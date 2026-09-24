"""Reduce runs to the shareable fields and write `loopmath.share/1` (spec 03 section 7).

Owner: lane 08. The reduction is an allowlist: every shared document is built
fresh from the fields named below, so a field this module does not name never
leaves. Identifier-like strings pass through only when they look like
identifiers; anything else becomes a salted hash label. Repo, task, run and
slate ids are always salted hashes. The salt is a per-store secret
(`$LOOPMATH_HOME/share/salt`, decision D23), so two shares from one store agree
and nobody can reverse a hash by trying names. Timestamps are not shared (D24).
"""

from __future__ import annotations

import copy
import datetime as _dt
import gzip
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from .. import __version__
from ..ingest.base import canonical_model
from ..ocp.canonical import config_id
from ..ocp.conformance import catalog_resolver
from ..store.home import Store
from ..taskmodel import normalize_features
from ..types import TIERS, AcceptanceRule, Evidence

SCHEMA = "loopmath.share/1"
EXT_KEY = "dev.loopmath.share"
OCP_VERSION = "0.3"

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,63}\Z")
# Identifier-shaped values that are still locators: session ids (UUIDs) and full commit shas.
LOCATOR_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|(?<![0-9a-f])[0-9a-f]{40}(?:[0-9a-f]{24})?(?![0-9a-f])", re.I)
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
CFG_RE = re.compile(r"cfg_[0-9a-f]{12}\Z")
# A subtype is a short category path such as `payments/webhooks`; anything that reads as a file path is hashed whole.
MAX_SUBTYPE_DEPTH = 4
_PATH_ROOTS = frozenset({"users", "home", "volumes", "mnt", "private", "tmp", "var"})
_DRIVE_RE = re.compile(r"[A-Za-z]:(\Z|[\\/])")

COST_COUNTS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "cache_creation_5m_tokens",
               "cache_creation_1h_tokens", "output_tokens", "reasoning_tokens", "requests")
STATUSES = ("queued", "working", "done", "failed", "rejected", "canceled", "settled_unverified", "lost")
RESULTS = STATUSES[2:]  # terminal results: OCP outcome.result
BASES = ("measured", "allocated")
# D56: an allocated cost keeps exactly this of the log match record; sessions, clips, parts and reasons stay home.
ALLOCATED_EXT = {"dev.loopmath.logmatch": {"tier": "heuristic"}}
(LOGMATCH_KEY,) = ALLOCATED_EXT
# D88: when attempts split one session's cost, the record also keeps its tier and `shared_session: true`, a bare
# boolean; the session id and the attempt ids stay home.
SHARED_SESSION_TIERS = ("verified", "heuristic")
# D67, D71: the per-model token split travels on any cost so a mixed-model cost can be repriced. Absent means one
# model; `{}` means more than one model and no split; `unknown` holds unlabelled tokens, which are never priced.
MODEL_TOKENS_KEY = "dev.loopmath.model_tokens"
MODEL_TOKEN_COUNTS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")
UNKNOWN_MODEL = "unknown"
# D46: the sender's rule text stays home; the structured parts travel under this fixed name and definition.
SHARED_RULE = {"name": "shared", "definition": "withheld"}
VERDICT_VALUES = ("accept", "reject", "pass", "fail", "error")
CONFIG_SOURCES = ("usual", "alternative", "exploration", "user_edit", "habit", "designed")
BETTER = ("higher", "lower")
SCALES = ("linear", "log", "fraction")
# OCP 2.10 spells two `touches` values differently from taskmodel; map them onto taskmodel's buckets.
_OCP_TOUCHES = {"2-3": "few", "4+": "many"}

EvidenceFn = Callable[..., Evidence]


def is_identifier(text: str) -> bool:
    """A short identifier that is not a session id or a commit sha: the only strings a share passes through."""
    return bool(TOKEN_RE.match(text)) and not LOCATOR_RE.search(text)


def is_pathlike(text: str) -> bool:
    """A subtype that reads as a file path: absolute, home-relative, Windows, dotted, under a home root, or deep."""
    segments = text.split("/")
    return (text.startswith(("/", "~")) or "\\" in text or bool(_DRIVE_RE.match(text))
            or any(seg in ("", ".", "..") for seg in segments) or len(segments) > MAX_SUBTYPE_DEPTH
            or segments[0].lower() in _PATH_ROOTS)


def config_id_of(workflow: dict | None, settings: dict | None) -> str | None:
    """The `cfg_` id of the reduced configuration (D2, D46), computed by lane 1's canonical form.

    None for a missing workflow, and for a reference the catalog cannot resolve
    (the OCP checker skips E190 then too).
    """
    if workflow is None:
        return None
    try:
        return config_id(workflow, settings or {}, resolve=catalog_resolver)
    except ValueError:  # UnresolvedWorkflow
        return None


# ---------------------------------------------------------------- salt and hashes
def salt_path(home: Path) -> Path:
    return Path(home) / "share" / "salt"


def load_salt(home: Path) -> bytes:
    """The store's share salt, created once (32 random bytes as hex, mode 0600)."""
    path = salt_path(home)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".salt.")
        try:
            os.write(fd, secrets.token_hex(32).encode("ascii"))
            os.fsync(fd)
            os.close(fd)
            os.chmod(tmp, 0o600)
            try:
                os.link(tmp, path)  # atomic create-if-absent: two first shares still end with one salt
            except FileExistsError:
                pass
        finally:
            os.unlink(tmp)
    text = path.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"share salt at {path} is not 64 hex digits")
    return bytes.fromhex(text)


def digest(salt: bytes, kind: str, value: str) -> str:
    return hmac.new(salt, f"{kind}\0{value}".encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def hashed(salt: bytes, prefix: str, value: Any) -> str:
    """`<prefix>_` plus 16 hex of HMAC-SHA256 under the store salt."""
    return f"{prefix}_{digest(salt, prefix, str(value))}"


def org_hash(salt: bytes, org: str | None) -> str:
    return digest(salt, "org", org or "")


# ---------------------------------------------------------------- reduction
class _Reducer:
    """Builds one shared run. Node and attempt ids are renumbered per run (`n1`, `a1`)."""

    def __init__(self, salt: bytes):
        self.salt = salt
        self.nodes: dict[str, str] = {}
        self.attempts: dict[str, str] = {}

    def token(self, value: Any) -> str | None:
        if value is None or isinstance(value, bool):
            return None
        text = str(value)
        if not text:
            return None
        return text if is_identifier(text) else hashed(self.salt, "h", text)

    def new_ref(self, table: dict[str, str], prefix: str, value: Any) -> str:
        return table.setdefault(str(value), f"{prefix}{len(table) + 1}")

    def subtype(self, value: Any) -> str | None:
        """Identifier segments stay plain (D24); a path-like value becomes one hash, so no part of it leaves."""
        if not isinstance(value, str) or not value:
            return None
        if is_pathlike(value):
            return hashed(self.salt, "h", value)
        return "/".join(self.token(seg) for seg in value.split("/"))

    def model(self, value: Any) -> dict | None:
        """An OCP modelRef: `{id, family, provider}`; the raw label stays home."""
        if isinstance(value, str):
            value = {"id": canonical_model(value)}
        if not isinstance(value, dict):
            return None
        return _clean({"id": self.token(value.get("id") or canonical_model(value.get("raw"))),
                       "family": self.token(value.get("family")),
                       "provider": self.token(value.get("provider"))}) or None

    def keyed(self, mapping: Any, fn: Callable[[Any], Any]) -> dict | None:
        if not isinstance(mapping, dict):
            return None
        out = {}
        for key, value in mapping.items():
            tok, mapped = self.token(key), fn(value)
            if tok and mapped is not None:
                out[tok] = mapped
        return out

    # -- task
    def task(self, task: dict) -> dict:
        return _clean({
            "id": hashed(self.salt, "tsk", task.get("id", "")),
            "type": self.token(task.get("type")),
            "subtype": self.subtype(task.get("subtype")),
            "repo": hashed(self.salt, "repo", task.get("repo", "")),
            "features": self.features(task.get("features")),
        })

    def features(self, raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        prepared: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                value = "yes" if value else "no"
            if str(key).lower() == "touches" and str(value) in _OCP_TOUCHES:
                value = _OCP_TOUCHES[str(value)]
            if isinstance(value, (str, int, float)):
                prepared[str(key)] = str(value)
        out = {}
        for key, value in normalize_features(prepared).items():
            if key.startswith("extra:"):
                continue  # the user's own keys may hold anything: spec 03 section 7 drops `extra`
            tok = self.token(value)
            if tok:
                out[key] = tok
        return out

    # -- configuration
    def configuration(self, cfg: Any) -> dict:
        """Workflow and settings as reduced; the id is recomputed from them (D46), since `options` and titles are gone."""
        if not isinstance(cfg, dict):
            return {}
        workflow = self.workflow(cfg.get("workflow"))
        settings = self.keyed(cfg.get("settings"), self.setting)
        return _clean({
            "id": config_id_of(workflow, settings),
            "source": cfg.get("source") if cfg.get("source") in CONFIG_SOURCES else None,
            "workflow": workflow,
            "settings": settings,
        })

    def setting(self, s: Any) -> dict:
        if not isinstance(s, dict):
            return {}
        # `options` is dropped whole: it holds gate commands (`options.command`) and harness flags.
        return _clean({"harness": self.token(s.get("harness")), "model": self.model(s.get("model")),
                       "effort": self.token(s.get("effort")), "context_policy": self.token(s.get("context_policy"))})

    def workflow(self, wf: Any) -> dict | None:
        """OCP 2.3 forms only (a reference, or pieces, artifacts, edges and control); other shapes are dropped."""
        if not isinstance(wf, dict):
            return None
        if "ref" in wf and "pieces" not in wf:
            ref = self.token(wf.get("ref"))
            return _clean({"ref": ref, "version": _pos(wf.get("version"))}) if ref else None
        pieces = []
        for p in _list(wf.get("pieces")):
            if isinstance(p, dict) and self.token(p.get("id")):
                pieces.append(_clean({"id": self.token(p.get("id")), "role": self.token(p.get("role")),
                                      "width": _pos(p.get("width")), "workflow": self.workflow(p.get("workflow"))}))
        if not pieces:
            return None
        artifacts = [_clean({"id": self.token(a.get("id")), "kind": self.token(a.get("kind"))})
                     for a in _list(wf.get("artifacts")) if isinstance(a, dict) and self.token(a.get("id"))]
        edges = [[self.token(e[0]), self.token(e[1])] for e in _list(wf.get("edges"))
                 if isinstance(e, (list, tuple)) and len(e) == 2 and self.token(e[0]) and self.token(e[1])]
        # The title is free text; the configuration id excludes it.
        return _clean({"id": self.token(wf.get("id")), "version": _pos(wf.get("version")), "pieces": pieces,
                       "artifacts": artifacts, "edges": edges, "control": self.control(wf.get("control"))})

    def control(self, c: Any) -> dict | None:
        """Gates, repair, budget and rescue. `ext` (with any gate rule commands) stays home."""
        if not isinstance(c, dict):
            return None
        gates = dict.fromkeys(t for t in (self.token(g) for g in _list(c.get("gates")) if isinstance(g, str)) if t)
        rescue = c.get("rescue")
        rescue = _clean({"kind": self.token(rescue.get("kind")), "ref": self.token(rescue.get("ref")),
                         "cost_usd": _usd(rescue.get("cost_usd"))}) if isinstance(rescue, dict) else {}
        return _clean({
            "gates": list(gates),
            "repair": self.keyed(c.get("repair"), self.token),
            "budget": _int(c["budget"] if "budget" in c else c.get("budget_rounds")),  # D30
            "rescue": rescue if "kind" in rescue else None,
        }) or None

    # -- run parts
    def slate(self, slate: Any) -> dict | None:
        if not isinstance(slate, dict) or not slate.get("id"):
            return None
        members = dict.fromkeys(hashed(self.salt, "run", m) for m in _list(slate.get("members")) if isinstance(m, str))
        if not members:
            return None  # OCP requires at least one member
        return _clean({
            "id": hashed(self.salt, "slt", slate["id"]),
            "members": list(members),
            "isolated": slate.get("isolated") if isinstance(slate.get("isolated"), bool) else None,
            "blinded": slate.get("blinded") if isinstance(slate.get("blinded"), bool) else None,
        })

    def rule(self, rule: AcceptanceRule) -> dict:
        """The rule's structure, so z can be read, under the fixed name and definition of D46."""
        score = None
        s = rule.score
        if s is not None and self.token(s.name) and _num(s.target) is not None and s.better in BETTER:
            score = _clean({"name": self.token(s.name), "target": _num(s.target), "better": s.better,
                            "scale": s.scale if s.scale in SCALES else None})
        return {**SHARED_RULE, **_clean({
            "requires": [t for t in (self.token(r) for r in rule.requires or ()) if t],
            "score": score,
            "excludes_events": [t for t in (self.token(e) for e in rule.excludes_events or ()) if t],
            "window_days": _int(rule.window_days),
        })}

    def node(self, n: dict) -> dict:
        gate = n.get("gate")
        return _clean({
            "id": self.new_ref(self.nodes, "n", n["id"]),
            "kind": self.token(n.get("kind")) or "unknown",
            "vertex": self.token(n.get("vertex")),
            "gate": (_clean({"rule": self.token(gate.get("rule"))}) or None) if isinstance(gate, dict) else None,
        })

    def attempt(self, a: dict) -> dict:
        cause = a.get("cause")
        outcome = a.get("outcome")
        return _clean({
            "id": self.new_ref(self.attempts, "a", a["id"]),
            "node": self.new_ref(self.nodes, "n", a["node"]),
            "n": _pos(a.get("n")),
            "vertex": self.token(a.get("vertex")),
            "round": _pos(a.get("round")),
            "status": a.get("status") if a.get("status") in STATUSES else "settled_unverified",
            "harness": self.token(a.get("harness")),
            "model": self.model(a.get("model")),
            "effort": self.token(a.get("effort")),
            "cause": (_clean({"type": self.token(cause.get("type"))}) or None) if isinstance(cause, dict) else None,
            "outcome": self.outcome(outcome) if isinstance(outcome, dict) else None,
            "cost": self.cost(a.get("cost")),
        })

    def outcome(self, o: dict) -> dict | None:
        """Gate and attempt results: the terminal result, its evidence tier, and the judging node."""
        if o.get("result") not in RESULTS:
            return None
        via = o.get("via")
        return _clean({"result": o["result"], "evidence": o.get("evidence") if o.get("evidence") in TIERS else None,
                       "via": self.nodes.get(str(via)) if via is not None else None})

    def cost(self, c: Any) -> dict | None:
        """Tokens, dollars, basis and tariff. `ext` keeps an allocated cost's log match tier (D56), whether the
        attempts split a shared session (D88), and the per-model split (D67, D71).

        A cost whose basis is neither measured nor allocated is left out: a document
        from producer loopmath may carry no other (OCP E171).
        """
        if not isinstance(c, dict) or c.get("basis") not in BASES:
            return None
        out: dict[str, Any] = {k: _int(c.get(k)) for k in COST_COUNTS}
        out["usd"] = _usd(c.get("usd"))
        out["basis"] = c["basis"]
        out["tier"] = c.get("tier") if c.get("tier") in TIERS else None
        tariff = c.get("tariff")
        if isinstance(tariff, dict):
            date = tariff.get("date")
            out["tariff"] = _clean({"id": self.token(tariff.get("id")),
                                    "date": date if isinstance(date, str) and DATE_RE.match(date) else None}) or None
        source = c.get("ext") if isinstance(c.get("ext"), dict) else {}
        ext = {LOGMATCH_KEY: _logmatch(source.get(LOGMATCH_KEY))} if c["basis"] == "allocated" else {}
        if MODEL_TOKENS_KEY in source:
            ext[MODEL_TOKENS_KEY] = self.model_tokens(source[MODEL_TOKENS_KEY])
        out["ext"] = ext
        return _clean(out)

    def model_tokens(self, split: Any) -> dict[str, dict[str, int]]:
        """D71: `{model id: {the four OCP token counts}}`, nothing else.

        A model id goes through the same rule as an attempt's model; one with no usable
        label is `unknown`, and parts that reduce to one id are summed. A split that does
        not have this shape becomes `{}`: more than one model, no usable split.
        """
        if not isinstance(split, dict):
            return {}
        out: dict[str, dict[str, int]] = {}
        for key, counts in split.items():
            model = self.token(canonical_model(key) if isinstance(key, str) else None) or UNKNOWN_MODEL
            part = {k: _int(counts.get(k)) for k in MODEL_TOKEN_COUNTS} if isinstance(counts, dict) else {}
            if len(part) != len(MODEL_TOKEN_COUNTS) or None in part.values():
                return {}
            total = out.setdefault(model, dict.fromkeys(MODEL_TOKEN_COUNTS, 0))
            for k, n in part.items():
                total[k] += n
        return out

    def verdict(self, s: dict) -> dict | None:
        if s.get("kind") != "verdict" or s.get("value") not in VERDICT_VALUES or not self.token(s.get("name")):
            return None
        at = s.get("at_attempt")
        return _clean({"name": self.token(s.get("name")), "value": s["value"],
                       "at_attempt": self.attempts.get(str(at)) if at is not None else None,
                       "tier": s.get("tier") if s.get("tier") in TIERS else "asserted"})


def run_rule(doc: dict) -> AcceptanceRule:
    """The acceptance rule recorded on the run, else the default binary rule (`tests`)."""
    raw = (doc.get("run") or {}).get("acceptance_rule")
    if isinstance(raw, dict) and raw.get("name"):
        return AcceptanceRule.from_dict({"definition": "", **raw})
    return AcceptanceRule(name="tests", definition="tests pass")


def reduce_run(doc: dict, *, salt: bytes, evidence: Evidence, rule: AcceptanceRule) -> dict:
    """One OCP v0.3 run document reduced to the fields spec 03 section 7 keeps."""
    r = _Reducer(salt)
    run = doc.get("run") or {}
    nodes = [r.node(n) for n in _list(doc.get("nodes")) if isinstance(n, dict) and n.get("id") is not None]
    attempts = [r.attempt(a) for a in _list(doc.get("attempts"))
                if isinstance(a, dict) and a.get("id") is not None and a.get("node") is not None]
    signals = [s for s in _list(run.get("signals")) if isinstance(s, dict)]
    share_ext = {
        "outcome": {"z": _num(evidence.z), "q": _num(evidence.q),
                    "tier": evidence.tier if evidence.tier in TIERS else "asserted"},
        "scores": _scores(r, evidence, signals),
        "verdicts": [v for v in (r.verdict(s) for s in signals) if v],
        "rounds": max((a.get("round") or 1 for a in attempts), default=0),
    }
    cfg = run.get("configuration")
    original = cfg.get("id") if isinstance(cfg, dict) else None
    if isinstance(original, str) and CFG_RE.match(original):
        share_ext["original_config_id"] = original  # the sender's id before reduction (D46)
    return {
        "ocp": OCP_VERSION,
        "producer": {"name": "loopmath", "version": __version__},
        "privacy": {"profile": "metadata_only"},
        "run": _clean({
            "id": hashed(salt, "run", run.get("id", "")),
            "task": r.task(run.get("task") or {}),
            "configuration": r.configuration(run.get("configuration")),
            "slate": r.slate(run.get("slate")),
            "acceptance_rule": r.rule(rule),
            "ext": {EXT_KEY: share_ext},
        }),
        "nodes": nodes,
        "attempts": attempts,
    }


def _scores(r: _Reducer, evidence: Evidence, signals: list[dict]) -> list[dict]:
    """Every measured score from the outcome function, with unit, direction and scale from its last signal."""
    meta = {s["name"]: s for s in signals if s.get("kind") == "score" and isinstance(s.get("name"), str)}
    out = []
    for name, value in sorted((evidence.scores or {}).items()):
        v = _num(value)
        if v is None or not r.token(name):
            continue
        s = meta.get(name, {})
        out.append(_clean({"name": r.token(name), "value": v, "unit": r.token(s.get("unit")),
                           "better": s.get("better") if s.get("better") in BETTER else None,
                           "scale": s.get("scale") if s.get("scale") in SCALES else None}))
    return out


# ---------------------------------------------------------------- the share object
def build_share(docs: Iterable[dict], *, salt: bytes, org: str | None, evidence_fn: EvidenceFn,
                since: _dt.datetime | None, now: _dt.datetime) -> tuple[dict, dict[str, int]]:
    """The `loopmath.share/1` object, and the runs left out counted by reason.

    `evidence_fn(doc, rule, now=now)` is the outcome function (spec 03 section 5),
    applied with the rule recorded on each run. A run it cannot read is left out
    and counted.
    """
    runs: list[dict] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for doc in docs:
        if since is not None:
            at = run_time(doc)
            if at is None:
                skip("no start time (--since)")
                continue
            if at < since:
                skip("before --since")
                continue
        rule = run_rule(doc)
        try:
            evidence = evidence_fn(doc, rule, now=now)
        except Exception:  # noqa: BLE001 - one unreadable run is counted, not fatal
            skip("outcome unreadable")
            continue
        runs.append(reduce_run(doc, salt=salt, evidence=evidence, rule=rule))
    runs.sort(key=lambda d: d["run"]["id"])
    obj = {
        "schema": SCHEMA,
        "org_hash": org_hash(salt, org),
        "created_at": now.isoformat(timespec="seconds"),
        "loopmath_version": __version__,
        "runs": runs,
    }
    return obj, skipped


def dumps(obj: dict, *, indent: int | None = None) -> str:
    """The share as JSON text, keys in the order they were built (`schema` first)."""
    return json.dumps(obj, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))


def write_share(obj: dict, path: Path) -> Path:
    """Gzip JSON, written atomically (temp file in the same folder, fsync, rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
                gz.write(dumps(obj).encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


# ---------------------------------------------------------------- store access
def store_docs(home: Path) -> Iterator[dict]:
    """Finished run documents from the store (spec 03 section 4), through lane 7's index."""
    return Store(home).finished_docs()


def store_org(home: Path) -> str | None:
    """The store's `org` config value, if set."""
    try:
        with (Path(home) / "config.toml").open("rb") as f:
            org = tomllib.load(f).get("org")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return org if isinstance(org, str) and org else None


# ---------------------------------------------------------------- time
def run_time(doc: dict) -> _dt.datetime | None:
    run = doc.get("run") or {}
    for key in ("started_at", "ended_at"):
        value = run.get(key)
        if isinstance(value, str):
            try:
                at = _dt.datetime.fromisoformat(value)
            except ValueError:
                continue
            return at if at.tzinfo else at.astimezone()
    return None


# ---------------------------------------------------------------- small helpers
def _list(value: Any) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value) if value >= 0 else None


def _pos(value: Any) -> int | None:
    v = _int(value)
    return v if v else None


def _num(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _usd(value: Any) -> float | int | None:
    v = _num(value)
    return v if v is not None and v >= 0 else None


def _logmatch(match: Any) -> dict[str, Any]:
    """An allocated cost's log match record: the D56 tier, or with a shared session its own tier and a bare
    `shared_session: true` (D88). A shared session is a non-empty object in the store, or `true`, as lane 1's
    E171 reads it; any other value means none."""
    out = copy.deepcopy(ALLOCATED_EXT[LOGMATCH_KEY])
    shared = match.get("shared_session") if isinstance(match, dict) else None
    if shared is True or (isinstance(shared, dict) and shared):
        if match.get("tier") in SHARED_SESSION_TIERS:
            out["tier"] = match["tier"]
        out["shared_session"] = True
    return out


def _clean(d: dict) -> dict:
    """Drop None values and empty containers; keep zeros and False."""
    return {k: v for k, v in d.items() if v is not None and v != {} and v != []}
