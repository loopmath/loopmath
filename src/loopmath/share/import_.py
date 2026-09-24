"""Import a shared file as an organization group (spec 03 section 7, spec 04 section 5).

Owner: lane 08. `prior import-shared FILE` checks a `loopmath.share/1` file
strictly (every key at every level must be one `share.export` writes, and every
string must be an identifier, a salted hash or a closed value), then with the
OCP v0.3 checker (D46), and stores its runs in
`$LOOPMATH_HOME/priors/shared-<org_hash>.json.gz`, merged by run id with
earlier imports from the same organization. Each stored run keeps the time of
the share it came from (`shared_at`), and on a clash the run from the later
share wins, whatever the import order. `shared_runs(home)` is the one reader
for the fit (decision D22): each run comes back with
`run.task.org = "shared:<org_hash>"` and its outcome, scores and source node in
`run.ext["dev.loopmath.share"]`.
"""

from __future__ import annotations

import copy
import datetime as _dt
import gzip
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from ..ocp.conformance import validate_doc
from ..types import TIERS
from .export import (
    BASES, BETTER, CFG_RE, CONFIG_SOURCES, COST_COUNTS, DATE_RE, EXT_KEY, LOGMATCH_KEY, MODEL_TOKEN_COUNTS,
    MODEL_TOKENS_KEY, OCP_VERSION, RESULTS, SCALES, SCHEMA, SHARED_RULE, SHARED_SESSION_TIERS, STATUSES,
    VERDICT_VALUES, is_identifier, is_pathlike, write_share,
)


ORG_PREFIX = "shared:"
MAX_BYTES = 256 * 1024 * 1024  # decompressed size cap for one shared file
MAX_PROBLEMS = 50
HEX16_RE = re.compile(r"[0-9a-f]{16}\Z")
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})\Z")

Check = Callable[[Any, str, list], None]


# ---------------------------------------------------------------- reading
def read_share(path: Path) -> dict:
    """A shared file as a dict: gzip JSON (plain JSON is accepted too). Raises ValueError when unreadable.

    At most MAX_BYTES are read from disk, and at most MAX_BYTES once decompressed.
    """
    path = Path(path)
    too_big = f"larger than {MAX_BYTES} bytes"
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(too_big)
    with path.open("rb") as f:
        raw = f.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError(too_big)
    try:
        if raw[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                raw = gz.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise ValueError(f"larger than {MAX_BYTES} bytes once decompressed")
        obj = json.loads(raw.decode("utf-8"))
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"not a gzip JSON share file ({type(exc).__name__})") from None
    if not isinstance(obj, dict):
        raise ValueError("not a JSON object")
    return obj


# ---------------------------------------------------------------- the strict check
def _fail(errs: list, where: str, why: str) -> None:
    errs.append(f"{where}: {why}")  # never echo the value: a rejected file may hold private text


def _pattern(regex: re.Pattern, why: str) -> Check:
    def check(v: Any, where: str, errs: list) -> None:
        if not (isinstance(v, str) and regex.match(v)):
            _fail(errs, where, why)
    return check


def _enum(*values: Any) -> Check:
    def check(v: Any, where: str, errs: list) -> None:
        if isinstance(v, bool) or v not in values:
            _fail(errs, where, "not one of " + ", ".join(map(str, values)))
    return check


def _hash(prefix: str) -> Check:
    return _pattern(re.compile(rf"{prefix}_[0-9a-f]{{16}}\Z"), f"not a salted {prefix} hash")


def _int(v: Any, where: str, errs: list) -> None:
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        _fail(errs, where, "not a non-negative integer")


def _pos(v: Any, where: str, errs: list) -> None:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        _fail(errs, where, "not a positive integer")


def _num(v: Any, where: str, errs: list) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        _fail(errs, where, "not a finite number")


def _usd(v: Any, where: str, errs: list) -> None:
    _num(v, where, errs)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
        _fail(errs, where, "negative")


def _bool(v: Any, where: str, errs: list) -> None:
    if not isinstance(v, bool):
        _fail(errs, where, "not a boolean")


def _true(v: Any, where: str, errs: list) -> None:
    if v is not True:
        _fail(errs, where, "not true")


def _subtype(v: Any, where: str, errs: list) -> None:
    if not (isinstance(v, str) and v and not is_pathlike(v) and all(is_identifier(seg) for seg in v.split("/"))):
        _fail(errs, where, "not a short category path of identifiers separated by /")


def _z(v: Any, where: str, errs: list) -> None:
    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1):
        _fail(errs, where, "not null or a number in [0, 1]")


def _q(v: Any, where: str, errs: list) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0.5 <= v <= 1:
        _fail(errs, where, "not a number in [0.5, 1]")


def _tok(v: Any, where: str, errs: list) -> None:
    if not (isinstance(v, str) and is_identifier(v)):
        _fail(errs, where, "not an identifier")


def _either(*checks: Check) -> Check:
    def check(v: Any, where: str, errs: list) -> None:
        tries = []
        for c in checks:
            tries.append([])
            c(v, where, tries[-1])
            if not tries[-1]:
                return
        errs.extend(min(tries, key=len))
    return check


def _list_of(item: Check) -> Check:
    def check(v: Any, where: str, errs: list) -> None:
        if not isinstance(v, list):
            return _fail(errs, where, "not a list")
        for i, x in enumerate(v):
            item(x, f"{where}[{i}]", errs)
    return check


def _map_of(value: Check) -> Check:
    def check(v: Any, where: str, errs: list) -> None:
        if not isinstance(v, dict):
            return _fail(errs, where, "not an object")
        for k, x in v.items():
            if not is_identifier(k):
                _fail(errs, f"{where}.<key>", "not an identifier")
                continue
            value(x, f"{where}.{k}", errs)
    return check


def _obj(fields: dict[str, Check], required: tuple[str, ...] = ()) -> Check:
    def check(v: Any, where: str, errs: list) -> None:
        if not isinstance(v, dict):
            return _fail(errs, where, "not an object")
        for key in required:
            if key not in v:
                _fail(errs, f"{where}.{key}", "missing")
        for key, x in v.items():
            if key not in fields:
                _fail(errs, f"{where}.<key>", "a field loopmath.share/1 does not carry")
                continue
            fields[key](x, f"{where}.{key}", errs)
    return check


def _workflow(v: Any, where: str, errs: list) -> None:
    _WORKFLOW(v, where, errs)


def _edge(v: Any, where: str, errs: list) -> None:
    if not (isinstance(v, list) and len(v) == 2):
        return _fail(errs, where, "not a pair")
    _list_of(_tok)(v, where, errs)


_MODEL = _obj({"id": _tok, "family": _tok, "provider": _tok})
_SETTING = _obj({"harness": _tok, "model": _MODEL, "effort": _tok, "context_policy": _tok})
_CONTROL = _obj({
    "gates": _list_of(_tok),
    "repair": _map_of(_tok),
    "budget": _int,
    "rescue": _obj({"kind": _tok, "ref": _tok, "cost_usd": _usd}, required=("kind",)),
})
_WORKFLOW = _either(
    _obj({"ref": _tok, "version": _pos}, required=("ref",)),
    _obj({
        "id": _tok, "version": _pos,
        "pieces": _list_of(_obj({"id": _tok, "role": _tok, "width": _pos, "workflow": _workflow}, required=("id",))),
        "artifacts": _list_of(_obj({"id": _tok, "kind": _tok}, required=("id",))),
        "edges": _list_of(_edge),
        "control": _CONTROL,
    }, required=("pieces",)),
)
_SHARE_EXT = _obj({
    "outcome": _obj({"z": _z, "q": _q, "tier": _enum(*TIERS)}, required=("z", "q", "tier")),
    "scores": _list_of(_obj({"name": _tok, "value": _num, "unit": _tok, "better": _enum(*BETTER),
                             "scale": _enum(*SCALES)}, required=("name", "value"))),
    "verdicts": _list_of(_obj({"name": _tok, "value": _enum(*VERDICT_VALUES), "at_attempt": _tok,
                               "tier": _enum(*TIERS)}, required=("name", "value", "tier"))),
    "rounds": _int,
    "original_config_id": _pattern(CFG_RE, "not a configuration id"),
    "shared_at": _pattern(TS_RE, "not an ISO 8601 time with offset"),  # set on import, from the file
}, required=("outcome",))
_RUN = _obj({
    "id": _hash("run"),
    "task": _obj({"id": _hash("tsk"), "type": _tok, "subtype": _subtype, "repo": _hash("repo"),
                  "features": _map_of(_tok)}, required=("id", "repo")),
    "configuration": _obj({"id": _pattern(CFG_RE, "not a configuration id"), "source": _enum(*CONFIG_SOURCES),
                           "workflow": _workflow, "settings": _map_of(_SETTING)}),
    "slate": _obj({"id": _hash("slt"), "members": _list_of(_hash("run")), "isolated": _bool, "blinded": _bool},
                  required=("id", "members")),
    "acceptance_rule": _obj({
        "name": _enum(SHARED_RULE["name"]),
        "definition": _enum(SHARED_RULE["definition"]),
        "requires": _list_of(_tok),
        "score": _obj({"name": _tok, "target": _num, "better": _enum(*BETTER), "scale": _enum(*SCALES)},
                      required=("name", "target", "better")),
        "excludes_events": _list_of(_tok),
        "window_days": _int,
    }, required=("name", "definition")),
    "ext": _obj({EXT_KEY: _SHARE_EXT}, required=(EXT_KEY,)),
}, required=("id", "task", "ext"))
_LOGMATCH_FIELDS = _obj({"tier": _enum(*SHARED_SESSION_TIERS), "shared_session": _true}, required=("tier",))


def _logmatch(v: Any, where: str, errs: list) -> None:
    """`{"tier": "heuristic"}` (D56), or a tier and `shared_session: true` when attempts split a session (D88)."""
    _LOGMATCH_FIELDS(v, where, errs)
    if isinstance(v, dict) and "shared_session" not in v and v.get("tier") == "verified":
        _fail(errs, f"{where}.tier", "verified only with shared_session")


_COST_FIELDS = _obj({
    **{k: _int for k in COST_COUNTS}, "usd": _usd, "basis": _enum(*BASES), "tier": _enum(*TIERS),
    "tariff": _obj({"id": _tok, "date": _pattern(DATE_RE, "not a date")}),
    "ext": _obj({LOGMATCH_KEY: _logmatch,
                 MODEL_TOKENS_KEY: _map_of(_obj({k: _int for k in MODEL_TOKEN_COUNTS}, required=MODEL_TOKEN_COUNTS))}),
}, required=("basis",))


def _cost(v: Any, where: str, errs: list) -> None:
    """`ext` holds the log match record exactly when the cost is allocated (D56, D88), and may hold the per-model
    split (D71; `{}` is a valid split). The exporter never writes an empty `ext`."""
    _COST_FIELDS(v, where, errs)
    if not isinstance(v, dict):
        return
    ext = v.get("ext", {})
    if not isinstance(ext, dict):
        return
    if (LOGMATCH_KEY in ext) != (v.get("basis") == "allocated"):
        _fail(errs, f"{where}.ext.{LOGMATCH_KEY}", "present exactly when basis is allocated")
    if "ext" in v and not ext:
        _fail(errs, f"{where}.ext", "empty")
_ATTEMPT = _obj({
    "id": _tok, "node": _tok, "n": _pos, "vertex": _tok, "round": _pos, "status": _enum(*STATUSES),
    "harness": _tok, "model": _MODEL, "effort": _tok,
    "cause": _obj({"type": _tok}, required=("type",)),
    "outcome": _obj({"result": _enum(*RESULTS), "evidence": _enum(*TIERS), "via": _tok}, required=("result",)),
    "cost": _cost,
}, required=("id", "node", "status"))
_DOC = _obj({
    "ocp": _enum(OCP_VERSION),
    "producer": _obj({"name": _enum("loopmath"), "version": _tok}, required=("name",)),
    "privacy": _obj({"profile": _enum("metadata_only")}, required=("profile",)),
    "run": _RUN,
    "nodes": _list_of(_obj({"id": _tok, "kind": _tok, "vertex": _tok, "gate": _obj({"rule": _tok})},
                           required=("id", "kind"))),
    "attempts": _list_of(_ATTEMPT),
}, required=("ocp", "producer", "privacy", "run", "nodes"))
_SHARE = _obj({
    "schema": _enum(SCHEMA),
    "org_hash": _pattern(HEX16_RE, "not 16 hex digits"),
    "created_at": _pattern(TS_RE, "not an ISO 8601 time with offset"),
    "loopmath_version": _tok,
    "runs": _list_of(_DOC),
}, required=("schema", "org_hash", "created_at", "loopmath_version", "runs"))


def check_share(obj: Any) -> list[str]:
    """Problems with a `loopmath.share/1` object; empty when it can be imported."""
    errs: list[str] = []
    if isinstance(obj, dict) and obj.get("schema") != SCHEMA:
        return [f"schema: expected {SCHEMA}"]
    _SHARE(obj, "share", errs)
    if not errs:
        ids = [d["run"]["id"] for d in obj["runs"]]
        if len(ids) != len(set(ids)):
            errs.append("share.runs: the same run id appears twice")
    return errs[:MAX_PROBLEMS]


def ocp_problems(obj: dict) -> list[str]:
    """OCP v0.3 errors in a checked share's runs (D46), from lane 1's checker.

    Run after `check_share` passes, so every key in a reported place is an identifier;
    the checker's messages are left out because they quote values.
    """
    errs = []
    for i, doc in enumerate(obj["runs"]):
        for finding in validate_doc(doc):
            if finding.level == "error":
                errs.append(f"share.runs[{i}]{finding.path.removeprefix('$')}: OCP rule {finding.code}")
    return errs[:MAX_PROBLEMS]


# ---------------------------------------------------------------- import and read back
def org_node(org_hash: str) -> str:
    return ORG_PREFIX + org_hash


def priors_dir(home: Path) -> Path:
    return Path(home) / "priors"


def priors_path(home: Path, org_hash: str) -> Path:
    return priors_dir(home) / f"shared-{org_hash}.json.gz"


def import_share(obj: dict, home: Path) -> dict:
    """Store a checked share under its organization, merged by run id.

    Every run keeps the `created_at` of the share it came from as `shared_at`; on
    a clash the run with the later `shared_at` wins (the incoming one on a tie),
    so the result does not depend on import order. Returns
    `{org, path, added, updated, unchanged, total}`.
    """
    org_hash = obj["org_hash"]
    path = priors_path(home, org_hash)
    previous = None
    if path.exists():
        previous = read_share(path)
        if check_share(previous):
            raise ValueError(f"{path} is not a valid {SCHEMA} file; move it away and import again")
    before = {d["run"]["id"]: d for d in previous["runs"]} if previous else {}
    merged = dict(before)
    for doc in obj["runs"]:
        doc = copy.deepcopy(doc)
        doc["run"]["ext"][EXT_KEY]["shared_at"] = obj["created_at"]
        old = before.get(doc["run"]["id"])
        if old is None or _time(_shared_at(old, previous)) <= _time(obj["created_at"]):
            merged[doc["run"]["id"]] = doc
    latest = obj if not previous or _time(obj["created_at"]) >= _time(previous["created_at"]) else previous
    stored = {
        "schema": SCHEMA,
        "org_hash": org_hash,
        "created_at": latest["created_at"],
        "loopmath_version": latest["loopmath_version"],
        "runs": [merged[k] for k in sorted(merged)],
    }
    write_share(stored, path)
    incoming = {d["run"]["id"] for d in obj["runs"]}
    changed = {k for k in incoming & set(before) if _content(before[k]) != _content(merged[k])}
    return {
        "org": org_node(org_hash),
        "path": str(path),
        "added": len(incoming - set(before)),
        "updated": len(changed),
        "unchanged": len(incoming & set(before)) - len(changed),
        "total": len(merged),
    }


def _shared_at(doc: dict, stored: dict) -> str:
    """When a stored run was shared; a run without `shared_at` takes the stored file's time."""
    return doc["run"]["ext"][EXT_KEY].get("shared_at") or stored["created_at"]


def _content(doc: dict) -> dict:
    """A run without its `shared_at`, to tell a changed run from the same run shared again."""
    doc = copy.deepcopy(doc)
    doc["run"]["ext"][EXT_KEY].pop("shared_at", None)
    return doc


def shared_runs(home: Path) -> Iterator[dict]:
    """Every imported run, as an OCP v0.3 shaped document under its organization node (D22).

    `run.task.org` and `run.ext["dev.loopmath.share"]["source"]` are `shared:<org_hash>`;
    the outcome the sender's rule gave is `run.ext["dev.loopmath.share"]["outcome"]` = {z, q, tier},
    and `["shared_at"]` is the time of the share the run came from.
    A stored file that fails the check is skipped with a warning on stderr.
    """
    for path in sorted(priors_dir(home).glob("shared-*.json.gz")):
        try:
            obj = read_share(path)
        except (OSError, ValueError) as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            continue
        problems = check_share(obj)
        if problems:
            print(f"warning: skipping {path}: {problems[0]}", file=sys.stderr)
            continue
        org = org_node(obj["org_hash"])
        for doc in obj["runs"]:
            doc = copy.deepcopy(doc)
            doc["run"]["task"]["org"] = org
            doc["run"]["ext"][EXT_KEY]["source"] = org
            yield doc


def _time(text: str) -> _dt.datetime:
    return _dt.datetime.fromisoformat(text)
