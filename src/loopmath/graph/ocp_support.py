"""Graph to OCP v0.3 (EXTRACTOR-SPEC.md section 5, task P2; spec/OCP.md section 8).

`to_ocp(graph, *, producer, privacy)` turns a `Graph` into one Orchestration
Context Protocol v0.3 document (`spec/ocp-v0.3.schema.json`). The mapping is
one node, one attempt, one synthetic task per node: a session is the unit of
work and the attempt that ran it at once. Every edge keeps the tier the graph
gave it; every attempt carries `origin`, `role` and `phase` (the extractor's
output always does, this is where that is enforced); artifacts become the
v0.2 entity; `Graph.meta` and the emitter's own counters go to
`ext["dev.loopmath.graph"]`.

v0.3 is a superset of v0.2, and the graph has no source for the fields v0.3
adds (task, configuration, signals, receipt, vertices, `cost.tariff`: the
graph's dollars carry no record of the table that priced them), so they are
left out; attempts settled through `loopmath.logmatch` carry a tariff.
Extension keys use loopmath's namespace `dev.loopmath.` (OCP.md 8.4);
readers still take the `dev.dagr.` keys of documents written before 0.3.

What the graph does not know stays unknown here, and nothing is guessed:

- an unknown role, phase or artifact kind is an explicit `null` beside its
  tier; a label whose tier is not one of the three the spec allows is
  withheld (value `null`, the evidence says why) and counted, never relabeled;
- the origin of a launched node copies the tier of its launch edge, never
  upgraded; a node the graph names a launcher for without an emitted launch
  edge gets `launched_by: null` with the claim in the evidence, and is counted;
  `origin.external` is true only when the harness itself reported a launcher
  that no scanned session is (the L2 case, tier `reported`); a scanned
  launcher session outside the requested workspaces is not that case and gets
  `external: null`;
- an edge whose tier is invalid or whose endpoints are unknown is not emitted;
  it is kept whole under `ext["dev.loopmath.graph"]["edges_not_emitted"]` with the
  reason, and counted;
- the graph joins a read to the latest writer before it, while OCP's
  `artifact` edge runs from the producer (first writer). Such an edge is
  emitted from the producer to the reader with the same tier, the later
  writer named in the evidence and in the edge's `ext`, and the rewrite
  counted (`artifact_edges_rerouted_from_later_writer`);
- a timestamp that does not parse is omitted and counted;
- a token stream, a cache retention bucket, a whole token record or a dollar
  figure the graph does not have is an explicit `null` under
  `ext["dev.loopmath.graph"]["cost_unknown"]` with its reason under `cost_missing`,
  and counted; the cost record never carries a zero for it (the schema types
  cost fields as integers, so the null lives beside the reason in `ext`);
- an artifact whose path exceeds the schema's id length gets a hashed id, and
  one whose path exceeds the path length gets a cut path; in both cases the
  full path sits untouched in the artifact's `ext["dev.loopmath.graph"]["path"]` and
  the derivation is counted.

Privacy: under `metadata_only` no transcript text leaves the graph. Role
evidence that quotes a spawn description or a declared subagent type is
reduced to the rule name and the structural facts the rule used; a node's
title is the task the spawn description names only under `full`, and under
`metadata_only` it stays synthesized from role and harness with the withheld
title counted; spawn records and launch commands appear only under `full`,
in `ext`.

Vocabulary (spec section 0, rule 4): every producer-authored string passes
one choke point, `sanitize`, which replaces em-dashes and counts what it
changed. The vocabulary filter was lifted in 0.1.0 (Q3).

No attempt is ever marked `done`: the graph carries no acceptance signal, so
every attempt settles `settled_unverified` with evidence `heuristic`.

Deterministic: same graph, same document, byte for byte (sorted iteration,
no `emitted_at` unless the caller passes one in `producer`).
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone

from .. import __version__
from .schema import Artifact, Graph, GraphEdge, GraphNode

PRIVACY_PROFILES = ("metadata_only", "full")
TIERS = ("verified", "heuristic", "reported")
OCP_VERSION = "0.3"
SOURCE_CONTRACT = "dagr_graph/1"
EXT_KEY = "dev.loopmath.graph"
# Documents written before the namespace switch; read through v0.4 (W182).
LEGACY_EXT_KEY = "dev.dagr.graph"
# The contract run file (`graph --format run`) is not OCP and keeps its key.
RUN_EXT_KEY = "dev.dagr.graph"

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")
_SHORT = 500
_ID_MAX = 200
_PATH_MAX = 1000

# Spec section 0, rule 4: no em-dashes in a string a user can see; replaced,
# never passed through, and counted. The technical-term filter that sat here
# was lifted in loopmath 0.1.0 at the user's request (2026-09-23).
_EM_DASH_RE = re.compile("\\s*[\u2014\u2015]\\s*")

# Graph role -> OCP node kind (open set; recommended vocabulary in the schema).
_NODE_KIND = {"planner": "plan", "dev": "impl", "reviewer": "review", "lead": "ops", "solo": "ops", "cli": "ops", "external": "ops"}
# Graph token stream -> OCP cost field. The parsers keep four billable streams
# (ingest/base.py `Tokens`); the two cache retention buckets are read when a
# record carries them under either the graph's or OCP's name.
_COST_FIELDS = (("in", "input_tokens"), ("cache_read", "cached_input_tokens"), ("cache_write", "cache_creation_tokens"), ("out", "output_tokens"))
_BUCKET_FIELDS = (("cache_write_5m", "cache_creation_5m_tokens"), ("cache_write_1h", "cache_creation_1h_tokens"))
_BUCKET_REASON = "no retention split in the session record: the parsers keep one cache-write stream ({total}); unknown, never zero"
# Role evidence prefixes (graph/labels.py) that quote transcript text, with the
# rule name and the field it read, for the metadata_only rewrite.
_ROLE_TEXT_RULES = (
    ("spawn description ", "spawn_description_pattern", "the Task call's description"),
    ("declared type ", "declared_type_pattern", "the Task call's subagent type"),
)

COUNTERS = (
    "artifact_edges_rerouted_from_later_writer",
    "edges_not_emitted_invalid_tier",
    "edges_not_emitted_unknown_endpoint",
    "edges_not_emitted_unknown_artifact",
    "edges_not_emitted_writer_not_listed",
    "edges_not_emitted_reader_not_consumer",
    "labels_withheld_invalid_tier",
    "model_tiers_omitted_invalid",
    "origins_without_launch_edge",
    "origins_reported_unscanned_launcher",
    "role_evidence_reduced_to_rule",
    "titles_from_spawn_description",
    "titles_withheld_metadata_only",
    "attempts_without_start_ts",
    "attempts_missing_wall_s",
    "attempts_without_tokens",
    "attempts_unpriced",
    "token_streams_missing",
    "cache_retention_buckets_missing",
    "cache_retention_buckets_inconsistent",
    "artifacts_first_write_ts_unparseable",
    "artifact_ids_hashed",
    "artifact_paths_cut",
    "strings_with_em_dash_replaced",
)


def attempt_id(node_id: str) -> str:
    """The single attempt of a node: `<node id>.a1`."""
    return f"{node_id}.a1"


def sanitize(s: object, counters: dict | None = None) -> str:
    """The one choke point for a string a user can see (spec section 0, rule 4):
    em-dashes become a spaced hyphen, and each string changed is counted when
    `counters` is given."""
    out, n_dash = _EM_DASH_RE.subn(" - ", str(s))
    if counters is not None and n_dash:
        counters["strings_with_em_dash_replaced"] += 1
    return out


class _Text:
    """Sanitize and cap producer-authored strings, counting into one dict."""

    def __init__(self, counters: dict) -> None:
        self.counters = counters

    def __call__(self, s: object) -> str:
        return sanitize(s, self.counters)[:_SHORT]


def _parse_ts(ts: str | None) -> datetime | None:
    if not isinstance(ts, str) or not _TS_RE.match(ts):
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_ts(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _artifact_identity(path: str, counters: dict) -> tuple[str, str, dict]:
    """(id, path, ext): the path itself as id when it fits the schema's id
    length, else a hash of it; the path itself when it fits the path length,
    else its head. Whenever either is derived the full path goes untouched
    into `ext` and the derivation is counted. Nothing is lost."""
    ext: dict = {}
    aid = path
    if len(path) > _ID_MAX:
        counters["artifact_ids_hashed"] += 1
        aid = "sha256:" + hashlib.sha256(path.encode("utf-8")).hexdigest()
        ext["id_derived"] = f"sha256 of the full path: {len(path)} characters exceed the {_ID_MAX} character id limit"
    shown = path
    if len(path) > _PATH_MAX:
        counters["artifact_paths_cut"] += 1
        shown = path[:_PATH_MAX]
        ext["path_cut"] = f"first {_PATH_MAX} of {len(path)} characters shown; the full path is beside this note"
    if ext:
        ext["path"] = path
    return aid, shown, ext


def _producer(producer) -> dict:
    base = {"name": "loopmath", "version": __version__, "source_contract": SOURCE_CONTRACT}
    if producer is None:
        return base
    if isinstance(producer, str):
        return {**base, "name": producer}
    if isinstance(producer, dict):
        return {**base, **producer}
    raise TypeError("producer must be a name, a dict or None")


def _privacy(privacy) -> dict:
    if isinstance(privacy, dict):
        prof = privacy.get("profile")
        if prof not in PRIVACY_PROFILES:
            raise ValueError(f"privacy profile must be one of {PRIVACY_PROFILES}, got {prof!r}")
        return dict(privacy)
    if privacy not in PRIVACY_PROFILES:
        raise ValueError(f"privacy profile must be one of {PRIVACY_PROFILES}, got {privacy!r}")
    if privacy == "metadata_only":
        note = "session logs reduced to structure; paths, rule names and short producer-authored labels only"
    else:
        note = "spawn descriptions, declared subagent types and launch commands appear in titles, evidence and ext namespaces"
    return {"profile": privacy, "note": note}


def _spawn_task(n: GraphNode) -> str | None:
    """The task the spawn description names, when it names one."""
    sp = n.spawn if isinstance(n.spawn, dict) else {}
    for key in ("description", "meta_description"):
        d = sp.get(key)
        if isinstance(d, str) and d.strip():
            return d.strip()
    return None


def _synth_title(n: GraphNode) -> str:
    unit = "subagent" if n.source == "subagent" else "session"
    if n.source == "external":
        what = "external launcher session"
    elif n.role:
        what = f"{n.role} {unit}"
    else:
        what = f"unlabeled {unit}"
    return f"{what} ({n.harness})"


def _node_title(n: GraphNode, full: bool, t: _Text) -> tuple[str, dict]:
    """(title, node ext). The task the spawn description names under `full`;
    otherwise a short producer-authored label from role and harness. Under
    `metadata_only` a description is transcript text: the title stays
    synthesized and the withheld one is counted and noted."""
    task = _spawn_task(n)
    if task is None:
        return t(_synth_title(n)), {}
    if full:
        t.counters["titles_from_spawn_description"] += 1
        return t(task), {"title_source": "spawn description"}
    t.counters["titles_withheld_metadata_only"] += 1
    return t(_synth_title(n)), {"title_withheld": f"the spawn description names the task ({len(task)} chars); withheld under metadata_only"}


def _label(value, tier, field: str, counters: dict) -> tuple[object, str, str | None]:
    """(value, tier, note). A valid tier passes through. A `None` value with no
    tier is the graph saying "no rule matched", a heuristic unknown. A value
    whose tier is not one of the three the spec allows is withheld and
    counted: the emitter never picks a tier the graph did not give."""
    if tier in TIERS:
        return value, tier, None
    if value is None:
        return None, "heuristic", None
    counters["labels_withheld_invalid_tier"] += 1
    return None, "heuristic", f"graph gave {field} {value!r} with tier {tier!r}, not one of {'/'.join(TIERS)}; value withheld"


def _origin(n: GraphNode, launch_edge: tuple[str, str] | None, spawn_edge: str | None, t: _Text) -> dict:
    """The attempt's origin. A launched node copies the tier of its emitted
    launch edge; `external` is true only for a launcher the harness reported
    and nothing scanned (L2); a scanned launcher outside the requested
    workspaces is not that case."""
    counters = t.counters
    lb = n.launched_by if isinstance(n.launched_by, dict) else None
    base = {"launched_by": None, "workspace": n.workspace, "external": None, "how": None}
    if lb and lb.get("id"):
        launcher = str(lb["id"])
        how = t(lb["how"]) if lb.get("how") else None
        if launch_edge is not None and launch_edge[0] == launcher:
            return {**base, "launched_by": attempt_id(launcher), "external": False, "how": how, "tier": launch_edge[1], "evidence": t(f"launch edge ({launch_edge[1]}): {lb.get('how')}; lag {lb.get('lag_s')} s")}
        counters["origins_without_launch_edge"] += 1
        return {**base, "tier": "heuristic", "evidence": t(f"the graph names launcher {attempt_id(launcher)} but no launch edge from it is emitted; launcher withheld")}
    if lb and lb.get("external") is True:
        # L2: the harness reported a launcher (codex originator) that no scanned session is.
        tier = lb.get("tier")
        how = t(lb["how"]) if lb.get("how") else None
        if tier in TIERS:
            counters["origins_reported_unscanned_launcher"] += 1
            return {**base, "external": True, "how": how, "tier": tier, "evidence": t(lb.get("evidence") or "the harness reports a launcher that no scanned session is")}
        counters["labels_withheld_invalid_tier"] += 1
        return {**base, "how": how, "tier": "heuristic", "evidence": t(f"graph reports an unscanned launcher with tier {tier!r}, not one of {'/'.join(TIERS)}; value withheld")}
    if n.source == "subagent" and n.parent:
        if spawn_edge in TIERS:
            return {**base, "external": False, "tier": spawn_edge, "evidence": "in-harness subagent (spawn edge), not a CLI launch"}
        return {**base, "external": False, "tier": "heuristic", "evidence": t(f"in-harness subagent; its spawn edge carries tier {spawn_edge!r} and is not emitted")}
    if n.source == "subagent":
        return {**base, "tier": "heuristic", "evidence": "subagent transcript with no matching Task call or enclosing session"}
    if n.source == "external":
        return {**base, "tier": "heuristic", "evidence": "scanned top-level session outside the requested workspaces; it launched sessions inside them and nothing in scope launched it, so it is not an unscanned launcher"}
    return {**base, "tier": "heuristic", "evidence": "no launch command in scope matched this session's start; the harness gives no launcher signal in the graph"}


def _role_evidence(n: GraphNode, full: bool, t: _Text) -> str | None:
    """The graph's role evidence; under `metadata_only`, when the graph quoted
    transcript text, the rule name and the structural facts it used instead."""
    ev = n.role_evidence
    if not ev:
        return None
    if not full:
        for prefix, rule, field in _ROLE_TEXT_RULES:
            if ev.startswith(prefix):
                t.counters["role_evidence_reduced_to_rule"] += 1
                quoted = max(0, len(ev) - len(prefix) - 2)
                return t(f"rule {rule}: {field} matched the {n.role} pattern ({quoted} chars of text withheld under metadata_only)")
            if ev.startswith(f"rule {rule}:"):
                # An OCP read sees the already privacy-reduced evidence rather
                # than the transcript text.  Keep the provenance count stable
                # while passing that reduced evidence through unchanged.
                t.counters["role_evidence_reduced_to_rule"] += 1
                return t(ev)
    return t(ev)


def _role(n: GraphNode, full: bool, t: _Text) -> dict:
    value, tier, note = _label(n.role, n.role_tier, "role", t.counters)
    if note:
        return {"value": None, "tier": tier, "evidence": t(note)}
    ev = _role_evidence(n, full, t)
    if value is None:
        return {"value": None, "tier": tier, "evidence": ev or "role rules matched nothing"}
    return {"value": t(value), "tier": tier, "evidence": ev}


def _phase(n: GraphNode, t: _Text) -> dict:
    value, tier, note = _label(n.phase, n.phase_tier, "phase", t.counters)
    if note:
        return {"value": None, "tier": tier, "evidence": t(note)}
    if value is None:
        return {"value": None, "tier": tier, "evidence": "no lead interval or start time to place the session against"}
    ev = {"build": "started inside the lead's interval", "post": "started more than 60 s after the lead ended", "external": "launched from outside the requested workspaces"}.get(value, "phase rule of the graph")
    return {"value": t(value), "tier": tier, "evidence": ev}


def _count(v) -> int | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
        return int(v)
    return None


def _cost(n: GraphNode, counters: dict) -> tuple[dict | None, dict]:
    """(cost record or None, missing) where `missing` names every stream,
    retention bucket and figure the graph does not have for this attempt,
    with the reason. Nothing missing is ever written as zero."""
    missing: dict = {}
    cost: dict | None = None
    if not isinstance(n.tokens, dict):
        counters["attempts_without_tokens"] += 1
        missing["tokens"] = "no token usage in the session record; no cost record emitted"
    else:
        cost = {}
        for src, dst in _COST_FIELDS:
            if src not in n.tokens:
                counters["token_streams_missing"] += 1
                missing[dst] = f"stream {src!r} absent from the session record"
                continue
            v = _count(n.tokens[src])
            if v is None:
                counters["token_streams_missing"] += 1
                missing[dst] = f"stream {src!r} is {str(n.tokens[src])[:40]!r} in the session record, not a count"
            else:
                cost[dst] = v
        buckets: dict = {}
        for src, dst in _BUCKET_FIELDS:
            key = src if src in n.tokens else (dst if dst in n.tokens else None)
            if key is None or n.tokens[key] is None:
                counters["cache_retention_buckets_missing"] += 1
                missing[dst] = _BUCKET_REASON.format(total="absent too" if "cache_creation_tokens" not in cost else f"{cost['cache_creation_tokens']} tokens")
                continue
            v = _count(n.tokens[key])
            if v is None:
                counters["cache_retention_buckets_missing"] += 1
                missing[dst] = f"bucket {key!r} is {str(n.tokens[key])[:40]!r} in the session record, not a count"
            else:
                buckets[dst] = v
        if len(buckets) == 2 and "cache_creation_tokens" in cost and sum(buckets.values()) != cost["cache_creation_tokens"]:
            counters["cache_retention_buckets_inconsistent"] += 1
            for dst, v in buckets.items():
                missing[dst] = f"session record gives {v}, but the two buckets sum to {sum(buckets.values())} against {cost['cache_creation_tokens']} cache-write tokens; both withheld"
        else:
            cost.update(buckets)
        cost["basis"] = "measured"
    usd = n.usd if isinstance(n.usd, (int, float)) and not isinstance(n.usd, bool) and n.usd >= 0 else None
    if usd is not None:
        if cost is not None:
            cost["usd"] = float(usd)
        else:
            missing["usd"] = f"the graph carries {float(usd)} USD for this session but no token usage to place it against; not emitted"
    else:
        counters["attempts_unpriced"] += 1
        missing["usd"] = "not priced: the graph carries no dollar figure for this session (no price entry for the model, or no token stream to price); never counted as zero"
    return cost, missing


def _edge_evidence(e: GraphEdge, t: _Text) -> str:
    d = e.detail or {}
    # OCP readers carry the producer's evidence verbatim.  Native ingesters do
    # not set this key, so their established evidence wording is unchanged.
    if isinstance(d.get("evidence"), str):
        return t(d["evidence"])
    if e.kind == "spawn":
        return t(d.get("reason") or "Task tool_use id matched subagent meta")
    if e.kind == "launch":
        return t(f"{d.get('how')}; lag {d.get('lag_s')} s")
    if e.kind in ("dep", "fan_in"):
        # Scheduling edges are only ever read from a document, never inferred
        # from a session, so an edge that arrives without its source evidence
        # says that and nothing more; the artifact wording below would claim a
        # write and a read that no session shows.
        return t(f"{e.kind} edge declared by the source document, evidence not carried")
    return t(f"write ({d.get('write_tier')}) then read ({d.get('read_tier')}) {d.get('lag_s')} s later")


def _edge_record(e: GraphEdge, full: bool, t: _Text) -> dict:
    """The graph edge as it was, for `edges_not_emitted` (raw tier kept)."""
    d = e.detail or {}
    rec = {"from": e.src, "to": e.dst, "kind": e.kind, "tier": e.tier, "evidence": _edge_evidence(e, t)}
    if d.get("path"):
        rec["path"] = str(d["path"])
    if full and e.kind == "launch" and d.get("command"):
        rec["command"] = d["command"]
    return rec
