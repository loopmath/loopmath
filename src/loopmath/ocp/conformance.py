#!/usr/bin/env python3
"""OCP conformance checker (v0.1, v0.2 and v0.3 documents).

Moved into the package from spec/ocp_conformance.py (spec 01 section 3);
spec/ocp_conformance.py is now a shim over this module.

Checks an OCP run document twice over:

1. Schema validation, when the `jsonschema` package is importable, against the
   schema the document's `ocp` field selects: "0.1" -> ocp-v0.schema.json,
   "0.2" -> ocp-v0.2.schema.json, "0.3" -> ocp-v0.3.schema.json (package data
   under loopmath/ocp/schema/). Any other version is an error (E003); the
   referential checks below still run. Skipped (with a notice) when
   jsonschema is missing.
2. Referential rules the JSON Schema cannot express, from the schemas' own
   descriptions:
     - edge endpoints and attempt.node name existing node ids
     - event node/attempt refs name existing nodes/attempts
     - group.parent names an existing group and parent links are acyclic
     - dep and fan_in edges together form a DAG (spawn, launch and artifact
       edges are lineage, not scheduling, and may form cycles)
     - a terminal attempt's outcome.result equals its status
   v0.2 rules, run when `ocp` is at least "0.2" (a v0.1 document carrying any
   of these fields is treated as carrying unknown fields, which are ignored):
     - artifact ids are unique; artifact producer, writers and consumers
       name existing attempts
     - every edge carries a tier, whatever its kind and whoever the
       producer is, and the tier is one of
       verified, heuristic, reported
     - role.value, phase.value and artifact.kind.value may be null (unknown)
       with the tier still present; a null list field (top-level lists,
       artifact writers and consumers) is a finding, never an exception
     - an 'artifact' edge names an existing artifact and carries both
       from_attempt and to_attempt; they name existing attempts at its
       from / to nodes; from_attempt is the artifact's producer and
       to_attempt is one of its consumers
     - artifact.writers is non-empty with no duplicates, consumers has no
       duplicates, producer equals writers[0], a known (non-null) n_writes
       is at least the number of writers and a known n_reads at least the
       number of consumers. A null count is unknown: it is never treated as satisfying or failing the
       undercount check, and the schema accepts null for first_write_at,
       n_writes and n_reads
     - attempts need not carry origin, role or phase at all (a v0.1
       attempt has none; the extractor's own output always carries all
       three, and the extractor, not this checker, enforces that). Once a record is present, every field
       listed for it is required inside it, unknown being an explicit null
     - origin.launched_by names another existing attempt, and a 'launch'
       edge from the launcher's node to the launched node exists; an
       origin with external true and a non-null launched_by contradicts
       itself
     - cache_creation_5m_tokens + cache_creation_1h_tokens equals
       cache_creation_tokens whenever all three are present
     - when producer.name is the extractor ('loopmath', or the pre-rename
       'dagr', optionally suffixed after a non-word character): every cost
       record has basis 'measured' (E171). One exception, v0.3 and later
       with producer loopmath only (D48): basis 'allocated' is allowed when
       cost.ext["dev.loopmath.logmatch"] is present with tier 'heuristic'
       (the store's heuristic log match, D27), or with shared_session (a
       session several attempts name, its cost split among them; an object
       or, in a shared record, true; D88)
     - producer.capabilities is recommended; its absence is a warning, and a
       capability declared false while its element is present is an error
   The v0.2 consistency rules are strict: every internal inconsistency
   is an error. A string
   outside an open field's recommended vocabulary produces a warning.
   The capabilities absence warning
   joins v0.1's advisories: dangling cause.ref / outcome.via, terminal attempts
   missing an outcome, outcomes on non-terminal attempts, and events not
   ascending in time.
   v0.3 rules, run when `ocp` is at least "0.3" (OCP.md section 6.3):
     - E190 run.configuration.id is the canonical hash of its workflow and
       settings (loopmath.ocp.canonical); a workflow given by reference is
       resolved through the workflow catalog, and skipped when the catalog
       does not hold that id and version
     - E191 every top-level piece has a setting, unless it holds a nested
       workflow and has no role
     - E192 workflow edges join a piece and an artifact, name existing
       vertices, and form a DAG; piece and artifact ids do not collide
     - E193 control.gates name pieces; control.repair maps a gate to a piece
     - E194 the run is a member of its slate and shares the slate's
       base_commit; a preference's winner is a member or 'tie'; across
       files (validate_many), members of one slate share task.id and
       base_commit
     - E195 signal values match their kind, only a score may be null,
       fraction scores lie in [0, 1], the rule's score names no verdict or
       event signal and its requires name no score or event signal
     - W196 run.receipt is present but producer.name is not loopmath
     - capabilities task, configuration, signals, slate and receipt join E180
3. Extension namespace compatibility: an `ext` key beginning with a retired
   OCP or dagr producer prefix is an error naming the replacement prefix
   (E181). In v0.3 documents loopmath's namespace is 'dev.loopmath.' and a
   'dev.dagr.' key is still read, with warning W182.

Library use:  findings = validate_file(path)      # or validate_doc(dict)
              errors = [f for f in findings if f.level == "error"]
              findings = validate_many(docs)      # adds cross-file E194
CLI use:      python3 -m loopmath.ocp.conformance FILE [FILE ...]
              (or python3 spec/ocp_conformance.py FILE [FILE ...])
              exit 0 iff every file has zero errors (warnings do not fail).

Stdlib only; `jsonschema` is an optional extra.
"""

from __future__ import annotations

import json
import re
import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from .canonical import UnresolvedWorkflow, config_id, is_workflow_ref
from .contractv3 import is_contract_run
from .version import version_at_least

_HERE = Path(__file__).resolve().parent / "schema"
SCHEMA_PATHS = {
    "0.1": _HERE / "ocp-v0.schema.json",
    "0.2": _HERE / "ocp-v0.2.schema.json",
    "0.3": _HERE / "ocp-v0.3.schema.json",
}
SCHEMA_PATH = SCHEMA_PATHS["0.1"]  # kept for callers that pinned the v0.1 name
CURRENT_VERSION = "0.3"

TERMINAL_STATUSES = {
    "done", "failed", "rejected", "canceled", "settled_unverified", "lost",
}
DAG_EDGE_KINDS = {"dep", "fan_in"}  # spawn, launch and artifact edges are exempt
EVIDENCE_TIERS = ("verified", "heuristic", "reported")
TOKEN_COST_FIELDS = {
    "input_tokens", "cached_input_tokens", "cache_creation_tokens",
    "cache_creation_5m_tokens", "cache_creation_1h_tokens", "output_tokens",
    "reasoning_tokens",
}
RECOMMENDED_VOCABULARY = {
    "node.kind": frozenset((
        "impl", "review", "test", "gate", "question", "docs", "ship", "plan", "ops", "task",
        "unknown",
    )),
    "node.state": frozenset((
        "queued", "working", "review", "blocked", "done", "failed", "rejected", "canceled",
        "settled_unverified",
    )),
    "edge.kind": frozenset(("dep", "fan_in", "spawn", "launch", "artifact")),
    "artifact.kind.value": frozenset((
        "plan", "review", "spec", "report", "code", "test", "config", "doc", "data", "log",
        "other",
    )),
    "role.value": frozenset((
        "lead", "solo", "planner", "dev", "reviewer", "cli", "subagent",
    )),
    "phase.value": frozenset(("build", "post", "external")),
    "cost.basis": frozenset(("measured", "allocated")),
    "cause.type": frozenset((
        "initial", "sent_back", "gate_failed", "followup", "superseded", "other",
    )),
    "outcome.evidence": frozenset(("verified", "reported", "heuristic", "asserted")),
    "event.type": frozenset((
        "attempt_started", "attempt_settled", "promoted", "directive", "note", "launched",
        "artifact_written", "artifact_read",
    )),
}
# v0.3 additions to the recommended vocabularies above (checked in v0.3 documents only).
RECOMMENDED_VOCABULARY_V03 = {
    "event.type": RECOMMENDED_VOCABULARY["event.type"] | frozenset((
        "signal_observed", "receipt_written", "run_finished",
    )),
    "task.type": frozenset((
        "bug_fix", "feature", "refactor", "tests", "docs", "research", "infra", "data",
    )),
    # D17: late events are matched to a run by its commit or merge artifacts.
    # D91: the artifact kinds of workflow pieces are recorded as the workflow names them.
    "artifact.kind.value": RECOMMENDED_VOCABULARY["artifact.kind.value"] | frozenset((
        "commit", "merge", "issue", "repo", "diff", "verdict", "test_record",
    )),
    # D32: loopmath writes the workflow role words of its pieces verbatim.
    "role.value": RECOMMENDED_VOCABULARY["role.value"] | frozenset((
        "implementer", "tester", "referee", "worker",
    )),
}
VERDICT_VALUES = frozenset(("accept", "reject", "pass", "fail", "error"))
V03_CAPABILITIES = ("task", "configuration", "signals", "slate", "receipt")
# The extractor's own producer name; switches on the extractor-only rules.
_EXTRACTOR_NAME = re.compile(r"^(loopmath|dagr)(\W|$)")  # dagr: pre-rename producer name in recorded runs
# Only loopmath writes receipts (W196); 'dagr' never did.
_LOOPMATH_NAME = re.compile(r"^loopmath(\W|$)")
# The store's log-match evidence on an allocated cost (D27, D48).
LOGMATCH_KEY = "dev.loopmath.logmatch"

Finding = namedtuple("Finding", ["level", "code", "path", "message"])


def _err(code, path, message):
    return Finding("error", code, path, message)


def _warn(code, path, message):
    return Finding("warning", code, path, message)


def _known(value, ids):
    """`value in ids` for a set of string ids: any other value (a list, an object)
    is unknown, never a TypeError from hashing it."""
    return isinstance(value, str) and value in ids


def _parse_ts(value):
    """Return a datetime for an OCP timestamp string, or None."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _recommended_findings(doc):
    """Warn about strings outside the recommended open vocabularies (v0.2 and later)."""
    if not version_at_least(doc, "0.2"):
        return []
    findings = []
    vocabulary = dict(RECOMMENDED_VOCABULARY)
    if version_at_least(doc, "0.3"):
        vocabulary.update(RECOMMENDED_VOCABULARY_V03)

    def check(field, path, value):
        if isinstance(value, str) and value not in vocabulary[field]:
            findings.append(_warn(
                "W200",
                path,
                f"{field} value {value!r} is outside the recommended vocabulary; "
                "readers must accept it",
            ))

    def records(name):
        value = doc.get(name)
        return value if isinstance(value, list) else []

    for i, node in enumerate(records("nodes")):
        if isinstance(node, dict):
            check("node.kind", f"$['nodes'][{i}]['kind']", node.get("kind"))
            check("node.state", f"$['nodes'][{i}]['state']", node.get("state"))
    for i, edge in enumerate(records("edges")):
        if isinstance(edge, dict):
            check("edge.kind", f"$['edges'][{i}]['kind']", edge.get("kind", "dep"))
    for i, artifact in enumerate(records("artifacts")):
        kind = artifact.get("kind") if isinstance(artifact, dict) else None
        if isinstance(kind, dict):
            check("artifact.kind.value", f"$['artifacts'][{i}]['kind']['value']", kind.get("value"))
    for i, attempt in enumerate(records("attempts")):
        if not isinstance(attempt, dict):
            continue
        for record, field in (
            ("role", "role.value"),
            ("phase", "phase.value"),
            ("cost", "cost.basis"),
            ("cause", "cause.type"),
            ("outcome", "outcome.evidence"),
        ):
            value = attempt.get(record)
            if not isinstance(value, dict):
                continue
            key = field.rsplit(".", 1)[1]
            check(field, f"$['attempts'][{i}]['{record}']['{key}']", value.get(key))
    for i, event in enumerate(records("events")):
        if isinstance(event, dict):
            check("event.type", f"$['events'][{i}]['type']", event.get("type"))
    run = doc.get("run")
    task = run.get("task") if isinstance(run, dict) else None
    if "task.type" in vocabulary and isinstance(task, dict):
        check("task.type", "$['run']['task']['type']", task.get("type"))
    return findings


def _is_loopmath_producer(doc):
    producer = doc.get("producer")
    name = producer.get("name") if isinstance(producer, dict) else None
    return isinstance(name, str) and bool(_LOOPMATH_NAME.match(name))


def _logmatch_allocation(cost):
    """An allocated cost the store's log match accounts for.

    D48: a heuristic match (D27); a shared record keeps only `tier` (D56).
    D88: a session several attempts name, split among them, at any tier;
    `shared_session` is an object in the store and `true` in a shared record.
    """
    ext = cost.get("ext")
    match = ext.get(LOGMATCH_KEY) if isinstance(ext, dict) else None
    if cost.get("basis") != "allocated" or not isinstance(match, dict):
        return False
    shared = match.get("shared_session")
    return match.get("tier") == "heuristic" or shared is True or (isinstance(shared, dict) and bool(shared))


def is_extractor_producer(doc):
    """True when the document says the dagr extractor emitted it."""
    producer = doc.get("producer")
    name = producer.get("name") if isinstance(producer, dict) else None
    return isinstance(name, str) and bool(_EXTRACTOR_NAME.match(name))


def schema_path_for(doc):
    """The schema file the document's `ocp` field selects, or None."""
    version = doc.get("ocp")
    return SCHEMA_PATHS.get(version) if isinstance(version, str) else None


_VALIDATORS = {}


def _validator(jsonschema, schema_path):
    """A checked validator for the schema file, reused while the file is unchanged."""
    path = Path(schema_path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    validator = _VALIDATORS.get(key)
    if validator is None:
        schema = json.loads(path.read_text())
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = _VALIDATORS[key] = validator_cls(schema)
    return validator


def _schema_findings(doc, schema_path):
    """Schema validation via jsonschema, if available."""
    try:
        import jsonschema
    except ImportError:
        return [_warn("W001", "$", "jsonschema not installed; schema validation skipped")]
    try:
        validator = _validator(jsonschema, schema_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [_err("E011", "$", f"cannot load schema {schema_path}: {exc}")]
    findings = []
    for error in sorted(validator.iter_errors(doc), key=str):
        loc = "$" + "".join(
            f"[{p!r}]" if isinstance(p, str) else f"[{p}]" for p in error.absolute_path
        )
        findings.append(_err("E010", loc, f"schema: {error.message}"))
    return findings


def _extension_prefix_findings(doc):
    """Reject retired prefixes on keys inside any nested `ext` object.

    From v0.3 loopmath's namespace is 'dev.loopmath.': the retired bare prefix
    names it as the replacement, and a 'dev.dagr.' key is read with warning
    W182 (for two minor versions). In v0.2 and earlier 'dev.dagr.' is still
    the loopmath namespace and draws no warning.
    """
    findings = []
    v03 = version_at_least(doc, "0.3")
    replacements = (
        ("da" "gr.", "dev.loopmath." if v03 else "dev.dagr."),
        ("oc" "p.", "io.orchestrationcontextprotocol."),
    )

    def walk(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}[{key!r}]"
                if key == "ext":
                    if isinstance(child, dict):
                        for extension_key in child:
                            if not isinstance(extension_key, str):
                                continue
                            for retired, expected in replacements:
                                if extension_key.startswith(retired):
                                    findings.append(_err(
                                        "E181",
                                        f"{child_path}[{extension_key!r}]",
                                        f"extension key {extension_key!r} uses retired prefix "
                                        f"{retired!r}; expected prefix {expected!r}",
                                    ))
                                    break
                            else:
                                if v03 and extension_key.startswith("dev.dagr."):
                                    findings.append(_warn(
                                        "W182",
                                        f"{child_path}[{extension_key!r}]",
                                        f"extension key {extension_key!r} uses the pre-rename "
                                        "namespace 'dev.dagr.'; it is still read through v0.4, "
                                        "write 'dev.loopmath.' instead",
                                    ))
                    # Stop here so foreign payloads below a known ext slot are
                    # not judged, while ext objects below unknown members are.
                    # See test_extension_prefix_rule_does_not_enter_foreign_payload
                    # and test_extension_prefix_decision_pins_unknown_*.
                    continue
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(doc, "$")
    return findings


def _detect_dag_cycle(node_ids, edges):
    """Kahn's algorithm over dep/fan_in edges; returns node ids stuck in cycles."""
    adjacency = {n: [] for n in node_ids}
    indegree = {n: 0 for n in node_ids}
    for frm, to in edges:
        adjacency[frm].append(to)
        indegree[to] += 1
    queue = [n for n, d in indegree.items() if d == 0]
    seen = 0
    while queue:
        current = queue.pop()
        seen += 1
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if seen == len(indegree):
        return []
    return sorted(n for n, d in indegree.items() if d > 0)


def _referential_findings(doc):
    findings = []
    # Every v0.2 rule is gated on the declared version: artifacts, launch
    # edges, origin, edge tiers and the cache buckets are unknown fields to a
    # v0.1 consumer and must stay ignored there.
    v02 = version_at_least(doc, "0.2")
    extractor = v02 and is_extractor_producer(doc)
    v03_loopmath = version_at_least(doc, "0.3") and _is_loopmath_producer(doc)

    producer = doc.get("producer")
    capabilities = producer.get("capabilities") if isinstance(producer, dict) else None
    if v02 and isinstance(producer, dict) and "capabilities" not in producer:
        findings.append(_warn(
            "W180", "$['producer']",
            "producer.capabilities is absent; consumers cannot distinguish unsupported elements",
        ))

    def ids_of(entities, label, code):
        seen = {}
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue
            eid = entity.get("id")
            if not isinstance(eid, str):
                continue
            if eid in seen:
                findings.append(_err(code, f"$['{label}'][{i}]",
                                     f"duplicate {label[:-1]} id {eid!r} (first at index {seen[eid]})"))
            else:
                seen[eid] = i
        return seen

    def as_list(key):
        value = doc.get(key)
        if isinstance(value, list):
            return value
        if key in doc:
            # A null (or otherwise non-list) list field is a finding, never
            # an exception; the referential checks then treat it as empty.
            findings.append(_err("E104", f"$['{key}']",
                                 f"{key} is {'null' if value is None else type(value).__name__}, "
                                 "not a list"))
        return []

    nodes = as_list("nodes")
    edges = as_list("edges")
    attempts = as_list("attempts")
    events = as_list("events")
    groups = as_list("groups")
    artifacts = as_list("artifacts") if v02 else []

    if v02 and isinstance(capabilities, dict):
        present = {
            "groups": bool(groups),
            "events": bool(events),
            "artifacts": bool(artifacts),
            "edges_dep": any(
                isinstance(edge, dict) and _known(edge.get("kind", "dep"), DAG_EDGE_KINDS)
                for edge in edges
            ),
            "edges_spawn": any(
                isinstance(edge, dict) and edge.get("kind") == "spawn" for edge in edges
            ),
            "edges_launch": any(
                isinstance(edge, dict) and edge.get("kind") == "launch" for edge in edges
            ),
            "edges_artifact": any(
                isinstance(edge, dict) and edge.get("kind") == "artifact" for edge in edges
            ),
            "cost_usd": any(
                isinstance(attempt, dict)
                and isinstance(attempt.get("cost"), dict)
                and "usd" in attempt["cost"]
                for attempt in attempts
            ),
            "cost_tokens": any(
                isinstance(attempt, dict)
                and isinstance(attempt.get("cost"), dict)
                and bool(TOKEN_COST_FIELDS.intersection(attempt["cost"]))
                for attempt in attempts
            ),
            "outcome_evidence": any(
                isinstance(attempt, dict)
                and isinstance(attempt.get("outcome"), dict)
                and "evidence" in attempt["outcome"]
                for attempt in attempts
            ),
        }
        if version_at_least(doc, "0.3"):
            run = doc.get("run") if isinstance(doc.get("run"), dict) else {}
            present.update({
                "task": "task" in run,
                "configuration": "configuration" in run,
                "signals": bool(run.get("signals")),
                "slate": "slate" in run,
                "receipt": "receipt" in run,
            })
        for capability, has_element in present.items():
            if capabilities.get(capability) is False and has_element:
                findings.append(_err(
                    "E180", f"$['producer']['capabilities']['{capability}']",
                    f"capability {capability!r} is false but the document contains that element",
                ))

    node_ids = ids_of(nodes, "nodes", "E100")
    attempt_ids = ids_of(attempts, "attempts", "E101")
    group_ids = ids_of(groups, "groups", "E102")
    artifact_ids = ids_of(artifacts, "artifacts", "E103")
    attempt_node = {
        a["id"]: a.get("node") for a in attempts
        if isinstance(a, dict) and isinstance(a.get("id"), str)
    }
    artifact_by_id = {
        a["id"]: a for a in artifacts
        if isinstance(a, dict) and isinstance(a.get("id"), str)
    }

    # Node group refs
    for i, node in enumerate(nodes):
        if isinstance(node, dict) and isinstance(node.get("group"), str) \
                and node["group"] not in group_ids:
            findings.append(_err("E115", f"$['nodes'][{i}]['group']",
                                 f"node {node.get('id')!r} names unknown group {node['group']!r}"))

    # Edge endpoints + DAG over dep/fan_in + v0.2 edge fields
    dag_edges = []
    launch_pairs = set()  # (from node, to node) of launch edges, for origin checks
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        loc = f"$['edges'][{i}]"
        frm, to = edge.get("from"), edge.get("to")
        kind = edge.get("kind", "dep")
        ok = True
        if not _known(frm, node_ids):
            findings.append(_err("E110", f"{loc}['from']",
                                 f"edge.from names unknown node {frm!r}"))
            ok = False
        if not _known(to, node_ids):
            findings.append(_err("E111", f"{loc}['to']",
                                 f"edge.to names unknown node {to!r}"))
            ok = False
        if ok and _known(kind, DAG_EDGE_KINDS):
            dag_edges.append((frm, to))
        if ok and kind == "launch":
            launch_pairs.add((frm, to))
        if not v02:
            continue
        if edge.get("tier") not in EVIDENCE_TIERS:
            findings.append(_err("E160", loc,
                                 f"edge {frm!r}->{to!r} ({kind}) has no evidence tier "
                                 f"(got {edge.get('tier')!r}); every v0.2 edge carries one of "
                                 + ", ".join(EVIDENCE_TIERS)))
        artifact_ref = edge.get("artifact")
        artifact = None
        if kind == "artifact":
            if not isinstance(artifact_ref, str):
                findings.append(_err("E116", loc,
                                     f"artifact edge {frm!r}->{to!r} names no artifact"))
            elif artifact_ref not in artifact_ids:
                findings.append(_err("E117", f"{loc}['artifact']",
                                     f"edge names unknown artifact {artifact_ref!r}"))
            else:
                artifact = artifact_by_id[artifact_ref]
        for field, end, node_ref in (("from_attempt", "from", frm), ("to_attempt", "to", to)):
            ref = edge.get(field)
            if ref is None:
                if kind == "artifact":
                    findings.append(_err("E125", loc,
                                         f"artifact edge {frm!r}->{to!r} has no {field}; "
                                         "an artifact edge runs from the producing attempt "
                                         "to the consuming attempt"))
                continue
            if not _known(ref, attempt_ids):
                findings.append(_err("E123", f"{loc}['{field}']",
                                     f"edge names unknown attempt {ref!r}"))
                continue
            if attempt_node.get(ref) != node_ref:
                findings.append(_err("E124", f"{loc}['{field}']",
                                     f"attempt {ref!r} is at node {attempt_node.get(ref)!r}, "
                                     f"not at edge.{end} {node_ref!r}"))
            if artifact is None:
                continue
            if field == "from_attempt":
                if ref != artifact.get("producer"):
                    findings.append(_err("E164", f"{loc}['{field}']",
                                         f"attempt {ref!r} is not the producer "
                                         f"{artifact.get('producer')!r} of artifact "
                                         f"{artifact_ref!r}; an artifact edge runs from the "
                                         "producer, and the edge and the artifact record "
                                         "contradict each other"))
            else:
                consumers = artifact.get("consumers")
                if not isinstance(consumers, list) or ref not in consumers:
                    findings.append(_err("E164", f"{loc}['{field}']",
                                         f"attempt {ref!r} is not among artifact "
                                         f"{artifact_ref!r} consumers; the edge and the "
                                         "artifact record contradict each other"))
    cyclic = _detect_dag_cycle(set(node_ids), dag_edges)
    if cyclic:
        findings.append(_err("E122", "$['edges']",
                             "dep/fan_in edges contain a cycle through nodes: "
                             + ", ".join(cyclic)))

    # Group parent refs + acyclicity
    parent_of = {}
    for i, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        parent = group.get("parent")
        if parent is None:
            continue
        if not _known(parent, group_ids):
            findings.append(_err("E120", f"$['groups'][{i}]['parent']",
                                 f"group {group.get('id')!r} names unknown parent {parent!r}"))
        elif isinstance(group.get("id"), str):
            parent_of[group["id"]] = parent
    for start in parent_of:
        trail, current = set(), start
        while current in parent_of:
            if current in trail:
                findings.append(_err("E121", "$['groups']",
                                     f"group parent cycle through {current!r}"))
                break
            trail.add(current)
            current = parent_of[current]
        else:
            continue
        break  # report one cycle, not one per member

    # Attempts: node refs, cause refs, outcome/status agreement, origin, cost
    for i, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            continue
        loc = f"$['attempts'][{i}]"
        aid = attempt.get("id")
        if not _known(attempt.get("node"), node_ids):
            findings.append(_err("E112", f"{loc}['node']",
                                 f"attempt {aid!r} names unknown node {attempt.get('node')!r}"))
        cause = attempt.get("cause")
        if isinstance(cause, dict) and isinstance(cause.get("ref"), str) \
                and cause["ref"] not in attempt_ids:
            findings.append(_warn("W140", f"{loc}['cause']['ref']",
                                  f"attempt {aid!r} cause.ref names unknown attempt {cause['ref']!r}"))
        status = attempt.get("status")
        outcome = attempt.get("outcome")
        if isinstance(outcome, dict):
            via = outcome.get("via")
            if isinstance(via, str) and via not in node_ids:
                findings.append(_warn("W141", f"{loc}['outcome']['via']",
                                      f"attempt {aid!r} outcome.via names unknown node {via!r}"))
        if _known(status, TERMINAL_STATUSES):
            if not isinstance(outcome, dict):
                findings.append(_warn("W131", loc,
                                      f"terminal attempt {aid!r} (status {status!r}) has no outcome record"))
            elif outcome.get("result") != status:
                findings.append(_err("E130", f"{loc}['outcome']['result']",
                                     f"attempt {aid!r}: outcome.result {outcome.get('result')!r} "
                                     f"!= terminal status {status!r}"))
        elif isinstance(outcome, dict):
            findings.append(_warn("W132", f"{loc}['outcome']",
                                  f"attempt {aid!r} carries an outcome but status {status!r} is not terminal"))

        if not v02:
            continue
        origin = attempt.get("origin")
        if isinstance(origin, dict) and origin.get("launched_by") is not None:
            launcher = origin["launched_by"]
            if origin.get("external") is True:
                findings.append(_err("E165", f"{loc}['origin']",
                                     f"attempt {aid!r} origin says external (launched by an "
                                     f"unscanned session) yet names launcher {launcher!r}; "
                                     "the two contradict each other"))
            launch_pair = (attempt_node.get(launcher) if isinstance(launcher, str) else None, attempt.get("node"))
            if not _known(launcher, attempt_ids) or launcher == aid:
                findings.append(_err("E119", f"{loc}['origin']['launched_by']",
                                     f"attempt {aid!r} origin.launched_by names "
                                     f"{'itself' if launcher == aid else 'unknown attempt ' + repr(launcher)}"))
            elif not (all(isinstance(end, str) for end in launch_pair) and launch_pair in launch_pairs):
                findings.append(_err("E161", f"{loc}['origin']['launched_by']",
                                     f"attempt {aid!r} says {launcher!r} launched it but no launch edge "
                                     f"{attempt_node.get(launcher)!r}->{attempt.get('node')!r} exists"))

        cost = attempt.get("cost")
        if isinstance(cost, dict):
            if extractor and cost.get("basis") != "measured" and not (
                    v03_loopmath and _logmatch_allocation(cost)):
                findings.append(_err("E171", f"{loc}['cost']['basis']",
                                     f"attempt {aid!r}: cost.basis is {cost.get('basis')!r}; "
                                     "the extractor must emit 'measured'"))
            buckets = (cost.get("cache_creation_5m_tokens"), cost.get("cache_creation_1h_tokens"))
            total = cost.get("cache_creation_tokens")
            if all(isinstance(v, int) for v in buckets) and isinstance(total, int) \
                    and sum(buckets) != total:
                findings.append(_err("E170", f"{loc}['cost']",
                                     f"attempt {aid!r}: cache_creation_5m_tokens + cache_creation_1h_tokens "
                                     f"= {sum(buckets)} but cache_creation_tokens = {total}"))

    # Artifacts: attempt refs, producer == writers[0], unique members, counts
    for i, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        loc = f"$['artifacts'][{i}]"
        art_id = artifact.get("id")
        producer = artifact.get("producer")
        if not _known(producer, attempt_ids):
            findings.append(_err("E118", f"{loc}['producer']",
                                 f"artifact {art_id!r} producer names unknown attempt {producer!r}"))
        for field in ("writers", "consumers"):
            members = artifact.get(field)
            if not isinstance(members, list):
                if field in artifact:
                    # null-safe: a null list
                    # field is a finding, never a raised exception
                    findings.append(_err("E167", f"{loc}['{field}']",
                                         f"artifact {art_id!r} {field} is "
                                         f"{'null' if members is None else type(members).__name__}, "
                                         "not a list of attempt ids"))
                continue
            seen_members = set()
            for j, ref in enumerate(members):
                if not _known(ref, attempt_ids):
                    findings.append(_err("E118", f"{loc}['{field}'][{j}]",
                                         f"artifact {art_id!r} {field} name unknown attempt {ref!r}"))
                scalar = not isinstance(ref, (list, dict))  # a JSON scalar hashes; the schema flags the rest
                if scalar and ref in seen_members:
                    findings.append(_err("E166", f"{loc}['{field}'][{j}]",
                                         f"artifact {art_id!r} lists {ref!r} twice in {field}; "
                                         "each attempt is listed once"))
                if scalar:
                    seen_members.add(ref)
            count_field = "n_writes" if field == "writers" else "n_reads"
            count = artifact.get(count_field)
            # A null count is unknown: neither an undercount nor a match.
            # Only a known integer is held against the listed members.
            if isinstance(count, int) and not isinstance(count, bool) and count < len(members):
                findings.append(_err("E163", f"{loc}['{count_field}']",
                                     f"artifact {art_id!r}: {count_field} = {count} but "
                                     f"{len(members)} {field} are listed; the count undercounts"))
        writers = artifact.get("writers")
        if isinstance(writers, list):
            if not writers:
                findings.append(_err("E166", f"{loc}['writers']",
                                     f"artifact {art_id!r} has no writers; an artifact exists "
                                     "because an attempt wrote it"))
            elif producer != writers[0]:
                findings.append(_err("E162", f"{loc}['producer']",
                                     f"artifact {art_id!r} producer {producer!r} is not writers[0] "
                                     f"({writers[0]!r}); the producer is the first writer"))

    # Events: refs exist, timestamps ascend
    previous_ts = None
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if isinstance(event.get("node"), str) and event["node"] not in node_ids:
            findings.append(_err("E113", f"$['events'][{i}]['node']",
                                 f"event names unknown node {event['node']!r}"))
        if isinstance(event.get("attempt"), str) and event["attempt"] not in attempt_ids:
            findings.append(_err("E114", f"$['events'][{i}]['attempt']",
                                 f"event names unknown attempt {event['attempt']!r}"))
        ts = _parse_ts(event.get("at"))
        if ts is not None and previous_ts is not None and ts < previous_ts:
            findings.append(_warn("W150", f"$['events'][{i}]['at']",
                                  "events not ascending in time"))
        if ts is not None:
            previous_ts = ts

    return findings


def catalog_resolver(ref, version=None):
    """Resolve a workflow reference through loopmath's workflow catalog, as OCP JSON.

    Returns None when the catalog does not hold that id and version.
    """
    if not isinstance(ref, str):
        return None
    from ..workflows.format import catalog
    from .emit import workflow_to_ocp

    workflow = catalog().get(ref)
    if workflow is None or (version is not None and workflow.version != version):
        return None
    return workflow_to_ocp(workflow)


def _run_of(doc):
    run = doc.get("run")
    return run if isinstance(run, dict) else {}


def _records(value):
    return value if isinstance(value, list) else []


def _workflow_findings(workflow, loc):
    """E192 (typed, known, acyclic edges; unique vertex ids) and E193 (gates, repair)."""
    findings = []
    vertex_type = {}
    for field, kind in (("pieces", "piece"), ("artifacts", "artifact")):
        for i, record in enumerate(_records(workflow.get(field))):
            vid = record.get("id") if isinstance(record, dict) else None
            if not isinstance(vid, str):
                continue
            if vid in vertex_type:
                findings.append(_err("E192", f"{loc}['{field}'][{i}]['id']",
                                     f"vertex id {vid!r} is used twice (as a {vertex_type[vid]} and as "
                                     f"a {kind}); piece and artifact ids are one namespace"))
                continue
            vertex_type[vid] = kind
    dag_edges = []
    for j, edge in enumerate(_records(workflow.get("edges"))):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue  # the schema reports the shape
        frm, to = edge
        types = tuple(vertex_type.get(v) if isinstance(v, str) else None for v in (frm, to))
        if None in types:
            unknown = frm if types[0] is None else to
            findings.append(_err("E192", f"{loc}['edges'][{j}]",
                                 f"workflow edge {frm!r}->{to!r} names unknown vertex {unknown!r}"))
        elif types[0] == types[1]:
            findings.append(_err("E192", f"{loc}['edges'][{j}]",
                                 f"workflow edge {frm!r}->{to!r} joins two {types[0]}s; an edge joins "
                                 "a piece and an artifact (produces or consumes)"))
        else:
            dag_edges.append((frm, to))
    cyclic = _detect_dag_cycle(set(vertex_type), dag_edges)
    if cyclic:
        findings.append(_err("E192", f"{loc}['edges']",
                             "workflow edges contain a cycle through: " + ", ".join(cyclic)
                             + "; the repair loop belongs in control.repair"))

    control = workflow.get("control")
    if isinstance(control, dict):
        gates = [g for g in _records(control.get("gates")) if isinstance(g, str)]
        for k, gate in enumerate(_records(control.get("gates"))):
            if isinstance(gate, str) and vertex_type.get(gate) != "piece":
                findings.append(_err("E193", f"{loc}['control']['gates'][{k}]",
                                     f"gate {gate!r} names no piece of the workflow"))
        repair = control.get("repair")
        if isinstance(repair, dict):
            for gate, target in repair.items():
                if gate not in gates:
                    findings.append(_err("E193", f"{loc}['control']['repair'][{gate!r}]",
                                         f"repair key {gate!r} is not one of control.gates"))
                if not isinstance(target, str) or vertex_type.get(target) != "piece":
                    findings.append(_err("E193", f"{loc}['control']['repair'][{gate!r}]",
                                         f"repair target {target!r} names no piece of the workflow"))

    for i, piece in enumerate(_records(workflow.get("pieces"))):
        nested = piece.get("workflow") if isinstance(piece, dict) else None
        if isinstance(nested, dict) and not is_workflow_ref(nested):
            findings.extend(_workflow_findings(nested, f"{loc}['pieces'][{i}]['workflow']"))
    return findings


def _configuration_findings(configuration, resolve):
    """E190 (id is the canonical hash), E191 (every piece has a setting), E192, E193."""
    if not isinstance(configuration, dict):
        return []
    loc = "$['run']['configuration']"
    workflow = configuration.get("workflow")
    if is_workflow_ref(workflow):
        workflow = resolve(workflow.get("ref"), workflow.get("version")) if resolve else None
        inline = False
    else:
        inline = True
    if not isinstance(workflow, dict):
        return []
    findings = _workflow_findings(workflow, f"{loc}['workflow']") if inline else []

    settings = configuration.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    for piece in _records(workflow.get("pieces")):
        if not isinstance(piece, dict) or ("workflow" in piece and "role" not in piece):
            continue  # a piece holding a nested workflow needs no setting of its own
        pid = piece.get("id")
        if isinstance(pid, str) and not isinstance(settings.get(pid), dict):
            findings.append(_err("E191", f"{loc}['settings']",
                                 f"piece {pid!r} has no setting; every piece vertex has one"))

    cid = configuration.get("id")
    if isinstance(cid, str):
        try:
            expected = config_id(configuration.get("workflow"), settings, resolve=resolve)
        except (UnresolvedWorkflow, ValueError, TypeError, AttributeError, KeyError):
            expected = None  # malformed shape: the schema findings say why
        if expected is not None and cid != expected:
            findings.append(_err("E190", f"{loc}['id']",
                                 f"configuration id {cid!r} is not the canonical hash of its workflow "
                                 f"and settings ({expected!r}); see loopmath.ocp.canonical"))
    return findings


def _slate_findings(run):
    """E194 within one document: membership, base commit, preference winners."""
    findings = []
    slate = run.get("slate")
    members = None
    if isinstance(slate, dict):
        members = slate.get("members") if isinstance(slate.get("members"), list) else None
        run_id = run.get("id")
        if members is not None and isinstance(run_id, str) and run_id not in members:
            findings.append(_err("E194", "$['run']['slate']['members']",
                                 f"run {run_id!r} is not among its slate's members"))
        task = run.get("task")
        task_commit = task.get("base_commit") if isinstance(task, dict) else None
        slate_commit = slate.get("base_commit")
        if isinstance(task_commit, str) and isinstance(slate_commit, str) and task_commit != slate_commit:
            findings.append(_err("E194", "$['run']['task']['base_commit']",
                                 f"task.base_commit {task_commit!r} differs from the slate's base_commit "
                                 f"{slate_commit!r}; slate members start from one commit"))
    for i, preference in enumerate(_records(run.get("preferences"))):
        if not isinstance(preference, dict):
            continue
        loc = f"$['run']['preferences'][{i}]"
        if isinstance(slate, dict) and isinstance(preference.get("slate"), str) \
                and isinstance(slate.get("id"), str) and preference["slate"] != slate["id"]:
            findings.append(_err("E194", f"{loc}['slate']",
                                 f"preference names slate {preference['slate']!r} but this run is in "
                                 f"slate {slate['id']!r}"))
        pool = preference.get("members") if isinstance(preference.get("members"), list) else members
        winner = preference.get("winner")
        if isinstance(winner, str) and winner != "tie" and pool is not None and winner not in pool:
            findings.append(_err("E194", f"{loc}['winner']",
                                 f"preference winner {winner!r} is neither a member nor 'tie'"))
    return findings


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _signal_findings(run):
    """E195: signal values by kind, fraction range, and the rule's names by kind."""
    findings = []
    kinds_by_name = {}
    for i, signal in enumerate(_records(run.get("signals"))):
        if not isinstance(signal, dict):
            continue
        loc = f"$['run']['signals'][{i}]['value']"
        kind, name, value = signal.get("kind"), signal.get("name"), signal.get("value")
        if kind == "verdict" and not (isinstance(value, str) and value in VERDICT_VALUES):
            findings.append(_err("E195", loc, f"verdict {name!r} has value {value!r}; a verdict is one of "
                                 + ", ".join(sorted(VERDICT_VALUES))))
        elif kind == "score" and value is not None:
            if not _is_number(value):
                findings.append(_err("E195", loc, f"score {name!r} has value {value!r}; a score is a number, "
                                     "or null when declared and not measured yet"))
            elif signal.get("scale") == "fraction" and not 0 <= value <= 1:
                findings.append(_err("E195", loc, f"score {name!r} has scale 'fraction' but value {value!r} "
                                     "is outside [0, 1]"))
        elif kind == "event" and not isinstance(value, str):
            findings.append(_err("E195", loc, f"event {name!r} has value {value!r}; an event's value is a "
                                 "reference string (only a score may be null)"))
        if isinstance(name, str) and kind in ("verdict", "score", "event"):
            kinds_by_name.setdefault(name, set()).add(kind)

    rule = run.get("acceptance_rule")
    if not isinstance(rule, dict):
        return findings
    score = rule.get("score")
    if isinstance(score, dict):
        name = score.get("name")
        other = sorted(kinds_by_name.get(name, set()) - {"score"}) if isinstance(name, str) else []
        if other:
            findings.append(_err("E195", "$['run']['acceptance_rule']['score']['name']",
                                 f"the rule's score {name!r} names a {' and '.join(other)} signal, not a score"))
        target = score.get("target")
        if score.get("scale") == "fraction" and _is_number(target) and not 0 <= target <= 1:
            findings.append(_err("E195", "$['run']['acceptance_rule']['score']['target']",
                                 f"the rule's score has scale 'fraction' but target {target!r} is outside [0, 1]"))
    for j, name in enumerate(_records(rule.get("requires"))):
        other = sorted(kinds_by_name.get(name, set()) - {"verdict"}) if isinstance(name, str) else []
        if other:
            findings.append(_err("E195", f"$['run']['acceptance_rule']['requires'][{j}]",
                                 f"the rule requires {name!r}, which is a {' and '.join(other)} signal; "
                                 "requires names verdicts"))
    return findings


def _v03_findings(doc, resolve=None):
    """Rules E190 to E195 and W196, for documents at version 0.3 or later."""
    if not version_at_least(doc, "0.3"):
        return []
    run = _run_of(doc)
    findings = []
    findings.extend(_configuration_findings(run.get("configuration"), resolve))
    findings.extend(_slate_findings(run))
    findings.extend(_signal_findings(run))
    producer = doc.get("producer")
    name = producer.get("name") if isinstance(producer, dict) else None
    if "receipt" in run and not (isinstance(name, str) and _LOOPMATH_NAME.match(name)):
        findings.append(_warn("W196", "$['run']['receipt']",
                              f"run.receipt is written only by loopmath, but producer.name is {name!r}"))
    return findings


def _slate_key(doc):
    run = _run_of(doc)
    task = run.get("task") if isinstance(run.get("task"), dict) else {}
    slate = run.get("slate") if isinstance(run.get("slate"), dict) else {}
    commit = task.get("base_commit") if isinstance(task.get("base_commit"), str) else slate.get("base_commit")
    return task.get("id"), commit


def validate_many(docs, schema_path=None, resolve=catalog_resolver):
    """Validate several documents; adds the cross-file part of E194.

    Returns one findings list per document, in order. Documents whose
    run.slate.id match are members of one slate: they must share task.id and
    base_commit (task.base_commit, else slate.base_commit).
    """
    results = [validate_doc(doc, schema_path, resolve=resolve) for doc in docs]
    by_slate = {}
    for index, doc in enumerate(docs):
        if not isinstance(doc, dict) or not version_at_least(doc, "0.3"):
            continue
        slate = _run_of(doc).get("slate")
        if isinstance(slate, dict) and isinstance(slate.get("id"), str):
            by_slate.setdefault(slate["id"], []).append(index)
    for slate_id, indexes in by_slate.items():
        first = indexes[0]
        first_task, first_commit = _slate_key(docs[first])
        for index in indexes[1:]:
            task_id, commit = _slate_key(docs[index])
            other = _run_of(docs[first]).get("id")
            if task_id != first_task:
                results[index].append(_err("E194", "$['run']['task']['id']",
                                           f"slate {slate_id!r}: task.id {task_id!r} differs from member "
                                           f"{other!r} ({first_task!r}); pair members share task.id"))
            if commit != first_commit:
                results[index].append(_err("E194", "$['run']['task']['base_commit']",
                                           f"slate {slate_id!r}: base_commit {commit!r} differs from member "
                                           f"{other!r} ({first_commit!r}); pair members share base_commit"))
    return results


def validate_doc(doc, schema_path=None, resolve=catalog_resolver):
    """Validate one already-parsed OCP document. Returns a list of Finding tuples.

    `schema_path` overrides the schema the document's `ocp` field selects.
    `resolve(ref, version)` returns an inline OCP workflow for a workflow
    reference (default: loopmath's catalog); None skips E190 and E191 for it.
    A document conforms when no finding has level 'error'.
    """
    if not isinstance(doc, dict):
        return [_err("E002", "$", "top-level JSON value is not an object")]
    findings = []
    if schema_path is None:
        schema_path = schema_path_for(doc)
        if schema_path is None:
            known = f"known: {', '.join(sorted(SCHEMA_PATHS))}"
            if "ocp" not in doc:
                message = f"no 'ocp' version field; {known}"
                if is_contract_run(doc):
                    message += " (this is a contract run file: 'loopmath ocp migrate' converts it to OCP)"
            else:
                message = f"unknown OCP version {doc['ocp']!r}; {known}"
            findings.append(_err("E003", "$['ocp']", message))
    if schema_path is not None:
        findings.extend(_schema_findings(doc, schema_path))
    findings.extend(_extension_prefix_findings(doc))
    findings.extend(_referential_findings(doc))
    findings.extend(_v03_findings(doc, resolve))
    findings.extend(_recommended_findings(doc))
    return findings


def read_file(path):
    """(doc, None) for a readable JSON file, else (None, E001 finding)."""
    try:
        return json.loads(Path(path).read_text()), None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, _err("E001", "$", f"cannot read/parse {path}: {exc}")


def validate_file(path, schema_path=None, resolve=catalog_resolver):
    """Validate one OCP document on disk. Returns a list of Finding tuples.

    A file conforms when no finding has level 'error'.
    """
    doc, error = read_file(path)
    if error is not None:
        return [error]
    return validate_doc(doc, schema_path, resolve=resolve)


def validate_files(paths, schema_path=None, resolve=catalog_resolver):
    """Validate several files together (cross-file E194). One findings list per path."""
    read = [read_file(path) for path in paths]
    readable = [i for i, (doc, error) in enumerate(read) if error is None]
    checked = validate_many([read[i][0] for i in readable], schema_path, resolve=resolve)
    results = [[error] if error is not None else None for _, error in read]
    for i, findings in zip(readable, checked):
        results[i] = findings
    return results


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    any_errors = False
    for path, findings in zip(argv, validate_files(argv)):
        errors = [f for f in findings if f.level == "error"]
        warnings = [f for f in findings if f.level == "warning"]
        verdict = "PASS" if not errors else "FAIL"
        print(f"{verdict}  {path}  ({len(errors)} errors, {len(warnings)} warnings)")
        for f in findings:
            print(f"  {f.level} {f.code} {f.path}: {f.message}")
        any_errors = any_errors or bool(errors)
    return 1 if any_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
