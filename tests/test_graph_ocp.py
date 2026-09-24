"""`graph/ocp.py`: the workflow graph as an OCP v0.2 document (spec section 5, P2).

The skeleton fixture graph (tests/fixtures/graph/skeleton) covers a lead with
spawned subagents, a launched codex session, an external launcher, unlabeled
sessions and two artifacts; a hand-built graph pins the cases the fixture does
not reach (a later writer's artifact edge, unparseable timestamps, long paths).
Every assertion names a value the emitter must produce, and the document is
run through the reference conformance checker.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from loopmath.graph import extract, sanitize, to_ocp
from loopmath.graph.ocp import attempt_id
from loopmath.graph.schema import Artifact, Graph, GraphEdge, GraphNode
from loopmath.ocp.emit import validate_strict
from tests.test_graph_extract import ALPHA, skeleton_records

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ocp_conformance", ROOT / "spec" / "ocp_conformance.py")
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)

REQUIRED_ORIGIN = {"launched_by", "workspace", "external", "how", "tier"}
REQUIRED_LABEL = {"value", "tier", "evidence"}
TIERS = {"verified", "heuristic", "reported"}


def errors(doc: dict) -> list:
    return [f for f in conf.validate_doc(doc) if f.level == "error"]


@pytest.fixture(scope="module")
def g() -> Graph:
    return extract(skeleton_records(), workspaces=[ALPHA])


@pytest.fixture(scope="module")
def doc(g) -> dict:
    return to_ocp(g, producer=None, privacy="metadata_only")


def _as_pre_03(doc: dict) -> dict:
    """`doc` with every graph writer key renamed to its pre-0.3 `dev.dagr.` name."""
    text = json.dumps(doc).replace('"dev.loopmath.graph"', '"dev.dagr.graph"').replace('"dev.loopmath.artifact"', '"dev.dagr.artifact"')
    return json.loads(text)


def _attempt(doc, node_id):
    return next(a for a in doc["attempts"] if a["node"] == node_id)


def test_document_passes_conformance_as_the_extractor(doc):
    assert doc["ocp"] == "0.3"
    assert doc["producer"]["name"] == "loopmath"
    assert doc["producer"]["source_contract"] == "dagr_graph/1"
    assert errors(doc) == []
    # The store's strict check takes it (v0.3, E005), and no key is in the
    # pre-0.3 `dev.dagr.` namespace (W182).
    strict = validate_strict(doc)
    assert [f for f in strict if f["level"] == "error"] == []
    assert [f for f in strict if f["code"] == "W182"] == []
    assert any(f["code"] == "W182" for f in validate_strict(_as_pre_03(doc)))


def test_one_node_one_attempt_per_graph_node(g, doc):
    assert [n["id"] for n in doc["nodes"]] == sorted(n.id for n in g.nodes)
    assert [a["node"] for a in doc["attempts"]] == [n["id"] for n in doc["nodes"]]
    assert all(a["id"] == attempt_id(a["node"]) and a["n"] == 1 for a in doc["attempts"])
    assert {n["group"] for n in doc["nodes"]} == {ALPHA, "/ws/other"}
    assert doc["run"]["workspace"] == ALPHA  # the external launcher's workspace is a group, not the run


def test_every_attempt_carries_origin_role_and_phase_with_tiers(doc):
    for a in doc["attempts"]:
        assert REQUIRED_ORIGIN <= set(a["origin"]) and a["origin"]["tier"] in TIERS
        assert set(a["role"]) >= REQUIRED_LABEL and a["role"]["tier"] in TIERS
        assert set(a["phase"]) >= REQUIRED_LABEL and a["phase"]["tier"] in TIERS
        assert a["status"] == "settled_unverified" == a["outcome"]["result"]
        assert a["outcome"]["evidence"] == "heuristic"


def test_labels_follow_the_graph_and_unknowns_are_explicit_nulls(g, doc):
    plan = _attempt(doc, "plan")
    # The graph's evidence quotes the declared type ("declared type 'Plan Plan'"); under
    # metadata_only the document names the rule and the field, never the text (item 6).
    assert g.node("plan").role_evidence.startswith("declared type ")
    assert plan["role"] == {"value": "planner", "tier": "reported", "evidence": "rule declared_type_pattern: the Task call's subagent type matched the planner pattern (9 chars of text withheld under metadata_only)"}
    assert "Plan Plan" not in json.dumps(doc)
    assert doc["ext"]["dev.loopmath.graph"]["emitter"]["role_evidence_reduced_to_rule"] == 1
    assert plan["phase"]["value"] == "build" and plan["phase"]["tier"] == "heuristic"
    assert plan["origin"]["external"] is False and plan["origin"]["launched_by"] is None
    assert plan["origin"]["tier"] == "verified"  # the spawn edge is verified
    reader = _attempt(doc, "reader")
    assert reader["role"]["value"] is None and reader["role"]["tier"] == "heuristic"
    assert reader["role"]["evidence"] == g.node("reader").role_evidence
    assert reader["phase"]["value"] == "post"
    analyst = _attempt(doc, "analyst")
    assert analyst["role"]["value"] == "external" and analyst["phase"]["value"] == "external"
    assert "started_at" not in analyst
    assert doc["ext"]["dev.loopmath.graph"]["emitter"]["attempts_without_start_ts"] == 1
    assert doc["nodes"][[n["id"] for n in doc["nodes"]].index("dev")]["kind"] == "impl"
    assert doc["nodes"][[n["id"] for n in doc["nodes"]].index("reader")]["kind"] == "unknown"


def test_launch_becomes_origin_and_a_launch_edge(doc):
    rev = _attempt(doc, "codex-review")
    assert rev["origin"]["launched_by"] == "lead.a1" and rev["origin"]["external"] is False
    assert rev["origin"]["how"] and rev["origin"]["tier"] == "heuristic"
    launch = [e for e in doc["edges"] if e["kind"] == "launch"]
    assert {(e["from"], e["to"], e["tier"]) for e in launch} == {("lead", "codex-review", "heuristic"), ("analyst", "codex-audit", "heuristic")}
    assert all(e["from_attempt"] == attempt_id(e["from"]) and e["to_attempt"] == attempt_id(e["to"]) for e in doc["edges"])


def test_edges_keep_their_tiers(g, doc):
    graph_edges = sorted((e.kind, e.src, e.dst, e.tier) for e in g.edges)
    ocp_edges = sorted((e["kind"], e["from"], e["to"], e["tier"]) for e in doc["edges"])
    assert ocp_edges == graph_edges
    assert all(e["tier"] in TIERS and e["evidence"] for e in doc["edges"])


def test_artifacts_and_artifact_edges(g, doc):
    arts = {a["id"]: a for a in doc["artifacts"]}
    assert set(arts) == {a.id for a in g.artifacts}
    plan_md = arts["/ws/alpha/plan.md"]
    assert plan_md["producer"] == "plan.a1" == plan_md["writers"][0]
    assert plan_md["consumers"] == ["dev.a1", "reader.a1"]
    assert plan_md["n_writes"] == 1 and plan_md["n_reads"] == 4
    assert plan_md["first_write_at"] == "2026-08-31T10:01:00.000Z"
    assert plan_md["kind"] == {"value": "plan", "tier": "heuristic", "evidence": "path pattern"}
    source_artifact = next(artifact for artifact in g.artifacts if artifact.id == plan_md["id"])
    recording_fields = (
        "bytes", "bytes_tier", "lines_added", "lines_added_tier",
        "lines_removed", "lines_removed_tier", "language", "language_tier",
        "tests_touched", "tests_touched_tier", "fate", "fate_tier", "meta",
    )
    assert plan_md["ext"]["dev.loopmath.artifact"] == {
        field: getattr(source_artifact, field) for field in recording_fields
    }
    art_edges = [e for e in doc["edges"] if e["kind"] == "artifact"]
    # A1 (merged 2f40fb4): lead's 10:05:00 `codex exec ... -o /ws/alpha/review.md` is a heuristic write attributed
    # to codex-review; lead reads review.md at 10:08:10, so there is a third artifact edge codex-review -> lead.
    assert {(e["artifact"], e["from_attempt"], e["to_attempt"]) for e in art_edges} == {("/ws/alpha/plan.md", "plan.a1", "dev.a1"), ("/ws/alpha/plan.md", "plan.a1", "reader.a1"), ("/ws/alpha/review.md", "codex-review.a1", "lead.a1")}


def test_cost_is_measured_and_maps_the_four_streams(doc):
    lead = _attempt(doc, "lead")
    assert lead["cost"] == {"input_tokens": 1000, "output_tokens": 5000, "usd": 1.25, "basis": "measured"}
    assert lead["model"] == {"raw": "claude-opus-5", "tier": "verified"}
    assert lead["effort"] == "high"
    assert lead["started_at"] == "2026-08-31T10:00:00.000Z" and lead["ended_at"] == "2026-08-31T10:10:00.000Z"
    audit = _attempt(doc, "codex-audit")
    assert "usd" not in audit["cost"]  # unpriced stays absent, never 0


def test_meta_and_events_land_in_the_document(g, doc):
    assert doc["ext"]["dev.loopmath.graph"]["meta"] == g.meta
    assert doc["run"]["started_at"] == "2026-08-31T10:00:00.000Z"
    ats = [e["at"] for e in doc["events"]]
    assert ats == sorted(ats) and len(ats) == 2 * 9  # nine timed attempts, start and end each


def test_privacy_profiles_gate_transcript_text(g):
    meta_only = json.dumps(to_ocp(g, privacy="metadata_only"))
    full = to_ocp(g, privacy="full")
    assert "Plan the extractor" not in meta_only
    assert full["privacy"]["profile"] == "full"
    assert _attempt(full, "plan")["ext"]["dev.loopmath.graph"]["spawn"]["description"] == "Plan the extractor"
    assert _attempt(full, "plan")["role"]["evidence"] == g.node("plan").role_evidence  # raw evidence only under full
    assert full["ext"]["dev.loopmath.graph"]["emitter"]["role_evidence_reduced_to_rule"] == 0
    assert "codex" in _attempt(full, "codex-review")["ext"]["dev.loopmath.graph"]["launch_command"]
    assert errors(full) == []
    with pytest.raises(ValueError):
        to_ocp(g, privacy="secret")


def test_producer_override_and_determinism(g):
    a = to_ocp(g, producer={"name": "loopmath/graph", "emitted_at": "2026-09-01T00:00:00Z"})
    assert a["producer"]["name"] == "loopmath/graph" and a["producer"]["emitted_at"] == "2026-09-01T00:00:00Z"
    assert errors(a) == []
    assert json.dumps(to_ocp(g)) == json.dumps(to_ocp(g))


def test_producer_capability_extensions_survive_while_core_values_are_forced(g):
    core = {
        "groups": True,
        "events": True,
        "artifacts": True,
        "edges_dep": True,
        "edges_spawn": True,
        "edges_launch": True,
        "edges_artifact": True,
        "cost_usd": True,
        "cost_tokens": True,
        "outcome_evidence": True,
    }
    supplied = {name: False for name in core}
    supplied["my_extension"] = True

    document = to_ocp(
        g,
        producer={"name": "probe", "capabilities": supplied},
    )

    assert document["producer"]["capabilities"] == {**core, "my_extension": True}
    assert errors(document) == []
    with pytest.raises(
        ValueError,
        match="producer capability 'my_extension' must be boolean",
    ):
        to_ocp(
            g,
            producer={"capabilities": {"my_extension": "yes"}},
        )


def _hand_graph() -> Graph:
    long_path = "/ws/" + "x" * 250 + "/notes.txt"
    nodes = [
        GraphNode(id="a", harness="claude-code", source="top", session_path="/s/a.jsonl", workspace="w", ts="2026-08-31T10:00:00Z", wall_s=100.0, tokens={"in": 1, "out": 2}, role="lead", role_tier="heuristic", role_evidence="e", phase="build", phase_tier="heuristic"),
        GraphNode(id="b", harness="claude-code", source="subagent", session_path="/s/b.jsonl", workspace="w", ts="not a time", wall_s=10.0, tokens=None, parent="a", role="dev", role_tier="heuristic", role_evidence="e", phase="build", phase_tier="heuristic"),
        GraphNode(id="c", harness="claude-code", source="subagent", session_path="/s/c.jsonl", workspace="w", ts="2026-08-31T10:00:30Z", wall_s=None, parent="a", role="reviewer", role_tier="reported", role_evidence="e", phase=None, phase_tier=None),
    ]
    edges = [
        GraphEdge("a", "b", "spawn", "verified", {}),
        GraphEdge("a", "c", "spawn", "heuristic", {"reason": "path containment"}),
        GraphEdge("a", "c", "artifact", "verified", {"path": "/ws/plan.md", "lag_s": 5.0, "write_tier": "verified", "read_tier": "verified"}),
        # b is a later writer of plan.md, so c's read joins to b in the graph; OCP runs artifact edges from the producer.
        GraphEdge("b", "c", "artifact", "heuristic", {"path": "/ws/plan.md", "lag_s": 1.0, "write_tier": "heuristic", "read_tier": "verified"}),
        GraphEdge("a", "b", "artifact", "verified", {"path": long_path, "lag_s": 2.0, "write_tier": "verified", "read_tier": "verified"}),
    ]
    artifacts = [
        Artifact(id="/ws/plan.md", producer="a", writers=["a", "b"], consumers=["c"], first_write_ts="2026-08-31T10:00:05Z", n_writes=2, n_reads=2, kind="plan", kind_tier="heuristic"),
        Artifact(id=long_path, producer="a", writers=["a"], consumers=["b"], first_write_ts="bogus", n_writes=1, n_reads=1, kind=None, kind_tier=None),
    ]
    return Graph(nodes=nodes, edges=edges, artifacts=artifacts, meta={"unlinked_subagents": 0})


def test_later_writer_edges_are_rerouted_from_the_producer_and_counted():
    """Item 4: the graph's b->c read edge (b is a later writer of plan.md) becomes the
    conforming producer-to-consumer edge a->c with the same tier, b named in the
    evidence and in ext, and the rewrite counted. Nothing is dropped."""
    doc = to_ocp(_hand_graph())
    assert errors(doc) == []
    art_edges = [e for e in doc["edges"] if e["kind"] == "artifact" and e["artifact"] == "/ws/plan.md"]
    assert [(e["from"], e["to"], e["tier"]) for e in art_edges] == [("a", "c", "verified"), ("a", "c", "heuristic")]
    rerouted = art_edges[1]
    assert rerouted["from_attempt"] == "a.a1" and rerouted["to_attempt"] == "c.a1"
    assert rerouted["evidence"] == "read joined to later writer b.a1 in the graph (write (heuristic) then read (verified) 1.0 s later); OCP runs the artifact edge from the producer a.a1"
    assert rerouted["ext"]["dev.loopmath.graph"] == {"graph_from": "b", "graph_from_attempt": "b.a1", "rerouted": "later writer to producer"}
    assert "ext" not in art_edges[0]
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["artifact_edges_rerouted_from_later_writer"] == 1
    assert doc["ext"]["dev.loopmath.graph"]["edges_not_emitted"] == []
    assert len(doc["edges"]) == len(_hand_graph().edges)
    plan = next(a for a in doc["artifacts"] if a["path"] == "/ws/plan.md")
    assert plan["writers"] == ["a.a1", "b.a1"] and plan["n_writes"] == 2


def test_edges_with_invalid_tiers_or_bad_endpoints_are_kept_in_ext_never_retiered():
    """Item 5: a tier the graph did not give is never invented. The edge is left out of
    `edges`, kept whole with its raw tier and a reason under `edges_not_emitted`, and
    counted. Valid tiers pass through untouched."""
    g = _hand_graph()
    g.edges.append(GraphEdge("a", "b", "spawn", "guessed", {"reason": "bad tier"}))
    g.edges.append(GraphEdge("a", "c", "launch", "", {"how": "no tier", "lag_s": 1.0}))
    g.edges.append(GraphEdge("a", "zzz", "spawn", "verified", {}))
    g.edges.append(GraphEdge("a", "c", "artifact", "verified", {"path": "/ws/never-written.md", "lag_s": 1.0, "write_tier": "verified", "read_tier": "verified"}))
    g.edges.append(GraphEdge("c", "b", "artifact", "verified", {"path": "/ws/plan.md", "lag_s": 1.0, "write_tier": "verified", "read_tier": "verified"}))  # c never wrote plan.md
    g.edges.append(GraphEdge("a", "a", "artifact", "verified", {"path": "/ws/plan.md", "lag_s": 1.0, "write_tier": "verified", "read_tier": "verified"}))  # a is not a consumer
    doc = to_ocp(g)
    assert errors(doc) == []
    assert sorted(e["tier"] for e in doc["edges"]) == ["heuristic", "heuristic", "verified", "verified", "verified"]
    held = doc["ext"]["dev.loopmath.graph"]["edges_not_emitted"]
    assert [(h["from"], h["to"], h["kind"], h["tier"]) for h in held] == [
        ("a", "a", "artifact", "verified"), ("a", "c", "artifact", "verified"), ("c", "b", "artifact", "verified"), ("a", "c", "launch", ""), ("a", "b", "spawn", "guessed"), ("a", "zzz", "spawn", "verified"),
    ]
    assert held[4]["reason"] == "tier 'guessed' is not one of verified/heuristic/reported; the emitter does not pick one"
    assert held[5]["reason"] == "an endpoint is not a node of the graph"
    assert held[1]["reason"] == "the edge's path is not an artifact of the graph" and held[1]["path"] == "/ws/never-written.md"
    assert held[2]["reason"] == "the edge's writer is not among the artifact's writers"
    assert held[0]["reason"] == "the edge's reader is not among the artifact's consumers"
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["edges_not_emitted_invalid_tier"] == 2 and em["edges_not_emitted_unknown_endpoint"] == 1
    assert em["edges_not_emitted_unknown_artifact"] == 1 and em["edges_not_emitted_writer_not_listed"] == 1 and em["edges_not_emitted_reader_not_consumer"] == 1
    assert len(doc["edges"]) + len(held) == len(g.edges)


def test_labels_with_invalid_tiers_are_withheld_and_counted():
    """Item 5 for role, phase, artifact kind and model: a value whose tier is not one
    of the three is withheld as an explicit null (evidence says why), the model label is
    kept without a tier, and each case is counted."""
    g = _hand_graph()
    a = g.node("a")
    a.role_tier, a.phase_tier, a.model, a.model_tier = "asserted", None, "claude-opus-5", "guess"
    g.artifacts[0].kind_tier = "maybe"
    doc = to_ocp(g)
    assert errors(doc) == []
    at = _attempt(doc, "a")
    assert at["role"] == {"value": None, "tier": "heuristic", "evidence": "graph gave role 'lead' with tier 'asserted', not one of verified/heuristic/reported; value withheld"}
    assert at["phase"] == {"value": None, "tier": "heuristic", "evidence": "graph gave phase 'build' with tier None, not one of verified/heuristic/reported; value withheld"}
    assert at["model"] == {"raw": "claude-opus-5"}
    assert at["ext"]["dev.loopmath.graph"]["model_tier_omitted"].startswith("graph gave model tier 'guess'")
    plan = next(x for x in doc["artifacts"] if x["path"] == "/ws/plan.md")
    assert plan["kind"] == {"value": None, "tier": "heuristic", "evidence": "graph gave artifact kind 'plan' with tier 'maybe', not one of verified/heuristic/reported; value withheld"}
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["labels_withheld_invalid_tier"] == 3 and em["model_tiers_omitted_invalid"] == 1
    # A null value with no tier is the graph saying no rule matched: heuristic unknown, not counted as invalid.
    c = _attempt(doc, "c")
    assert c["phase"]["value"] is None and c["phase"]["tier"] == "heuristic"


def test_role_evidence_from_transcript_text_is_reduced_to_the_rule_under_metadata_only():
    """Item 6: a role inferred from a spawn description names the rule and the facts it
    used under metadata_only; the quoted text appears only under full."""
    g = _hand_graph()
    b = g.node("b")
    b.role, b.role_tier, b.role_evidence = "planner", "heuristic", "spawn description 'Plan the SECRET migration'"
    b.spawn = {"description": "Plan the SECRET migration"}
    meta_only = to_ocp(g)
    assert "SECRET" not in json.dumps(meta_only)
    assert _attempt(meta_only, "b")["role"] == {"value": "planner", "tier": "heuristic", "evidence": "rule spawn_description_pattern: the Task call's description matched the planner pattern (25 chars of text withheld under metadata_only)"}
    assert meta_only["ext"]["dev.loopmath.graph"]["emitter"]["role_evidence_reduced_to_rule"] == 1
    full = to_ocp(g, privacy="full")
    assert _attempt(full, "b")["role"]["evidence"] == "spawn description 'Plan the SECRET migration'"
    # Evidence that never quoted text passes through unchanged under both profiles.
    assert _attempt(meta_only, "a")["role"]["evidence"] == "e" == _attempt(full, "a")["role"]["evidence"]


def test_missing_cost_streams_and_prices_are_named_per_attempt():
    """Item 7: every stream, token record or dollar figure the graph lacks is named on
    the attempt with its reason, never silently absent, and counted by the emitter."""
    g = _hand_graph()
    g.node("c").tokens = {"in": 5, "cache_read": None, "out": -1}
    g.node("c").usd = 0.5
    doc = to_ocp(g)
    assert errors(doc) == []
    a = _attempt(doc, "a")
    assert a["cost"] == {"input_tokens": 1, "output_tokens": 2, "basis": "measured"}
    no_split = "no retention split in the session record: the parsers keep one cache-write stream (absent too); unknown, never zero"
    assert a["ext"]["dev.loopmath.graph"]["cost_missing"] == {
        "cached_input_tokens": "stream 'cache_read' absent from the session record",
        "cache_creation_tokens": "stream 'cache_write' absent from the session record",
        "cache_creation_5m_tokens": no_split,
        "cache_creation_1h_tokens": no_split,
        "usd": "not priced: the graph carries no dollar figure for this session (no price entry for the model, or no token stream to price); never counted as zero",
    }
    # Item 4: every missing figure is an explicit null beside its reason, never a zero.
    assert a["ext"]["dev.loopmath.graph"]["cost_unknown"] == {k: None for k in a["ext"]["dev.loopmath.graph"]["cost_missing"]}
    assert "cache_creation_5m_tokens" not in a["cost"] and 0 not in a["cost"].values()
    b = _attempt(doc, "b")
    assert "cost" not in b
    assert b["ext"]["dev.loopmath.graph"]["cost_missing"]["tokens"] == "no token usage in the session record; no cost record emitted"
    assert "usd" in b["ext"]["dev.loopmath.graph"]["cost_missing"]
    assert b["ext"]["dev.loopmath.graph"]["cost_unknown"] == {"tokens": None, "usd": None}
    c = _attempt(doc, "c")
    assert c["cost"] == {"input_tokens": 5, "usd": 0.5, "basis": "measured"}
    assert c["ext"]["dev.loopmath.graph"]["cost_missing"] == {
        "cached_input_tokens": "stream 'cache_read' is 'None' in the session record, not a count",
        "cache_creation_tokens": "stream 'cache_write' absent from the session record",
        "cache_creation_5m_tokens": no_split,
        "cache_creation_1h_tokens": no_split,
        "output_tokens": "stream 'out' is '-1' in the session record, not a count",
    }
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["attempts_without_tokens"] == 1 and em["attempts_unpriced"] == 2 and em["token_streams_missing"] == 5
    assert em["cache_retention_buckets_missing"] == 4  # two buckets for each of the two attempts with a token record


def test_cache_retention_buckets_are_emitted_only_when_the_record_carries_them():
    """Item 4: `cache_creation_5m_tokens` and `cache_creation_1h_tokens` come from the
    record (either the graph's or OCP's key), are withheld when they contradict the
    cache-write total, and are explicit unknowns otherwise; none of it is ever 0."""
    g = _hand_graph()
    a = g.node("a")
    a.tokens = {"in": 1, "cache_read": 2, "cache_write": 30, "cache_write_5m": 10, "cache_creation_1h_tokens": 20, "out": 3}
    doc = to_ocp(g)
    assert errors(doc) == []
    at = _attempt(doc, "a")
    assert at["cost"] == {"input_tokens": 1, "cached_input_tokens": 2, "cache_creation_tokens": 30, "cache_creation_5m_tokens": 10, "cache_creation_1h_tokens": 20, "output_tokens": 3, "basis": "measured"}
    assert set(at["ext"]["dev.loopmath.graph"]["cost_missing"]) == {"usd"}
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["cache_retention_buckets_missing"] == 0 and em["cache_retention_buckets_inconsistent"] == 0  # a is the only node with a token record
    # The buckets must sum to the total; otherwise both are withheld with the numbers in the reason.
    a.tokens["cache_write_5m"] = 11
    doc = to_ocp(g)
    assert errors(doc) == []
    at = _attempt(doc, "a")
    assert "cache_creation_5m_tokens" not in at["cost"] and "cache_creation_1h_tokens" not in at["cost"] and at["cost"]["cache_creation_tokens"] == 30
    assert at["ext"]["dev.loopmath.graph"]["cost_missing"]["cache_creation_5m_tokens"] == "session record gives 11, but the two buckets sum to 31 against 30 cache-write tokens; both withheld"
    assert at["ext"]["dev.loopmath.graph"]["cost_unknown"]["cache_creation_1h_tokens"] is None
    assert doc["ext"]["dev.loopmath.graph"]["emitter"]["cache_retention_buckets_inconsistent"] == 1
    # A bucket that is present but not a count is named as such.
    a.tokens["cache_write_5m"] = "ten"
    doc = to_ocp(g)
    assert _attempt(doc, "a")["ext"]["dev.loopmath.graph"]["cost_missing"]["cache_creation_5m_tokens"] == "bucket 'cache_write_5m' is 'ten' in the session record, not a count"
    assert _attempt(doc, "a")["cost"]["cache_creation_1h_tokens"] == 20


def test_bad_timestamps_long_paths_and_unknown_kind():
    doc = to_ocp(_hand_graph())
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    b = _attempt(doc, "b")
    assert "started_at" not in b and "ended_at" not in b and "cost" not in b
    c = _attempt(doc, "c")
    assert c["started_at"] == "2026-08-31T10:00:30.000Z" and "ended_at" not in c
    assert c["phase"] == {"value": None, "tier": "heuristic", "evidence": "no lead interval or start time to place the session against"}
    assert em["attempts_without_start_ts"] == 1 and em["attempts_missing_wall_s"] == 1
    long_path = "/ws/" + "x" * 250 + "/notes.txt"
    long_art = next(a for a in doc["artifacts"] if a["path"] != "/ws/plan.md")
    assert long_art["id"].startswith("sha256:") and len(long_art["id"]) <= 200
    # Item 5: a derived id never loses the path; the full path sits in ext and the derivation is counted.
    assert long_art["path"] == long_path
    assert long_art["ext"]["dev.loopmath.graph"] == {"id_derived": "sha256 of the full path: 264 characters exceed the 200 character id limit", "path": long_path}
    assert long_art["ext"]["dev.loopmath.artifact"]["fate"] == "unknown"
    assert long_art["first_write_at"] is None and em["artifacts_first_write_ts_unparseable"] == 1
    assert long_art["kind"] == {"value": None, "tier": "heuristic", "evidence": "path pattern; no rule matched"}
    assert em["artifact_ids_hashed"] == 1 and em["artifact_paths_cut"] == 0
    plan_art = next(a for a in doc["artifacts"] if a["path"] == "/ws/plan.md")
    assert set(plan_art["ext"]) == {"dev.loopmath.artifact"}
    edge = next(e for e in doc["edges"] if e["kind"] == "artifact" and e["to"] == "b")
    assert edge["artifact"] == long_art["id"]


def test_paths_over_the_schema_limit_are_cut_in_place_and_kept_whole_in_ext():
    """Item 5: the schema caps `path` at 1000 characters. The shown path is the head,
    the full path is untouched in ext beside a note, and the cut is counted."""
    g = _hand_graph()
    huge = "/ws/" + "y" * 1200 + "/deep.md"
    g.artifacts.append(Artifact(id=huge, producer="a", writers=["a"], consumers=["c"], first_write_ts="2026-08-31T10:00:06Z", n_writes=1, n_reads=1, kind="doc", kind_tier="heuristic"))
    g.edges.append(GraphEdge("a", "c", "artifact", "verified", {"path": huge, "lag_s": 1.0, "write_tier": "verified", "read_tier": "verified"}))
    doc = to_ocp(g)
    assert errors(doc) == []
    art = next(a for a in doc["artifacts"] if a["path"].startswith("/ws/yyy"))
    assert len(art["path"]) == 1000 and art["path"] == huge[:1000]
    assert art["id"].startswith("sha256:")
    assert art["ext"]["dev.loopmath.graph"]["path"] == huge
    assert art["ext"]["dev.loopmath.graph"]["path_cut"] == "first 1000 of 1212 characters shown; the full path is beside this note"
    assert art["ext"]["dev.loopmath.graph"]["id_derived"].startswith("sha256 of the full path: 1212 characters")
    assert set(art["ext"]) == {"dev.loopmath.graph", "dev.loopmath.artifact"}
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["artifact_paths_cut"] == 1 and em["artifact_ids_hashed"] == 2
    assert any(e["kind"] == "artifact" and e["artifact"] == art["id"] and e["to"] == "c" for e in doc["edges"])


def test_artifact_recording_emits_every_value_and_preserves_meta_without_aliasing():
    g = _hand_graph()
    artifact = g.artifacts[0]
    artifact.bytes = 0
    artifact.bytes_tier = "verified"
    artifact.lines_added = 4
    artifact.lines_added_tier = "reported"
    artifact.lines_removed = 0
    artifact.lines_removed_tier = "verified"
    artifact.language = "py"
    artifact.language_tier = "heuristic"
    artifact.tests_touched = 0
    artifact.tests_touched_tier = "heuristic"
    artifact.fate = "edited"
    artifact.fate_tier = "verified"
    artifact.meta = {
        "unknown.key": {"nested": [0, False, None], "unicode": "αλφα"},
    }

    doc = to_ocp(g)
    emitted = next(item for item in doc["artifacts"] if item["path"] == artifact.id)
    assert emitted["ext"]["dev.loopmath.artifact"] == {
        "bytes": 0,
        "bytes_tier": "verified",
        "lines_added": 4,
        "lines_added_tier": "reported",
        "lines_removed": 0,
        "lines_removed_tier": "verified",
        "language": "py",
        "language_tier": "heuristic",
        "tests_touched": 0,
        "tests_touched_tier": "heuristic",
        "fate": "edited",
        "fate_tier": "verified",
        "meta": {"unknown.key": {"nested": [0, False, None], "unicode": "αλφα"}},
    }
    emitted["ext"]["dev.loopmath.artifact"]["meta"]["unknown.key"]["nested"].append(1)
    assert artifact.meta["unknown.key"]["nested"] == [0, False, None]


def test_origin_tier_is_the_launch_edge_tier_copied_never_upgraded():
    """Item 3: a launched node's origin carries exactly the tier of its emitted launch
    edge; a launcher the graph names without an emitted launch edge is withheld (null)
    with the claim in the evidence, and counted."""
    g = _hand_graph()
    for nid, tier in (("x", "verified"), ("y", "reported"), ("z", "heuristic")):
        g.nodes.append(GraphNode(id=nid, harness="codex", source="codex", session_path=f"/s/{nid}.jsonl", workspace="w", ts="2026-08-31T10:01:00Z", wall_s=5.0, tokens={"in": 1, "out": 1}, parent="a", launched_by={"id": "a", "workspace": "w", "how": f"{tier} join", "lag_s": 2.0, "command": "codex exec x"}, role="cli", role_tier="heuristic", role_evidence="e", phase="build", phase_tier="heuristic"))
        g.edges.append(GraphEdge("a", nid, "launch", tier, {"how": f"{tier} join", "lag_s": 2.0, "command": "codex exec x"}))
    # `q` names a launcher but its launch edge has an invalid tier and is not emitted; `r` has no edge at all.
    for nid, edge in (("q", GraphEdge("a", "q", "launch", "guessed", {"how": "bad", "lag_s": 1.0})), ("r", None)):
        g.nodes.append(GraphNode(id=nid, harness="codex", source="codex", session_path=f"/s/{nid}.jsonl", workspace="w", ts="2026-08-31T10:01:00Z", wall_s=5.0, tokens={"in": 1, "out": 1}, parent="a", launched_by={"id": "a", "workspace": "w", "how": "x", "lag_s": 1.0}, role="cli", role_tier="heuristic", role_evidence="e", phase="build", phase_tier="heuristic"))
        if edge:
            g.edges.append(edge)
    doc = to_ocp(g)
    assert errors(doc) == []
    for nid, tier in (("x", "verified"), ("y", "reported"), ("z", "heuristic")):
        o = _attempt(doc, nid)["origin"]
        assert o == {"launched_by": "a.a1", "workspace": "w", "external": False, "how": f"{tier} join", "tier": tier, "evidence": f"launch edge ({tier}): {tier} join; lag 2.0 s"}
    for nid in ("q", "r"):
        o = _attempt(doc, nid)["origin"]
        assert o == {"launched_by": None, "workspace": "w", "external": None, "how": None, "tier": "heuristic", "evidence": "the graph names launcher a.a1 but no launch edge from it is emitted; launcher withheld"}
    em = doc["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["origins_without_launch_edge"] == 2 and em["edges_not_emitted_invalid_tier"] == 1


def test_external_is_true_only_for_a_reported_unscanned_launcher(doc):
    """Other arm's item 1: a scanned launcher session outside the requested workspaces
    is not an unscanned launcher: `external` is null there and false on the nodes it
    launched. True is reserved for the L2 case, a harness-reported launcher that no
    scanned session is, at the tier the graph gives it."""
    analyst = _attempt(doc, "analyst")["origin"]
    assert analyst == {"launched_by": None, "workspace": "/ws/other", "external": None, "how": None, "tier": "heuristic", "evidence": "scanned top-level session outside the requested workspaces; it launched sessions inside them and nothing in scope launched it, so it is not an unscanned launcher"}
    audit = _attempt(doc, "codex-audit")["origin"]
    assert audit["launched_by"] == "analyst.a1" and audit["external"] is False and audit["tier"] == "heuristic"
    assert all(a["origin"]["external"] is not True for a in doc["attempts"])
    g = _hand_graph()
    g.nodes.append(GraphNode(id="l2", harness="codex", source="codex", session_path="/s/l2.jsonl", workspace="w", ts="2026-08-31T10:01:00Z", wall_s=5.0, tokens={"in": 1, "out": 1}, launched_by={"id": None, "external": True, "tier": "reported", "how": "codex originator codex_exec", "evidence": "session_meta originator codex_exec, cwd is the workspace; no launcher in scope"}, role=None, role_tier=None, role_evidence="not launched by any session in scope; no prompt found", phase="build", phase_tier="heuristic"))
    out = to_ocp(g)
    assert errors(out) == []
    assert _attempt(out, "l2")["origin"] == {"launched_by": None, "workspace": "w", "external": True, "how": "codex originator codex_exec", "tier": "reported", "evidence": "session_meta originator codex_exec, cwd is the workspace; no launcher in scope"}
    assert out["ext"]["dev.loopmath.graph"]["emitter"]["origins_reported_unscanned_launcher"] == 1
    g.node("l2").launched_by["tier"] = "asserted"
    out = to_ocp(g)
    assert errors(out) == []
    assert _attempt(out, "l2")["origin"]["external"] is None and _attempt(out, "l2")["origin"]["tier"] == "heuristic"
    assert _attempt(out, "l2")["origin"]["evidence"] == "graph reports an unscanned launcher with tier 'asserted', not one of verified/heuristic/reported; value withheld"
    assert out["ext"]["dev.loopmath.graph"]["emitter"]["labels_withheld_invalid_tier"] == 1


def test_titles_name_the_spawned_task_under_full_and_stay_synthesized_under_metadata_only(g):
    """Item 6: the spawn description names the task, so under `full` it is the node's
    title; under `metadata_only` it is transcript text, the title stays synthesized
    from role and harness, and the withheld title is counted and noted on the node."""
    full = to_ocp(g, privacy="full")
    meta_only = to_ocp(g)
    assert errors(full) == [] and errors(meta_only) == []
    node_full = {n["id"]: n for n in full["nodes"]}
    node_meta = {n["id"]: n for n in meta_only["nodes"]}
    assert node_full["plan"]["title"] == "Plan the extractor"
    assert node_full["plan"]["ext"]["dev.loopmath.graph"] == {"title_source": "spawn description"}
    assert node_meta["plan"]["title"] == "planner subagent (claude-code)"
    assert node_meta["plan"]["ext"]["dev.loopmath.graph"] == {"title_withheld": "the spawn description names the task (18 chars); withheld under metadata_only"}
    assert "Plan the extractor" not in json.dumps(meta_only)
    # A node without a description keeps the synthesized title under both profiles, with no note.
    assert node_full["lead"]["title"] == "lead session (claude-code)" == node_meta["lead"]["title"]
    assert "ext" not in node_full["lead"] and "ext" not in node_meta["lead"]
    assert node_full["analyst"]["title"] == "external launcher session (claude-code)"
    n_described = sum(1 for n in g.nodes if n.spawn and (n.spawn.get("description") or n.spawn.get("meta_description")))
    assert n_described >= 2
    assert full["ext"]["dev.loopmath.graph"]["emitter"]["titles_from_spawn_description"] == n_described
    assert meta_only["ext"]["dev.loopmath.graph"]["emitter"]["titles_withheld_metadata_only"] == n_described
    assert full["ext"]["dev.loopmath.graph"]["emitter"]["titles_withheld_metadata_only"] == 0


@pytest.mark.parametrize("term", ["knowledge gradient", "posterior", "prior", "experimental design", "value of information", "bandit", "arms", "Knowledge  Gradient", "BANDIT"])
def test_em_dashes_never_reach_a_user_facing_string_and_the_lifted_vocabulary_passes(term):
    """Spec section 0, rule 4: every title, evidence and reason passes one choke
    point that replaces em-dashes and counts what it changed. The vocabulary
    filter was lifted in 0.1.0 (Q3): its old terms pass through unchanged. Ids
    and paths are data and stay as they are."""
    g = _hand_graph()
    a = g.node("a")
    a.role_evidence = f"lead picked by the {term} rule \u2014 top of the {term}"
    b = g.node("b")
    b.spawn = {"description": f"Compare the {term} across runs"}
    b.role_evidence = f"spawn description 'Compare the {term} across runs'"
    b.launched_by = None
    g.artifacts.append(Artifact(id=f"/ws/{term.replace(' ', '-')}.md", producer="a", writers=["a"], consumers=[], first_write_ts=None, n_writes=1, n_reads=0, kind="doc", kind_tier="heuristic"))
    full = to_ocp(g, privacy="full")
    assert errors(full) == []
    strings = [n["title"] for n in full["nodes"]] + [at["role"]["evidence"] for at in full["attempts"]]
    joined = "\n".join(s for s in strings if s)
    assert "\u2014" not in joined
    assert _attempt(full, "a")["role"]["evidence"] == f"lead picked by the {term} rule - top of the {term}"
    assert next(n for n in full["nodes"] if n["id"] == "b")["title"] == f"Compare the {term} across runs"
    em = full["ext"]["dev.loopmath.graph"]["emitter"]
    assert em["strings_with_em_dash_replaced"] == 1 and "strings_with_forbidden_term_replaced" not in em
    # The path is an identifier, not prose: it is untouched.
    assert any(x["path"] == f"/ws/{term.replace(' ', '-')}.md" for x in full["artifacts"])
    assert sanitize("posteriors and priors, bandits") == "posteriors and priors, bandits"
    assert sanitize("no arms race\u2014none") == "no arms race - none"


def test_v01_example_still_passes_and_a_tierless_edge_fails(doc):
    broken = copy.deepcopy(doc)
    del broken["edges"][0]["tier"]
    assert any(f.code == "E160" for f in errors(broken))
    v01 = json.loads((ROOT / "spec" / "examples" / "minimal.ocp.json").read_text())
    assert errors(v01) == []
