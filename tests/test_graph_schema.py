"""Graph schema: constants and the `to_dict` round trip."""

from __future__ import annotations

import json

from loopmath.graph.artifacts import build_artifacts
from loopmath.graph.schema import EDGE_KINDS, PHASES, SOURCES, TIERS, Artifact, Graph, GraphEdge, GraphNode


def _sample() -> Graph:
    lead = GraphNode(id="lead", harness="claude-code", source="top", session_path="/p/lead.jsonl", model="claude-opus-5", workspace="/ws/alpha", ts="2026-08-31T10:00:00Z", wall_s=600.0, tokens={"in": 10, "out": 5}, usd=1.25, role="lead", role_tier="heuristic", role_evidence="most out-edges", phase="build", phase_tier="heuristic")
    sub = GraphNode(id="plan", harness="claude-code", source="subagent", session_path="/p/lead/subagents/a.jsonl", parent="lead", spawn={"tool_use_id": "toolu_1"}, role="planner", role_tier="reported", phase="build", phase_tier="heuristic")
    cx = GraphNode(id="cx", harness="codex", source="codex", session_path="/c/r.jsonl", model="gpt-5", model_tier="reported", parent="lead", launched_by={"id": "lead", "lag_s": 3.0}, role="reviewer", role_tier="heuristic", phase="build", phase_tier="heuristic")
    edges = [
        GraphEdge("lead", "plan", "spawn", "verified", {"tool_use_id": "toolu_1"}),
        GraphEdge("lead", "cx", "launch", "heuristic", {"lag_s": 3.0}),
        GraphEdge("plan", "cx", "artifact", "verified", {"path": "/ws/alpha/plan.md", "lag_s": 240.0}),
    ]
    art = Artifact(id="/ws/alpha/plan.md", producer="plan", writers=["plan"], consumers=["cx"], first_write_ts="2026-08-31T10:01:00Z", n_writes=1, n_reads=1, kind="plan", kind_tier="heuristic")
    return Graph(nodes=[lead, sub, cx], edges=edges, artifacts=[art], meta={"n_nodes": 3, "unlinked_subagents": 0, "nodes_by_source": {"top": 1, "subagent": 1, "codex": 1}, "external_launchers": []})


def test_constants_are_the_documented_vocabularies():
    # dep and fan_in are the OCP core scheduling kinds; they are read from a
    # document, never inferred, and the viewer draws them like any other kind.
    assert EDGE_KINDS == ("dep", "fan_in", "spawn", "launch", "artifact")
    assert TIERS == ("verified", "heuristic", "reported")
    assert SOURCES == ("top", "subagent", "codex", "external")
    assert PHASES == ("build", "external", "post")


def test_to_dict_top_level_shape_and_version():
    d = _sample().to_dict()
    assert d["dagr_graph"] == 1
    assert list(d) == ["dagr_graph", "meta", "nodes", "edges", "artifacts"]
    assert [n["id"] for n in d["nodes"]] == ["lead", "plan", "cx"]
    assert [(e["src"], e["dst"], e["kind"], e["tier"]) for e in d["edges"]] == [("lead", "plan", "spawn", "verified"), ("lead", "cx", "launch", "heuristic"), ("plan", "cx", "artifact", "verified")]
    assert d["artifacts"][0]["consumers"] == ["cx"]


def test_to_dict_round_trips_through_json():
    g = _sample()
    d = json.loads(json.dumps(g.to_dict()))
    rebuilt = Graph(
        nodes=[GraphNode(**n) for n in d["nodes"]],
        edges=[GraphEdge(**e) for e in d["edges"]],
        artifacts=[Artifact(**a) for a in d["artifacts"]],
        meta=d["meta"],
    )
    assert rebuilt == g
    assert rebuilt.to_dict() == g.to_dict()


def test_every_edge_and_node_in_sample_carries_a_known_tier():
    g = _sample()
    assert all(e.kind in EDGE_KINDS and e.tier in TIERS for e in g.edges)
    assert all(n.source in SOURCES and n.phase in PHASES for n in g.nodes)
    assert all(n.role_tier in TIERS for n in g.nodes if n.role is not None)
    # A phase is an inference (spec section 0.2), so every set phase carries a tier.
    assert [(n.phase, n.phase_tier) for n in g.nodes] == [("build", "heuristic")] * 3
    assert all(n.phase_tier in TIERS for n in g.nodes if n.phase is not None)
    assert g.artifacts[0].kind_tier in TIERS


def test_node_defaults_leave_unknowns_none_not_zero():
    n = GraphNode(id="x", harness="codex", source="codex", session_path="/c/x.jsonl")
    assert n.usd is None and n.model is None and n.model_tier is None
    assert n.parent is None and n.spawn is None and n.launched_by is None
    assert (n.role, n.role_tier, n.role_evidence, n.phase, n.phase_tier) == (None, None, None, None, None)
    # Unknown duration and token counts are None, not 0.0 or {}: a zero would read as a
    # measured value. extract() counts them in meta (nodes_missing_wall_s, nodes_missing_tokens).
    assert n.wall_s is None and n.tokens is None


def test_default_factories_do_not_share_state():
    a, b = GraphEdge("s", "d", "spawn", "verified"), GraphEdge("s", "d", "spawn", "verified")
    a.detail["x"] = 1
    assert b.detail == {}
    n1, n2 = GraphNode(id="1", harness="h", source="top", session_path="/1", tokens={}), GraphNode(id="2", harness="h", source="top", session_path="/2", tokens={})
    n1.tokens["out"] = 3
    assert n2.tokens == {}
    g1, g2 = Graph(), Graph()
    g1.nodes.append(n1)
    assert g2.nodes == [] and g2.edges == [] and g2.artifacts == [] and g2.meta == {}


def test_artifact_recording_defaults_keep_unknown_distinct_from_zero_and_meta_independent():
    a = Artifact("/ws/a.py", "writer")
    b = Artifact("/ws/b.py", "writer")
    assert (
        a.bytes,
        a.bytes_tier,
        a.lines_added,
        a.lines_added_tier,
        a.lines_removed,
        a.lines_removed_tier,
        a.language,
        a.language_tier,
        a.tests_touched,
        a.tests_touched_tier,
        a.fate,
        a.fate_tier,
        a.meta,
    ) == (None, None, None, None, None, None, None, None, None, None, "unknown", None, {})
    a.bytes = 0
    a.bytes_tier = "verified"
    a.tests_touched = 0
    a.tests_touched_tier = "heuristic"
    a.meta["nested"] = {"unknown-key": [0, False, None]}
    assert b.bytes is None and b.tests_touched is None and b.meta == {}


def test_artifact_recording_survives_graph_json_round_trip_exactly():
    artifact = Artifact(
        "/ws/empty.py",
        "writer",
        bytes=0,
        bytes_tier="verified",
        lines_added=2,
        lines_added_tier="reported",
        lines_removed=0,
        lines_removed_tier="verified",
        language="py",
        language_tier="heuristic",
        tests_touched=0,
        tests_touched_tier="heuristic",
        fate="kept",
        fate_tier="reported",
        meta={"future": {"list": [0, False, None], "object": {"x.y": "z"}}},
    )
    document = json.loads(json.dumps(Graph(artifacts=[artifact]).to_dict()))
    rebuilt = Artifact(**document["artifacts"][0])
    assert rebuilt == artifact
    assert rebuilt.bytes == 0 and rebuilt.lines_removed == 0 and rebuilt.tests_touched == 0
    assert rebuilt.meta is not artifact.meta


def test_graph_node_lookup():
    g = _sample()
    assert g.node("plan").parent == "lead"
    assert g.node("cx").launched_by == {"id": "lead", "lag_s": 3.0}
    assert g.node("missing") is None


def test_to_dict_copies_meta_rather_than_aliasing_it():
    g = _sample()
    d = g.to_dict()
    d["meta"]["n_nodes"] = 99
    d["nodes"][0]["id"] = "changed"
    assert g.meta["n_nodes"] == 3
    assert g.nodes[0].id == "lead"


def test_to_dict_deep_copies_nested_meta():
    # meta holds nested tables (nodes_by_source, external_launchers); a shallow copy would
    # leave them aliased, so mutating the output would change the graph.
    g = _sample()
    d = g.to_dict()
    d["meta"]["nodes_by_source"]["top"] = 99
    d["meta"]["nodes_by_source"]["ghost"] = 1
    d["meta"]["external_launchers"].append("intruder")
    assert g.meta["nodes_by_source"] == {"top": 1, "subagent": 1, "codex": 1}
    assert g.meta["external_launchers"] == []
    assert d["meta"]["nodes_by_source"] is not g.meta["nodes_by_source"]
    assert g.to_dict()["meta"] == {"n_nodes": 3, "unlinked_subagents": 0, "nodes_by_source": {"top": 1, "subagent": 1, "codex": 1}, "external_launchers": []}


def _scan_nodes(*ids):
    return {nid: GraphNode(id=nid, harness="claude-code", source="top", session_path=f"/p/{nid}.jsonl") for nid in ids}


def test_artifact_counts_are_computed_from_the_scans_not_defaulted():
    # n_writes and n_reads have no unknown state: build_artifacts always counts them from
    # the per-node scans, so a 0 is a computed "nobody read this", never a placeholder.
    # Every scan event carries its tier, as scan.py emits them (Write/Edit/Read are verified).
    nodes = _scan_nodes("w", "r")
    scans = {
        "w": {
            "writes": [
                {"ts": "2026-08-31T10:01:00Z", "path": "/ws/x.md", "tier": "verified", "how": "Write"},
                {"ts": "2026-08-31T10:03:00Z", "path": "/ws/x.md", "tier": "verified", "how": "Edit"},
                {"ts": "2026-08-31T10:04:00Z", "path": "/ws/y.py", "tier": "verified", "how": "Write"},
            ],
            "reads": [{"ts": "2026-08-31T10:01:30Z", "path": "/ws/x.md", "tier": "verified", "how": "Read"}],
        },
        "r": {"writes": [], "reads": [{"ts": "2026-08-31T10:02:00Z", "path": "/ws/x.md", "tier": "verified", "how": "Read"}, {"ts": "2026-08-31T10:05:00Z", "path": "/ws/x.md", "tier": "verified", "how": "Read"}]},
    }
    arts, edges = build_artifacts(scans, nodes)
    by_id = {a.id: a for a in arts}
    x, y = by_id["/ws/x.md"], by_id["/ws/y.py"]
    assert (x.n_writes, x.n_reads) == (2, 3)  # the writer's own read counts as a read, not a consumption
    assert (x.producer, x.writers, x.consumers, x.first_write_ts) == ("w", ["w"], ["r"], "2026-08-31T10:01:00Z")
    assert (y.n_writes, y.n_reads, y.consumers) == (1, 0, [])
    assert (y.kind, y.kind_tier) == ("code", "heuristic")
    assert [(e.src, e.dst, e.tier, e.detail["path"]) for e in edges] == [("w", "r", "verified", "/ws/x.md")]
    assert (edges[0].detail["write_tier"], edges[0].detail["read_tier"]) == ("verified", "verified")


def test_artifact_edge_tier_is_the_weaker_of_write_and_read():
    # Spec section 0.2: an edge is only as strong as its weaker end. A write inferred from a
    # Bash command (heuristic) read by the Read tool (verified) cannot become a verified edge.
    nodes = _scan_nodes("w", "r")
    scans = {
        "w": {"writes": [{"ts": "2026-08-31T10:01:00Z", "path": "/ws/review.md", "tier": "heuristic", "how": "bash -o"}], "reads": []},
        "r": {"writes": [], "reads": [{"ts": "2026-08-31T10:02:00Z", "path": "/ws/review.md", "tier": "verified", "how": "Read"}]},
    }
    _, edges = build_artifacts(scans, nodes)
    assert [(e.src, e.dst, e.kind, e.tier) for e in edges] == [("w", "r", "artifact", "heuristic")]
    assert edges[0].detail == {"path": "/ws/review.md", "lag_s": 60.0, "write_tier": "heuristic", "read_tier": "verified"}
    # verified plus verified is the only way to a verified edge.
    scans["w"]["writes"][0]["tier"] = "verified"
    _, edges = build_artifacts(scans, nodes)
    assert edges[0].tier == "verified" and edges[0].detail["write_tier"] == "verified"
    # A verified write read by a heuristic read is heuristic too; reported sits between the two.
    scans["r"]["reads"][0]["tier"] = "heuristic"
    assert build_artifacts(scans, nodes)[1][0].tier == "heuristic"
    scans["r"]["reads"][0]["tier"] = "reported"
    assert build_artifacts(scans, nodes)[1][0].tier == "reported"
    assert all(e.tier in TIERS for e in edges)


def test_artifact_edge_event_without_a_tier_counts_as_heuristic():
    # A scan event that carries no tier is not silently promoted: it is treated as heuristic.
    nodes = _scan_nodes("w", "r")
    scans = {
        "w": {"writes": [{"ts": "2026-08-31T10:01:00Z", "path": "/ws/x.md"}], "reads": []},
        "r": {"writes": [], "reads": [{"ts": "2026-08-31T10:02:00Z", "path": "/ws/x.md", "tier": "verified", "how": "Read"}]},
    }
    _, edges = build_artifacts(scans, nodes)
    assert edges[0].tier == "heuristic"
    assert (edges[0].detail["write_tier"], edges[0].detail["read_tier"]) == ("heuristic", "verified")
