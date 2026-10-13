"""DOT rendering: the expected nodes, fills, edge styles and tier colours."""

from __future__ import annotations

from loopmath.graph.extract import extract
from loopmath.graph.render import to_dot
from loopmath.graph.schema import Artifact, Graph, GraphEdge, GraphNode

from test_graph_extract import ALPHA, skeleton_records

VERIFIED = "#111827"
UNVERIFIED = "#9ca3af"


def _graph() -> Graph:
    nodes = [
        GraphNode(id="lead", harness="claude-code", source="top", session_path="/p/lead.jsonl", model="claude-opus-5", usd=1.25, role="lead", role_tier="heuristic", phase="build", phase_tier="heuristic"),
        GraphNode(id="plan", harness="claude-code", source="subagent", session_path="/p/a.jsonl", model="claude-opus-5", usd=0.4, role="planner", role_tier="reported", phase="build", phase_tier="heuristic"),
        GraphNode(id="orphan", harness="claude-code", source="subagent", session_path="/p/b.jsonl", role="dev", role_tier="heuristic", phase="build", phase_tier="heuristic"),
        GraphNode(id="cx", harness="codex", source="codex", session_path="/c/r.jsonl", model="gpt-5", usd=0.55, role="reviewer", role_tier="heuristic", phase="build", phase_tier="heuristic"),
        GraphNode(id="reader", harness="claude-code", source="top", session_path="/p/r.jsonl", role=None, phase="post", phase_tier="heuristic"),
        GraphNode(id="analyst", harness="claude-code", source="external", session_path="/o/a.jsonl", role="external", role_tier="heuristic", phase="external", phase_tier="heuristic"),
        GraphNode(id='q"uote', harness="codex", source="codex", session_path="/c/q.jsonl", model='m"x'),
    ]
    edges = [
        GraphEdge("lead", "plan", "spawn", "verified", {"tool_use_id": "toolu_1"}),
        GraphEdge("lead", "orphan", "spawn", "heuristic", {"reason": "path containment"}),
        GraphEdge("lead", "cx", "launch", "heuristic", {"lag_s": 3.0, "how": "inside a running Bash call naming the CLI"}),
        GraphEdge("analyst", "cx", "launch", "heuristic", {"how": "no lag recorded"}),
        GraphEdge("plan", "reader", "artifact", "verified", {"path": "/ws/alpha/plan.md", "lag_s": 1150.0, "write_tier": "verified", "read_tier": "verified"}),
        # A Bash-inferred write (heuristic) read by the Read tool (verified): the edge is heuristic.
        GraphEdge("cx", "lead", "artifact", "heuristic", {"path": "/ws/alpha/review.md", "lag_s": 10.0, "write_tier": "heuristic", "read_tier": "verified"}),
    ]
    arts = [
        Artifact(id="/ws/alpha/plan.md", producer="plan", writers=["plan"], consumers=["reader"], n_writes=1, n_reads=1, kind="plan", kind_tier="heuristic"),
        Artifact(id="/ws/alpha/src/scan.py", producer="orphan", writers=["orphan"], consumers=[], n_writes=1, kind="code", kind_tier="heuristic"),
        Artifact(id="/ws/alpha/review.md", producer="cx", writers=["cx"], consumers=["lead"], n_writes=1, n_reads=1, kind="review", kind_tier="heuristic"),
    ]
    return Graph(nodes=nodes, edges=edges, artifacts=arts, meta={})


def _lines(**kw) -> list[str]:
    return to_dot(_graph(), **kw).splitlines()


def test_header_and_footer():
    dot = to_dot(_graph())
    lines = dot.splitlines()
    assert lines[0] == "digraph loopmath {"
    assert lines[1] == "  rankdir=LR;"
    assert lines[2].startswith("  node [shape=box")
    assert dot.endswith("\n}\n")


def test_node_label_is_role_model_and_cost():
    # Labels hold literal newlines between role, model and cost (valid inside a DOT string).
    dot = to_dot(_graph())
    assert '\n  "lead" [label="lead\nclaude-opus-5\n$1.25", fillcolor="#dbeafe", color="#9ca3af"];\n' in dot
    # A reported role tier gets no grey border; the heuristic one above does.
    assert '\n  "plan" [label="planner\nclaude-opus-5\n$0.40", fillcolor="#dcfce7"];\n' in dot


def test_unlabeled_node_falls_back_to_source_and_post_phase_is_dashed():
    lines = _lines()
    assert '  "reader" [label="top", fillcolor="#dbeafe", style="filled,rounded,dashed"];' in lines


def test_fill_colour_per_source():
    dot = to_dot(_graph())
    assert '\n  "cx" [label="reviewer\ngpt-5\n$0.55", fillcolor="#fef3c7", color="#9ca3af"];\n' in dot
    assert '\n  "analyst" [label="external", fillcolor="#eeeeee", color="#9ca3af"];\n' in dot
    assert '\n  "orphan" [label="dev", fillcolor="#dcfce7", color="#9ca3af"];\n' in dot


def test_quotes_in_ids_and_labels_are_escaped():
    dot = to_dot(_graph())
    assert '\n  "q\\"uote" [label="codex\nm\\"x", fillcolor="#fef3c7"];\n' in dot


def test_spawn_edges_solid_with_tier_colour():
    lines = _lines()
    assert f'  "lead" -> "plan" [style=solid, color="{VERIFIED}"];' in lines
    assert f'  "lead" -> "orphan" [style=solid, color="{UNVERIFIED}"];' in lines


def test_launch_edges_dashed_with_lag_label_when_known():
    lines = _lines()
    assert f'  "lead" -> "cx" [style=dashed, color="{UNVERIFIED}", label="+3.0s"];' in lines
    assert f'  "analyst" -> "cx" [style=dashed, color="{UNVERIFIED}"];' in lines


def test_artifact_edges_are_drawn_through_note_nodes_not_directly():
    lines = _lines()
    assert '  "/ws/alpha/plan.md" [shape=note, fillcolor="#f3f4f6", label="plan.md"];' in lines
    assert f'  "plan" -> "/ws/alpha/plan.md" [style=dotted, color="{VERIFIED}"];' in lines
    assert f'  "/ws/alpha/plan.md" -> "reader" [style=dotted, color="{VERIFIED}"];' in lines
    assert not any(l.startswith('  "plan" -> "reader"') for l in lines)


def test_verified_artifact_connectors_are_dark():
    # Both connectors of a verified artifact relation carry the verified colour: the tier
    # of the artifact edge is not erased by drawing it through the note node.
    lines = _lines()
    plan_lines = [l for l in lines if "/ws/alpha/plan.md" in l and "->" in l]
    assert len(plan_lines) == 2
    assert all(f'color="{VERIFIED}"' in l for l in plan_lines)


def test_heuristic_artifact_connectors_are_grey():
    # review.md was written by a Bash command (heuristic write) and read with the Read
    # tool: the edge is heuristic, so producer and consumer connectors are both grey.
    lines = _lines()
    assert '  "/ws/alpha/review.md" [shape=note, fillcolor="#f3f4f6", label="review.md"];' in lines
    assert f'  "cx" -> "/ws/alpha/review.md" [style=dotted, color="{UNVERIFIED}"];' in lines
    assert f'  "/ws/alpha/review.md" -> "lead" [style=dotted, color="{UNVERIFIED}"];' in lines
    assert not any(l.startswith('  "cx" -> "lead"') for l in lines)


def test_consumed_default_hides_unread_artifacts():
    lines = _lines()
    assert not any("scan.py" in l for l in lines)
    assert sum(1 for l in lines if "[shape=note" in l) == 2  # plan.md and review.md


def test_artifacts_all_draws_unread_artifacts_too():
    lines = _lines(artifacts="all")
    assert '  "/ws/alpha/src/scan.py" [shape=note, fillcolor="#f3f4f6", label="scan.py"];' in lines
    # No artifact edge ends at scan.py, so its producer connector has no tier to inherit and stays grey.
    assert f'  "orphan" -> "/ws/alpha/src/scan.py" [style=dotted, color="{UNVERIFIED}"];' in lines
    assert sum(1 for l in lines if "[shape=note" in l) == 3


def test_artifacts_none_draws_no_notes_but_keeps_other_edges():
    lines = _lines(artifacts="none")
    assert not any("[shape=note" in l or "style=dotted" in l for l in lines)
    assert sum(1 for l in lines if "style=solid" in l) == 2
    assert sum(1 for l in lines if "style=dashed" in l) == 2


def test_dot_of_the_skeleton_fixture():
    g = extract(skeleton_records(), workspaces=[ALPHA])
    dot = to_dot(g)
    for nid in ("lead", "plan", "dev", "orphan", "lost", "reader", "codex-review", "codex-audit", "codex-stray", "analyst"):
        assert f'  "{nid}" [' in dot
    lines = dot.splitlines()
    assert sum(1 for l in lines if "style=solid" in l) == 3  # two verified spawns, one heuristic
    assert sum(1 for l in lines if "style=dashed" in l) == 2  # both launches
    # Dark lines: the two verified spawn edges plus the three connectors of the verified plan.md relation.
    assert sum(1 for l in lines if f'color="{VERIFIED}"' in l) == 5
    assert '  "lead" -> "codex-review" [style=dashed, color="#9ca3af", label="+3.0s"];' in lines
    assert f'  "plan" -> "/ws/alpha/plan.md" [style=dotted, color="{VERIFIED}"];' in lines
    assert f'  "/ws/alpha/plan.md" -> "dev" [style=dotted, color="{VERIFIED}"];' in lines
    assert f'  "/ws/alpha/plan.md" -> "reader" [style=dotted, color="{VERIFIED}"];' in lines
    assert not any('"/ws/alpha/plan.md" -> "lead"' in l for l in lines)  # lead's pre-write read is not a consumption
    assert "scan.py" not in dot
    assert to_dot(g) == dot
