"""Workflow inference on graphs, OCP documents, sweep run files and RQ1 configs (D25, D33; lane 04)."""

from __future__ import annotations

import copy
import glob
import json
import os
from pathlib import Path

import pytest

from loopmath.types import Setting
from loopmath.workflows.format import catalog
from loopmath.workflows.ids import config_id, make_config
from loopmath.workflows.infer import InferError, infer, infer_detail, infer_workflow, load_source
from loopmath.workflows.ocp import configuration_to_ocp

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "tests" / "fixtures" / "graph" / "roundtrip"
SWARM_V02 = ROOT / "spec" / "examples" / "swarm-v02.ocp.json"


def _data_dir(env: str, sub: str | None = None) -> Path | None:
    """Real data named by an environment variable (never a path in the repo); None when unset."""
    value = os.environ.get(env)
    if not value:
        return None
    path = Path(value).expanduser()
    if sub and not any(path.glob("*.run.json")) and (path / sub).is_dir():
        path = path / sub  # the variable names the results folder rather than its run records
    return path


# The research verbs' variable: the sweep run records (`*.run.json`), or the results folder above them.
SWEEP = _data_dir("LOOPMATH_SWEEP_DIR", "dagr")
# RQ1 phase 1 runs: a folder of `<run>/config.json`. Test only.
RQ1 = _data_dir("LOOPMATH_RQ1_RUNS")


def _check(result, nodes):
    config = result.configuration
    assert config.id == config_id(config.workflow, config.settings)
    assert set(config.extra["node_vertex"]) == set(nodes)  # D25: every node
    assert set(config.extra["node_vertex"].values()) <= {p.id for p in config.workflow.pieces}
    assert set(config.settings) == {p.id for p in config.workflow.pieces}
    assert config.extra["shape"] == result.shape and config.extra["nearest_catalog"] == result.nearest_catalog
    assert 0.05 <= result.confidence <= 0.99 and config.extra["confidence"] == result.confidence


def _native(name):
    source = load_source(NATIVE / f"native-swarm-{name}.json")
    return source, [n["id"] for n in source["nodes"]]


def test_native_swarm_graphs():
    src, nodes = _native("b")
    r = infer_detail(src)
    _check(r, nodes)
    assert r.shape == "swarm" and r.nearest_catalog == "swarm"
    s = r.configuration.settings
    assert (s["plan"].model, s["plan"].effort) == ("opus-5", "xhigh")
    assert (s["work"].harness, s["work"].model, s["work"].effort) == ("claude-code", "sonnet-5", "xhigh")
    assert (s["review"].harness, s["review"].model, s["review"].effort) == ("codex", "gpt-5.6-sol", "medium")
    assert r.configuration.workflow.pieces[1].width == 6  # 8 at once, capped
    assert 0.6 < r.confidence < 0.9

    src, nodes = _native("c")
    r = infer_detail(src)
    _check(r, nodes)
    assert r.shape == "solo"
    assert (r.configuration.settings["implement"].model, r.configuration.settings["implement"].effort) == ("opus-5", "xhigh")
    assert any("outside the build" in x for x in r.reasons)  # the external codex review does not shape it

    src, nodes = _native("a")
    r = infer_detail(src)
    _check(r, nodes)
    assert r.shape == "swarm"


def test_graph_objects_and_graph_dicts_agree():
    from loopmath.graph.schema import Artifact, Graph, GraphEdge, GraphNode

    src, _ = _native("c")
    graph = Graph(nodes=[GraphNode(**n) for n in src["nodes"]], edges=[GraphEdge(**e) for e in src["edges"]],
                  artifacts=[Artifact(**a) for a in src["artifacts"]], meta=src["meta"])
    config, confidence = infer(graph)
    assert (config.id, confidence) == (infer(src)[0].id, infer(src)[1])


def test_ocp_v02_example_and_other_versions():
    doc = json.loads(SWARM_V02.read_text())
    r = infer_detail(doc)
    _check(r, [n["id"] for n in doc["nodes"]])
    assert r.shape == "swarm" and r.inferred_from == "ocp"
    assert r.configuration.settings["plan"].model == "opus-5"  # the planner session (canonical id), not the lead
    v01 = copy.deepcopy(doc)
    v01["ocp"] = "0.1"
    r1 = infer_detail(v01)
    _check(r1, [n["id"] for n in doc["nodes"]])
    assert r1.shape == "swarm" and r1.confidence < r.confidence


def test_a_declared_configuration_is_taken_as_is():
    cfg = make_config(catalog()["implement_review"], {"implement": Setting("claude-code", "claude-opus-5-5", "high"),
                                                      "review": Setting("codex", "gpt-6-astra", "high")})
    doc = {"ocp": "0.3", "run": {"id": "r1", "configuration": configuration_to_ocp(cfg)},
           "nodes": [{"id": "n1", "kind": "impl", "vertex": "implement"}, {"id": "n2", "kind": "review"}],
           "attempts": [{"id": "n2.a1", "node": "n2", "n": 1, "role": {"value": "reviewer", "tier": "reported"}}]}
    config, confidence = infer(doc)
    assert config.id == cfg.id and confidence == 0.95
    assert config.extra["node_vertex"] == {"n1": "implement", "n2": "review"}
    assert config.extra["node_vertex_basis"] == {"n1": "node", "n2": "role"}


def _declared(workflow, nodes, attempts=()):
    s = Setting("claude-code", "claude-opus-5-5", "high")
    cfg = make_config(workflow, {p.id: s for p in workflow.pieces})
    doc = {"ocp": "0.3", "run": {"id": "r1", "configuration": configuration_to_ocp(cfg)},
           "nodes": list(nodes), "attempts": list(attempts)}
    r = infer_detail(doc)
    assert r.inferred_from == "configuration" and r.configuration.id == cfg.id
    pieces = {p.id for p in r.configuration.workflow.pieces}
    assert set(r.node_vertex) == {n["id"] for n in nodes}  # D25: every node
    assert set(r.node_vertex.values()) <= pieces  # only pieces of the returned workflow
    assert r.configuration.extra["node_vertex"] == r.node_vertex
    return r


def _custom(pieces, edges, gates=()):
    from loopmath.types import Control, Gate, Piece, Workflow

    arts = sorted({x for e in edges for x in e} - {p for p, _ in pieces})
    return Workflow(id="custom", version=1, title="custom", pieces=tuple(Piece(p, role) for p, role in pieces),
                    artifacts=tuple(arts), edges=tuple(edges),
                    control=Control(gates=tuple(Gate(f"g_{a}", a, "review_approve") for a in gates), budget_rounds=1))


def _attempt(node, n=1, role=None, vertex=None):
    a = {"id": f"{node}.a{n}", "node": node, "n": n}
    if role:
        a["role"] = {"value": role, "tier": "reported"}
    if vertex:
        a["vertex"] = vertex
    return a


def test_declared_custom_piece_found_through_the_attempt_vertex():
    # lane 04 review of b90bdbb: a custom piece id, no node.vertex, the attempt names the piece
    wf = _custom([("builder", "implementer")], [("issue", "builder"), ("builder", "diff")])
    r = _declared(wf, [{"id": "implement", "kind": "impl"}], [_attempt("implement", vertex="builder")])
    assert r.node_vertex == {"implement": "builder"} and r.confidence == 0.95
    assert r.configuration.extra["node_vertex_basis"] == {"implement": "attempt"}


def test_declared_custom_pieces_found_by_role():
    wf = _custom([("architect", "planner"), ("builder", "implementer"), ("checker", "reviewer")],
                 [("issue", "architect"), ("architect", "spec"), ("spec", "builder"), ("builder", "diff"),
                  ("diff", "checker"), ("checker", "verdict")], gates=["checker"])
    nodes = [{"id": "p"}, {"id": "d"}, {"id": "o", "kind": "ops"}, {"id": "g", "kind": "gate"}, {"id": "v"}]
    attempts = [_attempt("p", role="planner"), _attempt("d", role="developer"), _attempt("v", role="critic")]
    r = _declared(wf, nodes, attempts)
    assert r.node_vertex == {"p": "architect", "d": "builder", "o": "architect", "g": "checker", "v": "checker"}
    assert set(r.configuration.extra["node_vertex_basis"].values()) == {"role"} and r.confidence == 0.95


def test_declared_unmatched_and_ambiguous_nodes_lower_confidence():
    wf = _custom([("implement", "implementer"), ("r1", "reviewer"), ("r2", "reviewer")],
                 [("issue", "implement"), ("implement", "diff"), ("diff", "r1"), ("diff", "r2"),
                  ("r1", "v1"), ("r2", "v2")])
    nodes = [{"id": "a", "vertex": "implement"}, {"id": "b"}, {"id": "c"}, {"id": "d", "vertex": "nope"}]
    attempts = [_attempt("b", role="reviewer"), _attempt("c", 1, vertex="r2"), _attempt("c", 2, vertex="r1")]
    r = _declared(wf, nodes, attempts)
    basis = r.configuration.extra["node_vertex_basis"]
    assert r.node_vertex == {"a": "implement", "b": "r1", "c": "r1", "d": "implement"}
    assert basis == {"a": "node", "b": "ambiguous", "c": "ambiguous", "d": "default"}
    assert r.confidence == round(0.95 - 0.1 * 3 / 4, 3)
    assert any("vertex 'nope'" in x for x in r.reasons) and any("fits pieces r1, r2" in x for x in r.reasons)


def _sweep(planner=True):
    exp = {"sweep": "s", "task": "t1", "arm": {"id": "a", "family": "claude", "model": "claude-fable-5", "effort": "high"},
           "reviewer": {"model": "gpt-5.6-sol", "effort": "xhigh", "family": "codex"}, "accepted": True,
           "dev_attempts": 2, "planner_shared": True}
    if planner:
        exp["planner"] = {"model": "gpt-5.6-luna", "effort": "low", "family": "codex"}
    tasks = [{"id": "PLAN", "kind": "plan", "owner": "planner", "attempts": [{"model": "opus5·xhigh"}]},
             {"id": "DEV", "kind": "impl", "owner": "developer", "attempts": [{"model": "fable·high"}]},
             {"id": "REV", "kind": "review", "owner": "reviewer", "attempts": [{"model": "sol·xhigh"}]},
             {"id": "GATE", "kind": "gate", "owner": None, "attempts": [{"actor": "harness"}]}]
    return {"dagr": 3, "run": {}, "ext": {"experiment": exp}, "tasks": tasks}


def test_sweep_run_files():
    r = infer_detail(_sweep())
    _check(r, ["PLAN", "DEV", "REV", "GATE"])
    assert r.shape == "plan_implement_review" and r.confidence == 0.98 and r.inferred_from == "sweep"
    s = r.configuration.settings
    assert (s["plan"].harness, s["plan"].model, s["plan"].effort) == ("codex", "gpt-5.6-luna", "low")
    assert (s["implement"].harness, s["implement"].model) == ("claude-code", "fable-5")
    assert r.configuration.extra["node_vertex"]["GATE"] == "review"
    shared = infer_detail(_sweep(planner=False))
    assert shared.configuration.settings["plan"].model == "opus-5" and shared.confidence == 0.95
    assert any("shared" in x for x in shared.reasons)


@pytest.mark.parametrize("topology,agents,shape,nearest", [
    ("solo", [["solo", "solo", "gpt-5.6-sol", "max"]], "solo", "solo"),
    ("reviewer", [["worker", "worker", "gpt-5.6-luna", "high"], ["reviewer", "reviewer", "gpt-5.6-sol", "max"]],
     "implement_review", "implement_review"),
    ("team", [[f"worker-{i}", "worker", "gpt-5.6-sol", "xhigh"] for i in (1, 2, 3)], "team", "swarm"),
    ("best", [[f"worker-{i}", "worker", "gpt-5.6-sol", "xhigh"] for i in (1, 2, 3)], "best_of_n", "best_of_n"),
    ("planner", [["planner", "planner", "gpt-5.6-sol", "max"]] + [[f"worker-{i}", "worker", "gpt-5.6-luna", "high"]
                                                                  for i in (1, 2, 3)], "plan_team", "swarm"),
])
def test_rq1_topologies(topology, agents, shape, nearest):
    r = infer_detail({"arm": "Axx", "topology": topology, "agents": agents})
    _check(r, [a[0] for a in agents])
    assert (r.shape, r.nearest_catalog, r.confidence) == (shape, nearest, 0.98)
    if shape in ("team", "best_of_n", "plan_team"):
        assert max(p.width for p in r.configuration.workflow.pieces) == 3


def test_rq1_mixed_settings_lower_confidence():
    agents = [["worker-1", "worker", "gpt-5.6-sol", "max"], ["worker-2", "worker", "gpt-5.6-luna", "xhigh"],
              ["worker-3", "worker", "gpt-5.6-luna", "xhigh"]]
    r = infer_detail({"topology": "team", "agents": agents})
    assert r.configuration.settings["work"].model == "gpt-5.6-luna" and r.confidence < 0.98


def _node(i, role, *, source="subagent", start="2026-09-01T10:00:00Z", wall=600.0, model="opus-5", effort="high",
          tier="heuristic", phase="build", harness="claude-code"):
    return {"id": i, "harness": harness, "source": source, "session_path": i, "model": model, "effort": effort,
            "ts": start, "wall_s": wall, "role": role, "role_tier": tier, "phase": phase}


def test_sequential_workers_implement_and_overlapping_workers_team():
    lead = _node("L", "lead", source="top", wall=7200.0)
    seq = [_node("d1", "dev", start="2026-09-01T10:00:00Z"), _node("d2", "dev", start="2026-09-01T10:20:00Z")]
    r = infer_detail({"nodes": [lead] + seq})
    assert r.shape == "plan_implement" and r.configuration.extra["node_vertex"] == {"L": "plan", "d1": "implement",
                                                                                    "d2": "implement"}
    par = [_node("d1", "dev"), _node("d2", "dev", start="2026-09-01T10:05:00Z")]
    assert infer_detail({"nodes": [lead] + par}).shape == "plan_team"


def test_missing_labels_timing_and_models():
    nodes = [_node("top", None, source="top", tier=None), _node("sub", None, start=None, tier=None)]
    r = infer_detail({"nodes": nodes})
    assert r.shape == "plan_implement" and r.confidence < 0.8
    assert any("no role label" in x for x in r.reasons)
    two = [_node("top", "lead", source="top"), _node("a", "dev", start=None), _node("b", "dev", start=None)]
    r = infer_detail({"nodes": two})
    assert r.shape == "plan_team" and any("timing is missing" in x for x in r.reasons)
    with pytest.raises(InferError):
        infer({"nodes": []})
    with pytest.raises(InferError):
        infer({"nodes": [_node("x", "solo", source="top", model=None)]})
    with pytest.raises(InferError):
        infer({"something": "else"})


def test_a_reviewer_loop_in_a_single_session_line_raises_rounds():
    nodes = [_node("s", "solo", source="top", wall=9000.0)] + [
        _node(f"r{i}", "reviewer", source="codex", harness="codex", model="gpt-5.6-sol",
              start=f"2026-09-01T1{i}:00:00Z") for i in range(5)]
    r = infer_detail({"nodes": nodes})
    assert r.shape == "implement_review" and r.configuration.workflow.control.budget_rounds == 5


def test_infer_workflow_is_the_migration_entry_point():
    assert infer_workflow is infer


@pytest.mark.skipif(SWEEP is None or not SWEEP.is_dir(), reason="LOOPMATH_SWEEP_DIR is not set to the sweep results")
def test_every_sweep_run_file_is_plan_implement_review():
    files = sorted(glob.glob(str(SWEEP / "*.run.json")))
    assert files
    for f in files:
        r = infer_detail(f)
        assert r.shape == "plan_implement_review" and r.confidence >= 0.95, f


@pytest.mark.skipif(RQ1 is None or not RQ1.is_dir(), reason="LOOPMATH_RQ1_RUNS is not set to the RQ1 phase 1 runs")
def test_every_rq1_run_config():
    expected = {"solo": "solo", "reviewer": "implement_review", "team": "team", "best": "best_of_n",
                "planner": "plan_team"}
    runs = sorted(RQ1.glob("*/config.json"))
    assert runs
    for path in runs:
        topology = json.loads(path.read_text())["topology"]
        assert infer_detail(path.parent).shape == expected[topology], path
