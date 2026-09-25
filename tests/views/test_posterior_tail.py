"""A mean above its interval's upper end means a real heavy tail. The view keeps the mean and adds
"the average is pulled up by rare very large outcomes" in text and HTML; the JSON is unchanged."""

from __future__ import annotations

import copy

from loopmath import cli
from loopmath.types import NodeSummary
from loopmath.views import common, posterior as P
from tests.views._posterior_helpers import FakeState, iv, needs_node, run_page

NOTE = "the average is pulled up by rare very large outcomes"


def _display(mean: float, lo: float, hi: float, head: str = "cost") -> dict:
    return {"level": "model", "key": "claude-opus-5", "head": head, "display": {"mean": mean, "lo": lo, "hi": hi}}


def test_terminal_values_keep_the_mean_and_add_the_note():
    assert P.TAIL_NOTE == NOTE
    assert P.fmt_display(_display(3.0, 0.5, 2.5)) == f"x3.00 (0.50 to 2.50; {NOTE})"
    assert P.fmt_display(_display(1.2, 0.5, 2.5)) == "x1.20 (0.50 to 2.50)"
    assert P.fmt_display(_display(0.2, 0.5, 2.5)) == "x0.20 (0.50 to 2.50)"  # below the interval: no note
    assert P.fmt_display(_display(12.0, 1.0, 10.0, "success")) == f"+12 pp (+1 to +10; {NOTE})"
    heads = {"score:rank": {"kind": "shift", "unit": "rank"}}
    assert P.fmt_display(_display(1_100_000.0, 0.2, 19_497.0, "score:rank"), heads) == f"+1100000 rank (0 to +19497; {NOTE})"
    assert P._iv({"mean": 5.2, "lo": 0.25, "hi": 2.25}, P._usd) == f"$5.20 ($0.25 to $2.25; {NOTE})"
    assert P._iv({"mean": 2.25, "lo": 0.25, "hi": 2.25}, P._usd) == "$2.25 ($0.25 to $2.25)"


def _tail_workflow() -> dict:
    """One piece whose dollars per round and per run have their means above the interval (a heavy tail)."""
    def money(mean, lo, hi):
        return {"usd": {"mean": mean, "lo": lo, "hi": hi}, "tokens": {"mean": 900_000.0, "lo": 50_000.0, "hi": 800_000.0}}
    graph = {"config": "cfg_7a17a17a17a1", "nodes": [
                 {"id": "issue", "kind": "artifact"}, {"id": "implement", "kind": "piece", "role": "implementer"},
                 {"id": "diff", "kind": "artifact"}],
             "edges": [["issue", "implement"], ["implement", "diff"]], "gates": [], "budget_rounds": 1}
    one = {"mean": 1.0, "lo": 1.0, "hi": 1.0}
    per_piece = {"implement": {"piece": "implement", "cost": money(5.2, 0.25, 2.25), "cost_per_round": money(5.2, 0.25, 2.25),
                               "gate_pass": None, "rounds": one}}
    prediction = {"cost": money(5.2, 0.25, 2.25), "p_success": {"mean": 0.9, "lo": 0.7, "hi": 1.0}, "rounds": one, "support": 3}
    return {"config": "cfg_7a17a17a17a1", "label": "solo: tail", "graph": graph, "per_piece": per_piece,
            "prediction": prediction}


def test_terminal_workflow_lines_carry_the_note():
    data = {"fit": {"id": "fit_x", "at": "t", "n_runs": 3}, "levels": {}, "workflow": P.enrich_workflow(_tail_workflow())}
    line = next(ln for ln in P.summary_lines(data) if ln.startswith("  implement:"))
    assert line == (f"  implement: $5.20 ($0.25 to $2.25; {NOTE}) per round, $5.20 ($0.25 to $2.25; {NOTE}) per run,"
                    f" 900k tokens ({NOTE}) per run, expected rounds 1, 100% of cost")


def _gate_workflow() -> dict:
    """implement then review, with a gate after review: review's pass chance and rounds have their means above the interval."""
    def money(x):
        return {"usd": {"mean": x, "lo": x, "hi": x}, "tokens": {"mean": 1000.0 * x, "lo": 1000.0 * x, "hi": 1000.0 * x}}
    rounds = {"mean": 1.6, "lo": 1.0, "hi": 1.5}
    graph = {"config": "cfg_9a7e9a7e9a7e", "nodes": [
                 {"id": "issue", "kind": "artifact"}, {"id": "implement", "kind": "piece", "role": "implementer"},
                 {"id": "diff", "kind": "artifact"}, {"id": "review", "kind": "piece", "role": "reviewer"},
                 {"id": "verdict", "kind": "artifact"}],
             "edges": [["issue", "implement"], ["implement", "diff"], ["diff", "review"], ["review", "verdict"]],
             "gates": [{"id": "g_review", "after": "review", "rule": "review_approve", "on_fail": "implement"}], "budget_rounds": 3}
    per_piece = {"implement": {"piece": "implement", "cost": money(1.0), "cost_per_round": money(1.0), "gate_pass": None, "rounds": rounds},
                 "review": {"piece": "review", "cost": money(1.0), "cost_per_round": money(1.0),
                            "gate_pass": {"mean": 0.9, "lo": 0.5, "hi": 0.85}, "rounds": rounds}}
    return {"config": "cfg_9a7e9a7e9a7e", "label": "implement_review: gate", "graph": graph, "per_piece": per_piece,
            "gates": [{"id": "g_review", "pass_by_round": [{"mean": 0.9, "lo": 0.5, "hi": 0.85}, {"mean": 0.7, "lo": 0.5, "hi": 0.8}]}]}


def test_terminal_bare_means_carry_the_note_too():
    """Expected rounds and tokens are shown without bounds, and still get the note."""
    data = {"fit": {"id": "fit_x", "at": "t", "n_runs": 3}, "levels": {}, "workflow": P.enrich_workflow(_gate_workflow())}
    lines = P.summary_lines(data)
    review = next(ln for ln in lines if ln.startswith("  review:"))
    assert f"1k tokens per run, expected rounds 1.6 ({NOTE}), 50% of cost" in review
    gate = next(ln for ln in lines if ln.startswith("  gate g_review"))
    assert gate == f"  gate g_review after review: pass 90% (50% to 85%; {NOTE}) per round, expected rounds 1.6 ({NOTE})"
    loop = next(ln for ln in lines if ln.startswith("  repair loop"))
    assert loop.startswith(f"  repair loop, review back to implement: expected rounds 1.6 ({NOTE});")


def _tail_nodes() -> list[NodeSummary]:
    return [NodeSummary("family", "opus", "cost", iv(0.5, -1.0, 0.9), iv(3.0, 0.5, 2.5), 12, "provider:anthropic", {"user": 12}),
            NodeSummary("family", "sol", "cost", iv(0.1, -0.2, 0.3), iv(1.1, 0.8, 1.4), 12, "provider:openai", {"user": 12})]


def test_json_is_unchanged_and_the_terminal_says_it(tmp_path, capsys, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    state = FakeState(_tail_nodes())
    monkeypatch.setattr(P, "_load_state", lambda h: state)
    assert cli.main(["posterior", "--home", str(home), "--json"]) == 0
    out = capsys.readouterr().out
    assert NOTE not in out and "pulled up" not in out
    assert cli.main(["posterior", "--home", str(home), "--level", "model"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert f"  family opus: x3.00 (0.50 to 2.50; {NOTE}), 12 runs" in lines
    assert "  family sol: x1.10 (0.80 to 1.40), 12 runs" in lines


def test_page_and_python_use_the_same_words():
    assert P.TAIL_NOTE is common.TAIL_NOTE  # one string, lane 12's views/common.py
    assert f"var TAIL_NOTE = '{common.TAIL_NOTE}';" in common.asset("estimates.js")


@needs_node
def test_page_shows_the_note_in_cells_tips_prose_and_under_the_graph(tmp_path):
    workflow = _tail_workflow()
    data = {"schema": P.SCHEMA, "generated_at": "x", "fit": {"id": "fit_x", "at": "y", "n_runs": 12},
            "levels": P.group_levels(_tail_nodes()), "workflow": workflow, "workflows": [workflow]}
    before = copy.deepcopy(data)
    page = P.render(data)
    assert data == before and P.extract_data(page)["workflow"]["per_piece"] == workflow["per_piece"]
    out = run_page(page, tmp_path, "#graph")
    cost = out["levels"]["cost"]
    block = f'<span class="why">{NOTE}</span>'
    assert f'<td class="eff">x3.00 (0.50 to 2.50){block}</td>' in cost
    assert '<td class="eff">x1.10 (0.80 to 1.40)</td>' in cost
    assert f"x3.00 (0.50 to 2.50; {NOTE})" in cost  # the row's hover and the bar's label
    graph = out["graph"][0]
    assert f"Predicted run: $5.20 ($0.25 to $2.25; {NOTE}), 900k tokens ({NOTE}); chance" in graph
    assert f'<p class="note">In the graph, cost of implement, tokens of implement: {NOTE}.</p>' in graph
    assert "$5.20 ($0.25 to $2.25) per round" in graph  # the box keeps the value; the caption carries the note
    assert f'<td class="num">$5.20 ($0.25 to $2.25){block}</td>' in graph
    assert f'<td class="num">900k (50k to 800k){block}</td>' in graph
    assert f"cost per run: $5.20 ($0.25 to $2.25; {NOTE})" in graph  # the piece's hover


@needs_node
def test_page_bare_means_in_the_graph_gates_and_heatmap(tmp_path):
    """On the page: the graph's gate and loop labels, per-round gate passes and heatmap cells show means only."""
    workflow = _gate_workflow()
    cells = [NodeSummary("family_effort", "opus|max", "cost", iv(0.5, -1.0, 0.9), iv(3.0, 0.5, 2.5), 12, "family:opus", {"user": 12}),
             NodeSummary("family_effort", "opus|low", "cost", iv(-0.2, -0.4, 0.0), iv(0.8, 0.7, 1.0), 12, "family:opus", {"user": 12})]
    data = {"schema": P.SCHEMA, "generated_at": "x", "fit": {"id": "fit_x", "at": "y", "n_runs": 12},
            "levels": P.group_levels(cells), "workflow": workflow, "workflows": [workflow]}
    out = run_page(P.render(data), tmp_path, "#graph")
    heat = out["levels"]["cost"]
    assert f'<p class="legend">opus|max: {NOTE}.</p>' in heat and "opus|low:" not in heat
    graph = out["graph"][0]
    assert (f'<p class="note">In the graph, pass chance of gate g_review, rounds of gate g_review,'
            f' rounds of the repair review back to implement: {NOTE}.</p>') in graph
    assert f"round 1: 90% ({NOTE}), round 2: 70%<" in graph
