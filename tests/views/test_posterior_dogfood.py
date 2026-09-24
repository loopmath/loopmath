"""Found by running `loopmath posterior` on a real local store (dogfood, 2026-09-23): the terminal
words keys as the page does, says what a repair loop's share is, and a missing --workflow says what to pass."""

from __future__ import annotations

from loopmath import cli
from loopmath.output import EXIT_NOT_FOUND
from loopmath.views import posterior as P
from tests.views._posterior_helpers import needs_node, run_page


def iv(mean: float, lo: float, hi: float) -> dict:
    return {"mean": mean, "lo": lo, "hi": hi}


def _node(level: str, key: str) -> dict:
    return {"level": level, "key": key, "head": "cost", "effect": iv(0, -0.1, 0.1), "display": iv(1.0, 0.9, 1.1),
            "support": 12, "parent": None, "source_mix": {"sweep": 12}}


def test_terminal_keys_read_like_the_page():
    assert P.key_text(_node("position", "plan_implement_review#0")) == "piece 1 of plan_implement_review"
    assert P.key_text(_node("position", "solo")) == "solo"
    assert P.key_text(_node("feature:has_tests", "has_tests=yes")) == "yes"
    assert P.key_text(_node("family_effort", "opus|high")) == "opus|high"
    levels = {"topology": [_node("position", "swarm#2")], "feature": [_node("feature:size", "size=l")]}
    lines = P.summary_lines({"fit": {"id": "fit_x", "at": "t", "n_runs": 12}, "levels": levels})
    assert "  position piece 3 of swarm: x1.00 (0.90 to 1.10), 12 runs, shared data" in lines
    assert "  feature:size l: x1.00 (0.90 to 1.10), 12 runs, shared data" in lines


def test_terminal_repair_loop_names_the_pieces_behind_its_share():
    loop = {"gate": "g_review", "from": "implement", "to": "review", "pieces": ["implement", "review"],
            "rounds": iv(1.03, 1.0, 1.2), "share": 1.0}
    data = {"fit": {"id": "fit_x", "at": "t", "n_runs": 12}, "levels": {},
            "workflow": {"label": "implement_review: x", "per_piece": {}, "gates": [], "loops": [loop]}}
    assert P.summary_lines(data)[-1] == ("  repair loop, review back to implement: expected rounds 1.03;"
                                         " the pieces it reruns (implement, review) are 100% of cost")


def test_missing_workflow_says_what_to_pass(tmp_path, capsys, monkeypatch):
    from tests.views._posterior_helpers import FakeState, lane5_nodes

    home = tmp_path / "home"
    home.mkdir()
    state = FakeState(lane5_nodes())
    monkeypatch.setattr(P, "_load_state", lambda h: state)
    assert cli.main(["posterior", "--home", str(home), "--workflow", "implement_review"]) == EXIT_NOT_FOUND
    err = capsys.readouterr().err
    assert "implement_review is a catalog workflow with no models set" in err
    assert "`loopmath recommend --json`" in err and "a workflow file with settings" in err
    assert cli.main(["posterior", "--home", str(home), "--workflow", "cfg_ffffffffffff"]) == EXIT_NOT_FOUND
    assert "configuration ids (cfg_...) come from `loopmath recommend --json` and your runs" in capsys.readouterr().err


def test_untyped_group_is_named_for_people():
    """D92: the prior's `unknown` task type (the E0 runs) reads as untyped, in the terminal and on the page."""
    assert P.key_text(_node("type", "unknown")) == "untyped (no task type recorded)" == P.UNTYPED
    assert P.key_text(_node("type", "feature")) == "feature"
    assert P.key_text(_node("repo", "unknown")) == "unknown"
    lines = P.summary_lines({"fit": {"id": "fit_x", "at": "t", "n_runs": 12}, "levels": {"type": [_node("type", "unknown")]}})
    assert "  type untyped (no task type recorded): x1.00 (0.90 to 1.10), 12 runs, shared data" in lines
    child = {**_node("repo", "unknown/acme/app"), "support": 0, "parent": "type:unknown"}
    assert P._support_text(child) == "no runs here: the parent's estimate (type untyped (no task type recorded)), widened"
    assert f"var UNTYPED = '{P.UNTYPED}';" in P.PAGE_JS


@needs_node
def test_page_names_the_untyped_group(tmp_path):
    from loopmath.types import NodeSummary
    from tests.views._posterior_helpers import iv as interval

    nodes = [NodeSummary("type", "unknown", "cost", interval(0, -0.1, 0.1), interval(1.12, 0.58, 1.79), 809, None,
                         {"e0": 809}),
             NodeSummary("repo", "unknown/acme/app", "cost", interval(0, -0.1, 0.1), interval(1.0, 0.9, 1.1), 0,
                         "type:unknown", {})]
    data = {"schema": P.SCHEMA, "generated_at": "x", "fit": {"id": "fit_x", "at": "y", "n_runs": 809},
            "levels": P.group_levels(nodes), "workflow": None, "workflows": []}
    cost = run_page(P.render(data), tmp_path)["levels"]["cost"]
    assert '<span class="lvl">type</span> untyped (no task type recorded)' in cost
    assert "the parent's estimate (type untyped (no task type recorded)), widened" in cost
    assert '<span class="lvl">type</span> unknown' not in cost
