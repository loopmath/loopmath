"""The posterior view on lane 5's own names and metadata (belief/forest.py, belief/fit.py):
levels `model`, `family_effort`, `role_family`, `position`, parents as node ids, meta.json per head."""

from __future__ import annotations

import json
import re

from loopmath.views import posterior as P
from tests.views._posterior_helpers import FakeState, lane5_nodes, needs_node, run_page


def _levels() -> dict:
    return P.group_levels(lane5_nodes())


def _order(section: list[dict]) -> list[str]:
    return [f"{n['level']}:{n['key']}" for n in section]


def test_lane5_levels_land_in_their_sections():
    levels = _levels()
    assert list(levels)[:7] == list(P.SECTIONS)
    assert {n["level"] for n in levels["effort"]} == {"effort", "family_effort"}
    assert {n["level"] for n in levels["role"]} == {"role", "role_family"}
    assert {n["level"] for n in levels["topology"]} == {"topology", "position"}
    assert {n["level"] for n in levels["repo"]} == {"repo", "subtype"}
    assert [n["key"] for n in levels["feature"]] == ["size=s"]
    assert {"harness", "gate", "source"} <= set(levels)
    assert P.interaction_parts("family_effort") == ["family", "effort"]
    assert P.interaction_parts("role_family") == ["role", "family"]


def test_lane5_parent_ids_give_the_model_tree():
    assert _order(_levels()["model"]) == [
        "provider:anthropic", "family:opus", "model:claude-opus-5-5",
        "provider:openai", "family:astra", "model:gpt-6-astra"]
    assert _order(_levels()["repo"]) == ["repo:loopmath/loopmath", "subtype:loopmath/loopmath/views"]
    assert P.parent_ref("family:opus") == ("family", "opus")
    assert P.parent_ref("repo:feature/loopmath/loopmath") == ("repo", "loopmath/loopmath")
    assert P.parent_ref("opus") == (None, "opus")
    assert P.parent_ref(None) is None


def test_lane5_piece_effects_cover_every_level_of_the_setting():
    nodes = [n for section in _levels().values() for n in section]
    piece = {"id": "review", "kind": "piece", "role": "reviewer",
             "setting": {"harness": "codex", "model": "gpt-6-astra", "effort": "xhigh"}}
    found = {(n["level"], n["key"]) for n in P.piece_effects(piece, nodes, "implement_review#1")}
    assert found == {("provider", "openai"), ("family", "astra"), ("model", "gpt-6-astra"), ("effort", "xhigh"),
                     ("family_effort", "astra|xhigh"), ("role", "reviewer"), ("role_family", "reviewer|astra"),
                     ("harness", "codex"), ("position", "implement_review#1")}


def test_lane5_meta_fills_the_data_tab():
    meta = {"fit": "fit_x", "n_runs": {"prior": 1500, "user": 23},
            "runs_by_source": {"sweep": 1400, "rq1": 100, "user": 23},
            "dropped": {"duplicate run id": 2, "without benchmark": 5},
            "heads": {"cost": {"kind": "cost", "phi": {"family": 0.6, "model": 0.3},
                               "rows_by_source": {"sweep": 1549, "user": 61}},
                      "success": {"kind": "logit", "phi": {"family": 0.9}, "rows_by_source": {"sweep": 660}}},
            "scores": {"perf": {"scale": "linear", "better": "higher", "unit": "points"}},
            "seconds": 3.25, "code_version": "0.1.0+abc", "options": {"without": ["benchmark"]}}
    block = P.data_block(meta)
    assert block["rows"] == {"cost": {"sweep": 1549, "user": 61}, "success": {"sweep": 660}}
    assert block["scales"] == {"cost": {"family": 0.6, "model": 0.3}, "success": {"family": 0.9}}
    assert block["dropped"] == [{"reason": "duplicate run id", "n": 2}, {"reason": "without benchmark", "n": 5}]
    assert block["fit_time_s"] == 3.25 and block["code_version"] == "0.1.0+abc"
    assert block["without"] == ["benchmark"] and block["runs_by_source"]["rq1"] == 100


def test_head_info_follows_the_score_scale():
    meta = {"scores": {"ms": {"scale": "log", "better": "lower", "unit": "ms"},
                       "frac": {"scale": "fraction", "better": "higher"},
                       "perf": {"scale": "linear", "better": "higher", "unit": "points"}}}
    assert P.head_info("cost", meta) == {"kind": "multiplier"}
    assert P.head_info("tokens", meta) == {"kind": "multiplier"}
    assert P.head_info("success", meta) == {"kind": "pp"} and P.head_info("gate", meta) == {"kind": "pp"}
    assert P.head_info("score:ms", meta) == {"kind": "multiplier", "scale": "log", "better": "lower", "unit": "ms"}
    assert P.head_info("score:frac", meta)["kind"] == "pp"
    assert P.head_info("score:perf", meta) == {"kind": "shift", "scale": "linear", "better": "higher", "unit": "points"}

    class State:
        def score_info(self, name):
            return {"scale": "log", "unit": "s"}

    assert P.head_info("score:other", {}, State())["kind"] == "multiplier"


def test_build_view_ignores_lane5_head_objects(tmp_path):
    """FitState.heads maps names to HeadState objects; the view builds its own head info."""
    class Lane5State(FakeState):
        heads = {"cost": object()}

    home = tmp_path / "home"
    (home / "fits" / "fit_x").mkdir(parents=True)
    (home / "fits" / "fit_x" / "meta.json").write_text(json.dumps({
        "heads": {"cost": {"kind": "cost", "phi": {"family": 0.6}, "rows_by_source": {"sweep": 10}}}, "seconds": 1.5}))
    state = Lane5State(lane5_nodes())
    state.fit_id = "fit_x"
    obj = P.build_view(state, home=home, level="all", head=None, workflow=None, task_type=None, repo=None,
                       now="2026-09-23T17:00:00-07:00")
    assert obj["heads"]["cost"] == {"kind": "multiplier"} and obj["heads"]["gate"] == {"kind": "pp"}
    assert obj["data"]["rows"] == {"cost": {"sweep": 10}} and obj["data"]["fit_time_s"] == 1.5


@needs_node
def test_page_draws_lane5_names_for_people(tmp_path):
    data = {"schema": P.SCHEMA, "generated_at": "2026-09-23T17:00:00-07:00",
            "fit": {"id": "fit_x", "at": "2026-09-23T16:00:00-07:00", "n_runs": {"prior": 1500, "user": 23}},
            "levels": _levels(), "workflow": None, "workflows": []}
    out = run_page(P.render(data), tmp_path)
    cost = out["levels"]["cost"]
    assert "family x effort" in cost and "role x family" in cost and "family_effort" not in cost
    assert "the parent's estimate (provider openai), widened" in cost
    assert '<span class="lvl">version</span> claude-opus-5-5' in cost
    assert '<span class="lvl">feature:size</span> s<' in cost
    model_row = cost.split('<span class="lvl">version</span> claude-opus-5-5')[0].rsplit("padding-left:", 1)[1]
    assert model_row.startswith("44px")  # provider, family, version: depth 2
    assert '<span class="lvl">position</span> piece 2 of implement_review' in cost
    assert cost.count("* fewer than 5 runs.") == 1  # family x effort has a 3-run cell, role x family has none


def test_terminal_notes_follow_d21():
    """The terminal says what the page says: no runs here (the parent's estimate, widened), shared data, 1 run."""
    none = {"support": 0, "parent": "family:opus", "source_mix": {}}
    assert P._support_text(none) == "no runs here: the parent's estimate (family opus), widened"
    assert P._support_text({**none, "parent": "family_effort:opus|high"}) == \
        "no runs here: the parent's estimate (family x effort opus|high), widened"
    assert P._support_text({"support": 0, "parent": None}) == "no runs here: the prior for this level, widened"
    assert P._support_text({"support": 1, "source_mix": {"sweep": 1}}) == "1 run, shared data"
    assert P._support_text({"support": 12, "source_mix": {"sweep": 10, "user": 2}}) == "12 runs"


@needs_node
def test_bundle_fit_order_of_efforts_heads_and_source_rows(tmp_path):
    """Seen on the shipped bundle fit: efforts run low to max, Tokens sits by Cost, one head per row of the sources table."""
    from loopmath.types import NodeSummary
    from tests.views._posterior_helpers import iv

    nodes = [NodeSummary("family_effort", f"opus|{e}", "cost", iv(0, -0.1, 0.1), iv(1.0, 0.9, 1.1), 10, "family:opus",
                         {"sweep": 10}) for e in ("high", "low", "max", "medium", "xhigh", "turbo")]
    nodes += [NodeSummary("effort", "high", h, iv(0, -0.1, 0.1), iv(1.0, 0.9, 1.1), 10, None, {"sweep": 10})
              for h in ("score:perf", "gate", "tokens", "success", "cost")]
    data = {"schema": P.SCHEMA, "generated_at": "x", "fit": {"id": "fit_x", "at": "y", "n_runs": 10},
            "levels": P.group_levels(nodes), "workflow": None, "workflows": [],
            "data": {"rows": {"cost": {"sweep": 2063, "e0": 859}, "tokens": {"sweep": 2063}, "score:perf": {"rq1": 44}}}}
    out = run_page(P.render(data), tmp_path)
    assert out["heads"] == ["cost", "tokens", "success", "gate", "score:perf"]
    heat = out["levels"]["cost"].split('class="heat"')[1].split("</thead>")[0]
    assert re.findall(r"<th>([^<]*)</th>", heat) == ["low", "medium", "high", "xhigh", "max", "turbo"]
    table = out["data"].split("Rows per source and head")[1].split("</table>")[0]
    assert re.findall(r"<th[^>]*>([^<]*)</th>", table) == ["Head", "sweep", "e0", "rq1"]
    assert "<td>Tokens</td>" in table and ">2063<" in table


def test_page_effort_order_is_the_fit_order():
    from loopmath.fit_pricing import EFFORT_ORDER

    assert "var EFFORTS = ['" + "', '".join(EFFORT_ORDER) + "'];" in P.PAGE_JS


def test_loop_text_names_a_piece_that_retries_itself():
    assert P.loop_text({"from": "implement", "to": "review"}) == "review back to implement"
    assert P.loop_text({"from": "implement", "to": "implement"}) == "implement retries itself"


@needs_node
def test_long_sections_collapse_after_30_rows(tmp_path):
    from tests.views._posterior_helpers import iv
    from loopmath.types import NodeSummary

    tasks = [NodeSummary("task", f"t{i:02d}", "cost", iv(0, -1, 1), iv(1.0, 0.8, 1.2), 60 - i, "subtype:x/y", {"sweep": 1})
             for i in range(60)]
    data = {"schema": P.SCHEMA, "generated_at": "x", "fit": {"id": "fit_x", "at": "y", "n_runs": 60},
            "levels": P.group_levels(tasks), "workflow": None, "workflows": []}
    cost = run_page(P.render(data), tmp_path)["levels"]["cost"]
    assert cost.count('<details class="more"><summary>Show 30 more (fewer runs)</summary>') == 1
    shown, hidden = cost.split('<details class="more">')
    assert "t00" in shown and "t29" in shown and "t30" not in shown and "t59" in hidden
