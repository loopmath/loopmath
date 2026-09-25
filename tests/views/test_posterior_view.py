"""The posterior page (spec 06 section 3): the fixture, derived numbers, every section and head, every shape."""

from __future__ import annotations

import json
import math

import pytest

from loopmath.types import Task
from loopmath.views import posterior as P
from loopmath.workflows.format import catalog
from tests.graph_html_static_probe import network_findings
from tests.views._posterior_helpers import (
    FakeState, balance_errors, configure, fixture, fixture_nodes, hand_shapes, needs_node, run_page,
)


# ---------------------------------------------------------------- the object
def test_page_embeds_the_enriched_fixture_and_nothing_reaches_the_network():
    data = fixture()
    page = P.render(data)
    assert page.startswith("<!doctype html>") and page.count("<html") == 1
    assert P.extract_data(page) == P.enrich(data)
    assert network_findings(page) == []
    assert "<script src" not in page and "http://" not in page and "https://" not in page


def test_enrich_is_idempotent_and_keeps_every_fixture_field():
    data = fixture()
    once = P.enrich(data)
    assert P.enrich(once) == once
    for key, value in data.items():
        if key != "workflow":
            assert once[key] == value
    for key in ("config", "label", "graph", "per_piece"):
        assert once["workflow"][key] == data["workflow"][key]
    assert set(once["heads"]) == {"cost", "success", "gate", "score:heldout_perf"}
    assert once["heads"]["cost"]["kind"] == "multiplier" and once["heads"]["gate"]["kind"] == "pp"


def test_fixture_workflow_shares_gates_and_repair_loop():
    w = P.enrich(fixture())["workflow"]
    assert math.isclose(sum(w["shares"].values()), 1.0, abs_tol=1e-3)
    # Each piece's cost is its whole-run contribution (1.13 each), never multiplied by rounds
    assert w["shares"]["plan"] == pytest.approx(1 / 3, abs=1e-3)
    assert w["shares"]["review"] == pytest.approx(1 / 3, abs=1e-3)
    (loop,) = w["loops"]
    assert loop["from"] == "implement" and loop["to"] == "review" and loop["pieces"] == ["implement", "review"]
    assert loop["share"] == pytest.approx(w["shares"]["implement"] + w["shares"]["review"], abs=1e-3)
    assert loop["rounds"]["mean"] == 1.2
    (gate,) = w["gates"]
    assert gate["after"] == "review" and gate["on_fail"] == "implement"
    assert gate["pass"]["mean"] == 0.72 and gate["rounds"]["mean"] == 1.2


def test_piece_effects_follow_each_setting():
    data = fixture()
    data["levels"] = P.group_levels(fixture_nodes())
    obj = P.enrich(data)
    effects = obj["workflow"]["effects"]["pieces"]
    assert all(isinstance(r, str) for refs in effects.values() for r in refs)
    plan = set(effects["plan"])
    assert {"version:claude-opus-5-5", "family:opus", "provider:anthropic", "effort:xhigh",
            "family x effort:opus|xhigh"} <= plan
    assert "family x effort:opus|high" not in plan
    review = {(n["level"], n["key"]) for n in P.resolve_refs(effects["review"], obj["levels"])}
    assert {("version", "gpt-6-astra"), ("family", "astra"), ("provider", "openai"), ("role", "reviewer"),
            ("role x family", "reviewer|astra"), ("harness", "codex"), ("family x effort", "astra|xhigh")} <= review
    assert {n["head"] for n in P.resolve_refs(effects["review"], obj["levels"], "gate")} == {"gate"}


def test_repair_loops_come_from_control_on_acyclic_edges():
    wf = hand_shapes()["implement_test_review"]
    config = configure(wf)
    entry = P.enrich_workflow({"config": config.id, "label": config.label(),
                               "graph": P.workflow_graph(config, FakeState().predict(_task(), config))})
    loops = {lp["gate"]: lp for lp in entry["loops"]}
    assert loops["g_tests"]["pieces"] == ["implement", "test"]
    assert loops["g_review"]["pieces"] == ["implement", "test", "review"]
    stop = P.enrich_workflow({"graph": P.workflow_graph(configure(hand_shapes()["implement_test"]), None)})
    assert stop["loops"] == [] and stop["gates"][0]["on_fail"] is None


def _task() -> Task:
    return Task(id="t", type="feature", repo="o/r")


def test_nan_and_infinity_become_null():
    data = fixture()
    data["levels"]["model"][0]["display"]["hi"] = float("inf")
    data["data"]["fit_time_s"] = float("nan")
    out = P.enrich(data)
    assert out["levels"]["model"][0]["display"]["hi"] is None and out["data"]["fit_time_s"] is None
    json.loads(P.embed_json(out))


def test_embed_cannot_close_the_script_early():
    data = fixture()
    data["levels"]["model"][0]["key"] = "</script><b>x</b>&\u2028"
    page = P.render(data)
    assert page.count("</script>") == 2
    assert P.extract_data(page)["levels"]["model"][0]["key"] == "</script><b>x</b>&\u2028"


# ---------------------------------------------------------------- levels
def test_sections_every_level_even_empty_and_extras_after():
    levels = P.group_levels(fixture_nodes())
    assert list(levels)[:7] == list(P.SECTIONS)
    assert "harness" in levels and list(levels).index("harness") >= 7
    assert {n["level"] for n in levels["effort"]} == {"effort", "family x effort"}
    assert {n["level"] for n in levels["role"]} == {"role", "role x family"}
    empty = P.group_levels([])
    assert list(empty) == list(P.SECTIONS) and all(v == [] for v in empty.values())


def test_section_of():
    assert P.section_of("provider") == P.section_of("version") == "model"
    assert P.section_of("feature:size") == "feature"
    assert P.section_of("family_x_effort") == "effort" and P.section_of("role x family") == "role"
    assert P.section_of("source") == "source" and P.section_of("position") == "topology"


def test_model_tree_parents_before_children():
    keys = [n["key"] for n in P.group_levels(fixture_nodes())["model"] if n["head"] == "cost"]
    assert keys.index("anthropic") < keys.index("opus") < keys.index("claude-opus-5-5")
    assert keys.index("openai") < keys.index("astra") < keys.index("gpt-6-astra")
    assert keys.index("fable") > keys.index("anthropic")


def test_filter_levels_by_section_and_head():
    levels = P.group_levels(fixture_nodes())
    only = P.filter_levels(levels, level="model", head="cost")
    assert list(only) == ["model"] and all(n["head"] == "cost" for n in only["model"])
    assert P.filter_levels(levels, head="gate")["effort"] == []


def test_display_for_people():
    heads = {"score:heldout_perf": {"kind": "shift", "unit": "perf"}}
    node = {"head": "cost", "display": {"mean": 1.4, "lo": 1.1, "hi": 1.8}}
    assert P.fmt_display(node) == "x1.40 (1.10 to 1.80)"
    assert P.fmt_display({"head": "success", "display": {"mean": 6, "lo": -1, "hi": 12}}) == "+6 pp (-1 to +12)"
    assert P.fmt_display({"head": "score:heldout_perf", "display": {"mean": 60, "lo": -40, "hi": 160}},
                         heads) == "+60 perf (-40 to +160)"


def test_summary_lines_stay_within_25():
    data = P.enrich(fixture())
    many = [dict(data["levels"]["model"][0], key=f"m{i}") for i in range(60)]
    data["levels"]["model"] = many
    lines = P.summary_lines(data)
    assert len(lines) <= 25
    assert lines[-1].endswith("use --level, --head, --json or --html")
    assert lines[0].startswith("Current estimates")


# ---------------------------------------------------------------- the page under node
@needs_node
def test_page_draws_every_head_section_and_block(tmp_path):
    data = fixture()
    data["levels"] = P.group_levels(fixture_nodes())
    data["levels"]["score:heldout_perf"] = fixture()["levels"]["score:heldout_perf"]
    obj = P.enrich(data)
    out = run_page(P.render(data), tmp_path)
    assert out["initialHead"] == "cost" and out["heads"][:3] == ["cost", "success", "gate"]
    assert out["drawn"] == {"levels": True, "graph": True, "data": True}  # every block drawn into its host
    for head, html in out["levels"].items():
        assert balance_errors(html) == [], head
        for name in obj["levels"]:
            assert f'data-section="{name}"' in html, (head, name)
    cost = out["levels"]["cost"]
    assert "x1.28 (1.05 to 1.57)" in cost and "from shared data" in cost
    assert "No runs here: the parent's estimate (openai), widened." in cost
    assert 'class="heat"' in cost and "opus" in cost
    assert 'class="heat"' in out["levels"]["gate"]
    assert "No cost estimates at this level" in cost
    assert "+60 (-40 to +160)" in out["levels"]["score:heldout_perf"]
    assert out["afterHeadClick"] == "all"
    assert out["tip"] == "line one\nline two" and out["tipHidden"] is False and out["tipLeft"] == "820px"
    assert "from 1,173 runs, 23 of them yours" in out["lede"]


@needs_node
def test_js_and_python_format_every_node_alike(tmp_path):
    data = fixture()
    data["levels"] = P.group_levels(fixture_nodes())
    obj = P.enrich(data)
    out = run_page(P.render(data), tmp_path)
    for nodes in obj["levels"].values():
        for n in nodes:
            assert out["display"][f"{n['level']}|{n['key']}|{n['head']}"] == P.fmt_display(n, obj["heads"])


@needs_node
def test_graph_and_data_blocks_on_the_fixture(tmp_path):
    out = run_page(P.render(fixture()), tmp_path, "#graph")
    assert out["drawn"]["graph"]
    (graph,) = out["graph"]
    assert balance_errors(graph) == []
    for text in ("plan_implement_review", "$1.13 ($0.73 to $1.76) per run", "203k tokens per run, 33% of cost",
                 "gate g_review: pass 72%", "repair, review back to implement: 1.2 rounds, reruns pieces with 67% of cost",
                 "Repair loops", "Gates", "Pieces", "Cost per run"):
        assert text in graph, text
    data = out["data"]
    assert balance_errors(data) == []
    for text in ("Rows per source and head", "sweep", "asserted cost", "0.62", "loopmath fit --without benchmark"):
        assert text in data, text


@needs_node
def test_graph_words_for_one_round_no_support_and_small_amounts(tmp_path):
    data = fixture()
    w = data["workflow"]
    w["gates"] = []
    w["per_piece"]["review"]["rounds"] = {"mean": 1.0, "lo": 1.0, "hi": 1.0, "level": 0.8}
    w["per_piece"]["review"]["cost"]["usd"] = {"mean": 0.007, "lo": 0.004, "hi": 0.012, "level": 0.8}
    w["prediction"] = {"support": 0}
    graph = run_page(P.render(data), tmp_path, "#graph")["graph"][0]
    assert "gate g_review: pass 72%, 1 round<" in graph and "1 rounds" not in graph
    assert "no runs yet in any group close to this configuration" in graph and "; 0 runs" not in graph
    assert "$0.007 ($0.004 to $0.01)" in graph
    assert "203k (132k to 317k)" in graph and "k tokens (" not in graph


def _d60_workflow() -> dict:
    """The reviewer's case through lane 5's composer: plan costs 1 once; implement costs 1 per round,
    passes its gate half the time and runs at most 2 rounds, so 1.5 over the run."""
    one = {"mean": 1.0, "lo": 1.0, "hi": 1.0, "level": 0.8}
    def money(x):
        return {"usd": {**one, "mean": x, "lo": x, "hi": x}, "tokens": {**one, "mean": 1000 * x, "lo": 1000 * x, "hi": 1000 * x}}
    graph = {"config": "cfg_d60d60d60d60", "nodes": [
                 {"id": "issue", "kind": "artifact"}, {"id": "plan", "kind": "piece", "role": "planner"},
                 {"id": "plan_doc", "kind": "artifact"}, {"id": "implement", "kind": "piece", "role": "implementer"},
                 {"id": "diff", "kind": "artifact"}],
             "edges": [["issue", "plan"], ["plan", "plan_doc"], ["plan_doc", "implement"], ["implement", "diff"]],
             "gates": [{"id": "g", "after": "implement", "rule": "tests_pass", "on_fail": "implement"}], "budget_rounds": 2}
    per_piece = {"plan": {"piece": "plan", "cost": money(1.0), "cost_per_round": money(1.0), "gate_pass": None, "rounds": one},
                 "implement": {"piece": "implement", "cost": money(1.5), "cost_per_round": money(1.0),
                               "gate_pass": {**one, "mean": 0.5}, "rounds": {**one, "mean": 1.5, "hi": 2.0}}}
    return {"config": "cfg_d60d60d60d60", "label": "plan_implement: d60", "graph": graph, "per_piece": per_piece}


def test_d60_shares_come_from_run_totals():
    w = P.enrich_workflow(_d60_workflow())
    assert w["shares"] == {"plan": 0.4, "implement": 0.6}
    (loop,) = w["loops"]
    assert loop["pieces"] == ["implement"] and loop["share"] == 0.6
    data = {"fit": {"id": "fit_x", "at": "t", "n_runs": 10}, "levels": {}, "workflow": w}
    lines = [ln for ln in P.summary_lines(data) if ln.startswith("  implement:")]
    assert lines == ["  implement: $1.00 ($1.00 to $1.00) per round, $1.50 ($1.50 to $1.50) per run, 2k tokens per run,"
                     " expected rounds 1.5, 60% of cost"]


@needs_node
def test_d60_labels_per_round_and_per_run(tmp_path):
    data = fixture()
    data["workflow"] = _d60_workflow()
    graph = run_page(P.render(data), tmp_path, "#graph")["graph"][0]
    assert "$1.00 ($1.00 to $1.00) per round" in graph and "$1.50 ($1.50 to $1.50) per round" not in graph
    assert "2k tokens per run, 60% of cost" in graph and "1k tokens per run, 40% of cost" in graph
    row = graph.split("<td>implement ")[1].split("</tr>")[0]
    assert '<td class="num">$1.00 ($1.00 to $1.00)</td><td class="num">$1.50 ($1.50 to $1.50)</td>' in row
    assert "implement retries itself: 1.5 rounds, reruns pieces with 60% of cost" in graph


@needs_node
def test_heat_colors_follow_the_score_better_direction(tmp_path):
    """A score that is better higher on a log scale, and one better lower on a fraction scale (review of 99548fc)."""
    blue, red = "rgb(42,120,214)", "rgb(227,73,72)"
    cases = [("score:log_higher", "multiplier", "higher", 2.0, blue), ("score:log_lower", "multiplier", "lower", 2.0, red),
             ("score:frac_lower", "pp", "lower", -15.0, blue), ("score:frac_higher", "pp", "higher", -15.0, red),
             ("cost", "multiplier", None, 2.0, red), ("success", "pp", None, -15.0, red)]
    for head, kind, better, mean, want in cases:
        heads = {head: {"kind": kind, **({"better": better} if better else {})}}
        data = {"schema": P.SCHEMA, "fit": {"id": "fit_x", "at": "t"}, "heads": heads, "levels": {"effort": [
            {"level": "family_effort", "key": "opus|high", "head": head, "display": {"mean": mean, "lo": mean, "hi": mean},
             "effect": {"mean": 0, "lo": 0, "hi": 0}, "support": 10, "source_mix": {"user": 10}}]}}
        drawn = run_page(P.render(data), tmp_path)["levels"][head]
        assert f'<td style="background:{want};' in drawn, (head, better)
    heads = {"score:plain": {"kind": "shift"}}
    data["heads"] = heads
    data["levels"]["effort"][0]["head"] = "score:plain"
    drawn = run_page(P.render(data), tmp_path)["levels"]["score:plain"]
    assert '<td style="background:rgb(240,239,236);' in drawn and "no better direction" in drawn


@needs_node
def test_empty_view_draws_its_messages(tmp_path):
    data = {"schema": P.SCHEMA, "generated_at": "2026-09-23T16:00:00-07:00",
            "fit": {"id": "fit_x", "at": "2026-09-23T16:00:00-07:00", "n_runs": None},
            "levels": P.group_levels([]), "workflow": None}
    out = run_page(P.render(data), tmp_path)
    assert "No configuration to draw" in out["graph"][0]
    assert out["levels"]["all"].count("No estimates at this level yet.") == len(P.SECTIONS)
    assert "Not recorded by this fit." in out["data"]


@needs_node
@pytest.mark.parametrize("name", sorted(catalog()))
def test_graph_tab_draws_every_catalog_shape(name, tmp_path):
    workflow = catalog()[name]
    config = configure(workflow)
    state = FakeState()
    entry = P._entry(state, _task(), config, "argument")
    data = {"schema": P.SCHEMA, "generated_at": "2026-09-23T16:00:00-07:00",
            "fit": {"id": state.fit_id, "at": state.created_at, "n_runs": 3},
            "levels": P.group_levels(state.node_summary()), "workflows": [entry], "workflow": entry}
    obj = P.enrich(data)
    w = obj["workflow"]
    assert math.isclose(sum(w["shares"].values()), 1.0, abs_tol=1e-3)
    assert len(w["loops"]) == sum(1 for g in workflow.control.gates if g.on_fail)
    out = run_page(P.render(data), tmp_path, "#graph")
    (graph,) = out["graph"]
    assert balance_errors(graph) == []
    (layout,) = out["layouts"]
    ids = [p.id for p in workflow.pieces] + list(workflow.artifacts)
    assert set(layout["pos"]) == set(ids) and layout["back"] == []
    for a, b in workflow.edges:
        assert layout["pos"][a]["x"] < layout["pos"][b]["x"], (a, b)
    for piece in workflow.pieces:
        if piece.width > 1:  # I12: one box per worker, each with its setting
            assert all(f"{piece.id} {k} of {piece.width} <tspan" in graph for k in range(1, piece.width + 1))
            assert f"{piece.id} <tspan" not in graph
        else:
            assert f"{piece.id} <tspan" in graph


@needs_node
def test_workflow_picker_switches_configurations(tmp_path):
    state = FakeState()
    entries = [P._entry(state, _task(), configure(w), "recorded") for w in hand_shapes().values()]
    data = {"schema": P.SCHEMA, "generated_at": "2026-09-23T16:00:00-07:00",
            "fit": {"id": state.fit_id, "at": state.created_at, "n_runs": 3},
            "levels": P.group_levels(state.node_summary()), "workflows": entries, "workflow": entries[0]}
    out = run_page(P.render(data), tmp_path)
    assert len(out["graph"]) == len(entries) and out["afterPick"] == len(entries) - 1
    assert 'id="wfpick"' in out["graph"][0]
    assert all(e["config"] in g for e, g in zip(entries, out["graph"]))
