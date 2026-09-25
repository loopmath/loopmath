"""views/plans.py and assets/plans.js: the planning page, `loopmath recommend --html` (lane 12, spec 06 section 2;
rebuilt in 0.2 by lane 2E as one decision, D119 Z3)."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

from loopmath.views import common, plans
from tests.views._plans_helpers import money, read_page

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v0_1"
VARIANTS = ("binary", "score")

def _fixture(variant: str) -> dict:
    return json.loads((FIXTURES / f"view-plans-{variant}.json").read_text(encoding="utf-8"))


def _builder():
    spec = importlib.util.spec_from_file_location("lm12_build_fixtures_plans", FIXTURES / "build_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", VARIANTS)
def test_fixture_round_trips_through_the_page(variant):
    data = _fixture(variant)
    page = plans.render(data)
    assert common.extract_data(page) == data
    assert not re.search(r"https?://", page)
    assert chr(0x2014) not in page  # no em dashes


@pytest.mark.parametrize("score_rule", [False, True])
def test_build_view_from_the_recommend_object_matches_the_fixture(score_rule):
    b = _builder()
    payload, cands, preds = b.recommend_payload(score_rule)
    fixture = _fixture("score" if score_rule else "binary")
    view = plans.build_view(payload, cands, now=None)
    assert view["schema"] == plans.SCHEMA
    assert view["candidates"] == fixture["candidates"]
    for key in ("task", "rule", "fit", "usual", "curve", "default_pick", "goal", "alternatives", "exploration", "pair", "message", "rec"):
        assert view[key] == fixture[key], key
    assert set(view["graphs"]) == set(fixture["graphs"])
    for cid, want in fixture["graphs"].items():
        got = view["graphs"][cid]
        for key in ("config", "label", "edges", "gates"):
            assert got[key] == want[key], (cid, key)
        assert [n["id"] for n in got["nodes"]] == [n["id"] for n in want["nodes"]]
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d$", view["generated_at"])
    json.dumps(view, allow_nan=False)  # plain JSON values only


def test_build_view_keeps_the_top_200_and_every_linked_candidate():
    data = _fixture("score")
    payload = {k: v for k, v in data.items() if k not in ("candidates", "graphs", "schema", "generated_at")}
    ranked = []
    for i in range(250):
        c = copy.deepcopy(data["candidates"][1])
        c["config"]["id"] = c["prediction"]["config"] = f"cfg_{i:012x}"
        ranked.append(c)
    ranked += data["candidates"]  # the linked ones rank last, past the cap
    view = plans.build_view(payload, ranked)
    ids = [c["config"]["id"] for c in view["candidates"]]
    assert ids[:200] == [f"cfg_{i:012x}" for i in range(200)]
    assert set(plans.referenced(payload)) <= set(ids)
    assert set(ids) <= set(view["graphs"])


def test_render_completes_a_bare_recommend_object():
    data = _fixture("binary")
    bare = {k: v for k, v in data.items() if k not in ("graphs", "schema")}
    back = common.extract_data(plans.render(bare))
    assert back["schema"] == plans.SCHEMA and set(back["graphs"]) == set(data["graphs"])


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_option_row_opens_its_graph_and_command(variant, tmp_path, probe):
    data = _fixture(variant)
    r = read_page(tmp_path, probe, data, f"plans-{variant}")
    ids = [row["cfg"] for row in r["rows"]]
    usual = data["usual"]["config"]["id"]
    best = data["exploration"]["best_value"]["candidate"]["config"]["id"]
    most = data["exploration"]["max_gain"]["candidate"]["config"]["id"]
    marked = {data["goal"]["config"], data["default_pick"]["config"], usual, best, most}
    assert marked <= set(ids) and len(ids) == len(set(ids))
    assert all(o["svg"] and o["cmd"] for o in r["opened"].values()), r["opened"]
    marks = {row["cfg"]: row["marks"] for row in r["rows"]}
    assert marks[data["goal"]["config"]] == ["recommended (80% row)"]
    assert "default pick" in marks[data["default_pick"]["config"]] and "your usual" in marks[usual]
    assert "best value to try" in marks[best] and "biggest gain to try" in marks[most]
    assert [row["cfg"] for row in r["rows"] if "pick" in row["cls"].split()] == [data["goal"]["config"]]
    # P4 (0.2.1): no numbered options in this fixture, so every button copies `workflow <config id>: <label>`
    assert all(o["cmd"].startswith(f"workflow {cfg}: ") for cfg, o in r["opened"].items())
    assert r["pickCmd"].startswith(f"workflow {data['goal']['config']}: ") and "loopmath run start" not in r["pickCmd"]
    # cheapest per accepted result first
    ells = [money(row["cells"][4]) for row in r["rows"]]
    assert ells == sorted(ells)
    assert r["graphBoxes"] >= 1 and r["forest"] >= 3 and r["pay"] == 2
    assert r["three"][:2] == [data["goal"]["config"], usual] and r["threeLabels"][:2] == ["recommended", "your usual"]
    head = "chance to reach 2400" if variant == "score" else "chance of an accepted result"
    assert r["heads"] == ["workflow", head, "cost per run", "expected rescue", "cost per accepted result", "runs behind it"]


def test_paused_same_as_and_none_picks(tmp_path, probe):
    data = _fixture("binary")
    best = data["exploration"]["best_value"]
    data["exploration"] = {"best_value": {"paused": "budget cap reached", "would_have_been": best},
                           "max_gain": {"same_as": "best_value"}}
    r = read_page(tmp_path, probe, data, "plans-paused")
    cid = best["candidate"]["config"]["id"]
    row = next(x for x in r["rows"] if x["cfg"] == cid)
    assert row["marks"] == ["best value to try (paused)", "biggest gain to try (paused)"]
    assert "Best value (paused)" in r["cards"][0] and "Paused: budget cap reached" in r["cards"][0]
    assert "Biggest gain (paused)" in r["cards"][1] and "Same candidate as best value" in r["cards"][1]
    assert r["pay"] == 1  # the paused pick is still drawn once; its twin is not drawn again

    data["exploration"] = {"best_value": {"none": "no candidate has positive gain"}, "max_gain": {"none": "no candidate has positive gain"}}
    r = read_page(tmp_path, probe, data, "plans-none")
    assert not any("to try" in m for x in r["rows"] for m in x["marks"])
    assert all("no candidate has positive gain" in card for card in r["cards"]) and r["pay"] == 0
    assert any(s.startswith("Worth trying something new") and "no candidate has positive gain" in s for s in r["summaries"])


def test_pick_cards_and_headers_read_plainly(tmp_path, probe):
    data = _fixture("score")
    data["exploration"]["best_value"]["payback_runs"] = 0.821
    data["exploration"]["max_gain"]["payback_runs"] = 4.3
    r = read_page(tmp_path, probe, data, "plans-words")
    best, most = r["cards"]
    assert "pays for itself afterabout 1 similar run" in best and "0.821" not in best  # whole runs, at least one
    assert "about 5 similar runs" in most  # I20 (0.2.1): ceil(price now / gain), as the message rounds it
    # The expected saving in dollars, as lane 6's message words it; its parts stay in the JSON
    assert "trying it onceexpected to save about $0.42 per future similar run" in best
    assert "expected to save about $0.95 per future similar run" in most
    assert all("gain if" not in c and "lower cost" not in c and "higher cost" not in c and "pp success" not in c
               and "-60" not in c for c in (best, most))
    assert data["exploration"]["best_value"]["gain_per_run"]["score"] == -60.0
    # P3a: renamed, beside the chance to reach the target, with one line on the difference
    assert "chance it beats the recommended pick" in best and "chance to beat the goal" not in best
    assert "chance of reaching heldout_perf >= 2400" in best and "Not the chance of reaching heldout_perf >= 2400" in best
    assert "ell" not in r["heads"] and r["heads"].count("cost per accepted result") == 1
    ids = [data["exploration"][k]["candidate"]["config"]["id"] for k in ("best_value", "max_gain")]
    assert [c.split(": ")[0] for c in r["betCmds"]] == [f"workflow {i}" for i in ids]  # P4 (0.2.1): Copy option


def test_two_hundred_candidates_stay_under_one_and_a_half_megabytes(tmp_path, probe):
    data = _fixture("score")
    payload = {k: v for k, v in data.items() if k not in ("candidates", "graphs", "schema", "generated_at")}
    big = max(data["candidates"], key=lambda c: len(json.dumps(c)))  # the three-piece workflow
    ranked = list(data["candidates"])
    for i in range(300):
        c = copy.deepcopy(big)
        c["config"]["id"] = c["prediction"]["config"] = f"cfg_{i:012x}"
        c["prediction"]["cost"]["usd"]["mean"] = 0.3 + 0.02 * i
        ranked.append(c)
    view = plans.build_view(payload, ranked)
    assert len(view["candidates"]) == 200 and len(view["graphs"]) >= 200
    text = plans.render(view)
    assert len(text.encode("utf-8")) < 1_500_000
    r = read_page(tmp_path, probe, view, "plans-200", limit=8)
    assert len(r["rows"]) >= 200 and all(o["svg"] and o["cmd"] for o in r["opened"].values())


def test_an_ell_above_its_interval_keeps_its_mean_with_the_note(tmp_path, probe):
    """The cost per accepted result keeps the mean and adds the note; the embedded JSON is unchanged."""
    data = _fixture("binary")
    goal = data["goal"]["config"]
    for c in data["candidates"]:
        if c["config"]["id"] == goal:
            c["prediction"]["ell"]["usd"] = {"mean": 3.0, "lo": 1.1, "hi": 2.64, "level": 0.8}
    page_data = copy.deepcopy(data)
    assert common.extract_data(plans.render(data)) == page_data
    r = read_page(tmp_path, probe, data, "plans-tail")
    row = next(x for x in r["rows"] if x["cfg"] == goal)
    assert row["tail"] == [False, False, False, False, True, False]
    assert row["cells"][4] == "$3.00$1.10 to $2.64 the average is pulled up by rare very large outcomes"
    assert "$1.10 to $2.64 the average is pulled up by rare very large outcomes" in r["nums"]
    assert sum(x["tail"][4] for x in r["rows"]) == 1


def test_a_bare_mean_gets_the_note_when_its_prediction_is_pulled_up(tmp_path, probe):
    """The rule is about the prediction, not what is printed: a pick card's price carries the note when its tokens
    are pulled up, though only dollars are shown; the JSON is unchanged."""
    note = "the average is pulled up by rare very large outcomes"
    data = _fixture("binary")
    data["exploration"]["best_value"]["price"]["tokens"] = {"mean": 900000.0, "lo": 50000.0, "hi": 800000.0, "level": 0.8}
    r = read_page(tmp_path, probe, data, "plans-bare")
    assert r["cardTail"] == [True, False] and note in r["cards"][0]
