"""The 0.2 planning page (lane 2E, D119): one decision first, lane 2A's strategy sentence verbatim (Z2 A), copy
commands that the real `run start` parser accepts, and a page that fits a phone."""

from __future__ import annotations

import argparse
import copy
import shlex
from pathlib import Path

import pytest

from loopmath.views import common, plans
from tests.views._plans_helpers import _score, money, read_page, recommend2

ASSETS = Path(plans.__file__).resolve().parent / "assets"
# The sentence 2A writes (DESIGN 10.3), with characters that must reach the page as text, not markup.
TEXT = ("Try implement_review: claude-fable-5-1/medium <first>; if it misses (about 3 in 10 tasks), your best recorded "
        "workflow rescues it; expected $11.21 per accepted result, against $24.01 with your best recorded workflow alone & more.")

ORDER = r"""(() => {
  const s = document.getElementById('strategy'), h1 = document.querySelector('h1'), nums = document.getElementById('nums');
  const after = (a, b) => !!(a && b && (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING));
  return { text: s ? s.textContent : null, html: s ? s.innerHTML : null, afterH1: after(h1, s), beforeNums: after(s, nums),
           count: document.getElementById('app').textContent.split('if it misses').length - 1 };
})()"""


def _strategy(data: dict) -> dict:
    goal = data["goal"]["config"]
    return {"kind": "try_then_rescue", "text": TEXT, "pick": {"config": goal, "label": "implement_review"},
            "reference": {"config": data["reference"]["config"]["id"], "kind": "best_recorded", "name": "your best recorded workflow"},
            "chance": {"pick": 0.69, "reference": 0.70}, "misses": {"n": 3, "of": 10, "share": 0.31},
            "rescue": {"kind": "redo_usual", "usd": 24.01}, "run_cost_usd": 3.81, "expected_rescue_usd": 7.40,
            "cost_per_accepted_usd": 11.21, "reference_cost_per_accepted_usd": 24.01}


def _run_start_parser() -> argparse.ArgumentParser:
    from loopmath import cli_registry

    parser = argparse.ArgumentParser(prog="loopmath")
    cli_registry.register(parser.add_subparsers(dest="command", required=True))
    return parser


def _parse(parser, line: str) -> argparse.Namespace:
    words = shlex.split(line)
    assert words[0] == "loopmath"
    return parser.parse_args(words[1:])


@pytest.mark.parametrize("where", ["goal", "choice"])
def test_the_strategy_sentence_is_shown_verbatim_under_the_heading(where, tmp_path, probe):
    data = recommend2(_score())
    strategy = _strategy(data)
    if where == "goal":
        data["goal"]["strategy"] = strategy
    data["choices"] = [{"key": "goal", "config": data["goal"]["config"], "strategy": copy.deepcopy(strategy)}]
    page = tmp_path / f"plans-strategy-{where}.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, ORDER)
    assert got["exceptions"] == [] and got["console_errors"] == []
    r = got["result"]
    assert r["text"] == TEXT and "&lt;first&gt;" in r["html"] and "&amp; more" in r["html"]
    assert r["afterH1"] and r["beforeNums"] and r["count"] == 1  # once, where the page states the pick


def test_no_strategy_sentence_when_it_is_null(tmp_path, probe):
    data = recommend2(_score())
    data["goal"]["strategy"] = None
    data["choices"] = [{"key": "goal", "config": data["goal"]["config"], "strategy": None}]
    page = tmp_path / "plans-no-strategy.html"
    page.write_text(plans.render(data), encoding="utf-8")
    got = probe(page, ORDER)
    assert got["exceptions"] == [] and got["result"]["text"] is None and got["result"]["count"] == 0


def test_every_copy_command_parses_with_run_start(tmp_path, probe):
    data = recommend2(_score())
    data["task"]["title"] = "Fix the 'quoted' thing"
    data["task"]["features"] = {"size": "large files"}
    data["pair"] = {"members": [data["goal"]["config"], data["exploration"]["best_value"]["candidate"]["config"]["id"]],
                    "explore_pick": "best_value", "instructions": ["run start --new-slate, then run start --slate SLT"]}
    r = read_page(tmp_path, probe, data, "plans-commands")
    parser = _run_start_parser()
    lines = [r["pickCmd"], *r["betCmds"], *(o["cmd"] for o in r["opened"].values()), *r["pairCmd"].split("\n")]
    assert len(lines) >= 8
    for line in lines:
        args = _parse(parser, line)
        assert (args.command, args.run_command) == ("run", "start") and args.rec == data["rec"]
        assert args.title == "Fix the 'quoted' thing" and args.feature == ["size=large files"] and args.task_type == "feature"
    first, second = (_parse(parser, x) for x in r["pairCmd"].split("\n"))
    assert first.config == data["goal"]["config"] and first.new_slate and first.source == "alternative"
    assert second.config == data["pair"]["members"][1] and second.slate == "SLT" and second.source == "exploration"
    assert _parse(parser, r["pickCmd"]).config == data["goal"]["config"]


def test_the_options_a_stored_recommendation_names_start_with_choice(tmp_path, probe):
    """Lane 2D's `run start --rec REC --choice KEY` for the goal, the reference, the cheapest run and the pair;
    any other option keeps the full `--config` command."""
    data = recommend2(_score())
    goal, ref = data["goal"]["config"], data["reference"]["config"]["id"]
    other = next(c["config"]["id"] for c in data["candidates"] if c["config"]["id"] not in (goal, ref))
    explore = data["exploration"]["best_value"]["candidate"]["config"]["id"]
    data["pair"] = {"members": [goal, explore], "explore_pick": "best_value", "instructions": []}
    data["choices"] = [{"key": "goal", "config": goal, "members": [goal]},
                       {"key": "pair", "config": goal, "members": [goal, explore]},
                       {"key": "reference", "config": ref, "members": [ref]},
                       {"key": "cheapest_run", "config": other, "members": [other]}]
    r = read_page(tmp_path, probe, data, "plans-choice")
    parser = _run_start_parser()
    want = {goal: "goal", ref: "reference", other: "cheapest_run"}
    for cfg, key in [(None, "goal"), *want.items()]:
        line = r["pickCmd"] if cfg is None else r["opened"][cfg]["cmd"]
        args = _parse(parser, line)
        assert (args.choice, args.rec, args.config, args.source) == (key, data["rec"], None, None), line
    pair = _parse(parser, r["pairCmd"])
    assert "\n" not in r["pairCmd"] and (pair.choice, pair.rec, pair.new_slate, pair.slate) == ("pair", data["rec"], False, None)
    rest = [c for c in r["opened"] if c not in want]
    assert rest and all(_parse(parser, r["opened"][c]["cmd"]).config == c for c in rest)


def test_search_origins_read_as_words(tmp_path, probe):
    """Lane 2A's search adds the origins front, thompson and polish; the page names them in words."""
    data = recommend2(_score())
    rows = data["candidates"] + data["alternatives"]
    for c, origin in zip(rows, ["front", "thompson", "polish"]):
        c["origin"] = origin
    r = read_page(tmp_path, probe, data, "plans-origins")
    subs = " ".join(row["cells"][0] for row in r["rows"])
    for word in ("found by the search", "found in a search draw", "near the pick"):
        assert word in subs and word in r["body"]
    assert not any(w in subs for w in ("front", "thompson", "polish"))


@pytest.mark.parametrize("horizon,features,want", [
    ({"seconds": 5400.0, "from": "given", "used": True}, {"size": "large", "horizon_s": "5400"}, "5400"),
    ({"seconds": None, "from": "given", "used": False}, {"size": "large", "horizon_s": "none"}, "none"),
    ({"seconds": 7200.0, "from": "repo", "used": True}, {"size": "large"}, "7200"),
    ({"seconds": None, "from": "none", "used": False}, {"size": "large"}, None),
    (None, {"size": "large", "horizon_s": "1800"}, "1800"),
])
def test_the_copy_command_carries_the_horizon_the_prediction_used(horizon, features, want, tmp_path, probe):
    """Lane 2C's horizon goes as `--horizon`, never as a feature: the one given, or the one the fit filled in, so the
    run is recorded under the time budget it was priced for; none when the prediction had none."""
    parser = _run_start_parser()
    data = recommend2(_score())
    data["task"]["features"] = features
    if horizon is None:
        data["task"].pop("horizon", None)
    else:
        data["task"]["horizon"] = horizon
    r = read_page(tmp_path, probe, data, f"plans-horizon-{want}")
    for line in [r["pickCmd"], *(o["cmd"] for o in r["opened"].values())]:
        args = _parse(parser, line)
        assert args.horizon == want and args.feature == ["size=large"], line


def test_the_arithmetic_and_the_order_hold_on_the_page(tmp_path, probe):
    data = recommend2(_score())
    r = read_page(tmp_path, probe, data, "plans-arith")
    ells = [money(row["cells"][4]) for row in r["rows"]]
    assert ells == sorted(ells) and len(ells) == len({c["config"]["id"] for c in data["candidates"] + data["alternatives"]} | {data["reference"]["config"]["id"]})
    for row in r["rows"]:
        run, rescue, ell = money(row["cells"][2]), money(row["cells"][3]), money(row["cells"][4])
        assert abs(run + rescue - ell) <= 0.015, row  # run cost + expected rescue = cost per accepted result, to the cent
    assert r["summaries"][0].startswith("The arithmetic") and "rescue =" in r["summaries"][0]


def test_the_page_fits_a_phone(tmp_path, probe):
    data = recommend2(_score())
    data["goal"]["strategy"] = _strategy(data)
    r = read_page(tmp_path, probe, data, "plans-phone", limit=3, width=390)
    assert r["width"] <= 390 and r["strategy"] == TEXT and r["three"]


def test_the_page_is_offline_and_has_no_em_dashes():
    page = plans.render(recommend2(_score()))
    assert "https:" not in page and "http:" not in page.replace("'ht' + 'tp:'", "")
    for name in ("viz.js", "viz.css", "plans.js", "plans.css"):
        assert chr(0x2014) not in (ASSETS / name).read_text(encoding="utf-8"), name
    assert common.extract_data(page)["schema"] == plans.SCHEMA
