"""Scaffold contracts for 0.1.0 (design/0.1/02-commands.md, 03-interfaces.md).

These tests pin what every lane builds against: the command registry, the
shared types' JSON round trip, configuration ids, and the view fixtures.
Lanes replace stubs; they should not need to change these tests.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib

import pytest

from loopmath import cli, cli_registry
from loopmath.taskmodel import FEATURES, TASK_TYPES, group_chain, normalize_features
from loopmath.types import (
    AcceptanceRule, Configuration, Interval, Prediction, ScoreTarget, Setting, Signal, Task,
)
from loopmath.workflows.ids import config_id

FIX = pathlib.Path(__file__).parent / "fixtures" / "v0_1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopmath")
    sub = parser.add_subparsers(dest="command", required=True)
    cli_registry.register(sub)
    return parser


def test_every_handler_target_imports():
    for target in cli_registry.targets(_parser()):
        module_name, fn_name = target.split(":")
        module = importlib.import_module(module_name)
        assert callable(getattr(module, fn_name)), target


@pytest.mark.parametrize("argv", [
    ["task-types", "--json"],
    ["workflows", "list"],
    ["recommend", "--type", "feature", "--repo", "r", "--target", "runtime_s<=200", "--json"],
    ["plan", "--backlog", "t.jsonl", "--budget-usd", "10"],
    ["run", "start", "--type", "bug_fix", "--repo", "r", "--config", "cfg_000000000000", "--source", "usual", "--new-slate"],
    ["run", "attempt", "--run", "run_x", "--piece", "implement", "--harness", "codex", "--model", "gpt-6-sol", "--session", "self"],
    ["run", "finish", "--run", "run_x"],
    ["outcome", "--run", "run_x", "--signal", "quality=", "--kind", "score", "--scale", "fraction"],
    ["outcome", "--slate", "slt_x", "--prefer", "tie", "--judge", "referee", "--blinded"],
    ["fit", "--background"],
    ["onboard", "--dry-run"],
    ["share", "--preview"],
    ["runs", "--html"],
    ["posterior", "--level", "model", "--head", "score:heldout_perf"],
    ["status"],
    ["report"],
    ["doctor"],
    ["skill", "install", "--target", "both"],
    ["ocp", "validate", "x.json"],
    ["prior", "show"],
])
def test_registered_commands_parse(argv):
    args = _parser().parse_args(argv)
    assert callable(args.func)


def test_research_verbs_moved():
    pytest.importorskip("loopmath.fit")
    with pytest.raises(SystemExit):
        cli.main(["research", "--help"])


def test_task_types_json(capsys):
    assert cli.main(["task-types", "--json"]) == 0
    obj = json.loads(capsys.readouterr().out)
    assert next(iter(obj)) == "schema" and obj["schema"] == "loopmath.task-types/1"
    assert [t["id"] for t in obj["types"]] == list(TASK_TYPES)
    assert {f["key"] for f in obj["features"]} == set(FEATURES)


def test_features_and_chain():
    assert normalize_features({"Size": "250", "touches": "3", "color": "Blue", "has_tests": "maybe"}) == {
        "size": "m", "touches": "few", "extra:color": "blue", "has_tests": "unknown"}
    t = Task(id="t1", type="feature", repo="o/r", subtype="cli", org="acme")
    assert group_chain(t) == [("org", "acme"), ("type", "feature"), ("repo", "o/r"), ("subtype", "cli"), ("task", "t1")]


def test_types_round_trip_and_extra():
    rule = AcceptanceRule(name="perf>=2400", definition="d", requires=(), score=ScoreTarget("heldout_perf", 2400, "higher"))
    assert AcceptanceRule.from_dict(json.loads(json.dumps(rule.to_dict()))) == rule
    sig = Signal.from_dict({"id": "sig_1", "run": "run_1", "kind": "score", "name": "quality", "value": None,
                            "scale": "fraction", "future_field": 7})
    assert sig.value is None and sig.extra == {"future_field": 7}
    assert sig.to_dict()["future_field"] == 7
    assert Interval.from_dict({"mean": 1, "lo": 0, "hi": 2}).level == 0.8


def test_fixture_views_load_and_round_trip():
    for name, schema in [("view-runs.json", "loopmath.view.runs/1"), ("view-plans-binary.json", "loopmath.view.plans/1"),
                         ("view-plans-score.json", "loopmath.view.plans/1"), ("view-posterior.json", "loopmath.view.posterior/1"),
                         ("recommend-binary.json", "loopmath.recommend/1"), ("recommend-score.json", "loopmath.recommend/1")]:
        obj = json.loads((FIX / name).read_text())
        assert next(iter(obj)) == "schema" and obj["schema"] == schema, name
    plans = json.loads((FIX / "view-plans-score.json").read_text())
    usual = Prediction.from_dict(plans["usual"]["prediction"])
    assert usual.success_from == "score_head" and usual.scores["heldout_perf"].p_reach is not None
    for c in plans["candidates"]:
        cfg = Configuration.from_dict(c["config"])
        assert config_id(cfg.workflow, cfg.settings) == cfg.id


def test_config_id_is_stable_and_ignores_titles():
    from loopmath.types import Piece, Workflow

    wf = Workflow(id="solo", version=1, title="One agent", pieces=(Piece("implement", "implementer"),),
                  artifacts=("patch",), edges=(("implement", "patch"),))
    wf2 = Workflow(id="solo", version=1, title="Renamed", pieces=wf.pieces, artifacts=wf.artifacts, edges=wf.edges)
    s = {"implement": Setting("claude-code", "claude-opus-5-5", "high")}
    assert config_id(wf, s) == config_id(wf2, s)
    assert config_id(wf, s).startswith("cfg_") and len(config_id(wf, s)) == 16
    assert config_id(wf, {"implement": Setting("claude-code", "claude-opus-5-5", "xhigh")}) != config_id(wf, s)
