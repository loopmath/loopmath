"""The `loopmath eval` group: every command and option parses, and a command that is not built says so."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath import cli, cli_registry
from loopmath.eval import commands
from loopmath.eval import contracts as C

ROOT = Path(__file__).resolve().parents[2]
FULL = {
    "scores import": ["eval", "scores", "import", "--ocp", "a.ocp.json", "b.ocp.json", "--rows", "rows.jsonl",
                      "--csv", "labels.csv", "--pairs-csv", "pairs.csv", "--items", "items.jsonl", "--labels",
                      "plan.json", "--rubric", "rubric.json", "--pair-component", "overall", "--allow-simulated",
                      "--out", "study", "--json"],
    "scores check": ["eval", "scores", "check", "study", "--rubric", "rubric.json", "--judges", "judges.json",
                     "--require-assignment", "--json"],
    "scores add": ["eval", "scores", "add", "study", "--rubric", "rubric.json", "--run", "r1", "--step", "draft",
                   "--component", "depth", "--judge", "reviewer", "--value", "7", "--of", "9", "--call", "c1",
                   "--observed-at", "2026-01-15T09:30:00-08:00", "--replaces", "s1", "--assignment", "a1", "--id",
                   "s2", "--json"],
    "scores export": ["eval", "scores", "export", "study", "--run", "r1", "--run", "r2", "--json"],
    "simulate": ["eval", "simulate", "halo", "--out", "study", "--seed", "3", "--json"],
    "fit": ["eval", "fit", "study", "--rubric", "rubric.json", "--judges", "judges.json", "--seed", "1",
            "--structure", "factor:2", "--halo", "full", "--cuts", "free", "--terms", "family=off", "--terms",
            "judge_input=auto", "--draws", "500", "--chains", "2", "--keep-draws", "400", "--components", "depth",
            "clarity", "--check-only", "--allow-simulated", "--out", "study/fit.json", "--json"],
    "judges": ["eval", "judges", "study/fit.json", "--target-error", "0.05", "--budget-usd", "0.2",
               "--allow-unconverged", "--study", "study", "--strict", "--json"],
    "rubric check": ["eval", "rubric", "check", "rubric.json", "--judges", "judges.json", "--value", "value.json",
                     "--items", "items.jsonl", "--fit", "study/fit.json", "--allow-unconverged", "--json"],
    "value": ["eval", "value", "study/fit.json", "--value", "value.json", "--reference", "baseline", "--check",
              "design", "--labels", "labels-001.json", "labels-002.json", "--seed", "2", "--allow-unconverged",
              "--study", "study", "--strict", "--json"],
    "steps": ["eval", "steps", "study/fit.json", "--value", "value.json", "--reference", "baseline", "--pool",
              "--runs", "study/runs.jsonl", "--seed", "2", "--allow-unconverged", "--study", "study", "--strict",
              "--json"],
    "labels next": ["eval", "labels", "next", "study/fit.json", "--value", "value.json", "--mode", "active", "--n",
                    "40", "--minutes", "60", "--ask", "auto", "--reviewer", "reviewer", "--overlap", "second",
                    "--components", "depth", "--arms", "baseline", "strong", "--uniform-share", "0.3",
                    "--repeat-share", "0.2", "--seed", "1", "--out", "labels.json", "--form", "labels.csv",
                    "--explain", "a1", "--allow-unconverged", "--study", "study", "--strict", "--json"],
}
MINIMAL = {
    "scores import": ["eval", "scores", "import", "--out", "study"],
    "scores check": ["eval", "scores", "check", "study", "--rubric", "rubric.json"],
    "scores add": ["eval", "scores", "add", "study", "--rubric", "rubric.json", "--judge", "reviewer", "--pair",
                   "r1", "r2", "--verdict", "tie", "--shown-first", "b"],
    "scores export": ["eval", "scores", "export", "study"],
    "simulate": ["eval", "simulate", "--list"],
    "fit": ["eval", "fit", "study", "--rubric", "rubric.json", "--judges", "judges.json"],
    "judges": ["eval", "judges", "fit.json"],
    "rubric check": ["eval", "rubric", "check", "rubric.json"],
    "value": ["eval", "value", "fit.json", "--value", "value.json"],
    "steps": ["eval", "steps", "fit.json", "--value", "value.json"],
    "labels next": ["eval", "labels", "next", "fit.json"],
}
MODULE = {"scores import": "scores", "scores check": "scores", "scores add": "scores", "scores export": "scores",
          "simulate": "simulate", "fit": "fit", "judges": "judges", "rubric check": "rubric", "value": "value",
          "steps": "steps", "labels next": "labels"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopmath")
    sub = parser.add_subparsers(dest="command", required=True)
    cli_registry.register(sub)
    return parser


def test_the_group_declares_every_command():
    assert commands.COMMANDS == tuple(FULL) == tuple(MINIMAL) == tuple(MODULE)
    assert set(MODULE.values()) == set(commands.MODULES)


@pytest.mark.parametrize("command", list(FULL))
def test_every_option_parses(command):
    args = _parser().parse_args(FULL[command])
    assert args.command == "eval" and args.eval_command == command and args.json is True
    assert args.func.target == f"loopmath.eval.{MODULE[command]}:run"
    minimal = _parser().parse_args(MINIMAL[command])
    assert minimal.eval_command == command and minimal.json is False


def test_option_values_and_defaults():
    p = _parser()
    fit = p.parse_args(FULL["fit"])
    assert (fit.study, fit.rubric, fit.judges, fit.seed, fit.structure, fit.halo, fit.cuts) == (
        "study", "rubric.json", "judges.json", 1, "factor:2", "full", "free")
    assert fit.terms == ["family=off", "judge_input=auto"] and fit.components == ["depth", "clarity"]
    assert (fit.draws, fit.chains, fit.keep_draws, fit.check_only, fit.out) == (500, 2, 400, True, "study/fit.json")
    fit = p.parse_args(MINIMAL["fit"])
    assert (fit.seed, fit.structure, fit.halo, fit.cuts, fit.terms, fit.draws, fit.chains, fit.keep_draws,
            fit.components, fit.check_only, fit.allow_simulated, fit.out) == (
        0, None, "factor", "fixed", None, None, None, 1000, None, False, False, None)
    labels = p.parse_args(MINIMAL["labels next"])
    assert (labels.fit, labels.mode, labels.ask, labels.n, labels.minutes, labels.uniform_share,
            labels.repeat_share, labels.seed, labels.value, labels.reviewer, labels.strict, labels.study) == (
        "fit.json", "pilot", "scores", None, None, 0.2, 0.1, 0, None, None, False, None)
    labels = p.parse_args(FULL["labels next"])
    assert (labels.mode, labels.ask, labels.n, labels.minutes, labels.arms, labels.overlap, labels.explain) == (
        "active", "auto", 40, 60.0, ["baseline", "strong"], "second", "a1")
    add = p.parse_args(MINIMAL["scores add"])
    assert (add.pair, add.verdict, add.shown_first, add.step, add.run, add.value) == (
        ["r1", "r2"], "tie", "b", "final", None, None)
    add = p.parse_args(FULL["scores add"])
    assert (add.run, add.step, add.component, add.value, add.of, add.item, add.replaces) == (
        "r1", "draft", "depth", "7", 9, None, "s1")
    imported = p.parse_args(FULL["scores import"])
    assert (imported.ocp, imported.rows, imported.csv, imported.pairs_csv, imported.items, imported.labels,
            imported.pair_component, imported.allow_simulated, imported.out) == (
        ["a.ocp.json", "b.ocp.json"], "rows.jsonl", "labels.csv", "pairs.csv", "items.jsonl", "plan.json", "overall",
        True, "study")
    value = p.parse_args(FULL["value"])
    assert (value.fit, value.value, value.reference, value.check, value.labels, value.allow_unconverged) == (
        "study/fit.json", "value.json", "baseline", "design", ["labels-001.json", "labels-002.json"], True)
    steps = p.parse_args(FULL["steps"])
    assert (steps.reference, steps.pool, steps.runs) == ("baseline", True, "study/runs.jsonl")
    assert p.parse_args(FULL["scores export"]).run == ["r1", "r2"]
    assert p.parse_args(MINIMAL["simulate"]).list is True and p.parse_args(FULL["simulate"]).spec == "halo"
    rubric = p.parse_args(FULL["rubric check"])
    assert (rubric.rubric, rubric.judges, rubric.value, rubric.items, rubric.fit) == (
        "rubric.json", "judges.json", "value.json", "items.jsonl", "study/fit.json")
    judges = p.parse_args(FULL["judges"])
    assert (judges.target_error, judges.budget_usd, judges.study, judges.strict) == (0.05, 0.2, "study", True)


@pytest.mark.parametrize("argv", [
    ["eval"], ["eval", "scores"], ["eval", "scores", "import"], ["eval", "fit", "study"],
    ["eval", "fit", "study", "--rubric", "r", "--judges", "j", "--halo", "none"], ["eval", "value", "fit.json"],
    ["eval", "labels", "next", "fit.json", "--mode", "greedy"], ["eval", "scores", "check", "study"],
    ["eval", "scores", "add", "study", "--rubric", "r"], ["eval", "rubric", "check"], ["eval", "nothing"],
])
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as caught:
        _parser().parse_args(argv)
    assert caught.value.code == 2
    capsys.readouterr()


def test_every_handler_target_imports_and_is_callable():
    targets = [t for t in cli_registry.targets(_parser()) if t.startswith("loopmath.eval.")]
    assert targets == sorted(f"loopmath.eval.{m}:run" for m in commands.MODULES)
    for target in targets:
        module, name = target.split(":")
        assert callable(getattr(importlib.import_module(module), name))


def test_every_module_of_the_layout_exists_with_run():
    for name in ("scores", "simulate", "model", "engine_pymc", "fit", "judges", "rubric", "value", "steps",
                 "designcheck", "labels", "adapters.ocp"):
        assert callable(importlib.import_module(f"loopmath.eval.{name}").run)
    importlib.import_module("loopmath.eval.specs")


def test_not_built_names_the_command_and_exits_3(capsys):
    assert commands.not_built(argparse.Namespace(eval_command="scores check")) == 3
    assert capsys.readouterr().err == "loopmath eval scores check: not built yet\n"
    assert commands.not_built(argparse.Namespace()) == 3
    assert capsys.readouterr().err == "loopmath eval: not built yet\n"


def test_a_contract_error_becomes_an_error_line_and_exit_1(monkeypatch, capsys):
    module = importlib.import_module("loopmath.eval.judges")

    def run(args):
        raise C.ContractError("fit.json", "schema", "missing", "add it", "missing_schema")

    monkeypatch.setattr(module, "run", run)
    assert cli.main(["eval", "judges", "fit.json"]) == 1
    captured = capsys.readouterr()
    assert captured.err == "error: fit.json: schema: missing; add it\n" and captured.out == ""
    assert cli.main(["eval", "judges", "fit.json", "--json"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"schema": "loopmath.error/1", "ok": False, "command": "eval judges",
                                        "error": "fit.json: schema: missing; add it", "exit": 1}


def test_the_group_is_listed_as_experimental(capsys):
    assert "eval" not in cli_registry.HIDDEN
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    entry = [line for line in out.splitlines() if line.split()[:1] == ["eval"]]
    assert len(entry) == 1 and entry[0].split()[1] == "experimental:"
    with pytest.raises(SystemExit):
        cli.main(["eval", "--help"])
    out = capsys.readouterr().out
    assert out.split("\n\n")[1].startswith("Experimental: ")
    for name in ("scores", "simulate", "fit", "judges", "rubric", "value", "steps", "labels"):
        assert any(line.split()[:1] == [name] for line in out.splitlines())


def test_help_does_not_load_numpy_or_any_stage():
    code = ("import sys\nfrom loopmath import cli\n"
            "try:\n    cli.main(['eval', 'fit', '--help'])\nexcept SystemExit:\n    pass\n"
            "bad = [m for m in sys.modules if m.startswith('loopmath.eval.') and "
            "m not in ('loopmath.eval.commands', 'loopmath.eval.contracts')]\n"
            "print('LOADED', bad)\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(ROOT / "src"), "LOOPMATH_NO_UPDATE_CHECK": "1", "NO_COLOR": "1"})
    assert run.stdout.strip().endswith("LOADED []")
