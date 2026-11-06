"""The `loopmath flow` command group: its arguments, its handlers, and its place on the command list. No other
CLI output moves: the stored outputs from before the group gain only `flow` (in the command lists, the usage line
and one help entry), `loopmath --version` is pinned, and the unknown-command error gains only `'flow'`."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath import cli, cli_registry
from loopmath.flows import commands

VERBS = ("init", "import", "export", "show", "plan", "next", "record", "score", "status", "check")
ROOT = Path(__file__).resolve().parents[2]
BASE = json.loads((ROOT / "tests" / "fixtures" / "flows" / "to_dict.base.json").read_text(encoding="utf-8"))
CLI = sorted(k for k in BASE if k.startswith("cli/"))
UNKNOWN = ("cli/loopmath not-a-command", "cli/loopmath not-a-command --json")
GAINS_FLOW = ("cli/loopmath", "cli/loopmath --json", "cli/loopmath --help")
HARNESS_ENTRY = ("    harness             experimental: tune a Claude Code harness: init, plan,\n"
                 "                        run, watch, show, apply\n")
FLOW_ENTRY = ("    flow                run a flow experiment: a fixed workflow, edits to try,\n"
              "                        judges and a margin\n")
DROP = ("PYTHONPATH", "COLUMNS", "LINES", "FORCE_COLOR", "PYTHON_COLORS", "NO_COLOR")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loopmath")
    sub = parser.add_subparsers(dest="command", required=True)
    cli_registry.register(sub)
    return parser


def test_module_imports():
    assert importlib.import_module("loopmath.flows.commands") is commands


@pytest.mark.parametrize("verb", VERBS)
def test_signature_is_pinned(verb):
    assert str(inspect.signature(getattr(commands, f"{verb}_verb"))) == "(args: 'argparse.Namespace') -> 'int'"


# 14. The arguments


@pytest.mark.parametrize("argv", [
    ["flow", "init"],
    ["flow", "init", "lit-brief", "--answers", "answers.json", "--from-ocp", "a.ocp.json", "b.ocp.json",
     "--from-rows", "rows.jsonl", "--toml", "--home", "h", "--json"],
    ["flow", "import", "flow.toml"],
    ["flow", "import", "flow.toml", "lit-brief"],
    ["flow", "export", "lit-brief", "--out", "flow.toml"],
    ["flow", "show", "lit-brief/flow.ocp.json"],
    ["flow", "plan", "lit-brief", "--budget-usd", "6", "--margin", "0.1"],
    ["flow", "next", "lit-brief", "--n", "4", "--seed", "7", "--json"],
    ["flow", "record", "lit-brief", "--rows", "rows.jsonl"],
    ["flow", "record", "lit-brief", "--ocp", "a.ocp.json", "b.ocp.json", "--allow-unplanned"],
    ["flow", "record", "lit-brief", "--ocp", "a.ocp.json", "--allow-simulated"],
    ["flow", "score", "lit-brief", "--run", "run_a", "--judge", "person", "0.9"],
    ["flow", "score", "lit-brief", "--run", "run_a", "--judge", "j-claude", "0.5", "--replace", "--observed-at",
     "2026-11-04T09:30:00-08:00", "--at-attempt", "att_1", "--home", "h", "--json"],
    ["flow", "status", "lit-brief/flow.toml"],
    ["flow", "check", "lit-brief", "--arm", "extract-sonnet"],
])
def test_flow_subcommands_parse(argv):
    args = _parser().parse_args(argv)
    assert args.command == "flow" and args.flow_command == argv[1]
    assert args.func.target == f"loopmath.flows.commands:{argv[1]}_verb"


def test_every_flow_handler_is_a_target():
    found = cli_registry.targets(_parser())
    assert {f"loopmath.flows.commands:{v}_verb" for v in VERBS} <= set(found)


@pytest.mark.parametrize("argv", [
    ["flow", "record", "lit-brief"],
    ["flow", "record", "lit-brief", "--rows", "rows.jsonl", "--ocp", "a.ocp.json"],
    ["flow", "check", "lit-brief"],
    ["flow"],
])
def test_missing_required_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv, named", [
    (["flow", "score", "lit-brief", "--judge", "person", "0.9"], "the following arguments are required: --run"),
    (["flow", "score", "lit-brief", "--run", "run_a"], "the following arguments are required: --judge"),
    (["flow", "score", "lit-brief", "--run", "run_a", "--judge", "person"], "argument --judge: expected 2 arguments"),
    (["flow", "score", "--run", "run_a", "--judge", "person", "0.9"], "the following arguments are required: FLOW"),
])
def test_flow_score_needs_the_flow_a_run_and_a_judge_with_its_value(argv, named, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    err = capsys.readouterr().err
    assert exc.value.code == 2 and err.startswith("usage: loopmath flow score ") and f": error: {named}\n" in err


def test_a_refused_flow_next_json_writes_one_error_object(capsys, tmp_path):
    folder = tmp_path / "lit-brief"
    folder.mkdir()
    shutil.copyfile(ROOT / "examples" / "flows" / "lit-brief" / "flow.ocp.json", folder / "flow.ocp.json")
    assert cli.main(["flow", "next", str(folder), "--home", str(tmp_path / "home"), "--json"]) == 1
    captured = capsys.readouterr()
    error = json.loads(captured.out)
    assert error["schema"] == "loopmath.error/1" and error["ok"] is False and error["exit"] == 1
    assert error["command"] == "flow next" and error["error"] == (f"there is no plan.json in {folder}; run loopmath "
                                                                  "flow plan first")
    assert sorted(p.name for p in folder.iterdir()) == ["flow.ocp.json"]


# 15. On the command list


def test_flow_is_listed(capsys):
    assert "flow" not in cli_registry.HIDDEN
    sub = next(a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction))
    assert "flow" in sub.choices and "flow" in cli_registry.command_names(sub)
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    entries = [line for line in out.splitlines() if line.split()[:1] == ["flow"]]
    assert len(entries) == 1 and "run a flow experiment" in entries[0]
    listed = out.split("{", 1)[1].split("}", 1)[0].split(",")
    assert "workflows" in listed and listed[-2:] == ["harness", "flow"]


def _run_cli(argv: list[str], tmp_path) -> dict:
    """`python -m loopmath` as the stored outputs were made: fresh state, an empty folder, 80 columns."""
    for name in ("work", "home", "lmhome", "cache"):
        (tmp_path / name).mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LOOPMATH_") and k not in DROP}
    env.update(HOME=str(tmp_path / "home"), LOOPMATH_HOME=str(tmp_path / "lmhome"),
               LOOPMATH_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(ROOT / "src"), COLUMNS="80",
               LOOPMATH_NO_UPDATE_CHECK="1")  # no daily pypi.org check (loopmath/update.py), as conftest sets
    run = subprocess.run([sys.executable, "-m", "loopmath", *argv], cwd=tmp_path / "work", env=env,
                         capture_output=True, check=False)
    return {"exit": run.returncode, "stdout": run.stdout.decode("utf-8"), "stderr": run.stderr.decode("utf-8")}


def _stored(key: str) -> dict:
    stored, python = BASE[key], "%d.%d" % sys.version_info[:2]
    if stored["python"] != python:
        pytest.skip(f"the stored output was made with Python {stored['python']}; argparse output differs on {python}")
    return {k: stored[k] for k in ("exit", "stdout", "stderr")}


def test_stored_cli_outputs_cover_the_six_commands():
    assert CLI == ["cli/loopmath", "cli/loopmath --help", "cli/loopmath --json", "cli/loopmath --version",
                   "cli/loopmath not-a-command", "cli/loopmath not-a-command --json"]
    assert all(BASE[k]["argv"] == k.split()[1:] for k in CLI)


@pytest.mark.parametrize("key", [k for k in CLI if k not in UNKNOWN + GAINS_FLOW])
def test_cli_output_is_unchanged(key, tmp_path):
    assert _run_cli(BASE[key]["argv"], tmp_path) == _stored(key)


@pytest.mark.parametrize("key", GAINS_FLOW)
def test_cli_output_gains_only_flow(key, tmp_path):
    """`flow` joins the command lists after `harness`, and `--help` gains its entry after harness's. The usage
    line keeps its wrapping (argparse does not break the braces), so the comparison is exact."""
    stored = _stored(key)
    expected = dict(stored)
    if key.endswith("--help"):
        assert stored["exit"] == 0 and stored["stderr"] == ""
        assert stored["stdout"].count(",harness}") == 2 and stored["stdout"].count(HARNESS_ENTRY) == 1
        expected["stdout"] = (stored["stdout"].replace(",harness}", ",harness,flow}")
                              .replace(HARNESS_ENTRY, HARNESS_ENTRY + FLOW_ENTRY))
    else:
        assert stored["exit"] == 2 and stored["stderr"].count(", harness\n") == 1
        expected["stderr"] = stored["stderr"].replace(", harness\n", ", harness, flow\n")
        if key.endswith("--json"):
            assert stored["stdout"].count(', harness",') == 1
            expected["stdout"] = stored["stdout"].replace(', harness",', ', harness, flow",')
        else:
            assert stored["stdout"] == ""
    assert _run_cli(BASE[key]["argv"], tmp_path) == expected


@pytest.mark.parametrize("key", UNKNOWN)
def test_unknown_command_error_gains_only_flow(key, tmp_path):
    """`'flow'` joins the choices, and, now that it is listed, the usage braces."""
    stored = _stored(key)
    assert stored["exit"] == 2 and stored["stderr"].count("'harness'") == 1
    assert stored["stderr"].count(",harness}") == 2
    expected = dict(stored, stderr=stored["stderr"].replace("'harness'", "'harness', 'flow'")
                    .replace(",harness}", ",harness,flow}"))
    if key.endswith("--json"):
        assert stored["stdout"].count("'harness'") == 1 and stored["stdout"].count(",harness}") == 1
        expected["stdout"] = (stored["stdout"].replace("'harness'", "'harness', 'flow'")
                              .replace(",harness}", ",harness,flow}"))
    else:
        assert stored["stdout"] == ""
    assert _run_cli(BASE[key]["argv"], tmp_path) == expected


def test_cli_runs_write_no_update_json(tmp_path, monkeypatch):
    """The update check stays off in every CLI run here, as release-check checks: no `update.json`. `CI` is
    dropped so only `LOOPMATH_NO_UPDATE_CHECK` guards, and a closed local proxy keeps a failure offline."""
    monkeypatch.delenv("CI", raising=False)
    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for i, key in enumerate(CLI):
        (tmp_path / str(i)).mkdir()
        _run_cli(BASE[key]["argv"], tmp_path / str(i))
    assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("update.json")] == []
