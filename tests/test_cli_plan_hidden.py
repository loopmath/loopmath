"""`loopmath plan` is off the user surface but still callable for loopmath-exp's sweep, `prior` has a
description in `loopmath --help`, and `adapt` says which OCP version it emits."""

from __future__ import annotations

import json

import pytest

from loopmath import cli
from loopmath.recommend.backlog import EXPERIMENTAL_NOTE

# The keys `plan --members grid --json` printed before it was hidden; the sweep reads this object.
GRID_KEYS = ["schema", "members", "slates", "spent_usd", "remaining_usd", "budget_usd", "tasks", "cells", "task_types",
             "coverage", "priced_by", "notes", "rule", "created_at"]


def _help(capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--help"])
    assert exc.value.code == 0
    return capsys.readouterr().out


def test_plan_is_not_in_the_main_help_but_prior_is_described(capsys):
    out = _help(capsys)
    usage, listing = out.split("\n\n", 1)
    assert "plan" not in usage.replace("\n", " ").split("{", 1)[1].split("}", 1)[0].split(",")
    commands = [line.split()[0] for line in listing.splitlines() if line.startswith("    ") and line.split()]
    assert "plan" not in commands and "recommend" in commands
    prior = next(line for line in listing.splitlines() if line.strip().startswith("prior "))
    assert len(prior.split()) > 1  # a description after the name
    assert "OCP v0.2" in out and "ocp migrate" in out


def test_plan_help_still_works_and_says_experimental(capsys):
    assert "Experimental" in _help(capsys, "plan")


def _backlog(tmp_path):
    path = tmp_path / "tasks.jsonl"
    lines = [{"id": "tsk_a", "type": "feature", "repo": "bench", "subtype": "p1", "title": "one"},
             {"id": "tsk_b", "type": "feature", "repo": "bench", "subtype": "p2", "title": "two"}]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def _sweep_argv(backlog, home, **over):
    """What `loopmath_exp.lm.Loopmath.plan()` passes: `plan --backlog B --budget-usd X --slate-size N --members M
    [--max-slates K] [--models A,B] [--target T | --rule R] --home H --json`."""
    args = ["plan", "--backlog", str(backlog), "--budget-usd", f"{over.get('budget', 40.0):g}", "--slate-size", "2",
            "--members", "grid", "--max-slates", "3", "--models", "gpt-5.6-sol,gpt-5.6-luna",
            "--target", "heldout_perf>=2400"]
    return args + ["--home", str(home), "--json"]


def test_the_sweeps_exact_invocation_parses_and_prints_the_same_keys(capsys, tmp_path):
    home = tmp_path / "lm"
    code = cli.main(_sweep_argv(_backlog(tmp_path), home))
    cap = capsys.readouterr()
    assert code == 0, cap.err
    obj = json.loads(cap.out.strip())  # one JSON object on stdout, as lm.call() requires
    assert list(obj) == GRID_KEYS and obj["schema"] == "loopmath.plan/1" and obj["members"] == "grid"
    assert obj["slates"] and obj["spent_usd"] <= 40.0
    assert cap.err.splitlines() == [EXPERIMENTAL_NOTE]
    assert EXPERIMENTAL_NOTE not in cap.out


def test_the_note_goes_to_stderr_in_text_mode_too(capsys, tmp_path):
    argv = [a for a in _sweep_argv(_backlog(tmp_path), tmp_path / "lm") if a != "--json"]
    assert cli.main(argv) == 0
    cap = capsys.readouterr()
    assert cap.err.splitlines()[0] == EXPERIMENTAL_NOTE and EXPERIMENTAL_NOTE not in cap.out
