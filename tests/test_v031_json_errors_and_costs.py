"""0.3.1: errors under `--json`, an unknown cost, and a store path that is a file.

- Under `--json`, a command that fails prints one JSON object on stdout (`"ok": false`, `error`, `exit`) with the
  same exit code and the same message on stderr: `fit`, `posterior`, `recommend` and `runs`, and every usage error.
  A command that printed its own JSON error is left alone, and without `--json` nothing changes.
- `runs`, `status` and `budget` never show an unknown cost as $0: `cost unknown`, or `$X known`.
- `status` with a store path that is a file says so in one line and exits 1, with no traceback.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pytest

from loopmath import cli

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "spec" / "examples" / "harness-runs"


@pytest.fixture
def store(tmp_path, monkeypatch):
    home = tmp_path / "lm"
    monkeypatch.setenv("LOOPMATH_HOME", str(home))
    return home


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    try:
        code = cli.main(list(argv))
    except SystemExit as exc:
        code = exc.code
    out, err = capsys.readouterr()
    return code, out, err


def _one_error(out: str, code: int) -> dict:
    obj = json.loads(out)  # exactly one object: json.loads refuses trailing text
    assert obj["schema"].startswith("loopmath.") and obj["ok"] is False and obj["exit"] == code
    assert obj["error"]
    return obj


@pytest.mark.parametrize("argv, code, words", [
    (["posterior", "--json"], 5, "no fit yet"),
    (["recommend", "--type", "bugfix", "--repo", "acme/api", "--json"], 1, "task type must be one of"),
    (["runs", "--since", "xyz", "--json"], 1, "--since xyz"),
    (["runs", "--run", "run_nope", "--json"], 2, "no run run_nope"),
    (["fit", "--without", "nosuch", "--json"], 2, "unknown source nosuch"),
])
def test_handler_errors_print_one_json_object(store, capsys, argv, code, words):
    got, out, err = _run(capsys, *argv)
    assert got == code
    obj = _one_error(out, code)
    assert words in obj["error"] and obj["schema"] == "loopmath.error/1"
    assert obj["command"] == argv[0]
    assert f"error: {obj['error']}" in err  # the stderr message is unchanged


@pytest.mark.parametrize("argv, command", [
    (["gate", "--base", "arm=a", "--json"], "gate"),
    (["check-change", "start", "--base", "a", "--json"], "check-change start"),
    (["runs", "--bogus", "--json"], "runs"),
    (["posterior", "--level", "nosuch", "--json"], "posterior"),
    (["fit", "--json", "--seed", "x"], "fit"),
])
def test_usage_errors_print_one_json_object(store, capsys, argv, command):
    got, out, err = _run(capsys, *argv)
    assert got == 2
    obj = _one_error(out, 2)
    assert obj["command"] == command
    assert "usage:" in err and f"error: {obj['error']}" in err


def test_without_json_stdout_stays_empty(store, capsys):
    for argv in (["posterior"], ["gate", "--base", "arm=a"]):
        got, out, err = _run(capsys, *argv)
        assert got in (2, 5) and out == "" and "error:" in err


def test_a_command_with_its_own_json_error_prints_one_object(store, capsys):
    got, out, _ = _run(capsys, "check-change", "status", "--json")
    assert got == 2
    obj = _one_error(out, 2)
    assert obj["schema"] == "loopmath.check-change/1"


def test_a_hint_line_goes_to_its_own_field(capsys):
    from loopmath import cli_errors

    def fails(args):
        print("error: no fit yet", file=sys.stderr)
        print("hint: `loopmath plan --members grid` needs no fit", file=sys.stderr)
        return 5

    assert cli_errors.run_guarded(fails, None, "plan") == 5
    obj = _one_error(capsys.readouterr().out, 5)
    assert obj["error"] == "no fit yet" and obj["hint"] == "`loopmath plan --members grid` needs no fit"


def test_json_after_a_double_dash_is_not_the_flag():
    from loopmath import cli_errors

    assert cli_errors.wants_json(["runs", "--json"]) and not cli_errors.wants_json(["runs", "--", "--json"])


def test_an_uncaught_exception_prints_the_object_and_still_raises(capsys):
    from loopmath import cli_errors

    def boom(args):
        raise RuntimeError("broken store")

    with pytest.raises(RuntimeError):
        cli_errors.run_guarded(boom, None, "status")
    obj = _one_error(capsys.readouterr().out, 1)
    assert obj["error"] == "RuntimeError: broken store" and obj["command"] == "status"


# ---------------------------------------------------------------- #5 an unknown cost is not $0
def _import_rows(capsys, tmp_path: Path, rows_name: str, *, keep_cost: bool, suffix: str) -> None:
    src = tmp_path / "hr"
    if not src.exists():
        shutil.copytree(EXAMPLES, src)
    now = datetime.now().astimezone().replace(microsecond=0).isoformat()
    lines = []
    for line in (EXAMPLES / "hand-ab.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not keep_cost:
                row.pop("cost", None)
            row["id"] += suffix
            row["started_at"] = now  # this month, so status counts them
            lines.append(json.dumps(row))
    (src / rows_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / f"ocp-{suffix}"
    assert cli.main(["adapt", "harness-runs", str(src / rows_name), "--out", str(out)]) == 0
    assert cli.main(["run", "import", str(out), "--finish", "--no-fit"]) == 0
    capsys.readouterr()


def test_runs_status_and_budget_say_unknown_not_zero(store, tmp_path, capsys):
    _import_rows(capsys, tmp_path, "nocost.jsonl", keep_cost=False, suffix=":n")
    _, out, _ = _run(capsys, "runs")
    head = out.splitlines()[0]
    assert head.startswith("6 runs, cost unknown;") and "$0" not in head
    _, out, _ = _run(capsys, "status")
    spend = [line for line in out.splitlines() if line.startswith("spend this month")][0]
    assert spend == "spend this month: unknown (tokens unknown); no cap"
    assert "  6 runs, cost unknown" in out.splitlines() and "$0" not in out
    _, out, _ = _run(capsys, "budget")
    assert "unknown, tokens unknown; 6 runs, cost unknown" in out and "spent since" in out
    _, out, _ = _run(capsys, "status", "--json")
    spent = json.loads(out)["budget"]["spent"]
    assert spent["runs"] == 6 and spent["runs_with_usd"] == 0 and spent["runs_with_tokens"] == 0


def test_some_costs_known_read_as_known(store, tmp_path, capsys):
    _import_rows(capsys, tmp_path, "nocost.jsonl", keep_cost=False, suffix=":n")
    _import_rows(capsys, tmp_path, "costed.jsonl", keep_cost=True, suffix=":c")
    _, out, _ = _run(capsys, "runs")
    assert out.splitlines()[0].startswith("12 runs, $1.29 known, 6 with no cost (tokens unknown);")
    _, out, _ = _run(capsys, "status")
    assert "spend this month: $1.29 known (tokens unknown); no cap" in out
    assert "  runs $1.29 known over 12 runs, 6 with no full cost" in out


def test_every_cost_known_reads_as_before(store, tmp_path, capsys):
    _import_rows(capsys, tmp_path, "costed.jsonl", keep_cost=True, suffix=":c")
    _, out, _ = _run(capsys, "status")
    assert "  runs $1.29 over 6 runs" in out  # every run has dollars; the rows give no tokens
    assert "spend this month: $1.29 (tokens unknown); no cap" in out and "0 tokens" not in out


# ---------------------------------------------------------------- #15 a store path that is a file
@pytest.mark.parametrize("how", ["env", "flag"])
@pytest.mark.parametrize("json_flag", [False, True])
def test_status_on_a_file_store_path_is_one_error_line(tmp_path, monkeypatch, capsys, how, json_flag):
    afile = tmp_path / "afile"
    afile.write_text("not a store\n")
    argv = ["status"] + (["--home", str(afile)] if how == "flag" else []) + (["--json"] if json_flag else [])
    if how == "env":
        monkeypatch.setenv("LOOPMATH_HOME", str(afile))
    got, out, err = _run(capsys, *argv)
    assert got == 1
    assert err == f"error: the store path {afile} is not a folder: it is a file; point --home or LOOPMATH_HOME at " \
                  "a folder\n"
    if json_flag:
        assert "is a file" in _one_error(out, 1)["error"]
    else:
        assert out == ""


def test_a_file_on_the_store_path_is_named(tmp_path, capsys):
    afile = tmp_path / "afile"
    afile.write_text("x")
    got, _, err = _run(capsys, "status", "--home", str(afile / "sub"))
    assert got == 1 and f"{afile} on its path is a file" in err and "Traceback" not in err
