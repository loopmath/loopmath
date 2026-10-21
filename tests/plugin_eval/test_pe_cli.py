"""`loopmath adapt plugin-eval` as a new user meets it: exit codes, one JSON object under --json (errors too),
nothing written on a refusal, a results folder as input, the rows file, the summary and its next commands.
No test starts Claude Code."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pefixture as F  # noqa: E402

from loopmath import harness_runs as HR  # noqa: E402
from loopmath.cli import main  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    (tmp_path / "user-home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))

    def no_process(*a, **k):
        raise AssertionError(f"adapt plugin-eval started a process: {a!r}")

    monkeypatch.setattr(subprocess, "Popen", no_process)   # never claude, never anything


@pytest.fixture
def suite(tmp_path) -> F.Built:
    return F.write(tmp_path / "lint-helper", F.docs_suite(), manifest={"name": "lint-helper", "version": "0.4.0"})


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_a_result_file_becomes_one_run_document_per_run_and_the_rows_file(suite, tmp_path, capsys):
    out, rows = tmp_path / "ocp", tmp_path / "rows.jsonl"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(suite.path), "--out", str(out), "--rows", str(rows),
                             "--json")
    assert code == 0, err
    s = json.loads(stdout)
    assert s["schema"] == "loopmath.adapt.runs/1" and s["ok"] and s["importer"] == "plugin-eval"
    assert s["format"] == "claude-plugin-eval/1" and s["written"] == 24 and len(list(out.glob("*.ocp.json"))) == 24
    assert s["arms"] == {"with": 12, "without": 12} and s["scored"] == 23
    assert len(HR.read(rows)) == 24
    pe = s["plugin_eval"]
    for key in ("plugin", "cases", "runs", "pairs", "suite_cost_usd", "partial", "dropped_text", "errors",
                "threshold", "claude_version", "next"):
        assert key in pe, key


def test_the_text_summary_names_the_plugin_the_suite_cost_and_the_next_commands(suite, tmp_path, capsys):
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(suite.path), "--out", str(tmp_path / "ocp"))
    assert code == 0, err
    assert "plugin lint-helper 0.4.0" in stdout and "4 cases" in stdout and "12 pairs" in stdout
    assert "suite cost $1.84 (list price, with judge calls; not split over runs)" in stdout
    assert "cost unknown" in stdout and "--cost-from" in stdout
    assert f"loopmath run import {tmp_path / 'ocp'} --finish" in stdout
    lines = stdout.splitlines()
    assert "loopmath check-change from-runs --base-label arm:without --cand-label arm:with" in lines   # its own cap
    assert "loopmath gate --base arm:without --cand arm:with --rule accept --r-max 24" in lines   # the store's runs


def test_a_results_folder_finds_its_aggregate_result(suite, tmp_path, capsys):
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(suite.path.parent), "--out", str(tmp_path / "o"),
                             "--json")
    assert code == 0 and json.loads(stdout)["written"] == 24, err
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(tmp_path), "--json")
    assert code == 1 and "aggregate-result.json" in json.loads(stdout)["error"]


def test_a_missing_input_exits_2_and_a_refused_file_exits_1_writing_nothing(suite, tmp_path, capsys):
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(tmp_path / "nope.json"), "--json")
    assert code == 2 and json.loads(stdout)["exit"] == 2
    doc = F.docs_suite()
    doc["schemaVersion"] = 2
    bad = F.write(tmp_path / "bad", doc)
    out = tmp_path / "o"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(bad.path), "--out", str(out), "--json")
    one = json.loads(stdout)
    assert code == 1 and one["ok"] is False and one["exit"] == 1 and "schemaVersion 2" in one["error"]
    assert "error: adapt plugin-eval: " in err and not out.exists()


def test_a_cost_file_line_that_matches_nothing_exits_1_writing_nothing(suite, tmp_path, capsys):
    costs = F.write_costs(tmp_path / "c.jsonl", [{"case": "nope", "arm": "with", "run": 1, "usd": 0.1}])
    out = tmp_path / "o"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(suite.path), "--cost-from", str(costs), "--out", str(out))
    assert code == 1 and "c.jsonl:1" in err and not out.exists()
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(suite.path), "--cost-from", str(tmp_path / "x.jsonl"))
    assert code == 2 and "x.jsonl" in err


@pytest.mark.parametrize("flag", [["--cost-from", "c.jsonl"], ["--outcome", "score"], ["--model", "m"],
                                  ["--plugin", "p"], ["--keep-text"]])
def test_plugin_eval_flags_on_another_importer_exit_2_naming_the_flag(tmp_path, capsys, flag):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"schema": HR.SCHEMA, "task_id": "t1"}) + "\n", encoding="utf-8")
    code, stdout, err = _run(capsys, "adapt", "harness-runs", str(rows), *flag)
    assert code == 2 and flag[0] in err and "plugin-eval" in err


def test_privacy_full_exits_2_writing_nothing_and_points_to_keep_text(suite, tmp_path, capsys):
    out, rows = tmp_path / "o", tmp_path / "rows.jsonl"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(suite.path), "--privacy", "full",
                             "--out", str(out), "--rows", str(rows))
    assert code == 2 and not out.exists() and not rows.exists()
    assert "adapt plugin-eval: --privacy full applies to adapt harness2 only" in err
    assert "--keep-text" in err and "rows file" not in err


def _next(capsys, tmp_path, built: F.Built, *flags: str) -> list[str]:
    out = tmp_path / f"o-{built.root.name}-{len(flags)}"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(built.path), "--out", str(out), "--json", *flags)
    assert code == 0, err
    return json.loads(stdout)["plugin_eval"]["next"]


def test_the_next_commands_give_the_gate_its_own_cap_and_both_the_rule(tmp_path, capsys):
    twin = F.write(tmp_path / "twin", F.twin_suite())
    rows = str(tmp_path / "rows.jsonl")
    nxt = _next(capsys, tmp_path, twin, "--rows", rows)
    check = next(line for line in nxt if line.startswith("loopmath check-change from-runs "))
    gate = next(line for line in nxt if line.startswith("loopmath gate "))
    assert "--max-pairs" not in check and "--rule" not in check   # from-runs picks its cap; the runs' own rule
    assert gate == f"loopmath gate --base arm:without --cand arm:with --runs {rows} --rule accept --r-max 6"
    nxt = _next(capsys, tmp_path, twin, "--outcome", "score")
    check = next(line for line in nxt if line.startswith("loopmath check-change from-runs "))
    gate = next(line for line in nxt if line.startswith("loopmath gate "))
    assert check.endswith("--rule score:plugin-eval") and "--rule score:plugin-eval --r-max 6" in gate
    assert not any("calibrated" in line and "bound" in line and "gate" in line for line in nxt)   # the gate says


def test_past_the_largest_calibrated_cap_the_next_block_says_so(tmp_path, capsys):
    many = F.document([F.case(f"c{i}", [F.run([F.grader("g", True)])] * 10, [F.run([F.grader("g", False)])] * 10)
                       for i in range(5)])
    nxt = _next(capsys, tmp_path, F.write(tmp_path / "many", many))
    assert any("--r-max 96" in line for line in nxt if line.startswith("loopmath gate ")), nxt   # 50 pairs
    assert any("50 pairs" in line and "past the largest calibrated cap, 48" in line for line in nxt), nxt


def test_a_withheld_case_is_named_and_left_out_of_the_rows(suite, tmp_path, capsys):
    doc = F.docs_suite()
    del doc["cases"][1]["aggregates"]["delta"]   # rename-config-key: both arms, no delta
    built = F.write(tmp_path / "held", doc, manifest={"name": "lint-helper", "version": "0.4.0"})
    rows = tmp_path / "rows.jsonl"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(built.path), "--out", str(tmp_path / "o"),
                             "--rows", str(rows))
    assert code == 0, err
    line = next(ln for ln in stdout.splitlines() if "withheld from" in ln)   # the tmp path holds "withheld" too
    assert "rename-config-key" in line and "not comparable" in line, stdout
    assert "9 pairs" in stdout and {r["task_id"] for r in HR.read(rows)} == {"fix-null-check", "add-skill-call",
                                                                            "write-docs-page"}
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(built.path), "--out", str(tmp_path / "o2"), "--json")
    s = json.loads(stdout)
    assert s["written"] == 18 and [w["case"] for w in s["plugin_eval"]["withheld"]] == ["rename-config-key"]
    for c in doc["cases"]:
        c["aggregates"].pop("delta", None)
    none = F.write(tmp_path / "none", doc)
    out = tmp_path / "o3"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(none.path), "--out", str(out))
    assert code == 1 and "no case is comparable" in err and not out.exists()


def test_an_ungraded_errored_run_exits_1_naming_it_and_writes_nothing(tmp_path, capsys):
    doc = F.docs_suite(run_score=False)
    run = next(c for c in doc["cases"] if c["name"] == "rename-config-key")["arms"]["with"][2]
    del run["graders"]   # error "timed out after 300s", no score, no grader list: not a documented 0
    out = tmp_path / "o"
    code, stdout, err = _run(capsys, "adapt", "plugin-eval", str(F.write(tmp_path / "p", doc).path), "--out", str(out))
    assert code == 1 and not out.exists()
    assert "rename-config-key" in err and "run 3" in err and "no score and no grader list" in err


def test_help_names_the_importer(capsys):
    with pytest.raises(SystemExit):
        main(["adapt", "--help"])
    out = capsys.readouterr().out
    assert "plugin-eval" in out and "--cost-from" in out
