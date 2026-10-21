"""`loopmath adapt harness-runs` and `adapt harness2` as a new user meets them: refusals name the file and line,
--json prints one JSON object for errors too, a missing input exits 2, an empty input exits 1, the summary names
its experiments and its unpriced models, and imported runs count as the user's habit."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2fixture as F  # noqa: E402

from loopmath.cli import main  # noqa: E402
from loopmath.recommend import storeread  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "spec" / "examples" / "harness-runs"
SCHEMA = "loopmath.harness-runs/0.1"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    home = tmp_path / "user-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no-claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no-codex"))


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    capsys.readouterr()
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def _row(task_id: str = "t1", **over) -> dict:
    row = {"schema": SCHEMA, "task_id": task_id, "task": {"type": "bug_fix", "repo": "acme/hand"}, "arm": "base",
           "agent": "claude-code", "model": "claude-sonnet-5-5", "outcome": {"accepted": True, "by": "me"},
           "cost": {"usd": 0.1}, "started_at": "2026-01-15T10:00:00-08:00"}
    row.update(over)
    return row


def _rows(path: Path, *rows: dict) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _one_json(out: str) -> dict:
    value = json.loads(out)     # the whole of stdout is one object
    assert isinstance(value, dict)
    return value


# N8: a task type loopmath does not know is refused, in a row and in --type


def test_an_unknown_task_type_in_a_row_is_refused_by_file_line_with_the_closest_type(tmp_path, capsys):
    path = _rows(tmp_path / "runs.jsonl", _row("t1"), _row("t2", task={"type": "bugfix"}))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"))
    assert code == 1 and not (tmp_path / "o").exists(), (out, err)
    assert f"{path}:2: task.type: 'bugfix' is not a loopmath task type" in err and "did you mean 'bug_fix'?" in err


def test_an_unknown_type_flag_exits_2_and_a_known_one_fills_rows(tmp_path, capsys):
    path = _rows(tmp_path / "runs.jsonl", _row("t1", task={"repo": "acme/hand"}))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--type", "bugfix", "--out", str(tmp_path / "o"))
    assert code == 2 and "--type 'bugfix' is not a loopmath task type" in err and "'bug_fix'" in err, err
    assert not (tmp_path / "o").exists()
    code, out, err = _run(capsys, "adapt", "harness2", str(tmp_path), "--type", "docz", "--json")
    assert code == 2 and _one_json(out)["exit"] == 2 and "did you mean 'docs'?" in _one_json(out)["error"]
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--type", "docs", "--out", str(tmp_path / "o"))
    assert code == 0, err
    doc = json.loads(next((tmp_path / "o").glob("*.ocp.json")).read_text(encoding="utf-8"))
    assert doc["run"]["task"]["type"] == "docs"


# N9: the summary names the experiments


def test_the_summary_names_the_experiments_rows_give_or_the_flag_fills(tmp_path, capsys):
    code, out, err = _run(capsys, "adapt", "harness-runs", str(EXAMPLES / "plugin-evals.jsonl"),
                          "--out", str(tmp_path / "pe"), "--json")
    assert code == 0, err
    assert _one_json(out)["experiments"] == [{"experiment": "lint-plugin-eval-1", "runs": 6}]
    code, out, err = _run(capsys, "adapt", "harness-runs", str(EXAMPLES / "hand-ab.jsonl"), "--experiment", "my-ab",
                          "--out", str(tmp_path / "hab"), "--json")
    assert code == 0 and _one_json(out)["experiments"] == [{"experiment": "my-ab", "runs": 6}], err
    code, out, err = _run(capsys, "adapt", "harness-runs", str(EXAMPLES / "hand-ab.jsonl"), "--experiment", "my-ab",
                          "--out", str(tmp_path / "hab"))
    assert "  experiment my-ab: 6 runs" in out.splitlines(), out
    code, out, err = _run(capsys, "adapt", "harness-runs", str(EXAMPLES / "hand-ab.jsonl"), "--out",
                          str(tmp_path / "hab2"), "--json")
    assert _one_json(out)["experiments"] == []


# N10: an empty input is refused, with no next step


@pytest.mark.parametrize("text", ["", "\n  \n\t\n"], ids=["empty", "blank-only"])
def test_an_empty_or_blank_rows_file_exits_1_with_no_next_hint(tmp_path, capsys, text):
    path = tmp_path / "empty.jsonl"
    path.write_text(text, encoding="utf-8")
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"))
    assert code == 1 and "has no rows" in err and "next:" not in out + err, (out, err)
    assert not (tmp_path / "o").exists()
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"), "--json")
    assert code == 1 and _one_json(out)["ok"] is False and "has no rows" in _one_json(out)["error"]


def test_a_harness2_folder_with_no_importable_rollout_exits_1_and_says_why_each_was_skipped(tmp_path, capsys,
                                                                                         monkeypatch):
    from loopmath.adapters import harness2 as H2

    skipped = [{"run_id": "jb-oc-agg2/t1/base1", "reason": "unavailable: no trajectory"},
               {"run_id": "jb-oc-agg2/t1/agg1", "reason": "unavailable: no summary row"}]
    monkeypatch.setattr(H2, "read", lambda path, experiment=None, privacy="metadata_only": SimpleNamespace(
        rows=[], format="harness2/method-2", skipped=skipped, notes=[], experiments=[], counts={}))
    code, out, err = _run(capsys, "adapt", "harness2", str(tmp_path), "--out", str(tmp_path / "o"))
    assert code == 1 and not (tmp_path / "o").exists(), (out, err)
    lines = err.splitlines()
    assert lines[0].startswith("error: adapt harness2: no rollout under") and "(2 skipped, below)" in lines[0]
    assert lines[1:] == ["  skipped jb-oc-agg2/t1/base1: unavailable: no trajectory",
                         "  skipped jb-oc-agg2/t1/agg1: unavailable: no summary row"]
    code, out, err = _run(capsys, "adapt", "harness2", str(tmp_path), "--json")
    got = _one_json(out)
    assert code == 1 and got["exit"] == 1 and got["skipped"] == skipped


# N11: an experiment is a name, never a path


def test_a_row_whose_experiment_is_a_path_is_refused(tmp_path, capsys):
    path = _rows(tmp_path / "runs.jsonl", _row("t1", experiment="/Users/someone/runs/exp1"))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"))
    assert code == 1 and f"{path}:1: experiment: '/Users/someone/runs/exp1' is a path" in err, err
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("name,value", [("harness-runs", "../exp1"), ("harness-runs", "~/exp"),
                                        ("harness2", "a/b"), ("harness2", "two words")])
def test_an_experiment_flag_that_is_a_path_or_a_bad_label_exits_2(tmp_path, capsys, name, value):
    path = _rows(tmp_path / "runs.jsonl", _row("t1")) if name == "harness-runs" else tmp_path
    code, out, err = _run(capsys, "adapt", name, str(path), "--experiment", value, "--out", str(tmp_path / "o"))
    assert code == 2 and f"--experiment {value!r}" in err and not (tmp_path / "o").exists(), err


# N12: messages a user can act on


def test_a_harness_tree_error_names_its_file_and_line(tmp_path, capsys):
    (tmp_path / "trees" / "a").mkdir(parents=True)
    path = _rows(tmp_path / "runs.jsonl", _row("t1", harness="trees/a"), _row("t2", harness="../../etc"),
                 _row("t3", harness="trees/missing"))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"))
    assert code == 1 and err.startswith(f"error: adapt harness-runs: {path}:2: harness: tree path '../../etc' leaves"), err
    _rows(path, _row("t1", harness="trees/a"), _row("t3", harness="trees/missing"))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"))
    assert code == 1 and f"{path}:2: harness: tree path 'trees/missing' is not a folder" in err, err


def test_a_bom_and_a_missing_schema_get_plain_messages(tmp_path, capsys):
    bom = tmp_path / "bom.jsonl"
    bom.write_bytes(b"\xef\xbb\xbf" + (json.dumps(_row()) + "\n").encode("utf-8"))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(bom), "--out", str(tmp_path / "o"))
    assert code == 1 and "byte order mark" in err and "utf-8-sig" not in err, err
    row = _row()
    del row["schema"]
    path = _rows(tmp_path / "runs.jsonl", row)
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"))
    assert code == 1 and f"{path}:1: schema: missing" in err and "None" not in err, err


# N13: --json prints one JSON object on stdout, for success and for errors


def test_adapt_json_errors_are_one_object_on_stdout_and_a_line_on_stderr(tmp_path, capsys):
    path = _rows(tmp_path / "runs.jsonl", _row("t1", harness="../../etc"))
    for argv in (["adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"), "--json"],
                 ["adapt", "harness-runs", str(path), "--json"],
                 ["adapt", "harness2", str(tmp_path / "nope"), "--json"],
                 ["adapt", "harness-runs", "--json"]):
        code, out, err = _run(capsys, *argv)
        got = _one_json(out)
        assert code in (1, 2) and got["ok"] is False and got["exit"] == code, argv
        assert got["schema"] == "loopmath.adapt.runs/1" and err.startswith(f"error: {got['error']}"), (argv, err)


def test_adapt_json_success_is_one_object_and_documents_without_out_go_to_stdout(tmp_path, capsys):
    path = EXAMPLES / "hand-ab.jsonl"
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"), "--json")
    got = _one_json(out)
    assert code == 0 and got["ok"] is True and got["written"] == 6 and err == ""
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--json")
    docs = [json.loads(line) for line in out.splitlines()]
    assert code == 0 and len(docs) == 6 and all("run" in d for d in docs)
    assert json.loads(err)["ok"] is True       # the summary moves to stderr: stdout is the documents


def test_an_output_write_error_exits_1_with_one_json_error_and_no_traceback(tmp_path, capsys):
    blocker = tmp_path / "afile"
    blocker.write_text("x", encoding="utf-8")
    path = EXAMPLES / "hand-ab.jsonl"
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(blocker / "child"), "--json")
    got = _one_json(out)
    assert code == 1 and got["ok"] is False and got["exit"] == 1 and got["written"] == 0, (out, err)
    assert err.startswith("error: adapt harness-runs: cannot write:") and "Traceback" not in err, err
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--rows", str(blocker / "rows.jsonl"), "--json")
    got = _one_json(out)       # no documents went to stdout before the refusal
    assert code == 1 and got["exit"] == 1 and "cannot write" in got["error"], (out, err)
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(blocker / "child"))
    assert code == 1 and out == "" and "(0 of 6 run files written)" in err, err
    assert _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "probe"))[0] == 0
    names = sorted(p.name for p in (tmp_path / "probe").glob("*.ocp.json"))
    out_dir = tmp_path / "o"
    (out_dir / names[-1]).mkdir(parents=True)     # a folder where the last run file goes
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(out_dir), "--json")
    got = _one_json(out)
    assert code == 1 and got["exit"] == 1 and "cannot write" in got["error"] and "Traceback" not in err, (out, err)
    assert not list(out_dir.glob(".*.tmp")), "a failed write leaves no temp file"


# N14: a missing input exits 2


@pytest.mark.parametrize("name", ["harness-runs", "harness2"])
def test_a_missing_input_path_exits_2(tmp_path, capsys, name):
    code, out, err = _run(capsys, "adapt", name, str(tmp_path / "nope"), "--out", str(tmp_path / "o"))
    assert code == 2 and "nope: no such" in err and not (tmp_path / "o").exists(), err


# N17: the summary names the models it could not price


def test_the_summary_names_the_unpriced_models(tmp_path, capsys):
    path = _rows(tmp_path / "runs.jsonl",
                 _row("t1", model="acme-model-9", cost={"tokens": {"input": 100, "output": 10}}),
                 _row("t2", model="acme-model-9", cost={"tokens": {"input": 200, "output": 20}}),
                 _row("t3", model=None, cost={"tokens": {"input": 50, "output": 5}}),
                 _row("t4", model="claude-sonnet-5-5", cost={"usd": 0.2}))
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"), "--json")
    got = _one_json(out)
    assert code == 0 and got["cost"]["unpriced"] == 3, err
    assert [(e["model"], e["runs"]) for e in got["unpriced_models"]] == [("acme-model-9", 2), (None, 1)]
    assert "acme-model-9" in got["unpriced_models"][0]["reason"]
    code, out, err = _run(capsys, "adapt", "harness-runs", str(path), "--out", str(tmp_path / "o"))
    line = next(x for x in out.splitlines() if x.startswith("  unpriced:"))
    assert "2 acme-model-9 (" in line and "1 with no model (" in line and "loopmath prices" in line, line


def test_harness2_unpriced_rollouts_name_their_model(tmp_path, capsys):
    built = F.build(tmp_path / "runs", variants={"zero-cost"})
    code, out, err = _run(capsys, "adapt", "harness2", str(built.root), "--out", str(tmp_path / "o"), "--json")
    got = _one_json(out)
    assert code == 0 and got["cost"]["unpriced"] == len(built.rollouts), err
    assert [(e["model"], e["runs"]) for e in got["unpriced_models"]] == [("gemini-3.5-flash", len(built.rollouts))]
    code, out, err = _run(capsys, "adapt", "harness2", str(built.root), "--out", str(tmp_path / "o"))
    assert any(x.startswith(f"  unpriced: {len(built.rollouts)} gemini-3.5-flash (") for x in out.splitlines()), out


# N21: imported harness2 runs are the user's habit runs


def test_imported_harness2_runs_are_habit_runs_of_their_type_and_repo(tmp_path, capsys):
    built = F.build(tmp_path / "runs")
    home = tmp_path / "lm"
    assert _run(capsys, "adapt", "harness2", str(built.root), "--out", str(tmp_path / "o"))[0] == 0
    assert _run(capsys, "run", "import", str(tmp_path / "o"), "--finish", "--no-fit", "--home", str(home))[0] == 0
    rows = storeread.run_rows(home)
    assert rows and all(storeread.is_habit(r) and r["config"] for r in rows)
    started = max(storeread.parse_ts(r["started_at"]) for r in rows)
    cfg, level = storeread.usual_from_history(home, rows[0]["task_type"], rows[0]["repo"],
                                              now=started + dt.timedelta(days=1))
    assert (cfg, level) == (rows[0]["config"], "repo")
    # the usual counts habit runs of the last 90 days: the fixture's rollouts ran long before today
    assert storeread.usual_from_history(home, rows[0]["task_type"], rows[0]["repo"],
                                        now=started + dt.timedelta(days=storeread.HISTORY_DAYS + 1)) == (None, None)
