"""`loopmath tasks pick`: the next tasks for a candidate, its suite score so far, and stop or continue.

It reads the store's finished runs, or `loopmath.harness-runs/0.1` rows with `--runs`. From a store it writes
each next task's file under `<home>/tasks-pick/` and prints the `run record` line that records the run on that
task with the candidate's label, so the next `tasks pick` reads it. Synthetic runs, faked log settling.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from loopmath import cli
from loopmath.store import finish as finish_mod

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "store"))
from store_helpers import fake_settle  # noqa: E402

SOLO = ["--workflow", "solo", "--set", "implement=claude-code:claude-opus-5:high", "--source", "designed"]
SCHEMA = "loopmath.harness-runs/0.1"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("LOOPMATH_HOME", str(h))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "logs"))
    (tmp_path / "wt").mkdir()
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle(usd=0.5))
    h.mkdir()
    return h


def lm(capsys, *argv):
    capsys.readouterr()
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, (json.loads(out) if "--json" in argv and out.strip() else out), err


def history_rows(arms=6, tasks=20, seed=5):
    """Each history arm runs every task twice; the task's level and the arm's level set the chance."""
    rng = np.random.default_rng(seed)
    b = np.linspace(-2.5, 2.5, tasks)
    rows = []
    for a in range(arms):
        th = -1.0 + 2.0 * a / max(arms - 1, 1)
        for t in range(tasks):
            for r in range(2):
                ok = bool(rng.random() < 1 / (1 + np.exp(-(th - b[t]))))
                rows.append({"schema": SCHEMA, "id": f"h:{a}:{t}:{r}", "task_id": f"task-{t:02d}",
                             "task": {"type": "bug_fix", "repo": "acme/app", "title": f"task {t}"},
                             "arm": f"arm{a}", "status": "done", "outcome": {"accepted": ok, "by": "test"}})
    return rows


def write_rows(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_the_command_is_registered(capsys):
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main(["tasks", "--help"])
    assert exc.value.code == 0
    assert "pick" in capsys.readouterr().out


def test_pick_from_a_runs_file_gives_next_tasks_an_estimate_and_a_decision(tmp_path, capsys):
    rows = history_rows()
    for t, ok in (("task-03", True), ("task-10", True), ("task-15", False)):
        rows.append({"schema": SCHEMA, "id": f"c:{t}", "task_id": t, "arm": "new", "status": "done",
                     "outcome": {"accepted": ok, "by": "test"}})
    f = write_rows(tmp_path / "runs.jsonl", rows)
    code, out, err = lm(capsys, "tasks", "pick", "--runs", str(f), "--candidate", "arm=new", "--n", "3", "--json")
    assert code == 0, err
    assert out["schema"] == "loopmath.tasks-pick/1"
    assert out["suite"]["tasks"] == 20 and out["suite"]["history_arms"] == 6
    assert out["candidate"]["sessions"] == 3
    est = out["estimate"]
    assert est["range80"][0] <= est["score"] <= est["range80"][1]
    assert est["range95"][0] <= est["range80"][0] and est["range95"][1] >= est["range80"][1]
    assert out["decision"]["action"] in ("stop", "continue") and out["decision"]["reason"]
    names = [n["task"] for n in out["next"]]
    assert len(names) == 3 and not {"task-03", "task-10", "task-15"} & set(names)
    assert all(0.0 <= n["p"] <= 1.0 for n in out["next"])
    code, text, err = lm(capsys, "tasks", "pick", "--runs", str(f), "--candidate", "arm=new")
    assert code == 0, err
    assert "continue" in text or "stop" in text


def test_pick_refuses_a_candidate_with_no_history_to_learn_from_and_names_the_arms(tmp_path, capsys):
    f = write_rows(tmp_path / "runs.jsonl", [r for r in history_rows() if r["arm"] == "arm0"])
    code, _, err = lm(capsys, "tasks", "pick", "--runs", str(f), "--candidate", "arm=arm0", "--json")
    assert code == 1 and "history" in err


def test_the_store_round_trip_through_run_record(home, tmp_path, capsys):
    rows = write_rows(tmp_path / "hist.jsonl", history_rows())
    assert lm(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "ocp"))[0] == 0
    code, _, err = lm(capsys, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json")
    assert code == 0, err
    code, out, err = lm(capsys, "tasks", "pick", "--candidate", "config=new", "--json")
    assert code == 0, err
    assert out["candidate"]["sessions"] == 0 and out["decision"]["action"] == "continue"
    first = out["next"][0]
    task_file = Path(first["task_file"])
    assert task_file.is_file() and task_file.parent == home / "tasks-pick"
    assert json.loads(task_file.read_text())["id"] == first["task"]
    assert "run record" in first["record"] and "--label config=new" in first["record"]
    code, _, err = lm(capsys, "run", "record", "--task-file", str(task_file), *SOLO, "--label", "config=new",
                      "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h", "--verified", "tests=pass",
                      "--no-commits", "--no-fit", "--json")
    assert code == 0, err
    code, out, err = lm(capsys, "tasks", "pick", "--candidate", "config=new", "--json")
    assert code == 0, err
    assert out["candidate"]["sessions"] == 1 and out["candidate"]["passes"] == 1
    assert first["task"] not in [n["task"] for n in out["next"]]


def test_every_estimate_warns_where_the_ranges_fail_in_json_text_and_help(tmp_path, capsys):
    rows = history_rows()
    rows.append({"schema": SCHEMA, "id": "c:1", "task_id": "task-05", "arm": "new", "status": "done",
                 "outcome": {"accepted": True, "by": "test"}})
    f = write_rows(tmp_path / "runs.jsonl", rows)
    code, out, err = lm(capsys, "tasks", "pick", "--runs", str(f), "--candidate", "arm=new", "--json")
    assert code == 0, err
    warnings = out["estimate"]["warnings"]
    assert len(warnings) == 2
    assert "unlike every history arm" in warnings[0] and "--method random or stratified" in warnings[0]
    assert "below every history arm" in warnings[1] and "too high by every method" in warnings[1]
    code, text, err = lm(capsys, "tasks", "pick", "--runs", str(f), "--candidate", "arm=new")
    assert code == 0, err
    lines = text.splitlines()
    score = next(i for i, line in enumerate(lines) if line.startswith("score "))
    assert [line for line in lines if line.startswith("warning: ")] == [f"warning: {w}" for w in warnings]
    assert lines[score + 1].startswith("warning: ")  # next to the estimate
    with pytest.raises(SystemExit):
        cli.main(["tasks", "pick", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "unlike all of them" in help_text and "below every history arm" in help_text


def test_the_record_line_names_the_chosen_store_and_records_there(home, tmp_path, capsys):
    chosen = tmp_path / "chosen store"
    rows = write_rows(tmp_path / "hist.jsonl", history_rows())
    assert lm(capsys, "adapt", "harness-runs", str(rows), "--out", str(tmp_path / "ocp"))[0] == 0
    code, _, err = lm(capsys, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json",
                      "--home", str(chosen))
    assert code == 0, err
    code, out, err = lm(capsys, "tasks", "pick", "--candidate", "config=new", "--home", str(chosen), "--json")
    assert code == 0, err
    first = out["next"][0]
    assert Path(first["task_file"]).parent == chosen / "tasks-pick"
    argv = shlex.split(first["record"])
    assert argv[:3] == ["loopmath", "run", "record"] and argv[argv.index("--home") + 1] == str(chosen)
    code, _, err = lm(capsys, *argv[1:], *SOLO, "--cwd", f"implement={tmp_path / 'wt'}", "--since", "1h",
                      "--verified", "tests=pass", "--no-commits", "--no-fit", "--json")
    assert code == 0, err
    code, out, err = lm(capsys, "tasks", "pick", "--candidate", "config=new", "--home", str(chosen), "--json")
    assert code == 0, err
    assert out["candidate"]["sessions"] == 1 and out["candidate"]["passes"] == 1
    default_runs = home / "runs"  # LOOPMATH_HOME's store got nothing
    assert not default_runs.exists() or not any(default_runs.iterdir())
