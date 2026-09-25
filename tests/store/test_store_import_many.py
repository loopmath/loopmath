"""`run import` with several files or a directory, one refit at the end, and `--no-fit`."""

from __future__ import annotations

import json

import pytest
from store_helpers import cli, fake_settle, finished_doc

from loopmath.store import Store
from loopmath.store import finish as finish_mod
from loopmath.store import fitjob


@pytest.fixture
def spawned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


@pytest.fixture
def home(tmp_path, monkeypatch, spawned):
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    return tmp_path / "lm"


def _write(folder, *runs):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in runs:
        path = folder / f"{run}.ocp.json"
        path.write_text(json.dumps(finished_doc(run, usd=0.5)), encoding="utf-8")
        paths.append(path)
    return paths


def test_a_directory_with_finish_refits_once_at_the_end(capsys, home, tmp_path, spawned):
    folder = tmp_path / "ocp"
    _write(folder, "run-a", "run-b", "run-c")
    (folder / "MANIFEST.json").write_text("{}")  # only *.ocp.json files are read
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish")
    assert code == 0, err
    lines = out.splitlines()
    assert lines[-2:] == ["3 runs imported, 0 failed; 3 finished", "refit: started in the background (pid 0)"]
    assert {line.split()[0] for line in lines[:-2]} == {"run-a", "run-b", "run-c"}
    assert spawned == [str(home)]  # one refit for the whole call, not one per run
    rows = Store(home).index_rows()
    assert {r: rows[r]["state"] for r in rows} == {"run-a": "finished", "run-b": "finished", "run-c": "finished"}


def test_no_fit_starts_no_refit(capsys, home, tmp_path, spawned):
    paths = _write(tmp_path / "ocp", "run-a", "run-b")
    code, out, err = cli(capsys, home, "run", "import", *map(str, paths), "--finish", "--no-fit", "--json")
    assert code == 0, err
    assert spawned == []
    assert out["schema"] == "loopmath.run.import/1" and out["imported"] == 2 and out["failed"] == 0
    assert out["finished"] == 2 and out["fit"] == {"started": False, "reason": "--no-fit"}
    assert [f["run"] for f in out["files"]] == ["run-a", "run-b"] and all(f["ok"] for f in out["files"])
    assert out["runs"] == ["run-a", "run-b"]
    assert all(f["finished"]["fit_started"] is False for f in out["files"])


def test_one_file_with_no_fit_keeps_the_single_file_output(capsys, home, tmp_path, spawned):
    (path,) = _write(tmp_path / "ocp", "run-one")
    code, out, err = cli(capsys, home, "run", "import", str(path), "--finish", "--no-fit", "--json")
    assert code == 0, err
    assert spawned == []
    assert set(out) == {"schema", "run", "path", "migrated_from", "state", "finished",  # as in 0.1.0,
                        "shipped_overlap"}  # plus the shipped-source overlap count
    assert out["finished"]["fit"] == {"started": False, "reason": "--no-fit"}
    code, out, err = cli(capsys, home, "run", "import", str(path), "--finish")
    assert code == 0, err
    assert out.splitlines()[-1] == "refit: started in the background (pid 0)" and spawned == [str(home)]


def test_a_bad_file_is_reported_and_the_rest_still_import(capsys, home, tmp_path, spawned):
    good = _write(tmp_path / "ocp", "run-a", "run-b")
    bad = tmp_path / "ocp" / "bad.ocp.json"
    bad.write_text("{not json")
    future = tmp_path / "ocp" / "future.ocp.json"
    future.write_text(json.dumps({**finished_doc("run-f"), "ocp": "9.9"}))
    code, out, err = cli(capsys, home, "run", "import", str(good[0]), str(bad), str(future), str(good[1]), "--finish")
    assert code == 1, err
    lines = out.splitlines()
    assert lines[0].startswith(f"FAIL  {bad}: ") and "not JSON" in lines[0]
    assert lines[1].startswith(f"FAIL  {future}: ") and "0.3" in lines[1]
    assert lines[-2] == "2 runs imported, 2 failed; 2 finished"
    assert spawned == [str(home)]
    assert set(Store(home).index_rows()) == {"run-a", "run-b"}


def test_a_missing_file_or_empty_directory_exits_2(capsys, home, tmp_path, spawned):
    good = _write(tmp_path / "ocp", "run-a")
    empty = tmp_path / "empty"
    empty.mkdir()
    code, out, err = cli(capsys, home, "run", "import", str(good[0]), str(tmp_path / "none.ocp.json"), str(empty), "--json")
    assert code == 2
    assert out["imported"] == 1 and out["failed"] == 2
    assert [f["exit"] for f in out["files"] if not f["ok"]] == [2, 2]
    assert "no *.ocp.json files" in out["files"][2]["error"]
    # without --finish nothing is finished, so nothing refits
    assert spawned == [] and out["fit"]["started"] is False and "--finish" in out["fit"]["reason"]


def test_one_run_reads_in_the_singular(capsys, home, tmp_path, spawned):
    folder = tmp_path / "ocp"
    _write(folder, "run-a")
    code, out, err = cli(capsys, home, "run", "import", str(folder))
    assert code == 0, err
    assert "1 run imported, 0 failed" in out.splitlines()


def test_many_files_keep_the_summary_within_25_lines(capsys, home, tmp_path, spawned):
    folder = tmp_path / "ocp"
    _write(folder, *[f"run-{i:02d}" for i in range(30)])
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit")
    assert code == 0, err
    assert out.splitlines() == ["30 runs imported, 0 failed; 30 finished", "refit: not started (--no-fit)"]


BRIEF_KEYS = {"schema", "imported", "failed", "finished", "already_finished", "failures", "more_failures",
              "shipped_overlap", "overlap_note", "fit", "next"}


def test_brief_is_counts_failures_overlap_and_the_next_step(capsys, home, tmp_path, spawned):
    good = _write(tmp_path / "ocp", "run-a", "run-b")
    bad = tmp_path / "ocp" / "bad.ocp.json"
    bad.write_text("{not json")
    code, out, err = cli(capsys, home, "run", "import", str(tmp_path / "ocp"), "--finish", "--no-fit", "--json",
                         "--brief")
    assert code == 1, err  # a failed file still exits 1, as without --brief
    assert set(out) == BRIEF_KEYS and out["schema"] == "loopmath.run.import/1"
    assert (out["imported"], out["failed"], out["finished"], out["already_finished"]) == (2, 1, 2, 0)
    assert len(out["failures"]) == 1 and out["failures"][0]["file"] == str(bad)
    assert set(out["failures"][0]) == {"file", "error"} and "not JSON" in out["failures"][0]["error"]
    assert out["more_failures"] == 0 and out["shipped_overlap"] == {} and out["overlap_note"] is None
    assert out["fit"] == {"started": False, "reason": "--no-fit"} and out["next"] == "loopmath fit --json"
    assert spawned == [] and set(Store(home).index_rows()) == {"run-a", "run-b"} and len(good) == 2


def test_brief_stays_short_for_many_files_and_takes_one_file_too(capsys, home, tmp_path, spawned):
    folder = tmp_path / "ocp"
    _write(folder, *[f"run-{i:02d}" for i in range(30)])
    for i in range(12):
        (folder / f"bad-{i:02d}.ocp.json").write_text("{not json")
    capsys.readouterr()
    from store_helpers import main
    code = main(["run", "import", str(folder), "--finish", "--no-fit", "--json", "--brief", "--home", str(home)])
    text = capsys.readouterr().out
    assert code == 1 and len(text.splitlines()) < 70  # without --brief: every file with its run and cost
    out = json.loads(text)
    assert out["imported"] == 30 and out["failed"] == 12 and len(out["failures"]) == 10 and out["more_failures"] == 2
    (one,) = _write(tmp_path / "one", "run-one")
    code, out, err = cli(capsys, home, "run", "import", str(one), "--finish", "--json", "--brief")
    assert code == 0, err
    assert set(out) == BRIEF_KEYS and out["imported"] == 1 and out["finished"] == 1
    assert out["fit"]["started"] is True and out["next"].startswith("loopmath posterior --html")


def test_brief_says_when_runs_stay_open_or_nothing_imported(capsys, home, tmp_path, spawned):
    paths = _write(tmp_path / "ocp", "run-a")
    code, out, err = cli(capsys, home, "run", "import", str(paths[0]), str(tmp_path / "none.ocp.json"), "--json",
                         "--brief")
    assert code == 2 and out["imported"] == 1 and out["failed"] == 1
    assert out["next"] == "1 run stored open: finish each with loopmath run finish --run RUN --json"
    code, out, err = cli(capsys, home, "run", "import", str(tmp_path / "none.ocp.json"), "--json", "--brief")
    assert code == 2 and out["imported"] == 0 and out["next"].startswith("nothing was imported")
