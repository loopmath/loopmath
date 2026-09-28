"""P3-23: `run import` says how many runs were new and how many were already in the store, and a
`--finish` import of runs that are already finished refits a store that has no fit."""

from __future__ import annotations

import json
import os

import pytest
from store_helpers import cli, fake_settle, finished_doc

from loopmath.store import Store
from loopmath.store import finish as finish_mod
from loopmath.store import fitjob
from loopmath.store import runs as R


@pytest.fixture
def spawned(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    return calls


def _store_files(tmp_path, *runs):
    """Run files as a store writes them: finished, with the store's own ext (what the tester copied)."""
    folder = tmp_path / "from-store"
    folder.mkdir()
    for run in runs:
        doc = finished_doc(run, usd=0.5)
        R.set_store_ext(doc, state=R.FINISHED)
        (folder / f"{run}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    return folder


def _with_fit(home):
    fit = home / "fits" / "fit_1"
    fit.mkdir(parents=True)
    os.symlink("fit_1", home / "fits" / "latest")
    assert Store(home).latest_fit() is not None


def test_reimport_counts_new_and_already_in_the_store(capsys, tmp_path, spawned):
    folder = _store_files(tmp_path, "run-a", "run-b")
    home = tmp_path / "lm"
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish")
    assert code == 0, err
    assert "2 runs imported (2 new), 0 failed; 0 finished, 2 already finished" in out.splitlines()
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish", "--json")
    assert code == 0, err
    assert (out["imported"], out["new"], out["already_in_store"]) == (2, 0, 2)
    assert [f["already_in_store"] for f in out["files"]] == [True, True]
    code, out, err = cli(capsys, home, "run", "import", str(folder))
    assert "2 runs imported (0 new, 2 already in the store and replaced), 0 failed" in out.splitlines()


def test_finish_of_already_finished_runs_refits_a_store_with_no_fit(capsys, tmp_path, spawned):
    """The tester's case: the runs were finished by another store, so nothing is finished now; the new
    store has runs and no fit, so one refit starts (0.2.3 left it at `fit: none yet`)."""
    folder = _store_files(tmp_path, "run-a", "run-b")
    home = tmp_path / "lm"
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish")
    assert code == 0, err
    assert out.splitlines()[-1] == "refit: started in the background (pid 0)" and spawned == [str(home)]


def test_a_pure_reimport_into_a_store_with_a_fit_does_not_refit_and_says_why(capsys, tmp_path, spawned):
    folder = _store_files(tmp_path, "run-a")
    home = tmp_path / "lm"
    cli(capsys, home, "run", "import", str(folder), "--finish", "--no-fit")
    _with_fit(home)
    code, out, err = cli(capsys, home, "run", "import", str(folder / "run-a.ocp.json"), "--finish", "--json")
    assert code == 0, err
    assert out["already_in_store"] is True and spawned == []
    assert out["fit"] == {"started": False, "reason": "every run was already in the store and finished, and the store has a fit"}
    code, out, err = cli(capsys, home, "run", "import", str(folder), "str-not-there", "--finish")
    assert "refit: not started (every run was already in the store and finished, and the store has a fit)" in out


def test_a_finished_run_new_to_a_store_with_a_fit_refits(capsys, tmp_path, spawned):
    home = tmp_path / "lm"
    _with_fit(home)
    folder = _store_files(tmp_path, "run-new")
    code, out, err = cli(capsys, home, "run", "import", str(folder / "run-new.ocp.json"), "--finish")
    assert code == 0, err
    assert out.splitlines()[0] == "run-new"
    assert out.splitlines()[-1] == "refit: started in the background (pid 0)" and spawned == [str(home)]
    assert "already finished by loopmath; stored as finished, --finish skipped" in err


def test_one_file_already_in_the_store_says_so(capsys, tmp_path, spawned):
    folder = _store_files(tmp_path, "run-a")
    home = tmp_path / "lm"
    cli(capsys, home, "run", "import", str(folder / "run-a.ocp.json"), "--no-fit")
    code, out, err = cli(capsys, home, "run", "import", str(folder / "run-a.ocp.json"), "--no-fit")
    assert code == 0, err
    assert out.splitlines()[0] == "run-a  (already in the store; replaced by this file)"
