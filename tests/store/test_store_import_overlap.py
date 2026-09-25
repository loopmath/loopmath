"""`run import` of several files says once how many of them a shipped source also holds."""

from __future__ import annotations

import json

import pytest
from store_helpers import cli, fake_settle, finished_doc

from loopmath.priors import shipped_ids
from loopmath.store import finish as finish_mod
from loopmath.store import fitjob


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    return tmp_path / "lm"


def test_a_batch_import_names_the_shipped_overlap_once(capsys, home, tmp_path):
    shipped = sorted(run for run, source in shipped_ids().items() if source == "rq1")[:2]
    assert len(shipped) == 2
    folder = tmp_path / "ocp"
    folder.mkdir()
    for run in [*shipped, "run-mine"]:
        (folder / f"{run}.ocp.json").write_text(json.dumps(finished_doc(run, usd=0.5)), encoding="utf-8")
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish")
    assert code == 0, err
    note = "2 of your runs are also in the shipped rq1 prior (same run ids): fits use your copies"
    assert out.splitlines()[-1] == note and (out + err).count("shipped") == 1
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--no-fit", "--json")
    assert code == 0, err
    assert out["runs"] == sorted([*shipped, "run-mine"]) and out["shipped_overlap"] == {"rq1": 2}


def test_a_batch_import_with_no_shipped_run_prints_no_note(capsys, home, tmp_path):
    folder = tmp_path / "ocp"
    folder.mkdir()
    for run in ("run-a", "run-b"):
        (folder / f"{run}.ocp.json").write_text(json.dumps(finished_doc(run, usd=0.5)), encoding="utf-8")
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--finish")
    assert code == 0, err
    assert "shipped" not in out + err
    code, out, err = cli(capsys, home, "run", "import", str(folder), "--no-fit", "--json")
    assert out["shipped_overlap"] == {}
