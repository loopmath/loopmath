"""A fit written under other node names is never read (spec 04 section 4, `design_version`; lane 2B)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief import state as S
from loopmath.belief.design import DESIGN_VERSION
from loopmath.belief.fit import UnknownFit, load_fit
from loopmath.belief.state import load_latest


@pytest.fixture()
def home(tmp_path):
    docs, _ = simdata.simulate(60, seed=2, source="live", n_tasks=20)
    F.fit(tmp_path, docs=docs, no_prior=True, now=datetime.fromisoformat("2026-09-24T12:00:00-07:00"))
    return tmp_path


def _set_version(home, version):
    path = S.fit_dir(home) / "meta.json"
    meta = json.loads(path.read_text())
    if version is None:
        meta.pop("design_version")
    else:
        meta["design_version"] = version
    path.write_text(json.dumps(meta))
    return meta["fit"]


def test_a_new_fit_records_the_design_version(home):
    assert json.loads((S.fit_dir(home) / "meta.json").read_text())["design_version"] == DESIGN_VERSION == 3
    assert load_latest(home) is not None


def test_a_version_2_fit_without_the_reference_horizon_is_refused_in_the_same_line(home, capsys, monkeypatch):
    """Fits made between 2C (422873a) and the reference horizon code log2h from 1 h and have no
    `horizons.reference_s`: design version 2, refused like any older fit (lane 2C, HORIZON-CHECK)."""
    from loopmath.cli import main
    from loopmath.output import EXIT_NO_FIT

    monkeypatch.setattr(S, "_NOTED", set())
    path = S.fit_dir(home) / "meta.json"
    meta = json.loads(path.read_text())
    assert meta["horizons"]["reference_s"] is None  # written by every fit, None without timeboxed runs
    del meta["horizons"]["reference_s"]
    path.write_text(json.dumps({**meta, "design_version": 2}))
    assert main(["posterior", "--home", str(home)]) == EXIT_NO_FIT
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "no fit yet" not in err
    assert f"fit {meta['fit']} is from design version 2 (made by an older loopmath)" in err
    assert "run `loopmath fit`" in err


@pytest.mark.parametrize("version", [None, 1])
def test_a_fit_of_another_design_version_is_not_loaded(home, capsys, monkeypatch, version):
    monkeypatch.setattr(S, "_NOTED", set())
    fit_id = _set_version(home, version)
    assert load_latest(home) is None
    assert load_latest(home) is None
    err = capsys.readouterr().err
    assert err.count("\n") == 1  # once per process
    assert f"fit {fit_id} is from design version 1" in err and "run `loopmath fit`" in err
    with pytest.raises(UnknownFit, match="design version 1"):
        load_fit(home, fit_id)
    _set_version(home, DESIGN_VERSION)
    assert load_latest(home).fit_id == load_fit(home, fit_id).fit_id == fit_id


def _one_line(err: str, fit_id: str) -> None:
    assert err.count("\n") == 1 and err.startswith("error: ") and "no fit yet" not in err
    assert f"fit {fit_id} is from design version 1 (made by an older loopmath)" in err and "run `loopmath fit`" in err


def test_recommend_asks_for_a_fit_in_one_line(home, capsys, monkeypatch):
    from loopmath.recommend.commands import EXIT_NO_FIT, load_belief

    monkeypatch.setattr(S, "_NOTED", set())
    fit_id = _set_version(home, 1)
    belief, code = load_belief(home)
    assert belief is None and code == EXIT_NO_FIT
    _one_line(capsys.readouterr().err, fit_id)


def test_posterior_asks_for_a_fit_in_one_line(home, capsys, monkeypatch):
    from loopmath.cli import main
    from loopmath.output import EXIT_NO_FIT

    monkeypatch.setattr(S, "_NOTED", set())
    fit_id = _set_version(home, 1)
    assert main(["posterior", "--home", str(home)]) == EXIT_NO_FIT
    out, err = capsys.readouterr()
    assert out == ""
    _one_line(err, fit_id)


def test_status_says_the_fit_is_not_usable(home, capsys):
    from loopmath.cli import main

    fit_id = _set_version(home, 1)
    assert main(["status", "--home", str(home), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    fit = payload["fit"]
    assert fit["latest"] == fit_id and fit["usable"] is False and "is from design version 1" in fit["problem"]
    assert payload["ok"] is False and [h["code"] for h in payload["health"]] == ["fit_unusable"]
    assert main(["status", "--home", str(home)]) == 0
    out = capsys.readouterr().out
    assert f"fit: {fit_id}," in out and ", not usable" in out and "run `loopmath fit`" in out
    assert "health: ok" not in out
    _set_version(home, DESIGN_VERSION)
    assert main(["status", "--home", str(home), "--json"]) == 0
    fit = json.loads(capsys.readouterr().out)["fit"]
    assert fit["usable"] is True and fit["problem"] is None


def test_doctor_warns_that_the_fit_is_not_usable(home):
    import time

    from loopmath.skill.doctor import check_fit

    fit_id = _set_version(home, 1)
    check = check_fit(home, time.time())
    assert check["status"] == "warn" and check["detail"]["usable"] is False and check["detail"]["fit"] == fit_id
    assert "is from design version 1" in check["summary"] and "run `loopmath fit`" in check["summary"]
    _set_version(home, DESIGN_VERSION)
    check = check_fit(home, time.time())
    assert check["status"] == "ok" and check["detail"]["usable"] is True
