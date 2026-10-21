"""A fit written under other node names is never read (spec 04 section 4, `design_version`)."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief import state as S
from loopmath.belief.design import DESIGN_VERSION, OPT_IN_DESIGN_VERSION, other_design
from loopmath.belief.fit import UnknownFit, load_fit
from loopmath.belief.state import load_latest
from loopmath.store.config import CHOICES


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
    meta = json.loads((S.fit_dir(home) / "meta.json").read_text())
    assert meta["design_version"] == DESIGN_VERSION == 4
    assert "model_options" not in meta and other_design(meta) is None
    assert load_latest(home) is not None


def test_a_version_2_fit_without_the_reference_horizon_is_refused_in_the_same_line(home, capsys, monkeypatch):
    """Fits made with the first horizon coding, before the reference horizon, code log2h from 1 h and have no
    `horizons.reference_s`: design version 2, refused like any older fit (HORIZON-CHECK)."""
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


# ---------------------------------------------------------------- design version 5: a fit with model options

NOW = datetime.fromisoformat("2026-09-24T12:00:00-07:00")
SUCCESS_HEAD = "belief.success_head"


def _opt_in(options, fit_id="fit_x"):
    return {"fit": fit_id, "design_version": OPT_IN_DESIGN_VERSION, "model_options": options}


def _set_options(home, options):
    """The latest fit's meta as a fit with these model options writes it (design version 5)."""
    path = S.fit_dir(home) / "meta.json"
    meta = json.loads(path.read_text())
    path.write_text(json.dumps({**meta, "design_version": OPT_IN_DESIGN_VERSION, "model_options": options}))
    return meta["fit"]


@pytest.mark.parametrize("options", [{"prior.version": 2}, {SUCCESS_HEAD: "sampled", "prior.version": 2}])
def test_a_fit_with_a_model_option_this_loopmath_does_not_know_is_refused(home, capsys, monkeypatch, options):
    """A newer loopmath may write a model option this one has no nodes for. Read without them, the fit would give
    other numbers than its writer's, so every reader refuses it by the option's name and asks for a new fit."""
    import time

    from loopmath.recommend.commands import EXIT_NO_FIT, load_belief
    from loopmath.skill.doctor import check_fit
    from loopmath.store.status import fit_summary

    monkeypatch.setattr(S, "_NOTED", set())
    fit_id = _set_options(home, options)
    line = f"fit {fit_id} is from design version 5 (made by another loopmath) with the model option `prior.version`,"
    why = other_design(json.loads((S.fit_dir(home) / "meta.json").read_text()))
    assert why is not None and why.startswith(line) and why.endswith("run `loopmath fit` for a new fit")
    assert SUCCESS_HEAD not in why  # only the option it does not know is named
    assert S.latest_problem(home) == why
    assert load_latest(home) is None
    assert capsys.readouterr().err == f"loopmath: {why}\n"
    with pytest.raises(UnknownFit, match="`prior.version`"):
        load_fit(home, fit_id)
    monkeypatch.setattr(S, "_NOTED", set())
    belief, code = load_belief(home)  # what `recommend` reads
    assert belief is None and code == EXIT_NO_FIT
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("error: ") and line in err and "no fit yet" not in err
    assert fit_summary(home, fit_id)["problem"] == why  # `status`
    check = check_fit(home, time.time())  # `doctor`
    assert check["status"] == "warn" and check["detail"]["usable"] is False and check["summary"] == why


def test_every_option_this_loopmath_does_not_know_is_named():
    why = other_design(_opt_in({"prior.version": 2, SUCCESS_HEAD: "sampled", "a.b": True}))
    assert why is not None
    assert "with the model options `a.b`, `prior.version`, which this loopmath does not know;" in why


@pytest.mark.parametrize("value", ["full", "integrated", "eb", "", 2, True, None, ["sampled"], {"sampled": 1}])
def test_a_value_this_loopmath_does_not_write_is_refused(value):
    """Only the values this loopmath's `fit` writes are read: `sampled` (`eb` and unset write design version 4
    with no options, and `integrated` was never written by a release). Any JSON value is named, never a TypeError."""
    why = other_design(_opt_in({SUCCESS_HEAD: value}))
    assert why == (f"fit fit_x is from design version 5 (made by another loopmath) with the model option "
                   f"`{SUCCESS_HEAD}` = {json.dumps(value)}, a value this loopmath does not know; "
                   "run `loopmath fit` for a new fit")


def test_a_fit_with_an_unknown_value_is_not_loaded(home, capsys, monkeypatch):
    monkeypatch.setattr(S, "_NOTED", set())
    fit_id = _set_options(home, {SUCCESS_HEAD: "full"})
    assert load_latest(home) is None
    assert f'`{SUCCESS_HEAD}` = "full", a value this loopmath does not know' in capsys.readouterr().err
    with pytest.raises(UnknownFit, match="a value this loopmath does not know"):
        load_fit(home, fit_id)


def test_an_empty_options_block_is_refused():
    """`fit` writes design version 5 only with an option set: an empty block is not a fit this loopmath wrote."""
    assert other_design(_opt_in({})) == (
        "fit fit_x is from design version 5 (made by another loopmath) with no model options, which this loopmath "
        "does not write; run `loopmath fit` for a new fit")


def test_every_opt_in_value_config_takes_is_read():
    """Each value `config set` takes that makes `fit` write design version 5 is one this loopmath reads back."""
    for value in CHOICES[SUCCESS_HEAD]:
        why = other_design(_opt_in({SUCCESS_HEAD: value}))
        assert (why is None) == (value != "eb"), value


@pytest.mark.parametrize("meta", [{"design_version": 5}, {"design_version": 5, "model_options": "sampled"},
                                  {"design_version": 5, "model_options": None}])
def test_version_5_without_an_options_block_keeps_its_refusal(meta):
    assert other_design({"fit": "fit_x", **meta}) == (
        "fit fit_x is from design version 5 (made by another loopmath) and this loopmath reads version 4; "
        "run `loopmath fit` for a new fit")


def test_the_check_stays_light():
    """`status` and `doctor` call `other_design` without the engine: no numpy and no fitting module."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import loopmath

    code = ("import sys\nfrom loopmath.belief.design import other_design\n"
            "assert other_design({'design_version': 5, 'model_options': {'prior.version': 2}})\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('numpy', 'scipy')"
            " or m.startswith(('loopmath.belief.', 'loopmath.store'))))")
    env = {**os.environ, "PYTHONPATH": str(Path(loopmath.__file__).resolve().parents[1])}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
    assert json.loads(out.replace("'", '"')) == ["loopmath.belief.design", "loopmath.belief.forest",
                                                  "loopmath.belief.pricing"]


def test_a_sampled_fit_still_loads(tmp_path, capsys, monkeypatch):
    """The fit the opt-in writes (`belief.success_head = "sampled"`) is read by every reader, as before."""
    import time

    from loopmath.skill.doctor import check_fit
    from loopmath.store.status import fit_summary

    monkeypatch.setattr(S, "_NOTED", set())
    (tmp_path / "config.toml").write_text('[belief]\nsuccess_head = "sampled"\n', encoding="utf-8")
    docs, _ = simdata.simulate(60, seed=2, source="live", n_tasks=20)
    path = F.fit(tmp_path, docs=docs, no_prior=True, now=NOW)
    meta = json.loads((path / "meta.json").read_text())
    assert meta["design_version"] == OPT_IN_DESIGN_VERSION and meta["model_options"] == {SUCCESS_HEAD: "sampled"}
    assert other_design(meta) is None and S.latest_problem(tmp_path) is None
    assert load_latest(tmp_path).fit_id == load_fit(tmp_path, meta["fit"]).fit_id == meta["fit"]
    assert fit_summary(tmp_path, meta["fit"])["problem"] is None
    assert check_fit(tmp_path, time.time())["status"] == "ok"
    assert "design version" not in capsys.readouterr().err
