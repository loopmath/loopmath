"""`--fit ID` on recommend and posterior: read a kept fit instead of `fits/latest` (FINDINGS I9, D91)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
import simdata

from loopmath import cli
from loopmath.belief import fit as F

NOW = datetime.fromisoformat("2026-09-24T17:00:00-07:00")


@pytest.fixture(scope="module")
def two_fits(tmp_path_factory):
    """A store of 60 runs and two fits of it: the older one on 40 runs, the newer on all 60."""
    home = tmp_path_factory.mktemp("select") / "home"
    docs, _ = simdata.simulate(60, seed=41, source="live", prefix="run_sel")
    (home / "runs").mkdir(parents=True)
    for doc in docs:
        (home / "runs" / f"{doc['run']['id']}.ocp.json").write_text(json.dumps(doc), encoding="utf-8")
    old = F.fit(home, docs=docs[:40], no_prior=True, now=NOW).name
    new = F.fit(home, docs=docs, no_prior=True, now=NOW + timedelta(minutes=5)).name
    return home, old, new


def test_kept_fits_and_the_folder_of_one(two_fits):
    home, old, new = two_fits
    assert F.kept_fits(home) == [old, new]
    assert F.fit_folder(home, old) == home / "fits" / old
    assert F.fit_folder(home, "latest").name == new
    assert F.load_fit(home, old).fit_id == old
    with pytest.raises(F.UnknownFit, match=f"no fit fit_nope in .*; kept fits, newest last: {old}, {new}"):
        F.fit_folder(home, "fit_nope")
    with pytest.raises(F.UnknownFit):
        F.fit_folder(home, "../fits/" + old)
    with pytest.raises(F.UnknownFit, match="none yet"):
        F.fit_folder(home / "empty", old)


def _json(capsys, argv):
    assert cli.main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_recommend_reads_the_fit_it_is_given(two_fits, capsys):
    home, old, new = two_fits
    base = ["recommend", "--home", str(home), "--type", "feature", "--repo", "acme/api", "--json"]
    assert _json(capsys, base)["fit"]["id"] == new
    picked = _json(capsys, base + ["--fit", old])["fit"]
    assert picked["id"] == old and picked["n_runs"] == {"prior": 0, "user": 40}
    assert cli.main(base + ["--fit", "fit_nope"]) == 2
    assert f"kept fits, newest last: {old}, {new}" in capsys.readouterr().err


def test_posterior_reads_the_fit_it_is_given(two_fits, capsys):
    home, old, new = two_fits
    base = ["posterior", "--home", str(home), "--level", "model", "--json"]
    assert _json(capsys, base)["fit"]["id"] == new
    assert _json(capsys, base + ["--fit", old])["fit"]["id"] == old
    assert cli.main(base + ["--fit", "fit_nope"]) == 2
    assert "no fit fit_nope" in capsys.readouterr().err


def test_both_commands_document_the_flag(capsys):
    for command in ("recommend", "posterior"):
        with pytest.raises(SystemExit):
            cli.main([command, "--help"])
        assert "--fit ID" in capsys.readouterr().out
