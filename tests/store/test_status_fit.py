"""`loopmath status` names the current fit's options and run counts (FINDINGS I8, spec 02)."""

from __future__ import annotations

import json
import os
from datetime import datetime

from loopmath import cli
from loopmath.belief.design import DESIGN_VERSION
from loopmath.store.status import fit_options_text, fit_summary


def _fit(home, fit_id, meta):
    folder = home / "fits" / fit_id
    folder.mkdir(parents=True)
    meta = {"fit": fit_id, "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "design_version": DESIGN_VERSION, **meta}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    link = home / "fits" / "latest"
    if link.is_symlink():
        link.unlink()
    os.symlink(fit_id, link)


def _status(capsys, home, *extra):
    code = cli.main(["status", "--home", str(home), *extra])
    return code, capsys.readouterr().out


def test_status_names_the_options_and_counts(tmp_path, capsys):
    _fit(tmp_path, "fit_20260924170000", {
        "options": {"no_prior": False, "without": ["rq1"], "full": False},
        "runs_by_source": {"e0": 809, "sweep": 660, "user": 44}, "n_runs": {"prior": 1469, "user": 44},
        "shipped_overlap": {"rq1": 44}, "user_labels": {"rq1": 44}})
    code, out = _status(capsys, tmp_path)
    assert code == 0
    assert "fit: fit_20260924170000, 0 min old, --without rq1, runs 1469 prior + 44 yours\n" in out
    assert out.count("44 of your runs are also in the shipped rq1 prior (same runs): fits use your copies") == 1
    code, out = _status(capsys, tmp_path, "--json")
    fit = json.loads(out)["fit"]
    assert fit["options"] == {"no_prior": False, "without": ["rq1"], "full": False}
    assert fit["runs"] == {"prior": 1469, "user": 44, "shared": 0} and fit["shipped_overlap"] == {"rq1": 44}


def test_status_shows_shared_imports_apart_and_no_prior(tmp_path, capsys):
    _fit(tmp_path, "fit_20260924170100", {
        "options": {"no_prior": True, "without": [], "full": False},
        "runs_by_source": {"user": 22, "shared:acme": 5, "shared:beta": 2}, "n_runs": {"prior": 7, "user": 22}})
    _, out = _status(capsys, tmp_path)
    assert "fit: fit_20260924170100, 0 min old, --no-prior, runs 0 prior + 22 yours + 7 shared\n" in out
    assert "also in the shipped" not in out


def test_status_on_a_fit_from_before_the_options_were_recorded(tmp_path, capsys):
    _fit(tmp_path, "fit_20260924170200", {})
    _, out = _status(capsys, tmp_path)
    assert "fit: fit_20260924170200, 0 min old, runs 0 prior + 0 yours\n" in out
    (tmp_path / "fits" / "fit_20260924170200" / "meta.json").write_text("not json")
    assert fit_summary(tmp_path, "fit_20260924170200") == {}
    assert fit_summary(tmp_path, None) == {}


def test_options_text():
    assert fit_options_text({}) == ""
    assert fit_options_text({"no_prior": True, "without": ["rq1", "benchmark"], "full": True}) == \
        "--no-prior --without rq1 --without benchmark --full"
