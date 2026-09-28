"""Research verbs for a new user (new-user test P2-19): out of the main help, into `loopmath research
--help`, and a clean error, not a traceback, on a folder without sweep data."""

from __future__ import annotations

import pytest

import loopmath.fit
from loopmath.cli import main


def _help(capsys, *argv) -> str:
    with pytest.raises(SystemExit) as stop:
        main([*argv, "--help"])
    assert stop.value.code == 0
    return capsys.readouterr().out


def test_research_verbs_are_under_research_help(capsys):
    top = _help(capsys)
    assert "analyze-e0" not in top and "transfer-test" not in top and "loopmath research --help" in top
    research = _help(capsys, "research")
    for verb in ("fit", "transfer-test", "analyze-e0"):
        assert verb in research, verb
    assert "not part of the package" in research


@pytest.fixture
def no_pymc(monkeypatch):
    """The sweep check runs before any sampling, so the bayes extra is not needed to reach it."""
    monkeypatch.setattr(loopmath.fit, "require_bayes", lambda: None)


@pytest.mark.parametrize("verb", ["fit", "transfer-test"])
def test_a_folder_without_sweep_data_is_a_clean_error(verb, tmp_path, capsys, no_pymc):
    empty = tmp_path / "out"
    empty.mkdir()
    (empty / "notes.txt").write_text("not a run record\n")
    assert main(["research", verb, "--sweep-dir", str(empty)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"error: no sweep run records in {empty}") and "Traceback" not in err
    assert main(["research", verb, "--sweep-dir", str(tmp_path / "missing")]) == 1
    assert f"the sweep folder {tmp_path / 'missing'} does not exist" in capsys.readouterr().err


@pytest.fixture
def a_sweep(tmp_path, monkeypatch, no_pymc):
    """A sweep folder whose records assemble to three runs (the table is stubbed; nothing samples)."""
    import pandas as pd

    import loopmath.fit_assembly

    cells = [("gpt-5.6-luna", "low"), ("gpt-5.6-sol", "medium"), ("claude-opus-5", "high")]
    df = pd.DataFrame([{"model": m, "effort": e, "task": "t1", "task_group": "t1"} for m, e in cells])
    monkeypatch.setattr(loopmath.fit_assembly, "assemble_table", lambda **kw: df)
    (tmp_path / "sweep").mkdir()
    return tmp_path / "sweep"


@pytest.mark.parametrize("verb", ["fit", "transfer-test"])
def test_a_bad_mask_is_a_clean_error(verb, a_sweep, capsys):
    assert main(["research", verb, "--sweep-dir", str(a_sweep), "--observe", "nosuch:low"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: unknown model 'nosuch' in mask spec") and "Traceback" not in err
    assert main(["research", verb, "--sweep-dir", str(a_sweep), "--observe", "fable:low"]) == 1
    assert capsys.readouterr().err == ("error: the mask selected 0 of 3 runs to observe (--observe 'fable:low', "
                                       "--holdout 'rest'); check it against the models and efforts in this sweep\n")


@pytest.mark.parametrize("verb", ["fit", "transfer-test"])
def test_an_error_inside_the_fit_keeps_its_traceback(verb, a_sweep, monkeypatch):
    """Only the sweep folder and the mask are the user's to fix; a ValueError in the model code is a bug."""
    def bug(**kw):
        raise ValueError("a bug in the model code")

    monkeypatch.setattr(loopmath.fit, "run_fit", bug)
    with pytest.raises(ValueError, match="a bug in the model code"):
        main(["research", verb, "--sweep-dir", str(a_sweep)])


def test_analyze_e0_is_still_callable_at_the_top(capsys):
    help_text = _help(capsys, "analyze-e0")
    assert "e0" in help_text
