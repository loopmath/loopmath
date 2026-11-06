"""`loopmath outcome --observed-at`, of a signal and of a `--slate` preference: read as `parse_ts` reads it and stored
in UTC with a `Z`, OCP 0.4's timestamp, the form `flow score` stores. A time with no offset is local time and no
time is now, as before. A time past what a UTC time holds is refused by name, exit 1, and nothing is written."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from loopmath.store import fitjob
from loopmath.store.home import Store
from loopmath.store.ids import now_iso, parse_ts
from tests.store.store_helpers import EXPLORE_CFG, REC_ID, cli, install_rec

OBSERVED = {"20261104T093000Z": "2026-11-04T09:30:00Z", "2026-W45-3T09:30:00+00:00": "2026-11-04T09:30:00Z",
            "2026-11-04 09:30:00+00:00": "2026-11-04T09:30:00Z", "2026-11-04T09:30:00+0000": "2026-11-04T09:30:00Z",
            "2026-11-04T09:30Z": "2026-11-04T09:30:00Z", "2026-11-04T09:30:00,5+00:00": "2026-11-04T09:30:00.500Z",
            "2026-11-04T09+00:00": "2026-11-04T09:00:00Z", "2026-11-04T09:30:00-08:00": "2026-11-04T17:30:00Z",
            "2026-11-04T09:30:00.5+00:00": "2026-11-04T09:30:00.500Z"}  # each time given, as stored
EDGES = ["0001-01-01T00:00:00+01:00", "9999-12-31T23:59:59-01:00"]


@pytest.fixture(autouse=True)
def refits(monkeypatch):
    calls = []
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: calls.append(str(home)) or {"started": True, "pid": 0})
    return calls


@pytest.fixture
def pacific(monkeypatch):
    """Local time on the US Pacific coast, so a time with no offset reads the same on any machine."""
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture
def home(tmp_path) -> Path:
    home = tmp_path / "lm"
    install_rec(home)
    return home


def _run(capsys, home: Path, state: str) -> tuple[str, str]:
    """(run, slate): a run started in a slate of its own, then finished unless `state` is open."""
    code, out, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath", "--title",
                         "t", "--config", EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration", "--rule", "tests",
                         "--new-slate", "--json")
    assert code == 0, err
    if state == "finished":
        assert cli(capsys, home, "run", "finish", "--run", out["run"], "--no-fit")[0] == 0
    return out["run"], out["slate"]


def _argv(form: str, run: str, slate: str) -> list[str]:
    return ["outcome", "--run", run, "--signal", "tests=pass"] if form == "signal" else [
        "outcome", "--slate", slate, "--prefer", run]


def _times(home: Path, run: str, form: str) -> list[str]:
    """The stored observed_at of the run's signals, or of its preferences."""
    held = Store(home).run_doc(run)["run"].get("signals" if form == "signal" else "preferences") or []
    return [x["observed_at"] for x in held]


def _store(home: Path) -> dict[str, bytes]:
    """Every file of the store but its lock, by path."""
    return {p.relative_to(home).as_posix(): p.read_bytes() for p in sorted(home.rglob("*"))
            if p.is_file() and p.name != "lock"}


@pytest.mark.parametrize("state", ["open", "finished"])
@pytest.mark.parametrize("form", ["signal", "slate"])
@pytest.mark.parametrize("given", list(OBSERVED))
def test_a_time_python_reads_is_stored_in_utc_as_flow_score_stores_it(given, form, state, capsys, home):
    run, slate = _run(capsys, home, state)
    before = _times(home, run, form)
    code, out, err = cli(capsys, home, *_argv(form, run, slate), "--observed-at", given)
    assert code == 0 and err == ""
    assert _times(home, run, form) == [*before, OBSERVED[given]]


@pytest.mark.parametrize("form", ["signal", "slate"])
def test_no_offset_is_local_time_and_no_time_is_now(form, capsys, home, pacific):
    run, slate = _run(capsys, home, "finished")
    before = _times(home, run, form)
    assert cli(capsys, home, *_argv(form, run, slate), "--observed-at", "2026-11-04T09:30:00")[0] == 0
    start = parse_ts(now_iso())
    assert cli(capsys, home, *_argv(form, run, slate))[0] == 0
    *kept, naive, now = _times(home, run, form)
    assert kept == before and naive == "2026-11-04T17:30:00Z"
    assert now.endswith("Z") and start <= parse_ts(now) <= parse_ts(now_iso())


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("form", ["signal", "slate"])
@pytest.mark.parametrize("given", EDGES)
def test_a_time_past_what_utc_holds_is_refused_by_name_and_nothing_is_written(given, form, as_json, capsys, home,
                                                                              refits):
    run, slate = _run(capsys, home, "finished")
    before, fits = _store(home), list(refits)
    reason = f"--observed-at {given!r} cannot be stored: it is past what a UTC time holds"
    code, out, err = cli(capsys, home, *_argv(form, run, slate), "--observed-at", given, *(["--json"] * as_json))
    assert code == 1 and err == f"error: {reason}\n"
    assert out == ({"schema": "loopmath.outcome/1", "ok": False, "error": reason, "exit": 1} if as_json else "")
    assert _store(home) == before and refits == fits


@pytest.mark.parametrize("as_json", [False, True], ids=["text", "json"])
@pytest.mark.parametrize("form", ["signal", "slate"])
@pytest.mark.parametrize("given", ["0001-01-01T00:00:00", "9999-12-31T23:59:59"], ids=["first", "last"])
def test_a_local_time_at_a_year_edge_is_refused_naming_the_flag_and_the_value(given, form, as_json, capsys, home,
                                                                              refits, pacific):
    """A time with no offset at either edge of the years is local time that no UTC time holds: refused in the words
    of a time with an offset past it, exit 1, and nothing is written."""
    run, slate = _run(capsys, home, "finished")
    before, fits = _store(home), list(refits)
    reason = f"--observed-at {given!r} cannot be stored: it is past what a UTC time holds"
    code, out, err = cli(capsys, home, *_argv(form, run, slate), "--observed-at", given, *(["--json"] * as_json))
    assert code == 1 and err == f"error: {reason}\n"
    assert out == ({"schema": "loopmath.outcome/1", "ok": False, "error": reason, "exit": 1} if as_json else "")
    assert _store(home) == before and refits == fits


@pytest.mark.parametrize("form", ["signal", "slate"])
def test_text_that_is_no_time_keeps_its_words(form, capsys, home, refits, pacific):
    run, slate = _run(capsys, home, "finished")
    before, fits = _store(home), list(refits)
    code, out, err = cli(capsys, home, *_argv(form, run, slate), "--observed-at", "soon")
    assert code == 1 and out == "" and err == "error: not an ISO 8601 timestamp: 'soon'\n"
    assert _store(home) == before and refits == fits
