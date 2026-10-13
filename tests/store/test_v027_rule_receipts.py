"""Receipts scored before 0.2.7 under a rule `run finish` read otherwise than the fit (0.2.7).

`run finish` used to read `excludes_events: []` as no exclusions and `window_days: 0` as no window; the fit read
them as revert and incident, and 14 days, and from 0.2.7 every reader does. Every fit of the store, `loopmath fit`
or the background job, first settles such runs' receipts, doing only what is still to do: a receipt whose stored z
differs from the z the run has now, or still flagged `rescore`, is re-scored and written with a mark; the run file's
embedded receipt follows a marked receipt; and a marked receipt gets one note event on its run. A step that fails is
named in the fit's note on stderr and done by the next fit; a step done is never done twice, and a fit with nothing
left writes nothing. Only those runs' receipt files are opened. `report` reads only: it counts the receipts still to
re-score and names `loopmath fit`. A store with no such rule reads no receipt and re-scores nothing. Synthetic runs
only.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from loopmath.store import finish as finish_mod
from loopmath.store import fitjob
from loopmath.store.finish import read_otherwise_before_027, settle_rule_receipts
from loopmath.store.home import Store
from tests.store.store_helpers import EXPLORE_CFG, REC_ID, cli, fake_settle, install_rec


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(fitjob, "spawn_fit", lambda home, **kw: {"started": True, "pid": 0})
    monkeypatch.setattr(finish_mod, "default_settle", lambda: fake_settle())
    h = tmp_path / "lm"
    install_rec(h)
    return h


def _finished(capsys, home, tmp_path):
    code, start, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                           "--config", EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration", "--rule", "tests", "--json")
    assert code == 0, err
    run = start["run"]
    cli(capsys, home, "run", "attempt", "--run", run, "--piece", "implement", "--harness", "codex",
        "--model", "gpt-6-sol", "--cwd", str(tmp_path))
    assert cli(capsys, home, "outcome", "--run", run, "--signal", "tests=pass", "--tier", "verified")[0] == 0
    assert cli(capsys, home, "run", "finish", "--run", run, "--no-fit")[0] == 0
    return run


def _as_before_027(home, run, **rule):
    """The run as a 0.2.6 store holds it: its rule has `rule`'s keys, it ended two days ago, and a revert came a
    day after the end that the old reading did not count, so the receipt still says accepted."""
    def apply(doc):
        doc["run"]["acceptance_rule"] = {**doc["run"]["acceptance_rule"], **rule}
        ended = datetime.now().astimezone() - timedelta(days=2)
        doc["run"]["ended_at"] = ended.isoformat(timespec="seconds")
        at = (ended + timedelta(days=1)).isoformat(timespec="seconds")
        doc["run"]["signals"].append({"id": "sig_revert1", "kind": "event", "name": "revert", "value": "PR-9",
                                      "observed_at": at, "tier": "reported", "source": {"kind": "user"}})
    Store(home)._update(run, apply)


def test_a_store_with_no_such_rule_reads_no_receipt_and_rescores_nothing(capsys, home, tmp_path, monkeypatch):
    run = _finished(capsys, home, tmp_path)
    assert not read_otherwise_before_027(Store(home).run_doc(run))
    monkeypatch.setattr(Store, "receipts", lambda self: pytest.fail("receipts were read"))
    assert settle_rule_receipts(Store(home)) == {"checked": [], "rescored": [], "failed": []}


@pytest.mark.parametrize("rule", [{"excludes_events": []}, {"window_days": 0}], ids=["no exclusions", "window 0"])
def test_a_receipt_scored_under_the_old_reading_is_rescored_with_an_event(capsys, home, tmp_path, rule):
    run = _finished(capsys, home, tmp_path)
    rct = Store(home).receipts_for(run)[0]
    assert rct["after"]["z"] == 1.0
    _as_before_027(home, run, **rule)
    assert read_otherwise_before_027(Store(home).run_doc(run))
    out = settle_rule_receipts(Store(home))
    assert out == {"checked": [run], "rescored": [rct["id"]], "failed": []}
    again = Store(home).receipts_for(run)[0]
    assert again["after"]["z"] == 0.0 and again["rescore"] is False and again["settled"] == "0.2.7 rule reading"
    events = Store(home).run_doc(run)["events"]
    assert events[-1]["type"] == "note" and events[-1]["detail"].startswith(f"receipt {rct['id']} re-scored")
    assert settle_rule_receipts(Store(home))["rescored"] == []  # a second pass changes nothing


def _notes(home, run):
    """The run's receipt notes (`run start` writes a note of its own)."""
    return [e["detail"] for e in Store(home).run_doc(run)["events"] if e["type"] == "note" and e["detail"].startswith("receipt ")]


def _with_runs_to_fit(home):
    from tests.belief import simdata

    for doc in simdata.simulate(30, seed=47, source="live", prefix="run_f")[0]:
        Store(home).import_run(doc, finished=True, check=False)


def test_report_counts_them_a_foreground_fit_settles_them_and_report_agrees(capsys, home, tmp_path):
    runs = [_finished(capsys, home, tmp_path) for _ in range(3)]
    _as_before_027(home, runs[0], window_days=0)
    _as_before_027(home, runs[1], excludes_events=[])
    _with_runs_to_fit(home)
    before = {p.name: p.read_bytes() for p in (home / "receipts").glob("*.json")}

    def report_z(rep):
        return [{e["run"]: e["z"] for e in rep["exploration"]["runs"]}[r] for r in runs]

    code, rep, err = cli(capsys, home, "report", "--json")
    assert code == 0, err
    assert rep["unsettled_receipts"] == 2 and report_z(rep) == [1.0, 1.0, 1.0]
    accepted = rep["exploration"]["accepted"]
    code, text, err = cli(capsys, home, "report")
    assert "2 receipts scored before 0.2.7 are not re-scored yet to read their run's rule as the fit does; " \
           "`loopmath fit` re-scores them" in text
    assert {p.name: p.read_bytes() for p in (home / "receipts").glob("*.json")} == before  # report wrote nothing

    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert out["receipts_rescored"] == 2
    assert [Store(home).receipts_for(r)[0]["after"]["z"] for r in runs] == [0.0, 0.0, 1.0]
    for run in runs[:2]:
        rct = Store(home).receipts_for(run)[0]["id"]
        assert _notes(home, run) == [f"receipt {rct} re-scored: the run's rule is read as the fit reads it"]
    assert _notes(home, runs[2]) == []
    code, rep, err = cli(capsys, home, "report", "--json")
    assert code == 0, err
    assert "unsettled_receipts" not in rep and report_z(rep) == [0.0, 0.0, 1.0]
    assert rep["exploration"]["accepted"] == accepted - 2

    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")  # nothing left: no count, no new event
    assert code == 0, err
    assert "receipts_rescored" not in out and [len(_notes(home, r)) for r in runs] == [1, 1, 0]


def test_a_settlement_that_fails_leaves_the_fit_going_and_says_so(capsys, home, tmp_path, monkeypatch):
    run = _finished(capsys, home, tmp_path)
    _as_before_027(home, run, excludes_events=[])
    _with_runs_to_fit(home)

    def broken(store):
        raise OSError("disk full")
    real = finish_mod.settle_rule_receipts
    monkeypatch.setattr(finish_mod, "settle_rule_receipts", broken)
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert "note: receipts scored before 0.2.7 were not re-scored: OSError: disk full" in err
    assert "receipts_rescored" not in out and Store(home).receipts_for(run)[0]["after"]["z"] == 1.0
    monkeypatch.setattr(finish_mod, "settle_rule_receipts", real)
    code, text, err = cli(capsys, home, "fit", "--no-prior")  # the next fit settles them
    assert code == 0, err
    assert "1 receipt scored before 0.2.7 re-scored first" in text
    assert Store(home).receipts_for(run)[0]["after"]["z"] == 0.0



MARK = "0.2.7 rule reading"


def _report(capsys, home):
    code, rep, err = cli(capsys, home, "report", "--json")
    assert code == 0, err
    return rep


def _files(home):
    """Every run and receipt file's bytes: a fit with nothing left to do writes none."""
    return {f"{sub}/{p.name}": p.read_bytes() for sub in ("runs", "receipts") for p in sorted((home / sub).glob("*.json"))}


def _z(home, runs):
    return [Store(home).receipts_for(r)[0]["after"]["z"] for r in runs]


def _embedded_signals(home, run):
    return Store(home).run_doc(run)["run"]["receipt"]["after"]["signals"]


def _a_later_fit_writes_nothing(capsys, home, runs, notes):
    before = _files(home)
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert "note:" not in err and "receipts_rescored" not in out
    assert _files(home) == before and [len(_notes(home, r)) for r in runs] == notes


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file with no read permission")
def test_a_receipt_that_cannot_be_read_is_named_and_the_next_fit_rescores_it(capsys, home, tmp_path):
    runs = [_finished(capsys, home, tmp_path) for _ in range(3)]
    _as_before_027(home, runs[0], window_days=0)
    _as_before_027(home, runs[1], excludes_events=[])
    _with_runs_to_fit(home)
    rep = _report(capsys, home)
    assert rep["unsettled_receipts"] == 2
    accepted = rep["exploration"]["accepted"]
    locked = Store(home).receipts_for(runs[0])[0]["id"]
    path = home / "receipts" / f"{locked}.json"
    mode = path.stat().st_mode
    path.chmod(0)
    try:
        code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    finally:
        path.chmod(mode)
    assert code == 0, err
    assert _z(home, runs) == [1.0, 0.0, 1.0] and [len(_notes(home, r)) for r in runs] == [0, 1, 0]
    said = f"receipts/{locked}.json: PermissionError: Permission denied"
    assert "are not all settled yet; the next fit does what is left: " in err and said in err
    assert out["receipts_rescored"] == 1 and any(said in n for n in out["notes"])
    rep = _report(capsys, home)
    assert rep["unsettled_receipts"] == 1 and rep["exploration"]["accepted"] == accepted - 1

    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")  # readable again: the fit retries it
    assert code == 0, err
    assert out["receipts_rescored"] == 1 and "note:" not in err and "notes" not in out
    assert _z(home, runs) == [0.0, 0.0, 1.0] and [len(_notes(home, r)) for r in runs] == [1, 1, 0]
    assert [Store(home).receipts_for(r)[0].get("settled") for r in runs] == [MARK, MARK, None]
    rep = _report(capsys, home)
    assert "unsettled_receipts" not in rep and rep["exploration"]["accepted"] == accepted - 2
    _a_later_fit_writes_nothing(capsys, home, runs, [1, 1, 0])


def test_a_receipt_that_stays_unreadable_is_named_by_every_fit(capsys, home, tmp_path):
    run = _finished(capsys, home, tmp_path)
    _as_before_027(home, run, excludes_events=[])
    _with_runs_to_fit(home)
    bad = Store(home).receipts_for(run)[0]["id"]
    (home / "receipts" / f"{bad}.json").write_text("{not json")
    for _ in range(2):
        code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
        assert code == 0, err
        assert f"run {run}, receipts/{bad}.json: JSONDecodeError: " in err
        assert "receipts_rescored" not in out and _notes(home, run) == []


def _foreign(home, run):
    def other(doc):
        doc["producer"] = {"name": "other-orchestrator", "version": "1"}
        doc["events"] = [e for e in doc["events"] if e.get("type") != "receipt_written"]
    Store(home)._update(run, other)


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists a folder with no read permission")
def test_a_receipts_folder_that_cannot_be_listed_is_named_and_the_next_fit_rescores_them(capsys, home, tmp_path):
    runs = [_finished(capsys, home, tmp_path) for _ in range(2)]
    _as_before_027(home, runs[0], window_days=0)
    _as_before_027(home, runs[1], excludes_events=[])
    for run in runs:  # written by another tool: found by listing the receipts
        _foreign(home, run)
    _with_runs_to_fit(home)
    rep = _report(capsys, home)
    assert rep["unsettled_receipts"] == 2
    accepted = rep["exploration"]["accepted"]
    folder = home / "receipts"
    mode = folder.stat().st_mode
    folder.chmod(0)
    try:
        code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    finally:
        folder.chmod(mode)
    assert code == 0, err
    for run in runs:
        assert f"run {run}, the receipts folder: PermissionError: Permission denied" in err
    assert "receipts_rescored" not in out and _z(home, runs) == [1.0, 1.0]
    rep = _report(capsys, home)
    assert rep["unsettled_receipts"] == 2 and rep["exploration"]["accepted"] == accepted

    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert out["receipts_rescored"] == 2 and "note:" not in err
    assert _z(home, runs) == [0.0, 0.0] and [len(_notes(home, r)) for r in runs] == [1, 1]
    rep = _report(capsys, home)
    assert "unsettled_receipts" not in rep and rep["exploration"]["accepted"] == accepted - 2
    _a_later_fit_writes_nothing(capsys, home, runs, [1, 1])


def test_a_corrected_write_that_fails_is_done_by_the_next_fit(capsys, home, tmp_path, monkeypatch):
    run = _finished(capsys, home, tmp_path)
    _as_before_027(home, run, excludes_events=[])
    _with_runs_to_fit(home)
    accepted = _report(capsys, home)["exploration"]["accepted"]
    rct = Store(home).receipts_for(run)[0]["id"]
    real = Store.write_receipt

    def full(self, receipt):
        if receipt.get("id") == rct:
            raise OSError(28, "No space left on device")
        return real(self, receipt)
    monkeypatch.setattr(Store, "write_receipt", full)
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    monkeypatch.setattr(Store, "write_receipt", real)
    assert code == 0, err
    held = Store(home).receipts_for(run)[0]
    assert held["after"]["z"] == 1.0 and "settled" not in held and _notes(home, run) == []
    assert f"run {run}, receipts/{rct}.json: OSError: No space left on device" in err
    assert "receipts_rescored" not in out
    rep = _report(capsys, home)
    assert rep["unsettled_receipts"] == 1 and rep["exploration"]["accepted"] == accepted

    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert out["receipts_rescored"] == 1 and "note:" not in err
    held = Store(home).receipts_for(run)[0]
    assert held["after"]["z"] == 0.0 and held["rescore"] is False and held["settled"] == MARK
    assert _notes(home, run) == [f"receipt {rct} re-scored: the run's rule is read as the fit reads it"]
    rep = _report(capsys, home)
    assert "unsettled_receipts" not in rep and rep["exploration"]["accepted"] == accepted - 1

    opened = []
    read = finish_mod.read_json
    monkeypatch.setattr(finish_mod, "read_json", lambda path: opened.append(path.name) or read(path))
    _a_later_fit_writes_nothing(capsys, home, [run], [1])
    assert opened == [f"{rct}.json"]  # a later fit reads only the named receipts of such runs


def test_a_receipt_left_flagged_is_rescored_by_the_next_fit(capsys, home, tmp_path):
    """A re-score cut short leaves the receipt flagged `rescore` at its old z; the next foreground fit finishes it."""
    run = _finished(capsys, home, tmp_path)
    _as_before_027(home, run, excludes_events=[])
    _with_runs_to_fit(home)
    accepted = _report(capsys, home)["exploration"]["accepted"]
    held = Store(home).receipts_for(run)[0]
    Store(home).write_receipt({**held, "rescore": True})
    rep = _report(capsys, home)
    assert rep["unsettled_receipts"] == 1 and rep["exploration"]["accepted"] == accepted
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert out["receipts_rescored"] == 1
    held = Store(home).receipts_for(run)[0]
    assert held["after"]["z"] == 0.0 and held["rescore"] is False and len(_notes(home, run)) == 1
    assert "unsettled_receipts" not in _report(capsys, home)
    _a_later_fit_writes_nothing(capsys, home, [run], [1])


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a folder with no write permission")
def test_a_run_file_that_cannot_be_written_gets_its_note_and_receipt_on_the_next_fit(capsys, home, tmp_path):
    runs = [_finished(capsys, home, tmp_path) for _ in range(2)]
    _as_before_027(home, runs[0], window_days=0)
    _as_before_027(home, runs[1], excludes_events=[])
    _with_runs_to_fit(home)
    assert not any("sig_revert1" in _embedded_signals(home, r) for r in runs)
    folder = home / "runs"
    mode = folder.stat().st_mode
    folder.chmod(0o500)
    try:
        code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    finally:
        folder.chmod(mode)
    assert code == 0, err
    for run in runs:
        assert f"run {run}, runs/{run}.ocp.json: PermissionError: Permission denied" in err
    assert out["receipts_rescored"] == 2 and _z(home, runs) == [0.0, 0.0]  # the receipts are right already
    assert [_notes(home, r) for r in runs] == [[], []]
    assert not any("sig_revert1" in _embedded_signals(home, r) for r in runs)
    assert "unsettled_receipts" not in _report(capsys, home)

    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert "receipts_rescored" not in out and "note:" not in err  # nothing to re-score: the run file is caught up
    assert [len(_notes(home, r)) for r in runs] == [1, 1]
    assert all("sig_revert1" in _embedded_signals(home, r) for r in runs)
    _a_later_fit_writes_nothing(capsys, home, runs, [1, 1])


@pytest.mark.parametrize("fails", ["the note", "the embedded receipt"])
def test_each_run_file_step_left_undone_is_done_once(capsys, home, tmp_path, monkeypatch, fails):
    run = _finished(capsys, home, tmp_path)
    _as_before_027(home, run, excludes_events=[])
    _with_runs_to_fit(home)
    real = Store._update
    monkeypatch.setattr(Store, "_update", lambda self, r, fn: (_ for _ in ()).throw(OSError(28, "No space left on device")))
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    monkeypatch.setattr(Store, "_update", real)
    assert code == 0, err
    assert f"run {run}, runs/{run}.ocp.json: OSError: No space left on device" in err
    rct = Store(home).receipts_for(run)[0]
    assert rct["settled"] == MARK and _notes(home, run) == [] and "sig_revert1" not in _embedded_signals(home, run)
    if fails == "the note":  # the receipt was pointed at by another writer, the note is still to add
        from loopmath.store.finish import _point_run_at
        _point_run_at(Store(home), run, Store(home).run_doc(run), rct)
    else:  # the note is there, the embedded receipt still stale
        Store(home)._update(run, lambda d: d["events"].append(
            {"type": "note", "at": d["events"][-1]["at"],
             "detail": f"receipt {rct['id']} re-scored: the run's rule is read as the fit reads it"}))
    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert len(_notes(home, run)) == 1 and "sig_revert1" in _embedded_signals(home, run)
    _a_later_fit_writes_nothing(capsys, home, [run], [1])


def test_the_background_fit_job_settles_them_too(capsys, home, tmp_path):
    run = _finished(capsys, home, tmp_path)
    _as_before_027(home, run, excludes_events=[])
    _with_runs_to_fit(home)
    out = fitjob.run_job(home, no_prior=True)
    assert out["ran"] == 1 and out["failed"] == 0
    assert Store(home).receipts_for(run)[0]["after"]["z"] == 0.0 and len(_notes(home, run)) == 1


class _CutOff(BaseException):
    """The process ends here."""


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a folder with no write permission")
@pytest.mark.parametrize("pointing", ["normal writes", "a failed pointing write"])
@pytest.mark.parametrize("flagged", ["before the upgrade", "by an interrupted outcome"])
def test_a_receipt_the_background_job_corrects_first_gets_its_note_and_receipt(capsys, home, tmp_path, monkeypatch,
                                                                               flagged, pointing):
    """The job's flagged pass re-scores a flagged receipt before the fit settles: it writes the mark too."""
    runs = [_finished(capsys, home, tmp_path) for _ in range(2)]
    _as_before_027(home, runs[0], window_days=0)
    _as_before_027(home, runs[1], excludes_events=[])
    _with_runs_to_fit(home)
    target = runs[0]
    if flagged == "before the upgrade":  # a late signal whose re-score never ran under 0.2.6
        Store(home).flag_rescore(target)
        signal = "sig_revert1"
    else:  # `outcome` on the finished run ends just after it saves the flag
        real = Store.flag_rescore

        def cut(self, run):
            real(self, run)
            raise _CutOff
        monkeypatch.setattr(Store, "flag_rescore", cut)
        with pytest.raises(_CutOff):
            cli(capsys, home, "outcome", "--run", target, "--kind", "event", "--signal", "incident=PR-10")
        monkeypatch.setattr(Store, "flag_rescore", real)
        signal = Store(home).run_doc(target)["run"]["signals"][-1]["id"]
    held = Store(home).receipts_for(target)[0]
    assert held["rescore"] is True and held["after"]["z"] == 1.0 and "settled" not in held
    assert signal not in _embedded_signals(home, target)
    folder = home / "runs"
    mode = folder.stat().st_mode
    if pointing == "a failed pointing write":
        folder.chmod(0o500)
    try:
        out = fitjob.run_job(home, no_prior=True)
    finally:
        folder.chmod(mode)
    assert out["ran"] == 1
    held = Store(home).receipts_for(target)[0]
    assert held["after"]["z"] == 0.0 and held["rescore"] is False and held["settled"] == MARK
    if pointing == "normal writes":  # the job's own fit adds the notes
        assert [len(_notes(home, r)) for r in runs] == [1, 1] and signal in _embedded_signals(home, target)
    else:
        assert [len(_notes(home, r)) for r in runs] == [0, 0] and signal not in _embedded_signals(home, target)

    code, out, err = cli(capsys, home, "fit", "--no-prior", "--json")
    assert code == 0, err
    assert "note:" not in err and "receipts_rescored" not in out
    assert _z(home, runs) == [0.0, 0.0] and [len(_notes(home, r)) for r in runs] == [1, 1]
    assert signal in _embedded_signals(home, target) and "sig_revert1" in _embedded_signals(home, runs[1])
    assert "unsettled_receipts" not in _report(capsys, home)
    _a_later_fit_writes_nothing(capsys, home, runs, [1, 1])


def test_only_the_affected_runs_receipts_are_opened_and_a_later_settlement_writes_nothing(capsys, home, tmp_path,
                                                                                          monkeypatch):
    affected, other = _finished(capsys, home, tmp_path), _finished(capsys, home, tmp_path)
    _as_before_027(home, affected, window_days=0)
    for i in range(5):  # receipts of no run in the store
        Store(home).write_receipt({"id": f"rct_unrelated{i}", "run": f"run_gone{i}", "after": {"z": 1.0}})
    mine = Store(home).receipts_for(affected)[0]["id"]
    opened = []
    real = finish_mod.read_json
    monkeypatch.setattr(finish_mod, "read_json", lambda path: opened.append(path.name) or real(path))
    monkeypatch.setattr(Store, "receipts", lambda self, errors=None: pytest.fail("every receipt was read"))
    assert settle_rule_receipts(Store(home)) == {"checked": [affected], "rescored": [mine], "failed": []}
    assert opened == [f"{mine}.json"]
    before = _files(home)
    opened.clear()
    assert settle_rule_receipts(Store(home)) == {"checked": [affected], "rescored": [], "failed": []}
    assert opened == [f"{mine}.json"] and _files(home) == before and len(_notes(home, affected)) == 1
    assert _notes(home, other) == []


def test_a_run_another_tool_wrote_is_found_by_reading_the_receipts(capsys, home, tmp_path):
    run = _finished(capsys, home, tmp_path)
    _as_before_027(home, run, excludes_events=[])
    _foreign(home, run)
    rct = Store(home).receipts_for(run)[0]["id"]
    assert settle_rule_receipts(Store(home)) == {"checked": [run], "rescored": [rct], "failed": []}
    assert Store(home).receipts_for(run)[0]["after"]["z"] == 0.0 and len(_notes(home, run)) == 1


def test_a_receipt_not_in_the_store_is_nothing_to_do_and_one_that_cannot_be_read_is_named_for_a_foreign_run(
        capsys, home, tmp_path):
    gone, foreign = _finished(capsys, home, tmp_path), _finished(capsys, home, tmp_path)
    for run in (gone, foreign):
        _as_before_027(home, run, excludes_events=[])
    (home / "receipts" / f"{Store(home).receipts_for(gone)[0]['id']}.json").unlink()
    _foreign(home, foreign)
    broken = home / "receipts" / "rct_broken.json"
    broken.write_text("[")  # may be the foreign run's: named against it
    rct = Store(home).receipts_for(foreign)[0]["id"]
    out = settle_rule_receipts(Store(home))
    assert out["rescored"] == [rct] and len(out["failed"]) == 1
    assert out["failed"][0].startswith(f"run {foreign}, receipts/rct_broken.json: JSONDecodeError: ")
    broken.unlink()
    assert settle_rule_receipts(Store(home)) == {"checked": [foreign, gone] if foreign < gone else [gone, foreign],
                                                 "rescored": [], "failed": []}
    assert len(_notes(home, foreign)) == 1 and _notes(home, gone) == []


def _fit_z(doc):
    from loopmath.belief.design import parse_run

    return parse_run(doc, now=datetime.now().astimezone()).evidence.z


def test_an_upgraded_store_s_report_and_fit_agree_with_no_step(capsys, home, tmp_path):
    """A store upgraded from 0.2.6: a live run finished with a review verdict only (its receipt says z None), and
    an onboarded review-only run. `report` recomputes a stored z of None through the same resolver as the fit."""
    from tests.belief import simdata

    code, start, err = cli(capsys, home, "run", "start", "--type", "feature", "--repo", "loopmath/loopmath",
                           "--config", EXPLORE_CFG, "--rec", REC_ID, "--source", "exploration", "--rule", "tests", "--json")
    assert code == 0, err
    live = start["run"]
    cli(capsys, home, "run", "attempt", "--run", live, "--piece", "implement", "--harness", "codex",
        "--model", "gpt-6-sol", "--cwd", str(tmp_path))
    assert cli(capsys, home, "outcome", "--run", live, "--signal", "review=accept")[0] == 0
    assert cli(capsys, home, "run", "finish", "--run", live, "--no-fit")[0] == 0
    assert Store(home).receipts_for(live)[0]["after"]["z"] is None
    onboarded = simdata.simulate(1, seed=43, source="live", prefix="run_hist")[0][0]
    onboarded["run"]["task"]["source"] = {"kind": "history"}
    onboarded["run"].pop("acceptance_rule", None)
    onboarded["run"]["signals"] = [{"id": "sig_r", "kind": "verdict", "name": "review", "value": "accept",
                                    "observed_at": "2026-09-20T11:30:00-07:00", "tier": "reported"}]
    Store(home).import_run(onboarded, finished=True, check=False)

    def both():
        code, rep, err = cli(capsys, home, "report", "--json")
        assert code == 0, err
        report = {e["run"]: e["z"] for e in rep["exploration"]["runs"]}
        fit = {run: _fit_z(Store(home).run_doc(run)) for run in (live, onboarded["run"]["id"])}
        return report, fit, rep["runs"]["history"]

    report, fit, history = both()
    assert report == {live: None} and fit[live] is None and history == 1
    assert fit[onboarded["run"]["id"]] == 1.0  # read by the review rule with no step
    assert cli(capsys, home, "outcome", "--rule", "review", live)[0] == 0
    report, fit, _ = both()
    assert report == {live: 1.0} and fit[live] == 1.0
