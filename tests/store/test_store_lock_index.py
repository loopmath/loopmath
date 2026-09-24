"""Locking, atomic writes, the append-only index with reindex, status and report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
from store_helpers import SRC, cli, finished_doc

from loopmath.store import Store, StoreLocked
from loopmath.store.lock import append_line, atomic_write_json, read_jsonl, store_lock


def test_atomic_write_leaves_no_temp_and_replaces(tmp_path):
    path = tmp_path / "d" / "x.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert [p.name for p in path.parent.iterdir()] == ["x.json"]


def test_read_jsonl_skips_a_torn_last_line(tmp_path):
    path = tmp_path / "i.jsonl"
    append_line(path, {"run": "a"})
    with open(path, "a") as fh:
        fh.write('{"run": "b", "sta')
    assert read_jsonl(path) == [{"run": "a"}]


def test_lock_is_reentrant_in_a_thread(tmp_path):
    with store_lock(tmp_path):
        with store_lock(tmp_path, timeout=0.1):
            pass


def test_lock_held_by_another_process_times_out_with_exit_4(tmp_path, capsys):
    home = tmp_path / "lm"
    home.mkdir()
    holder = subprocess.Popen([sys.executable, "-c", (
        "import sys, time; from loopmath.store.lock import store_lock\n"
        f"with store_lock({str(home)!r}):\n    print('held', flush=True); time.sleep(20)")],
        env={**os.environ, "PYTHONPATH": SRC}, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        t0 = time.monotonic()
        with pytest.raises(StoreLocked):
            with store_lock(home, timeout=0.3):
                pass
        assert time.monotonic() - t0 < 5
        import loopmath.store.lock as lock_mod

        old = lock_mod.LOCK_TIMEOUT_S
        lock_mod.LOCK_TIMEOUT_S = 0.3
        try:
            code, out, err = cli(capsys, home, "config", "set", "org", "acme", "--json")
        finally:
            lock_mod.LOCK_TIMEOUT_S = old
        assert code == 4 and out["exit"] == 4 and "lock" in err
    finally:
        holder.kill()
        holder.wait()


def _doc(run, started="2026-09-20T10:00:00-07:00"):
    return finished_doc(run, started=started, open_=True)


def test_index_last_row_wins_and_reindex_rewrites(tmp_path, capsys):
    store = Store(tmp_path / "lm")
    store.import_run(_doc("run_a"), finished=False)
    store.import_run(_doc("run_a"))
    store.import_run(_doc("run_b"))
    assert len(read_jsonl(store.index_path)) == 3
    assert store.index_rows()["run_a"]["state"] == "finished"
    store.index_path.unlink()
    code, out, err = cli(capsys, store.home, "status", "--json")
    assert code == 0, err
    assert "index_missing" in [h["code"] for h in out["health"]]
    code, out, err = cli(capsys, store.home, "status", "--reindex", "--json")
    assert code == 0, err
    assert out["reindex"] == {"runs": 2, "unreadable": []} and out["counts"]["index_rows"] == 2
    assert "index_missing" not in [h["code"] for h in out["health"]]


def test_status_reports_open_runs_fit_and_health(tmp_path, capsys):
    store = Store(tmp_path / "lm")
    store.import_run(_doc("run_open", started="2020-01-01T00:00:00+00:00"), finished=False)
    (store.home / "runs" / ".run_x.ocp.json.abc.tmp").write_text("{")
    code, out, err = cli(capsys, store.home, "status", "--json")
    assert code == 0, err
    assert list(out)[0] == "schema" and out["schema"] == "loopmath.status/1"
    codes = {h["code"] for h in out["health"]}
    assert {"open_runs_stale", "temp_files", "no_fit"} <= codes
    assert out["open_runs"][0]["run"] == "run_open" and out["counts"]["open"] == 1
    code, text, _ = cli(capsys, store.home, "status")
    assert "open run_open" in text and len(text.splitlines()) <= 25


def test_status_on_an_empty_home(tmp_path, capsys):
    code, text, err = cli(capsys, tmp_path / "none", "status")
    assert code == 0, err
    assert "fit: none yet" in text


def _receipt(store, run, p, z, usd, lo, hi):
    store.write_receipt({"id": f"rct_{run[4:]}", "rec": None, "run": run, "fit": "fit_1",
                         "before": {"config": "cfg_aaaaaaaaaaaa", "p_success": {"mean": p, "lo": p, "hi": p, "level": 0.8},
                                    "cost": {"usd": {"mean": (lo + hi) / 2, "lo": lo, "hi": hi, "level": 0.8},
                                             "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}}},
                         "after": {"cost": {"usd": usd, "tokens": 1}, "z": z, "q": 0.95, "tier": "reported", "scores": {},
                                   "rounds": 1},
                         "scored": {"cost_in_interval": lo <= usd <= hi, "tokens_in_interval": True,
                                    "log_score_z": -0.1, "cost_log_ratio": 0.0}})


def test_report_calibration_cost_per_accepted_and_html(tmp_path, capsys):
    store = Store(tmp_path / "lm")
    for i, (source, p, z, usd) in enumerate([("usual", 0.75, 1.0, 1.0), ("usual", 0.72, 0.0, 3.0),
                                            ("exploration", 0.35, 1.0, 2.0), ("habit", 0.9, 1.0, 40.0)]):
        run = f"run_r{i}"
        doc = _doc(run)
        doc["run"]["configuration"]["source"] = source
        doc["attempts"] = finished_doc(run, usd=usd)["attempts"]
        store.import_run(doc)
        _receipt(store, run, p, z, usd, 0.5, 2.5)
    code, out, err = cli(capsys, store.home, "report", "--json")
    assert code == 0, err
    c = out["calibration"]
    assert c["scored"] == 4 and c["cost_coverage"] == 0.5
    assert {d["bin"][0]: (d["n"], d["realized"]) for d in c["success_by_decile"]} == {0.3: (1, 1.0), 0.7: (2, 0.5),
                                                                                     0.9: (1, 1.0)}
    k = out["cost_per_accepted"]
    assert k["accepted"] == 2 and k["usd"] == pytest.approx(3.0)  # habit history is not counted
    assert out["runs"]["history"] == 1
    code, text, err = cli(capsys, store.home, "report")  # dogfood: the text says what it leaves out
    assert text.splitlines()[0] == ("report: 4 finished run(s), 0 open; 1 of them from onboard history, "
                                    "left out of cost per accepted change")
    assert "recommendation: no stored recommendations yet" in text
    assert out["exploration"]["taken"] == 1 and out["exploration"]["runs"][0]["predicted_p_success"] == 0.35
    page = tmp_path / "r.html"
    code, text, err = cli(capsys, store.home, "report", "--html", str(page))
    assert code == 0, err
    html = page.read_text()
    assert "<script" not in html and "Cost per accepted change by source" in html
    code, text, err = cli(capsys, store.home, "report", "--since", "2d")
    assert code == 0 and "0 finished run(s)" in text
    code, _, err = cli(capsys, store.home, "report", "--since", "yesterday")
    assert code == 1


def test_finished_docs_reads_run_files_not_the_index(tmp_path):
    """Lane 5's fit reads `finished_docs`: a finished file with no index row counts, an open one never does."""
    store = Store(tmp_path / "lm")
    store.import_run(_doc("run_open_orch"), finished=False)
    placed = finished_doc("run_placed")
    placed["run"]["ended_at"] = "2026-09-20T11:00:00-07:00"  # an OCP producer's own end, no store ext, no events
    (store.home / "runs" / "run_placed.ocp.json").write_text(json.dumps(placed))
    ended_but_open = finished_doc("run_orch2")
    ended_but_open["run"]["ended_at"] = "2026-09-20T11:00:00-07:00"
    store.import_run(ended_but_open, finished=False)  # the store's own state decides: still open
    store.import_run(finished_doc("run_done"))
    (store.home / "runs" / "run_torn.ocp.json").write_text('{"ocp": "0.3", "run": {"id": "run_to')
    running = finished_doc("run_running")
    running["run"]["ended_at"] = "2026-09-20T11:00:00-07:00"
    running["run"]["ext"] = {"dev.loopmath": {"state": "running"}}  # the plan's key still says it is not done
    (store.home / "runs" / "run_running.ocp.json").write_text(json.dumps(running))
    assert sorted(d["run"]["id"] for d in store.finished_docs()) == ["run_done", "run_placed"]
    assert "run_placed" not in store.index_rows()
