"""Two writers and one fit at once (lane 07 milestone F), the background fit job, and finish under contention."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from store_helpers import SRC, fake_settle, task, usual_config

from loopmath.store import Store, fit_lock, fit_state, spawn_fit
from loopmath.store import runs as R
from loopmath.store.finish import finish_run
from loopmath.store.fitjob import MAX_RERUNS, run_job
from loopmath.types import DEFAULT_RULE

WRITER = r'''
import json, sys, time
from pathlib import Path
from loopmath.store import Store, spawn_fit
from loopmath.store.finish import finish_run
from loopmath.store.ids import new_id, now_iso
from loopmath.types import DEFAULT_RULE, Signal, Task

home, r0, tag, n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
store = Store(home)
cfg = store.run_doc(r0)["run"]["configuration"]

def settle(doc):
    for a in doc.get("attempts") or []:
        a["cost"] = {"input_tokens": 10, "output_tokens": 5, "usd": 0.01, "basis": "measured"}
    return doc, {"unmatched": {}}

while not (home / "go").exists():
    time.sleep(0.005)
fits = []
for i in range(n):
    store.add_attempt(r0, {"piece": "implement", "harness": "codex", "model": "m", "cwd": "/tmp"})
    store.add_signal(r0, Signal(id=new_id("sig"), run=r0, kind="score", name="s_" + tag, value=float(i),
                                observed_at=now_iso()))
    run = store.new_run(Task(id="tsk_" + tag + str(i), type="docs", repo="r"), cfg, source="usual", rec=None,
                        slate=None, rule=DEFAULT_RULE, base_commit=None)
    store.add_attempt(run, {"piece": "implement", "harness": "codex", "model": "m", "cwd": "/tmp"})
    fits.append(finish_run(store, run, settle=settle, spawn=spawn_fit)["fit"])
print(json.dumps(fits))
'''

FITTER = r'''
import json, sys, time
from pathlib import Path
from loopmath.store import Store
from loopmath.store import runs as R
from loopmath.store.fitjob import run_job

home = Path(sys.argv[1])
calls = home / "fit_calls.jsonl"

def fake_fit(home, **opts):
    first = not calls.exists()
    (home / "fit_started").write_text("1")
    deadline = time.time() + 60
    while first and not (home / "writers_done").exists() and time.time() < deadline:
        time.sleep(0.02)
    docs = list(Store(home).finished_docs())
    with open(calls, "a") as fh:
        fh.write(json.dumps({"n": len(docs), "open": [d["run"]["id"] for d in docs if not R.is_finished(d)],
                             "ids": sorted(d["run"]["id"] for d in docs)}) + "\n")
    return None

print(json.dumps(run_job(home, fit_fn=fake_fit)))
'''


def _env():
    return {**os.environ, "PYTHONPATH": SRC + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}


def _wait_for(path: Path, seconds: float = 30) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"timed out waiting for {path}"
        time.sleep(0.02)


def test_two_writers_and_one_fit(tmp_path):
    home = tmp_path / "lm"
    store = Store(home)
    r0 = store.new_run(task(), usual_config(), source="usual", rec=None, slate=None, rule=DEFAULT_RULE, base_commit=None)
    n = 10
    fitter = subprocess.Popen([sys.executable, "-c", FITTER, str(home)], env=_env(), stdout=subprocess.PIPE, text=True)
    writers = []
    try:
        _wait_for(home / "fit_started")
        writers = [subprocess.Popen([sys.executable, "-c", WRITER, str(home), r0, tag, str(n)], env=_env(),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for tag in ("a", "b")]
        (home / "go").write_text("1")
        outs = [w.communicate(timeout=120) for w in writers]
        for w, (out, err) in zip(writers, outs):
            assert w.returncode == 0, err
        (home / "writers_done").write_text("1")
        job_out, _ = fitter.communicate(timeout=120)
    finally:
        for p in [fitter, *writers]:
            if p.poll() is None:
                p.kill()
                p.wait()
    # every trigger while the fit ran was queued, and they cost exactly one more fit
    triggers = [f for out, _ in outs for f in json.loads(out)]
    assert len(triggers) == 2 * n and all(t.get("queued") and not t.get("started") for t in triggers)
    assert json.loads(job_out) == {"ran": 2, "failed": 0, "queued": False}
    calls = [json.loads(line) for line in (home / "fit_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert all(c["open"] == [] and r0 not in c["ids"] for c in calls)  # a fit never reads an open run
    assert calls[1]["n"] == 2 * n
    assert not (home / "fits" / "pending").exists()
    # no write was lost: both writers' attempts and signals are all in the open run
    doc = store.run_doc(r0)
    assert len(doc["attempts"]) == 2 * n and len({a["id"] for a in doc["attempts"]}) == 2 * n
    assert R.rev(doc) == 1 + 2 * n
    names = [s["name"] for s in doc["run"]["signals"]]
    assert names.count("s_a") == n and names.count("s_b") == n
    rows = store.index_rows()
    assert sum(1 for r in rows.values() if r["state"] == "finished") == 2 * n and rows[r0]["state"] == "open"
    assert store.reindex()["runs"] == 2 * n + 1


def test_finish_retries_once_when_the_run_changes_underneath(tmp_path):
    store = Store(tmp_path / "lm")
    run = store.new_run(task(), usual_config(), source="usual", rec=None, slate=None, rule=DEFAULT_RULE, base_commit=None)
    store.add_attempt(run, {"piece": "implement", "harness": "codex", "model": "m", "cwd": "/tmp"})
    calls = []
    inner = fake_settle()

    def racing_settle(doc):
        if not calls:  # another writer adds an attempt while this finish is matching logs
            store.add_attempt(run, {"piece": "review", "harness": "claude-code", "model": "m", "cwd": "/tmp"})
        calls.append(1)
        return inner(doc)

    res = finish_run(store, run, settle=racing_settle, no_fit=True)
    assert len(calls) == 2 and res["matched"]["verified"] == 2
    assert len(store.run_doc(run)["attempts"]) == 2


def test_run_job_records_a_failed_fit_without_raising(tmp_path):
    def broken(home, **opts):
        raise RuntimeError("no data")

    assert run_job(tmp_path, fit_fn=broken) == {"ran": 0, "failed": 1, "queued": False}
    state = fit_state(tmp_path)
    assert state["job"]["status"] == "failed" and "RuntimeError: no data" in state["job"]["error"] and not state["running"]


def test_run_job_coalesces_and_bounds_reruns(tmp_path):
    seen = []

    def trigger_twice(home, **opts):
        seen.append(opts)
        if len(seen) == 1:
            for _ in range(3):  # three triggers during one fit
                assert spawn_fit(home, full=True)["queued"] is True
        return None

    assert run_job(tmp_path, fit_fn=trigger_twice)["ran"] == 2
    assert seen[1]["full"] is True  # the queued options are used

    def always(home, **opts):
        spawn_fit(home)
        return None

    assert run_job(tmp_path, fit_fn=always)["ran"] == MAX_RERUNS


def test_spawn_fit_starts_a_detached_job(tmp_path):
    out = spawn_fit(tmp_path)
    assert out["started"] is True and out["pid"] > 0
    deadline = time.monotonic() + 30
    while True:
        job = fit_state(tmp_path)["job"] or {}
        if job.get("status") in ("done", "failed") and not fit_state(tmp_path)["running"]:
            break
        assert time.monotonic() < deadline, job
        time.sleep(0.05)
    assert job["pid"] == out["pid"]


def test_second_fit_waits_on_the_lock(tmp_path):
    with fit_lock(tmp_path) as got:
        assert got
        assert fit_state(tmp_path)["running"] is True
        assert run_job(tmp_path, fit_fn=lambda home, **o: pytest.fail("ran under a held lock")) == {"ran": 0,
                                                                                                    "queued": True}
    assert (tmp_path / "fits" / "pending").exists()


def test_default_fit_waits_out_a_foreground_fit(tmp_path, monkeypatch):
    """Lane 5's fit has its own lock and raises FitBusy at once; the background job asks it to wait instead."""
    import loopmath.belief.fit as belief_fit
    from loopmath.store import fitjob

    seen = {}

    def fake(home, *, no_prior=False, without=(), full=False, wait_s=0.0):
        seen.update(no_prior=no_prior, without=without, full=full, wait_s=wait_s)

    monkeypatch.setattr(belief_fit, "fit", fake, raising=False)
    assert run_job(tmp_path, without=["habit"])["ran"] == 1
    assert seen == {"no_prior": False, "without": ("habit",), "full": False, "wait_s": fitjob.FIT_WAIT_S}
