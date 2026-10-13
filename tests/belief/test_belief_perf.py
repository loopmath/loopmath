"""Performance targets of spec 04 section 7, on a generated store of 3,000 runs."""

from __future__ import annotations

import itertools
import time
from datetime import datetime

import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief.design import task_from_doc
from loopmath.belief.state import load_latest
from loopmath.types import Setting


def _candidates(n: int) -> list:
    """n distinct configurations: every shape with settings drawn from 14 per piece, some unseen."""
    pool = list(simdata.SETTINGS) + [Setting(s.harness, s.model, e) for s in simdata.SETTINGS[:4]
                                     for e in ("low", "max")] + [Setting("codex", "gpt-9-nova", "high"),
                                                                 Setting("claude-code", "claude-opus-9-7", "high")]
    out, seen = [], set()
    for wf in simdata.SHAPES:
        for combo in itertools.product(pool, repeat=len(wf.pieces)):
            cfg = simdata.config(wf, *combo)
            if cfg.id not in seen:
                seen.add(cfg.id)
                out.append(cfg)
    rotated = [out[i::7] for i in range(7)]  # mix the shapes
    return [c for part in rotated for c in part][:n]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    docs, _ = simdata.simulate(3000, seed=77, source="user", n_tasks=400)
    home = tmp_path_factory.mktemp("perf")
    t = time.perf_counter()
    F.fit(home, docs=docs, no_prior=True, now=datetime.fromisoformat("2026-09-23T12:00:00-07:00"))
    return home, docs, time.perf_counter() - t


def test_fit_under_two_minutes(store):
    home, docs, seconds = store
    assert sum(len(d["attempts"]) for d in docs) >= 10_000
    assert seconds < 120


def test_load_under_300_ms(store):
    home, _, _ = store
    t = time.perf_counter()
    state = load_latest(home)
    _ = state.meta, state.heads
    assert time.perf_counter() - t < 0.3


def test_predict_many_2000_under_3_seconds(store):
    home, docs, _ = store
    cands = _candidates(2000)
    assert len(cands) == 2000
    best = float("inf")
    for _ in range(2):  # best of two cold runs (a fresh state each), so a busy machine does not fail it
        state = load_latest(home)
        t = time.perf_counter()
        preds = state.predict_many(task_from_doc(docs[5]), cands)
        best = min(best, time.perf_counter() - t)
    assert best < 3.0
    assert len(preds) == 2000


def test_lookahead_top_200_under_10_seconds(store):
    home, docs, _ = store
    state = load_latest(home)
    task = task_from_doc(docs[9])
    top = _candidates(200)
    goal = top[0]
    t = time.perf_counter()
    for explore in top:
        state.lookahead(task, explore, goal, top, rescue_usd=4.0)
    assert time.perf_counter() - t < 10.0
