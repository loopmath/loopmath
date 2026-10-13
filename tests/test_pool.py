"""The worker pool returns what the one-process loop returns, in the same order."""

from __future__ import annotations

import dataclasses

from loopmath import pool
from loopmath.graph.extract import extract

from test_graph_extract import ALPHA, skeleton_records


def test_ordered_map_matches_loop_in_workers(monkeypatch):
    monkeypatch.setenv("LOOPMATH_WORKERS", "2")
    monkeypatch.setattr(pool, "MIN_ITEMS", 1)
    items = list(range(-50, 50))
    seen = []
    out = pool.ordered_map(abs, items, weights=[i % 7 for i in items], done=lambda i, n: seen.append((i, n)))
    assert out == [abs(x) for x in items]
    assert seen[-1] == (100, 100)


def test_workers_env(monkeypatch):
    monkeypatch.setenv("LOOPMATH_WORKERS", "1")
    assert pool.workers() == 1
    monkeypatch.setenv("LOOPMATH_WORKERS", "junk")
    assert 1 <= pool.workers() <= pool.MAX_DEFAULT


def test_extract_same_graph_with_workers(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "one"))
    monkeypatch.setenv("LOOPMATH_WORKERS", "1")
    one = extract(skeleton_records(), workspaces=[ALPHA])
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "two"))
    monkeypatch.setenv("LOOPMATH_WORKERS", "2")
    monkeypatch.setattr(pool, "MIN_ITEMS", 1)
    two = extract(skeleton_records(), workspaces=[ALPHA])
    assert dataclasses.asdict(one) == dataclasses.asdict(two)
