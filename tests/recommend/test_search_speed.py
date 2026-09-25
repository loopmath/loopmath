"""The workflow search's speed target (spec 05 section 1a): the benchmark space, 12 shapes over widths 2, 3, 4 and
6 and round limits 1 to 6 (126 cases) with 6 models x 5 efforts x 2 harnesses per piece (60 settings), searched in
under 2 seconds on the synthetic fit. A busy host (load above the core count) skips rather than fails."""

from __future__ import annotations

import os
import time

import pytest

from loopmath.recommend import search as S
from loopmath.types import Setting

import search_synth as H

MODELS = ("claude-opus-9", "claude-opus-9-5", "claude-fable-9", "gpt-9-astra", "gpt-9-sol", "gpt-9-luna")
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def busy() -> bool:
    try:
        return os.getloadavg()[0] > (os.cpu_count() or 1)
    except OSError:
        return False


def test_the_benchmark_space_is_searched_in_under_two_seconds(tmp_path_factory):
    fs = H.synth_fit(tmp_path_factory)["state"]
    settings = [Setting(h, m, e) for m in MODELS for h in ("codex", "claude-code") for e in EFFORTS]
    sp = S.space_from([], H.USUAL, widths=(2, 3, 4, 6), rounds=tuple(range(1, 7)), settings=settings)
    assert len(sp.cases) == 126 and len(sp.settings) == 60
    assert sp.configurations() == 323_708_700
    if busy():
        pytest.skip(f"host busy (load {os.getloadavg()[0]:.1f} on {os.cpu_count()} cores)")
    best = None
    for _ in range(2):
        t0 = time.perf_counter()
        found = S.search(fs, H.TASK, H.RULE, 1.5, sp, draws=200, per_objective=40)
        took = time.perf_counter() - t0
        best = took if best is None else min(best, took)
    assert found.stats["exact"] and found.front and found.wins
    assert best < 2.0, f"search took {best:.2f} s"
