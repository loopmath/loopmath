"""Shared fixtures for the lane 5 tests: one simulated store fitted once per session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simdata  # noqa: E402

SCORE = {"name": "perf", "scale": "linear", "better": "higher", "center": 2000.0, "spread": 300.0}


@pytest.fixture(scope="session")
def sim_fit(tmp_path_factory):
    """800 simulated user runs with a binary verdict and a `perf` score, fitted without priors."""
    from datetime import datetime

    from loopmath.belief import fit as fitmod
    from loopmath.belief.state import load_latest

    docs, truth = simdata.simulate(800, seed=11, source="live", n_tasks=120, score=SCORE)
    home = tmp_path_factory.mktemp("belief_home")
    # a fixed clock fixes the fit id, and with it every draw seed
    path = fitmod.fit(home, docs=docs, no_prior=True, now=datetime.fromisoformat("2026-09-23T12:00:00-07:00"))
    return {"home": home, "path": path, "docs": docs, "truth": truth, "state": load_latest(home)}
