"""Real data smoke test (spec 04 section 8): a fit on the shipped bundle, predictions for every shape.

Fits the bundle the package ships (lane 11), or the bundle directory `LOOPMATH_BUNDLE_DIR` names.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import simdata

from loopmath.belief import fit as F
from loopmath.belief.state import load
from loopmath.types import Task
from loopmath.workflows.format import catalog


def _bundle_dir() -> Path | None:
    env = os.environ.get("LOOPMATH_BUNDLE_DIR")
    return Path(env).expanduser() if env else None  # None: the packaged bundle


def test_fit_on_the_bundle_and_predict_every_shape(tmp_path):
    path = F.fit(tmp_path, bundle_dir=_bundle_dir())
    state = load(path)
    assert state.meta["n_runs"]["prior"] > 0 and state.meta["n_runs"]["user"] == 0
    assert {"cost", "success"} <= set(state.heads)
    task = Task("tsk_smoke", "feature", "newco/new-repo")
    shapes = list(catalog().values())
    assert shapes
    for wf in shapes:
        cfg = simdata.config(wf, *simdata.SETTINGS[:3])
        p = state.predict(task, cfg)
        for iv in (p.p_success, p.cost.usd, p.cost.tokens, p.ell.usd, p.rounds):
            assert all(math.isfinite(v) for v in (iv.mean, iv.lo, iv.hi))
            assert iv.lo <= iv.hi
        assert p.cost.usd.hi > p.cost.usd.lo > 0 and p.p_success.hi > p.p_success.lo
