"""Family x source (spec 04 section 1; lane 2B). Synthetic bundles only."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief.design import cost_rest, structure
from loopmath.belief.state import load_latest
from loopmath.types import Setting

NOW = datetime.fromisoformat("2026-09-24T12:00:00-07:00")
Z80 = 1.2815515655446004
OPUS = Setting("claude-code", "claude-opus-9", "high")
FABLE = Setting("claude-code", "claude-fable-9", "high")
USER_GAP = 0.7  # fable against opus for the user; 0 in source e0, -1.0 in source sweep


def _bundle(user_runs: int) -> list[tuple[str, dict]]:
    rng = np.random.default_rng(4)
    truth = simdata.Truth(np.random.default_rng(5), scale_mult=0.0)
    truth.effects["cost"] = {"fsrc:fable|sweep": -1.0, "fsrc:fable|user": USER_GAP}
    truth.effects["tokens"] = dict(truth.effects["cost"])
    out = []
    for origin, n in (("sweep", 150), ("e0", 150), ("user", user_runs)):
        for i in range(n):
            task = simdata.make_task(rng, i, n_tasks=40)
            cfg = simdata.config(simdata.SOLO, (OPUS, FABLE)[i % 2])
            out.append((origin, simdata.simulate_run(truth, task, cfg, f"run_{origin}{i:04d}", rng, source=origin)))
    return out


def _fable_vs_opus(state):
    """The user's cost ratio of solo at fable against solo at opus: mean, 80% interval, unseen nodes."""
    h = state.heads["cost"]
    rows = [cost_rest(structure(simdata.config(simdata.SOLO, s)), "implement", 1, source="user") for s in (FABLE, OPUS)]
    Xk, Xv, vnodes, vphi = h.matrices(rows)
    dk, dv = Xk[0] - Xk[1], (Xv[0] - Xv[1]).toarray().ravel()
    mu = float((dk @ h.mean).ravel()[0])
    sd = math.sqrt(float(np.sum(np.asarray(dk @ h.U) ** 2)) + float(np.sum(dv ** 2 * vphi ** 2)))
    return math.exp(mu), math.exp(mu - Z80 * sd), math.exp(mu + Z80 * sd), dict(zip(vnodes, vphi))


@pytest.fixture(scope="module")
def fits(tmp_path_factory):
    out = {}
    for n in (0, 80):
        home = tmp_path_factory.mktemp(f"fsrc{n}")
        F.fit(home, docs=_bundle(n), no_prior=True, now=NOW)
        out[n] = load_latest(home)
    return out


def test_a_new_users_family_node_is_unseen_at_the_family_x_source_width(fits):
    state = fits[0]
    _, lo, hi, unseen = _fable_vs_opus(state)
    phi = state.heads["cost"].node_scale("fsrc:fable|user")
    assert {"fsrc:fable|user", "fsrc:opus|user"} <= set(unseen)
    assert unseen["fsrc:fable|user"] == phi and phi == state.heads["cost"].phi["fsrc"]
    assert "fsrc:fable|sweep" in state.heads["cost"].index
    assert lo < 1 < hi


def test_the_users_family_price_follows_the_users_own_runs(fits):
    state = fits[80]
    assert "fsrc:fable|user" in state.heads["cost"].index
    mean, lo, hi, unseen = _fable_vs_opus(state)
    assert not any(n.startswith("fsrc:") for n in unseen)
    assert lo < math.exp(USER_GAP) < hi and abs(math.log(mean) - USER_GAP) < 0.1
    _, lo0, hi0, _ = _fable_vs_opus(fits[0])
    assert math.log(hi / lo) < math.log(hi0 / lo0)
