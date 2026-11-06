"""`compare.paired`'s loss margin and P(cheaper): a candidate may be worse by less than the margin, and the
chance that it costs less per run. Synthetic runs only."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import stdtr

from loopmath.compare import Observation, paired
from loopmath.compare import model as M


def scored(n, loss=0.0, *, cost_base=1.0, cost_cand=1.0, seed=0, priced=True, scale=None):
    rng = np.random.default_rng(seed)
    base, cand = [], []
    for i in range(n):
        qb, qc = 0.6 + 0.15 * rng.standard_normal(), 0.6 - loss + 0.15 * rng.standard_normal()
        ub = float(cost_base * np.exp(0.3 * rng.standard_normal())) if priced else None
        uc = float(cost_cand * np.exp(0.3 * rng.standard_normal())) if priced else None
        base.append(Observation(f"t{i:02d}", "base", float(qb), ub, run=f"b{i:02d}", pair=f"p{i:02d}", scale=scale))
        cand.append(Observation(f"t{i:02d}", "cand", float(qc), uc, run=f"c{i:02d}", pair=f"p{i:02d}", scale=scale))
    return base, cand


def test_a_loss_margin_reads_p_better_above_minus_the_margin():
    base, cand = scored(16, loss=0.05)
    c = paired(base, cand, rule="score:quality", loss_margin=0.1)
    plain = paired(base, cand, rule="score:quality")
    assert c.floor == -0.1 and plain.floor == 0.0
    diffs = [o2.outcome - o1.outcome for o1, o2 in zip(base, cand)]
    data = M.Data(kind="score", diffs=diffs,
                  scale=M.score_scale([o.outcome for pair in zip(base, cand) for o in pair]))
    nu, loc, s = M.score_t(data)
    assert c.p_better == pytest.approx(float(stdtr(nu, (loc + 0.1) / s)), abs=1e-6)
    assert c.p_better > plain.p_better
    assert any("loss margin is 0.1" in r for r in c.reasons)


@pytest.mark.parametrize("kw, match", [
    ({"rule": "tests", "loss_margin": 0.1}, "score rule"),
    ({"rule": "score:quality", "loss_margin": 0.1, "floor": 0.0}, "not both"),
    ({"rule": "score:quality", "loss_margin": 0}, "above 0"),
    ({"rule": "score:quality", "loss_margin": -0.1}, "above 0"),
    ({"rule": "score:quality", "loss_margin": math.nan}, "above 0"),
    ({"rule": "score:quality", "loss_margin": True}, "above 0"),
])
def test_a_bad_loss_margin_is_refused(kw, match):
    base, cand = scored(6)
    if kw["rule"] == "tests":
        base = [Observation(o.task, o.arm, True, o.usd, run=o.run, pair=o.pair) for o in base]
        cand = [Observation(o.task, o.arm, False, o.usd, run=o.run, pair=o.pair) for o in cand]
    with pytest.raises(ValueError, match=match):
        paired(base, cand, **kw)


def test_a_fraction_score_takes_a_loss_margin_under_1_only():
    base, cand = scored(6, scale="fraction")
    assert paired(base, cand, rule="score:quality", loss_margin=0.1).floor == -0.1
    with pytest.raises(ValueError, match="less than 1"):
        paired(base, cand, rule="score:quality", loss_margin=1.0)
    with pytest.raises(ValueError):
        paired(base, cand, rule="score:quality", floor=-0.5)  # a negative floor itself stays refused


def test_the_calibration_states_no_bound_under_a_loss_margin():
    base, cand = scored(10)
    c = paired(base, cand, rule="score:quality", loss_margin=0.1, max_pairs=24)
    assert c.bound is None
    assert any("the floor -0.1 is under the calibrated 0" in r for r in c.reasons)


def test_p_cheaper_reads_the_joint_cost_draws():
    dear = paired(*scored(16, cost_cand=1.6), rule="score:quality", seed=3)
    cheap = paired(*scored(16, cost_cand=0.6), rule="score:quality", seed=3)
    assert cheap.p_cheaper > 0.95 and dear.p_cheaper < 0.05
    assert cheap.p_cheaper == paired(*scored(16, cost_cand=0.6), rule="score:quality", seed=3).p_cheaper
    assert cheap.to_dict()["p_cheaper"] == cheap.p_cheaper


def test_p_cheaper_is_none_when_a_cost_is_unknown():
    base, cand = scored(8)
    cand[3] = Observation(cand[3].task, "cand", cand[3].outcome, None, run=cand[3].run, pair=cand[3].pair)
    c = paired(base, cand, rule="score:quality")
    assert c.outcomes_only and c.p_cheaper is None
    assert paired(*scored(8, priced=False), rule="score:quality").p_cheaper is None
