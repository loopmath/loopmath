"""D60 against lane 5's own composer (review of 99548fc): piece shares come from each piece's
whole-run cost, never multiplied by its expected rounds again."""

from __future__ import annotations

import numpy as np
import pytest

from loopmath.belief import compose as compose_mod
from loopmath.belief.design import GateInfo, Structure
from loopmath.views import posterior as P


def test_shares_from_the_composer_totals():
    # plan runs once; implement costs 1 per round, passes its gate half the time, at most 2 rounds
    st = Structure(workflow_id="plan_implement", pieces=["plan", "implement"], roles={}, settings={},
                   widths={"plan": 1, "implement": 1},
                   gates=[GateInfo(id="g", after="implement", rule="tests_pass", on_fail="implement", judged="implement")],
                   loops=[((0,), ["implement"])], k_max=2, piece_loop={"plan": None, "implement": 0}, gate_loop={0: 0})
    zeros = np.zeros(400)
    cost = {("plan", 1): zeros, ("implement", 1): zeros, ("implement", 2): zeros}
    rd = compose_mod.compose(st, cost, cost, {(0, 1): zeros, (0, 2): zeros}, sigma_usd=0, sigma_tokens=0,
                             rng=np.random.default_rng(1))
    per_piece = {p: {"cost": {"usd": {"mean": float(np.mean(d.exp_usd))}, "tokens": {"mean": float(np.mean(d.exp_tokens))}},
                     "rounds": {"mean": float(np.mean(d.rounds))}} for p, d in rd.pieces.items()}
    assert float(np.mean(rd.exp_usd)) == pytest.approx(2.5)
    assert per_piece["implement"]["cost"]["usd"]["mean"] == pytest.approx(1.5)
    assert per_piece["implement"]["rounds"]["mean"] == pytest.approx(1.5)
    assert P.cost_shares(per_piece) == {"plan": 0.4, "implement": 0.6}
