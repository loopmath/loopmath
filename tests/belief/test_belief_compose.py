"""Composition against Monte Carlo (spec 04 sections 3 and 8)."""

from __future__ import annotations

import numpy as np
import simdata

from loopmath.belief.compose import Intervals, compose, cost_per_round, interval
from loopmath.belief.design import structure
from loopmath.types import Control, Gate, Piece, Workflow

N = 200_000


def _etas(st, cost, gate):
    ck = [(p, k) for p in st.pieces for k in range(1, st.k_max + 1)]
    gk = [(gi, k) for gi in range(len(st.gates)) for k in range(1, st.k_max + 1)]
    return ({k: np.full(N, cost(*k)) for k in ck}, {k: np.full(N, cost(*k) + 11.0) for k in ck},
            {k: np.full(N, gate(*k)) for k in gk})


def _sig(x):
    return 1 / (1 + np.exp(-x))


def test_single_piece_with_a_gate_and_three_rounds_matches_monte_carlo():
    # one implementer with a tests gate that sends it back to itself, K_max 3
    wf = Workflow("solo_gate", 1, "", simdata.SOLO.pieces, simdata.SOLO.artifacts, simdata.SOLO.edges,
                  Control(gates=(Gate("g", "implement", "tests_pass", on_fail="implement"),), budget_rounds=3))
    st = structure(simdata.config(wf, simdata.SETTINGS[0]))
    assert st.k_max == 3 and st.loops == [((0,), ["implement"])]
    cost = {1: -0.5, 2: -0.9, 3: -1.1}
    gate = {1: 0.2, 2: 0.6, 3: 0.9}
    ce, te, ge = _etas(st, lambda p, k: cost[k], lambda gi, k: gate[k])
    rd = compose(st, ce, te, ge, sigma_usd=0.5, sigma_tokens=0.4, rng=np.random.default_rng(3))
    s2 = 0.5 ** 2 / 2
    r = [1.0, 1 - _sig(0.2), (1 - _sig(0.2)) * (1 - _sig(0.6))]
    exact = sum(r[k - 1] * np.exp(cost[k] + s2) for k in (1, 2, 3))
    np.testing.assert_allclose(rd.exp_usd, exact, rtol=1e-12)
    assert abs(rd.sim_usd.mean() / exact - 1) < 0.01
    np.testing.assert_allclose(rd.rounds, sum(r), rtol=1e-12)
    np.testing.assert_allclose(rd.pieces["implement"].rounds, sum(r), rtol=1e-12)


def _brute_sweep(cost, gate, rng, n):
    """Plan once; per round implement, tests gate, review, review gate; a failed gate starts the next round."""
    total = np.zeros(n)
    total += np.exp(cost["plan", 1] + 0.5 * rng.standard_normal(n))
    running = np.ones(n, bool)
    for k in (1, 2, 3):
        total += running * np.exp(cost["implement", k] + 0.5 * rng.standard_normal(n))
        tests_ok = rng.random(n) < _sig(gate[0, k])
        reviewing = running & tests_ok
        total += reviewing * np.exp(cost["review", k] + 0.5 * rng.standard_normal(n))
        review_ok = rng.random(n) < _sig(gate[1, k])
        running = running & ~(tests_ok & review_ok)
    return total


def test_merged_repair_loop_matches_a_brute_force_simulation():
    st = structure(simdata.config(simdata.SWEEP, simdata.SETTINGS[0]))
    cost = {("plan", 1): -1.2, **{("implement", k): -0.4 - 0.2 * k for k in (1, 2, 3)},
            **{("review", k): -1.0 - 0.1 * k for k in (1, 2, 3)}}
    gate = {(0, k): 0.1 + 0.3 * k for k in (1, 2, 3)} | {(1, k): -0.2 + 0.4 * k for k in (1, 2, 3)}
    ce, te, ge = _etas(st, lambda p, k: cost.get((p, k), 0.0), lambda gi, k: gate[(gi, k)])
    rd = compose(st, ce, te, ge, sigma_usd=0.5, sigma_tokens=0.4, rng=np.random.default_rng(4))
    brute = _brute_sweep(cost, gate, np.random.default_rng(5), N)
    assert abs(rd.exp_usd[0] / brute.mean() - 1) < 0.01
    assert abs(rd.sim_usd.mean() / brute.mean() - 1) < 0.01
    # review runs in round 1 only when the tests gate passed
    np.testing.assert_allclose(rd.reach[("review", 1)], _sig(gate[(0, 1)]))


def test_width_multiplies_the_piece_cost():
    st = structure(simdata.config(simdata.BON, simdata.SETTINGS[0]))
    ce, te, ge = _etas(st, lambda p, k: -1.0, lambda gi, k: 0.0)
    rd = compose(st, ce, te, ge, sigma_usd=0.5, sigma_tokens=0.4, rng=np.random.default_rng(6))
    np.testing.assert_allclose(rd.pieces["implement"].exp_usd, 3 * np.exp(-1.0 + 0.125), rtol=1e-12)
    assert abs(rd.pieces["implement"].sim_usd.mean() / (3 * np.exp(-1.0 + 0.125)) - 1) < 0.01


def test_batched_intervals_equal_one_at_a_time():
    rng = np.random.default_rng(7)
    arrays = [rng.lognormal(size=400) for _ in range(5)]
    iv = Intervals()
    handles = [iv.add(a) for a in arrays]
    res = iv.resolve()
    for h, a in zip(handles, arrays):
        np.testing.assert_allclose(res[h], interval(a))


PLAN_IMPLEMENT = Workflow("plan_implement", 1, "", (Piece("plan", "planner"), Piece("implement", "implementer")),
                          ("plan_doc", "patch"), (("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch")),
                          Control(gates=(Gate("g", "implement", "tests_pass", on_fail="implement"),), budget_rounds=2))


def _plan_implement(implement_round_2_usd: float):
    st = structure(simdata.config(PLAN_IMPLEMENT, simdata.SETTINGS[0]))
    cost = {("plan", 1): 0.0, ("implement", 1): 0.0, ("implement", 2): np.log(implement_round_2_usd)}
    ce, te, ge = _etas(st, lambda p, k: cost.get((p, k), 0.0), lambda gi, k: 0.0)
    return compose(st, ce, te, ge, sigma_usd=0.0, sigma_tokens=0.0, rng=np.random.default_rng(8))


def test_piece_cost_is_the_full_run_contribution_and_cost_per_round_one_execution():
    # D60, the reviewer's case: plan costs 1 once, implement 1 per round, gate pass 0.5, 2 rounds at most
    rd = _plan_implement(1.0)
    plan, impl = rd.pieces["plan"], rd.pieces["implement"]
    np.testing.assert_allclose([plan.exp_usd.mean(), impl.exp_usd.mean(), rd.exp_usd.mean()], [1.0, 1.5, 2.5])
    shares = {p: d.exp_usd.mean() / rd.exp_usd.mean() for p, d in rd.pieces.items()}
    np.testing.assert_allclose([shares["plan"], shares["implement"]], [0.4, 0.6])
    np.testing.assert_allclose(impl.rounds, 1.5)
    assert impl.round_free
    (usd_mean, usd), (tok_mean, _) = cost_per_round(impl)
    assert usd_mean == 1.0 and np.isclose(tok_mean, np.exp(11.0)) and np.all(usd == 1.0)
    assert cost_per_round(plan)[0][0] == 1.0


def test_cost_per_round_is_a_ratio_of_expectations_when_rounds_differ():
    # implement costs 1 in round 1 and 2 in round 2: E[C] = 1 + 0.5 * 2 = 2 over E[executions] = 1.5
    rd = _plan_implement(2.0)
    impl = rd.pieces["implement"]
    assert not impl.round_free
    np.testing.assert_allclose(impl.exp_usd, 2.0)
    (usd_mean, usd), _ = cost_per_round(impl)
    np.testing.assert_allclose(usd_mean, 2.0 / 1.5)
    assert set(np.round(usd, 12)) == {1.0, 1.5}  # one simulated run: 1 alone, or (1 + 2) / 2


def test_intervals_skip_draws_where_the_piece_did_not_run():
    iv = Intervals()
    h = iv.add(np.array([1.0, np.nan, 3.0, np.nan, 5.0]), 3.0)
    mean, lo, hi = iv.resolve()[h]
    assert mean == 3.0 and np.isclose(lo, np.percentile([1, 3, 5], 10)) and np.isclose(hi, np.percentile([1, 3, 5], 90))


def test_gates_control_reach_in_a_single_round():
    # D69, the reviewer's case: the sweep with one round, every piece $1, both gates pass half the time
    from dataclasses import replace

    wf = replace(simdata.SWEEP, control=replace(simdata.SWEEP.control, budget_rounds=1))
    st = structure(simdata.config(wf, simdata.SETTINGS[0]))
    assert st.k_max == 1 and st.loops == [((0, 1), ["implement", "review"])]
    n = 10
    ce = {(p, 1): np.zeros(n) for p in st.pieces}
    ge = {(gi, 1): np.zeros(n) for gi in range(len(st.gates))}
    rd = compose(st, ce, ce, ge, sigma_usd=0.0, sigma_tokens=0.0, rng=np.random.default_rng(1))
    np.testing.assert_allclose(rd.exp_usd, 2.5)
    np.testing.assert_allclose(rd.reach[("review", 1)], 0.5)
    np.testing.assert_allclose(rd.rounds, 1.0)
    assert set(np.round(rd.sim_usd, 12)) <= {2.0, 3.0}  # one simulated run: review ran or it did not
