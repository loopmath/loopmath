"""Look-ahead gain against brute force, and its bookkeeping (spec 04 section 6)."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
import simdata
from scipy import linalg

from loopmath.belief import fit as F
from loopmath.belief import lookahead as L
from loopmath.belief.design import task_from_doc, task_terms
from loopmath.belief.gaussian import gaussian_rank_one
from loopmath.belief.state import load
from loopmath.types import AcceptanceRule, ScoreTarget, Setting


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    docs, _ = simdata.simulate(300, seed=41, source="live", n_tasks=40)
    path = F.fit(tmp_path_factory.mktemp("la"), docs=docs, no_prior=True,
                 now=datetime.fromisoformat("2026-09-23T12:00:00-07:00"))
    state = load(path)
    task = task_from_doc(docs[0])
    # the cheap families' unseen versions contest the cheapest known configurations
    new = [simdata.config(simdata.SOLO, Setting("claude-code", "claude-fable-9-5", "medium")),
           simdata.config(simdata.SOLO, Setting("codex", "gpt-9-astra-2", "xhigh"))]
    cfgs = [simdata.config(simdata.SOLO, st) for st in simdata.SETTINGS] + new
    return state, task, cfgs


def _brute_gain(state, task, cfgs, explore, m=40_000, seed=0):
    """E[min ell after] by sampling effects, a noisy observation, and an exact rank-one update."""
    head = state.heads["cost"]
    sig2 = head.sigma ** 2
    prep = L._prepare(state, task, cfgs + [c for c in (explore,) if c not in cfgs], None)
    tt = task_terms(task, "user")
    rows = [tt + list(rest) for cfg in prep.configs for rest in state._plan(cfg)["cost"].values()]
    obs, n = L._explore_cost_obs(state, task, explore)
    Xk, Xv, _, vphi = head.matrices(rows + [obs])
    X = np.hstack([Xk.toarray(), Xv.toarray()])
    x, X = X[-1], X[:-1]
    mean = np.concatenate([head.mean, np.zeros(len(vphi))])
    U = linalg.block_diag(head.U, np.diag(vphi)) if len(vphi) else head.U

    def ell(mu, var):
        return prep.owner @ (prep.coef.reshape(-1, *([1] * (mu.ndim - 1))) * np.exp(mu + 0.5 * (var + sig2)))

    now = ell(X @ mean, np.sum((X @ U) ** 2, 1))
    tau2 = sig2 / n
    _, U2 = gaussian_rank_one(mean, U, x, 0.0, tau2)
    v = U.T @ x
    rng = np.random.default_rng(seed)
    b = mean[:, None] + U @ rng.standard_normal((len(mean), m))
    y = x @ b + math.sqrt(tau2) * rng.standard_normal(m)
    mu_after = (X @ mean)[:, None] + np.outer(X @ (U @ v), (y - x @ mean) / (v @ v + tau2))
    mins = ell(mu_after, np.sum((X @ U2) ** 2, 1)[:, None]).min(axis=0)
    return max(0.0, float(now.min() - mins.mean())), float(mins.std() / math.sqrt(m))


def test_gain_matches_brute_force(small):
    state, task, cfgs = small
    for explore in cfgs[-2:]:
        analytic = state.lookahead(task, explore, cfgs[0], cfgs).gain_per_run["usd"]
        brute, se = _brute_gain(state, task, cfgs, explore)
        assert brute > 10 * se  # a contested choice, not a zero gain
        assert abs(analytic - brute) < 4 * se + 0.03 * brute


def test_gain_is_nonnegative_and_zero_without_a_contest(small):
    state, task, cfgs = small
    for explore in cfgs:
        res = state.lookahead(task, explore, cfgs[0], cfgs, rescue_usd=2.0)
        assert res.gain_per_run["usd"] >= 0 and 0.0 <= res.p_beats_goal <= 1.0
    brute, se = _brute_gain(state, task, cfgs, cfgs[0])  # an expensive, well known configuration
    assert state.lookahead(task, cfgs[0], cfgs[0], cfgs).gain_per_run["usd"] <= brute + 4 * se
    assert state.lookahead(task, cfgs[0], cfgs[0], cfgs).p_beats_goal == 0.0


def test_the_posterior_shift_is_a_martingale(small):
    state, task, cfgs = small
    explore = cfgs[-1]
    res = state.lookahead(task, explore, cfgs[0], cfgs, rescue_usd=3.0)
    prep = L._prepare(state, task, cfgs, None)
    i = prep.index[explore.id]
    g_now = L._probit(prep.success.mu, prep.success.var)[i]
    assert res.posterior_shift["p_success"].mean == pytest.approx(g_now, abs=0.02)
    pred = state.predict(task, explore)
    assert res.posterior_shift["cost_usd"].mean == pytest.approx(pred.cost.usd.mean, rel=0.02)
    iv = res.posterior_shift["cost_usd"]
    assert iv.lo < iv.mean < iv.hi and iv.level == 0.8


def test_score_rule_lookahead(sim_fit):
    state = sim_fit["state"]
    task = task_from_doc(sim_fit["docs"][0])
    rule = AcceptanceRule("perf>=2100", "perf", requires=(), score=ScoreTarget("perf", 2100.0, "higher"))
    cfgs = simdata.all_configs()[:10]
    res = state.lookahead(task, cfgs[3], cfgs[0], cfgs, rule, rescue_usd=5.0)
    assert res.gain_per_run["usd"] >= 0 and res.gain_per_run["score"] is not None
    assert 0 <= res.posterior_shift["p_success"].mean <= 1
