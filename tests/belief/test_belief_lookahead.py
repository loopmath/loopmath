"""Look-ahead gain against brute force, and its bookkeeping (spec 04 section 6)."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest
import simdata
from numpy.polynomial.hermite_e import hermegauss
from scipy import linalg
from scipy.special import expit

from loopmath.belief import fit as F
from loopmath.belief import lookahead as L
from loopmath.belief import state as S
from loopmath.belief.design import task_from_doc, task_terms
from loopmath.belief.gaussian import gaussian_rank_one
from loopmath.belief.state import load
from loopmath.types import AcceptanceRule, ScoreTarget, Setting

MARTINGALE_DRAWS = 8000


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


def test_the_posterior_shift_is_a_martingale(small, monkeypatch):
    """The shift's means are the current means. On cost the look-ahead's is closed form and predict's
    averages the known draws, so the two may differ by 4 Monte Carlo standard errors of that average,
    computed from the same draws (antithetic pairs), and never by more than 3%. With the fit's 400
    draws 4 SE is about 3.7% of the mean here (7.4% with fsrc), so the test loads the fit with 8000
    (0.9%, and 1.6% with fsrc)."""
    state, task, cfgs = small
    monkeypatch.setattr(S, "N_DRAWS", MARTINGALE_DRAWS)
    monkeypatch.setattr(L, "N_DRAWS", MARTINGALE_DRAWS)
    state = load(state.path)
    explore = cfgs[-1]
    res = state.lookahead(task, explore, cfgs[0], cfgs, rescue_usd=3.0)
    prep = L._prepare(state, task, cfgs, None)
    i = prep.index[explore.id]
    g_now = L._probit(prep.success.mu, prep.success.var)[i]
    assert res.posterior_shift["p_success"].mean == pytest.approx(g_now, abs=0.02)
    # On success the update is exact over the draws: every candidate's g after, averaged over the verdict, is its
    # g now. Explore's g now is predict's chance within the noise of the unseen nodes' draws, which predict
    # integrates per known draw instead.
    o = L._outcomes(state, task, explore, cfgs[0], cfgs, None, 3.0)
    np.testing.assert_allclose(o.pz @ o.g_after, o.g_now, rtol=1e-9, atol=0)
    assert res.posterior_shift["p_success"].mean == pytest.approx(o.g_now[i], rel=1e-9)
    pred = state.predict(task, explore)
    known, s2 = state._known(state.heads["success"], state._task_parts(task)["success"], state._plan(explore)["run"])
    chance = S.normal_mean(expit, known, s2)  # predict's, one value per known draw
    assert s2 > 0 and pred.p_success.mean == pytest.approx(float(np.mean(chance)), rel=1e-12)
    per_draw = (prep.weight * prep.chance[i]).reshape(-1, MARTINGALE_DRAWS).sum(axis=0) * MARTINGALE_DRAWS
    assert float(np.mean(per_draw)) == pytest.approx(o.g_now[i], rel=1e-12)
    assert abs(o.g_now[i] - pred.p_success.mean) < 4 * _pair_se(per_draw - chance)
    mr, per_draw = state._mean_run(state._plan(explore), state._task_parts(task))
    m = per_draw(mr.exp_usd)  # predict's cost mean, one value per known draw
    assert len(m) == MARTINGALE_DRAWS and pred.cost.usd.mean == pytest.approx(float(np.mean(m)), rel=1e-12)
    half = len(m) // 2
    pairs = 0.5 * (m[:half] + m[half:])
    se = float(np.std(pairs, ddof=1)) / math.sqrt(half)
    exact = res.posterior_shift["cost_usd"].mean
    assert 4 * se < 0.03 * exact, "draw noise too large for a 4 SE test: raise MARTINGALE_DRAWS"
    assert abs(pred.cost.usd.mean - exact) < 4 * se
    iv = res.posterior_shift["cost_usd"]
    assert iv.lo < iv.mean < iv.hi and iv.level == 0.8


def _pair_se(t: np.ndarray) -> float:
    """Standard error of the mean of per-draw values, from antithetic pair means (draw j and j + half)."""
    half = len(t) // 2
    return float(np.std(0.5 * (t[:half] + t[half:]), ddof=1)) / math.sqrt(half)


def _known_at_means(state, name: str = "success"):
    """The state with one head's known nodes at their means, so only its unseen nodes vary between draws."""
    h, n = state.heads[name], S.N_DRAWS
    h0 = S.HeadState(h.name, h.meta, h.nodes, None, h.fit_id, mean=h.mean, U=np.zeros_like(h.U),
                     draws=np.repeat(h.mean[:, None], n, axis=1), unit=np.zeros((len(h.nodes), n)))
    h0.fitted, h0.phi = h.fitted, dict(h.phi)
    return S.FitState(state.path, meta=state.meta, heads={**state.heads, name: h0}, design=state.design)


def test_one_verdict_moves_a_candidate_through_a_shared_unseen_node(small, monkeypatch):
    """Spec 04 section 6: the success update moves every candidate through what it shares with explore, unseen
    nodes included. With the known nodes at their means, explore (a new model alone) shares only that model's node
    with a candidate that has four unseen nodes of its own. The candidate's shift after each verdict, by 2-D
    Gauss-Hermite over the shared node and its own, matches the update over 8000 draws within 4 standard errors
    (antithetic pair means), and is at least 6 of them, so an update through known nodes alone (no shift) fails.
    A candidate that shares nothing with explore moves by noise only."""
    state, task, _ = small
    monkeypatch.setattr(S, "N_DRAWS", MARTINGALE_DRAWS)
    monkeypatch.setattr(L, "N_DRAWS", MARTINGALE_DRAWS)
    state = _known_at_means(load(state.path))
    new_x, new_y = Setting("claude-code", "claude-fable-9-5", "medium"), Setting("codex", "gpt-9-astra-2", "xhigh")
    explore = simdata.config(simdata.SOLO, new_x)
    shared = simdata.config(simdata.IR, new_y, new_x)
    apart = simdata.config(simdata.SOLO, new_y)
    prep = L._prepare(state, task, [explore, shared, apart], None)
    assert state._task_parts(task)["success"].s2 == 0  # a known task: no unseen task effect, equal weights
    pz, g_now, g_after = L._success_update(prep, 0)
    np.testing.assert_allclose(pz @ g_after, g_now, rtol=1e-9, atol=0)
    head, tt = state.heads["success"], state.task_terms(task)

    def loads(cfg):
        Xk, Xv, nodes, phi = head.matrices([tt + list(state._plan(cfg)["run"])])
        return float((Xk @ head.mean)[0]), {n: float(Xv[0, j] * phi[j]) for j, n in enumerate(nodes)}

    (me, le), (mc, lc), (_, la) = loads(explore), loads(shared), loads(apart)
    assert len(le) == 1 and set(le) <= set(lc) and len(lc) == 5 and not set(la) & set(le)
    (node, ae), = le.items()
    ac, sc = lc[node], math.sqrt(sum(v * v for n, v in lc.items() if n != node))
    x, w = hermegauss(80)
    w = w / w.sum()
    pe = expit(me + ae * x)  # explore's chance at each value of the shared node
    pc = expit(mc + ac * x[:, None] + sc * x[None, :]) @ w  # the candidate's, its own nodes integrated out
    now = float(w @ pc)
    p = prep.chance
    assert abs(g_now[1] - now) < 4 * _pair_se(p[1] - g_now[1])
    q = L.LOOKAHEAD_Q
    for zi, z in enumerate((1, 0)):
        def like(s, z=z):
            return q * s + (1 - q) * (1 - s) if z else (1 - q) * s + q * (1 - s)

        want = float(w @ (like(pe) * pc)) / float(w @ like(pe)) - now
        lk = like(p[0]) / np.mean(like(p[0]))
        se = _pair_se((lk - 1.0) * (p[1] - g_after[zi, 1]))
        assert abs(g_after[zi, 1] - g_now[1] - want) < 4 * se, (z, g_after[zi, 1] - g_now[1], want, se)
        assert abs(want) > 6 * se, (z, want, se)
        assert abs(g_after[zi, 2] - g_now[2]) < 4 * _pair_se((lk - 1.0) * (p[2] - g_after[zi, 2]))


def test_score_rule_lookahead(sim_fit):
    state = sim_fit["state"]
    task = task_from_doc(sim_fit["docs"][0])
    rule = AcceptanceRule("perf>=2100", "perf", requires=(), score=ScoreTarget("perf", 2100.0, "higher"))
    cfgs = simdata.all_configs()[:10]
    res = state.lookahead(task, cfgs[3], cfgs[0], cfgs, rule, rescue_usd=5.0)
    assert res.gain_per_run["usd"] >= 0 and res.gain_per_run["score"] is not None
    assert 0 <= res.posterior_shift["p_success"].mean <= 1
