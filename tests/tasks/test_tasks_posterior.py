"""`loopmath.tasks.posterior`: the candidate's ability on a grid and its suite score S with a range.

S is the mean over the suite of the candidate's chance of acceptance per task. Checks: the posterior mean
against brute-force Monte Carlo, the prior estimate before any run, a range about as narrow as a full run's
after every task once, and 80% coverage near nominal on banks where the model holds.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from loopmath.tasks.fit import TaskFit
from loopmath.tasks.posterior import CandidatePosterior
from loopmath.tasks.sim import simulate_bank


def small_fit(sigma_u=0.5):
    tasks = [f"t{i}" for i in range(5)]
    return TaskFit(model="2pl", tasks=tasks, b=np.array([-1.5, -0.5, 0.0, 0.8, 2.0]),
                   b_se=np.array([0.3, 0.3, 0.4, 0.4, 0.9]), a=np.array([1.0, 1.3, 0.8, 1.0, 1.2]),
                   theta={"h1": 0.2, "h2": -0.4}, theta_mean=0.0, theta_sd=1.2, tau=1.5, sigma_u=sigma_u,
                   slope_sd=0.3, converged=True, grad_norm=0.0)


def brute_force_mean(f, results, draws=400_000, seed=7, arm_slope_sd=0.0):
    rng = np.random.default_rng(seed)
    th = rng.normal(f.theta_mean, f.theta_sd, draws)
    lam = np.exp(arm_slope_sd * rng.normal(0.0, 1.0, draws))
    v = f.a**2 * f.b_se**2 + f.sigma_u**2
    e = rng.normal(0.0, 1.0, (draws, len(f.tasks))) * np.sqrt(v)
    p = expit(lam[:, None] * f.a * (th[:, None] - f.b) + e)
    w = np.ones(draws)
    for t, (s, n) in results.items():
        j = f.tasks.index(t)
        w *= p[:, j] ** s * (1 - p[:, j]) ** (n - s)
    return float(np.sum(w * p.mean(1)) / np.sum(w))


def test_the_posterior_mean_of_s_matches_monte_carlo():
    f = small_fit()
    results = {"t1": (1, 1), "t3": (0, 1), "t4": (1, 2)}
    est = CandidatePosterior(f, results).score()
    assert abs(est["mean"] - brute_force_mean(f, results)) < 0.01
    lo, hi = est["range80"]
    assert lo < est["mean"] < hi and est["range95"][0] <= lo and est["range95"][1] >= hi


def test_with_no_runs_the_estimate_is_the_prior_and_wide():
    f = small_fit()
    est = CandidatePosterior(f, {}).score()
    assert abs(est["mean"] - brute_force_mean(f, {})) < 0.01
    assert est["range80"][1] - est["range80"][0] > 0.2
    assert CandidatePosterior(f, {}).sessions == 0


def test_after_every_task_once_the_range_is_about_as_narrow_as_a_full_run():
    truth, bank = simulate_bank(np.random.default_rng(11), arms=10, tasks=60, trials=5, model="rasch", tau=1.8)
    from loopmath.tasks.fit import fit_bank

    f = fit_bank(bank, model="rasch", sigma_u=0.3)
    rng = np.random.default_rng(12)
    p = expit(0.3 - f.b)
    results = {t: (int(rng.random() < p[j]), 1) for j, t in enumerate(f.tasks)}
    post = CandidatePosterior(f, results)
    half = (post.score()["range80"][1] - post.score()["range80"][0]) / 2
    assert half <= 1.15 * post.full_run_halfwidth(0.8)


def test_80_percent_coverage_is_near_nominal_where_the_model_holds():
    rng = np.random.default_rng(21)
    truth, bank = simulate_bank(rng, arms=12, tasks=50, trials=5, model="rasch", tau=1.6, sigma_u=0.4)
    from loopmath.tasks.fit import fit_bank

    f = fit_bank(bank, model="rasch", sigma_u=0.4)
    hits = 0
    reps = 120
    for _ in range(reps):
        th = rng.normal(f.theta_mean, f.theta_sd)
        p = expit(truth.a * (th - truth.b) + rng.normal(0, 0.4, len(truth.b)))
        seen = rng.choice(len(f.tasks), 15, replace=False)
        results = {f.tasks[j]: (int(rng.random() < p[j]), 1) for j in seen}
        lo, hi = CandidatePosterior(f, results).score()["range80"]
        hits += lo <= p.mean() <= hi
    assert 0.68 <= hits / reps <= 0.92


def joint_monte_carlo(f, results, draws=300_000, seed=9):
    """Weighted draws of S from the same task fit: theta from the prior, e_t per task, weights from the runs."""
    rng = np.random.default_rng(seed)
    th = rng.normal(f.theta_mean, f.theta_sd, draws)
    v = f.a**2 * f.b_se**2 + f.sigma_u**2
    p = expit(f.a * (th[:, None] - f.b) + rng.normal(0.0, 1.0, (draws, len(f.tasks))) * np.sqrt(v))
    logw = np.zeros(draws)
    for t, (s, n) in results.items():
        j = f.tasks.index(t)
        logw += s * np.log(p[:, j]) + (n - s) * np.log1p(-p[:, j])
    w = np.exp(logw - logw.max())
    s_draws = p.mean(1)
    order = np.argsort(s_draws)
    cw = np.cumsum(w[order]) / w.sum()
    return {q: float(s_draws[order][np.searchsorted(cw, q)]) for q in (0.025, 0.1, 0.9, 0.975)}


def fit_of_size(n, seed):
    rng = np.random.default_rng(seed)
    return TaskFit(model="2pl", tasks=[f"t{i}" for i in range(n)], b=rng.normal(0, 1.5, n),
                   b_se=rng.uniform(0.2, 1.0, n), a=np.exp(rng.normal(0, 0.3, n)), theta={"h1": 0.0, "h2": 1.0},
                   theta_mean=0.2, theta_sd=1.1, tau=1.5, sigma_u=0.6, slope_sd=0.3, converged=True, grad_norm=0.0)


def test_the_80_and_95_percent_endpoints_match_a_joint_monte_carlo_on_a_small_and_a_mostly_unseen_suite():
    for n_tasks, results in ((12, {"t0": (1, 1), "t3": (0, 1), "t5": (1, 1), "t8": (2, 3)}),
                             (40, {"t1": (0, 1), "t2": (1, 1)})):
        f = fit_of_size(n_tasks, seed=n_tasks)
        est = CandidatePosterior(f, results).score()
        ref = joint_monte_carlo(f, results)
        assert abs(est["range80"][0] - ref[0.1]) < 0.015 and abs(est["range80"][1] - ref[0.9]) < 0.015
        assert abs(est["range95"][0] - ref[0.025]) < 0.02 and abs(est["range95"][1] - ref[0.975]) < 0.02
        assert 0.0 <= est["range95"][0] and est["range95"][1] <= 1.0


def test_grid_and_node_refinement_moves_the_estimate_and_its_range_by_little():
    f = fit_of_size(30, seed=4)
    results = {"t0": (1, 1), "t4": (0, 1), "t9": (1, 1), "t11": (0, 2)}
    coarse = CandidatePosterior(f, results).score()
    fine = CandidatePosterior(f, results, grid=401, nodes=32).score()
    assert abs(coarse["mean"] - fine["mean"]) < 0.002
    for k in ("range80", "range95"):
        assert max(abs(x - y) for x, y in zip(coarse[k], fine[k])) < 0.003


def test_after_every_task_once_the_models_range_is_still_not_zero():
    f = fit_of_size(40, seed=5)
    rng = np.random.default_rng(6)
    post = CandidatePosterior(f, {t: (int(rng.random() < 0.5), 1) for t in f.tasks})
    lo, hi = post.score()["range80"]
    assert hi - lo > 0.5 * 2 * post.full_run_halfwidth(0.8) > 0


def test_ranges_are_censored_to_0_and_1_on_one_task_and_at_extreme_rates():
    """A normal component can put mass outside [0, 1]; the ranges are the censored mixture's quantiles."""
    for b, runs in (([-8.0], {}), ([8.0], {}), ([-8.0], {"t0": (3, 3)}), ([0.0], {"t0": (1, 2)}),
                    ([-9.0, 9.0], {"t0": (1, 1), "t1": (0, 1)})):
        n = len(b)
        f = TaskFit(model="rasch", tasks=[f"t{i}" for i in range(n)], b=np.array(b), b_se=np.full(n, 0.5),
                    a=np.ones(n), theta={"h": 0.0}, theta_mean=0.0, theta_sd=1.0, tau=1.5, sigma_u=0.5,
                    slope_sd=0.3, converged=True, grad_norm=0.0)
        est = CandidatePosterior(f, runs).score()
        (l8, h8), (l9, h9) = est["range80"], est["range95"]
        assert 0.0 <= l9 <= l8 <= h8 <= h9 <= 1.0
        assert l9 <= est["mean"] <= h9
        mc = joint_monte_carlo(f, runs, draws=200_000)
        assert abs(l9 - mc[0.025]) < 0.02 and abs(h9 - mc[0.975]) < 0.02  # one task, mid rate: 0.015 wide


def test_with_a_candidate_slope_the_posterior_mean_matches_monte_carlo_and_0_changes_nothing():
    f = small_fit()
    results = {"t0": (0, 2), "t1": (1, 1), "t3": (1, 1), "t4": (1, 2)}
    for sl in (0.3, 0.6):
        est = CandidatePosterior(f, results, arm_slope_sd=sl).score()
        assert abs(est["mean"] - brute_force_mean(f, results, arm_slope_sd=sl)) < 0.01
    off, base = CandidatePosterior(f, results, arm_slope_sd=0.0), CandidatePosterior(f, results)
    assert off.score() == base.score() and np.array_equal(off.var_cut(), base.var_cut())


def test_a_candidate_slope_widens_the_range_for_a_flatter_arm_seen_only_near_its_ability():
    """The history arms share one curve per task; this candidate's is half as steep. Its runs are on the 12
    tasks nearest its ability, where the two curves agree, so only the slope term can tell the model that the
    easy and hard tasks may be off. Its 80% range then covers the score more often, and is wider."""
    T = 40
    b = np.linspace(-3, 3, T)
    f = TaskFit(model="2pl", tasks=[f"t{i}" for i in range(T)], b=b, b_se=np.full(T, 0.2), a=np.ones(T),
                theta={"h1": 0.0}, theta_mean=0.0, theta_sd=1.5, tau=1.5, sigma_u=0.3, slope_sd=0.3,
                converged=True, grad_norm=0.0)
    rng = np.random.default_rng(3)
    hits, widths, reps = {0.0: 0, 0.4: 0}, {0.0: [], 0.4: []}, 120
    near = np.argsort(np.abs(b - 1.0))[:12]
    for _ in range(reps):
        p = expit(0.5 * (1.0 - b) + rng.normal(0, np.sqrt(0.2**2 + 0.3**2), T))
        runs = {f.tasks[j]: (float(rng.binomial(3, p[j])), 3.0) for j in near}
        for sl in hits:
            lo, hi = CandidatePosterior(f, runs, arm_slope_sd=sl).score()["range80"]
            hits[sl] += lo <= p.mean() <= hi
            widths[sl].append(hi - lo)
    assert hits[0.4] / reps >= 0.75 and hits[0.4] > hits[0.0]
    assert np.mean(widths[0.4]) > np.mean(widths[0.0])
