"""`loopmath.tasks.select`: which task to run next, and when to stop.

Methods: `loss` (the largest expected cut in Var(S)), `info` (Fisher information at the current ability),
`random` (a seeded order per candidate), `stratified` (strata by predicted rate, Neyman allocation) and `fixed`
(one informative order for every candidate). Stops: precision (a full run's half-width, or a given one),
versus a bar, and the end of the suite.
"""

from __future__ import annotations

import itertools

import numpy as np

from loopmath.tasks.fit import TaskFit
from loopmath.tasks.posterior import CandidatePosterior
from loopmath.tasks.select import decide, next_tasks, random_order, stratified_mean


def fit_of(b, a=None, b_se=None, sigma_u=0.4):
    n = len(b)
    return TaskFit(model="2pl", tasks=[f"t{i}" for i in range(n)], b=np.asarray(b, float),
                   b_se=np.full(n, 0.3) if b_se is None else np.asarray(b_se, float),
                   a=np.ones(n) if a is None else np.asarray(a, float), theta={"h": 0.0}, theta_mean=0.0,
                   theta_sd=1.0, tau=1.5, sigma_u=sigma_u, slope_sd=0.3, converged=True, grad_norm=0.0)


def test_no_seen_task_is_picked_again_and_n_tasks_come_back_in_order_of_value():
    f = fit_of(np.linspace(-2, 2, 12))
    post = CandidatePosterior(f, {"t5": (1, 1), "t6": (0, 1)})
    for method in ("loss", "info", "random", "stratified", "fixed"):
        picks = next_tasks(post, method=method, n=4, seed=3, key="arm=new")
        names = [p["task"] for p in picks]
        assert len(names) == 4 and len(set(names)) == 4
        assert not {"t5", "t6"} & set(names)
    vals = [p["value"] for p in next_tasks(post, method="loss", n=4)]
    assert vals == sorted(vals, reverse=True)


def test_the_loss_rule_prefers_an_uncertain_task_to_a_floor_task():
    f = fit_of([6.0, 0.0], b_se=[0.3, 0.3])  # t0 is far out of reach, t1 sits at the candidate's level
    post = CandidatePosterior(f, {})
    assert next_tasks(post, method="loss", n=1)[0]["task"] == "t1"
    assert next_tasks(post, method="info", n=1)[0]["task"] == "t1"


def test_random_order_is_fixed_by_seed_and_candidate_and_differs_between_candidates():
    tasks = [f"t{i}" for i in range(30)]
    assert random_order(tasks, seed=1, key="arm=a") == random_order(tasks, seed=1, key="arm=a")
    assert random_order(tasks, seed=1, key="arm=a") != random_order(tasks, seed=1, key="arm=b")
    assert sorted(random_order(tasks, seed=1, key="arm=a")) == sorted(tasks)


def test_the_stratified_mean_is_exactly_unbiased_by_enumeration():
    strata = [[0.1, 0.3, 0.2], [0.9, 0.6, 0.7, 0.8]]  # true per-task rates, as the population
    truth = np.mean([x for s in strata for x in s])
    per = [list(itertools.combinations(s, 2)) for s in strata]  # 2 tasks drawn per stratum, without replacement
    ests = [stratified_mean([list(c0), list(c1)], [3, 4]) for c0 in per[0] for c1 in per[1]]
    assert abs(np.mean(ests) - truth) < 1e-12


def test_the_precision_stop_and_the_end_of_the_suite():
    f = fit_of(np.linspace(-2, 2, 10))
    post = CandidatePosterior(f, {f"t{i}": (i % 2, 1) for i in range(6)})
    assert decide(post, halfwidth=0.9, min_sessions=5).stop
    d = decide(post, halfwidth=0.001, min_sessions=5)
    assert not d.stop and "range" in d.reason
    assert not decide(post, halfwidth=0.9, min_sessions=8).stop
    done = CandidatePosterior(f, {f"t{i}": (1, 1) for i in range(10)})
    d = decide(done, halfwidth=0.001)
    assert d.stop and "every task" in d.reason
    assert decide(post, halfwidth=0.001, max_sessions=6).stop


def test_the_versus_stop_says_which_side_of_the_bar():
    f = fit_of(np.linspace(-3, -1, 12))  # easy tasks
    post = CandidatePosterior(f, {f"t{i}": (1, 1) for i in range(8)})
    d = decide(post, versus=0.3, level=0.9, min_sessions=5)
    assert d.stop and d.side == "above"
    d = decide(post, versus=0.99, level=0.9, min_sessions=5)
    assert d.side in ("below", None)


def test_the_sample_mean_range_is_not_zero_at_the_full_suite_and_covers_s_there_and_at_a_partial_n():
    """S is the mean of the probabilities; each run is a fresh Bernoulli draw, so run noise stays at n = T."""
    from loopmath.tasks.select import srs_estimate

    rng = np.random.default_rng(31)
    p = rng.beta(0.6, 0.6, 66)  # many tasks near 0 or 1, as in TB4
    truth = p.mean()
    for n in (66, 20):
        hits = 0
        reps = 2000
        widths = []
        for _ in range(reps):
            seen = rng.choice(66, n, replace=False)
            y = (rng.random(n) < p[seen]).astype(float)
            est = srs_estimate(y)
            lo, hi = est["range80"]
            widths.append(hi - lo)
            hits += lo <= truth <= hi
        assert min(widths) >= 0 and np.mean(widths) > 0.05
        assert hits / reps >= 0.78  # conservative: at or above nominal 80%
    est = srs_estimate(np.r_[np.ones(33), np.zeros(33)])
    assert est["range80"][1] - est["range80"][0] > 0.1


def test_a_near_tie_with_an_arm_of_few_runs_gives_about_even_odds_and_no_stop():
    f = fit_of(np.linspace(-1, 1, 20))
    runs = {f"t{i}": (i % 2, 1) for i in range(6)}
    cand, arm = CandidatePosterior(f, runs), CandidatePosterior(f, runs)
    d = decide(cand, versus=arm, level=0.9, min_sessions=5)
    assert not d.stop and abs(d.p_above - 0.5) < 0.05
    d = decide(cand, versus=0.5, level=0.9, min_sessions=5)
    assert d.p_above is not None and not d.stop


def test_one_run_or_all_equal_outcomes_keep_a_conservative_nonzero_range():
    from loopmath.tasks.select import difference_estimate, srs_estimate, stratified_estimate

    for y in ([1.0], [0.0], [1.0] * 5, [0.0] * 5):
        lo, hi = srs_estimate(y)["range80"]
        assert hi - lo > 0.05
    lo, hi = stratified_estimate([[1.0, 1.0, 1.0], [0.0]], [30, 36])["range80"]
    assert hi - lo > 0.1
    lo, hi = difference_estimate([1.0, 1.0, 1.0], [0.7, 0.7, 0.7], [0.5] * 10)["range80"]
    assert hi - lo > 0.1
    rng = np.random.default_rng(41)
    p = np.full(66, 0.95)  # most small samples pass every run
    for n in (8, 20):
        hits = 0
        for _ in range(2000):
            seen = rng.choice(66, n, replace=False)
            lo, hi = srs_estimate((rng.random(n) < p[seen]).astype(float))["range80"]
            hits += lo <= p.mean() <= hi
        assert hits / 2000 >= 0.78
