"""`compare/model.py`: proper small-n posteriors, means only where they exist, a lookahead drawn from the
posterior that gave the decision, and the paired cost model. Synthetic data only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from loopmath.compare import Observation, model as M, paired


def scores(base, cand):
    return ([Observation(f"t{i:02d}", "base", float(b), 1.0, run=f"b{i:02d}") for i, b in enumerate(base)],
            [Observation(f"t{i:02d}", "cand", float(c), 1.0, run=f"c{i:02d}") for i, c in enumerate(cand)])


def binary(cells, *, lone=None, k=8, ratio=1.0, seed=0):
    rng = np.random.default_rng(seed)
    d = M.Data(kind="binary")
    d.cells = np.asarray(cells, dtype=float)
    if lone:
        d.lone = lone
    t = rng.normal(0, 0.8, k)
    d.cost_pairs = list(zip(t + rng.normal(0, 0.5, k), t + math.log(ratio) + rng.normal(0, 0.5, k)))
    return d


# ---------------------------------------------------------------- B2: small n
def test_two_identical_score_differences_give_a_range_not_certainty():
    c = paired(*scores([0, 0], [1, 1]), rule="score:quality", lookahead=False)
    assert c.p_better < 0.99 and c.verdict != "adopt"
    assert c.gain["lo"] < 1 < c.gain["hi"] and c.gain["bands"]["95"][0] < c.gain["lo"]
    assert c.gain["mean"] == pytest.approx(1.0)  # two pairs: t with 2 df, whose mean exists


def test_one_score_pair_has_no_gain_mean_but_keeps_its_median_and_ranges():
    c = paired(*scores([0], [1]), rule="score:quality", lookahead=False)
    assert c.gain["mean"] is None and c.improvement["mean"] is None  # t with 1 df has no mean
    assert c.gain["median"] == pytest.approx(1.0, abs=0.1) and c.gain["lo"] < c.gain["hi"]
    assert c.p_better < 0.9


def test_constant_and_identical_scores_stay_proper():
    c = paired(*scores(range(12), [i + 0.5 for i in range(12)]), rule="score:quality", lookahead=False)
    assert c.gain["lo"] < 0.5 < c.gain["hi"] and c.p_better < 1.0
    same = paired(*scores([3.0] * 6, [3.0] * 6), rule="score:quality", lookahead=False)
    assert same.p_better == pytest.approx(0.5) and same.gain["lo"] < 0 < same.gain["hi"]


def runs(outcomes_b, outcomes_c, usd=1.0):
    return ([Observation(f"t{i:02d}", "base", o, usd, run=f"b{i:02d}") for i, o in enumerate(outcomes_b)],
            [Observation(f"t{i:02d}", "cand", o, usd, run=f"c{i:02d}") for i, o in enumerate(outcomes_c)])


def test_no_accepted_run_leaves_the_cpar_mean_null_and_keeps_the_median():
    c = paired(*runs([False] * 6, [False] * 6), rule="tests", lookahead=False)
    for iv in (c.cpar_base, c.cpar_cand, c.improvement):
        assert iv["mean"] is None
    assert c.cpar_base["median"] is not None and c.cpar_base["lo"] < c.cpar_base["hi"]
    assert c.gain["mean"] is not None
    assert any("no base run is accepted" in r for r in c.reasons)
    one = paired(*runs([False] * 6, [True] + [False] * 5), rule="tests", lookahead=False)
    assert one.cpar_base["mean"] is None and one.cpar_cand["mean"] is not None and one.improvement["mean"] is not None


# ---------------------------------------------------------------- B4: the lookahead's posterior
def test_the_predictive_reads_the_lone_runs():
    d = M.Data(kind="binary")
    d.lone = {"base": (0, 40), "cand": (40, 0)}  # 40 failed base runs, 40 accepted candidate runs, no pairs
    post = M.posterior(d, np.random.default_rng(0))
    pb, pc = float(post.chance("base").mean()), float(post.chance("cand").mean())
    assert pb < 0.05 and pc > 0.95
    rng = np.random.default_rng(1)
    picks = rng.integers(post.n, size=4000)
    cells = np.array([M.draw_future(d, post, int(i), 1, rng)["cells"][0] for i in picks])
    base_ok, cand_ok = np.isin(cells, (0, 2)), np.isin(cells, (0, 1))
    for got, want in ((base_ok.mean(), pb), (cand_ok.mean(), pc)):
        assert abs(got - want) <= 4 * math.sqrt(want * (1 - want) / picks.size) + 1e-3


def test_expected_updated_means_equal_the_current_means():
    """The martingale identity: averaged over futures drawn from the posterior, the updated posterior mean is
    the current one."""
    rng = np.random.default_rng(2)
    d = binary([2, 3, 1, 2], lone={"base": (1, 3), "cand": (0, 0)}, k=8)
    d.cost_lone = {"base": [0.4], "cand": []}
    post = M.posterior(d, np.random.default_rng(0))
    now_b, now_mu = float(post.chance("base").mean()), post.cmu[:, 0].mean()
    after_b, after_mu = [], []
    for i in rng.integers(post.n, size=300):
        f = M.future(d, post, int(i), 4, rng)
        p = M.posterior(f, rng, 2000)
        after_b.append(p.chance("base").mean())
        after_mu.append(p.cmu[:, 0].mean())
    for now, after in ((now_b, after_b), (now_mu, after_mu)):
        after = np.asarray(after)
        assert abs(after.mean() - now) <= 4 * after.std(ddof=1) / math.sqrt(after.size) + 0.005

    s = M.Data(kind="score", diffs=[0.3, -0.1, 0.8, 0.2], scale=0.6)
    sp = M.posterior(s, np.random.default_rng(0))
    locs = [M.score_t(M.future(s, sp, int(i), 3, rng))[1] for i in rng.integers(sp.n, size=2000)]
    assert abs(np.mean(locs) - M.score_t(s)[1]) <= 4 * np.std(locs) / math.sqrt(len(locs))


@pytest.mark.parametrize("cells,ratio,k,stop", [
    ([2, 2, 2, 2], 1.0, 8, True),  # flat: the next pairs cannot pay
    ([5, 1, 1, 1], 0.6, 8, True),
    ([2, 6, 0, 1], 0.8, 9, False),  # leaning to the candidate: worth more pairs
])
def test_the_stop_decision_holds_as_the_futures_grow(cells, ratio, k, stop):
    d = binary(cells, ratio=ratio, k=k)
    st = M.Settings(threshold=0.99)
    post = M.posterior(d, np.random.default_rng(0))
    dd = M.draws(d, "cpar", np.random.default_rng(0))
    price = float(np.mean(dd.cost_base) + np.mean(dd.cost_cand))
    values = [M.next_pair_value(d, st, post, np.random.default_rng(seed), max_pairs=16, sims=sims)[0]
              for seed, sims in ((1, M.LOOK_SIMS), (2, M.LOOK_SIMS), (3, 3 * M.LOOK_SIMS))]
    assert all((v < price) == stop for v in values), (values, price)
    if not stop:
        assert abs(values[0] - values[2]) <= 0.5 * values[2] and abs(values[1] - values[2]) <= 0.5 * values[2]


def test_the_lookahead_stops_at_the_cap():
    d = binary([2, 6, 0, 1], ratio=0.8, k=9)
    post = M.posterior(d, np.random.default_rng(0))
    st = M.Settings(threshold=0.99)
    assert M.next_pair_value(d, st, post, np.random.default_rng(1), max_pairs=0) == (0.0, None)
    _, m = M.next_pair_value(d, st, post, np.random.default_rng(1), max_pairs=3, sims=32)
    assert m is None or m <= 3


# ---------------------------------------------------------------- N1: the paired cost model
def _cost_data(rng, k, mu, sd, rho, drop):
    cov = np.array([[sd[0] ** 2, rho * sd[0] * sd[1]], [rho * sd[0] * sd[1], sd[1] ** 2]])
    xy = rng.multivariate_normal(mu, cov, size=k)
    d = M.Data(kind="binary")
    for x, y in xy:
        if rng.random() < drop:
            d.cost_lone["base"].append(float(x))  # the candidate's cost is missing: a lone base cost
        else:
            d.cost_pairs.append((float(x), float(y)))
    if not d.cost_pairs:
        d.cost_pairs.append(tuple(map(float, xy[0])))
        d.cost_lone["base"] = d.cost_lone["base"][1:]
    return d


@pytest.mark.parametrize("k", [6, 12])
def test_paired_costs_recover_the_truth_with_80_percent_ranges(k):
    mu, sd, rho = (0.0, -0.3), (0.9, 0.9), 0.9  # strong task correlation
    truth = [math.exp(mu[a] + sd[a] ** 2 / 2) for a in (0, 1)]
    rng = np.random.default_rng(10 + k)
    hits = np.zeros(3)
    reps = 200
    for _ in range(reps):
        d = _cost_data(rng, k, mu, sd, rho, drop=0.2)
        post = M.posterior(d, rng, 2000)
        cb, cc = post.mean_cost("base"), post.mean_cost("cand")
        for j, (draws, want) in enumerate(((cb, truth[0]), (cc, truth[1]), (cc / cb, truth[1] / truth[0]))):
            lo, hi = np.quantile(draws, [0.1, 0.9])
            hits[j] += lo <= want <= hi
    cover = hits / reps
    assert np.all((cover >= 0.7) & (cover <= 0.92)), cover


def test_one_costed_pair_and_lone_costs_give_finite_ranges_in_the_prior_support():
    d = M.Data(kind="binary", cost_pairs=[(0.1, -0.2)], cost_lone={"base": [0.3, -0.4], "cand": []})
    post = M.posterior(d, np.random.default_rng(0))
    for arm in ("base", "cand"):
        c = post.mean_cost(arm)
        assert np.all(np.isfinite(c)) and np.quantile(c, 0.1) < np.quantile(c, 0.9)
    var = post.ccov[:, [0, 1], [0, 1]]
    assert var.min() >= M.SIGMA_RANGE[0] ** 2 - 1e-12 and var.max() <= M.SIGMA_RANGE[1] ** 2 + 1e-12
    alone = M.posterior(M.Data(kind="binary", cost_lone={"base": [0.2, 0.5], "cand": []}), np.random.default_rng(0))
    assert alone.mean_cost("cand") is None and np.all(np.isfinite(alone.mean_cost("base")))


def test_costs_far_outside_the_sd_range_are_refused_and_every_returned_draw_is_in_the_support():
    tiny, huge = math.log(1e-6), math.log(1e9)  # $0.000001 and $1e9: a log-cost sd near 24, the support ends at 3
    for d in (M.Data(kind="binary", cost_pairs=[(tiny, tiny), (huge, huge)]),
              M.Data(kind="binary", cost_lone={"base": [tiny, huge], "cand": []})):
        p = M.posterior(d, np.random.default_rng(160), 256)
        assert p.ccov is None and p.cmu is None and "no cost range is stated" in p.cost_refused
    base = [Observation(f"t{i}", "base", True, usd) for i, usd in enumerate((1e-6, 1e9))]
    cand = [Observation(f"t{i}", "cand", True, usd) for i, usd in enumerate((1e-6, 1e9))]
    c = paired(base, cand, rule="tests", lookahead=False)
    assert c.cost_base is None and c.cpar_cand is None and any("no cost range is stated" in r for r in c.reasons)
    p = M.posterior(M.Data(kind="binary", cost_pairs=[(0.0, 0.1), (1.0, 1.4)]), np.random.default_rng(1), 256)
    assert p.cost_refused is None and M._in_range(p.ccov).all() and (np.linalg.eigvalsh(p.ccov)[:, 0] > 0).all()
