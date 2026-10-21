"""compare's cost per run when an arm's runs differ widely in cost: arms that mix cheap and dear models.

The target is each arm's mean dollars per run (cost per accepted result is a mean). A reviewed run that adds a
stage to the same sessions costs at least its unreviewed read, so a candidate whose every run costs at least
its partner is never reported cheaper. Single-model arms keep 0.3.1's numbers within a stated tolerance.
Synthetic data only.

Every test here runs under the switched cost model, `model.COST_MODEL = "sw"`: an experimental setting, not
the default. The default stays 0.3.1's lognormal. The fixture below sets it through monkeypatch, so no test
outside this file sees it.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from loopmath.compare import Observation, calibrate as C, model as M, paired


@pytest.fixture(autouse=True)
def switched_cost_model(monkeypatch):
    """The setting these tests check; the part is the default `COST_PART`."""
    monkeypatch.setattr(M, "COST_MODEL", "sw")


def reviewed_pairs(n: int, seed: int, cheap: float = 0.4):
    """Pairs whose candidate is its base run plus a positive extra stage. The base arm mixes a cheap model
    (about $0.05 a run) and a dear one (about $3); the extra stage costs about $1.40. Returns the runs and the
    two arms' true mean dollars per run."""
    rng = np.random.default_rng(seed)
    is_cheap = rng.random(n) < cheap
    b = np.where(is_cheap, np.exp(math.log(0.05) + 0.4 * rng.standard_normal(n)),
                 np.exp(math.log(3.0) + 0.5 * rng.standard_normal(n)))
    c = b + np.exp(math.log(1.4) + 0.4 * rng.standard_normal(n))
    ob, oc = rng.random(n) < 0.5, rng.random(n) < 0.6
    runs = ([Observation(f"t{i:03d}", "base", bool(ob[i]), float(b[i]), run=f"b{i:03d}") for i in range(n)],
            [Observation(f"t{i:03d}", "cand", bool(oc[i]), float(c[i]), run=f"c{i:03d}") for i in range(n)])
    mean_b = cheap * 0.05 * math.exp(0.4 ** 2 / 2) + (1 - cheap) * 3.0 * math.exp(0.5 ** 2 / 2)
    return runs, mean_b, mean_b + 1.4 * math.exp(0.4 ** 2 / 2)


def lognormal_pairs(n: int, seed: int, ratio: float = 1.0):
    """Single-model pairs as the calibration draws them: a task's log cost (sd 0.8) shared by both runs, run
    noise sd 0.5, the candidate at `ratio` times the base's cost."""
    rng = np.random.default_rng(seed)
    t = 0.8 * rng.standard_normal(n)
    xb = np.exp(t + 0.5 * rng.standard_normal(n))
    xc = np.exp(t + math.log(ratio) + 0.5 * rng.standard_normal(n))
    ob, oc = rng.random(n) < 0.6, rng.random(n) < 0.6
    return ([Observation(f"t{i:03d}", "base", bool(ob[i]), float(xb[i]), run=f"b{i:03d}") for i in range(n)],
            [Observation(f"t{i:03d}", "cand", bool(oc[i]), float(xc[i]), run=f"c{i:03d}") for i in range(n)])


# ---------------------------------------------------------------- the inversion (0.3.1 reported the reviewed arm cheaper)
@pytest.mark.parametrize("n", [200, 12])
def test_a_candidate_that_costs_more_on_every_pair_is_not_reported_cheaper(n):
    (base, cand), _, _ = reviewed_pairs(n, seed=3)
    assert all(c.usd >= b.usd for b, c in zip(base, cand))
    c = paired(base, cand, rule="tests", lookahead=False)
    assert c.cost_cand["mean"] >= c.cost_base["mean"]
    assert c.cost_ratio["median"] >= 1.0 and c.cost_ratio["mean"] >= 1.0
    if n >= 200:
        assert c.cost_ratio["bands"]["95"][0] >= 1.0  # P(ratio < 1) under 2.5%


@pytest.mark.parametrize("n", [1, 2, 3, 6, 24])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dominated_pairs_keep_their_order_at_any_number_of_pairs(n, seed):
    (base, cand), _, _ = reviewed_pairs(n, seed)
    c = paired(base, cand, rule="tests", lookahead=False)
    assert c.cost_cand["mean"] >= c.cost_base["mean"]
    assert c.cost_ratio["median"] >= 1.0


def test_a_mixed_arms_mean_cost_is_near_its_true_mean():
    (base, cand), mean_b, mean_c = reviewed_pairs(200, seed=4)
    c = paired(base, cand, rule="tests", lookahead=False)
    assert c.cost_base["mean"] == pytest.approx(mean_b, rel=0.25)
    assert c.cost_cand["mean"] == pytest.approx(mean_c, rel=0.25)
    assert c.cost_base["bands"]["95"][0] <= mean_b <= c.cost_base["bands"]["95"][1]


def test_lone_costed_runs_count_toward_their_arm():
    """Half the candidate's costs are missing, so those base runs are lone costed runs."""
    (base, cand), mean_b, mean_c = reviewed_pairs(200, seed=5)
    cand = [Observation(o.task, o.arm, o.outcome, o.usd if i % 2 else None, run=o.run) for i, o in enumerate(cand)]
    c = paired(base, cand, rule="tests", lookahead=False)
    assert c.cost_base["mean"] == pytest.approx(mean_b, rel=0.25)
    assert c.cost_cand["mean"] == pytest.approx(mean_c, rel=0.25)


def test_the_80_percent_range_holds_a_mixed_arms_true_mean_about_80_percent_of_the_time():
    hits = 0
    for seed in range(200):
        (base, cand), mean_b, _ = reviewed_pairs(24, seed=1000 + seed)
        c = paired(base, cand, rule="tests", lookahead=False)
        hits += c.cost_base["lo"] <= mean_b <= c.cost_base["hi"]
    assert 140 <= hits <= 180, hits  # 0.8 of 200, give or take about 3.5 binomial sds


def test_costs_spread_over_six_orders_of_magnitude_still_get_a_cost_range():
    """0.3.1 states no cost range here (its log-cost sds stop at 3) and falls back to the gain."""
    xs = [1e-4 if i % 2 == 0 else 100.0 for i in range(48)]
    base = [Observation(f"t{i}", "base", i % 3 == 0, x, run=f"b{i}") for i, x in enumerate(xs)]
    cand = [Observation(f"t{i}", "cand", i % 2 == 0, 1.1 * x, run=f"c{i}") for i, x in enumerate(xs)]
    c = paired(base, cand, rule="tests", lookahead=False)
    assert c.cost_base is not None and c.cost_cand is not None
    assert c.objective == "cpar" and not c.outcomes_only
    assert c.cost_base["bands"]["95"][0] <= 50.00005 <= c.cost_base["bands"]["95"][1]  # the arm's raw mean


# ---------------------------------------------------------------- single-model arms: 0.3.1's numbers within tolerance
# 0.3.1's (mean, lo, hi) of cost_base, cost_cand and cost_ratio on lognormal_pairs(n, seed, ratio), seeds 0 to 9.
V031 = Path(__file__).parents[1] / "fixtures" / "compare" / "cost_v031_single_model.json"
MEAN_TOL = 0.25  # in units of 0.3.1's 80% range width W on the same data: the posterior mean
END_TOL = 0.40  # in units of W: each end of the 80% range
# Both at the 90th percentile over the 10 data sets (numpy's linear percentile).


@pytest.mark.parametrize("key", ["12,1.0", "12,0.8", "48,1.0", "48,0.8"])
def test_single_model_arms_keep_their_numbers_within_tolerance(key):
    n, ratio = key.split(",")
    moves: dict = {}
    for seed, ref in enumerate(json.loads(V031.read_text())["sets"][key]):
        c = paired(*lognormal_pairs(int(n), seed, float(ratio)), rule="tests", lookahead=False)
        for name, (mean, lo, hi) in ref.items():
            iv, width = getattr(c, name), hi - lo
            for what, got, was in (("mean", iv["mean"], mean), ("lo", iv["lo"], lo), ("hi", iv["hi"], hi)):
                moves.setdefault((name, what), []).append(abs(got - was) / width)
    for (name, what), m in moves.items():
        tol = MEAN_TOL if what == "mean" else END_TOL
        assert np.percentile(m, 90) <= tol, (name, what, np.round(m, 3))


# ---------------------------------------------------------------- small n
def test_one_pair_states_a_cost_range_that_is_not_a_point():
    base = [Observation("t0", "base", True, 0.4, run="b0")]
    cand = [Observation("t0", "cand", False, 0.5, run="c0")]
    c = paired(base, cand, rule="tests", lookahead=False)
    for iv in (c.cost_base, c.cost_cand, c.cost_ratio):
        assert iv is not None and iv["lo"] < iv["hi"]


def test_ranges_narrow_as_pairs_come_in():
    def rel_width(n):  # the 80% range over the median, averaged over 10 draws of the same arms
        out = []
        for seed in range(10):
            (base, cand), _, _ = reviewed_pairs(n, seed=50 + seed)
            c = paired(base, cand, rule="tests", lookahead=False)
            out.append((c.cost_base["hi"] - c.cost_base["lo"]) / c.cost_base["median"])
        return float(np.mean(out))

    assert rel_width(2) > rel_width(24) > rel_width(200)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_where_the_dp_part_answers_a_few_one_way_pairs_its_ratio_range_cannot_hold_1(n):
    """The model's stated limit at few pairs: when every pair orders the arms one way and there are no other
    runs, every draw of the DP part keeps that order, so where it answers, the cost ratio's range lies wholly
    on that side of 1 and cannot hold a true ratio of 1. Under 6 pairs it answers only when the lognormal's
    means order the arms against every pair or the lognormal states no range, so not on every data set."""
    answered = 0
    for seed in range(10):
        (base, cand), _, _ = reviewed_pairs(n, seed=200 + seed)
        ob, oc = np.array([o.outcome for o in base]), np.array([o.outcome for o in cand])
        d = M.Data(kind="binary")
        d.cells = np.array([np.sum(ob & oc), np.sum(~ob & oc), np.sum(ob & ~oc), np.sum(~ob & ~oc)], dtype=float)
        d.cost_pairs = [(math.log(b.usd), math.log(c.usd)) for b, c in zip(base, cand)]
        post = M.posterior(d, np.random.default_rng(seed), 2000)
        if post.cost is None or not post.cost.dp.all():
            continue  # the lognormal answered
        answered += 1
        assert np.all(post.mean_cost("cand") / post.mean_cost("base") > 1.0)
    assert answered >= 1


# ---------------------------------------------------------------- the seams the value code reads
def _data(n=12):
    (base, cand), _, _ = reviewed_pairs(n, seed=6)
    d = M.Data(kind="binary")
    d.cells = np.array([3.0, 3.0, 3.0, 3.0])
    d.cost_pairs = [(math.log(b.usd), math.log(c.usd)) for b, c in zip(base, cand)]
    return d


@pytest.mark.parametrize("n", [1, 12])
def test_mean_cost_is_one_positive_draw_per_posterior_draw(n):
    post = M.posterior(_data(n), np.random.default_rng(0), 500)
    for arm in M.ARMS:
        m = post.mean_cost(arm)
        assert m.shape == (500,) and np.all(np.isfinite(m)) and np.all(m > 0)


def test_future_pairs_keep_their_keys_and_shapes():
    d = _data()
    post = M.posterior(d, np.random.default_rng(0), 500)
    fut = M.draw_future(d, post, 3, 4, np.random.default_rng(1))
    assert np.asarray(fut["cost_pairs"]).shape == (4, 2) and np.all(np.isfinite(fut["cost_pairs"]))
    after = M.extend(d, fut, 2)
    assert len(after.cost_pairs) == len(d.cost_pairs) + 2
    assert all(len(p) == 2 and all(math.isfinite(x) for x in p) for p in after.cost_pairs)


def test_future_pairs_come_from_the_same_model_as_the_mean_cost():
    base, cand = lognormal_pairs(12, seed=6)
    d = M.Data(kind="binary")
    d.cells = np.array([3.0, 3.0, 3.0, 3.0])
    d.cost_pairs = [(math.log(b.usd), math.log(c.usd)) for b, c in zip(base, cand)]
    post = M.posterior(d, np.random.default_rng(0), 200)
    for i in range(5):
        fut = M.draw_future(d, post, i, 20000, np.random.default_rng(10 + i))
        got = np.exp(np.asarray(fut["cost_pairs"])).mean(axis=0)
        for a, arm in enumerate(M.ARMS):
            assert got[a] == pytest.approx(post.mean_cost(arm)[i], rel=0.05)


def test_the_gain_reads_no_cost():
    (base, cand), _, _ = reviewed_pairs(24, seed=8)
    scaled = ([Observation(o.task, o.arm, o.outcome, 1000.0 * o.usd, run=o.run) for o in base],
              [Observation(o.task, o.arm, o.outcome, o.usd if i % 2 else 1e-4, run=o.run) for i, o in enumerate(cand)])
    a = paired(base, cand, rule="tests", objective="gain", lookahead=False)
    b = paired(*scaled, rule="tests", objective="gain", lookahead=False)
    assert a.p_better == b.p_better and a.gain == b.gain


# P(better) at looks 6, 12, 24 and 48 of the first two paths of each mode's first null family (tune seed 1,
# chunk 0), at 0.3.1. The binary paths carry costs, so the cost model runs beside the gain.
P031 = {"gain": [[0.1055, 0.071, 0.09825, 0.0215], [0.21325, 0.05075, 0.0175, 0.00975]],
        "score": [[0.420852, 0.699719, 0.782458, 0.617488], [0.839192, 0.989579, 0.834118, 0.672989]]}


@pytest.mark.parametrize("mode", ["gain", "score"])
def test_the_gain_and_the_score_keep_their_answers_on_calibration_paths(mode):
    fam = C.NULLS[mode][0]
    rng = C._rng(C.SEEDS["tune"], mode, fam, 0)
    for want in P031[mode]:
        path = C.simulate(fam, rng, 48)
        got = [paired(*C.observations(mode, path, k), rule=C.rule_for(mode), objective=C.OBJECTIVE[mode],
                      lookahead=False).p_better for k in (6, 12, 24, 48)]
        assert got == want


# ---------------------------------------------------------------- calibration nulls whose arms mix models
def test_mixed_model_costs_are_drawn_after_everything_else_and_shared_by_a_pair():
    for fam in C.MIXED_COST_NULLS:
        plain = dataclasses.replace(fam, cheap=0.0)
        a = C.simulate(plain, np.random.default_rng(5), 2000)
        b = C.simulate(fam, np.random.default_rng(5), 2000)
        for k in ("ob", "oc", "extra", "eo"):
            assert np.array_equal(a[k], b[k])
        sb, sc = b["xb"] - a["xb"], b["xc"] - a["xc"]
        assert np.array_equal(np.isnan(sb), np.isnan(a["xb"])) and np.array_equal(np.isnan(sc), np.isnan(a["xc"]))
        both = np.isfinite(sb) & np.isfinite(sc)
        assert np.allclose(sb[both], sc[both])  # one model per pair
        assert np.all(np.isclose(sb[both], 0.0) | np.isclose(sb[both], math.log(fam.cheap_cost)))
        assert abs(np.mean(sb[both] < 0) - fam.cheap) < 0.04


def test_the_mixed_model_families_are_nulls_for_cost_per_accepted_result():
    for fam in C.MIXED_COST_NULLS:
        # both arms share the mix, so each arm's mean cost is its single-model mean times the same factor
        assert fam.cost_ratio / fam.pc == pytest.approx(1 / fam.pb)
