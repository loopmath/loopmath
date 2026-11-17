"""designcheck.py: the design-based check of `eval value --check design`. All data is synthetic."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import hypergeom

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import designcheck as DC
from loopmath.eval import fitfile as F
from loopmath.eval import quality as Q
from loopmath.eval import testing

DEPTH = 2          # final/depth in the example rubric: linear, 0 to 10
STAMP = "2026-01-05T10:00:00+00:00"


# ---------------------------------------------------------------------------------------------------
# The estimate, its variance and its interval

def population(n: int, rng: np.random.Generator, shift: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Predictions `f` and human scores `y` for `n` targets. The prediction is off by `shift` more where
    it is low."""
    f = np.clip(0.55 + 0.15 * rng.standard_normal(n), 0, 1)
    y = np.clip(f + 0.05 - shift * (f < 0.5) + 0.1 * rng.standard_normal(n), 0, 1)
    return f, y


def two_rate_sample(f: np.ndarray, n_low: int, n_high: int, rng: np.random.Generator):
    """A sample stratified by prediction band, as a pilot draws: a fixed number from each band, so
    targets with a low prediction have a higher inclusion probability."""
    low, high = np.flatnonzero(f < 0.5), np.flatnonzero(f >= 0.5)
    pi = np.where(f < 0.5, n_low / len(low), n_high / len(high))
    picked = np.concatenate([rng.choice(low, n_low, replace=False), rng.choice(high, n_high, replace=False)])
    return picked, pi


def by_size_sample(size: np.ndarray, n: int, rng: np.random.Generator, floor: float = 0.05):
    """A fixed-size sample without replacement with inclusion probability proportional to `size` and a
    floor, as an active plan draws: systematic sampling in a random order."""
    pi = n * size / size.sum()
    for _ in range(50):
        pi = np.clip(pi, floor, 1.0)
        free = (pi > floor) & (pi < 1.0)
        pi[free] *= (n - pi[~free].sum()) / pi[free].sum()
    order = rng.permutation(len(size))
    edges = np.cumsum(pi[order])
    picks = np.searchsorted(edges, rng.random() + np.arange(int(round(edges[-1] - 1e-9))), side="right")
    return order[picks], pi


def coverage(design: str, n: int, shift: float, reps: int = 1000, seed: int = 11) -> tuple[float, str]:
    """The share of samples whose 90% interval holds the true mean human score of a fixed population."""
    rng = np.random.default_rng(seed)
    f, y = population(400, rng, shift)
    hits, kinds = 0, set()
    for _ in range(reps):
        if design == "bands":
            picked, pi = two_rate_sample(f, n // 2, n // 2, rng)
        else:
            picked, pi = by_size_sample(0.2 + np.abs(f - 0.5) ** 0.5, n, rng)
        out = DC.interval(f, y[picked], f[picked], pi[picked], rng)
        hits += out["lo"] <= y.mean() <= out["hi"]
        kinds.add(out["interval"])
    assert len(kinds) == 1
    return hits / reps, kinds.pop()


@pytest.mark.parametrize("design, n, interval", [("bands", 80, "normal"), ("bands", 40, "bootstrap"),
                                                 ("by_size", 80, "normal"), ("by_size", 40, "bootstrap")])
def test_interval_covers_the_true_human_mean_at_its_stated_level(design, n, interval):
    """The design-based part of acceptance check 12, at the level of the estimator: coverage of the 90%
    interval between 85% and 95%, for a sample stratified by prediction band (pilot) and for one drawn
    with unequal probabilities (active), with the normal interval (50 labels or more) and the bootstrap
    interval (below 50). Measured: 0.899, 0.920, 0.869 and 0.899."""
    rate, kind = coverage(design, n, shift=0.05)
    assert kind == interval and 0.85 <= rate <= 0.95, rate


def test_interval_is_wide_when_the_prediction_error_differs_by_band():
    """A known limit, kept visible: the variance has one centre, so when a sample is stratified by
    prediction band and the prediction is off by much more in one band (here by 0.2, twice the noise),
    the interval is wider than its level says. Measured: 0.960 and 0.976. It never covers less."""
    rate, kind = coverage("bands", 80, shift=0.2)
    assert kind == "normal" and 0.95 < rate <= 0.99, rate
    rate, kind = coverage("bands", 40, shift=0.2)
    assert kind == "bootstrap" and 0.95 < rate <= 0.99, rate


def test_estimate_is_unbiased_where_the_plain_mean_of_the_labels_is_not():
    rng = np.random.default_rng(3)
    f, y = population(400, rng)
    weighted, plain = [], []
    for _ in range(600):
        picked, pi = two_rate_sample(f, 40, 40, rng)
        weighted.append(DC.estimate(f, y[picked], f[picked], pi[picked]))
        plain.append(y[picked].mean())
    assert np.mean(weighted) == pytest.approx(y.mean(), abs=0.004)
    assert abs(np.mean(plain) - y.mean()) > 0.02
    # the variance estimate is close to the variance over samples
    picked, pi = two_rate_sample(f, 40, 40, rng)
    estimates = []
    variances = []
    for _ in range(600):
        picked, pi = two_rate_sample(f, 40, 40, rng)
        estimates.append(DC.estimate(f, y[picked], f[picked], pi[picked]))
        variances.append(DC.variance(y[picked], f[picked], pi[picked]))
    # the variance estimate does not fall under the variance over samples (see the limit above)
    assert np.var(estimates) <= np.mean(variances) <= 2.5 * np.var(estimates)
    picked, pi = by_size_sample(0.2 + np.abs(f - 0.5) ** 0.5, 80, rng)
    assert len(picked) == 80 and len(set(picked.tolist())) == 80 and pi.sum() == pytest.approx(80)


def test_estimate_and_variance_in_closed_form():
    rng = np.random.default_rng(4)
    f, y = population(60, rng)
    # a census: the estimate is the mean of the human scores, with no sampling variance
    ones = np.ones(60)
    assert DC.estimate(f, y, f, ones) == pytest.approx(y.mean(), abs=1e-12)
    assert DC.variance(y, f, ones) == 0.0 and DC.variance(y[:1], f[:1], np.array([0.5])) == 0.0
    # equal probabilities: the simple random sampling formula with the finite population correction
    n, share = 30, 0.5
    e = (y - f)[:n]
    pi = np.full(n, share)
    assert DC.estimate(f, y[:n], f[:n], pi) == pytest.approx(f.mean() + e.mean(), abs=1e-12)
    assert DC.variance(y[:n], f[:n], pi) == pytest.approx((1 - share) * e.var(ddof=1) / n, rel=1e-9)
    assert DC.variance(y[:n], f[:n], pi, finite=False) == pytest.approx(e.var(ddof=1) / n, rel=1e-9)
    # without predictions it is the weighted mean of the labels
    zeros = np.zeros(60)
    weights = np.where(np.arange(n) < 10, 0.2, 0.8)
    assert DC.estimate(zeros, y[:n], zeros[:n], weights) == pytest.approx(np.average(y[:n], weights=1 / weights))
    out = DC.interval(f, y[:n], f[:n], pi, np.random.default_rng(0))
    assert out["interval"] == "bootstrap" and out["lo"] < out["mean"] < out["hi"] and out["sd"] > 0
    out = DC.interval(f, y, f, np.full(60, 0.5), np.random.default_rng(0))
    assert out["interval"] == "normal" and out["hi"] - out["mean"] == pytest.approx(1.6449 * out["sd"], rel=1e-4)


def test_expected_reading_matches_a_simulation_for_every_scale():
    rng = np.random.default_rng(5)
    q = np.array([[-0.4, 0.1, 0.45, 0.9]])
    bias, noise = np.array([0.1]), np.array([0.3])
    eps = noise[0] * rng.standard_normal((400_000, 1))
    eta = q + bias[0] + eps
    assert DC.expected_reading(q, "linear", bias, noise) == pytest.approx(q + 0.1)
    assert DC.expected_reading(q, "fraction", bias, noise) == pytest.approx(q + 0.1)
    assert DC.expected_reading(q, "binary", bias, noise)[0] == pytest.approx((eta >= 0).mean(axis=0), abs=0.004)
    # each item passes with chance Phi(eta)
    assert DC.expected_reading(q, "items", bias, noise)[0] == pytest.approx(ndtr(eta).mean(axis=0), abs=0.004)
    cuts = np.array([[0.25, 0.75]])
    levels = (eta[:, :, None] >= cuts[0]).sum(axis=2) / 2
    assert DC.expected_reading(q, "levels", bias, noise, cuts)[0] == pytest.approx(levels.mean(axis=0), abs=0.004)
    # a judge without noise reads the level, or the pass, exactly
    still = np.array([0.0])
    assert DC.expected_reading(q, "levels", still, still, cuts)[0].tolist() == [0.0, 0.0, 0.5, 1.0]
    assert DC.expected_reading(q, "binary", still, still)[0].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_observed_puts_every_scale_on_0_to_1():
    dims = {dim["component"]: dim for dim in C.dimensions(testing.example_rubric())}
    assert DC.observed(dims["depth"], 7.5) == 0.75 and DC.observed(dims["clarity"], "ok") == 0.5
    assert DC.observed(dims["numbers_right"], 1) == 1.0 and DC.observed(dims["coverage"], 3, 4) == 0.75
    with pytest.raises(ValueError):
        DC.observed(dims["clarity"], "nothing")


# ---------------------------------------------------------------------------------------------------
# Bounded scores: a census, and samples that show no spread

def uniform_sample(n_targets: int, n: int):
    return lambda rng: (rng.choice(n_targets, n, replace=False), np.full(n_targets, n / n_targets))


def two_strata(n_targets: int, n_first: int, take_first: int, take_rest: int):
    """A fixed number from the first `n_first` targets and from the rest."""
    def draw(rng):
        pi = np.where(np.arange(n_targets) < n_first, take_first / n_first, take_rest / (n_targets - n_first))
        rest = n_first + rng.choice(n_targets - n_first, take_rest, replace=False)
        return np.concatenate([rng.choice(n_first, take_first, replace=False), rest]), pi
    return draw


def failures(n_targets: int, at) -> np.ndarray:
    y = np.ones(n_targets)
    y[np.asarray(at)] = 0.0
    return y


def binary_coverage(f: np.ndarray, y: np.ndarray, sampler, reps: int = 500, seed: int = 0):
    """`(coverage, share of intervals by kind)` of the true mean `y.mean()` over repeated samples."""
    rng = np.random.default_rng(seed)
    hits, kinds = 0, {}
    for _ in range(reps):
        picked, pi = sampler(rng)
        out = DC.interval(f, y[picked], f[picked], pi[picked], rng)
        hits += out["lo"] - 1e-12 <= y.mean() <= out["hi"] + 1e-12
        kinds[out["interval"]] = kinds.get(out["interval"], 0) + 1 / reps
    return hits / reps, kinds


def test_a_census_is_exact():
    """Twenty targets, all labelled with certainty, half of them failing: the mean is 0.5 and there is
    no sampling error to show."""
    y = np.array([0.0, 1.0] * 10)
    out = DC.interval(np.ones(20), y, np.ones(20), np.ones(20), np.random.default_rng(0))
    assert out == {"mean": 0.5, "sd": 0.0, "lo": 0.5, "hi": 0.5, "interval": "census", "unseen": 0}
    # targets drawn with certainty add no width to a sample either: only the 20 drawn at 0.5 do
    rng = np.random.default_rng(1)
    e = np.concatenate([rng.standard_normal(10), 0.05 * rng.standard_normal(20)])
    f = np.full(30, 0.5)
    pi = np.concatenate([np.ones(10), np.full(20, 0.5)])
    out = DC.interval(np.full(50, 0.5), f + e, f, pi, np.random.default_rng(0))
    # the textbook sd of a mean of 20 from 40, carried to all 50 targets
    textbook = 40 / 50 * np.sqrt((1 - 0.5) * e[10:].var(ddof=1) / 20)
    assert out["interval"] == "bootstrap" and out["sd"] == pytest.approx(textbook, rel=0.1)


def test_one_rare_failure_is_not_reported_as_a_point():
    """100 binary targets, one true failure, a judge that predicts a pass everywhere, and a uniform
    sample of 40. A sample without the failure (6 in 10 samples) must not report [1, 1]."""
    n_targets, n = 100, 40
    f, y = np.ones(n_targets), failures(n_targets, [0])
    seen_none = np.arange(1, n + 1)
    out = DC.interval(f, y[seen_none], f[seen_none], np.full(n, 0.4), np.random.default_rng(0))
    # independent of the code: the largest number of failures such a sample still misses with a chance
    # above 5%, from the hypergeometric distribution
    most = max(k for k in range(60) if hypergeom(n_targets, k, n).pmf(0) > 0.05)
    assert most == 5 and hypergeom(n_targets, 1, n).pmf(0) == pytest.approx(0.6)
    assert out["interval"] == "bound" and out["unseen"] == 5 and out["sd"] == 0.0
    assert out["hi"] == 1.0 and out["lo"] == pytest.approx(1 - most / n_targets) and out["lo"] <= 0.99
    rate, kinds = binary_coverage(f, y, uniform_sample(n_targets, n))
    assert rate >= 0.90 and kinds["bound"] == pytest.approx(0.6, abs=0.06), (rate, kinds)
    # a larger sample, with the normal interval when the failure is seen
    rate, kinds = binary_coverage(f, y, uniform_sample(n_targets, 60))
    assert rate >= 0.90 and set(kinds) == {"bound", "normal"}, (rate, kinds)


def by_size(size: np.ndarray, n: int):
    return lambda rng: by_size_sample(size.copy(), n, rng)


_SIZE = 0.2 + np.random.default_rng(2).random(400)
RARE = {
    "uniform, 3 of 100 fail, 40 labels": (failures(100, range(3)), uniform_sample(100, 40)),
    "uniform, 10 of 100 fail, 40 labels": (failures(100, range(10)), uniform_sample(100, 40)),
    "uniform, 10 of 100 fail, 60 labels": (failures(100, range(10)), uniform_sample(100, 60)),
    "uniform, 10 of 1000 fail, 100 labels": (failures(1000, range(10)), uniform_sample(1000, 100)),
    "strata 10 of 100 and 50 of 100, failures in the first": (failures(200, range(4)), two_strata(200, 100, 10, 50)),
    "strata 10 of 100 and 50 of 100, failures in the second": (failures(200, range(100, 104)),
                                                               two_strata(200, 100, 10, 50)),
    "strata 10 of 100 and 50 of 100, failures in both": (failures(200, [0, 1, 2, 3, 100, 101, 102, 103]),
                                                         two_strata(200, 100, 10, 50)),
    "strata 10 of 100 and 20 of 100, failures in both": (failures(200, [0, 1, 2, 3, 100, 101, 102, 103]),
                                                         two_strata(200, 100, 10, 20)),
    "strata 20 of 20 and 30 of 100, failures in both": (failures(120, [0, 1, 2, 3, 4, 20, 21, 22]),
                                                        two_strata(120, 20, 20, 30)),
    "by size, 80 labels, 8 fail at the lowest chance": (failures(400, np.argsort(_SIZE)[:8]), by_size(_SIZE, 80)),
    "by size, 40 labels, 8 fail at the lowest chance": (failures(400, np.argsort(_SIZE)[:8]), by_size(_SIZE, 40)),
    "by size, 80 labels, 8 fail anywhere": (failures(400, np.arange(8) * 50), by_size(_SIZE, 80)),
}


@pytest.mark.parametrize("case", list(RARE))
def test_rare_binary_failures_are_covered(case):
    """Rare failures on a binary criterion, a judge that predicts a pass everywhere: the 90% interval
    holds the true mean in at least 85% of samples under uniform sampling, strata with their own rates
    and unequal probabilities. With failures this rare it mostly holds it more often than 90%: a wide
    interval is the only honest one when the sample saw nothing. Measured over 500 samples each, in the
    order above: 0.934, 0.862, 0.934, 1.000, 0.998, 1.000, 0.998, 0.990, 1.000, 0.996, 1.000, 0.854."""
    y, sampler = RARE[case]
    rate, _ = binary_coverage(np.ones(len(y)), y, sampler)
    assert rate >= 0.85, rate


def test_a_stratum_that_saw_nothing_widens_the_interval():
    """Failures in both strata, but the stratum sampled at 1 in 10 shows none: its unseen targets are
    allowed for, though the other stratum gives the sample a spread."""
    n_targets = 200
    y = failures(n_targets, [0, 1, 2, 3, 100, 101, 102, 103])
    picked = np.concatenate([np.arange(10, 20), np.arange(103, 153)])     # none of the first four, one of the rest
    pi = np.concatenate([np.full(10, 0.1), np.full(50, 0.5)])
    out = DC.interval(np.ones(n_targets), y[picked], np.ones(60), pi, np.random.default_rng(0))
    # independent: failures among the 100 targets of the first stratum that 10 labels miss with a
    # chance above 5%
    most = max(k for k in range(90) if hypergeom(100, k, 10).pmf(0) > 0.05)
    assert out["interval"] == "normal" and out["unseen"] >= most
    assert out["lo"] <= y.mean() <= out["hi"] and out["lo"] <= out["mean"] - most / n_targets
    # the same labels without that stratum's allowance would miss the truth
    assert out["mean"] - 1.645 * out["sd"] > y.mean()


def test_groups_without_spread():
    y = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.3, 0.3])
    f = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.1, 0.2])
    pi = np.array([0.2, 0.2, 0.5, 0.5, 0.5, 0.3, 0.7])
    # 0.2: equal scores; 0.5: a spread; 0.3 and 0.7 are held by one target each and form one group
    assert [at.tolist() for at in DC.flat_groups(y, f, pi)] == [[5, 6], [0, 1]]
    assert [DC.unseen(p) for p in (0.4, 0.1, 0.95, 1.0)] == [5, 28, 0, 0]
    assert 0.6 ** 5 > 0.05 > 0.6 ** 6 and 0.9 ** 28 > 0.05 > 0.9 ** 29


# ---------------------------------------------------------------------------------------------------
# The check over a fit file, score rows and labels plans

def score(rid: str, run: str, judge: str, value, component: str = "depth", **more) -> dict:
    return C.validate_score({"schema": C.SCHEMA_SCORES, "id": rid, "run": run, "component": component,
                             "judge": judge, "call": f"{rid}-call", "value": value, "rubric": "example@1",
                             "observed_at": STAMP, **more})


def assignment(aid: str, run: str, pi: float, components=("depth",), kind: str = "active", **more) -> dict:
    return {"id": aid, "ask": "score", "target": {"run": run}, "components": list(components), "kind": kind,
            "inclusion_probability": pi, "minutes": 3, **more}


def plan(rows, mode: str = "active") -> dict:
    return C.validate_labels({"schema": C.SCHEMA_LABELS, "mode": mode, "reviewer": "reviewer", "assignments": rows})


def build_study(folder, *, n_inputs: int = 150, n_labels: int = 60, shift: float = 0.0, seed: int = 0,
                shifted_arm: str = "strong-all"):
    """A fit of known parameters, judge readings on every run, and human readings on a sample drawn with
    unequal inclusion probabilities per arm. `shift` moves the human readings of one arm away from the model."""
    runs = testing.example_runs(n_inputs=n_inputs)
    built = testing.build_fit(folder / "fit.json", runs=runs, n_draws=100, posterior_sd=0.005, seed=seed)
    rng = np.random.default_rng(seed + 100)
    q = built.truth["q_run"][:, DEPTH]
    bias = built.truth["bias"][built.fit.judge("strong"), DEPTH]
    rows, picked_rows, n = [], [], 0
    judge_reading = np.clip(q + bias + 0.05 * rng.standard_normal(len(q)), 0, 1)
    for r, run in enumerate(runs):
        rows.append(score(f"j{r:04d}", run["run"], "strong", round(float(10 * judge_reading[r]), 3)))
    for arm in built.fit.arms:
        of_arm = np.array([r for r, run in enumerate(runs) if run["arm"] == arm])
        picked, pi = by_size_sample(0.3 + np.abs(judge_reading[of_arm] - 0.6), n_labels, rng)
        for k in picked:
            r = int(of_arm[k])
            human = np.clip(q[r] + (shift if arm == shifted_arm else 0.0) + 0.05 * rng.standard_normal(), 0, 1)
            n += 1
            picked_rows.append(assignment(f"a{n:04d}", runs[r]["run"], float(pi[k])))
            rows.append(score(f"h{n:04d}", runs[r]["run"], "reviewer", round(float(10 * human), 3),
                              assignment=f"a{n:04d}"))
    return built, rows, plan(picked_rows)


@pytest.fixture(scope="module")
def agreeing(tmp_path_factory):
    return build_study(tmp_path_factory.mktemp("agreeing"))


def test_check_agrees_when_the_labels_follow_the_model(agreeing):
    built, rows, labels = agreeing
    out = DC.check(built.fit, rows, [labels], plan_names=["labels.json"])
    assert list(out) == ["method", "level", "bootstrap_below", "plans", "cells", "flagged", "not_used", "notes"]
    assert (out["level"], out["bootstrap_below"], out["plans"], out["flagged"]) == (0.90, 50, ["labels.json"], 0)
    assert out["method"].startswith("prediction-powered") and out["not_used"] == []
    assert [(cell["arm"], cell["dimension"]) for cell in out["cells"]] == [
        (arm, "final/depth") for arm in ("baseline", "strong-draft", "strong-all")]
    q = built.truth["q_run"][:, DEPTH]
    for a, cell in enumerate(out["cells"]):
        assert list(cell) == ["arm", "dimension", "step", "component", "n_targets", "n_labels", "n_predicted",
                              "reviewers", "design", "model", "gap", "joint_sd", "flagged", "own", "status"]
        assert (cell["n_targets"], cell["n_labels"], cell["n_predicted"]) == (150, 60, 150)
        assert cell["reviewers"] == ["reviewer"] and cell["status"] == "ok" and cell["flagged"] is False
        truth = float(np.clip(q[built.fit.run_arm == a], 0, 1).mean())
        assert cell["design"]["interval"] == "normal"
        assert cell["design"]["mean"] == pytest.approx(truth, abs=3 * cell["design"]["sd"] + 0.005)
        assert cell["model"]["mean"] == pytest.approx(truth, abs=0.01)
        assert cell["gap"] == pytest.approx(cell["design"]["mean"] - cell["model"]["mean"])
        # the joint sd reads the sample without the finite population correction: larger than with it
        assert np.hypot(cell["design"]["sd"], cell["model"]["sd"]) < cell["joint_sd"] < 2 * cell["design"]["sd"]
        assert abs(cell["gap"]) <= 1.645 * cell["joint_sd"]
        assert cell["own"] == {"design": pytest.approx(10 * cell["design"]["mean"]),
                               "model": pytest.approx(10 * cell["model"]["mean"])}
    assert out["notes"] == ["scores are on 0 to 1: the fitting scale for linear, fraction and levels, the share "
                            "passed for binary and items"]
    assert DC.check(built.fit, rows, [labels], plan_names=["labels.json"]) == out


def test_check_flags_the_arm_where_the_labels_disagree(tmp_path):
    built, rows, labels = build_study(tmp_path, shift=-0.15)
    out = DC.check(built.fit, rows, [labels])
    assert [cell["flagged"] for cell in out["cells"]] == [False, False, True] and out["flagged"] == 1
    cell = out["cells"][2]
    assert cell["gap"] == pytest.approx(-0.15, abs=0.03) and abs(cell["gap"]) > 1.645 * cell["joint_sd"]
    assert out["plans"] == ["plan 1"]


def test_flag_rate_stays_near_its_level_when_the_labels_follow_the_model(tmp_path):
    """Over 40 studies of 3 arms where the human readings follow the model, about 1 cell in 10 is flagged
    by chance. Measured: 14 of 120."""
    flagged = z = 0
    ratios = []
    for seed in range(40):
        built, rows, labels = build_study(tmp_path / str(seed), seed=seed)
        cells = DC.check(built.fit, rows, [labels])["cells"]
        flagged += sum(cell["flagged"] for cell in cells)
        ratios += [cell["gap"] / cell["joint_sd"] for cell in cells]
    assert 6 <= flagged <= 20, flagged
    assert np.std(ratios) == pytest.approx(1.0, abs=0.15) and abs(np.mean(ratios)) < 0.3


def test_small_samples_use_the_bootstrap_and_say_so(tmp_path):
    built, rows, labels = build_study(tmp_path, n_inputs=60, n_labels=20, seed=2)
    out = DC.check(built.fit, rows, [labels], seed=5)
    assert {cell["design"]["interval"] for cell in out["cells"]} == {"bootstrap"} and out["flagged"] == 0
    assert out["notes"][-1] == "below 50 labelled targets the interval is a bootstrap percentile interval"
    assert DC.check(built.fit, rows, [labels], seed=5) == out
    assert DC.check(built.fit, rows, [labels], seed=6)["cells"][0]["design"]["lo"] != out["cells"][0]["design"]["lo"]


def test_readings_outside_the_plans_are_listed_and_not_used(agreeing):
    built, rows, labels = agreeing
    run = built.fit.index["runs"][0]
    taken = {row["target"]["run"] for row in labels["assignments"]}
    free = next(name for name in built.fit.index["runs"] if name not in taken)
    first = next(row for row in rows if row["judge"] == "reviewer")
    extra = [
        score("x1", free, "reviewer", 2.0),                                        # names no assignment
        score("x2", free, "reviewer", 2.0, assignment="nothing"),                  # not in the plans
        score("x3", free, "reviewer", 2.0, assignment=first["assignment"]),        # another run than assigned
        score("x4", "no-such-run", "reviewer", 2.0),
        score("x5", run, "stranger", 2.0),
        score("x6", first["run"], "reviewer", 0.0, replaces=first["id"], assignment=first["assignment"]),
        score("x7", free, "reviewer", "ok", component="clarity", assignment=first["assignment"]),
    ]
    out = DC.check(built.fit, rows + extra, [labels])
    reasons = {row["id"]: row["reason"] for row in out["not_used"]}
    assert reasons == {
        first["id"]: "replaced",
        "x1": "a person's reading that names no assignment",
        "x2": 'its assignment "nothing" is not a score assignment of the plans given',
        "x3": f'it does not match its assignment "{first["assignment"]}"',
        "x4": "its run is not in the fit",
        "x5": "its judge is not in the fit",
        "x7": f'it does not match its assignment "{first["assignment"]}"',
    }
    # the correction is read in place of the reading it replaces
    base = DC.check(built.fit, rows, [labels])
    arm = next(a for a, cell in enumerate(base["cells"]) if first["run"].startswith(cell["arm"] + "-i"))
    assert out["cells"][arm]["design"]["mean"] < base["cells"][arm]["design"]["mean"]
    assert [cell["n_labels"] for cell in out["cells"]] == [60, 60, 60]


def test_plans_pairs_and_thin_cells(agreeing):
    built, rows, labels = agreeing
    runs = built.fit.index["runs"]
    half = len(labels["assignments"]) // 2
    first, second = plan(labels["assignments"][:half]), plan(labels["assignments"][half:], mode="pilot")
    out = DC.check(built.fit, rows, [first, second])
    assert [cell["n_labels"] for cell in out["cells"]] == [60, 60, 60] and out["plans"] == ["plan 1", "plan 2"]
    assert out["notes"][1].startswith("several plans: a target's inclusion probability is taken from")
    with pytest.raises(C.ContractError) as caught:
        DC.check(built.fit, rows, [labels, first], plan_names=["a.json", "b.json"])
    assert caught.value.code == "duplicate" and str(caught.value).startswith('b.json: assignments[0].id: "a0001" is')
    # a pair assignment is counted and left out; one labelled target gives a cell without an interval
    pair = {"id": "p1", "ask": "pair", "a": {"run": runs[0]}, "b": {"run": runs[1]}, "shown_first": "a",
            "components": ["overall"], "kind": "active", "inclusion_probability": 0.5, "minutes": 2}
    one = plan([assignment("c1", runs[0], 0.5, components=("clarity", "numbers_right", "coverage")), pair])
    extra = [score("c1", runs[0], "reviewer", "good", component="clarity", assignment="c1"),
             score("c2", runs[0], "reviewer", 1, component="numbers_right", assignment="c1"),
             score("c3", runs[0], "reviewer", 3, component="coverage", of=4, assignment="c1")]
    out = DC.check(built.fit, extra, [one])
    assert out["notes"][-1] == "1 pair assignments are not checked: the design-based check reads scores"
    assert [(cell["dimension"], cell["n_labels"], cell["n_predicted"], cell["design"], cell["flagged"])
            for cell in out["cells"]] == [("final/numbers_right", 1, 0, None, False),
                                          ("final/clarity", 1, 0, None, False),
                                          ("final/coverage", 1, 0, None, False)]
    assert {cell["status"] for cell in out["cells"]} == {"one labelled target: no interval, not checked"}


def test_every_scale_is_checked_without_judge_readings(tmp_path):
    """A census of human readings on every scale, with no judge reading: the estimate is the plain mean,
    and the model side agrees with readings simulated from the fit's own parameters."""
    runs = testing.example_runs(n_inputs=200)
    built = testing.build_fit(tmp_path / "fit.json", runs=runs, n_draws=50, posterior_sd=0.002)
    truth, fit = built.truth, built.fit
    rng = np.random.default_rng(8)
    j = fit.judge("reviewer")
    rows, picks = [], []
    for r, run in enumerate(runs):
        noise = np.hypot(truth["tau"][j], truth["sigma"][j])
        eta = truth["q_run"][r] + truth["bias"][j] + noise * rng.standard_normal(5)
        aid = f"a{r:04d}"
        picks.append(assignment(aid, run["run"], 1.0, components=("numbers_right", "clarity", "depth", "coverage"),
                                kind="uniform"))
        level = ["poor", "ok", "good"][int((eta[1] >= np.array([0.25, 0.75])).sum())]
        passed = int((rng.random(8) < ndtr(eta[3])).sum())
        rows += [score(f"n{r}", run["run"], "reviewer", int(eta[0] >= 0), component="numbers_right", assignment=aid),
                 score(f"c{r}", run["run"], "reviewer", level, component="clarity", assignment=aid),
                 score(f"d{r}", run["run"], "reviewer", round(float(10 * np.clip(eta[2], 0, 1)), 3), assignment=aid),
                 score(f"k{r}", run["run"], "reviewer", passed, component="coverage", of=8, assignment=aid)]
    out = DC.check(fit, rows, [plan(picks, mode="pilot")])
    assert len(out["cells"]) == 12 and {cell["n_predicted"] for cell in out["cells"]} == {0}
    for cell in out["cells"]:
        assert cell["n_labels"] == cell["n_targets"] == 200 and cell["design"]["sd"] == 0.0
        # a census has no sampling error: the gap is the simulation noise of 200 readings
        assert cell["gap"] == pytest.approx(0.0, abs=0.06), (cell["arm"], cell["dimension"], cell["gap"])
    assert sum(abs(cell["gap"]) < 0.03 for cell in out["cells"]) >= 9
    # item lines are read as one count per call
    lines = []
    for r in range(3):
        lines += [score(f"i{r}-{k}", runs[r * 3]["run"], "reviewer", int(k < 3), component="coverage", item=f"p{k}",
                        assignment=f"a{r * 3:04d}", call=f"call-{r}") for k in range(4)]
    out = DC.check(fit, lines, [plan(picks, mode="pilot")])
    assert out["cells"][0]["dimension"] == "final/coverage" and out["cells"][0]["n_labels"] == 3
    assert out["cells"][0]["design"]["mean"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------------------------------
# The reviewer's judge-by-input noise

BINARY = 0         # final/numbers_right in the example rubric


def by_input_study(folder, state: str = "identified", n_inputs: int = 400, seed: int = 0):
    """Binary quality 0.5 on every run, a reviewer without bias or judge-by-output noise, call noise 1
    and judge-by-input sd 2. Every run is read once by the reviewer."""
    runs = testing.example_runs(n_inputs=n_inputs)
    judges = testing.example_judges()
    j = [row["id"] for row in judges["judges"]].index("reviewer")
    base = testing.default_truth(F.build_index(testing.example_rubric(), judges, runs),
                                 C.dimensions(testing.example_rubric()), judges)
    tau, sigma, by_input = base["tau"].copy(), base["sigma"].copy(), base["sd_judge_input"].copy()
    tau[j, BINARY], sigma[j, BINARY], by_input[j, BINARY] = 0.0, 1.0, 2.0
    q = np.full((len(runs), 5), 0.5)
    built = testing.build_fit(folder / "fit.json", runs=runs, n_draws=20, posterior_sd=0.0, seed=seed,
                              truth={"q_run": q, "tau": tau, "sigma": sigma, "sd_judge_input": by_input},
                              terms=[F.term("judge_input", state, judge="reviewer", count=n_inputs)])
    rng = np.random.default_rng(seed + 50)
    effect = 2.0 * rng.standard_normal(n_inputs)            # one effect per input, shared by its runs
    rows, picks = [], []
    for r, run in enumerate(runs):
        reading = 0.5 + effect[built.fit.run_input[r]] + rng.standard_normal() >= 0
        picks.append(assignment(f"a{r:04d}", run["run"], 1.0, components=("numbers_right", "depth"), kind="uniform"))
        rows += [score(f"n{r}", run["run"], "reviewer", int(reading), component="numbers_right",
                       assignment=f"a{r:04d}"),
                 score(f"d{r}", run["run"], "reviewer", 5.0 + (r % 2), assignment=f"a{r:04d}")]
    return built, rows, plan(picks, mode="pilot")


def test_expected_human_reading_includes_judge_by_input_noise(tmp_path):
    """Quality 0.5, call noise 1, judge-by-input sd 2: a pass has chance `Phi(0.5 / sqrt(1 + 4))`, which
    is 0.588468, not the 0.691462 that leaves the judge-by-input term out."""
    rng = np.random.default_rng(9)
    simulated = (0.5 + 2.0 * rng.standard_normal(2_000_000) + rng.standard_normal(2_000_000) >= 0).mean()
    assert simulated == pytest.approx(0.588468, abs=0.002)
    cells = []
    for seed in range(6):
        built, rows, labels = by_input_study(tmp_path / str(seed), seed=seed)
        cells += [c for c in DC.check(built.fit, rows, [labels])["cells"] if c["dimension"] == "final/numbers_right"]
    assert len(cells) == 18
    for cell in cells:
        assert cell["model"]["mean"] == pytest.approx(0.588468, abs=1e-6)
        assert cell["design"]["interval"] == "census" and cell["status"] == "ok"
    # the human readings were drawn with the judge-by-input effect: they agree with the model, and
    # would be flagged nearly everywhere against the figure that leaves the term out
    assert np.mean([cell["design"]["mean"] for cell in cells]) == pytest.approx(0.588468, abs=0.03)
    assert sum(cell["flagged"] for cell in cells) <= 5
    assert sum(abs(cell["design"]["mean"] - 0.691462) > 1.645 * cell["joint_sd"] for cell in cells) >= 15


def test_a_noise_term_the_design_does_not_measure_is_not_checked(tmp_path):
    built, rows, labels = by_input_study(tmp_path, state="prior_only", n_inputs=30)
    cells = {(c["arm"], c["dimension"]): c for c in DC.check(built.fit, rows, [labels])["cells"]}
    binary, linear = cells[("baseline", "final/numbers_right")], cells[("baseline", "final/depth")]
    assert binary["status"] == ("not checked: the expected reading on this scale depends on a noise term the "
                                "design does not measure (judge_input of reviewer)")
    assert binary["model"] is None and binary["gap"] is None and binary["flagged"] is False
    assert binary["design"]["interval"] == "census" and 0.0 <= binary["design"]["mean"] <= 1.0
    # the expected reading on a linear scale is `q + bias` whatever the noise: still checked
    assert linear["status"] == "ok" and linear["model"]["mean"] == pytest.approx(0.5, abs=1e-9)
    assert linear["design"]["mean"] == pytest.approx(0.55)
    assert DC.term_state(built.fit, "judge_input", "reviewer", "final/depth") == "prior_only"
    assert DC.term_state(built.fit, "tau", "reviewer", "final/depth") is None


def test_readings_that_share_an_input_are_not_counted_as_independent(tmp_path):
    """A reviewer reads 4 runs of every input and carries a judge-by-input effect of sd 0.1 on a linear
    score; a judge reads every run. When the readings follow the model, about 1 cell in 10 is flagged
    (9 of 90 expected); counting every reading as independent would flag far more."""
    flagged, ratios, plain = 0, [], []
    for seed in range(30):
        runs = testing.example_runs(n_inputs=40, runs_per_input=4)
        judges = testing.example_judges()
        j = [row["id"] for row in judges["judges"]].index("reviewer")
        base = testing.default_truth(F.build_index(testing.example_rubric(), judges, runs),
                                     C.dimensions(testing.example_rubric()), judges)
        by_input = base["sd_judge_input"].copy()
        by_input[j, DEPTH] = 0.1
        built = testing.build_fit(tmp_path / str(seed) / "fit.json", runs=runs, n_draws=40, posterior_sd=0.002,
                                  seed=seed, truth={"sd_judge_input": by_input},
                                  terms=[F.term("judge_input", "identified", judge="reviewer", count=40)])
        rng = np.random.default_rng(seed + 500)
        q = built.truth["q_run"][:, DEPTH]
        bias = built.truth["bias"][built.fit.judge("strong"), DEPTH]
        effect = 0.1 * rng.standard_normal(40)
        rows, picks, residual = [], [], []
        for r, run in enumerate(runs):
            judge = np.clip(q[r] + bias + 0.05 * rng.standard_normal(), 0, 1)
            human = np.clip(q[r] + effect[built.fit.run_input[r]] + 0.05 * rng.standard_normal(), 0, 1)
            residual.append(human - judge)
            picks.append(assignment(f"a{r:04d}", run["run"], 1.0, kind="uniform"))
            rows += [score(f"j{r}", run["run"], "strong", round(float(10 * judge), 3)),
                     score(f"h{r}", run["run"], "reviewer", round(float(10 * human), 3), assignment=f"a{r:04d}")]
        for cell in DC.check(built.fit, rows, [plan(picks, mode="pilot")])["cells"]:
            flagged += cell["flagged"]
            ratios.append(cell["gap"] / cell["joint_sd"])
            # what the ratio would be with every reading counted as independent
            mine = np.array(residual)[[r for r, run in enumerate(runs) if run["arm"] == cell["arm"]]]
            plain.append(cell["gap"] / (mine.std(ddof=1) / np.sqrt(len(mine))))
    assert 3 <= flagged <= 16, flagged
    assert np.std(ratios) == pytest.approx(1.0, abs=0.2), np.std(ratios)
    assert np.mean(np.abs(plain) > 1.645) > 0.25 and np.std(plain) > 1.4


# ---------------------------------------------------------------------------------------------------
# Through `eval value --check design`

def run_json(capsys, *argv):
    code = cli.main(["eval", "value", *argv, "--json"])
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if captured.out else None), captured.err


def test_value_command_runs_the_check(tmp_path, capsys):
    built, rows, labels = build_study(tmp_path, shift=-0.15)
    C.write_lines(tmp_path / C.SCORES_FILE, C.SCHEMA_SCORES, rows)
    plan_path = C.write_document(tmp_path / "labels.json", labels)
    value = str(C.write_document(tmp_path / "value.json", testing.example_value()))
    code, out, err = run_json(capsys, str(built.path), "--value", value, "--check", "design", "--labels",
                              str(plan_path), "--seed", "3")
    # the fit file of this test records no input files, so the scores file is new to it
    assert code == 0 and out["warnings"] == ["scores: scores.jsonl was not read by the earlier stage"]
    assert err == "warning: scores: scores.jsonl was not read by the earlier stage\n"
    assert out["design_check"] == json.loads(json.dumps(DC.check(built.fit, rows, [labels], seed=3,
                                                                 plan_names=["labels.json"])))
    assert out["design_check"]["flagged"] == 1 and out["options"]["check"] == "design"
    assert sorted(out["inputs"]) == ["draws", "fit", "labels-1", "scores", "value"]
    assert list(out).index("design_check") == list(out).index("comparisons") + 1
    assert cli.main(["eval", "value", str(built.path), "--value", value, "--check", "design", "--labels",
                     str(plan_path), "--seed", "3"]) == 0
    text = capsys.readouterr().out
    assert "design-based check (90% intervals, scores on 0 to 1): 1 of 3 cells flagged" in text
    assert "FLAGGED" in text and text.count("FLAGGED") == 1
    # `designcheck.run` is `eval value` with the check on
    args = argparse.Namespace(fit=str(built.path), value=value, labels=[str(plan_path)], seed=3, json=True,
                              check=None, reference=None, study=None, strict=False, allow_unconverged=False)
    assert DC.run(args) == 0 and args.check == "design"
    assert json.loads(capsys.readouterr().out)["design_check"] == out["design_check"]
    # without scores.jsonl there is nothing to check
    other = tmp_path / "other"
    other.mkdir()
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--check", "design", "--labels",
                            str(plan_path), "--study", str(other))
    assert code == 1 and out["error"].startswith("scores.jsonl: no such file")
    code, out, _ = run_json(capsys, str(built.path), "--value", value, "--check", "design", "--labels",
                            str(tmp_path / "none.json"))
    assert code == 1 and out["error"].startswith("none.json: no such file")
