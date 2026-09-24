"""Tests for loopmath.scoring: the transfer-test metric library (SPEC section 7)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from loopmath.scoring import (
    acceptance_calibration,
    baseline_nearest_neighbor,
    baseline_pooled_mean,
    decision_regret,
    interval_coverage,
    kendall_tau,
    mlpd,
    summarize,
    within_factor,
)

FORBIDDEN = [
    "knowledge gradient",
    "posterior",
    "prior",
    "experimental design",
    "value of information",
    "bandit",
    "arms",
]


def _assert_clean_text(text: str) -> None:
    assert "—" not in text, f"em-dash found in: {text!r}"
    lowered = text.lower()
    for word in FORBIDDEN:
        assert word not in lowered, f"forbidden word {word!r} found in: {text!r}"


# --------------------------------------------------------------------------
# 1. interval_coverage
# --------------------------------------------------------------------------


def test_interval_coverage_exact_fraction():
    y = np.array([0.0] * 8 + [5.0, -5.0])
    lo = np.array([-1.0] * 10)
    hi = np.array([1.0] * 10)
    result = interval_coverage(y, lo, hi)
    assert result["n"] == 10
    assert result["coverage"] == pytest.approx(0.8)
    assert result["n_inside"] == 8
    assert result["mean_width"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# 2. mlpd
# --------------------------------------------------------------------------


def test_mlpd_higher_for_draws_near_truth():
    rng = np.random.default_rng(20260901)
    y = np.array([0.0])
    draws_close = rng.normal(loc=0.0, scale=0.1, size=(1000, 1))
    draws_far = rng.normal(loc=10.0, scale=0.1, size=(1000, 1))

    close = mlpd(y, draws_close)
    far = mlpd(y, draws_far)

    assert close["n"] == 1
    assert far["n"] == 1
    assert close["mlpd"] > far["mlpd"]
    assert far["n_floored"] == 1  # truth is nowhere near the far draws
    assert close["n_floored"] == 0


def test_mlpd_documents_floor_and_handles_zero_spread_draws():
    y = np.array([5.0])
    draws = np.full((50, 1), 5.0)  # zero spread, exact match to truth
    result = mlpd(y, draws)
    assert result["n"] == 1
    assert math.isfinite(result["mlpd"])


# --------------------------------------------------------------------------
# 3. baseline_pooled_mean
# --------------------------------------------------------------------------


def test_baseline_pooled_mean_reproduces_train_stats():
    train_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    test_y = np.array([1.0, 2.0, 3.0])
    result = baseline_pooled_mean(train_y, test_y)
    assert result["mean"] == pytest.approx(np.mean(train_y))
    assert result["sd"] == pytest.approx(np.std(train_y, ddof=1))
    assert result["n"] == 3
    assert math.isfinite(result["mlpd"])
    assert math.isfinite(result["rmse"])


# --------------------------------------------------------------------------
# 4. baseline_nearest_neighbor
# --------------------------------------------------------------------------


def test_baseline_nearest_neighbor_exact_match_beats_partial_and_counts_fallback():
    train_df = pd.DataFrame(
        {
            "a": [1, 1],
            "b": [1, 2],
            "y": [10.0, 100.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "a": [1, 999],
            "b": [1, 999],
            "y": [10.0, 10.0],
        }
    )
    result = baseline_nearest_neighbor(train_df, test_df, y_col="y", feature_cols=["a", "b"])
    assert result["n"] == 2
    assert result["n_fallback"] == 1
    # Row 0 gets an exact match (both features), so its prediction should be
    # exactly the matching training row's y (10.0), not a blend with the
    # partial match (100.0). Row 1 falls back to the pooled mean (55.0).
    # rmse over both rows should reflect: (10-10)^2 and (10-55)^2.
    expected_rmse = math.sqrt(((10.0 - 10.0) ** 2 + (10.0 - 55.0) ** 2) / 2)
    assert result["rmse"] == pytest.approx(expected_rmse)
    assert result["mean_matched"] == pytest.approx(1.0)  # (2 + 0) / 2


def test_baseline_nearest_neighbor_log_space_is_scale_invariant_raw_space_is_not():
    """Regression for the swarm-b-sol review's issue 1: an earlier version of
    loopmath.fit fed this baseline raw usd while the model's own MLPD and B1'
    were both computed on log10(usd), making the printed MLPD difference
    dimensionally meaningless (a log density is not invariant under a change
    of units).

    A Normal log-density is exactly invariant under an equal additive shift
    applied to every evaluated point and every training value it is compared
    to -- and log10(k * x) = log10(x) + log10(k) is exactly that: the same
    additive shift for every value. So computing this baseline on
    log10(usd) must be unaffected by multiplying every underlying dollar
    figure by a constant k. Computing it directly on raw usd is NOT
    invariant to that same multiplication, because the density's own scale
    (sd) does not stay fixed -- which is the concrete way "wrong units"
    shows up as a wrong number, not just an abstract labeling complaint.
    """
    rng = np.random.default_rng(3)
    train_usd = rng.uniform(1.0, 50.0, size=30)
    test_usd = rng.uniform(1.0, 50.0, size=10)
    train_df = pd.DataFrame({"task": ["t1"] * 30, "effort": ["low"] * 30, "usd": train_usd})
    test_df = pd.DataFrame({"task": ["t1"] * 10, "effort": ["low"] * 10, "usd": test_usd})

    def log_space_mlpd(k: float) -> float:
        tr = train_df.assign(log10_usd=np.log10(train_df["usd"] * k))
        te = test_df.assign(log10_usd=np.log10(test_df["usd"] * k))
        return baseline_nearest_neighbor(
            tr, te, y_col="log10_usd", feature_cols=["task", "effort"]
        )["mlpd"]

    def raw_space_mlpd(k: float) -> float:
        tr = train_df.assign(usd_scaled=train_df["usd"] * k)
        te = test_df.assign(usd_scaled=test_df["usd"] * k)
        return baseline_nearest_neighbor(
            tr, te, y_col="usd_scaled", feature_cols=["task", "effort"]
        )["mlpd"]

    k = 1000.0
    assert log_space_mlpd(1.0) == pytest.approx(log_space_mlpd(k), abs=1e-9)
    # This is exactly the old, broken call shape (raw dollars): it is NOT
    # scale invariant, which is why it was incomparable to a log10(usd)
    # model MLPD in the first place.
    assert raw_space_mlpd(1.0) != pytest.approx(raw_space_mlpd(k), abs=1e-6)


# --------------------------------------------------------------------------
# 5. kendall_tau
# --------------------------------------------------------------------------


def test_kendall_tau_perfect_and_reversed_order():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_same = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred_reversed = np.array([50.0, 40.0, 30.0, 20.0, 10.0])

    same = kendall_tau(y_true, y_pred_same)
    reversed_ = kendall_tau(y_true, y_pred_reversed)

    assert same["tau"] == pytest.approx(1.0)
    assert reversed_["tau"] == pytest.approx(-1.0)
    assert same["n_concordant_pairs"] == 10
    assert reversed_["n_concordant_pairs"] == 0


# --------------------------------------------------------------------------
# 6. within_factor
# --------------------------------------------------------------------------


def test_within_factor_log10_boundary():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([math.log10(1.9), math.log10(2.1)])
    result = within_factor(y_true, y_pred, factor=2.0, log10_space=True)
    assert result["n"] == 2
    assert result["within"] == 1
    assert result["share"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# 7. decision_regret
# --------------------------------------------------------------------------


def test_decision_regret_hand_computed():
    test_df = pd.DataFrame(
        {
            "task": ["A", "A", "B", "B"],
            "cost": [5.0, 10.0, 5.0, 10.0],
        }
    )
    # Group A: estimate picks the true cheapest (row 0, pred=5 < pred=10).
    # Group B: estimate picks the true second-cheapest (pred says row 1,
    # cost 10, is cheapest).
    y_pred = np.array([5.0, 10.0, 10.0, 5.0])

    result = decision_regret(test_df, y_pred, group_col="task", cost_col="cost")

    assert result["n_groups"] == 2
    assert result["n"] == 4
    assert result["n_optimal"] == 1
    assert result["share_optimal"] == pytest.approx(0.5)
    assert result["mean_regret"] == pytest.approx((0.0 + 5.0) / 2)
    assert result["median_regret"] == pytest.approx(2.5)
    assert result["mean_regret_ratio"] == pytest.approx((1.0 + 2.0) / 2)

    by_group = {g["group"]: g for g in result["per_group"]}
    assert by_group["A"]["regret"] == pytest.approx(0.0)
    assert by_group["A"]["chosen_cost"] == pytest.approx(5.0)
    assert by_group["B"]["regret"] == pytest.approx(5.0)
    assert by_group["B"]["chosen_cost"] == pytest.approx(10.0)
    assert by_group["B"]["best_cost"] == pytest.approx(5.0)
    assert by_group["B"]["regret_ratio"] == pytest.approx(2.0)


def test_decision_regret_config_aggregation_differs_from_row_level():
    """Regression: "decision regret" must pick a workflow configuration
    (model + effort together), not an individual held-out row. A single
    noisy replicate with an anomalously low prediction can make the
    row-level rule choose a configuration whose own mean predicted cost is
    actually the worse one; aggregating replicates by configuration before
    choosing (mean over replicates on both sides) fixes that. This fixture
    is built so the two rules give different answers, so it would fail
    against the old row-level-only behavior.
    """
    test_df = pd.DataFrame({
        "task_group": ["G1", "G1", "G1", "G1"],
        "model": ["m-a", "m-a", "m-b", "m-b"],
        "effort": ["low", "low", "low", "low"],
        "cost": [10.0, 10.0, 6.0, 6.0],
    })
    # config m-a: predictions [2, 18], mean predicted cost 10.
    # config m-b: predictions [5, 7], mean predicted cost 6.
    # One m-a replicate (pred=2) looks cheaper than any single m-b row.
    y_pred = np.array([2.0, 18.0, 5.0, 7.0])

    row_level = decision_regret(test_df, y_pred, group_col="task_group", cost_col="cost")
    config_level = decision_regret(
        test_df, y_pred, group_col="task_group", cost_col="cost",
        config_cols=["model", "effort"],
    )

    # Row level: the single lowest-predicted ROW (pred=2, cost=10, config
    # m-a) is "chosen", compared against the single cheapest realized ROW
    # (cost=6, config m-b) -- a nonzero regret that looks like "the estimate
    # picked wrong", when really one noisy replicate did.
    assert row_level["per_group"][0]["chosen_cost"] == pytest.approx(10.0)
    assert row_level["mean_regret"] == pytest.approx(4.0)

    # Configuration level: m-a's mean predicted cost (10.0) loses to m-b's
    # (6.0), so m-b is chosen -- which is also the true cheapest mean
    # realized cost. Zero regret, and it actually means the right
    # configuration was picked.
    assert config_level["per_group"][0]["chosen_cost"] == pytest.approx(6.0)
    assert config_level["mean_regret"] == pytest.approx(0.0)
    assert config_level["n_optimal"] == 1
    assert config_level["per_group"][0]["n_configs"] == 2


# --------------------------------------------------------------------------
# 8. acceptance_calibration
# --------------------------------------------------------------------------


def test_acceptance_calibration_perfect_vs_overconfident():
    # Perfectly calibrated: 5 bins of 5 points each, observed rate in each
    # bin exactly equals the stated probability.
    p_pred = np.array(
        [0.0] * 5 + [0.2] * 5 + [0.4] * 5 + [0.6] * 5 + [0.8] * 5
    )
    y_true = np.array(
        [0, 0, 0, 0, 0]  # 0/5 = 0.0
        + [1, 0, 0, 0, 0]  # 1/5 = 0.2
        + [1, 1, 0, 0, 0]  # 2/5 = 0.4
        + [1, 1, 1, 0, 0]  # 3/5 = 0.6
        + [1, 1, 1, 1, 0],  # 4/5 = 0.8
        dtype=float,
    )
    calibrated = acceptance_calibration(y_true, p_pred, n_bins=5)
    assert calibrated["n"] == 25
    assert calibrated["ece"] == pytest.approx(0.0, abs=1e-9)

    # Systematically overconfident: stated chance is always 0.9, actual
    # acceptance rate is 0.5. Rows are grouped, not interleaved (accepted
    # rows first, then rejected) precisely because the old row-position
    # binning needed interleaving to hide its tie-boundary bug (flag 10);
    # value-based binning gives the same answer regardless of row order, so
    # a grouped layout is no longer a special case that needs avoiding.
    # Every probability is tied at 0.9, so the value-based bin boundary
    # cannot split them: all 20 points land in a single realised bin.
    p_pred_over = np.full(20, 0.9)
    y_true_over = np.array([1.0] * 10 + [0.0] * 10)
    overconfident = acceptance_calibration(y_true_over, p_pred_over, n_bins=5)
    assert overconfident["ece"] == pytest.approx(0.4, abs=1e-9)
    assert overconfident["ece"] > calibrated["ece"]
    assert len(overconfident["bins"]) == 1
    assert overconfident["bins"][0]["n"] == 20


def test_acceptance_calibration_permutation_invariant():
    # Grouped ties of uneven size, deliberately not a multiple of n_bins, so
    # the naive row-position equal-count split would cut a tied group in two
    # depending on where it landed. Binning by value must give the identical
    # ece, brier, and bin structure no matter how the rows are ordered.
    p_pred = np.array(
        [0.1] * 3 + [0.3] * 4 + [0.5] * 2 + [0.7] * 5 + [0.9] * 3
    )
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=p_pred.size).astype(float)

    baseline = acceptance_calibration(y_true, p_pred, n_bins=5)

    perm = np.random.default_rng(20260901).permutation(p_pred.size)
    shuffled = acceptance_calibration(y_true[perm], p_pred[perm], n_bins=5)

    assert shuffled["ece"] == pytest.approx(baseline["ece"])
    assert shuffled["brier"] == pytest.approx(baseline["brier"])
    assert shuffled["bins"] == baseline["bins"]
    assert baseline["n"] == p_pred.size


# --------------------------------------------------------------------------
# 9. empty input: every function returns its documented keys without raising
# --------------------------------------------------------------------------


def test_empty_input_never_raises():
    empty = np.array([])

    ic = interval_coverage(empty, empty, empty)
    assert set(ic) >= {"n", "coverage", "n_inside", "mean_width", "median_width", "note"}
    assert ic["n"] == 0

    m = mlpd(empty, np.zeros((0, 0)))
    assert set(m) >= {"n", "mlpd", "n_floored", "bandwidth", "note"}
    assert m["n"] == 0

    bpm = baseline_pooled_mean(empty, empty)
    assert set(bpm) >= {"n", "mlpd", "rmse", "mean", "sd", "note"}
    assert bpm["n"] == 0

    empty_df = pd.DataFrame({"a": [], "b": [], "y": []})
    bnn = baseline_nearest_neighbor(empty_df, empty_df, y_col="y", feature_cols=["a", "b"])
    assert set(bnn) >= {"n", "mlpd", "rmse", "n_fallback", "mean_matched", "note"}
    assert bnn["n"] == 0

    kt = kendall_tau(empty, empty)
    assert set(kt) >= {"n", "tau", "p_value", "n_concordant_pairs", "note"}
    assert kt["n"] == 0

    wf = within_factor(empty, empty)
    assert set(wf) >= {"n", "within", "share", "factor", "median_abs_error", "note"}
    assert wf["n"] == 0

    empty_df2 = pd.DataFrame({"task": [], "cost": []})
    dr = decision_regret(empty_df2, empty, group_col="task", cost_col="cost")
    assert set(dr) >= {
        "n_groups",
        "n",
        "mean_regret",
        "median_regret",
        "mean_regret_ratio",
        "n_optimal",
        "share_optimal",
        "per_group",
        "note",
    }
    assert dr["n_groups"] == 0

    ac = acceptance_calibration(empty, empty)
    assert set(ac) >= {"n", "brier", "ece", "bins", "note"}
    assert ac["n"] == 0

    # Every default metric is "expected"; with nothing scored, each one gets
    # an explicit "not available" line rather than vanishing silently
    # (this is flag 2's fix: summarize used to skip unavailable metrics).
    expected_unavailable = [
        "interval coverage: not available (no held-out points were available)",
        "mean log predictive density: not available (no held-out points had usable draws)",
        "pooled-mean baseline: not available (not enough data to compute this baseline)",
        "nearest-neighbor baseline: not available (not enough data to compute this baseline)",
        "rank agreement: not available (fewer than two held-out points)",
        "within-factor share: not available (no held-out points were available)",
        "decision regret: not available (no groups had enough data to compare choices)",
        "acceptance calibration: not available (no usable held-out outcomes were available)",
    ]
    assert summarize({}) == expected_unavailable
    assert (
        summarize(
            {
                "interval_coverage": ic,
                "mlpd": m,
                "baseline_pooled_mean": bpm,
                "baseline_nearest_neighbor": bnn,
                "kendall_tau": kt,
                "within_factor": wf,
                "decision_regret": dr,
                "acceptance_calibration": ac,
            }
        )
        == expected_unavailable
    )
    # An empty `expected` list means nothing is checked, so nothing is
    # printed even though nothing was computed either.
    assert summarize({}, expected=[]) == []


def test_functions_accept_lists_and_series_not_just_arrays():
    y = [0.0, 0.0, 0.0]
    lo = pd.Series([-1.0, -1.0, -1.0])
    hi = [1.0, 1.0, 1.0]
    result = interval_coverage(y, lo, hi)
    assert result["n"] == 3
    assert result["coverage"] == pytest.approx(1.0)


def test_nonfinite_values_are_dropped_and_counted():
    y = np.array([0.0, 0.0, np.nan, np.inf])
    lo = np.array([-1.0, -1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0, 1.0])
    result = interval_coverage(y, lo, hi)
    assert result["n"] == 2
    assert result["n_dropped"] == 2


# --------------------------------------------------------------------------
# 10. no forbidden vocabulary, no em-dash
# --------------------------------------------------------------------------


def test_no_forbidden_vocabulary_or_em_dash():
    rng = np.random.default_rng(1)

    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = y - 0.5
    hi = y + 0.5
    ic = interval_coverage(y, lo, hi)

    draws = rng.normal(loc=y, scale=0.2, size=(200, 4))
    m = mlpd(y, draws)

    bpm = baseline_pooled_mean(np.array([1.0, 2.0, 3.0]), y)

    train_df = pd.DataFrame({"a": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    test_df = pd.DataFrame({"a": [1, 5], "y": [1.0, 4.0]})
    bnn = baseline_nearest_neighbor(train_df, test_df, y_col="y", feature_cols=["a"])

    kt = kendall_tau(y, y[::-1])
    wf = within_factor(y, y + 0.01)

    dr_df = pd.DataFrame({"task": ["A", "A", "B", "B"], "cost": [5.0, 10.0, 5.0, 10.0]})
    dr = decision_regret(dr_df, np.array([5.0, 10.0, 10.0, 5.0]), "task", "cost")

    ac = acceptance_calibration(np.array([1, 0, 1, 0, 1.0]), np.array([0.9, 0.1, 0.9, 0.1, 0.5]), n_bins=2)

    results = {
        "interval_coverage": ic,
        "mlpd": m,
        "baseline_pooled_mean": bpm,
        "baseline_nearest_neighbor": bnn,
        "kendall_tau": kt,
        "within_factor": wf,
        "decision_regret": dr,
        "acceptance_calibration": ac,
    }

    for name, res in results.items():
        _assert_clean_text(res["note"])
        for pg in res.get("per_group", []):
            _assert_clean_text(str(pg.get("group", "")))

    for line in summarize(results):
        _assert_clean_text(line)


# --------------------------------------------------------------------------
# 11. mlpd: shape validation and no-draws exclusion (flags 3, 9)
# --------------------------------------------------------------------------


def test_mlpd_shape_mismatch_reports_reason_without_indexing():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    draws = np.zeros((10, 7))  # 7 columns for 5 held-out points
    result = mlpd(y, draws)
    assert result["n"] == 0
    assert result["n_input"] == 5
    assert math.isnan(result["mlpd"])
    assert result["n_floored"] == 0
    assert result["n_no_draws"] == 0
    assert "reason" in result
    assert "7" in result["reason"] and "5" in result["reason"]


def test_mlpd_excludes_points_with_no_finite_draws():
    y = np.array([0.0, 5.0, 10.0])
    rng = np.random.default_rng(20260901)
    draws = np.empty((100, 3))
    draws[:, 0] = rng.normal(loc=0.0, scale=0.1, size=100)
    draws[:, 1] = np.nan  # this point has no usable draws at all
    draws[:, 2] = rng.normal(loc=10.0, scale=0.1, size=100)

    result = mlpd(y, draws)
    assert result["n_input"] == 3
    assert result["n_no_draws"] == 1
    assert result["n"] == 2  # the no-draws point is excluded, not scored
    # The two scored points have draws right on top of the truth, so
    # neither should have been floored; a pre-fix implementation would have
    # counted the no-draws point as floored instead of excluding it.
    assert result["n_floored"] == 0
    assert math.isfinite(result["mlpd"])


# --------------------------------------------------------------------------
# 12. baselines: training exclusions and floor counting (flags 4, 5)
# --------------------------------------------------------------------------


def test_baseline_pooled_mean_reports_train_exclusions_and_floor():
    train_y = np.array([1.0, 2.0, np.nan, 3.0])
    test_y = np.array([1000.0])  # nowhere near the training data
    result = baseline_pooled_mean(train_y, test_y)
    assert result["n_train_input"] == 4
    assert result["n_train_dropped"] == 1
    assert result["n_floored"] == 1


def test_baseline_nearest_neighbor_reports_train_exclusions_and_floor():
    train_df = pd.DataFrame({"a": [1, 1, 1], "y": [10.0, 12.0, np.nan]})
    test_df = pd.DataFrame({"a": [1], "y": [10000.0]})
    result = baseline_nearest_neighbor(train_df, test_df, y_col="y", feature_cols=["a"])
    assert result["n_train_input"] == 3
    assert result["n_train_dropped"] == 1
    assert result["n_floored"] == 1


# --------------------------------------------------------------------------
# 13. decision_regret: missing group and non-finite exclusions (flag 6)
# --------------------------------------------------------------------------


def test_decision_regret_excludes_missing_group_and_nonfinite_rows():
    test_df = pd.DataFrame(
        {
            "task": ["A", None, "B", "B"],
            "cost": [5.0, 10.0, np.inf, 10.0],
        }
    )
    y_pred = np.array([5.0, 10.0, 10.0, 5.0])
    result = decision_regret(test_df, y_pred, group_col="task", cost_col="cost")

    assert result["n_excluded_rows"] == 2
    assert result["excluded_reason"] == {
        "missing_group": 1,
        "non_finite_cost": 1,
        "non_finite_prediction": 0,
    }
    # n reflects only the rows actually used, not the 4 rows given.
    assert result["n"] == 2
    assert result["n_groups"] == 2


# --------------------------------------------------------------------------
# 14. acceptance_calibration: outcome/probability validation (flag 7)
# --------------------------------------------------------------------------


def test_acceptance_calibration_excludes_invalid_outcomes_and_probabilities():
    y = np.array([0.0, 1.0, 2.0, np.nan, 1.0])
    p = np.array([0.5, 0.3, 0.4, 0.2, 1.5])
    result = acceptance_calibration(y, p, n_bins=2)

    assert result["excluded_reason"] == {
        "outcome_not_binary": 2,
        "probability_out_of_range": 1,
    }
    assert result["n_excluded"] == 3
    assert result["n"] == 2


# --------------------------------------------------------------------------
# 15. summarize: neutral wording only, exclusions surfaced (flags 1, 2)
# --------------------------------------------------------------------------

EVALUATIVE = ["beats", "wins", "better", "worse", "good", "bad", "trails"]


def _assert_no_evaluative_language(lines: list[str]) -> None:
    for line in lines:
        lowered = line.lower()
        for word in EVALUATIVE:
            assert word not in lowered, f"evaluative word {word!r} found in: {line!r}"


def test_summarize_never_uses_evaluative_language():
    rng = np.random.default_rng(2026)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    draws = rng.normal(loc=y, scale=0.2, size=(200, 5))

    results = {
        "interval_coverage": interval_coverage(y, y - 0.5, y + 0.5),
        "mlpd": mlpd(y, draws),
        "baseline_pooled_mean": baseline_pooled_mean(np.array([1.0, 2.0, 3.0]), y),
        "baseline_nearest_neighbor": baseline_nearest_neighbor(
            pd.DataFrame({"a": [1, 2, 3], "y": [1.0, 2.0, 3.0]}),
            pd.DataFrame({"a": [1, 5, 9, 2, 3], "y": y}),
            y_col="y",
            feature_cols=["a"],
        ),
        "kendall_tau": kendall_tau(y, y[::-1]),
        "within_factor": within_factor(y, y + 0.01),
        "decision_regret": decision_regret(
            pd.DataFrame({"task": ["A", "A", "B", "B", "C"], "cost": [5.0, 10.0, 5.0, 10.0, 1.0]}),
            np.array([5.0, 10.0, 10.0, 5.0, 1.0]),
            "task",
            "cost",
        ),
        "acceptance_calibration": acceptance_calibration(
            np.array([1.0, 0.0, 1.0, 0.0, 1.0]), np.array([0.9, 0.1, 0.9, 0.1, 0.5]), n_bins=2
        ),
    }

    lines = summarize(results)
    assert len(lines) > 0
    _assert_no_evaluative_language(lines)

    # Also check the fully-unavailable case, and a hand-built tie, since
    # those are the paths most likely to have leaked a verdict word before.
    _assert_no_evaluative_language(summarize({}))


def test_summarize_reports_baseline_comparison_in_neutral_wording():
    model_mlpd = {"n": 5, "mlpd": -1.24, "n_floored": 0, "n_no_draws": 0}
    baseline = {"n": 5, "mlpd": -1.55, "n_train_input": 5, "n_train_dropped": 0, "n_floored": 0}

    lines = summarize(
        {"mlpd": model_mlpd, "baseline_pooled_mean": baseline},
        expected=["mlpd", "baseline_pooled_mean"],
    )
    assert (
        "mean log predictive density: estimate -1.24, pooled-mean baseline -1.55, "
        "difference +0.31 (higher is a closer fit)" in lines
    )
    _assert_no_evaluative_language(lines)


def test_summarize_reports_exact_tie_as_equal():
    model_mlpd = {"n": 5, "mlpd": -1.50, "n_floored": 0, "n_no_draws": 0}
    baseline = {"n": 5, "mlpd": -1.50, "n_train_input": 5, "n_train_dropped": 0, "n_floored": 0}

    lines = summarize(
        {"mlpd": model_mlpd, "baseline_pooled_mean": baseline},
        expected=["mlpd", "baseline_pooled_mean"],
    )
    assert any("difference 0.00 (equal)" in line for line in lines)
    _assert_no_evaluative_language(lines)


def test_summarize_surfaces_exclusion_counts():
    train_y = np.array([1.0, 2.0, np.nan, 3.0])
    test_y = np.array([1000.0])
    bpm = baseline_pooled_mean(train_y, test_y)

    lines = summarize({"baseline_pooled_mean": bpm}, expected=["baseline_pooled_mean"])
    assert len(lines) == 1
    line = lines[0]
    assert "1 training points dropped as non-finite" in line
    assert "1 at the density floor" in line
