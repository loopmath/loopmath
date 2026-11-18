"""The family flag of `eval judges` and how the recovery harness counts it.

- Holm's step-down rule against hand-worked cases, and the p-value against the mean and sd by hand.
- A direct simulation: reports of k family contrasts with no family effect, each with calibrated
  posterior draws (centred on an observation drawn around the truth 0, with the observation's sd). The
  share of reports with any flag must stay at or under 0.10 for every k, while the per-contrast 90% rule
  of 0.4.4 flags 1 - 0.9^k of them. The same with draws that repeat or follow one another within a fit
  (repeated draws, a sticky sampler, chains at the effective sample size floor).
- The harness on records shaped like `gated` (2 model judges, 3 families, 3 dimensions: 18 raw centred
  family terms, 6 printed contrasts): false alarms are counted from the printed contrasts with the
  product flag, not from the raw terms or from the interval, and the raw terms are still checked for
  undercoverage (the centred cancellation case); records made before the flag are incomplete, never
  counted by their interval; real contrasts are judged by coverage over enough real-effect studies, and
  found is counted only where the flag is known.

All data is synthetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import recovery_stats as RS
from loopmath.eval import judges as J


# ---------------------------------------------------------------------------------------------------
# The rule

@pytest.mark.parametrize("p, want", [
    # k = 4, level 0.10: bounds 0.025, 0.0333, 0.05, 0.10 for the 1st to 4th smallest
    ([0.02, 0.03, 0.06, 0.2], [True, True, False, False]),     # 0.06 > 0.05 stops the rule
    ([0.01, 0.04, 0.03, 0.005], [True, True, True, True]),      # 0.005, 0.01, 0.03, 0.04 each under its bound
    # k = 3: bounds 0.0333, 0.05, 0.10; 0.04 > 0.0333 stops at once, though 0.09 < 0.10
    ([0.04, 0.045, 0.09], [False, False, False]),
    ([0.0333], [True]),                                        # k = 1: the bound is the level
    ([0.1], [True]),
    ([0.1000001], [False]),
    ([], []),
])
def test_holm_against_hand_worked_cases(p, want):
    assert J.holm(p, 0.10) == want


def test_holm_keeps_more_than_bonferroni_in_a_hand_worked_case():
    """k = 4 at 0.10: Bonferroni rejects only p <= 0.025 (one of these); Holm also takes 0.03 <= 0.10 / 3."""
    p = [0.02, 0.03, 0.06, 0.2]
    assert sum(v <= 0.10 / 4 for v in p) == 1 and sum(J.holm(p, 0.10)) == 2


def test_the_p_value_from_the_mean_and_sd_by_hand():
    """2 T(-|mean| / (sd sqrt(1 + 1/100)); 99), T the t distribution, worked from the draws' mean and sd."""
    from scipy import stats
    # [1, 2, 3]: mean 2, sd (ddof 1) 1
    assert J.normal_p([1.0, 2.0, 3.0]) == pytest.approx(2 * stats.t.cdf(-2 / math.sqrt(1.01), 99), rel=1e-12)
    assert J.normal_p([1.0, 2.0, 3.0]) == pytest.approx(2 * stats.t.sf(2 / math.sqrt(1.01), 99), rel=1e-12)
    assert J.normal_p([1.0, 2.0, 3.0], ess=1e12) == pytest.approx(2 * stats.norm.cdf(-2), rel=1e-6)   # 0.0455
    assert J.normal_p([np.nan, 1.0, 2.0, 3.0]) == J.normal_p([1.0, 2.0, 3.0])
    assert J.normal_p([-1.0, 1.0]) == 1.0                                       # mean 0
    assert J.normal_p([5.0, 5.0]) == 0.0 and J.normal_p([0.0, 0.0]) == 1.0      # no spread
    assert J.normal_p([]) == 1.0
    assert J.FAMILY_ESS == 100.0


def test_the_ess_floor_is_the_one_eval_fit_checks():
    from loopmath.eval import fit
    assert J.FAMILY_ESS == fit.ESS_MIN and "family" in fit.CHECKED


def test_repeating_every_draw_leaves_the_p_value_alone():
    """The p-value reads the mean and sd: n draws each saved m times keep the mean, and the sd (ddof 1)
    becomes sd x sqrt(m (n - 1) / (m n - 1)), a factor within 0.3% of 1 at n = 200; a count of draws on
    one side of 0 would instead treat them as m n independent draws."""
    from scipy import stats
    x = np.random.default_rng(3).normal(0.4, 0.2, size=200)
    for m in (2, 5):
        sd = x.std(ddof=1) * math.sqrt(m * 199 / (m * 200 - 1))
        assert J.normal_p(np.repeat(x, m)) == pytest.approx(
            2 * stats.t.sf(abs(x.mean()) / (sd * math.sqrt(1.01)), 99), rel=1e-9)
        assert J.normal_p(np.repeat(x, m)) == pytest.approx(J.normal_p(x), rel=0.02)


def _within(rng: np.random.Generator, kind: str, k: int, S: int) -> np.ndarray:
    """k contrasts' posterior draws around 0, sd 1, shape (k, S), dependent within a fit as `kind` says:
    `independent`; `repeated-m` (S / m draws each saved m times, as Laplace repeats a theta point);
    `sticky-h` (a sampler that keeps its previous draw with chance h); `ar-rho-c` (c stationary AR(1)
    chains with lag-1 correlation rho, effective sample size S (1 - rho) / (1 + rho))."""
    if kind == "independent":
        return rng.standard_normal((k, S))
    name, *arg = kind.split("-")
    if name == "repeated":
        m = int(arg[0])
        return np.repeat(rng.standard_normal((k, S // m)), m, axis=1)
    if name == "sticky":
        fresh = rng.random((k, S)) >= float(arg[0])
        fresh[:, 0] = True
        idx = np.maximum.accumulate(np.where(fresh, np.arange(S), 0), axis=1)
        return np.take_along_axis(rng.standard_normal((k, S)), idx, axis=1)
    rho, c = float(arg[0]), int(arg[1])
    e = rng.standard_normal((k, c, S // c))
    x = np.empty_like(e)
    x[:, :, 0] = e[:, :, 0]
    for t in range(1, S // c):
        x[:, :, t] = rho * x[:, :, t - 1] + math.sqrt(1 - rho * rho) * e[:, :, t]
    return x.reshape(k, S)


def _payload(draws: np.ndarray, judges: int) -> list[dict]:
    """A judges list as `eval judges` builds it, with one own-family row per contrast (rows dealt to the
    judges in turn), each row's interval and p-value from its draws."""
    out = [{"id": f"j{n}", "family": {"own_family": {"dims": []}, "reference_family": None}} for n in range(judges)]
    for i, x in enumerate(draws):
        lo, hi = np.percentile(x, [5.0, 95.0])
        out[i % judges]["family"]["own_family"]["dims"].append(
            {"dimension": f"d{i}", "excludes_zero": bool(lo > 0 or hi < 0), "p_value": J.normal_p(x)})
    return out


def _false_flag_share(k: int, S: int, kind: str, rho: float, reports: int, seed: int) -> tuple[float, float]:
    """Reports of k contrasts with true value 0: each observation N(0, 1) (correlated between contrasts by
    rho as in the test below), posterior N(obs, 1): calibrated. Share with any flag, and with any 90%
    interval excluding 0."""
    rng = np.random.default_rng(seed)
    if rho < 0:                          # the two judges on one dimension: pairs at rho (FAM-A, linear)
        cov = np.kron(np.eye(k // 2), np.array([[1.0, rho], [rho, 1.0]]))
    else:                                # every two contrasts at rho
        cov = np.full((k, k), rho) + (1.0 - rho) * np.eye(k)
    obs = rng.multivariate_normal(np.zeros(k), cov, size=reports)
    any_flag = any_interval = 0
    for r in range(reports):
        judges = _payload(obs[r][:, None] + _within(rng, kind, k, S), 2)
        rule = J.flag_family(judges)
        assert rule["contrasts"] == k and rule["scope"] == "report" and rule["level"] == 0.10
        rows = [row for _, _, row in J.family_rows(judges)]
        assert all(row["tested_with"] == k for row in rows)
        any_flag += rule["flagged"] > 0
        any_interval += any(row["excludes_zero"] for row in rows)
    return any_flag / reports, any_interval / reports


MARGIN = 3 * math.sqrt(0.10 * 0.90 / 4000)   # 3 binomial standard errors at 0.10 over 4000 reports


@pytest.mark.parametrize("k, rho", [(2, 0.0), (6, 0.0), (10, 0.0), (12, 0.0), (20, 0.0), (6, -0.4), (12, 0.5)])
def test_reports_with_no_family_effect_show_a_false_flag_at_most_10_percent_of_the_time(k, rho):
    """Direct simulation, 4000 reports of k contrasts with true value 0, independent draws within each
    contrast. Correlation between contrasts: rho between every two, or, for rho < 0, between the two
    contrasts of each pair (two judges on one dimension share the anchor's error with opposite signs).
    Holm over the report bounds the chance of any flag by 0.10 under any such correlation; checked with
    a margin of 3 binomial standard errors. The 0.4.4 rule (any 90% interval excluding 0) is far above it
    for k >= 2, near 1 - 0.9^k when independent."""
    share, interval = _false_flag_share(k, 400, "independent", rho, 4000, 1000 + k + int(10 * rho))
    assert share <= 0.10 + MARGIN, (k, rho, share)
    if rho == 0.0:
        assert share >= 0.5 * (1 - (1 - 0.10 / k) ** k), (k, share)      # not trivially conservative
        assert interval == pytest.approx(1 - 0.9 ** k, abs=0.04)


@pytest.mark.parametrize("kind, k, S", [
    ("repeated-2", 20, 400),     # the review's case: 200 draws each saved twice (a count p-value: 0.18)
    ("repeated-5", 50, 1000),    # 200 draws each saved five times (a count p-value: 0.39)
    ("sticky-0.25", 20, 400),    # the review's sampler that keeps its previous draw a quarter of the time
    ("ar-0.6-2", 20, 400),       # two chains at the effective sample size floor: 400 x 0.4 / 1.6 = 100
])
def test_draws_that_depend_on_each_other_within_a_fit_still_meet_the_bound(kind, k, S):
    """The same simulation with draws that repeat or follow one another within each contrast, each with an
    effective sample size of at least 100 (what `eval fit` checks). The flag reads the mean and sd, widened
    for their Monte Carlo error at that floor, so the bound still holds (Round 2 measured 0.083 to 0.092 at
    20,000 reports; the plain normal p-value gave 0.106 to 0.112)."""
    share, _ = _false_flag_share(k, S, kind, 0.0, 4000, 2000 + k + S + len(kind))
    assert share <= 0.10 + MARGIN, (kind, k, S, share)


def test_a_real_effect_among_nulls_is_still_flagged():
    """One contrast at 6 posterior sds from 0 among 11 nulls at 0: its p-value is about 2 T(-6 / 1.005; 99),
    near 3e-8, under 0.10 / 12, and it is flagged; the nulls (observations at 0) are not."""
    rng = np.random.default_rng(7)
    draws = rng.standard_normal((12, 400))
    draws[0] += 6.0
    judges = _payload(draws, 3)
    rule = J.flag_family(judges)
    rows = {row["dimension"]: row for _, _, row in J.family_rows(judges)}
    assert rule["flagged"] == 1 and rows["d0"]["flagged"] is True


def test_a_flag_needs_the_shown_interval_to_exclude_0():
    judges = [{"id": "j", "family": {"own_family": {"dims": [
        {"dimension": "d", "excludes_zero": False, "p_value": 0.0}]}}}]
    assert J.flag_family(judges)["flagged"] == 0


def test_the_scope_is_one_function():
    """`family_groups` alone decides what is tested together: report (one group), judge, contrast."""
    rows = [("a", "own_family", {}), ("a", "own_family", {}), ("b", "own_family", {}), ("b", "reference_family", {})]
    assert J.family_groups(rows, "report") == [[0, 1, 2, 3]]
    assert J.family_groups(rows, "judge") == [[0, 1], [2, 3]]
    assert J.family_groups(rows, "contrast") == [[0], [1], [2], [3]]
    assert J.family_groups([], "report") == []
    with pytest.raises(ValueError):
        J.family_groups(rows, "dimension")
    assert J.FAMILY_SCOPE == "report" and J.FAMILY_LEVEL == 0.10


# ---------------------------------------------------------------------------------------------------
# The harness

JUDGES, FAMILIES, DIMS = ("strong", "cheap"), ("alpha", "beta", "gamma"), ("d1", "d2", "d3")


def _record(seed: int, *, raw_missed: int = 0, printed: list[dict] | None, real: bool = False) -> dict:
    """A record shaped like `gated`: 18 raw centred family rows with truth 0 (the first `raw_missed` with an
    interval that excludes 0) and the printed contrasts `printed`; with `real`, the arm mean only."""
    hit = seed % 10 != 0                 # the arm mean holds its truth in 9 of 10 studies
    lo, hi = (0.3, 0.7) if hit else (0.6, 0.8)
    rows = [{"class": "mu", "name": "mu[a,d1]", "truth": 0.5, "mean": 0.5, "sd": 0.1, "lo": lo, "hi": hi,
             "hit": hit}]
    n = 0
    if not real:
        for j in JUDGES:
            for f in FAMILIES:
                for d in DIMS:
                    lo, hi = (0.01, 0.05) if n < raw_missed else (-0.03, 0.03)
                    rows.append({"class": "family", "name": f"family[{j},{f},{d}]", "truth": 0.0,
                                 "mean": (lo + hi) / 2, "sd": 0.01, "lo": lo, "hi": hi, "hit": not (lo > 0)})
                    n += 1
    return {"schema": RS.RECORD_SCHEMA, "spec": "gated-shaped", "seed": seed, "engine": "none", "options": {},
            "converged": True, "seconds": 1.0, "cpu_seconds": 1.0, "rows": rows, "aggregates": {},
            "family": printed, "expected": [{"class": r["class"], "name": r["name"]} for r in rows],
            "expected_aggregates": {}, "degenerate": [], "no_truth": [], "excluded": {}}


def _contrasts(*, excludes: int = 0, flagged: int = 0, truth: float = 0.0, held: bool = True,
               with_flag: bool = True) -> list[dict]:
    """The 6 contrasts `eval judges` prints for 2 model judges on 3 dimensions: the first `excludes` with an
    interval that excludes 0, the first `flagged` of those with the product flag."""
    out = []
    for i, (j, d) in enumerate((j, d) for j in JUDGES for d in DIMS):
        lo, hi = (0.01, 0.05) if i < excludes else (-0.03, 0.03)
        if truth:
            lo, hi = (truth - 0.02, truth + 0.02) if held else (truth + 0.01, truth + 0.05)
        row = {"judge": j, "kind": "own_family", "family": FAMILIES[JUDGES.index(j)], "dimension": d,
               "truth": truth, "mean": (lo + hi) / 2, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}
        if with_flag:
            row.update({"flagged": i < flagged, "p_value": 0.001 if i < flagged else 0.5, "tested_with": 6})
        out.append(row)
    return out


def test_the_harness_counts_printed_contrasts_with_the_flag_on_a_gated_shaped_record():
    """200 studies. In 40 a raw centred term misses (any of 18), in 30 a printed contrast's interval
    excludes 0, and in 12 of those the product flag is set. The class `family` (truth 0 in every row) is
    judged by the 12 of 200 printed flags (0.06), not by the 40 raw misses (0.20, above 0.10) or the 30
    intervals (0.15); the raw count and the interval count are shown, not gated."""
    records = [_record(s, raw_missed=1 if s < 40 else 0,
                       printed=_contrasts(excludes=1 if s < 30 else 0, flagged=1 if s < 12 else 0))
               for s in range(200)]
    summary = RS.summarise(records)
    z = summary["zero_classes"]["family"]
    assert z["counted"] == "printed family contrasts with the product flag"
    assert (z["studies"], z["studies_flagged"], z["rows"], z["rows_flagged"]) == (200, 12, 1200, 12)
    assert (z["raw_studies"], z["raw_studies_flagged"], z["raw_rows"], z["raw_rows_flagged"]) == (200, 40, 3600, 40)
    assert z["share_of_studies"] == pytest.approx(0.06) and z["within_limit"] is True and z["studies_missing_flag"] == 0
    fam = summary["family"]
    assert (fam["studies_with_a_null_contrast"], fam["studies_flagged"], fam["null_contrasts"]) == (200, 12, 1200)
    assert fam["share_of_studies"] == pytest.approx(0.06) and fam["share_of_studies_interval"] == pytest.approx(0.15)
    items = summary["gate"]["items"]
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "pass"
    assert "(printed contrasts)" in items["zero_classes_false_alarms_within_limit"]["why"]
    assert items["family_false_alarms_within_limit"]["state"] == "pass"
    assert items["family_real_contrasts_covered"]["state"] == "not applicable"
    assert items["coverage_per_class_in_band"]["state"] == "pass"    # the class still leaves the band items


def test_printed_flags_above_the_limit_still_fail():
    """30 of 200 studies with a flagged printed contrast: 0.15, its exact interval above 0.10: both items fail."""
    records = [_record(s, printed=_contrasts(excludes=1 if s < 30 else 0, flagged=1 if s < 30 else 0))
               for s in range(200)]
    items = RS.summarise(records)["gate"]["items"]
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "fail"
    assert items["family_false_alarms_within_limit"]["state"] == "fail"


def test_records_before_the_flag_are_incomplete_not_counted_by_their_interval():
    """Records like gate run 3's: contrasts with an interval and no `flagged`. Both family items say so and
    are incomplete; no false alarm is counted from the intervals; coverage of real contrasts is still read."""
    records = [_record(s, printed=_contrasts(excludes=1 if s < 5 else 0, with_flag=False)) for s in range(200)]
    summary = RS.summarise(records)
    fam, z = summary["family"], summary["zero_classes"]["family"]
    assert fam["studies_without_flag"] == 200 and fam["studies_flagged"] == 0 and fam["share_of_studies"] is None
    assert z["studies_missing_flag"] == 200 and z["studies"] == 0
    items = summary["gate"]["items"]
    assert items["family_false_alarms_within_limit"]["state"] == "incomplete"
    assert "before the family flag" in items["family_false_alarms_within_limit"]["why"]
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "incomplete"
    assert summary["gate"]["state"] == "incomplete"
    # one old record among new ones is enough to leave the items incomplete
    mixed = [_record(s, printed=_contrasts(with_flag=s != 0)) for s in range(200)]
    assert RS.summarise(mixed)["gate"]["items"]["family_false_alarms_within_limit"]["state"] == "incomplete"


@pytest.mark.parametrize("held_every, state", [(10, "pass"), (2, "fail"), (1000, "fail")])
def test_real_family_contrasts_are_judged_by_coverage(held_every, state):
    """200 studies of 6 real contrasts (truth 0.06), all held except in every `held_every`-th study: held
    in 0.90 (inside 85 to 95%), 0.50 (below) or 0.995 (above). Found is reported from the flag."""
    records = [_record(s, real=True, printed=_contrasts(truth=0.06, held=s % held_every != 0, flagged=3))
               for s in range(200)]
    summary = RS.summarise(records)
    fam = summary["family"]
    assert fam["simulated_effects"] == 1200 and fam["found"] == pytest.approx(0.5)
    want = 1 - math.ceil(200 / held_every) / 200
    assert fam["held"] == pytest.approx(want)
    item = summary["gate"]["items"]["family_real_contrasts_covered"]
    assert item["state"] == state, item
    assert "found 0.500 of 1200" in item["why"]


def _cancelling_record(seed: int) -> dict:
    """The review's case: two model judges, three families, one dimension, every true family term 0.
    strong's centred estimates are (0, +1, -1) over (alpha, beta, gamma) and cheap's (+1, 0, -1), each with
    an interval of width 0.02. strong's own family is alpha and cheap's beta, so both printed contrasts are
    0 (the other two terms cancel) and nothing is flagged; 4 of the 6 raw intervals miss 0 in every study."""
    hit = seed % 10 != 0
    lo, hi = (0.3, 0.7) if hit else (0.6, 0.8)
    rows = [{"class": "mu", "name": "mu[a,d1]", "truth": 0.5, "mean": 0.5, "sd": 0.1, "lo": lo, "hi": hi,
             "hit": hit}]
    estimates = {"strong": (0.0, 1.0, -1.0), "cheap": (1.0, 0.0, -1.0)}
    for j, values in estimates.items():
        for f, m in zip(FAMILIES, values):
            rows.append({"class": "family", "name": f"family[{j},{f},d1]", "truth": 0.0, "mean": m, "sd": 0.006,
                         "lo": m - 0.01, "hi": m + 0.01, "hit": m - 0.01 <= 0.0 <= m + 0.01})
    printed = [{"judge": j, "kind": "own_family", "family": FAMILIES[n], "dimension": "d1", "truth": 0.0,
                "mean": 0.0, "lo": -0.01, "hi": 0.01, "excludes_zero": False, "flagged": False, "p_value": 1.0,
                "tested_with": 2} for n, j in enumerate(JUDGES)]
    return {"schema": RS.RECORD_SCHEMA, "spec": "cancelling", "seed": seed, "engine": "none", "options": {},
            "converged": True, "seconds": 1.0, "cpu_seconds": 1.0, "rows": rows, "aggregates": {},
            "family": printed, "expected": [{"class": r["class"], "name": r["name"]} for r in rows],
            "expected_aggregates": {}, "degenerate": [], "no_truth": [], "excluded": {}}


def test_raw_family_terms_that_cancel_in_every_printed_contrast_still_fail():
    """Regression for the centred cancellation case, 200 studies: 0 of 200 reports flagged, but the raw
    terms cover 400 / 1200 = 1/3 and four named terms never cover their truth. The printed-contrast count
    passes; the raw undercoverage item fails, and with it the spec."""
    summary = RS.summarise([_cancelling_record(s) for s in range(200)])
    z = summary["zero_classes"]["family"]
    assert (z["studies_flagged"], z["studies"]) == (0, 200) and (z["raw_rows"], z["raw_rows_flagged"]) == (1200, 800)
    assert summary["classes"]["family"]["coverage"] == pytest.approx(1 / 3)
    items = summary["gate"]["items"]
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "pass"
    assert items["family_false_alarms_within_limit"]["state"] == "pass"
    assert items["family_real_contrasts_covered"]["state"] == "not applicable"
    raw = items["family_raw_terms_not_undercovered"]
    assert raw["state"] == "fail", raw
    assert "cover 0.333 of 1200 in 200 studies" in raw["why"] and "4 terms clearly below 0.85" in raw["why"]
    assert summary["gate"]["state"] == "fail"


@pytest.mark.parametrize("raw_missed, state", [(1, "pass"), (0, "pass")])
def test_raw_family_terms_may_over_cover_at_the_all_zero_boundary(raw_missed, state):
    """Raw terms that cover 0 in nearly every study (above 0.95, the band's top) pass: over-coverage at the
    all-zero boundary is allowed. Only undercoverage fails."""
    records = [_record(s, raw_missed=raw_missed if s < 40 else 0, printed=_contrasts()) for s in range(200)]
    summary = RS.summarise(records)
    assert summary["classes"]["family"]["coverage"] > 0.95
    assert summary["gate"]["items"]["family_raw_terms_not_undercovered"]["state"] == state


def test_raw_family_terms_with_effects_are_left_to_the_band_items():
    """When a raw family term has a simulated effect the class is not all zero: the band items judge it,
    and the raw undercoverage item is not applicable."""
    records = [_record(s, real=True, printed=_contrasts(truth=0.06, flagged=6)) for s in range(200)]
    assert RS.summarise(records)["gate"]["items"]["family_raw_terms_not_undercovered"]["state"] == "not applicable"


def test_found_is_counted_only_where_the_flag_is_known():
    """200 studies of 6 real contrasts held in 9 of 10 studies; only study 0 has the flag, with 3 of its 6
    flagged. Coverage is read from every interval (1200 effects); found is 3/6 measured and 1194 unknown,
    never 3 of 1200 or 0.5 of 1200."""
    records = [_record(s, real=True, printed=_contrasts(truth=0.06, held=s % 10 != 0, flagged=3, with_flag=s == 0))
               for s in range(200)]
    summary = RS.summarise(records)
    fam = summary["family"]
    assert (fam["simulated_effects"], fam["effects_with_flag"], fam["effects_found"], fam["effects_without_flag"]) \
        == (1200, 6, 3, 1194)
    assert fam["found"] is None and fam["held"] == pytest.approx(0.9)
    item = summary["gate"]["items"]["family_real_contrasts_covered"]
    assert "found 3/6 measured; 1194 unknown" in item["why"], item
    assert "found 0.500" not in item["why"]


def test_real_contrast_coverage_needs_enough_real_effect_studies():
    """200 valid records, but only study 0 has real contrasts (10 of them, 9 held: 0.90). Ten rows of one
    study are not 200 studies: the item is incomplete, not a pass."""
    records = [_record(s, printed=_contrasts()) for s in range(1, 200)]
    real = [dict(row, dimension=f"d{i}", lo=0.04, hi=0.08) for i, row in
            enumerate(_contrasts(truth=0.06, flagged=6) + _contrasts(truth=0.06, flagged=6)[:4])]
    real[0].update(lo=0.07, hi=0.09)
    records.append(_record(0, printed=real))
    summary = RS.summarise(records)
    fam = summary["family"]
    assert fam["held"] == pytest.approx(0.9) and fam["held_studies"] == 1 and fam["held_studies_min"] == 1
    item = summary["gate"]["items"]["family_real_contrasts_covered"]
    assert item["state"] == "incomplete", item
    assert "1 of 200 real-effect studies" in item["why"]


def test_real_contrast_coverage_needs_every_real_contrast_in_enough_studies():
    """200 studies with real contrasts, but one contrast (strong on d3) is measured in only 50 of them: the
    sparsest contrast decides, and the item is incomplete though coverage is 0.90."""
    records = []
    for s in range(200):
        rows = _contrasts(truth=0.06, held=s % 10 != 0, flagged=6)
        if s >= 50:
            rows = [row for row in rows if not (row["judge"] == "strong" and row["dimension"] == "d3")]
        records.append(_record(s, real=True, printed=rows))
    summary = RS.summarise(records)
    assert summary["family"]["held_studies"] == 200 and summary["family"]["held_studies_min"] == 50
    item = summary["gate"]["items"]["family_real_contrasts_covered"]
    assert item["state"] == "incomplete" and "the sparsest real contrast in 50" in item["why"], item
