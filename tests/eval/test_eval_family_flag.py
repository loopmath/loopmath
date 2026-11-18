"""The family flag of `eval judges` and how the recovery harness counts it.

- Holm's step-down rule against hand-worked cases, and the tail share against counted draws.
- A direct simulation: reports of k family contrasts with no family effect, each with calibrated
  posterior draws (centred on an observation drawn around the truth 0, with the observation's sd). The
  share of reports with any flag must stay at or under 0.10 for every k, while the per-contrast 90% rule
  of 0.4.4 flags 1 - 0.9^k of them.
- The harness on records shaped like `gated` (2 model judges, 3 families, 3 dimensions: 18 raw centred
  family terms, 6 printed contrasts): false alarms are counted from the printed contrasts with the
  product flag, not from the raw terms or from the interval; records made before the flag are
  incomplete, never counted by their interval; real contrasts are judged by coverage.

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


def test_tail_share_counts_draws():
    """2 (b + 1) / (S + 1), b the draws on the smaller side of 0, worked by hand."""
    assert J.tail_share([-1.0, 1.0, 2.0, 3.0]) == pytest.approx(2 * 2 / 5)      # b = 1, S = 4
    assert J.tail_share([0.0, 1.0, 2.0, 3.0]) == pytest.approx(2 * 2 / 5)       # a draw at 0 counts on both sides
    assert J.tail_share([1.0, 2.0]) == pytest.approx(2 / 3)                     # b = 0: never 0
    assert J.tail_share([-1.0, 1.0]) == 1.0                                     # 4 / 3, capped at 1
    assert J.tail_share([np.nan, -1.0, 1.0, 2.0, 3.0]) == pytest.approx(2 * 2 / 5)
    assert J.tail_share(np.r_[-np.ones(20), np.ones(380)]) == pytest.approx(42 / 401)
    assert J.tail_share([]) == 1.0


def test_the_plain_share_would_not_hold_the_bound():
    """Why `+ 1`: with q the true chance of a draw at or below 0 drawn uniformly (a calibrated posterior at a
    true 0), b of S = 400 is uniform on 0..400, so P(2 b / S <= 0.005) = P(b <= 1) = 2 / 401, twice the
    0.0025 one side should get; P(2 (b + 1) / 401 <= 0.005) = P(b = 0) = 1 / 401 <= 0.0025. Checked by
    simulation of the draws."""
    rng = np.random.default_rng(11)
    q = rng.uniform(size=200_000)
    b = rng.binomial(400, q)
    assert np.mean(2 * b / 400 <= 0.005) == pytest.approx(2 / 401, abs=0.0006)
    assert np.mean(2 * (b + 1) / 401 <= 0.005) == pytest.approx(1 / 401, abs=0.0005)


def _payload(draws: np.ndarray, judges: int) -> list[dict]:
    """A judges list as `eval judges` builds it, with one own-family row per contrast (rows dealt to the
    judges in turn), each row's interval and tail share from its draws."""
    out = [{"id": f"j{n}", "family": {"own_family": {"dims": []}, "reference_family": None}} for n in range(judges)]
    for i, x in enumerate(draws):
        lo, hi = np.percentile(x, [5.0, 95.0])
        out[i % judges]["family"]["own_family"]["dims"].append(
            {"dimension": f"d{i}", "excludes_zero": bool(lo > 0 or hi < 0), "tail_share": J.tail_share(x)})
    return out


@pytest.mark.parametrize("k, rho", [(2, 0.0), (6, 0.0), (10, 0.0), (12, 0.0), (20, 0.0), (6, -0.4), (12, 0.5)])
def test_reports_with_no_family_effect_show_a_false_flag_at_most_10_percent_of_the_time(k, rho):
    """Direct simulation, 4000 reports of k contrasts with true value 0. Each contrast's observation is
    N(0, 1), correlated: rho between every two contrasts, or, for rho < 0, between the two contrasts of
    each pair (two judges on one dimension share the anchor's error with opposite signs); its posterior N(obs, 1)
    from 400 draws: calibrated. Holm over the report bounds the chance of any flag by 0.10 under any
    correlation; the bound is checked with a binomial margin of 3 standard errors at 0.10. The 0.4.4
    rule (any 90% interval excluding 0) is shown far above it for k >= 2, near 1 - 0.9^k when
    independent."""
    rng = np.random.default_rng(1000 + k + int(10 * rho))
    reports, S = 4000, 400
    if rho < 0:                          # the two judges on one dimension: pairs at rho (FAM-A, linear)
        cov = np.kron(np.eye(k // 2), np.array([[1.0, rho], [rho, 1.0]]))
    else:                                # every two contrasts at rho
        cov = np.full((k, k), rho) + (1.0 - rho) * np.eye(k)
    obs = rng.multivariate_normal(np.zeros(k), cov, size=reports)
    any_flag = any_interval = 0
    for r in range(reports):
        draws = obs[r][:, None] + rng.standard_normal((k, S))
        judges = _payload(draws, 2)
        rule = J.flag_family(judges)
        assert rule["contrasts"] == k and rule["scope"] == "report" and rule["level"] == 0.10
        rows = [row for _, _, row in J.family_rows(judges)]
        assert all(row["tested_with"] == k for row in rows)
        any_flag += rule["flagged"] > 0
        any_interval += any(row["excludes_zero"] for row in rows)
    share = any_flag / reports
    assert share <= 0.10 + 3 * math.sqrt(0.10 * 0.90 / reports), (k, rho, share)
    if rho == 0.0:
        assert share >= 0.5 * (1 - (1 - 0.10 / k) ** k), (k, share)      # not trivially conservative
        assert any_interval / reports == pytest.approx(1 - 0.9 ** k, abs=0.04)


def test_a_real_effect_among_nulls_is_still_flagged():
    """One contrast at 6 posterior sds from 0 among 11 nulls at 0: no draw of it is at or below 0, so its
    tail share is 2 / 401, under 0.10 / 12, and it is flagged; the nulls (observations at 0) are not."""
    rng = np.random.default_rng(7)
    draws = rng.standard_normal((12, 400))
    draws[0] += 6.0
    judges = _payload(draws, 3)
    rule = J.flag_family(judges)
    rows = {row["dimension"]: row for _, _, row in J.family_rows(judges)}
    assert rule["flagged"] == 1 and rows["d0"]["flagged"] is True


def test_a_flag_needs_the_shown_interval_to_exclude_0():
    judges = [{"id": "j", "family": {"own_family": {"dims": [
        {"dimension": "d", "excludes_zero": False, "tail_share": 0.0}]}}}]
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
            row.update({"flagged": i < flagged, "tail_share": 0.001 if i < flagged else 0.5, "tested_with": 6})
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
