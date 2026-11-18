"""The benchmark and recovery harness for comparing engines: `scripts/eval_engine_bench.py` and
`recovery_stats.py`.

Every test here is marked `slow` and runs only with LOOPMATH_EVAL_SLOW=1. Most need no fit and take
seconds; the last ones run the harness with real fits on small studies.

- The intervals the statistics use, against independent values: binomial tail sums for the exact
  interval, the score equation for the Wilson interval, the exact variance of a share over clusters of
  very different sizes, and direct simulations of clustered hits with equal and unequal sizes.
- The two ways a pooled check misleads, built as records: pooled coverage of 90% with one parameter at
  0%, and one false family contrast in ten in every study. The old checks pass both; these must not.
- A class whose truth is 0 in every row: it over-covers by nature, so it is judged by its false alarms
  once per study (limit 10%) and not by the band; one nonzero truth puts it back in the band. An item
  with nothing to judge (no such class, or no class left for the band) is not applicable: it never
  blocks a spec, and never stands in for missing studies or an empty inventory.
- Missing evidence is never a pass: a gate with specs or a scale kind missing, an unconverged or failed
  size-target timing, constant, missing or non-finite draws of an expected parameter, a class seen in
  few studies, and records made without the family measurement.
- The records of catalog designs, through the real `fit.fit_design` with the sampler swapped for draws
  of known law (truth plus a random offset plus noise, both of sd s): the 90% interval of such draws
  holds the truth with probability 0.90, so every class must cover near 0.90 and show no bias. One
  design is unanchored, so its location is the judges' mean bias.
- The worker budgets: workers together stay inside the run's cores, CPU minutes and fits.
- The harness with `eval fit` itself, PyMC first, on two small studies; and the Laplace engine once
  `fit.fit_design` takes an engine.
- With LOOPMATH_EVAL_BENCH_OUT naming the folder of a finished large run, the gate's spec items there.

Every study here is simulated.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, stats as sps

import recovery_stats as RS
from loopmath.eval import contracts as C
from loopmath.eval import engine_pymc
from loopmath.eval import fit as FIT
from loopmath.eval import model as M
from loopmath.eval import simulate, specs

pytestmark = pytest.mark.slow

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "scripts" / "eval_engine_bench.py"
FAKE_DRAWS, FAKE_CHAINS = 200, 2          # 400 saved draws, the fewest a fit file takes


# ---------------------------------------------------------------------------------------------------
# Intervals against independent values


@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (180, 200), (199, 200), (1, 1)])
def test_clopper_pearson_ends_are_binomial_tail_points(k, n):
    """At the upper end, k or fewer hits have probability 2.5%; at the lower end, k or more do. Checked
    with binomial tail sums, not the beta quantiles the helper uses; and the closed forms at k = 0 and n."""
    lo, hi = RS.clopper_pearson(k, n)
    if k < n:
        assert sps.binom.cdf(k, n, hi) == pytest.approx(0.025, abs=1e-9)
    else:
        assert hi == 1.0 and lo == pytest.approx(0.025 ** (1.0 / n), abs=1e-12)
    if k > 0:
        assert sps.binom.sf(k - 1, n, lo) == pytest.approx(0.025, abs=1e-9)
    else:
        assert lo == 0.0 and hi == pytest.approx(1.0 - 0.025 ** (1.0 / n), abs=1e-12)


@pytest.mark.parametrize("p,n", [(0.9, 200), (0.5, 7.5), (0.97, 40.0), (0.0, 30)])
def test_wilson_ends_solve_the_score_equation(p, n):
    """The Wilson interval is every p0 with |p - p0| <= z sqrt(p0 (1 - p0) / n). Its ends are found here by
    root finding on that equation, apart from the closed form."""
    z = sps.norm.ppf(0.975)
    lo, hi = RS.wilson(p, n)

    def gap(p0):
        return abs(p - p0) - z * np.sqrt(p0 * (1 - p0) / n)

    want_hi = optimize.brentq(gap, max(p, 1e-12), 1.0 - 1e-15)
    want_lo = 0.0 if p == 0 else optimize.brentq(gap, 1e-15, p)
    assert hi == pytest.approx(want_hi, abs=1e-9)
    assert lo == pytest.approx(want_lo, abs=1e-9)


def test_cluster_interval_holds_its_level_when_hits_share_a_study():
    """Hits of one study move together (each study has its own chance, drawn around 0.90). Over 2000
    simulated runs of 200 studies with 6 rows each, the cluster interval holds 0.90 at least at its 95%
    (it is conservative for equal sizes, allowing for Monte Carlo error); a plain binomial interval over
    the 1200 rows, which treats them as independent, holds it far less often."""
    rng = np.random.default_rng(20261116)
    reps, studies, rows, mean, spread = 2000, 200, 6, 0.90, 5.0
    chance = rng.beta(mean * spread, (1 - mean) * spread, size=(reps, studies))
    hits = rng.binomial(rows, chance)
    cluster = plain = 0
    for rep in range(reps):
        got = RS.cluster_share(hits[rep], np.full(studies, rows))
        cluster += got["lo"] <= mean <= got["hi"]
        lo, hi = RS.wilson(hits[rep].sum() / (studies * rows), studies * rows)
        plain += lo <= mean <= hi
    assert cluster / reps >= 0.94, cluster / reps
    assert plain / reps < 0.88, plain / reps


UNEQUAL = np.array([1.0] * 190 + [100.0] * 10)     # 190 studies of one row, 10 of 100 rows


def test_cluster_interval_on_190_single_rows_and_10_studies_of_100():
    """Every study covers with chance 0.9 and a study's rows move together. The pooled share then has
    variance 0.9 x 0.1 x (190 + 10 x 100^2) / 1190^2 = 0.00636755878822 (written out here, not taken from
    the helper), an effective 14.1341 trials, far below the 200 studies. With 171 single-row hits and 9
    large-study hits the share is 0.9; the interval must hold 0.9 and rest on about 14 trials."""
    exact_var = 0.9 * 0.1 * (190 + 10 * 100 ** 2) / 1190 ** 2
    assert exact_var == pytest.approx(0.00636755878822, abs=1e-14)
    hits = np.array([1.0] * 171 + [0.0] * 19 + [100.0] * 9 + [0.0])
    got = RS.cluster_share(hits, UNEQUAL)
    assert got["share"] == pytest.approx(0.9, abs=1e-12)
    assert got["n_size"] == pytest.approx(0.9 * 0.1 / exact_var, abs=1e-9)
    assert got["n_eff"] <= got["n_size"] + 1e-12 and got["n_eff"] < 15
    assert got["lo"] <= 0.9 <= got["hi"]
    assert got["lo"] < 0.75 and got["hi"] > 0.95, got     # about 14 trials, not 200


def test_cluster_interval_holds_its_level_with_unequal_study_sizes():
    """4000 simulated runs of the 190 + 10 layout, each study covering with chance 0.9: the interval holds
    0.9 in at least 94% of runs (95% less Monte Carlo error). The interval from the variance over studies
    alone, the part that fails, is shown to fail: when the ten large studies all hit, it is near 0 wide."""
    rng = np.random.default_rng(1190)
    reps, held, spread_only = 4000, 0, 0
    for _ in range(reps):
        hit = rng.random(200) < 0.9
        hits = UNEQUAL * hit
        got = RS.cluster_share(hits, UNEQUAL)
        held += got["lo"] <= 0.9 <= got["hi"]
        p, N, S = hits.sum() / UNEQUAL.sum(), UNEQUAL.sum(), 200
        v = S / (S - 1) * ((hits - p * UNEQUAL) ** 2).sum() / N ** 2
        lo, hi = RS.wilson(p, p * (1 - p) / v) if 0 < p < 1 else (p, p)
        spread_only += lo <= 0.9 <= hi
    assert held / reps >= 0.94, held / reps
    assert spread_only / reps < 0.90, spread_only / reps


# ---------------------------------------------------------------------------------------------------
# Where pooled checks mislead, as records


def synthetic(studies: int, rows_of) -> list[dict]:
    """Records with no fit behind them: `rows_of(seed)` gives (rows, family contrasts) of one study."""
    out = []
    for seed in range(studies):
        rows, family = rows_of(seed)
        out.append({"schema": RS.RECORD_SCHEMA, "spec": "synthetic", "seed": seed, "engine": "none", "options": {},
                    "converged": True, "seconds": 1.0, "cpu_seconds": 1.0, "rows": rows, "aggregates": {},
                    "family": family, "expected": [{"class": r["class"], "name": r["name"]} for r in rows],
                    "expected_aggregates": {}, "degenerate": [], "no_truth": [], "excluded": {}})
    return out


def row(klass: str, name: str, hit: bool, error: float = 0.0) -> dict:
    """One parameter row with truth 0.5 whose interval holds the truth or not, as `hit` says."""
    lo, hi = (0.3, 0.7) if hit else (0.6, 0.8)
    return {"class": klass, "name": name, "truth": 0.5, "mean": 0.5 + error, "sd": 0.1, "lo": lo, "hi": hi,
            "hit": hit}


def test_one_parameter_at_0_percent_is_caught_where_pooled_coverage_is_90_percent():
    """The first way: ten parameters of one class over 200 studies, nine always covered and one
    never. Pooled, that is 90%, inside the band, which is what the 0.4.4 check read. Read one by one, the
    bad parameter's exact interval lies far below the band, and the nine at 100% lie above it (an
    interval that always holds the truth is too wide), so the spec fails."""
    records = synthetic(200, lambda seed: ([row("tau", f"tau[{n}]", n != 0) for n in range(10)], []))
    summary = RS.summarise(records)
    assert summary["classes"]["tau"]["coverage"] == pytest.approx(0.90)
    assert summary["classes"]["tau"]["in_band"]
    outside = {p["name"]: p["coverage"] for p in summary["parameters_outside"]}
    assert outside == {f"tau[{n}]": (0.0 if n == 0 else 1.0) for n in range(10)}
    assert summary["gate"]["items"]["no_parameter_outside_band"]["state"] == "fail"
    assert summary["gate"]["pass"] is False


def test_one_parameter_at_50_percent_among_good_ones_is_the_only_one_named():
    """Nineteen parameters that hold the truth in 90% of studies and one that holds it in 50%: pooled
    coverage is 88%, inside the band; one by one, only the bad parameter is outside it."""
    def one(seed):
        rows = [row("sigma", f"sigma[{n}]", (seed + n) % 10 != 0) for n in range(1, 20)]
        return [row("sigma", "sigma[0]", seed % 2 == 0)] + rows, []

    summary = RS.summarise(synthetic(200, one))
    assert summary["classes"]["sigma"]["coverage"] == pytest.approx(0.88)
    assert summary["classes"]["sigma"]["in_band"]
    assert [p["name"] for p in summary["parameters_outside"]] == ["sigma[0]"]
    assert summary["gate"]["pass"] is False


def test_one_false_family_contrast_in_ten_flags_every_study():
    """The second way: every study has ten contrasts with no simulated effect and one of them
    excludes 0. Per contrast that is 10%, within the 0.4.4 check's limit; per study it is 100%."""
    def one(seed):
        family = [{"judge": "j", "kind": "own_family", "family": f"f{n}", "dimension": "d", "truth": 0.0,
                   "mean": 0.0, "lo": -1.0, "hi": 1.0, "excludes_zero": n == seed % 10,
                   "flagged": n == seed % 10} for n in range(10)]
        return [row("mu", "mu[a,d]", True)], family

    summary = RS.summarise(synthetic(200, one))
    fam = summary["family"]
    assert fam["share_of_contrasts"] == pytest.approx(0.10)
    assert fam["share_of_studies"] == 1.0 and fam["within_limit"] is False and fam["place"] == "above"
    assert summary["gate"]["pass"] is False


def test_arm_means_are_judged_one_by_one():
    """Two arm means with errors of +0.02 and -0.02 in every study: their pooled error is 0, but each one
    alone is biased, and the summary names both."""
    rng = np.random.default_rng(3)
    records = synthetic(50, lambda seed: ([row("mu", "mu[a,d]", True, 0.02 + rng.normal(0, 0.005)),
                                           row("mu", "mu[b,d]", True, -0.02 + rng.normal(0, 0.005))], []))
    summary = RS.summarise(records)
    assert abs(summary["classes"]["mu"]["bias"]) < 0.005
    assert sorted(summary["gate"]["arm_means_biased"]) == ["mu[a,d]", "mu[b,d]"]
    assert summary["gate"]["items"]["arm_means_unbiased"]["state"] == "fail"


def test_a_class_seen_in_few_studies_leaves_the_spec_incomplete():
    """A tau class seen in 10 of 200 studies, 9 of them covered: its rows look fine, but the spec has not
    measured it 200 times, so the inventory item is incomplete and the spec does not pass."""
    def one(seed):
        rows = [row("mu", "mu[a,d]", seed % 10 != 0)]
        if seed < 10:
            rows.append(row("tau", "tau[j,d]", seed != 0))
        return rows, []

    summary = RS.summarise(synthetic(200, one))
    item = summary["gate"]["items"]["inventory_complete"]
    assert item["state"] == "incomplete" and "tau" in item["why"], item
    assert summary["gate"]["state"] != "pass"


def test_records_made_without_the_family_measurement_leave_the_spec_incomplete():
    """200 otherwise clean records made with `--no-family` (`family` null): the family item is incomplete,
    never a pass. Records that measured and found no contrast to read (`family` an empty list) are a
    design with nothing to measure, and that item passes."""
    def clean(family):
        return lambda seed: ([row("mu", "mu[a,d]", seed % 10 != 0)], family)

    skipped = RS.summarise(synthetic(200, clean(None)))
    assert skipped["family"]["studies_skipped"] == 200
    assert skipped["gate"]["items"]["family_false_alarms_within_limit"]["state"] == "incomplete"
    assert skipped["gate"]["state"] == "incomplete"
    measured = RS.summarise(synthetic(200, clean([])))
    assert measured["gate"]["items"]["family_false_alarms_within_limit"]["state"] == "pass"
    half = RS.summarise(synthetic(200, lambda seed: ([row("mu", "mu[a,d]", seed % 10 != 0)],
                                                     None if seed % 2 else [])))
    assert half["gate"]["items"]["family_false_alarms_within_limit"]["state"] == "incomplete"


def zero_row(klass: str, name: str, flagged: bool) -> dict:
    """One parameter row with truth 0 whose interval excludes 0 (a false alarm) or holds it."""
    lo, hi = (0.1, 0.3) if flagged else (-0.2, 0.2)
    return {"class": klass, "name": name, "truth": 0.0, "mean": (lo + hi) / 2, "sd": 0.1, "lo": lo, "hi": hi,
            "hit": not flagged}


def zero_class_records(studies: int, flagged_studies: int, nonzero_in_seed=None, double_in_seed=None) -> list[dict]:
    """One arm mean that holds its truth in 9 of 10 seeds, and four `sd_judge_input` rows with truth 0 (the
    class `family` is counted from the printed contrasts instead, see the tests below); in the first
    `flagged_studies` seeds the first of them excludes 0. `nonzero_in_seed` gives the second such row a
    truth of 0.5 (held) in that seed; `double_in_seed` makes it exclude 0 too in that seed. The family
    contrasts were measured and none has truth 0."""
    def one(seed):
        rows = [row("mu", "mu[a,d]", seed % 10 != 0)]
        for n in range(4):
            if n == 1 and seed == nonzero_in_seed:
                rows.append(row("sd_judge_input", f"sd_judge_input[j{n},d]", True))
            else:
                rows.append(zero_row("sd_judge_input", f"sd_judge_input[j{n},d]", (n == 0 and seed < flagged_studies)
                                     or (n == 1 and seed == double_in_seed)))
        return rows, []
    return synthetic(studies, one)


def test_a_class_with_truth_0_in_every_row_is_judged_by_false_alarms_not_the_band():
    """Four judge-input sds with truth 0 in 200 studies, one interval excluding 0 in 15 studies: the class
    covers 785 of 800 rows (0.98, above the band) and three of its parameters cover 200 of 200 (outside
    the band by their own intervals). That over-coverage is what an interval does at a true 0, so the
    class leaves both band items, which pass, and its own item counts 15 of 200 studies (0.075, within
    0.10) and passes. The family contrast item is unchanged."""
    summary = RS.summarise(zero_class_records(200, 15))
    judged = summary["classes"]["sd_judge_input"]
    assert judged["always_zero"] is True and judged["in_band"] is False
    assert judged["coverage"] == pytest.approx(785 / 800)
    assert sorted(p["name"] for p in summary["parameters_outside"]) == [f"sd_judge_input[j{n},d]" for n in (1, 2, 3)]
    zero = summary["zero_classes"]
    assert set(zero) == {"sd_judge_input"}
    z = zero["sd_judge_input"]
    assert (z["studies"], z["studies_flagged"], z["rows"], z["rows_flagged"]) == (200, 15, 800, 15)
    assert z["share_of_studies"] == pytest.approx(0.075) and z["within_limit"] is True and z["limit"] == 0.10
    lo, hi = binomial_interval(15, 200)
    assert z["lo"] == pytest.approx(lo, abs=1e-9) and z["hi"] == pytest.approx(hi, abs=1e-9)
    assert z["place"] == "undecided"                      # 0.10 lies inside [0.043, 0.121]
    items = summary["gate"]["items"]
    assert items["coverage_per_class_in_band"]["state"] == "pass"
    assert "sd_judge_input" in items["coverage_per_class_in_band"]["why"]
    assert items["no_parameter_outside_band"]["state"] == "pass"
    assert "3 more in classes judged by false alarms" in items["no_parameter_outside_band"]["why"]
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "pass"
    assert items["family_false_alarms_within_limit"] == {
        "state": "pass", "why": "measured; the design has no contrast without a simulated effect"}
    assert summary["gate"]["classes_judged_by_false_alarms"] == ["sd_judge_input"]
    assert summary["gate"]["state"] == "pass", summary["gate"]
    shown = "\n".join(bench_module().render_summary("synthetic", summary))
    assert "zero_classes_false_alarms_within_limit: pass" in shown
    assert "sd_judge_input, 15 of 200 studies with an interval that excludes 0" in shown


def test_a_class_with_truth_0_fails_on_false_alarms_in_30_of_200_studies():
    """The same class with an interval excluding 0 in 30 of 200 studies: 0.15, above 0.10 (its exact
    interval [0.104, 0.207] lies above it), so its item fails and the spec fails, while the band items
    still pass. One study with two intervals excluding 0 counts once."""
    summary = RS.summarise(zero_class_records(200, 30, double_in_seed=0))
    z = summary["zero_classes"]["sd_judge_input"]
    assert (z["studies_flagged"], z["rows_flagged"]) == (30, 31)
    assert z["share_of_studies"] == pytest.approx(0.15) and z["within_limit"] is False
    assert z["lo"] == pytest.approx(binomial_interval(30, 200)[0], abs=1e-9) and z["place"] == "above"
    items = summary["gate"]["items"]
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "fail"
    assert "sd_judge_input" in items["zero_classes_false_alarms_within_limit"]["why"]
    assert items["coverage_per_class_in_band"]["state"] == "pass"
    assert items["no_parameter_outside_band"]["state"] == "pass"
    assert summary["gate"]["state"] == "fail"


def test_a_class_with_truth_0_short_of_studies_is_incomplete_until_it_is_above():
    """In 50 studies (the gate wants 200): 8 flagged (0.16) has an exact interval reaching below 0.10, so
    the item is incomplete; 15 flagged (0.30) lies above 0.10 whatever comes, so it fails now."""
    some = RS.summarise(zero_class_records(50, 8))
    assert binomial_interval(8, 50)[0] < 0.10 < binomial_interval(15, 50)[0]
    assert some["zero_classes"]["sd_judge_input"]["place"] == "undecided"
    assert some["gate"]["items"]["zero_classes_false_alarms_within_limit"]["state"] == "incomplete"
    many = RS.summarise(zero_class_records(50, 15))
    assert many["gate"]["items"]["zero_classes_false_alarms_within_limit"]["state"] == "fail"


def test_one_nonzero_truth_keeps_a_class_in_the_band():
    """The same records with one such row of truth 0.5 in one study: the class is not 0 in every row, so
    it keeps the band, and its over-coverage fails both band items as before."""
    summary = RS.summarise(zero_class_records(200, 15, nonzero_in_seed=7))
    assert summary["zero_classes"] == {} and summary["classes"]["sd_judge_input"]["always_zero"] is False
    items = summary["gate"]["items"]
    assert items["coverage_per_class_in_band"]["state"] == "fail"
    assert summary["gate"]["classes_out_of_band"] == ["sd_judge_input"]
    assert items["no_parameter_outside_band"]["state"] == "fail"
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "not applicable"
    assert "no gated class has truth 0" in items["zero_classes_false_alarms_within_limit"]["why"]
    assert summary["gate"]["state"] == "fail"


@pytest.mark.parametrize("studies", [200, 400, 1000])
def test_a_spec_whose_every_gated_class_has_truth_0_can_pass(studies):
    """One arm mean in each study with truth 0, mean 0, sd 0.1 and interval [-0.2, 0.2], the family measured
    with no contrast: every gated class has truth 0 in every row, so no class is left for the band. The
    band item is not applicable (not incomplete for ever); the false-alarm item counts 0 studies, whose exact
    interval is [0, 1 - 0.025^(1/n)]; and the spec passes."""
    summary = RS.summarise(synthetic(studies, lambda seed: ([zero_row("mu", "mu[a,d]", False)], [])))
    assert set(summary["zero_classes"]) == {"mu"}
    z = summary["zero_classes"]["mu"]
    assert (z["studies"], z["studies_flagged"]) == (studies, 0)
    assert z["lo"] == 0.0 and z["hi"] == pytest.approx(1.0 - 0.025 ** (1.0 / studies), abs=1e-9)
    items = summary["gate"]["items"]
    band = items["coverage_per_class_in_band"]
    assert band["state"] == "not applicable" and "every gated class has truth 0" in band["why"], band
    assert items["zero_classes_false_alarms_within_limit"]["state"] == "pass"
    assert items["no_parameter_outside_band"]["state"] == "pass"
    assert "1 more in classes judged by false alarms" in items["no_parameter_outside_band"]["why"]
    assert items["family_real_contrasts_covered"]["state"] == "not applicable"     # no real family contrast
    assert items["family_raw_terms_not_undercovered"]["state"] == "not applicable"  # no raw family term
    assert {k for k, v in items.items() if v["state"] != "pass"} == {"coverage_per_class_in_band",
                                                                    "family_real_contrasts_covered",
                                                                    "family_raw_terms_not_undercovered"}
    assert summary["gate"]["state"] == "pass" and summary["gate"]["pass"] is True


def test_not_applicable_never_stands_in_for_missing_evidence():
    """The same all-zero spec at 50 of 200 studies: the band item is still not applicable, but the study
    count item is incomplete, so the spec is. Records with no gated class at all measured nothing: there
    the band item stays incomplete, and so does the spec."""
    short = RS.summarise(synthetic(50, lambda seed: ([zero_row("mu", "mu[a,d]", False)], [])))
    assert short["gate"]["items"]["coverage_per_class_in_band"]["state"] == "not applicable"
    assert short["gate"]["items"]["enough_studies"]["state"] == "incomplete"
    assert short["gate"]["state"] == "incomplete"
    empty = RS.summarise(synthetic(200, lambda seed: ([], [])))
    assert empty["classes"] == {} and empty["zero_classes"] == {}
    assert empty["gate"]["items"]["coverage_per_class_in_band"]["state"] == "incomplete"
    assert empty["gate"]["items"]["inventory_complete"]["state"] == "incomplete"
    assert empty["gate"]["state"] == "incomplete"


@pytest.mark.parametrize("studies, state", [(50, "incomplete"), (199, "incomplete"), (200, "pass"), (400, "pass")])
def test_a_spec_with_no_truth_0_class_marks_that_item_not_applicable(studies, state):
    """A spec whose only class has truth 0.5: the false-alarm item has nothing to judge at any count, so it
    is not applicable, below and at the 200 studies the gate wants. The spec is incomplete below 200 because
    of the study count, and passes at 200 and above."""
    summary = RS.summarise(probe_records(studies=studies))
    item = summary["gate"]["items"]["zero_classes_false_alarms_within_limit"]
    assert item == {"state": "not applicable", "why": "no gated class has truth 0 in every row"}
    assert summary["gate"]["items"]["enough_studies"]["state"] == ("pass" if studies >= 200 else "incomplete")
    assert summary["gate"]["state"] == state
    shown = "\n".join(bench_module().render_summary("probe", summary))
    assert "zero_classes_false_alarms_within_limit: not applicable" in shown


def test_combine_leaves_not_applicable_items_out():
    """`not applicable` never blocks and never passes alone: with nothing else, nothing was judged."""
    assert RS.combine(["pass", "not applicable"]) == "pass"
    assert RS.combine(["incomplete", "not applicable"]) == "incomplete"
    assert RS.combine(["fail", "not applicable", "pass"]) == "fail"
    assert RS.combine(["not applicable"]) == "incomplete"
    assert RS.combine([]) == "incomplete"


# ---------------------------------------------------------------------------------------------------
# The engine gate: its required specs and the size-target timing


def passing(studies: int = 200) -> dict:
    """The summary fields `engine_gate` reads, for a spec whose own items all pass."""
    return {"studies": studies, "failed": [], "unconverged": [], "degenerate_gated": [], "gate": {"state": "pass"}}


def timing(**change) -> dict:
    """A size-target timing that meets every condition; `change` breaks one."""
    out = {"schema": RS.TIMING_SCHEMA, "spec": "size-target", "full_size": True, "started": "t", "exit_code": 0,
           "converged": True, "wall_seconds": 90.0, "cpu_seconds": 171.0, "cores_measured": 1.9, "threads": "1",
           "counts": {"runs": 500, "readings": 20000, "readings_used": 20000, "dimensions": 5, "judges": 4}}
    out.update(change)
    return out


ALL_SPECS = ("linear", "one-component", "binary-checks", "levels-3", "levels-5", "cheap-vs-strong-judge", "case-items",
             "mixed-rubric", "gated", "pairs", "overall-vs-criteria", "own-family", "reference-family", "halo",
             "panel-vs-per-component", "steps-swap", "case-items-uneven")


def test_the_gate_names_its_required_specs_apart_from_the_records():
    """The required lists are written in the module, and must match the catalog: every catalog spec, and
    the seven specs where a judge reads a binary or levels component more than once per output (worked
    out from the catalog). Written out here as well, so a change to either shows."""
    assert tuple(specs.names()) == RS.CATALOG_SPECS == ALL_SPECS
    assert tuple(RS.repeated_cut_specs()) == RS.REPEATED_SCALE_SPECS == (
        "binary-checks", "levels-3", "levels-5", "cheap-vs-strong-judge", "mixed-rubric", "gated",
        "panel-vs-per-component")
    assert RS.repeated_scale_kinds("binary-checks") == {"binary"} and RS.repeated_scale_kinds("levels-3") == {"levels"}


def test_four_passing_specs_and_a_clean_timing_are_not_a_passing_gate():
    """Four specs with 200 passing studies each and a clean 90-second size-target timing: the coverage item
    passes, but the catalog and repeated binary and levels items are incomplete and name what is missing
    (among them cheap-vs-strong-judge, gated and panel-vs-per-component, and every levels spec), so the
    gate is incomplete, never all_ok. With every spec present, the same gate passes."""
    four = {name: passing() for name in ("linear", "one-component", "own-family", "binary-checks")}
    gate = RS.engine_gate(four, [timing()])
    items = gate["items"]
    assert items["coverage_on_4_specs_at_200_studies"]["state"] == "pass"
    assert items["size_target_under_2_minutes"]["state"] == "pass"
    assert items["no_optimiser_failure_on_catalog"]["state"] == "incomplete"
    assert len(items["no_optimiser_failure_on_catalog"]["missing_or_short"]) == 13
    repeated = items["repeated_binary_levels_clean"]
    assert repeated["state"] == "incomplete" and repeated["kinds_missing"] == ["levels"]
    assert {"cheap-vs-strong-judge", "gated", "panel-vs-per-component", "levels-3", "levels-5"} <= set(
        repeated["missing_or_short"])
    assert gate["state"] == "incomplete" and gate["all_ok"] is False
    every = {name: passing() for name in ALL_SPECS}
    assert RS.engine_gate(every, [timing()])["all_ok"] is True
    short = dict(every, gated=passing(199))
    assert RS.engine_gate(short, [timing()])["items"]["repeated_binary_levels_clean"]["missing_or_short"] == ["gated"]


def test_a_narrower_gate_needs_a_note_and_both_scale_kinds():
    """A smaller catalog, fewer studies or fewer repeated specs is refused without a note saying who
    approved it; with a note it is shown as narrowed. Repeated specs that read no levels component (or no
    binary one) are refused even with a note."""
    with pytest.raises(ValueError, match="note"):
        RS.gate_scope(catalog=["linear", "binary-checks", "levels-3"], repeated=["binary-checks", "levels-3"])
    with pytest.raises(ValueError, match="note"):
        RS.gate_scope(studies=100)
    with pytest.raises(ValueError, match="note"):
        RS.gate_scope(catalog_studies=20)
    few = RS.gate_scope(catalog_studies=20, note="approved: 20 studies per spec for the clean-fit items")
    every = {name: passing(20) for name in ALL_SPECS}
    gate = RS.engine_gate(every, [timing()], few)
    assert gate["items"]["no_optimiser_failure_on_catalog"]["state"] == "pass"
    assert gate["items"]["coverage_on_4_specs_at_200_studies"]["state"] == "incomplete"   # 20 of 200 studies
    assert RS.engine_gate(every, [timing()])["items"]["no_optimiser_failure_on_catalog"]["state"] == "incomplete"
    with pytest.raises(ValueError, match="levels"):
        RS.gate_scope(repeated=["binary-checks"], note="approved for a pilot")
    with pytest.raises(ValueError, match="binary"):
        RS.gate_scope(repeated=["levels-3", "levels-5"], note="approved for a pilot")
    scope = RS.gate_scope(repeated=["binary-checks", "levels-3"], note="approved for a pilot")
    assert scope["narrowed"] and scope["repeated"] == ["binary-checks", "levels-3"] and not scope["omitted"]
    assert RS.gate_scope()["narrowed"] is False


@pytest.mark.parametrize("attempts,state", [
    ([], "incomplete"),
    ([timing()], "pass"),
    ([timing(converged=False)], "fail"),                                 # fast but not converged
    ([timing(), timing(exit_code=1, wall_seconds=300.0, converged=None, counts=None)], "fail"),   # a failed retry
    ([timing(wall_seconds=125.0)], "fail"),
    ([timing(cores_measured=3.4)], "fail"),
    ([timing(threads="4")], "fail"),
    ([timing(counts={"runs": 500, "readings": 15000, "readings_used": 15000, "dimensions": 5, "judges": 4})], "fail"),
    ([timing(schema="loopmath.eval-engine-bench-timing/1")], "fail"),
    ([timing(full_size=False, wall_seconds=5.0)], "incomplete"),       # a small copy is not the size target
])
def test_size_target_timing_counts_every_attempt(attempts, state):
    """The timing item passes only when every full-size attempt exited 0, converged, had the size target's
    counts, used at most 2 cores with one thread per process, and took under 120 s. A failed attempt
    beside a good one is shown and fails the item; no attempt leaves it incomplete."""
    got = RS.timing_item(attempts)
    assert got["state"] == state, got
    assert len(got["attempts"]) == sum(bool(t["full_size"]) for t in attempts)
    if state == "fail":
        assert got["failed_attempts"] >= 1 and any(a["problems"] for a in got["attempts"])
    every = {name: passing() for name in ALL_SPECS}
    assert RS.engine_gate(every, attempts)["all_ok"] is (state == "pass")


# ---------------------------------------------------------------------------------------------------
# Evidence checked before any statistic: named inventory, one record per study, finite numbers


def probe_records(name: str = "linear", studies: int = 200, extra=None) -> list[dict]:
    """Studies with one arm mean that holds its truth in 9 of 10 seeds (seed % 10 != 0), the family measured
    with no contrast; `extra(seed)` adds rows to a study."""
    def rows_of(seed):
        rows = [row("mu", "mu[a,d]", seed % 10 != 0)] + (extra(seed) if extra else [])
        return rows, []
    out = synthetic(studies, rows_of)
    for record in out:
        record.update(spec=name, engine="probe", options={"inputs": None})
    return out


def binomial_interval(k: int, n: int, tail: float = 0.025) -> tuple[float, float]:
    """The exact interval by bisection on binomial tail sums, written apart from the code under test."""
    def tail_at_least(p):
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))

    def tail_at_most(p):
        return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k + 1))

    def solve(f, rising):
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if (f(mid) < tail) == rising:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return solve(tail_at_least, True), solve(tail_at_most, False)


def test_a_parameter_gated_in_2_of_200_studies_leaves_the_spec_incomplete():
    """A tau gated in every study fills its class; a second tau is identified in only 2 of 200 studies and
    misses both times. The class covers 180 of 202 (in the band), but the rare parameter has 2 studies, so
    the inventory is incomplete and the spec does not pass. Without it, the same spec passes."""
    common = lambda seed: [row("tau", "tau[common,d]", seed % 10 != 0)]
    rare = lambda seed: common(seed) + ([row("tau", "tau[rare,d]", False)] if seed < 2 else [])
    control = RS.summarise(probe_records(extra=common))
    assert control["gate"]["state"] == "pass", control["gate"]
    got = RS.summarise(probe_records(extra=rare))
    assert got["classes"]["tau"]["rows"] == 202 and abs(got["classes"]["tau"]["coverage"] - 180 / 202) < 1e-12
    assert got["classes"]["tau"]["in_band"] is True
    assert got["parameters_short"] == ["tau[rare,d]"]
    item = got["gate"]["items"]["inventory_complete"]
    assert item["state"] == "incomplete" and "tau[rare,d]" in item["why"]
    assert got["gate"]["state"] == "incomplete" and not got["gate"]["pass"]


def test_a_record_that_breaks_its_declared_inventory_is_refused():
    """A record declaring four arm means, six taus and 320 run qualities, with one arm mean row and no
    aggregates, is refused before any statistic; so are the old count-only inventory, an extra row, a row
    of another class, a hit its own interval contradicts, and two rows of one parameter in one study."""
    def broken(change):
        records = probe_records(studies=1)
        change(records[0])
        with pytest.raises(RS.RecordError) as err:
            RS.summarise(records)
        return str(err.value)

    def declared(record):
        record["expected"] = ([{"class": "mu", "name": f"mu[{a},d]"} for a in "abcd"]
                              + [{"class": "tau", "name": f"tau[{j},d]"} for j in range(6)])
        record["expected_aggregates"] = {"q_run": 320}
    why = broken(declared)
    assert "9 expected parameters have no row" in why and "aggregates [], expected ['q_run']" in why
    assert "not a list of named parameters" in broken(lambda r: r.update(expected={"mu": 4, "tau": 6, "q_run": 320}))
    assert "not in `expected`" in broken(lambda r: r["rows"].append(row("mu", "mu[b,d]", True)))
    assert "has class tau" in broken(lambda r: r["rows"][0].update({"class": "tau"}))
    assert "its interval and truth say otherwise" in broken(lambda r: r["rows"][0].update(hit=not r["rows"][0]["hit"]))
    assert "two rows for mu[a,d]" in broken(lambda r: r["rows"].append(dict(r["rows"][0])))
    assert "aggregate q_run counts" in broken(lambda r: r.update(
        expected_aggregates={"q_run": 320}, aggregates={"q_run": {"n": 320, "hits": 400, "error_sum": 0.0, "err_n": 320,
                                                                  "error_sq_sum": 0.0, "sd_sum": 1.0, "degenerate": 0}}))


def test_identical_copies_count_once_and_differing_copies_are_refused():
    """Twenty seeds read ten times each are 20 studies, not 200: 18 of 20 hits, with the exact interval of
    n = 20, found here by bisection on binomial tails (0.6830172860 to 0.9876514728), and the spec is
    incomplete. A copy that differs from the first record of its study is refused. The same seed at
    another inputs setting is another study."""
    copies = [json.loads(json.dumps(r)) for r in probe_records(studies=20) for _ in range(10)]
    got = RS.summarise(copies)
    assert got["studies"] == 20 and got["copies_dropped"] == 180
    arm = got["arm_means"][0]
    assert (arm["hits"], arm["studies"]) == (18, 20)
    lo, hi = binomial_interval(18, 20)
    assert abs(lo - 0.6830172860) < 1e-9 and abs(hi - 0.9876514728) < 1e-9
    assert abs(arm["lo"] - lo) < 1e-8 and abs(arm["hi"] - hi) < 1e-8
    assert RS.parameter_table(copies)[0]["studies"] == 20 and RS.class_table(copies)["mu"]["studies"] == 20
    assert got["gate"]["items"]["enough_studies"]["state"] == "incomplete" and got["gate"]["state"] != "pass"
    differing = json.loads(json.dumps(copies))
    differing[57]["seconds"] = 2.0
    with pytest.raises(RS.RecordError, match="two different records of one study"):
        RS.summarise(differing)
    with pytest.raises(RS.RecordError, match="two different records of one study"):
        RS.parameter_table(differing)
    other = probe_records(studies=1)[0]
    unique, notes = RS.unique_studies([other, {**other, "options": {"inputs": 12}}])
    assert len(unique) == 2 and notes["copies_dropped"] == 0
    with pytest.raises(RS.RecordError, match="more than one engine, spec or inputs"):
        RS.summarise(unique)


def full_matrix(out: Path, copies: int, seeds: int) -> None:
    for name in RS.CATALOG_SPECS:
        for c in range(copies):
            for record in probe_records(name, seeds):
                RS.write_json_atomic(out / "records" / "probe" / f"{name}-iall" / f"copy{c}" / f"s{record['seed']}.json",
                                     record)
    RS.write_json_atomic(out / "timing" / "probe" / "size-target-iall-s0" / "timing-synthetic.json",
                         {**timing(), "inputs": None, "started": "synthetic"})


def test_stats_on_a_full_matrix_of_copies_counts_unique_studies(tmp_path):
    """The real CLI on all 17 catalog specs, each 20 seeds written in ten folders: 20 studies per spec, the
    3060 copies said, and the engine gate incomplete. A copy changed on disk stops `stats`, and so does a
    record with NaN."""
    out = tmp_path / "bench"
    full_matrix(out, copies=10, seeds=20)
    shown = bench("stats", "--out", out, "--engine", "probe")
    assert "3060 identical copies" in shown.stdout
    assert "probe linear-iall: 20 studies" in shown.stdout and "engine gate for probe: incomplete" in shown.stdout
    saved = json.loads((out / "stats-probe.json").read_text(encoding="utf-8"))
    assert {s["studies"] for s in saved["summaries"].values()} == {20} and saved["gate"]["all_ok"] is False
    assert saved["copies_dropped"] == 3060
    path = out / "records" / "probe" / "linear-iall" / "copy7" / "s3.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["rows"][0].update(hit=False, lo=0.6, hi=0.8)
    path.write_text(json.dumps(record), encoding="utf-8")
    refused = bench("stats", "--out", out, "--engine", "probe", check=False)
    assert refused.returncode != 0 and "two different records of one study" in refused.stderr
    assert "engine gate" not in refused.stdout
    record["rows"][0].update(hit=True, lo=0.3, hi=0.7, mean=float("nan"))
    path.write_text(json.dumps(record, allow_nan=True), encoding="utf-8")
    refused = bench("stats", "--out", out, "--engine", "probe", check=False)
    assert refused.returncode != 0 and "NaN is not a finite number" in refused.stderr


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("change", [
    {"wall_seconds": NAN}, {"wall_seconds": INF}, {"wall_seconds": -INF}, {"wall_seconds": 0.0},
    {"wall_seconds": None}, {"cores_measured": NAN}, {"cores_measured": INF}, {"cores_measured": None},
    {"cores_measured": -1.0}, {"cpu_seconds": NAN}, {"cpu_seconds": None}, {"wall_seconds": NAN, "cores_measured": NAN},
])
def test_a_timing_number_that_is_not_finite_fails(change):
    """NaN or infinity compares false with every limit, so a NaN wall time once passed. Every number of an
    attempt must be finite (wall above 0, CPU and cores at least 0), or the item and the engine gate fail."""
    every = {name: passing() for name in ALL_SPECS}
    assert RS.engine_gate(every, [timing()])["all_ok"] is True                 # the control
    got = RS.timing_item([timing(**change)])
    assert got["state"] == "fail" and got["attempts"][0]["problems"], got
    gate = RS.engine_gate(every, [timing(**change)])
    assert gate["state"] == "fail" and gate["all_ok"] is False


def test_nan_read_back_from_a_timing_file_fails_the_gate(tmp_path):
    """A timing file with NaN wall and cores (written by a JSON writer that allows NaN)
    is read, shown, and fails the item and the engine gate. A NaN anywhere in a spec summary fails it too."""
    folder = tmp_path / "timing" / "probe" / "size-target-iall-s0"
    folder.mkdir(parents=True)
    (folder / "timing-nan.json").write_text(json.dumps(timing(wall_seconds=NAN, cores_measured=NAN), allow_nan=True),
                                            encoding="utf-8")
    readback = bench_module().load_timings(tmp_path, "probe")
    assert len(readback) == 1 and math.isnan(readback[0]["wall_seconds"])
    assert RS.timing_item(readback)["state"] == "fail"
    every = {name: passing() for name in ALL_SPECS}
    gate = RS.engine_gate(every, readback)
    assert gate["state"] == "fail" and gate["all_ok"] is False
    every["linear"] = {**passing(), "seconds": {"median": NAN}}
    gate = RS.engine_gate(every, [timing()])
    assert gate["state"] == "fail" and "linear.seconds.median" in gate["nonfinite"]


def test_files_are_written_and_read_without_nan(tmp_path):
    """`write_json_atomic` refuses NaN and infinity and leaves no file; `finite_json` turns them into null
    for informational fields; reading refuses them; a record folder with a NaN record cannot be counted;
    a record with a NaN number is refused by the statistics."""
    path = tmp_path / "out" / "x.json"
    for bad in (NAN, INF, -INF):
        with pytest.raises(ValueError):
            RS.write_json_atomic(path, {"a": [1.0, bad]})
        assert list(path.parent.iterdir()) == []
    RS.write_json_atomic(path, RS.finite_json({"a": NAN, "b": [1.0, INF], "c": np.float64("nan"), "d": 2}))
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": None, "b": [1.0, None], "c": None, "d": 2}
    path.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a finite number"):
        RS.read_json_finite(path)
    record = probe_records(studies=1)[0]
    folder = tmp_path / "records" / "probe" / "linear-iall"
    folder.mkdir(parents=True)
    (folder / "s0.json").write_text(json.dumps({**record, "seconds": NAN}, allow_nan=True), encoding="utf-8")
    with pytest.raises(RS.RecordError, match="NaN is not a finite number"):
        RS.load_records(tmp_path / "records")
    for change in ({"cpu_seconds": NAN}, {"seconds": INF}):
        with pytest.raises(RS.RecordError, match="not a finite number"):
            RS.summarise([{**record, **change}])
    nan_row = {**record, "rows": [{**record["rows"][0], "mean": NAN}]}
    with pytest.raises(RS.RecordError, match="not finite"):
        RS.summarise([nan_row])


# ---------------------------------------------------------------------------------------------------
# Records of catalog designs, with draws of known law


def study(tmp: Path, name: str, seed: int = 0, inputs: int | None = None) -> tuple[M.Design, dict, Path]:
    folder = tmp / f"{name}-i{inputs}-s{seed}"
    if not (folder / simulate.TRUTH_FILE).exists():
        simulate.write(simulate.generate(specs.get(name, inputs=inputs), seed=seed), folder)
    design = M.build(C.read_study(folder), C.read_document(folder / "rubric.json", C.SCHEMA_RUBRIC),
                     C.read_document(folder / "judges.json", C.SCHEMA_JUDGES), allow_simulated=True)
    return design, simulate.read_truth(folder / simulate.TRUTH_FILE), folder


def in_engine_units(design: M.Design, truth: dict) -> dict:
    """The truth in the engine's terms, written from the units `engine_pymc` documents: read by position
    (simulate writes the fit's orders); family effects centred over the study's families where the
    family term is on, and each judge's mean family effect moved into its bias; then per dimension the
    biases sum to 0 with the weights `bias_w` (the anchors, or every judge with readings when the
    dimension is not anchored), the same amount going into the arm means and run qualities."""
    family_on = np.asarray(design.masks["family_on"], dtype=float)                    # J x D
    fam = np.nan_to_num(np.asarray(truth["family"], dtype=float))                      # J x F x D
    centre = fam.mean(axis=1) if fam.shape[1] else np.zeros(family_on.shape)
    target = {name: np.nan_to_num(np.asarray(truth[name], dtype=float))
              for name in ("mu", "sd_input", "sd_run", "corr_input", "corr_run", "q_run", "tau", "sigma", "halo_target",
                           "halo_call", "sd_judge_input", "sigma_pair", "position", "tie_width")}
    bias = np.nan_to_num(np.asarray(truth["bias"], dtype=float)) + centre
    target["family"] = (fam - centre[:, None, :]) * family_on[:, None, :]
    weights = np.asarray(design.data["bias_w"], dtype=float)
    shift = np.einsum("jd,jd->d", weights, bias)
    target["bias"] = bias - shift
    target["mu"] = target["mu"] + shift
    target["q_run"] = target["q_run"] + shift
    return target


def known_law(design: M.Design, truth: dict, rng: np.random.Generator, s: float) -> dict:
    """What a sampler returns, as draws of known law: per parameter, its value in the engine's terms
    (`in_engine_units`) plus one offset ~ N(0, s), plus independent N(0, s) noise per draw."""
    family_on = np.asarray(design.masks["family_on"], dtype=float)                    # J x D
    target = in_engine_units(design, truth)
    out = {}
    for name, value in target.items():
        offset = rng.normal(0.0, s, size=value.shape)
        noise = rng.normal(0.0, s, size=(FAKE_CHAINS, FAKE_DRAWS) + value.shape)
        x = value + offset + noise
        if name == "family":
            x = x * family_on[:, None, :]          # a family term that is off is 0 in the engine
        out[name] = x
    return {"draws": out, "divergences": 0, "run_seconds": 0.0, "version": "known-law"}


def known_law_fit(design: M.Design, truth: dict, rng: np.random.Generator, monkeypatch, seed: int = 0,
                  s: float = 0.02) -> tuple:
    fake = known_law(design, truth, rng, s)
    monkeypatch.setattr(engine_pymc, "sample", lambda data, **kw: fake)
    return FIT.fit_design(design, seed=seed, draws=FAKE_DRAWS, chains=FAKE_CHAINS, keep_draws=400, engine="pymc")


def known_law_records(tmp: Path, name: str, reps: int, monkeypatch, s: float = 0.02, family_every: int = 10,
                      seed: int = 0, inputs: int | None = None) -> tuple:
    design, truth, _ = study(tmp, name, seed, inputs)
    rng = np.random.default_rng(zlib.crc32(f"{name}-{inputs}-{seed}".encode()))
    records = []
    for rep in range(reps):
        meta, draws = known_law_fit(design, truth, rng, monkeypatch, rep, s)
        body = RS.study_record(design, meta, draws, truth, family=rep % family_every == 0)
        records.append({"schema": RS.RECORD_SCHEMA, "spec": name, "seed": rep, "engine": "known-law",
                        "options": {}, "converged": meta["diagnostics"]["converged"], "seconds": 0.0,
                        "cpu_seconds": 0.0, **body})
    return records, design, truth


def known_law_coverage(n: int) -> float:
    """The chance that the 5% to 95% points of n draws (numpy's linear interpolation) hold the truth, when
    the draws and the truth are independent from one normal law, as here (offset and noise of sd s).

    The truth is then like one more draw, so the chance is E[F(upper point)] - E[F(lower point)]. The
    k-th of n ordered draws has E[F] = k / (n + 1), and the points interpolate linearly between ranks
    1 + 0.05 (n - 1) and 1 + 0.95 (n - 1), so the chance is 0.9 (n - 1) / (n + 1): 0.8955 at 400 draws,
    0.8982 at 1000."""
    return 0.9 * (n - 1) / (n + 1)


KNOWN_LAW_SPECS = {"linear": {"mu", "arm_difference", "sd_input", "sd_run", "corr_input", "corr_run", "bias", "tau",
                              "sigma", "family", "q_run"},
                   "pairs": {"mu", "position", "tie_width", "sigma_pair"},
                   "halo": {"halo_target", "halo_call"},
                   "own-family": {"family"},
                   "binary-checks": {"mu", "tau", "sigma"}}


@pytest.mark.parametrize("name", sorted(KNOWN_LAW_SPECS))
def test_records_of_catalog_designs_cover_near_90_percent_under_a_known_law(name, tmp_path, monkeypatch):
    """Through the real `fit_design`, `finish` and `assemble`, with draws of known law in place of the
    sampler, over 200 repetitions: each class the spec should give is there, each gated class covers
    near its expected share (its cluster interval at a Bonferroni level over the classes holds it), no
    parameter is outside the band by its own interval, and no class is biased beyond 4 standard errors.
    A wrong index, unit or sign in the record would show here as coverage far from that share.

    The expected share is not 0.90 but `known_law_coverage(400)`, 0.8955: see there."""
    records, _, _ = known_law_records(tmp_path, name, 200, monkeypatch)
    summary = RS.summarise(records)
    classes = {k: v for k, v in summary["classes"].items() if v["gated"]}
    missing = KNOWN_LAW_SPECS[name] - set(classes)
    assert not missing, (missing, sorted(summary["classes"]))
    level, expected = 1.0 - 0.01 / len(classes), known_law_coverage(FAKE_DRAWS * FAKE_CHAINS)
    for klass in classes:
        hits, sizes = [], []
        for record in records:
            rows = [r for r in record["rows"] if r["class"] == klass]
            agg = record["aggregates"].get(klass)
            hits.append(agg["hits"] if agg else sum(r["hit"] for r in rows))
            sizes.append(agg["n"] if agg else len(rows))
        got = RS.cluster_share(hits, sizes, level)
        assert got["lo"] <= expected <= got["hi"], (klass, expected, got)
        c = classes[klass]
        if klass != "family":           # centred family effects sum to 0 in every study (see recovery_stats)
            assert abs(c["bias"]) <= 4.0 * c["bias_se"], (klass, c)
    assert not summary["parameters_outside"], summary["parameters_outside"]
    assert summary["family"]["studies_read"] == 20


def test_true_bias_and_family_effect_keep_the_judges_view(tmp_path, monkeypatch):
    """On `own-family` the record moves the judge's mean family effect into its true bias. Whatever it
    moves, a judge's view of an output of family f depends on bias + family effect on f, so that sum must
    be the simulated one for every model judge, family and dimension."""
    records, design, truth = known_law_records(tmp_path, "own-family", 1, monkeypatch)
    value = {r["name"]: r["truth"] for r in records[0]["rows"]}
    tj, tf = truth["index"]["judges"], truth["index"]["families"]
    dims = [d["id"] for d in truth["index"]["dimensions"]]
    seen = 0
    for j, judge in enumerate(tj):
        for d, dim in enumerate(dims):
            for f, family in enumerate(tf):
                key_b, key_f = f"bias[{judge},{dim}]", f"family[{judge},{family},{dim}]"
                if key_b in value and key_f in value:
                    simulated = truth["bias"][j, d] + truth["family"][j, f, d]
                    assert value[key_b] + value[key_f] == pytest.approx(simulated, abs=1e-12), (judge, family, dim)
                    seen += 1
    assert seen >= 6, seen
    real = [row for row in records[0]["family"] if row["truth"] != 0]
    assert {row["judge"] for row in real} == {"strong", "cheap"}, records[0]["family"]


def test_unanchored_study_is_compared_in_the_engine_location(tmp_path):
    """`one-component` with 1 input, seed 437: one anchor target, below the threshold, so the engine sets
    the mean bias of the three judges with readings to 0. Their simulated biases 0, 0.03 and 0.08 average
    0.0366666667, so in the engine's units the arm means are 0.42 + that = 0.4566666667 and 0.54 + that
    = 0.5766666667 (worked by hand), and each bias is 0.0366666667 lower."""
    design, truth, _ = study(tmp_path, "one-component", 437, 1)
    assert design.anchor["kind"] == "judges_mean"
    assert np.asarray(truth["bias"]).ravel().tolist() == pytest.approx([0.0, 0.03, 0.08])
    got = RS.engine_truth(design, truth)
    assert got["location"].tolist() == pytest.approx([0.0366666667], abs=1e-10)
    assert got["mu"].ravel().tolist() == pytest.approx([0.4566666667, 0.5766666667], abs=1e-10)
    assert got["bias"].ravel().tolist() == pytest.approx([-0.0366666667, -0.0066666667, 0.0433333333], abs=1e-10)


@pytest.mark.parametrize("name,seed,inputs", [("one-component", 437, 1), ("own-family", 0, None), ("pairs", 0, 8),
                                              ("halo", 0, 8), ("panel-vs-per-component", 0, 8)])
def test_truth_moves_keep_every_judges_view(name, seed, inputs, tmp_path):
    """Whatever `engine_truth` moves, what a judge sees of a run must stay: for every run, judge with
    readings, family where the family term is on, and dimension, q_run + bias + family is the simulated
    sum. And the units hold: the `bias_w` weighted biases are 0 per dimension, the family effects sum to
    0 over families, and arm differences are unchanged."""
    design, truth, _ = study(tmp_path, name, seed, inputs)
    got = RS.engine_truth(design, truth)
    q0, b0, f0 = (np.nan_to_num(np.asarray(truth[k], dtype=float)) for k in ("q_run", "bias", "family"))
    has = np.asarray(design.masks["has_score"], dtype=bool)
    on = np.asarray(design.masks["family_on"], dtype=bool)
    checked = 0
    for j, d in zip(*np.nonzero(has)):
        fams = range(f0.shape[1]) if on[j, d] else [None]
        for f in fams:
            before = q0[:, d] + b0[j, d] + (f0[j, f, d] if f is not None else f0[j, :, d].mean())
            after = got["q_run"][:, d] + got["bias"][j, d] + (got["family"][j, f, d] if f is not None else 0.0)
            assert after == pytest.approx(before, abs=1e-12), (j, d, f)
            checked += 1
    assert checked >= 3
    w = np.asarray(design.data["bias_w"], dtype=float)
    weighted = np.where(w > 0, w * got["bias"], 0.0)          # no truth (NaN) where a judge has no readings
    assert weighted.sum(axis=0) == pytest.approx(np.zeros(w.shape[1]), abs=1e-12)
    assert got["family"].sum(axis=1) == pytest.approx(np.zeros(on.shape), abs=1e-12)
    mu0 = np.asarray(truth["mu"], dtype=float)
    assert got["mu"] - got["mu"][0] == pytest.approx(mu0 - mu0[0], abs=1e-12)


def test_known_law_on_an_unanchored_design_covers_the_arm_means(tmp_path, monkeypatch):
    """The seed 437 design under draws of known law, 200 repetitions: in the engine's location the arm
    means and run qualities cover near the expected share and the arm means are unbiased. Compared in the
    simulator's location instead, the 0.0367 shift (about 1.3 sd of the law) would drop them far below."""
    records, _, _ = known_law_records(tmp_path, "one-component", 200, monkeypatch, family_every=1, seed=437, inputs=1)
    summary = RS.summarise(records)
    expected = known_law_coverage(FAKE_DRAWS * FAKE_CHAINS)
    for klass in ("mu", "arm_difference", "q_run"):
        c = summary["classes"][klass]
        assert c["lo"] <= expected <= c["hi"], (klass, c)
    assert summary["gate"]["items"]["arm_means_unbiased"]["state"] == "pass", summary["arm_means"]
    assert not summary["degenerate"]


def degenerate_case(tmp_path, monkeypatch, change) -> tuple[list, list, dict]:
    """The real `linear` design, one fit of known-law draws, then `change(draws)` before the record."""
    design, truth, _ = study(tmp_path, "linear")
    meta, draws = known_law_fit(design, truth, np.random.default_rng(5), monkeypatch)
    clean = RS.study_record(design, meta, draws, truth, family=False)
    draws = {k: np.array(v, dtype=float, copy=True) for k, v in draws.items()}
    change(draws)
    bad = RS.study_record(design, meta, draws, truth, family=False)
    return clean["rows"], bad["rows"], bad


def _tau(rows):
    return [r for r in rows if r["class"].split(":")[0] == "tau"]


@pytest.mark.parametrize("what", ["constant", "nonfinite", "missing"])
def test_bad_tau_draws_stay_as_rows_and_fail_the_spec(what, tmp_path, monkeypatch):
    """On `linear`, every tau array made constant 0, all NaN, or dropped: the tau rows are all still there,
    each marked degenerate and not covering its truth (the first is 0.03), and the spec fails its
    degenerate item. Nothing is read as fixed by the model."""
    def change(draws):
        if what == "constant":
            draws["tau"][...] = 0.0
        elif what == "nonfinite":
            draws["tau"][...] = np.nan
        else:
            del draws["tau"]

    clean, bad, record = degenerate_case(tmp_path, monkeypatch, change)
    assert len(_tau(clean)) >= 6 and not any(r.get("degenerate") for r in clean)
    assert [r["name"] for r in _tau(bad)] == [r["name"] for r in _tau(clean)]
    assert all(r["degenerate"] == what and r["hit"] is False for r in _tau(bad))
    assert len(record["degenerate"]) == len(_tau(clean))
    gated = [r for r in _tau(bad) if ":" not in r["class"]]
    summary = RS.summarise([{"schema": RS.RECORD_SCHEMA, "spec": "linear", "seed": 0, "engine": "known-law",
                             "options": {}, "converged": True, "seconds": 0.0, "cpu_seconds": 0.0, **record}])
    assert gated and summary["gate"]["items"]["no_degenerate_parameters"]["state"] == "fail"
    assert summary["gate"]["state"] == "fail"


def test_a_constant_arm_mean_of_999_is_a_degenerate_miss(tmp_path, monkeypatch):
    """One arm mean held at 999 against a truth near 0.4: its row stays, marked constant, and misses."""
    def change(draws):
        draws["mu"][:, 0, 0] = 999.0

    clean, bad, record = degenerate_case(tmp_path, monkeypatch, change)
    first = [r for r in bad if r["class"] == "mu"][0]
    assert first["degenerate"] == "constant" and first["hit"] is False and first["mean"] == 999.0
    assert len([r for r in bad if r["class"] == "mu"]) == len([r for r in clean if r["class"] == "mu"])


def test_parameters_fixed_by_the_units_are_left_out_by_the_design(tmp_path, monkeypatch):
    """The anchor's bias is 0 by the units (one per dimension on `linear`) and no pair parameter is used
    without pairs: both are left out from the design (`structural`, `unused`), whatever their draws, and
    are no failure."""
    clean, _, record = degenerate_case(tmp_path, monkeypatch, lambda draws: None)
    excluded = record["excluded"]
    assert sum(v.get("structural", 0) for k, v in excluded.items() if k.split(":")[0] == "bias") == 2, excluded
    assert sum(v.get("unused", 0) for k, v in excluded.items() if k.split(":")[0] == "sigma_pair") > 0, excluded
    assert not record["degenerate"] and not any(r.get("degenerate") for r in clean)


def test_size_target_is_the_study_of_check_14():
    """The size target here is the spec of `test_eval_recovery.size_spec`, and it gives 500 runs and
    20,000 readings over 5 components and 4 judges."""
    import test_eval_recovery

    assert RS.size_spec() == test_eval_recovery.size_spec()
    counts = simulate.generate(RS.size_spec(), seed=14).truth["counts"]
    assert (counts["runs"], counts["readings"], counts["dimensions"], counts["judges"]) == (500, 20000, 5, 4), counts


def test_repeated_binary_and_levels_specs():
    """The catalog specs the gate's last item reads: a judge reads a yes or no or levels component
    more than once per output."""
    got = RS.repeated_cut_specs()
    assert {"binary-checks", "levels-3", "levels-5", "mixed-rubric"} <= set(got)
    assert not {"linear", "one-component", "pairs", "own-family"} & set(got)


# ---------------------------------------------------------------------------------------------------
# Worker budgets


def bench_module():
    spec = importlib.util.spec_from_file_location("eval_engine_bench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workers_share_the_runs_cores_cpu_minutes_and_fits(tmp_path, monkeypatch):
    """`recover --workers 2 --cpu-minutes 20 --max-fits 2` on the default 2 cores starts two workers of one
    core per fit, 10 CPU minutes and one fit each (no process starts: Popen is replaced). Together they
    stay inside the totals. More workers than cores, or cores that pass the total, are refused; a worker
    whose share of fits is 0 is not started."""
    bench_py = bench_module()
    started = []

    class Fake:
        pid = 0

        def __init__(self, cmd, env=None):
            started.append(cmd)

        def wait(self):
            return 0

    monkeypatch.setattr(bench_py.subprocess, "Popen", Fake)
    base = ["recover", "--engine", "pymc", "--specs", "linear", "--out", str(tmp_path / "out"), "--studies", "4"]
    assert bench_py.main(base + ["--workers", "2", "--cpu-minutes", "20", "--max-fits", "2"]) == 0
    assert len(started) == 2
    for k, cmd in enumerate(started):
        value = {cmd[i]: cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i].startswith("--")}
        assert (value["--cores"], value["--total-cores"], value["--workers"]) == ("1", "1", "1")
        assert float(value["--cpu-minutes"]) == 10.0 and value["--max-fits"] == "1"
        assert value["--shard"] == f"{k}/2"
    started.clear()
    assert bench_py.main(base + ["--workers", "2", "--max-fits", "1"]) == 0
    assert len(started) == 1
    for bad in (["--workers", "3"], ["--workers", "2", "--cores", "2"], ["--total-cores", "1", "--workers", "2"]):
        with pytest.raises(SystemExit):
            bench_py.main(base + bad)


def test_time_refuses_more_chains_than_cores_when_eval_fit_has_no_cores_option(tmp_path):
    """`eval fit` runs its chains side by side; without a `--cores` option there, `time --chains 4` would use
    4 cores, so it is refused before any fit."""
    got = bench("time", "--engines", "pymc", "--specs", "one-component", "--inputs", 4, "--chains", 4,
                "--out", tmp_path / "out", check=False)
    help_text = subprocess.run([sys.executable, "-m", "loopmath", "eval", "fit", "--help"], capture_output=True,
                               text=True, env=dict(os.environ, LOOPMATH_NO_UPDATE_CHECK="1",
                                                   PYTHONPATH=str(REPO / "src"))).stdout
    if "--cores" in help_text:
        pytest.skip("eval fit takes --cores here, so the chains may exceed the cores")
    assert got.returncode != 0 and "--cores" in got.stderr, got.stderr[-1000:]
    assert not (tmp_path / "out" / "timing").exists()


# ---------------------------------------------------------------------------------------------------
# The harness with real fits


def bench(*argv: object, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, LOOPMATH_NO_UPDATE_CHECK="1", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(REPO / "src")] + ([os.environ["PYTHONPATH"]]
                                                                 if os.environ.get("PYTHONPATH") else [])))
    got = subprocess.run([sys.executable, str(SCRIPT), *[str(a) for a in argv]], capture_output=True, text=True,
                         env=env, cwd=REPO)
    if check:
        assert got.returncode == 0, (argv, got.stdout[-2000:], got.stderr[-3000:])
    return got


def need_engine(name: str) -> None:
    if not engine_pymc.available():
        pytest.skip(engine_pymc.INSTALL)
    if name != "pymc" and "engine" not in inspect.signature(FIT.fit_design).parameters:
        pytest.skip(f"fit.fit_design takes no engine argument in this checkout, so {name} cannot run")
    if name == "laplace" and importlib.util.find_spec("loopmath.eval.engine_laplace") is None:
        pytest.skip("loopmath.eval.engine_laplace is not in this checkout")


SMALL = ("--specs", "one-component", "--inputs", 12, "--draws", 200, "--keep-draws", 400)


def test_harness_with_pymc_on_two_small_studies(tmp_path):
    """`recover` writes one record per study, a second run keeps them and fits nothing, a run with other
    options is refused, `stats` reads them, and `time` times `eval fit` as its own process."""
    need_engine("pymc")
    out = tmp_path / "bench"
    bench("recover", "--engine", "pymc", "--studies", 2, "--out", out, *SMALL)
    paths = sorted((out / "records" / "pymc" / "one-component-i12").glob("s*.json"))
    assert [p.name for p in paths] == ["s0.json", "s1.json"]
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert not record.get("error") and record["rows"] and record["seconds"] > 0
        assert {"truth", "mean", "lo", "hi", "hit"} <= set(record["rows"][0])
        assert record["family"] is not None and "q_run" in record["aggregates"]
        assert record["schema"] == RS.RECORD_SCHEMA and sum(e["class"] == "mu" for e in record["expected"]) == 2
        assert {(r["class"], r["name"]) for r in record["rows"]} == {(e["class"], e["name"]) for e in record["expected"]}
        assert RS.record_problems(record) == []
    before = [p.stat().st_mtime_ns for p in paths]
    again = bench("recover", "--engine", "pymc", "--studies", 2, "--out", out, *SMALL)
    assert [p.stat().st_mtime_ns for p in paths] == before, again.stdout
    plan = bench("recover", "--engine", "pymc", "--studies", 3, "--out", out, "--dry-run", *SMALL)
    assert "3 studies planned, 2 kept, 1 to fit" in plan.stdout and "median" in plan.stdout
    other = bench("recover", "--engine", "pymc", "--studies", 2, "--out", out, "--specs", "one-component",
                  "--inputs", 12, "--draws", 300, "--keep-draws", 400, check=False)
    assert other.returncode != 0 and "never overwritten" in other.stderr
    shown = bench("stats", "--out", out, "--engine", "pymc")
    assert "pymc one-component-i12: 2 studies" in shown.stdout and "arm means" in shown.stdout
    assert "gate for this spec: " in shown.stdout and "gate for this spec: pass" not in shown.stdout   # 2 of 200
    assert "engine gate for pymc: incomplete" in shown.stdout
    saved = json.loads((out / "stats-pymc.json").read_text(encoding="utf-8"))
    assert saved["summaries"]["one-component-i12"]["studies"] == 2
    timed = bench("time", "--engines", "pymc", "--specs", "one-component", "--inputs", 10, "--draws", 200,
                  "--keep-draws", 400, "--out", out)
    line = json.loads(timed.stdout.strip().splitlines()[-1])
    assert line["exit_code"] == 0 and line["wall_seconds"] > 0 and line["engine_seconds"] > 0
    assert 0 < line["cores_measured"] <= 2.0 * (1 + RS.CORES_SLACK)


def test_harness_with_the_laplace_engine_beside_pymc(tmp_path):
    """Once `fit_design` takes an engine: the same two small studies with the Laplace engine, side by side
    with PyMC, and its size-target timing on a small copy."""
    need_engine("laplace")
    out = tmp_path / "bench"
    for engine in ("pymc", "laplace"):
        bench("recover", "--engine", engine, "--studies", 2, "--out", out, *SMALL)
    shown = bench("compare", "--out", out, "--engines", "pymc,laplace")
    assert "one-component-i12: 2 studies fitted by both" in shown.stdout
    timed = bench("time", "--engines", "laplace", "--specs", "size-target", "--inputs", 10, "--out", out)
    assert json.loads(timed.stdout.strip().splitlines()[-1])["exit_code"] == 0


# ---------------------------------------------------------------------------------------------------
# A finished large run


def test_gate_spec_items_of_a_finished_run():
    """With LOOPMATH_EVAL_BENCH_OUT naming the folder of a finished run (and LOOPMATH_EVAL_BENCH_ENGINE its
    engine, default laplace): every catalog-size spec there passes its part of the engine gate. The gate
    itself (4 specs, the size target, clean fits) is printed by `stats` and decided by people, not here."""
    folder = os.environ.get("LOOPMATH_EVAL_BENCH_OUT")
    if not folder:
        pytest.skip("set LOOPMATH_EVAL_BENCH_OUT to the folder of a finished run")
    engine = os.environ.get("LOOPMATH_EVAL_BENCH_ENGINE", "laplace")
    records = RS.load_records(Path(folder) / "records" / engine)
    full = [r for r in records if r["options"].get("inputs") is None]
    assert full, f"no catalog-size records for {engine} in {folder}"
    by_spec: dict[str, list] = {}
    for record in full:
        by_spec.setdefault(record["spec"], []).append(record)
    failed = {name: RS.summarise(group)["gate"] for name, group in by_spec.items()}
    failed = {name: gate for name, gate in failed.items() if not gate["pass"]}
    assert not failed, failed
