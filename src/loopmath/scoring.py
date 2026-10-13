"""Transfer-test metric library (SPEC section 7).

Every function here takes arrays or DataFrames of held-out data and a model's
guesses about that data, and returns a dict of plain numbers. Nothing in this
module judges whether a number is good, acceptable, or good enough: there is
no pass/fail, no verdict field, and no comparison to a cutoff anywhere. Each
function's dict carries an "n" (how many held-out points it scored) and a
"note" (one plain-language sentence saying what the number means).

What each metric can tell you, and what it cannot:

- `interval_coverage`: whether the model's stated bands actually contain the
  held-out truth, and how wide those bands are. It says nothing about whether
  a band is precise (a band could cover everything by being enormous).
- `mlpd`: how well the model's whole spread of guesses matches where the
  truth actually landed, point by point. It rewards a spread that is both
  well-placed and appropriately sized, but the number alone is only
  meaningful relative to another mlpd computed the same way (a baseline, or
  another model), not in isolation.
- `baseline_pooled_mean` / `baseline_nearest_neighbor`: two deliberately
  simple ways of guessing, for comparison. They say nothing about whether the
  model being tested is doing anything clever, only what a simple guess
  would have scored.
- `kendall_tau`: whether the model gets the relative order right (this one
  costs more than that one), independent of the actual dollar amounts.
- `within_factor`: how often the model's guess and the truth are within a
  given multiple of each other. It is blind to systematic bias in one
  direction if that bias never crosses the factor line.
- `decision_regret`: what actually happens if someone acts on the model's
  cheapest-looking option, versus what the truly cheapest option would have
  cost. It only measures the choices actually presented in the data.
- `acceptance_calibration`: whether stated acceptance chances line up with
  how often things were actually accepted, grouped into equal-count bins. It
  cannot tell you anything about a single prediction, only about groups.

Every function is pure (arrays/DataFrames in, a dict of numbers out), handles
empty or degenerate input without raising, and returns `float("nan")` for any
statistic that is undefined rather than raising or guessing. Every function
also reports what it excluded from its calculation and why (rows dropped for
being non-finite, points with no usable draws, groups with a missing key,
outcomes or probabilities outside their valid range, and so on): the count
lives in the result dict, `summarize` prints it whenever it is non-zero, and
nothing here judges whether the resulting number is good, acceptable, or
good enough.
"""

from __future__ import annotations

import math
import sys
import warnings
from types import ModuleType

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm

from . import scoring_comparisons as _comparisons
from . import scoring_predictive as _predictive
from .scoring_comparisons import (
    _drop_nonfinite,
    _value_based_bin_edges,
    acceptance_calibration,
    decision_regret,
    kendall_tau,
    within_factor,
)
from .scoring_predictive import (
    _DENSITY_FLOOR,
    _SCALE_EPS,
    _SQRT_2PI,
    _silverman_bandwidth,
    baseline_nearest_neighbor,
    baseline_pooled_mean,
    interval_coverage,
    mlpd,
)


_SUPPORT_REBINDINGS = {
    "_DENSITY_FLOOR": (_predictive,),
    "_SCALE_EPS": (_predictive,),
    "_SQRT_2PI": (_predictive,),
    "_drop_nonfinite": (_comparisons, _predictive),
    "_silverman_bandwidth": (_predictive,),
    "_value_based_bin_edges": (_comparisons,),
    "kendalltau": (_comparisons,),
    "math": (_comparisons, _predictive),
    "norm": (_predictive,),
    "np": (_comparisons, _predictive),
    "pd": (_comparisons, _predictive),
    "warnings": (_comparisons,),
}


class _ScoringModule(ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _SUPPORT_REBINDINGS.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _ScoringModule


_SUMMARIZE_METRICS = [
    "interval_coverage",
    "mlpd",
    "baseline_pooled_mean",
    "baseline_nearest_neighbor",
    "kendall_tau",
    "within_factor",
    "decision_regret",
    "acceptance_calibration",
]

_SUMMARIZE_LABEL = {
    "interval_coverage": "interval coverage",
    "mlpd": "mean log predictive density",
    "baseline_pooled_mean": "pooled-mean baseline",
    "baseline_nearest_neighbor": "nearest-neighbor baseline",
    "kendall_tau": "rank agreement",
    "within_factor": "within-factor share",
    "decision_regret": "decision regret",
    "acceptance_calibration": "acceptance calibration",
}

_SUMMARIZE_UNAVAILABLE_REASON = {
    "interval_coverage": "no held-out points were available",
    "mlpd": "no held-out points had usable draws",
    "baseline_pooled_mean": "not enough data to compute this baseline",
    "baseline_nearest_neighbor": "not enough data to compute this baseline",
    "kendall_tau": "fewer than two held-out points",
    "within_factor": "no held-out points were available",
    "decision_regret": "no groups had enough data to compare choices",
    "acceptance_calibration": "no usable held-out outcomes were available",
}


def _summarize_available(res: dict | None, headline_key: str) -> bool:
    """Whether `res` has a usable (non-NaN) value for `headline_key`, under
    the count field that metric uses ("n_groups" for decision_regret, "n"
    for everything else)."""
    if not res:
        return False
    n_key = "n_groups" if "n_groups" in res else "n"
    if res.get(n_key, 0) <= 0:
        return False
    return bool(np.isfinite(res.get(headline_key, float("nan"))))


def _summarize_unavailable(name: str) -> str:
    return f"{_SUMMARIZE_LABEL[name]}: not available ({_SUMMARIZE_UNAVAILABLE_REASON[name]})"


def _summarize_counts_suffix(res: dict, keys_phrases: list[tuple[str, str]]) -> str:
    parts = [f"{res[k]} {phrase}" for k, phrase in keys_phrases if res.get(k)]
    return f" ({', '.join(parts)})" if parts else ""


def summarize(results: dict, expected: list[str] | None = None) -> list[str]:
    """Render a dict of the above results as plain user-facing lines.

    `results` is keyed by metric name: "interval_coverage", "mlpd",
    "baseline_pooled_mean", "baseline_nearest_neighbor", "kendall_tau",
    "within_factor", "decision_regret", "acceptance_calibration".

    `expected` names which of those metrics should be reported; it defaults
    to all of them. A metric in `expected` that is missing from `results`,
    or whose headline value is undefined (NaN, or nothing was scored), gets
    an explicit "not available" line naming why, rather than being silently
    left out. A metric whose result carries a non-zero exclusion count
    (points dropped as non-finite, points with no usable draws, training
    points dropped, rows excluded before grouping, points at the density
    floor, points that fell back to a plain average, and so on) has that
    count appended to its line, so an exclusion is never left unmentioned.

    Descriptive only: never a verdict, never a threshold comparison. A line
    that reports one number is higher, lower, or equal to another is a plain
    statement of the two numbers, not a judgment of either one.
    """
    if expected is None:
        expected = list(_SUMMARIZE_METRICS)
    expected_set = set(expected)
    lines: list[str] = []

    if "interval_coverage" in expected_set:
        ic = results.get("interval_coverage")
        if _summarize_available(ic, "coverage"):
            lines.append(
                f"interval coverage: {ic['coverage'] * 100:.0f}% of {ic['n']} held-out points fell "
                f"inside their stated band (median band width {ic['median_width']:.2f})"
                + _summarize_counts_suffix(ic, [("n_dropped", "dropped as non-finite")])
            )
        else:
            lines.append(_summarize_unavailable("interval_coverage"))

    model_mlpd = results.get("mlpd")
    model_ok = _summarize_available(model_mlpd, "mlpd")
    if "mlpd" in expected_set:
        if model_ok:
            lines.append(
                f"mean log predictive density: {model_mlpd['mlpd']:.2f} across {model_mlpd['n']} points"
                + _summarize_counts_suffix(
                    model_mlpd,
                    [
                        ("n_dropped", "dropped as non-finite"),
                        ("n_no_draws", "excluded for having no usable draws"),
                        ("n_floored", "at the density floor"),
                    ],
                )
            )
        else:
            lines.append(_summarize_unavailable("mlpd"))

    for base_key, base_label in (
        ("baseline_pooled_mean", "pooled-mean"),
        ("baseline_nearest_neighbor", "nearest-neighbor"),
    ):
        if base_key not in expected_set:
            continue
        base = results.get(base_key)
        if not _summarize_available(base, "mlpd"):
            lines.append(_summarize_unavailable(base_key))
            continue
        base_counts = _summarize_counts_suffix(
            base,
            [
                ("n_dropped", "dropped as non-finite"),
                ("n_train_dropped", "training points dropped as non-finite"),
                ("n_floored", "at the density floor"),
                ("n_fallback", "matched nothing and fell back to the pooled mean"),
            ],
        )
        if model_ok:
            diff = model_mlpd["mlpd"] - base["mlpd"]
            diff_str = "0.00 (equal)" if diff == 0.0 else f"{diff:+.2f} (higher is a closer fit)"
            lines.append(
                f"mean log predictive density: estimate {model_mlpd['mlpd']:.2f}, "
                f"{base_label} baseline {base['mlpd']:.2f}, difference {diff_str}" + base_counts
            )
        else:
            lines.append(
                f"{base_label} baseline: mean log predictive density {base['mlpd']:.2f} across "
                f"{base['n']} points" + base_counts
            )

    if "kendall_tau" in expected_set:
        kt = results.get("kendall_tau")
        if _summarize_available(kt, "tau"):
            lines.append(
                f"rank agreement between estimated and actual cost across {kt['n']} points: "
                f"tau {kt['tau']:.2f}"
                + _summarize_counts_suffix(kt, [("n_dropped", "dropped as non-finite")])
            )
        else:
            lines.append(_summarize_unavailable("kendall_tau"))

    if "within_factor" in expected_set:
        wf = results.get("within_factor")
        if _summarize_available(wf, "share"):
            lines.append(
                f"{wf['share'] * 100:.0f}% of {wf['n']} held-out points landed within "
                f"{wf['factor']:g}x of the true cost"
                + _summarize_counts_suffix(wf, [("n_dropped", "dropped as non-finite")])
            )
        else:
            lines.append(_summarize_unavailable("within_factor"))

    if "decision_regret" in expected_set:
        dr = results.get("decision_regret")
        if dr and dr.get("n_groups", 0) > 0 and np.isfinite(dr.get("mean_regret", float("nan"))):
            per_group = dr.get("per_group") or []
            configs_bit = ""
            if per_group and "n_configs" in per_group[0]:
                median_configs = float(np.median([g["n_configs"] for g in per_group]))
                configs_bit = f"; a median of {median_configs:g} workflow configurations were available per group"
            lines.append(
                f"choosing the workflow configuration the estimate said was cheapest cost "
                f"{dr['mean_regret']:.2f} more on average than the true cheapest configuration "
                f"across {dr['n_groups']} groups "
                f"({dr['n_optimal']} of {dr['n_groups']} picked the true cheapest configuration)"
                f"{configs_bit}"
                + _summarize_counts_suffix(dr, [("n_excluded_rows", "rows excluded before grouping")])
            )
        else:
            lines.append(_summarize_unavailable("decision_regret"))

    if "acceptance_calibration" in expected_set:
        ac = results.get("acceptance_calibration")
        if _summarize_available(ac, "ece"):
            lines.append(
                f"stated acceptance chances were off by {ac['ece']:.3f} on average across bins "
                f"(Brier score {ac['brier']:.3f})"
                + _summarize_counts_suffix(
                    ac, [("n_excluded", "excluded as invalid outcomes or probabilities")]
                )
            )
        else:
            lines.append(_summarize_unavailable("acceptance_calibration"))

    return lines
