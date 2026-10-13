"""Predictive-density and baseline scoring metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from .scoring_comparisons import _drop_nonfinite

# mlpd / baseline predictive-density floor (SPEC section 7 implementation
# note): a point far outside the draws would otherwise score log(0) = -inf.
# Flooring the density at this value turns that into a large, finite,
# negative number instead, and n_floored/n_dropped keys say how often it fired.
_DENSITY_FLOOR = 1e-12

# Bandwidth/scale floor: avoids a division by zero when a set of draws, or a
# matched training group, has exactly zero spread. It does not change the
# reported spread ("sd", "bandwidth"), only the number actually used inside
# the density calculation.
_SCALE_EPS = 1e-12

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _silverman_bandwidth(draws_i: np.ndarray) -> float:
    """Silverman's rule-of-thumb kernel width for one point's draws.

    Returns 0.0 (the caller substitutes a tiny epsilon) when there are fewer
    than two finite draws, or the draws have zero spread.
    """
    draws_i = draws_i[np.isfinite(draws_i)]
    n = draws_i.size
    if n < 2:
        return 0.0
    sigma = float(np.std(draws_i, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    return 1.06 * sigma * n ** (-1.0 / 5.0)


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Share of held-out points inside their stated band.

    -> {"n", "coverage", "n_inside", "mean_width", "median_width", "note"}
    """
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    (y, lo, hi), n_dropped = _drop_nonfinite(y, lo, hi)
    n = int(y.size)

    out: dict = {"n": n}
    if n_dropped:
        out["n_dropped"] = n_dropped
    if n == 0:
        out.update(
            {
                "coverage": float("nan"),
                "n_inside": 0,
                "mean_width": float("nan"),
                "median_width": float("nan"),
                "note": "No held-out points were available to check, so there is nothing to report here.",
            }
        )
        return out

    inside = (y >= lo) & (y <= hi)
    n_inside = int(inside.sum())
    coverage = n_inside / n
    widths = hi - lo
    median_width = float(np.median(widths))
    out.update(
        {
            "coverage": coverage,
            "n_inside": n_inside,
            "mean_width": float(np.mean(widths)),
            "median_width": median_width,
            "note": (
                f"{coverage * 100:.0f}% of the {n} held-out points fell inside the band the model "
                f"stated for them (typical band width {median_width:.2f})."
            ),
        }
    )
    return out


def mlpd(y: np.ndarray, draws: np.ndarray, bandwidth: float | None = None) -> dict:
    """Mean log predictive density of the held-out points under the predictive draws.

    `draws` is shape (n_draws, n_points), where n_points must equal len(y)
    exactly as passed in. That shape is checked first, before any filtering,
    against the original held-out count: a mismatch is reported through
    "reason" and nothing is indexed or scored, rather than risking a
    misaligned column. Each point's density is a Gaussian kernel average
    over its own draws (Silverman bandwidth when `bandwidth` is None),
    floored at 1e-12 before taking the log so a point far outside every draw
    gives a large negative number instead of -inf. A point with no finite
    draws at all cannot be scored and is excluded from "n" (see
    "n_no_draws") rather than being floored; a point that had usable draws
    but landed at the floor is a different, separately counted condition
    ("n_floored").

    -> {"n", "n_input", "mlpd", "n_floored", "n_no_draws", "bandwidth", "note"}
    """
    y = np.asarray(y, dtype=float)
    draws = np.asarray(draws, dtype=float)
    n_input = int(y.size)

    if draws.ndim == 1:
        draws = draws.reshape(-1, 1) if y.size == 1 else draws.reshape(1, -1)

    if draws.ndim != 2 or draws.shape[1] != n_input:
        n_cols = int(draws.shape[1]) if draws.ndim == 2 else int(draws.size)
        return {
            "n": 0,
            "n_input": n_input,
            "mlpd": float("nan"),
            "n_floored": 0,
            "n_no_draws": 0,
            "bandwidth": float("nan"),
            "reason": f"draws have {n_cols} columns for {n_input} held-out points",
            "note": (
                "The draws matrix did not have one column per held-out point, so "
                "nothing was scored."
            ),
        }

    finite_y = np.isfinite(y)
    n_dropped = int((~finite_y).sum())
    y = y[finite_y]
    draws = draws[:, finite_y]

    finite_draws = np.isfinite(draws)
    has_draws = finite_draws.any(axis=0) if draws.size else np.zeros(draws.shape[1], dtype=bool)
    n_no_draws = int((~has_draws).sum())
    y = y[has_draws]
    draws = draws[:, has_draws]
    n = int(y.size)

    out: dict = {"n": n, "n_input": n_input, "n_no_draws": n_no_draws}
    if n_dropped:
        out["n_dropped"] = n_dropped
    if n == 0:
        out.update(
            {
                "mlpd": float("nan"),
                "n_floored": 0,
                "bandwidth": float("nan"),
                "note": "No held-out points had usable draws to score, so there is nothing to report here.",
            }
        )
        return out

    log_dens = np.empty(n)
    bws_used = np.empty(n)
    n_floored = 0
    for i in range(n):
        draws_i = draws[:, i]
        draws_i = draws_i[np.isfinite(draws_i)]
        bw = bandwidth if bandwidth is not None else _silverman_bandwidth(draws_i)
        if not np.isfinite(bw) or bw <= 0:
            bw = _SCALE_EPS
        bws_used[i] = bw
        u = (y[i] - draws_i) / bw
        density = float(np.mean(np.exp(-0.5 * u * u) / (bw * _SQRT_2PI)))
        floored = max(density, _DENSITY_FLOOR)
        if density <= _DENSITY_FLOOR:
            n_floored += 1
        log_dens[i] = math.log(floored)

    out.update(
        {
            "mlpd": float(np.mean(log_dens)),
            "n_floored": n_floored,
            "bandwidth": float(bandwidth) if bandwidth is not None else float(np.mean(bws_used)),
            "note": (
                "Average, per point, of how plausible the actual value looks under the model's "
                "spread of guesses; a higher number means the truth tended to land in a denser part "
                f"of the guesses. {n_floored} of {n} points scored landed at the density floor "
                "instead of an undefined value"
                + (f", and {n_no_draws} points had no usable draws and were excluded" if n_no_draws else "")
                + "."
            ),
        }
    )
    return out


def baseline_pooled_mean(train_y: np.ndarray, test_y: np.ndarray) -> dict:
    """Baseline B1': predict every held-out point by the pooled training mean,
    with the training spread as the predictive scale (a Normal predictive).

    -> {"n", "mlpd", "rmse", "mean", "sd", "n_train_input", "n_train_dropped",
        "n_floored", "note"}
    """
    train_y_all = np.asarray(train_y, dtype=float)
    n_train_input = int(train_y_all.size)
    finite_train = np.isfinite(train_y_all)
    n_train_dropped = int((~finite_train).sum())
    train_y = train_y_all[finite_train]

    test_y = np.asarray(test_y, dtype=float)
    finite = np.isfinite(test_y)
    n_dropped = int((~finite).sum())
    test_y = test_y[finite]
    n = int(test_y.size)

    out: dict = {
        "n": n,
        "n_train_input": n_train_input,
        "n_train_dropped": n_train_dropped,
    }
    if n_dropped:
        out["n_dropped"] = n_dropped
    if n == 0 or train_y.size == 0:
        out.update(
            {
                "mlpd": float("nan"),
                "rmse": float("nan"),
                "mean": float(np.mean(train_y)) if train_y.size else float("nan"),
                "sd": float("nan"),
                "n_floored": 0,
                "note": "Not enough data to compute this baseline.",
            }
        )
        return out

    mean = float(np.mean(train_y))
    sd = float(np.std(train_y, ddof=1)) if train_y.size > 1 else 0.0
    sd_eff = sd if sd > 0 else _SCALE_EPS
    dens = norm.pdf(test_y, loc=mean, scale=sd_eff)
    n_floored = int((dens <= _DENSITY_FLOOR).sum())
    log_dens = np.log(np.maximum(dens, _DENSITY_FLOOR))
    rmse = float(np.sqrt(np.mean((test_y - mean) ** 2)))

    out.update(
        {
            "mlpd": float(np.mean(log_dens)),
            "rmse": rmse,
            "mean": mean,
            "sd": sd,
            "n_floored": n_floored,
            "note": (
                f"Simplest comparison point: guess the training average ({mean:.2f}) for every "
                f"held-out point, using the training spread ({sd:.2f}) as the guessed range; its "
                f"typical miss size was {rmse:.2f}."
                + (
                    f" {n_floored} of {n} points were far enough from this guess to be capped at "
                    "a very low density instead of an undefined one."
                    if n_floored
                    else ""
                )
            ),
        }
    )
    return out


def baseline_nearest_neighbor(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_col: str, feature_cols: list[str]
) -> dict:
    """Baseline B2': predict each held-out point by the mean of the training rows
    that match on the most feature columns (Hamming distance over the categorical
    feature columns, ties averaged). Falls back to the pooled mean when nothing
    matches on anything, and counts how often that happened.

    UNIT CONTRACT: this function is space-agnostic -- it evaluates a Normal
    density directly on whatever `y_col` holds, in whatever units the caller
    passes. A log density is not invariant under a change of units, so
    `y_col` MUST already be in the same transformed space as any other MLPD
    this baseline's `mlpd` will be compared against (this module's own
    `mlpd` and `baseline_pooled_mean` both operate on log10(usd) when that is
    the model's likelihood scale). Pass a `log10(usd)`-valued column here,
    never raw `usd`, whenever the comparison is to a model fit on
    `log10(usd)` -- see `loopmath.fit.transfer_test` for the call site that
    enforces this.

    -> {"n", "mlpd", "rmse", "n_fallback", "mean_matched", "n_train_input",
        "n_train_dropped", "n_floored", "note"}
    """
    feature_cols = list(feature_cols)

    train_y_all = pd.to_numeric(train_df[y_col], errors="coerce").to_numpy(dtype=float)
    n_train_input = int(train_y_all.size)
    train_mask = np.isfinite(train_y_all)
    n_train_dropped = int((~train_mask).sum())
    train_y = train_y_all[train_mask]
    train_feats = train_df.loc[train_mask, feature_cols].reset_index(drop=True) if feature_cols else train_df.loc[train_mask, []].reset_index(drop=True)

    test_y_all = pd.to_numeric(test_df[y_col], errors="coerce").to_numpy(dtype=float)
    test_mask = np.isfinite(test_y_all)
    n_dropped = int((~test_mask).sum())
    test_y = test_y_all[test_mask]
    test_feats = test_df.loc[test_mask, feature_cols].reset_index(drop=True) if feature_cols else test_df.loc[test_mask, []].reset_index(drop=True)
    n = int(test_y.size)

    out: dict = {
        "n": n,
        "n_train_input": n_train_input,
        "n_train_dropped": n_train_dropped,
    }
    if n_dropped:
        out["n_dropped"] = n_dropped
    if n == 0 or train_y.size == 0:
        out.update(
            {
                "mlpd": float("nan"),
                "rmse": float("nan"),
                "n_fallback": 0,
                "mean_matched": float("nan"),
                "n_floored": 0,
                "note": "Not enough data to compute this baseline.",
            }
        )
        return out

    n_train = int(train_y.size)
    match_counts = np.zeros((n, n_train))
    for col in feature_cols:
        t_vals = test_feats[col].to_numpy()
        r_vals = train_feats[col].to_numpy()
        match_counts += t_vals[:, None] == r_vals[None, :]

    global_mean = float(np.mean(train_y))
    global_sd = float(np.std(train_y, ddof=1)) if n_train > 1 else 0.0

    y_pred = np.empty(n)
    scale = np.empty(n)
    best_matches = np.empty(n)
    n_fallback = 0
    for i in range(n):
        row_counts = match_counts[i]
        best = float(row_counts.max()) if n_train else 0.0
        best_matches[i] = best
        if best <= 0:
            y_pred[i] = global_mean
            scale[i] = global_sd
            n_fallback += 1
        else:
            grp = train_y[row_counts == best]
            y_pred[i] = float(grp.mean())
            if grp.size > 1:
                s = float(np.std(grp, ddof=1))
                scale[i] = s if s > 0 else global_sd
            else:
                scale[i] = global_sd

    scale_eff = np.where(scale > 0, scale, _SCALE_EPS)
    dens = norm.pdf(test_y, loc=y_pred, scale=scale_eff)
    n_floored = int((dens <= _DENSITY_FLOOR).sum())
    log_dens = np.log(np.maximum(dens, _DENSITY_FLOOR))
    rmse = float(np.sqrt(np.mean((test_y - y_pred) ** 2)))

    out.update(
        {
            "mlpd": float(np.mean(log_dens)),
            "rmse": rmse,
            "n_fallback": n_fallback,
            "mean_matched": float(np.mean(best_matches)),
            "n_floored": n_floored,
            "note": (
                "Comparison point that guesses each held-out point from the training rows sharing "
                f"the most features with it; its typical miss size was {rmse:.2f}, and {n_fallback} "
                f"of {n} points matched nothing so fell back to the plain training average."
                + (
                    f" {n_floored} of {n} points were far enough from their guess to be capped at "
                    "a very low density instead of an undefined one."
                    if n_floored
                    else ""
                )
            ),
        }
    )
    return out
