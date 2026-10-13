"""Ranking, decision, and calibration scoring metrics."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def _drop_nonfinite(*arrays: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Row-wise finite mask across equal-length 1-D arrays.

    Returns the filtered arrays (same order) and how many rows were dropped
    for having a non-finite value in at least one of them.
    """
    stacked = np.vstack([np.asarray(a, dtype=float) for a in arrays])
    mask = np.all(np.isfinite(stacked), axis=0)
    n_dropped = int((~mask).sum())
    filtered = [np.asarray(a, dtype=float)[mask] for a in arrays]
    return filtered, n_dropped


def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Rank agreement between what actually cost more and what was estimated to.

    Uses scipy.stats.kendalltau (tau-b).

    -> {"n", "tau", "p_value", "n_concordant_pairs", "note"}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    (y_true, y_pred), n_dropped = _drop_nonfinite(y_true, y_pred)
    n = int(y_true.size)

    out: dict = {"n": n}
    if n_dropped:
        out["n_dropped"] = n_dropped
    if n < 2:
        out.update(
            {
                "tau": float("nan"),
                "p_value": float("nan"),
                "n_concordant_pairs": 0,
                "note": "Not enough held-out points to compare any pairs.",
            }
        )
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tau, p_value = kendalltau(y_true, y_pred)

    dt = y_true[:, None] - y_true[None, :]
    dp = y_pred[:, None] - y_pred[None, :]
    iu = np.triu_indices(n, k=1)
    n_concordant = int(np.sum((dt[iu] * dp[iu]) > 0))

    out.update(
        {
            "tau": float(tau) if np.isfinite(tau) else float("nan"),
            "p_value": float(p_value) if np.isfinite(p_value) else float("nan"),
            "n_concordant_pairs": n_concordant,
            "note": (
                "How often the estimate and the actual cost agree on which of two held-out points "
                "cost more: 1 means they always agree on the order, -1 means they always disagree, "
                "0 means no relationship."
            ),
        }
    )
    return out


def within_factor(
    y_true: np.ndarray, y_pred: np.ndarray, factor: float = 2.0, log10_space: bool = True
) -> dict:
    """Share of held-out points whose estimate is within `factor` of the truth.

    When `log10_space` is True the inputs are log10 dollars, so the test is
    |y_true - y_pred| <= log10(factor). Otherwise it is a ratio test on
    positive values (non-positive values cannot form a ratio and are dropped
    like any other unusable pair).

    -> {"n", "within", "share", "factor", "median_abs_error", "note"}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    (y_true, y_pred), n_dropped = _drop_nonfinite(y_true, y_pred)

    if not log10_space:
        positive = (y_true > 0) & (y_pred > 0)
        n_dropped += int((~positive).sum())
        y_true = y_true[positive]
        y_pred = y_pred[positive]

    n = int(y_true.size)
    out: dict = {"n": n}
    if n_dropped:
        out["n_dropped"] = n_dropped
    if n == 0:
        out.update(
            {
                "within": 0,
                "share": float("nan"),
                "factor": factor,
                "median_abs_error": float("nan"),
                "note": "No held-out points were available to check.",
            }
        )
        return out

    if log10_space:
        abs_err = np.abs(y_true - y_pred)
        within_mask = abs_err <= math.log10(factor)
        unit = "in log10 dollars"
    else:
        ratio = np.maximum(y_true, y_pred) / np.minimum(y_true, y_pred)
        within_mask = ratio <= factor
        abs_err = np.abs(y_true - y_pred)
        unit = "in dollars"

    n_within = int(within_mask.sum())
    share = n_within / n
    median_abs_error = float(np.median(abs_err))
    out.update(
        {
            "within": n_within,
            "share": share,
            "factor": factor,
            "median_abs_error": median_abs_error,
            "note": (
                f"{share * 100:.0f}% of the {n} held-out points had an estimate within {factor:g}x "
                f"of the true cost (typical miss size {median_abs_error:.2f} {unit})."
            ),
        }
    )
    return out


def decision_regret(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str,
    cost_col: str,
    config_cols: list[str] | None = None,
) -> dict:
    """Per group (e.g. per task), pick the workflow configuration the estimate
    says is cheapest, then measure what that choice actually cost against the
    group's true cheapest workflow configuration.

    A "workflow configuration" is whatever `config_cols` names (typically
    model and effort together). Two held-out rows that share a configuration
    are replicate runs of the same choice, not two different choices to pick
    between; picking the single cheapest-looking ROW instead of the
    cheapest-looking CONFIGURATION would let a lucky replicate masquerade as
    "the right choice was made" even when it was not. So when `config_cols`
    is given, rows are first aggregated by (group, configuration) -- mean
    over replicates on both the estimate and the realized cost, since that is
    what "the row a caller would actually see for that configuration"
    means -- and the choice and the true-cheapest comparison are both made on
    those per-configuration means, never on an individual row. When
    `config_cols` is omitted, each row is its own configuration (a fixture
    with no replicate structure, or a caller that has already aggregated).

    Regret is reported both as a dollar difference and as a ratio (chosen
    configuration's cost divided by the true cheapest configuration's cost,
    so 1.0 means the choice was optimal). Each group's entry in `per_group`
    also reports `n_configs`, how many distinct workflow configurations were
    available to choose between in that group -- a reported regret of 0.00
    means little if there was only one configuration to pick.

    Rows whose group is missing, whose cost or prediction is non-finite, or
    (when `config_cols` is given) whose configuration key is missing, are
    excluded before grouping (a pandas groupby silently drops a missing group
    on its own, which would otherwise leave "n" counting rows that were never
    actually scored). "n" and "n_groups" both describe only what was scored.

    -> {"n_groups", "n", "mean_regret", "median_regret", "mean_regret_ratio",
        "n_optimal", "share_optimal", "per_group": [ {..., "n_configs"} ],
        "n_excluded_rows", "excluded_reason", "note"}
    """
    y_pred = np.asarray(y_pred, dtype=float)
    config_cols = list(config_cols) if config_cols else []
    cfg_internal = [f"__cfg{i}__" for i in range(len(config_cols))]

    df = pd.DataFrame(
        {
            "__group__": test_df[group_col].to_numpy(),
            "__cost__": pd.to_numeric(test_df[cost_col], errors="coerce").to_numpy(dtype=float),
            "__pred__": y_pred,
        }
    )
    for internal_col, real_col in zip(cfg_internal, config_cols):
        df[internal_col] = test_df[real_col].to_numpy()
    if "label" in test_df.columns:
        df["__label__"] = test_df["label"].to_numpy()
    else:
        df["__label__"] = test_df.index.to_numpy()

    n_input = int(len(df))
    group_missing = pd.isna(df["__group__"]).to_numpy()
    cost_bad = ~np.isfinite(df["__cost__"].to_numpy())
    pred_bad = ~np.isfinite(df["__pred__"].to_numpy())

    excluded_reason = {"missing_group": 0, "non_finite_cost": 0, "non_finite_prediction": 0}
    reasons_in_order = [
        ("missing_group", group_missing),
        ("non_finite_cost", cost_bad),
        ("non_finite_prediction", pred_bad),
    ]
    if cfg_internal:
        config_missing = df[cfg_internal].isna().any(axis=1).to_numpy()
        excluded_reason["missing_config"] = 0
        reasons_in_order.append(("missing_config", config_missing))

    exclude_mask = np.zeros(n_input, dtype=bool)
    for reason, bad in reasons_in_order:
        newly = bad & ~exclude_mask
        excluded_reason[reason] = int(newly.sum())
        exclude_mask |= bad

    n_excluded_rows = int(exclude_mask.sum())
    df = df.loc[~exclude_mask]
    n = int(len(df))

    out: dict = {
        "n_groups": 0,
        "n": n,
        "n_excluded_rows": n_excluded_rows,
        "excluded_reason": excluded_reason,
    }

    def _config_label(key) -> str:
        if len(config_cols) <= 1:
            return f"{config_cols[0]}={key}" if config_cols else str(key)
        return ", ".join(f"{c}={v}" for c, v in zip(config_cols, key))

    per_group = []
    for g, sub in df.groupby("__group__"):
        if sub.empty:
            continue
        if cfg_internal:
            agg = sub.groupby(cfg_internal).agg(
                __cost__=("__cost__", "mean"), __pred__=("__pred__", "mean")
            )
            n_configs = int(len(agg))
            chosen_key = agg["__pred__"].idxmin()
            best_key = agg["__cost__"].idxmin()
            chosen_cost = float(agg.loc[chosen_key, "__cost__"])
            best_cost = float(agg.loc[best_key, "__cost__"])
            chosen_label = _config_label(chosen_key)
            best_label = _config_label(best_key)
        else:
            n_configs = int(len(sub))
            chosen_row = sub.loc[sub["__pred__"].idxmin()]
            best_row = sub.loc[sub["__cost__"].idxmin()]
            chosen_cost = float(chosen_row["__cost__"])
            best_cost = float(best_row["__cost__"])
            chosen_label = chosen_row["__label__"]
            best_label = best_row["__label__"]

        regret = chosen_cost - best_cost
        regret_ratio = (chosen_cost / best_cost) if best_cost != 0 else float("nan")
        per_group.append(
            {
                "group": str(g),
                "chosen": chosen_label,
                "chosen_cost": chosen_cost,
                "best": best_label,
                "best_cost": best_cost,
                "regret": regret,
                "regret_ratio": regret_ratio,
                "n_configs": n_configs,
            }
        )

    n_groups = len(per_group)
    out["n_groups"] = n_groups
    if n_groups == 0:
        out.update(
            {
                "mean_regret": float("nan"),
                "median_regret": float("nan"),
                "mean_regret_ratio": float("nan"),
                "n_optimal": 0,
                "share_optimal": float("nan"),
                "per_group": [],
                "note": "No groups had enough held-out data to compare choices.",
            }
        )
        return out

    regrets = np.array([p["regret"] for p in per_group], dtype=float)
    ratios = np.array([p["regret_ratio"] for p in per_group], dtype=float)
    finite_ratios = ratios[np.isfinite(ratios)]
    n_optimal = int(np.sum(np.isclose(regrets, 0.0)))
    mean_regret = float(np.mean(regrets))
    median_n_configs = float(np.median([p["n_configs"] for p in per_group]))

    out.update(
        {
            "mean_regret": mean_regret,
            "median_regret": float(np.median(regrets)),
            "mean_regret_ratio": float(np.mean(finite_ratios)) if finite_ratios.size else float("nan"),
            "n_optimal": n_optimal,
            "share_optimal": n_optimal / n_groups,
            "per_group": per_group,
            "note": (
                f"Picking the workflow configuration the estimate said was cheapest cost "
                f"{mean_regret:.2f} more on average than the true cheapest configuration would "
                f"have, across {n_groups} groups ({n_optimal} of {n_groups} picked the true "
                f"cheapest configuration; a group had a median of {median_n_configs:g} "
                f"configurations available to choose between)."
            ),
        }
    )
    return out


def _value_based_bin_edges(p_sorted: np.ndarray, k: int) -> list[tuple[int, int]]:
    """Equal-count bin edges over an already-sorted array, with each boundary
    pushed forward past a run of tied values so every row sharing a
    probability lands in the same bin.

    This is what makes `acceptance_calibration` give the same answer no
    matter what order its rows were given in: the split points are a
    function of the sorted values only, never of row position. Ties heavier
    than one target bin's share make that bin larger than the rest, which is
    an honest reflection of the data, not an error.
    """
    n = int(p_sorted.size)
    if n == 0 or k <= 0:
        return []
    edges: list[tuple[int, int]] = []
    pos = 0
    remaining_bins = k
    while remaining_bins > 0 and pos < n:
        remaining_n = n - pos
        size = max(1, round(remaining_n / remaining_bins))
        end = min(pos + size, n)
        while end < n and p_sorted[end] == p_sorted[end - 1]:
            end += 1
        edges.append((pos, end))
        pos = end
        remaining_bins -= 1
    return edges


def acceptance_calibration(
    y_true_binary: np.ndarray, p_pred: np.ndarray, n_bins: int = 5
) -> dict:
    """Are the acceptance chances the model states borne out? Rows are sorted
    by their stated probability and split into `n_bins` groups of as-even-
    as-possible size, except that a bin boundary is pushed forward past any
    tied probability values so every row with the same stated probability
    lands in the same bin (this is what makes the result independent of the
    order rows were given in; see `_value_based_bin_edges`). Ties heavier
    than an equal share make that bin larger than the others.

    An outcome that does not coerce to exactly 0 or 1, or a probability
    outside [0, 1] or non-finite, cannot be scored; such rows are excluded
    rather than raising (see "n_excluded" and "excluded_reason").

    -> {"n", "brier", "ece", "bins": [{"lo","hi","n","mean_p","observed"}],
        "n_excluded", "excluded_reason", "note"}
    """
    y = np.asarray(y_true_binary, dtype=float)
    p = np.asarray(p_pred, dtype=float)
    n_input = int(y.size)

    outcome_bad = ~((y == 0.0) | (y == 1.0))
    prob_bad = ~(np.isfinite(p) & (p >= 0.0) & (p <= 1.0))

    excluded_reason = {"outcome_not_binary": 0, "probability_out_of_range": 0}
    exclude_mask = np.zeros(n_input, dtype=bool)
    for reason, bad in (
        ("outcome_not_binary", outcome_bad),
        ("probability_out_of_range", prob_bad),
    ):
        newly = bad & ~exclude_mask
        excluded_reason[reason] = int(newly.sum())
        exclude_mask |= bad

    n_excluded = int(exclude_mask.sum())
    y = y[~exclude_mask]
    p = p[~exclude_mask]
    n = int(y.size)

    out: dict = {"n": n, "n_excluded": n_excluded, "excluded_reason": excluded_reason}
    if n == 0:
        out.update(
            {
                "brier": float("nan"),
                "ece": float("nan"),
                "bins": [],
                "note": "No usable held-out outcomes were available to check.",
            }
        )
        return out

    brier = float(np.mean((p - y) ** 2))
    k = max(1, min(int(n_bins), n))
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    edges = _value_based_bin_edges(p_sorted, k)

    bins = []
    ece_num = 0.0
    for start, end in edges:
        p_bin = p_sorted[start:end]
        y_bin = y_sorted[start:end]
        mean_p = float(np.mean(p_bin))
        observed = float(np.mean(y_bin))
        bins.append(
            {
                "lo": float(np.min(p_bin)),
                "hi": float(np.max(p_bin)),
                "n": int(p_bin.size),
                "mean_p": mean_p,
                "observed": observed,
            }
        )
        ece_num += p_bin.size * abs(mean_p - observed)

    ece = ece_num / n
    out.update(
        {
            "brier": brier,
            "ece": ece,
            "bins": bins,
            "note": (
                f"Grouping held-out points by their stated acceptance chance, the average gap "
                f"between the stated chance and what actually happened was {ece:.3f} (0 is a "
                f"perfect match); the Brier score, one overall accuracy number for the stated "
                f"chances, was {brier:.3f}."
                + (
                    f" {n_excluded} points were excluded before scoring (see excluded_reason)."
                    if n_excluded
                    else ""
                )
            ),
        }
    )
    return out
