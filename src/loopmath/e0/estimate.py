"""Tokens-per-accepted estimation (design doc sections 2, 3, 5).

Implementation notes, honestly stated:
- Cell adjustment is the design's joint fixed-effects fit (step 2):
  log(1 + output tokens) = alpha_arm + gamma_cell, estimated by OLS on arm and
  cell dummies. An earlier version subtracted cell means in a single pass,
  which absorbs part of each arm's effect into the cell offsets in proportion
  to its within-cell share and attenuates arm contrasts; that estimator is
  gone. The x beta covariates of step 2 (log user turns, tool, week) are still
  not included: the turn bucket sits inside the cell, week is logged as a
  dropped deviation, and tool is nearly aliased with model. The report says
  which path ran.
- Shrinkage is method-of-moments James-Stein on the fitted arm effects:
  tau^2 by moments, per-arm shrink factor tau^2 / (tau^2 + se^2), with se^2
  from the OLS covariance.
- Acceptance rates are raw per-arm rates on the overlap-restricted set with
  known proxy, not the logit model of step 2. Acceptance sits near ceiling
  on this slice (risk 4), so the logit model would move little.
- Standardization: R_a = sum_c w_c * expm1(alpha_shrunk_a + gamma_c) / acc_a
  with w_c the pooled unit share of overlap cell c. Because the model has no
  arm-by-cell interaction, the mix factor is common across arms and S is
  driven by the shrunken arm effects; the explicit weighting keeps the
  reported levels honest about the reference mix.
- Naive per-arm tokens-per-accepted uses known-proxy sessions in both the
  numerator and the denominator (design section 3 excludes unknown-proxy
  sessions from both); the all-session token total is reported alongside.
- Price-weighted billed tokens: not implemented (secondary in the design).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from types import FunctionType as _FunctionType

from . import estimate_rework as _estimate_rework
from .estimate_rework import DAG_PAIR, FAILURE_CAUSES, _APDT_COLS

TRIVIAL_EFFORTS = ("low",)  # excluded from headline S per design section 2

# Prespecified pairwise guards (design section 5, named before the run):
SESSION_PAIR = ("opus5·medium", "fable5·xhigh")

_NAIVE_COLS = [
    "arm", "n", "known", "accepted", "acc_rate", "total_output_tokens",
    "known_output_tokens", "median_output_tokens", "naive_tokens_per_accepted",
]

def naive_arm_estimates(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """Unadjusted per-arm tokens-per-accepted (the first-cut number).

    numerator and denominator are both restricted to known-proxy sessions
    (design section 3: unknowns are excluded from both); total_output_tokens
    over all arm sessions is kept as a ledger column.
    """
    rows = []
    for arm, g in df[df["arm"].notna()].groupby("arm"):
        if len(g) < min_n:
            continue
        known = g[g["proxy_known"]] if "proxy_known" in g else g
        accepted = int(g["accepted"].sum())
        rows.append(
            {
                "arm": arm,
                "n": len(g),
                "known": len(known),
                "accepted": accepted,
                "acc_rate": accepted / len(known) if len(known) else np.nan,
                "total_output_tokens": int(g["output_tokens"].sum()),
                "known_output_tokens": int(known["output_tokens"].sum()),
                "median_output_tokens": float(g["output_tokens"].median()),
                "naive_tokens_per_accepted": (
                    known["output_tokens"].sum() / accepted if accepted >= 5 else np.nan
                ),
            }
        )
    if not rows:
        return pd.DataFrame(columns=_NAIVE_COLS)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def _overlap_cells(
    df: pd.DataFrame, eligible_arms: set[str], cell_col: str, min_cell_units: int = 2
) -> set[str]:
    """Cells containing at least 2 eligible arms with min_cell_units rows each.

    One grouped count over (cell, arm) pairs, not a Python loop over cells:
    this sits on the bootstrap's hot path (called once per draw), where the
    per-cell loop's thousands of small groupbys dominated `analyze` runtime.
    """
    sub = df[df["arm"].isin(eligible_arms)]
    if sub.empty:
        return set()
    counts = sub.groupby([cell_col, "arm"]).size()
    arms_ok = (counts >= min_cell_units).groupby(level=0).sum()
    return set(arms_ok[arms_ok >= 2].index)


def _prepare_overlap(
    df: pd.DataFrame,
    min_n: int,
    min_overlap_cells: int,
    restrict_to_overlap: bool,
    cell_col: str,
) -> tuple[pd.DataFrame, bool]:
    """Eligibility filter and overlap-cell restriction (design section 2)."""
    df = df[df["arm"].notna()].copy()
    sizes = df.groupby("arm").size()
    eligible = set(sizes[sizes >= min_n].index)
    if not eligible:
        return df.iloc[0:0], False
    overlap = _overlap_cells(df, eligible, cell_col)
    if restrict_to_overlap and overlap:
        in_overlap = df[df[cell_col].isin(overlap) & df["arm"].isin(eligible)]
        cells_per_arm = in_overlap.groupby("arm")[cell_col].nunique()
        eligible &= set(cells_per_arm[cells_per_arm >= min_overlap_cells].index)
        return in_overlap[in_overlap["arm"].isin(eligible)].copy(), True
    # fixture-sized data may have no overlap cells; fall back, flagged
    return df[df["arm"].isin(eligible)].copy(), False


def _joint_fit(work: pd.DataFrame, cell_col: str):
    """OLS fit of log1p(output_tokens) on arm dummies + cell dummies.

    Returns (arms, alpha, alpha_se2, gamma: pd.Series over cells). Arm dummies
    carry no intercept so alpha_a is the arm level in the reference cell;
    the first cell (sorted) is the reference with gamma = 0. Rank deficiency
    (possible in bootstrap draws) is handled with a pseudoinverse.
    """
    y = np.log1p(work["output_tokens"].to_numpy(dtype=float))
    arms = sorted(work["arm"].unique())
    cells = sorted(work[cell_col].unique())
    A = pd.get_dummies(work["arm"]).astype(float)[arms].to_numpy()
    if len(cells) > 1:
        Cd = pd.get_dummies(work[cell_col]).astype(float)[cells].to_numpy()[:, 1:]
        X = np.hstack([A, Cd])
    else:
        X = A
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = max(1, len(y) - np.linalg.matrix_rank(X))
    sigma2 = float(resid @ resid) / dof
    se2 = sigma2 * np.diag(xtx_inv)
    alpha = beta[: len(arms)]
    alpha_se2 = se2[: len(arms)]
    gamma = pd.Series(
        np.concatenate([[0.0], beta[len(arms):]]) if len(cells) > 1 else [0.0],
        index=cells,
    )
    return arms, alpha, alpha_se2, gamma


def shrunken_arm_estimates(
    df: pd.DataFrame,
    min_n: int = 20,
    min_overlap_cells: int = 2,
    restrict_to_overlap: bool = True,
    cell_col: str = "cell",
) -> pd.DataFrame:
    """Cell-adjusted, partially pooled, mix-standardized tokens-per-accepted (R_a).

    Steps: eligibility filter -> overlap-cell restriction -> joint OLS of
    log(1 + output tokens) on arm + cell effects -> James-Stein shrinkage of
    the fitted arm effects toward the pool -> R_a = mix-standardized level /
    acceptance rate.
    """
    work, restricted = _prepare_overlap(
        df, min_n, min_overlap_cells, restrict_to_overlap, cell_col
    )
    if work.empty or work["arm"].nunique() < 1:
        return pd.DataFrame()

    arms, alpha, alpha_se2, gamma = _joint_fit(work, cell_col)

    stats = pd.DataFrame(index=pd.Index(arms, name="arm"))
    stats["n"] = work.groupby("arm").size()
    stats["alpha"] = alpha
    stats["alpha_se"] = np.sqrt(alpha_se2)
    stats["se2"] = alpha_se2
    stats["n_overlap_cells"] = work.groupby("arm")[cell_col].nunique()

    # Method-of-moments tau^2 (between-arm variance beyond estimation noise)
    pool = stats["alpha"].mean()
    tau2 = max(
        0.0,
        stats["alpha"].var(ddof=1 if len(stats) > 1 else 0) - stats["se2"].mean(),
    )
    if np.isnan(tau2):
        tau2 = 0.0
    stats["shrink"] = tau2 / (tau2 + stats["se2"].replace(0, 1e-12))
    stats["alpha_shrunk"] = pool + stats["shrink"] * (stats["alpha"] - pool)

    # Mix standardization over the pooled overlap-cell distribution:
    # level_a = sum_c w_c * expm1(alpha_a + gamma_c)
    w = work.groupby(cell_col).size() / len(work)
    gam = gamma.loc[w.index].to_numpy()
    wc = w.to_numpy()
    stats["level"] = [
        float(np.sum(wc * np.expm1(m + gam))) for m in stats["alpha_shrunk"]
    ]
    stats["level_unshrunken"] = [
        float(np.sum(wc * np.expm1(m + gam))) for m in stats["alpha"]
    ]

    known = work[work["proxy_known"]] if "proxy_known" in work else work
    acc = known.groupby("arm")["accepted"].mean()
    stats["acc_rate"] = acc
    stats["acc_known_n"] = known.groupby("arm").size()
    # TODO(design step 3): shrink acceptance rates too (logit scale) instead of raw
    stats["R"] = stats["level"] / stats["acc_rate"]
    stats["R_unshrunken"] = stats["level_unshrunken"] / stats["acc_rate"]
    stats["overlap_restricted"] = restricted
    return stats.reset_index()


def arm_overlap_pairs(
    df: pd.DataFrame,
    min_n: int = 20,
    min_overlap_cells: int = 2,
    cell_col: str = "cell",
) -> pd.DataFrame:
    """Shared-overlap-cell counts for every eligible arm pair.

    Design section 2: arms that share no cell are compared only through the
    model, and the paper must say so. This table is the evidence for that
    statement.
    """
    work, _ = _prepare_overlap(df, min_n, min_overlap_cells, True, cell_col)
    if work.empty:
        return pd.DataFrame(columns=["arm_a", "arm_b", "shared_cells"])
    cells = {a: set(g[cell_col]) for a, g in work.groupby("arm")}
    arms = sorted(cells)
    rows = [
        {"arm_a": a, "arm_b": b, "shared_cells": len(cells[a] & cells[b])}
        for i, a in enumerate(arms)
        for b in arms[i + 1:]
    ]
    return pd.DataFrame(rows)


def within_cell_anchor(
    df: pd.DataFrame,
    min_n: int = 20,
    min_units: int = 5,
    cell_col: str = "cell",
) -> pd.DataFrame:
    """Design step 1: the model-free within-cell matched comparison.

    Within every cell holding >= 2 eligible arms with >= min_units sessions
    each, the median-cost ratio (oriented so ratio >= 1) and the acceptance
    rate difference for each arm pair. Exhaustive over eligible arms; no
    model, no extrapolation. The inverse-variance-weighted combination and
    figure F5 remain undone and are logged as such.
    """
    df = df[df["arm"].notna()]
    sizes = df.groupby("arm").size()
    eligible = set(sizes[sizes >= min_n].index)
    rows = []
    for cell, g in df[df["arm"].isin(eligible)].groupby(cell_col):
        counts = g.groupby("arm").size()
        arms = sorted(counts[counts >= min_units].index)
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                ga, gb = g[g["arm"] == a], g[g["arm"] == b]
                med_a = float(ga["output_tokens"].median())
                med_b = float(gb["output_tokens"].median())
                if min(med_a, med_b) <= 0:
                    continue
                hi, lo = (a, b) if med_a >= med_b else (b, a)
                ghi, glo = (ga, gb) if med_a >= med_b else (gb, ga)
                kh = ghi[ghi["proxy_known"]] if "proxy_known" in ghi else ghi
                kl = glo[glo["proxy_known"]] if "proxy_known" in glo else glo
                rows.append(
                    {
                        "cell": cell,
                        "arm_hi": hi,
                        "arm_lo": lo,
                        "n_hi": len(ghi),
                        "n_lo": len(glo),
                        "median_ratio": max(med_a, med_b) / min(med_a, med_b),
                        "acc_diff": (
                            float(kh["accepted"].mean() - kl["accepted"].mean())
                            if len(kh) and len(kl)
                            else np.nan
                        ),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["cell", "arm_hi", "arm_lo", "n_hi", "n_lo", "median_ratio", "acc_diff"]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("median_ratio", ascending=False)
        .reset_index(drop=True)
    )


def spread_S(est: pd.DataFrame, exclude_trivial: bool = True) -> float:
    """Headline S = max R / min R over eligible arms (design section 5)."""
    if est.empty:
        return np.nan
    e = est.dropna(subset=["R"])
    e = e[e["R"] > 0]
    if exclude_trivial:
        e = e[~e["arm"].str.endswith(TRIVIAL_EFFORTS)]
    if len(e) < 2:
        return np.nan
    return float(e["R"].max() / e["R"].min())


def bootstrap_S(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 20260830,
    min_n: int = 20,
    pair: tuple[str, str] = SESSION_PAIR,
    **est_kwargs,
) -> dict:
    """Cluster bootstrap over cells; every draw reruns adjustment + shrinkage.

    Records per-arm R draws (for per-arm intervals), the prespecified pairwise
    ratio (design section 5 guard against range-statistic bias), and per-arm
    draw diagnostics: because eligibility reruns inside each draw, the arm set
    behind S varies across draws in both directions (arms can drop out and
    arms outside the point-estimate set can enter). Both sides are reported.
    Seed is fixed and recorded in the report.
    """
    rng = np.random.default_rng(seed)
    cell_col = est_kwargs.get("cell_col", "cell")
    cells = df[cell_col].dropna().unique()
    groups = {c: df[df[cell_col] == c] for c in cells}
    draws = []
    arm_draws: dict[str, list[float]] = {}
    s_entry: dict[str, int] = {}
    s_as_max: dict[str, int] = {}
    s_as_min: dict[str, int] = {}
    pair_draws = []
    for _ in range(n_boot):
        picked = rng.choice(cells, size=len(cells), replace=True)
        boot = pd.concat([groups[c] for c in picked], ignore_index=True)
        est = shrunken_arm_estimates(boot, min_n=min_n, **est_kwargs)
        s = spread_S(est)
        if not np.isnan(s):
            draws.append(s)
        if not est.empty:
            r = est.set_index("arm")["R"]
            for arm, val in r.items():
                if np.isfinite(val) and val > 0:
                    arm_draws.setdefault(arm, []).append(float(val))
            in_s = r[np.isfinite(r) & (r > 0)]
            in_s = in_s[~in_s.index.str.endswith(TRIVIAL_EFFORTS)]
            if len(in_s) >= 2:
                for arm in in_s.index:
                    s_entry[arm] = s_entry.get(arm, 0) + 1
                s_as_max[in_s.idxmax()] = s_as_max.get(in_s.idxmax(), 0) + 1
                s_as_min[in_s.idxmin()] = s_as_min.get(in_s.idxmin(), 0) + 1
            if (
                pair[0] in r.index
                and pair[1] in r.index
                and np.isfinite(r[pair[0]])
                and np.isfinite(r[pair[1]])
                and r[pair[0]] > 0
                and r[pair[1]] > 0
            ):
                pair_draws.append(float(r[pair[1]] / r[pair[0]]))
    draws_arr = np.array(draws)
    point_est = shrunken_arm_estimates(df, min_n=min_n, **est_kwargs)
    point = spread_S(point_est)
    out = {
        "S_point": point,
        "n_boot_requested": n_boot,
        "n_boot_valid": len(draws_arr),
        "seed": seed,
        "draws": draws_arr,
        "pair": pair,
        "pair_draws": np.array(pair_draws),
        "arm_set_diag": pd.DataFrame(
            {
                "arm": sorted(s_entry),
                "in_S_draws": [s_entry[a] for a in sorted(s_entry)],
                "as_max": [s_as_max.get(a, 0) for a in sorted(s_entry)],
                "as_min": [s_as_min.get(a, 0) for a in sorted(s_entry)],
            }
        ),
    }
    if len(draws_arr):
        out["S_lo90"] = float(np.percentile(draws_arr, 5))
        out["S_hi90"] = float(np.percentile(draws_arr, 95))
        out["P_S_gt_10"] = float((draws_arr > 10).mean())
        out["n_gt_10"] = int((draws_arr > 10).sum())
        out["S_draw_max"] = float(draws_arr.max())
    else:
        out.update({"S_lo90": np.nan, "S_hi90": np.nan, "P_S_gt_10": np.nan,
                    "n_gt_10": 0, "S_draw_max": np.nan})
    if len(pair_draws):
        pd_arr = np.array(pair_draws)
        out["pair_ratio_lo90"] = float(np.percentile(pd_arr, 5))
        out["pair_ratio_hi90"] = float(np.percentile(pd_arr, 95))
    # per-arm 90% intervals
    rows = []
    for arm, vals in arm_draws.items():
        v = np.array(vals)
        rows.append(
            {
                "arm": arm,
                "R_lo90": float(np.percentile(v, 5)),
                "R_hi90": float(np.percentile(v, 95)),
                "n_draws": len(v),
            }
        )
    out["arm_intervals"] = pd.DataFrame(rows)
    # pair point ratio
    if not point_est.empty:
        r = point_est.set_index("arm")["R"]
        if pair[0] in r.index and pair[1] in r.index and r[pair[0]] > 0:
            out["pair_ratio_point"] = float(r[pair[1]] / r[pair[0]])
    return out


def unknown_brackets(
    df: pd.DataFrame, min_n: int = 20, **est_kwargs
) -> dict[str, float]:
    """Sensitivity brackets for unknown proxy outcomes (design section 3).

    Recomputes the point S under: all unknowns accepted; all unknowns not
    accepted (both entering the denominator as known).
    """
    out = {}
    hi = df.copy()
    hi.loc[~hi["proxy_known"], "accepted"] = True
    hi["proxy_known"] = True
    out["S_unknown_accepted"] = spread_S(
        shrunken_arm_estimates(hi, min_n=min_n, **est_kwargs)
    )
    lo = df.copy()
    lo.loc[~lo["proxy_known"], "accepted"] = False
    lo["proxy_known"] = True
    out["S_unknown_rejected"] = spread_S(
        shrunken_arm_estimates(lo, min_n=min_n, **est_kwargs)
    )
    return out


_moved_attempts_per_done_task = _estimate_rework.attempts_per_done_task
attempts_per_done_task = _FunctionType(
    _moved_attempts_per_done_task.__code__,
    globals(),
    _moved_attempts_per_done_task.__name__,
    _moved_attempts_per_done_task.__defaults__,
    _moved_attempts_per_done_task.__closure__,
)
attempts_per_done_task.__kwdefaults__ = _moved_attempts_per_done_task.__kwdefaults__
attempts_per_done_task.__annotations__ = _moved_attempts_per_done_task.__annotations__
attempts_per_done_task.__dict__.update(_moved_attempts_per_done_task.__dict__)
attempts_per_done_task.__doc__ = _moved_attempts_per_done_task.__doc__
attempts_per_done_task.__qualname__ = _moved_attempts_per_done_task.__qualname__
attempts_per_done_task.__module__ = __name__
_estimate_rework.attempts_per_done_task = attempts_per_done_task
del _FunctionType, _estimate_rework, _moved_attempts_per_done_task
