"""Bootstrap and spread calculations for the cost surface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from loopmath.e0.estimate import shrunken_arm_estimates, spread_S


def _spread_of(df: pd.DataFrame, level_col: str) -> float:
    """`spread_S` on an arbitrary per-configuration level column.

    `spread_S` hardcodes the column name `R`; this borrows it for whichever
    level column a walkdown step needs (the naive per-accepted level, the
    unshrunken mix-standardized level, or `R` itself) by projecting to a
    two-column frame instead of renaming in place, which avoids ever holding
    two columns both named `R` at once.

    PATCH 1 (reviewer round, HIGH): called with `exclude_trivial=False`.
    `spread_S`'s default (`True`) drops every configuration whose name ends
    in "low" (`TRIVIAL_EFFORTS`, `loopmath.e0.estimate`) -- a design decision
    E0 made for its OWN headline token study (design section 2), where a
    trivial-task low-effort arm was a known degenerate baseline not worth
    ranging over. This module's headline spread is a different report, over
    a different population, with its own documented promise that every row
    is accounted for and nothing is silently dropped; low-effort workflow
    configurations are exactly what this build exists to measure, so
    inheriting E0's exclusion here would silently zero out the cheapest
    configuration while `n_rows_used == n_rows` kept claiming full coverage.
    `spread_S` itself is untouched (frozen, still called unmodified) --
    only the argument this module passes to it changes.
    """
    if df.empty or level_col not in df.columns:
        return float("nan")
    return spread_S(
        df[["arm", level_col]].rename(columns={level_col: "R"}), exclude_trivial=False
    )


def _bootstrap_bands(
    view: pd.DataFrame,
    n_boot: int,
    seed: int,
    ci: float,
    min_n: int,
    min_overlap_cells: int,
    cell_col: str = "cell",
) -> dict:
    """Cell-cluster bootstrap at an arbitrary confidence level.

    Same resampling scheme as E0's `bootstrap_S` (whole cells resampled with
    replacement, matching how the fit pools; `shrunken_arm_estimates` refit
    on every draw) but not that function, because `bootstrap_S` hardcodes the
     90% (5th/95th percentile) band and SPEC section 6 requires 80%. This also
    returns per-configuration bands, which `bootstrap_S` does not.

    A draw whose refit comes back empty (possible when a resample happens to
    starve a configuration below `min_n`) is skipped and counted, never
    treated as a zero or dropped silently; `n_boot_valid` is how many draws
    refit at all. That is not the same claim as "how many draws back the
    headline band": a draw can refit non-empty and still come back with a
    NaN `spread_S` (fewer than two priced arms survive that draw's
    eligibility + overlap restriction), in which case it contributes nothing
    to `S_lo`/`S_hi`. `n_boot_valid_S` (`== len(s_draws)`) counts only the
    draws that actually landed in the headline band, so a caller can see the
    two numbers can differ instead of `n_boot_valid` silently overstating
    the band's support.

    FIX 3 (reviewer round, blocker): a per-configuration draw is kept in
    `arm_draws` (and so backs that configuration's `lo`/`hi` band) whenever it
    is finite and non-negative -- a resampled cost-per-accepted estimate of
    EXACTLY ZERO is a valid draw (a real, if extreme, outcome; the token
    basis in particular can genuinely total zero output tokens for a
    configuration), not an invalid one, and the old `val > 0` filter dropped
    it silently, shrinking that configuration's `n_draws` and biasing its
    band upward with no sign anything had been excluded. This does not touch
    `spread_S` (frozen, in `loopmath.e0.estimate`): that function's own `R > 0`
    filter exists because `spread_S` DIVIDES by the smallest `R` to form a
    ratio, where a zero or negative value is genuinely unusable; `arm_draws`
    here only ever feeds a percentile band on `R` itself, never a division,
    so zero is safe to keep. A draw that is NOT finite and non-negative (NaN,
    infinite -- which happens in practice when a resample's known-outcome
    rows are all rejected, giving a zero acceptance rate and so an infinite
    cost-per-accepted -- or, in principle, negative, which `spread_S` treats
    as equally unusable for the same reason) is counted per configuration in
    `n_invalid_draws` instead of vanishing with no trace.
    """
    rng = np.random.default_rng(seed)
    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 - (1.0 - ci) / 2.0) * 100.0

    cells = view[cell_col].dropna().unique()
    groups = {c: view[view[cell_col] == c] for c in cells}

    s_draws: list[float] = []
    arm_draws: dict[str, list[float]] = {}
    arm_invalid: dict[str, int] = {}
    n_valid = 0

    if len(cells):
        for _ in range(n_boot):
            picked = rng.choice(cells, size=len(cells), replace=True)
            boot = pd.concat([groups[c] for c in picked], ignore_index=True)
            est = shrunken_arm_estimates(
                boot, min_n=min_n, min_overlap_cells=min_overlap_cells, cell_col=cell_col
            )
            if est.empty:
                continue
            n_valid += 1
            # PATCH 1: exclude_trivial=False, matching `_spread_of` -- the
            # headline S_lo/S_hi band must bracket the same spread the point
            # estimate reports, and that point estimate no longer drops
            # low-effort configurations either. See `_spread_of`'s docstring.
            s = spread_S(est, exclude_trivial=False)
            # Finite, not merely non-NaN. `spread_S` is max(R)/min(R), so a
            # draw in which the cheapest configuration lands at exactly zero
            # returns positive infinity, and an infinite draw fed to
            # `np.percentile` can push the whole spread band to infinity. It
            # is not a usable spread, so it is dropped here for the same
            # reason a NaN draw is, and it stays visible the same way: the
            # gap between `n_boot_valid` and `n_boot_valid_S` counts every
            # draw that refit fine but produced no usable spread.
            if np.isfinite(s):
                s_draws.append(s)
            r = est.set_index("arm")["R"]
            for arm, val in r.items():
                # FIX 3: zero is a valid draw (kept); NaN/infinite/negative is
                # not (counted instead of silently vanishing). See the
                # docstring above for why this differs from `spread_S`'s own
                # `R > 0` filter, and why keeping the latter unchanged is
                # deliberate, not an oversight.
                if np.isfinite(val) and val >= 0:
                    arm_draws.setdefault(arm, []).append(float(val))
                else:
                    arm_invalid[arm] = arm_invalid.get(arm, 0) + 1

    per_arm = pd.DataFrame(
        [
            {
                "arm": arm,
                "lo": float(np.percentile(arm_draws[arm], lo_q)) if arm in arm_draws else float("nan"),
                "hi": float(np.percentile(arm_draws[arm], hi_q)) if arm in arm_draws else float("nan"),
                "n_draws": len(arm_draws.get(arm, [])),
                "n_invalid_draws": arm_invalid.get(arm, 0),
            }
            # Union of both dicts: an arm that appeared in every draw only as
            # an invalid estimate (never once finite and non-negative) still
            # gets a row here, with n_draws=0, rather than disappearing from
            # `per_arm` entirely with its invalid count going nowhere.
            for arm in (arm_draws.keys() | arm_invalid.keys())
        ],
        columns=["arm", "lo", "hi", "n_draws", "n_invalid_draws"],
    )

    return {
        "per_arm": per_arm,
        "S_lo": float(np.percentile(s_draws, lo_q)) if s_draws else float("nan"),
        "S_hi": float(np.percentile(s_draws, hi_q)) if s_draws else float("nan"),
        "n_boot_valid": n_valid,
        "n_boot_valid_S": len(s_draws),
        "seed": seed,
        "ci": ci,
    }
