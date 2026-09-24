"""Task-mix-adjusted cost-surface estimation."""

from __future__ import annotations

import pandas as pd

from loopmath.e0.estimate import (
    _prepare_overlap,
    arm_overlap_pairs,
    naive_arm_estimates,
    shrunken_arm_estimates,
    within_cell_anchor,
)

from .surface_bootstrap import _bootstrap_bands, _spread_of
from .surface_presentation import _band_note, _band_support_note
from .surface_support import (
    COST_DOLLARS,
    _BELOW_MIN_N,
    _BELOW_MIN_OVERLAP,
    _CELL_NOT_OVERLAP,
    _NO_COST_VALUE,
    _NO_MODEL_LABEL,
    _TABLE_COLUMNS,
    _UNKNOWN_ACCEPTANCE,
    _exclusion_detail,
    _tier_composition,
    _tier_note,
)


def cost_surface(
    df: pd.DataFrame,
    cost_col: str = COST_DOLLARS,
    min_n: int = 5,
    min_overlap_cells: int = 2,
    n_boot: int = 1000,
    seed: int = 20260831,
    ci: float = 0.80,
) -> dict:
    """The task-mix-honest cost surface: stratify by cell, pool thin configurations, band it.

    `df` is `build_frame`'s output (or anything with the same columns).
    `cost_col` selects the cost basis fed into the E0 view (`COST_DOLLARS` by
    default, `COST_TOKENS` for the secondary token picture). Every row is
    accounted for exactly once: it either ends up in the fit behind `table`,
    or it is placed in exactly one bucket of `exclusions`, first matching
    reason wins, so `n_rows == n_rows_used + sum(e["n"] for e in exclusions)`
    always holds. `exclusion_by_run_id` names the same partition at row
    granularity: a dict of `run_id -> reason` for every excluded row (and
    only excluded rows), so a caller (or a test) can check disjointness and
    membership without re-deriving it from the aggregate counts.

    When `min_overlap_cells` finds zero overlap cells at all, the eligibility
    filter this reuses from E0 (`_prepare_overlap`, frozen, not reimplemented)
    falls back to every min_n-eligible row and reports `restricted=False`
    instead of raising or leaving the table empty -- E0 was written to stay
    usable on fixture-sized data with no shared cells. That fallback table is
    NOT task-mix-adjusted, even though it is shaped identically to the honest
    case. This function will not silently accept that: `overlap_caveat` is a
    non-empty, user-facing sentence whenever `restricted` came back False on
    a non-empty fit, and `None` otherwise; `walkdown` and `walkdown_line`
    read it and say so out loud rather than presenting an unadjusted number
    as if it were mix-standardized.

    `table["estimate_outside_band"]` is True for a configuration whose point
    estimate sits outside its own `[lo, hi]`, False (never NaN) when the band
    is missing or the estimate sits inside it. `n_estimate_outside_band`
    counts that column; `band_n_draws_min`/`band_n_draws_max` are the
    smallest and largest `band_n_draws` among configurations that actually
    got a band (`None` when none did). `band_note` is a user-facing sentence
    naming the effect (only set when `n_estimate_outside_band > 0`);
    `band_support_note` is a user-facing sentence naming the thinnest band's
    draw support and/or its count of invalid resampling draws (only set when
    at least one of those applies; see `_bootstrap_bands`, FIX 3).

    `n_no_workspace` (FIX 5, reviewer round) counts input rows with no known
    workspace: `build_frame` gives each of those its own unmatchable cell, so
    none of them can ever take part in task-mix matching. This is reported
    separately from `exclusions` above (a workspace-less row is not itself
    dropped from the fit; its configuration can still be estimated normally
    on its OTHER rows, or via this row alone if its arm's other cells already
    clear the overlap requirement) rather than folded into that ladder.
    `no_workspace_note` is the user-facing sentence naming the count, set
    only when `n_no_workspace > 0`.

    `table["tiers"]` (PATCH 2, reviewer round) is each configuration's
    evidence-tier composition over the same fitted rows `acc_rate` was
    computed from (e.g. `"31v/2h"`: 31 verified, 2 heuristic). `tier_note`
    is a user-facing caveat, set only when the fitted population mixes
    `verified` rows with some other tier -- the specific case where
    "verified is accepted by construction" makes one configuration's
    acceptance rate a different measurement than another's, not merely a
    different result.
    """
    n_rows = len(df)
    reason = pd.Series([None] * n_rows, index=df.index, dtype=object)

    # FIX 5 (reviewer round, accepted in part): a run with no known workspace
    # got its own unmatchable cell in `build_frame`, precisely so it cannot be
    # mistaken for sharing task mix with any other row. Counted here (over
    # every input row, not just the ones that survive the exclusion ladder
    # below) so the reader is told how many runs that affects, whenever it is
    # more than zero.
    n_no_workspace = int(df["workspace"].isna().sum()) if "workspace" in df else 0
    no_workspace_note = (
        (
            f"{n_no_workspace} run has no recorded workspace, so it cannot take "
            "part in task-mix matching"
            if n_no_workspace == 1
            else f"{n_no_workspace} runs have no recorded workspace, so they "
            "cannot take part in task-mix matching"
        )
        if n_no_workspace > 0
        else None
    )

    # First-match-wins exclusion ladder. Order matters: a row with no model
    # label is never also counted as having no usable cost value, etc.
    reason[df["arm"].isna()] = _NO_MODEL_LABEL
    no_cost_value = df[cost_col].isna() & reason.isna()
    reason[no_cost_value] = _NO_COST_VALUE
    unknown_acc = df["accepted"].isna() & reason.isna()
    reason[unknown_acc] = _UNKNOWN_ACCEPTANCE

    remaining = df.loc[reason.isna()]

    # The E0 view trick: carry the chosen cost basis under the name E0's
    # functions expect. `accepted` is coerced to bool (never None) because by
    # this point every unknown-acceptance row has already been excluded above,
    # matching what E0's own session tables always look like (see
    # loopmath.e0.arms.build_session_table: accepted is always a plain bool,
    # proxy_known is the separate flag). `proxy_known` is therefore True for
    # every remaining row by construction.
    view = pd.DataFrame(
        {
            "arm": remaining["arm"].to_numpy(),
            "cell": remaining["cell"].to_numpy(),
            "output_tokens": remaining[cost_col].to_numpy(dtype=float),
            "accepted": remaining["accepted"].astype(bool).to_numpy(),
            "proxy_known": True,
        },
        index=remaining.index,
    )

    # Exactly the eligibility + overlap-cell restriction shrunken_arm_estimates
    # applies internally (imported, not reimplemented, so this can never
    # silently drift from what the table actually fits on).
    work, restricted = _prepare_overlap(view, min_n, min_overlap_cells, True, "cell")
    n_rows_used = len(work)

    sizes = view.groupby("arm").size()
    eligible_by_min_n = set(sizes[sizes >= min_n].index)
    # FIX 2 (final reviewer round, blocker): a row can be missing from `work`
    # for two different reasons, and they are NOT the same claim. Either (a)
    # its configuration never cleared the overlap requirement at all -- none
    # of that configuration's rows are in `work`, and `_BELOW_MIN_OVERLAP`'s
    # detail ("configuration does not share ... cells") is true of it -- or
    # (b) its configuration DID clear that requirement (some of its OTHER
    # rows are in `work`) but this particular row sits in a task-mix cell
    # that was never retained as an overlap cell (too few comparable
    # configurations worked in it, or this configuration did not have enough
    # rows there). Case (b) is `_CELL_NOT_OVERLAP`: measured per row, not
    # blamed on the whole configuration.
    work_arms = set(work["arm"].unique()) if not work.empty else set()
    for idx in view.index.difference(work.index):
        arm = view.at[idx, "arm"]
        if arm not in eligible_by_min_n:
            reason.at[idx] = _BELOW_MIN_N
        elif arm not in work_arms:
            reason.at[idx] = _BELOW_MIN_OVERLAP
        else:
            reason.at[idx] = _CELL_NOT_OVERLAP

    # `n_censored` is the measured intersection of this bucket with the
    # censored evidence tier (grade.py), computed per reason so the buckets
    # still sum to `n_rows` and nothing is double counted. A censored run
    # always has `accepted is None` by construction (grade.py's R5 rule only
    # ever demotes an accepted=None record), so it can only ever land here or
    # in `_NO_MODEL_LABEL` / `_NO_COST_VALUE` (whichever fires first on that
    # row); `_BELOW_MIN_N` / `_BELOW_MIN_OVERLAP` / `_CELL_NOT_OVERLAP` only
    # ever hold rows with a known acceptance outcome, so their `n_censored`
    # is always 0. This is what lets a caller (terminal.py) print "of which
    # N were censored" under whichever bucket actually holds them instead of
    # asserting they are all under "unknown acceptance".
    exclusions = [
        {
            "reason": r,
            "n": int((reason == r).sum()),
            "detail": _exclusion_detail(r, cost_col, min_n, min_overlap_cells),
            "n_censored": int(((reason == r) & (df["tier"] == "censored")).sum()),
        }
        for r in (
            _NO_MODEL_LABEL,
            _NO_COST_VALUE,
            _UNKNOWN_ACCEPTANCE,
            _BELOW_MIN_N,
            _BELOW_MIN_OVERLAP,
            _CELL_NOT_OVERLAP,
        )
    ]
    n_unpriced_dropped = next(e["n"] for e in exclusions if e["reason"] == _NO_COST_VALUE)
    excluded_mask = reason.notna()
    exclusion_by_run_id = dict(zip(df.loc[excluded_mask, "run_id"], reason[excluded_mask]))

    # FIX 3 (final reviewer round, blocker): on this fallback path,
    # `shrunken_arm_estimates` below still runs its full pipeline (joint fit,
    # James-Stein shrinkage toward the pool, mix standardization) over the
    # unmatched rows -- the numbers in the table are fitted, shrunken
    # cost-per-accepted estimates, never the raw per-run figures the old
    # sentence claimed. Read the code before touching this string again.
    overlap_caveat = (
        "No task-mix cell contains two comparable workflow configurations, so "
        "task-mix matching was not possible. These are cost-per-accepted "
        "estimates fitted without matching, not a like-for-like comparison."
        if (not restricted and len(work) > 0)
        else None
    )

    naive = naive_arm_estimates(view, min_n=min_n)
    # naive_arm_estimates casts its dollar-ledger columns through int(), which
    # is correct on E0's native token basis but truncates a dollar total to
    # whole dollars. Overwrite with exact float sums computed here (not in
    # the frozen estimate.py) and rename so no caller can mistake a dollar
    # sum for a token count (fix for the same reason `output_tokens` in
    # `view` is a cost, not necessarily tokens).
    naive = naive.rename(
        columns={
            "total_output_tokens": "total_cost",
            "known_output_tokens": "known_cost",
            "median_output_tokens": "median_cost",
        }
    )
    if not naive.empty:
        known_view = view[view["proxy_known"]] if "proxy_known" in view else view
        totals = view.groupby("arm")["output_tokens"].sum()
        known_totals = known_view.groupby("arm")["output_tokens"].sum()
        medians = view.groupby("arm")["output_tokens"].median()
        naive["total_cost"] = naive["arm"].map(totals).astype(float)
        naive["known_cost"] = naive["arm"].map(known_totals).astype(float)
        naive["median_cost"] = naive["arm"].map(medians).astype(float)

    anchor = within_cell_anchor(view, min_n=min_n, cell_col="cell")
    shrunk = shrunken_arm_estimates(view, min_n=min_n, min_overlap_cells=min_overlap_cells, cell_col="cell")
    overlap_pairs = arm_overlap_pairs(view, min_n=min_n, min_overlap_cells=min_overlap_cells, cell_col="cell")
    bands = _bootstrap_bands(view, n_boot=n_boot, seed=seed, ci=ci, min_n=min_n, min_overlap_cells=min_overlap_cells)

    table = shrunk.rename(columns={"R": "cost_per_accepted", "R_unshrunken": "cost_per_accepted_unshrunken"})
    if not table.empty:
        table = table.merge(
            naive[["arm", "naive_tokens_per_accepted"]].rename(
                columns={"naive_tokens_per_accepted": "naive_cost_per_accepted"}
            ),
            on="arm",
            how="left",
        )
        table = table.merge(
            bands["per_arm"][["arm", "lo", "hi", "n_draws", "n_invalid_draws"]].rename(
                columns={"n_draws": "band_n_draws", "n_invalid_draws": "band_n_invalid_draws"}
            ),
            on="arm",
            how="left",
        )
        # A configuration whose band is missing (no arm draw ever survived
        # eligibility + overlap in the resampling) is False here, never True
        # and never NaN: "no band to compare against" is not the same claim
        # as "the estimate sits outside its band."
        has_band = table["lo"].notna() & table["hi"].notna()
        outside_band = (table["cost_per_accepted"] < table["lo"]) | (table["cost_per_accepted"] > table["hi"])
        table["estimate_outside_band"] = (has_band & outside_band).fillna(False)

        # PATCH 2 (reviewer round, HIGH): tier composition over exactly the
        # rows `shrunken_arm_estimates` used for `acc_rate` above (`work`,
        # the eligibility- and overlap-restricted set), not the wider input
        # frame -- the composition must describe the same population the
        # printed acceptance rate was computed from.
        tier_by_row = df.loc[work.index, "tier"] if len(work) else pd.Series(dtype=object)
        tier_comp = (
            pd.DataFrame({"arm": work["arm"].to_numpy(), "tier": tier_by_row.to_numpy()})
            .groupby("arm")["tier"]
            .apply(_tier_composition)
            .rename("tiers")
            .reset_index()
            if len(work)
            else pd.DataFrame(columns=["arm", "tiers"])
        )
        table = table.merge(tier_comp, on="arm", how="left")
        table["tiers"] = table["tiers"].fillna("n/a")
    table = table.reindex(columns=_TABLE_COLUMNS)

    if len(table):
        n_estimate_outside_band = int(table["estimate_outside_band"].fillna(False).astype(bool).sum())
        band_draws = table["band_n_draws"].dropna()
        # FIX 3: total count of resampling draws that produced a NaN,
        # infinite, or negative point estimate for SOME configuration, summed
        # across all configurations (never silently dropped; see
        # `_bootstrap_bands`).
        band_n_invalid_draws_total = int(table["band_n_invalid_draws"].fillna(0).sum())
    else:
        n_estimate_outside_band = 0
        band_draws = pd.Series(dtype=float)
        band_n_invalid_draws_total = 0

    # Only over configurations that actually got a band at all (a config with
    # no surviving draw contributes nothing here, not a spurious 0).
    band_n_draws_min = int(band_draws.min()) if len(band_draws) else None
    band_n_draws_max = int(band_draws.max()) if len(band_draws) else None

    band_note = (
        _band_note(n_estimate_outside_band, len(table), cost_col)
        if n_estimate_outside_band > 0
        else None
    )
    # FIX 3: fires whenever the thinnest band rests on fewer than n_boot
    # draws (as before) OR whenever any configuration's resampling produced
    # an invalid draw at all -- the latter can be true even when every band
    # happens to be full, and must not stay silent just because it does.
    # The spread line prints its own 80% band from `s_draws`, whose support
    # was computed and returned but never shown to a reader. A spread draw is
    # dropped when that resample produced no usable spread at all, so the same
    # honesty that applies to a thin per-configuration band applies here.
    n_spread_draws = int(bands.get("n_boot_valid_S", 0))
    band_support_note = (
        _band_support_note(
            band_n_draws_min, n_boot, band_n_invalid_draws_total, n_spread_draws
        )
        if (band_n_draws_min is not None and band_n_draws_min < n_boot)
        or band_n_invalid_draws_total > 0
        or n_spread_draws < n_boot
        else None
    )

    # PATCH 2: the caveat fires only when the fitted population (`work`)
    # actually mixes `verified` rows with some other tier -- that is the
    # specific condition under which "accepted by construction" makes one
    # configuration's acceptance rate incomparable to another's. A table
    # that is all one tier, or has no verified rows at all, was measured by
    # one instrument throughout, so no caveat is raised about a confound
    # that is not live.
    tiers_present = set(df.loc[work.index, "tier"].dropna()) if len(work) else set()
    tier_note = _tier_note(tiers_present)

    # S_naive must describe the same population S_matched/S_honest do (the
    # overlap-eligible configurations behind `shrunk`), not every
    # min_n-eligible configuration -- otherwise the first step of the
    # walkdown compares a different set of configurations than the rest of
    # it. Filter naive down to shrunk's arm set before spreading it; the
    # unrestricted spread over every min_n-eligible configuration is still
    # exposed, under a name that says so, for a caller that wants it.
    shrunk_arms = set(shrunk["arm"]) if "arm" in shrunk.columns else set()
    naive_matched_population = naive[naive["arm"].isin(shrunk_arms)]

    return {
        "table": table,
        "cost_col": cost_col,
        "ci": ci,
        "seed": seed,
        "n_boot": n_boot,
        "n_boot_valid": bands["n_boot_valid"],
        "n_boot_valid_S": bands["n_boot_valid_S"],
        "naive": naive,
        "anchor": anchor,
        "overlap_pairs": overlap_pairs,
        # Walkdown steps (naive to honest); S_matched is not in SPEC's minimal
        # key list but is needed by `walkdown` and costs nothing to expose.
        "S_naive": _spread_of(naive_matched_population, "naive_tokens_per_accepted"),
        "S_naive_all_configs": _spread_of(naive, "naive_tokens_per_accepted"),
        "S_matched": _spread_of(shrunk, "R_unshrunken"),
        "S_honest": _spread_of(shrunk, "R"),
        "S_lo": bands["S_lo"],
        "S_hi": bands["S_hi"],
        "n_rows": n_rows,
        "n_rows_used": n_rows_used,
        "n_unpriced_dropped": n_unpriced_dropped,
        "n_cells": int(view["cell"].nunique()),
        "exclusions": exclusions,
        "exclusion_by_run_id": exclusion_by_run_id,
        "overlap_restricted": restricted,
        "overlap_caveat": overlap_caveat,
        "n_estimate_outside_band": n_estimate_outside_band,
        "band_n_draws_min": band_n_draws_min,
        "band_n_draws_max": band_n_draws_max,
        "band_n_invalid_draws_total": band_n_invalid_draws_total,
        "band_note": band_note,
        "band_support_note": band_support_note,
        "n_no_workspace": n_no_workspace,
        "no_workspace_note": no_workspace_note,
        "tier_note": tier_note,
    }
