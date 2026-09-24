"""Exclusion and evidence-tier metadata for the cost surface."""

from __future__ import annotations

import pandas as pd


# The two cost bases this module knows how to build a view for. Dollars is
# primary per SPEC section 1.5 / decision #4; output tokens is kept as the
# secondary basis so a caller can still reproduce the token-only picture.
COST_DOLLARS = "usd"
COST_TOKENS = "out_tokens"


_TABLE_COLUMNS = [
    "arm", "n", "n_overlap_cells", "acc_rate", "shrink", "cost_per_accepted",
    "lo", "hi", "band_n_draws", "band_n_invalid_draws", "naive_cost_per_accepted",
    "cost_per_accepted_unshrunken", "acc_known_n", "overlap_restricted",
    "estimate_outside_band", "tiers",
]

# PATCH 2 (reviewer round, HIGH): evidence-tier abbreviations for the
# `tiers` composition string, in grade.py's own evidentiary-strength order
# (`_TIERS` in grade.py; `censored`/`ungraded` never appear here since both
# always carry `accepted is None`, which excludes them before this table is
# built). A tier not in this map (should not happen given grade.py's fixed
# tier set, but must never silently disappear) falls back to its own first
# letter rather than being dropped.
_TIER_ABBREV = {"verified": "v", "reported": "r", "heuristic": "h", "asserted": "a", "censored": "c", "ungraded": "u"}
_TIER_ORDER = ["verified", "reported", "heuristic", "asserted", "censored", "ungraded"]

# Exclusion reasons, in first-match-wins priority order. A row is tested
# against these in exactly this order and lands in the first one that fires;
# this is what keeps n_rows == n_rows_used + sum(exclusion counts) exact
# instead of an approximation with double-counted rows.
_NO_MODEL_LABEL = "no model label"
_NO_COST_VALUE = "no usable cost value"
_UNKNOWN_ACCEPTANCE = "unknown acceptance"
_BELOW_MIN_N = "configuration below min_n"
_BELOW_MIN_OVERLAP = "configuration below min_overlap_cells"
# FIX 2 (final reviewer round, blocker): `_prepare_overlap` drops a row for
# two DIFFERENT reasons, and the old code called both of them
# `_BELOW_MIN_OVERLAP`. That reason's detail text says the row's
# CONFIGURATION does not share enough task-mix cells with another eligible
# configuration -- true for a configuration that never clears the overlap
# requirement at all, but false for a row that belongs to a configuration
# which DID clear that requirement (on its other rows) yet itself sits in a
# cell that was never retained as shared. `_CELL_NOT_OVERLAP` is that second,
# row-level reason: the configuration is fine, this particular row's cell is
# not.
_CELL_NOT_OVERLAP = "run's task-mix cell not shared with another configuration"


def _no_cost_value_detail(cost_col: str) -> str:
    """The `_NO_COST_VALUE` detail text, worded honestly for the basis in play.

    The dollar basis and the token basis fail to produce a usable cost value
    for different reasons (pricing failure vs. a missing token count), so a
    single sentence naming "usd" for both would misdescribe one of them.
    """
    if cost_col == COST_DOLLARS:
        return "run could not be priced, so it has no dollar cost and is never counted as zero"
    if cost_col == COST_TOKENS:
        return "run has no usable output-token count, so it is never counted as zero"
    return f"run has no usable value for the '{cost_col}' cost basis, so it is never counted as zero"


def _exclusion_detail(reason: str, cost_col: str, min_n: int, min_overlap_cells: int) -> str:
    """Jargon-free, user-facing detail text for one exclusion reason."""
    return {
        _NO_MODEL_LABEL: "run has no model label, so it cannot be placed in any workflow configuration",
        _NO_COST_VALUE: _no_cost_value_detail(cost_col),
        _UNKNOWN_ACCEPTANCE: "run has an unknown acceptance outcome, so it cannot inform a cost-per-accepted-run number",
        _BELOW_MIN_N: f"fewer than {min_n} runs in this workflow configuration",
        _BELOW_MIN_OVERLAP: (
            f"workflow configuration does not share at least {min_overlap_cells} "
            "task-mix cells with another eligible configuration"
        ),
        _CELL_NOT_OVERLAP: "run is in a task-mix cell no other comparable workflow configuration worked in",
    }[reason]


def _tier_composition(tiers: pd.Series) -> str:
    """Compact `"<n><letter>/<n><letter>"` evidence-tier breakdown, e.g. `"31v/2h"`.

    PATCH 2 (reviewer round, HIGH): the acceptance rate behind a
    configuration's cost-per-accepted number is measured by whichever
    evidence tier (grade.py) its rows landed on, and a `verified` row is
    accepted by construction while a `heuristic`/`reported` row is not. This
    string is that composition, so a reader can see, per configuration,
    which instrument produced its acceptance rate instead of the tier mix
    disappearing behind a single percentage. Order matches grade.py's own
    evidentiary-strength order (`_TIER_ORDER`); a tier with zero rows here is
    omitted rather than printed as `"0x"`. `"n/a"` only when `tiers` is
    empty, which should not happen for a configuration that made it into the
    table at all (every row there already has a known acceptance outcome).
    """
    if tiers.empty:
        return "n/a"
    counts = tiers.value_counts()
    parts = [
        f"{int(counts[t])}{_TIER_ABBREV.get(t, str(t)[:1])}"
        for t in _TIER_ORDER
        if t in counts.index and counts[t] > 0
    ]
    # A tier value outside grade.py's known set should never happen, but must
    # never silently vanish from the count either if it did.
    extra = sorted(t for t in counts.index if t not in _TIER_ORDER)
    parts.extend(f"{int(counts[t])}{str(t)[:1]}" for t in extra)
    return "/".join(parts) if parts else "n/a"


def _tier_note(tiers_present: set) -> str | None:
    """Plain-language caveat: acceptance is not measured the same way for every configuration.

    Only fires when the fitted population mixes `verified` rows (accepted by
    construction; grade.py has no verified-and-rejected path) with at least
    one other tier. When every fitted row shares one tier, or no `verified`
    row is present at all, the configurations in the table were measured by
    the same instrument as each other, and this returns `None` rather than
    raising an alarm about a confound that is not actually live here.
    """
    if "verified" in tiers_present and len(tiers_present) > 1:
        return (
            "Acceptance is not measured the same way for every workflow "
            "configuration above: rows graded at the verified tier are "
            "accepted by construction, so a configuration with more "
            "verified rows tends to show a higher acceptance rate for "
            "reasons about the evidence available, not about the work "
            "itself. See the tiers column for each configuration's mix."
        )
    return None
