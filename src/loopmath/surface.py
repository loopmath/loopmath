"""Task-mix-honest cost surface: dollars primary, output tokens secondary (SPEC section 6).

What this module is: a thin wrapper around the frozen E0 estimator
(`loopmath.e0.estimate`), pointed at a run's dollar cost instead of its raw
output-token count, with 80% bands instead of E0's hardcoded 90%. It does not
reimplement any of E0's statistics. Every number in `cost_surface`'s "table"
comes from `naive_arm_estimates` / `shrunken_arm_estimates` / `spread_S`
running unmodified, which is what "the E0 numbers keep reproducing" means in
practice: run the E0 test suite against `loopmath.e0.estimate` directly and it
still passes, because that file is never touched here.

The view trick, spelled out once so it never needs re-deriving: E0's
functions all read fixed column names -- `output_tokens`, `arm`, `cell`,
`accepted`, `proxy_known` -- because it was written for one hardcoded cost
basis (raw output tokens). To reuse it honestly for a different cost basis
(dollars, or output tokens again, or whatever a future `cost_col` names), this
module builds a small "view" DataFrame that carries the chosen cost column
under the name `output_tokens` and the same `arm`/`cell`/`accepted` columns
E0 already expects, then calls the real E0 functions on that view. The E0
code never sees that the numbers underneath are dollars; it just does the
same joint OLS + James-Stein shrinkage + mix standardization it always does.
That is also why every table this module hands back that did not go through
the rename (`anchor`, `overlap_pairs`) still carries E0's original column
names (`output_tokens`, `naive_tokens_per_accepted`, ...) even when the
values are dollars: renaming those too would be extra surface for no honesty
gain. `naive` is the one partial exception: its per-arm dollar-ledger columns
(`total_output_tokens`, `known_output_tokens`, `median_output_tokens` in E0's
naming) are recomputed here as exact floats and renamed to `total_cost`,
`known_cost`, `median_cost`, because E0's own ledger columns are int-cast
(correct on E0's native token basis, silently truncating on a dollar basis)
and a column still named "tokens" holding a dollar sum would be actively
misleading, not just inconsistent. `table`, the public-facing summary, gets
fully cost-generic names throughout.

What this module adds on top of the port:
- `build_frame`: turns graded, priced run-record dicts into the DataFrame
  shape the rest of this module (and E0) needs. Every input record keeps a
  row, even one with no model label or an unusable cost value, so nothing is
  dropped before `cost_surface` gets a chance to count and name it. The
  `cell` it builds -- a workspace paired with the most-written file
  extension -- is the task-mix stratum: it is the only shape of the work a
  metadata-only parse can see (SPEC's privacy contract forbids reading
  prompt or file content), so it is what "task-mix-honest" is standing on. A
  parse that cannot see task content can still see where files landed and
  what kind they were, and that is the cheapest available proxy for "what
  shape of work was this." A run with no known workspace gets a cell unique
  to that one row instead (FIX 5, reviewer round), so it can never be
  falsely matched against another workspace-less run it has nothing to do
  with; `cost_surface` counts these rows (`n_no_workspace`) and names them
  when the count is non-zero.
- 80% bands via `_bootstrap_bands`, a cell-cluster bootstrap matching E0's own
  `bootstrap_S` resampling scheme (whole cells resampled with replacement,
  `shrunken_arm_estimates` refit on every draw) but parameterized on `ci`
  instead of frozen at the 5th/95th percentile, and returning per-configuration
  bands (E0's `bootstrap_S` only returns the headline spread's band). Draw
  support is surfaced two ways: `n_boot_valid` (draws whose refit was
  non-empty) vs. `n_boot_valid_S` (the subset of those that actually produced
  a finite headline spread and so back `S_lo`/`S_hi`), and `band_n_draws` per
  configuration in `table`. A resampled per-configuration estimate of exactly
  zero counts as a valid draw (kept); one that is NaN, infinite, or negative
  does not, and is counted (not silently dropped) in `band_n_invalid_draws`
  per configuration and `band_n_invalid_draws_total` overall (FIX 3, reviewer
  round).
- `walkdown` / `walkdown_line`: the naive-to-honest walkdown SPEC section 6
  asks for, built entirely from tables the E0 engine already produced. Both
  name their cost basis (dollars, output tokens, ...) instead of assuming
  dollars, and both say so plainly on a surface whose `overlap_caveat` is
  set -- meaning no task-mix cell had two comparable configurations to match
  on, so the matching and pooling steps could not run and the numbers are
  raw, unmatched costs, not a task-mix-honest comparison. All three walkdown
  spreads, and the bootstrap band around the honest one, call `spread_S` with
  `exclude_trivial=False` (PATCH 1, reviewer round): E0's default silently
  drops every configuration whose name ends in "low", which would zero the
  cheapest configuration out of the headline number on exactly the build
  this exists to measure, with `n_rows_used == n_rows` still claiming full
  coverage. `spread_S` itself is frozen and unmodified; only the argument
  passed to it here changed. Per-configuration bands were never affected by
  this (they read `est.set_index("arm")["R"]` directly, with no trivial
  filter) -- only the four spread-ratio numbers were.
- Tier composition (PATCH 2, reviewer round): the acceptance rate behind
  each configuration's cost-per-accepted is measured by a different
  instrument depending on which evidence tier (grade.py) its rows landed
  on -- a `verified` row is accepted by construction (grade.py has no
  verified-and-rejected path), while `heuristic`/`reported` rows are not.
  `table["tiers"]` gives each configuration's tier composition as a compact
  string (`"31v/2h"`); `tier_note` is a plain-language caveat, set only
  when the fitted population mixes `verified` rows with any other tier, so
  a reader is told when (and only when) that confound is actually live.
- Band honesty: a percentile bootstrap on a ratio estimator can, and here
  routinely does, place the point estimate outside its own band (resampling
  whole cells trims cell diversity, and by Jensen's inequality the mean of
  the resampled ratios sits above the once-computed ratio). `table` carries
  `estimate_outside_band` per configuration; `n_estimate_outside_band`,
  `band_n_draws_min`/`band_n_draws_max`, `band_note`, and
  `band_support_note` say so in plain language rather than leaving a row
  that looks like a bug to speak for itself. This does not change the band
  itself: still the same percentile band E0 always produced, just named
  honestly.

Honesty commitments this module keeps: every row of the input frame is
accounted for. A row that does not make it into the final table is placed in
exactly one exclusion bucket (first matching reason wins, so the counts never
double up), and `n_rows == n_rows_used + sum(exclusion counts)` always holds;
`exclusion_by_run_id` names that same partition at row granularity. Nothing
is ever silently dropped, and a fit that could not be adjusted for task mix
(`overlap_caveat`) says so instead of looking identical to one that was.

Vocabulary note: internally this keeps E0's name for the grouping variable,
`arm`, in code and comments (SPEC exempts internal code from the forbidden
list). Every string in this module that a user might see -- exclusion
`detail` text, `walkdown_line`'s output -- says "workflow configuration"
instead, and none of it uses an em dash.
"""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pandas as pd

from loopmath.e0.estimate import (
    _prepare_overlap,
    arm_overlap_pairs,
    naive_arm_estimates,
    shrunken_arm_estimates,
    spread_S,
    within_cell_anchor,
)

from . import surface_bootstrap as _bootstrap
from . import surface_estimation as _estimation
from . import surface_frame as _frame
from . import surface_presentation as _presentation
from . import surface_support as _support
from .surface_bootstrap import _bootstrap_bands, _spread_of
from .surface_estimation import cost_surface
from .surface_frame import _FRAME_COLUMNS, _coerce_token_stream, _primary_kind, build_frame
from .surface_presentation import (
    _band_note,
    _band_support_note,
    _basis_noun,
    _fmt_x,
    walkdown,
    walkdown_line,
)
from .surface_support import (
    COST_DOLLARS,
    COST_TOKENS,
    _BELOW_MIN_N,
    _BELOW_MIN_OVERLAP,
    _CELL_NOT_OVERLAP,
    _NO_COST_VALUE,
    _NO_MODEL_LABEL,
    _TABLE_COLUMNS,
    _TIER_ABBREV,
    _TIER_ORDER,
    _UNKNOWN_ACCEPTANCE,
    _exclusion_detail,
    _no_cost_value_detail,
    _tier_composition,
    _tier_note,
)


_SUPPORT_REBINDINGS = {
    "COST_DOLLARS": (_presentation, _support),
    "COST_TOKENS": (_presentation, _support),
    "_BELOW_MIN_N": (_estimation, _support),
    "_BELOW_MIN_OVERLAP": (_estimation, _support),
    "_CELL_NOT_OVERLAP": (_estimation, _support),
    "_FRAME_COLUMNS": (_frame,),
    "_NO_COST_VALUE": (_estimation, _support),
    "_NO_MODEL_LABEL": (_estimation, _support),
    "_TABLE_COLUMNS": (_estimation,),
    "_TIER_ABBREV": (_support,),
    "_TIER_ORDER": (_support,),
    "_UNKNOWN_ACCEPTANCE": (_estimation, _support),
    "_band_note": (_estimation,),
    "_band_support_note": (_estimation,),
    "_basis_noun": (_presentation,),
    "_bootstrap_bands": (_estimation,),
    "_coerce_token_stream": (_frame,),
    "_exclusion_detail": (_estimation,),
    "_fmt_x": (_presentation,),
    "_no_cost_value_detail": (_support,),
    "_prepare_overlap": (_estimation,),
    "_primary_kind": (_frame,),
    "_spread_of": (_estimation,),
    "_tier_composition": (_estimation, _support),
    "_tier_note": (_estimation,),
    "arm_overlap_pairs": (_estimation,),
    "naive_arm_estimates": (_estimation,),
    "np": (_bootstrap, _frame, _presentation),
    "pd": (_bootstrap, _estimation, _frame, _support),
    "shrunken_arm_estimates": (_bootstrap, _estimation),
    "spread_S": (_bootstrap,),
    "walkdown": (_presentation,),
    "within_cell_anchor": (_estimation,),
}


class _SurfaceModule(ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _SUPPORT_REBINDINGS.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _SurfaceModule
