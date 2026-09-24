"""User-facing cost-surface explanations and walkdown rendering."""

from __future__ import annotations

import numpy as np

from .surface_support import COST_DOLLARS, COST_TOKENS


def _basis_noun(cost_col: str) -> str:
    """The user-facing noun for a cost basis: what the raw numbers are made of.

    `cost_col` names the column, not the units a reader should see; this is
    the one place that translates between the two so no step note or summary
    line ever hardcodes "dollars" while actually describing a token count
    (or vice versa).
    """
    if cost_col == COST_DOLLARS:
        return "dollars"
    if cost_col == COST_TOKENS:
        return "output tokens"
    return cost_col


def _band_note(n_outside: int, n_total: int, cost_col: str) -> str:
    """Plain-language explanation for a point estimate sitting outside its own band.

    Only called when `n_outside > 0`. Says three things, in order: how many
    configurations this is happening to; that this is expected behavior for a
    percentile bootstrap on a ratio estimator, not a bug (resampling whole
    cells trims cell diversity, and by Jensen's inequality the mean of the
    resampled ratios sits above the once-computed ratio); and that the band
    is the spread of the resampling, not error bars centered on the estimate.
    Also names the table marker so a reader can connect the sentence to the
    row it is talking about.
    """
    basis = _basis_noun(cost_col)
    if n_outside == 1:
        subject = "1 of "
        verb = "has"
        pronoun = "its"
    else:
        subject = f"{n_outside} of "
        verb = "have"
        pronoun = "their"
    return (
        f"{subject}{n_total} workflow configurations {verb} a point estimate that falls "
        f"outside {pronoun} own confidence band (marked with * in the table above). That is "
        "expected here, not a mistake: the band shows the spread of the resampling, not error "
        f"bars centered on the estimate, and because {basis} per accepted run is a ratio, "
        "resampling can push the band above the single estimate computed once on the full sample."
    )


def _band_support_note(
    band_n_draws_min: int | None,
    n_boot: int,
    n_invalid_draws: int = 0,
    n_spread_draws: int | None = None,
) -> str:
    """Plain-language note naming the weakest band's resampling support.

    Called when the thinnest band rests on fewer than `n_boot` draws, when
    some configuration's resampling produced at least one invalid (NaN,
    infinite, or negative -- FIX 3, reviewer round) point estimate, or both.
    `band_n_draws_min` may be `None` (no configuration ever got a band at
    all); the sentence about it is then omitted rather than printing a
    nonsense number.

    The first sentence says how many of the resampling draws the thinnest
    band actually rests on, and why a configuration drops out of a draw at
    all: it fell below the minimum run count, or the minimum number of shared
    task-mix cells, once that draw's resample was applied. The second, only
    added when `n_invalid_draws > 0`, says how many draws (summed across
    every configuration) produced an estimate that was not a usable number at
    all and so were excluded from every band above rather than being folded
    in as a false zero.
    """
    parts = []
    if band_n_draws_min is not None:
        parts.append(
            f"The thinnest confidence band above rests on only {band_n_draws_min} of {n_boot} "
            "resampling draws. A workflow configuration drops out of a draw when that resample "
            "leaves it below the minimum run count or the minimum number of shared task-mix cells, "
            "so a thin band carries less support than a wide one might suggest."
        )
    if n_invalid_draws > 0:
        plural = "s" if n_invalid_draws != 1 else ""
        verb = "were" if n_invalid_draws != 1 else "was"
        parts.append(
            f"{n_invalid_draws} resampling draw{plural} {verb} excluded from the bands above "
            "because the resample produced a workflow-configuration estimate that was not a "
            "usable number (not finite, or negative)."
        )
    if n_spread_draws is not None and n_spread_draws < n_boot:
        parts.append(
            f"The spread band on the line above rests on {n_spread_draws} of {n_boot} draws, "
            "for the same reason: a resample that leaves too few comparable workflow "
            "configurations cannot produce a spread at all."
        )
    return " ".join(parts)


def walkdown(surface: dict) -> list[dict]:
    """The naive-spread-to-honest-spread walkdown, one step per dict.

    Four steps: the raw naive spread, the spread once every configuration is
    compared inside the same task-mix cells (mix-standardized, not yet
    pooled), the spread after partially pooling thin configurations toward
    the group (the honest headline number), and the 80% band (or whatever
    `ci` the surface was built with) around that last number. The naive
    step's note names its basis (dollars, output tokens, or whatever
    `surface["cost_col"]` is) instead of assuming dollars, since this
    function runs on both bases.

    When `surface["overlap_caveat"]` is set (no task-mix cell contained two
    comparable workflow configurations, so E0's overlap restriction fell
    back to everything unmatched), the middle two steps say so plainly
    instead of claiming a matching step ran that did not. Their `value`s
    still come from `shrunk` (the fallback model fit over every eligible
    row), which is not necessarily numerically equal to the naive step's
    value -- the note says the comparison could not be matched, not that the
    number is identical to naive's.

    FIX 4 (final reviewer round, blocker): `shrunken_arm_estimates` runs its
    full pipeline on this fallback path too, INCLUDING the James-Stein
    shrinkage step that pools thin configurations toward the group -- only
    the task-mix matching (the overlap-cell restriction) is unavailable. The
    old `honest_note` claimed "no pooling ... was possible", which is false:
    pooling happened, on an unmatched fit. Say that instead of asserting a
    step never ran when it did.
    """
    ci = surface["ci"]
    basis = _basis_noun(surface.get("cost_col"))
    caveat = surface.get("overlap_caveat")
    if caveat:
        matched_note = (
            "no task-mix cell contains two comparable workflow configurations, "
            "so no matching on task mix was possible; this number comes from "
            "the fallback fit over every eligible run instead, not a "
            "task-mix-matched comparison"
        )
        honest_note = (
            "no task-mix cell contains two comparable workflow configurations, "
            "so task-mix matching was unavailable; pooling of thin "
            "configurations still happened on this unmatched fit, so despite "
            "the step name, the result is pooled but not task-mix-adjusted"
        )
    else:
        matched_note = "spread once every workflow configuration is compared inside the same task-mix cells, before pooling"
        honest_note = "spread after partially pooling thin workflow configurations toward the group; the honest headline number"
    return [
        {
            "step": "naive spread",
            "value": surface["S_naive"],
            "note": f"raw {basis} per accepted run, best over worst, no task-mix adjustment",
        },
        {
            "step": "after matching on task mix",
            "value": surface["S_matched"],
            "note": matched_note,
        },
        {
            "step": "after pooling thin configurations",
            "value": surface["S_honest"],
            "note": honest_note,
        },
        {
            "step": f"{ci * 100:.0f}% band",
            "value": None,
            "lo": surface["S_lo"],
            "hi": surface["S_hi"],
            "note": f"{ci * 100:.0f}% bootstrap band around the honest spread, resampled by task-mix cell",
        },
    ]


def _fmt_x(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.1f}x"


def walkdown_line(surface: dict) -> str:
    """One user-facing line rendering the walkdown. No forbidden vocabulary, no em dash.

    Prefixed with the cost basis (dollars, output tokens, ...) so the line
    never implies dollars while actually reporting token counts. When the
    surface carries an `overlap_caveat` (no task-mix cell had two comparable
    workflow configurations to match on), the line says so explicitly rather
    than presenting the unmatched numbers as if they were task-mix-honest.
    """
    steps = walkdown(surface)
    naive_s, matched_s, honest_s, band = steps[0], steps[1], steps[2], steps[3]
    ci_pct = f"{surface['ci'] * 100:.0f}%"
    basis = _basis_noun(surface.get("cost_col"))
    line = (
        f"{basis} spread: {_fmt_x(naive_s['value'])} naive, "
        f"{_fmt_x(matched_s['value'])} after matching on task mix, "
        f"{_fmt_x(honest_s['value'])} after pooling thin configurations; "
        f"{ci_pct} band {_fmt_x(band['lo'])} to {_fmt_x(band['hi'])}"
    )
    if surface.get("overlap_caveat"):
        line += "; not adjusted for task mix (no shared task-mix cells)"
    return line
