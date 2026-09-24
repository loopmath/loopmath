"""Terminal renderer for `loopmath analyze` (SPEC section 6): Beats 1 to 4.

This module renders data it is handed. It never reads a log file, never
calls a parser, and never makes a network call: everything it prints comes
from `ingest_diag` (from the parsers), `coverage`/`coverage_line` (from
`loopmath.grade`), `surface` (from `loopmath.surface.cost_surface`), and
`price_warnings` (from `loopmath.price`). Ownership of wording for the coverage
line and the walkdown line stays with the modules that compute them; this
module only places them in the report.

The four beats, in the fixed order SPEC section 6 requires:

1. **What I read** -- how many session files were seen, how many parsed into
   runs, how many were skipped, per harness and in total, followed by the
   mandatory coverage line and the evidence-tier breakdown. SPEC section 0's
   full-honesty rule makes this beat mandatory: a reader must see how much of
   the corpus actually turned into gradeable evidence before seeing any cost
   number.
2. **The configuration table** -- one row per workflow configuration (model x
   effort), cost per accepted run (dollars, or output tokens in token mode,
   per `surface["cost_col"]`) with a confidence band sized to whatever `ci`
   the surface was actually built with, the naive-vs-honest walkdown line,
   the token/dollar amplifier note whenever the surface carries a
   `ratio_pair`, and any price warnings. A row whose point estimate sits
   outside its own band is marked (`*`) and explained once by `band_note`;
   `band_support_note` names the weakest band's resampling support when it
   rests on less than the full number of resampling draws. Both print only
   when `surface` sets them. Each row also carries a `tiers` column (the
   evidence-tier composition behind that row's acceptance rate, e.g.
   "31v/2h"); `tier_note`, printed once directly under the table, states
   plainly when acceptance is not measured the same way for every
   configuration (a verified row is accepted by construction) -- only when
   the fitted rows actually mix that tier with another one.
3. **What I left out** -- every excluded group named with its count. Honesty
   about exclusions is as mandatory as the coverage line: a cost number with
   silently dropped runs behind it is not a cost number, it is a guess
   wearing a decimal point.
4. **Privacy** -- the fixed local-only line, on its own.

`render` must not raise on partial or empty input; every beat degrades to a
short, honest line rather than throwing when its input is missing or empty.
"""

from __future__ import annotations

import unicodedata
import sys
from dataclasses import dataclass
from numbers import Number
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence

from . import terminal_exclusions as _terminal_exclusions
from . import terminal_format as _terminal_format
from . import terminal_read as _terminal_read
from .terminal_exclusions import (
    _DEFAULT_SKIP_REASON_VERB,
    _SKIP_REASON_VERB,
    _UNKNOWN_ACCEPTANCE_REASON,
    _counted_exclusion_detail,
    _skip_reasons_line,
)
from .terminal_format import (
    ColumnSpec,
    TableColumn,
    _cell_text,
    _ci_presentation,
    _column,
    _cost_col_presentation,
    _display_width,
    _fmt_int,
    _fmt_money,
    _fmt_pct,
    _fmt_plain_number,
    _truncate_display,
    render_table,
)
from .terminal_read import _HARNESS_ORDER, beat1_what_i_read


_SUPPORT_REBINDINGS = {
    **dict.fromkeys(("Any", "Literal", "Mapping", "Sequence"), (_terminal_format,)),
    "unicodedata": (_terminal_format,),
    "Number": (_terminal_format,),
    "_fmt_money": (_terminal_format,),
    "_fmt_pct": (_terminal_format,),
    "_fmt_plain_number": (_terminal_format,),
    "TableColumn": (_terminal_format,),
    "ColumnSpec": (_terminal_format,),
    "_display_width": (_terminal_format,),
    "_truncate_display": (_terminal_format,),
    "_column": (_terminal_format,),
    "_cell_text": (_terminal_format,),
    "_fmt_int": (_terminal_format, _terminal_read, _terminal_exclusions),
    "render_table": (_terminal_format, _terminal_read),
    "_HARNESS_ORDER": (_terminal_read,),
    "_SKIP_REASON_VERB": (_terminal_exclusions,),
    "_DEFAULT_SKIP_REASON_VERB": (_terminal_exclusions,),
}


class _TerminalModule(ModuleType):
    """Keep moved terminal helpers responsive to module rebinding."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _SUPPORT_REBINDINGS.get(name, ()):
            setattr(module, name, value)


sys.modules[__name__].__class__ = _TerminalModule

PRIVACY_LINE = "Local only. Nothing uploaded."



# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Beat 1: what I read
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Beat 2: the configuration table
# ---------------------------------------------------------------------------




def beat2_configurations(
    surface: dict,
    walkdown_line: str,
    price_warnings: list[str],
    *,
    width: int | None = None,
) -> list[str]:
    """Per-configuration cost-per-accepted-run table plus the walkdown line.

    On narrow terminals the table drops columns in this documented order:
    ``tiers``, ``runs``, ``accepted``, then the confidence ``band``. Cost and
    configuration are the two most useful comparison fields and therefore
    survive longest. If only one can fit, configuration has final priority.
    ``width`` includes this beat's two-space table indentation.
    """
    lines = ["Configurations"]

    surface = surface or {}
    table = surface.get("table")

    if table is None:
        has_rows = False
    elif hasattr(table, "empty"):
        has_rows = not table.empty
    else:
        try:
            has_rows = len(table) > 0
        except TypeError:
            has_rows = bool(table)

    cost_subtitle, cost_header, value_fmt = _cost_col_presentation(surface.get("cost_col", "usd"))
    band_label, band_header = _ci_presentation(surface.get("ci", 0.80))

    if not has_rows:
        lines.append("  no configurations met the minimum run count")
    else:
        lines.append(f"  {cost_subtitle} ({band_label})")
        lines.append("")

        sorted_table = table.sort_values("cost_per_accepted", ascending=True)
        rows = []
        for _, row in sorted_table.iterrows():
            lo = row.get("lo")
            hi = row.get("hi")
            band = (
                f"{value_fmt(lo)} to {value_fmt(hi)}"
                if lo is not None and hi is not None
                else "n/a"
            )
            # A point estimate sitting outside its own band is expected
            # behavior for a percentile bootstrap on a ratio estimator (see
            # `band_note`), not a bug; mark the row plainly (plain ASCII, no
            # unicode symbol) so the eye catches it, and let `band_note`
            # explain the marker once rather than repeating the explanation
            # per row.
            if bool(row.get("estimate_outside_band")):
                band = f"{band} *"
            # PATCH 2 (reviewer round, HIGH): each configuration's evidence-tier
            # composition (surface.py's `_tier_composition`, e.g. "31v/2h"), so
            # a reader can see which instrument produced this row's acceptance
            # rate without leaving the table. Older surfaces built before this
            # column existed (or a hand-built fixture in a test) have no
            # "tiers" key at all; that renders as "n/a" rather than a blank or
            # a raised error.
            tiers_val = row.get("tiers")
            tiers_str = tiers_val if isinstance(tiers_val, str) and tiers_val else "n/a"
            rows.append(
                {
                    "configuration": str(row.get("arm", "?")),
                    "runs": _fmt_int(row.get("n", 0)),
                    "acceptance": _fmt_pct(row.get("acc_rate")),
                    "tiers": tiers_str,
                    "cost": value_fmt(row.get("cost_per_accepted")),
                    "band": band,
                }
            )

        table_lines = render_table(
            rows,
            [
                TableColumn("configuration", "configuration", "l", 60),
                TableColumn("runs", "runs", "r", 20),
                TableColumn("acceptance", "accepted", "r", 30),
                TableColumn("tiers", "tiers", "l", 10),
                TableColumn("cost", cost_header, "r", 50),
                TableColumn("band", band_header, "l", 40),
            ],
            width=None if width is None else max(1, width - 2),
        )
        lines.extend(f"  {line}" for line in table_lines)

        # PATCH 2: one plain-language caveat, directly under the table it
        # qualifies, stating that acceptance is not measured the same way for
        # every configuration whenever the fitted rows behind this table mix
        # the verified tier (accepted by construction) with any other tier.
        # Already a complete, jargon-free sentence from surface.py; printed
        # only when that confound is actually present, never asserted here.
        tier_note = surface.get("tier_note")
        if tier_note:
            lines.append(f"  {tier_note}")

    if walkdown_line:
        lines.append("")
        lines.append(f"  {walkdown_line}")
        # A task-mix cell is the stratum surface.py's `build_frame` groups on
        # (its `cell` column: workspace paired with the dominant kind of file
        # a run wrote). "after matching on task mix" just above and "does not
        # share at least N task-mix cells" in beat 3 both name that stratum
        # without ever defining it; say it once, immediately under the line
        # that first uses it.
        #
        # PATCH 3 (reviewer round, MEDIUM): the old second clause said
        # matching compares configurations that "worked on the same kinds of
        # task" -- workspace plus dominant written-file-extension is a proxy
        # for a task, not an identification of one (SPEC's privacy contract
        # forbids reading prompt or file content, so this is the only shape
        # of the work a metadata-only parse can see). The definition clause
        # itself is unchanged; only the claim about what matching on it
        # proves is corrected.
        lines.append(
            "  A task-mix cell is one workspace paired with the main kind of "
            "file the run wrote; matching compares configurations only where "
            "they share a cell, which is a proxy for the kind of task worked "
            "on, not an identification of the task itself."
        )

    # When no task-mix cell holds two comparable configurations, the estimator
    # falls back to an unmatched fit and says so via `overlap_caveat`. That
    # caveat changes what every number above means, so it prints right under
    # them rather than down in the exclusions beat.
    caveat = surface.get("overlap_caveat")
    if caveat:
        lines.append(f"  {caveat}")

    # A point estimate can sit outside its own band (a percentile bootstrap
    # on a ratio estimator, not a bug); `band_note` says so and names the
    # table marker, and `band_support_note` names the weakest band's
    # resampling support when it used less than the full n_boot. Both are
    # already user-facing sentences from surface.py; print each only when
    # present, same indentation as `overlap_caveat` above.
    band_note = surface.get("band_note")
    if band_note:
        lines.append(f"  {band_note}")

    band_support_note = surface.get("band_support_note")
    if band_support_note:
        lines.append(f"  {band_support_note}")

    # SPEC section 5: reports must show the token ratio AND the dollar ratio
    # side by side, because a cheaper model burning more tokens can still
    # cost far more in dollars (the amplifier); dropping the line whenever
    # either half is unavailable hides exactly that amplifier. So once a
    # `ratio_pair` exists AND was actually computed (no `"unavailable"` key),
    # always print the amplifier line: render whichever half is unavailable
    # as "n/a" with a short reason rather than omitting it.
    #
    # `_extremes_ratio` (cli.py) can fail for five different reasons (fewer
    # than two ranked configurations, an extreme with no accepted runs, a NaN
    # token total, a NaN dollar total, or a non-positive dollar total); it
    # says which one by returning `{"unavailable": "<measured reason>"}`
    # instead of `None`. This module prints that reason verbatim rather than
    # asserting one of its own. `ratio_pair` itself is only fully absent (or
    # `None`) when a caller other than the CLI never supplied one at all; that
    # case gets a neutral sentence that names no cause, since none was
    # measured here.
    ratio_pair = surface.get("ratio_pair")
    unavailable_reason = ratio_pair.get("unavailable") if ratio_pair else None
    if unavailable_reason:
        lines.append(f"  {unavailable_reason}")
    elif ratio_pair:
        token_ratio = ratio_pair.get("token_ratio")
        dollar_ratio = ratio_pair.get("dollar_ratio")
        amplifier = ratio_pair.get("amplifier")
        label = ratio_pair.get("label")
        prefix = f"{label}: " if label else "cheapest vs most expensive: "

        basis = ratio_pair.get("basis")
        basis_phrase = f" {basis}" if basis else ""
        token_phrase = (
            f"{token_ratio:.1f}x the tokens{basis_phrase}"
            if token_ratio is not None
            else f"n/a tokens{basis_phrase}"
        )
        dollar_phrase = f"{dollar_ratio:.1f}x the dollars" if dollar_ratio is not None else "n/a dollars"

        if token_ratio is not None and dollar_ratio is not None:
            tail = f" ({amplifier:.1f}x amplifier)" if amplifier is not None else ""
        elif dollar_ratio is None and token_ratio is None:
            tail = " (nothing comparable)"
        elif dollar_ratio is None:
            tail = " (some runs unpriced)"
        else:
            tail = " (no token counts)"

        lines.append(f"  {prefix}{token_phrase} but {dollar_phrase}{tail}")

        # FIX 1 (reviewer round, blocker): `_extremes_ratio` (cli.py) sets
        # `basis_note` whenever a configuration's numerator/denominator had to
        # be restricted to the subset of runs with both a known token count
        # and a known dollar cost (never left silent). Printed on its own
        # indented line directly under the amplifier line it qualifies, one
        # extra indent so it reads as a footnote to that line, not a sibling
        # fact.
        basis_note = ratio_pair.get("basis_note")
        if basis_note:
            lines.append(f"    {basis_note}")
    else:
        lines.append("  the cheapest and most expensive configurations were not compared")

    for warning in price_warnings or []:
        lines.append(f"  {warning}")

    return lines


# ---------------------------------------------------------------------------
# Beat 3: named exclusions
# ---------------------------------------------------------------------------




def beat3_exclusions(surface: dict, coverage: dict, ingest_diag: dict) -> list[str]:
    """Every excluded group named with its count. Never buried, never double-counted."""
    lines = ["What I left out"]

    # entries: list of (top_line, nested_line_or_None). A nested line prints
    # indented under its parent, as a subset breakdown, never as a sibling
    # count a reader would add to the parent.
    entries: list[tuple[str, str | None]] = []

    ingest_diag = ingest_diag or {}
    skipped_total = (ingest_diag.get("skipped") or {}).get("total")
    if skipped_total:
        # The old wording named a cause ("no assistant turns or unreadable")
        # `parse_all` never measured; it now measures `skip_reasons` and this
        # prints only what was actually counted, nesting the breakdown under
        # the top line exactly like every other "of which" line in this beat,
        # and saying nothing about cause when nothing was measured.
        nested = _skip_reasons_line(ingest_diag.get("skip_reasons") or {})
        entries.append(
            (f"{_fmt_int(skipped_total)} session {'file' if skipped_total == 1 else 'files'} did not parse into a run", nested)
        )

    exclusions = (surface or {}).get("exclusions") or []
    unknown_acceptance = None
    other_exclusions = []
    for exc in exclusions:
        if exc.get("reason") == _UNKNOWN_ACCEPTANCE_REASON:
            unknown_acceptance = exc
        else:
            other_exclusions.append(exc)

    ordered_exclusions = (
        ([unknown_acceptance] if unknown_acceptance is not None else []) + other_exclusions
    )

    if ordered_exclusions:
        # The censored tier (grade.py) and each of surface.py's exclusion
        # buckets can overlap: a censored run always has an unknown
        # acceptance outcome by construction, but the first-match-wins
        # exclusion ladder in `cost_surface` may have already claimed that
        # row under "no model label" or "no usable cost value" instead. Each
        # bucket now carries its OWN measured `n_censored` (the intersection
        # of that specific bucket with the censored tier), so the "of which"
        # line nests under whichever bucket actually holds those rows,
        # never asserted against a bucket that may not contain them.
        for exc in ordered_exclusions:
            n = exc.get("n")
            detail = exc.get("detail") or exc.get("reason") or "excluded"
            n_censored = exc.get("n_censored")
            nested = (
                f"of which {_fmt_int(n_censored)} {'was' if n_censored == 1 else 'were'} censored: instant retry under "
                "60 s with no approval signal"
                if n_censored
                else None
            )
            entries.append((f"{_fmt_int(n)} {_counted_exclusion_detail(n, exc.get('reason'), detail)}", nested))
    else:
        # No exclusions at all came back from the surface, so there is no
        # bucket left to nest a censored count under. Fall back to the
        # global censored tier count (grade.py) as its own top-level line
        # rather than silently dropping it from the report; this is the one
        # remaining case where that could happen.
        censored = ((coverage or {}).get("tiers") or {}).get("censored")
        if censored:
            entries.append(
                (
                    f"{_fmt_int(censored)} {'run' if censored == 1 else 'runs'} censored: instant retry under 60 s with "
                    "no approval signal",
                    None,
                )
            )

    # FIX 5 (reviewer round, accepted in part): a run with no recorded
    # workspace cannot take part in task-mix matching at all (surface.py
    # gives it a cell that can never match another row's); it is not dropped
    # from the fit the way the ladder above is, so it is named here as its
    # own informational line rather than folded into `exclusions`.
    no_workspace_note = (surface or {}).get("no_workspace_note")
    if no_workspace_note:
        entries.append((no_workspace_note, None))

    if not entries:
        lines.append("  nothing excluded")
    else:
        widths = [len(top.split(" ", 1)[0]) for top, _nested in entries]
        w = max(widths) if widths else 0
        for top, nested in entries:
            count_str, rest = top.split(" ", 1)
            lines.append(f"  {count_str.rjust(w)} {rest}")
            if nested:
                indent = " " * (2 + w + 1)
                lines.append(f"{indent}{nested}")

    return lines


# ---------------------------------------------------------------------------
# Beat 4: privacy
# ---------------------------------------------------------------------------


def beat4_privacy() -> list[str]:
    """The fixed local-only line, alone in its own beat."""
    return [PRIVACY_LINE]


# ---------------------------------------------------------------------------
# render: the whole report
# ---------------------------------------------------------------------------


def render(
    ingest_diag: dict,
    coverage: dict,
    coverage_line: str,
    surface: dict,
    walkdown_line: str,
    price_warnings: list[str],
    *,
    width: int | None = None,
) -> str:
    """The whole report as one string, beats in order, blank line between beats.

    Defensive by design: every beat function degrades to a short honest line
    on missing or empty input rather than raising, so a caller can run this
    on a partial pipeline result (a fresh corpus, a filtered surface, a run
    with no priced records) and always get readable output.
    """
    ingest_diag = ingest_diag or {}
    coverage = coverage or {}
    surface = surface or {}
    price_warnings = price_warnings or []

    beats = [
        beat1_what_i_read(ingest_diag, coverage, coverage_line or ""),
        beat2_configurations(surface, walkdown_line or "", price_warnings, width=width),
        beat3_exclusions(surface, coverage, ingest_diag),
        beat4_privacy(),
    ]

    return "\n\n".join("\n".join(beat) for beat in beats)
