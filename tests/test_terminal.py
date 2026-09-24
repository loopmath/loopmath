"""Tests for loopmath.report.terminal (SPEC section 6): Beats 1 to 4 terminal report."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from loopmath import price as price_mod
from loopmath.cli import _extremes_ratio
from loopmath.report.terminal import (
    PRIVACY_LINE,
    TableColumn,
    beat1_what_i_read,
    beat2_configurations,
    beat3_exclusions,
    beat4_privacy,
    render,
    render_table,
)

# `_extremes_ratio` lives in cli.py, but this fix round's editable test files
# do not include a tests/test_cli.py, so its dedicated tests live here
# instead, alongside the terminal.py rendering tests for what it returns.

FORBIDDEN_PHRASES = [
    "knowledge gradient",
    "posterior",
    "prior",
    "experimental design",
    "value of information",
    "bandit",
]


def _fake_ingest_diag():
    return {
        "files_seen": {"claude-code": 6218, "codex": 4158, "total": 10376},
        "records": {"claude-code": 5104, "codex": 3820, "total": 8924},
        "skipped": {"claude-code": 1114, "codex": 338, "total": 1452},
        "cache_hits": 2103,
        # FIX 2: measured reasons `parse_all` classified skipped files under,
        # not an asserted "no assistant turns or unreadable" guess. Sums to
        # the 1,452 skipped total above.
        "skip_reasons": {"no assistant turns": 1200, "unreadable or empty": 252},
    }


def _fake_coverage():
    tiers = {
        "verified": 5210,
        "reported": 1802,
        "heuristic": 1090,
        "asserted": 402,
        "censored": 420,
        "ungraded": 0,
    }
    return {
        "n_total": 8924,
        "n_graded": 8102,
        "pct_structural": 90.8,
        "tiers": tiers,
        "n_accepted": 7000,
        "n_rejected": 1102,
        "n_unknown": 402,
        "denominator": 8102,
    }


def _fake_coverage_line():
    return (
        "graded 8102 of 8924 runs (90.8% structural); tiers: "
        "verified 5210 / reported 1802 / heuristic 1090 / asserted 402 / censored 420"
    )


def _fake_surface():
    table = pd.DataFrame(
        [
            {
                "arm": "opus-5-xhigh",
                "n": 120,
                "n_overlap_cells": 4,
                "acc_rate": 0.84,
                "shrink": 0.3,
                "cost_per_accepted": 12.34,
                "lo": 9.10,
                "hi": 15.80,
                "naive_cost_per_accepted": 13.0,
            },
            {
                "arm": "sonnet-5-medium",
                "n": 340,
                "n_overlap_cells": 6,
                "acc_rate": 0.79,
                "shrink": 0.5,
                "cost_per_accepted": 1204.0,
                "lo": 980.0,
                "hi": 1450.0,
                "naive_cost_per_accepted": 1300.0,
            },
        ]
    )
    return {
        "table": table,
        "cost_basis": "dollars per accepted run",
        "exclusions": [
            {
                "reason": "unpriced",
                "n": 882,
                "detail": "runs unpriced: no price entry for the model",
                # FIX 1: the censored count is measured per bucket, not taken
                # from the global tier count, so a fixture that wants a
                # nested "of which" line has to say which bucket it is in.
                "n_censored": 420,
            },
            {
                "reason": "small_configuration",
                "n": 140,
                "detail": "runs in configurations with fewer than 5 runs",
                "n_censored": 0,
            },
        ],
        "ratio_pair": {
            "token_ratio": 3.1,
            "dollar_ratio": 10.4,
            "amplifier": 3.4,
            "label": "cheapest vs most expensive",
        },
        "band_note": (
            "1 of 2 workflow configurations have a point estimate that falls outside "
            "their own confidence band (marked with * in the table above). That is "
            "expected here, not a mistake: the band shows the spread of the "
            "resampling, not error bars centered on the estimate, and because the "
            "measure here is a ratio, resampling can push the band above the single "
            "estimate computed once on the full sample."
        ),
        "band_support_note": (
            "The thinnest confidence band above rests on only 940 of 1000 resampling "
            "draws. A workflow configuration drops out of a draw when that resample "
            "leaves it below the minimum run count or the minimum number of shared "
            "task-mix cells, so a thin band carries less support than a wide one "
            "might suggest."
        ),
    }


def _fake_walkdown_line():
    return "naive spread 8.2x; honest, cell-adjusted spread 5.6x"


def _fake_price_warnings():
    return [
        "WARNING: 12 runs priced from placeholder rates (sonnet-5: 12).",
        "Price table as of 2026-08-31.",
    ]


def _full_render():
    return render(
        ingest_diag=_fake_ingest_diag(),
        coverage=_fake_coverage(),
        coverage_line=_fake_coverage_line(),
        surface=_fake_surface(),
        walkdown_line=_fake_walkdown_line(),
        price_warnings=_fake_price_warnings(),
    )


# ---------------------------------------------------------------------------
# 1. All four beats present, in order
# ---------------------------------------------------------------------------


def test_render_has_all_four_beats_in_order():
    out = _full_render()
    i1 = out.find("What I read")
    i2 = out.find("Configurations")
    i3 = out.find("What I left out")
    i4 = out.find(PRIVACY_LINE)
    assert i1 != -1 and i2 != -1 and i3 != -1 and i4 != -1
    assert i1 < i2 < i3 < i4


def test_render_contains_privacy_line():
    out = _full_render()
    assert PRIVACY_LINE in out
    assert PRIVACY_LINE == "Local only. Nothing uploaded."


# New strings introduced by the reviewer fix round (FIXES 1-4) that are not
# reachable through `_full_render()`'s default fixture (its `ratio_pair` is
# always the successful, computed kind, never the "unavailable" kind). Swept
# below so a new user-facing string can never skip the no-em-dash /
# no-forbidden-vocabulary checks just because the default fixture never
# happens to trigger it.
_EXTRA_USER_FACING_STRINGS = [
    "the cheapest and most expensive configurations were not compared",
    "no workflow configuration met the minimum run count, so there is "
    "nothing to compare",
    "only one workflow configuration met the minimum run count, so there is "
    "nothing to compare",
    "the cheapest or most expensive configuration has no accepted run, so a "
    "per-accepted comparison is not defined",
    "the token total for one of the two configurations is unknown, so the "
    "comparison would be guesswork",
    "the dollar total for one of the two configurations is unknown, so the "
    "comparison would be guesswork",
    "not classified (skip count over the 1000-file classification cap)",
    "A task-mix cell is one workspace paired with the main kind of file the "
    "run wrote; matching compares configurations only where they worked on "
    "the same kinds of task.",
    # Final reviewer fix round additions, not reachable through
    # `_full_render()`'s default fixture:
    # FIX 1: the new distinguishable reason when neither side is wholly
    # unknown but no accepted run has both known together, and a
    # representative `basis_note`.
    "no accepted run in one of the two configurations has both a known "
    "token count and a known dollar cost, so the comparison would be "
    "guesswork",
    "computed on 40 of 62 runs considered (dear.high) where both the token "
    "count and the dollar cost are known",
    # FIX 2: the "what --limit cut" line.
    "--limit 50 per harness; 4,200 more files were found but not read "
    "because of it (raise --limit, or drop it, to read them)",
    # FIX 3: the band-support note's new invalid-draws sentence, singular and
    # plural.
    "1 resampling draw was excluded from the bands above because the "
    "resample produced a workflow-configuration estimate that was not a "
    "usable number (not finite, or negative).",
    "37 resampling draws were excluded from the bands above because the "
    "resample produced a workflow-configuration estimate that was not a "
    "usable number (not finite, or negative).",
    # FIX 5: the no-recorded-workspace note, singular and plural.
    "1 run has no recorded workspace, so it cannot take part in task-mix "
    "matching",
    "2 runs have no recorded workspace, so they cannot take part in "
    "task-mix matching",
    # Final reviewer fix round (LAST round), FIX 1: the row-based basis_note
    # wording, fired by a rejected attempt missing a value, not just an
    # accepted one.
    "computed on 2 of 3 runs considered (dear.high) where both the token "
    "count and the dollar cost are known",
    # FIX 2: the new row-level exclusion reason's detail text (a
    # configuration retained by the overlap requirement, but this row's own
    # cell was not one of the shared cells).
    "run is in a task-mix cell no other comparable workflow configuration "
    "worked in",
    # FIX 3: the corrected no-overlap caveat (the numbers are fitted,
    # shrunken cost-per-accepted estimates, never raw per-run costs).
    "No task-mix cell contains two comparable workflow configurations, so "
    "task-mix matching was not possible. These are cost-per-accepted "
    "estimates fitted without matching, not a like-for-like comparison.",
    # FIX 4: the corrected walkdown honest-step note (pooling did happen on
    # the unmatched fit; only the task-mix matching was unavailable).
    "no task-mix cell contains two comparable workflow configurations, so "
    "task-mix matching was unavailable; pooling of thin configurations "
    "still happened on this unmatched fit, so despite the step name, the "
    "result is pooled but not task-mix-adjusted",
]


# ---------------------------------------------------------------------------
# 2. No em-dash
# ---------------------------------------------------------------------------


def test_no_em_dash_or_en_dash():
    out = _full_render()
    assert "—" not in out  # em-dash
    assert "–" not in out  # en-dash
    for s in _EXTRA_USER_FACING_STRINGS:
        assert "—" not in s, f"em-dash found in {s!r}"
        assert "–" not in s, f"en-dash found in {s!r}"


# ---------------------------------------------------------------------------
# 3. No forbidden vocabulary
# ---------------------------------------------------------------------------


def test_no_forbidden_vocabulary():
    out = _full_render().lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in out, f"forbidden phrase found: {phrase!r}"
    assert re.search(r"\barms\b", out) is None
    for s in _EXTRA_USER_FACING_STRINGS:
        low = s.lower()
        for phrase in FORBIDDEN_PHRASES:
            assert phrase not in low, f"forbidden phrase found in {s!r}: {phrase!r}"
        assert re.search(r"\barms\b", low) is None, f"forbidden word 'arms' found in {s!r}"
    # sanity: near-miss words that share substrings must not false-positive
    # against the regex itself (this asserts the regex behavior we rely on,
    # not the rendered output).
    assert re.search(r"\barms\b", "the harness warms up") is None


# ---------------------------------------------------------------------------
# 4. Money and percent formatting
# ---------------------------------------------------------------------------


def test_money_and_percent_formatting():
    out = _full_render()
    assert "$12.34" in out
    assert "$1,204" in out
    assert "84%" in out


# ---------------------------------------------------------------------------
# 5. Graceful degradation
# ---------------------------------------------------------------------------


def test_render_empty_surface_does_not_raise():
    out = render(
        ingest_diag=_fake_ingest_diag(),
        coverage=_fake_coverage(),
        coverage_line=_fake_coverage_line(),
        surface={},
        walkdown_line="",
        price_warnings=[],
    )
    assert "no configurations met the minimum run count" in out
    assert PRIVACY_LINE in out


def test_render_empty_dataframe_table_does_not_raise():
    surface = {"table": pd.DataFrame(), "exclusions": []}
    out = render(
        ingest_diag=_fake_ingest_diag(),
        coverage=_fake_coverage(),
        coverage_line=_fake_coverage_line(),
        surface=surface,
        walkdown_line="walkdown unavailable",
        price_warnings=[],
    )
    assert "no configurations met the minimum run count" in out


def test_render_zero_records_does_not_raise():
    empty_diag = {
        "files_seen": {"claude-code": 0, "codex": 0, "total": 0},
        "records": {"claude-code": 0, "codex": 0, "total": 0},
        "skipped": {"claude-code": 0, "codex": 0, "total": 0},
        "cache_hits": 0,
    }
    empty_coverage = {
        "n_total": 0,
        "n_graded": 0,
        "pct_structural": 0.0,
        "tiers": {t: 0 for t in ("verified", "reported", "heuristic", "asserted", "censored")},
        "n_accepted": 0,
        "n_rejected": 0,
        "n_unknown": 0,
        "denominator": 0,
    }
    out = render(
        ingest_diag=empty_diag,
        coverage=empty_coverage,
        coverage_line="graded 0 of 0 runs (0.0% structural); tiers: verified 0 / reported 0 / heuristic 0 / asserted 0 / censored 0",
        surface={},
        walkdown_line="",
        price_warnings=[],
    )
    assert PRIVACY_LINE in out


def test_render_ratio_pair_none_shows_neutral_not_compared_sentence():
    # FIX 3 (reviewer round): the reviewer flagged the old version of this
    # test as vacuous -- its own fixture (`_fake_surface()`) has TWO
    # configurations, yet the assertion demanded the "only one configuration"
    # sentence. An absent (or None) `ratio_pair` means no caller supplied one
    # at all, which is not the same fact as "only one configuration exists";
    # this must print a neutral sentence that names no cause, since none was
    # measured here.
    surface = _fake_surface()
    del surface["ratio_pair"]
    out = render(
        ingest_diag=_fake_ingest_diag(),
        coverage=_fake_coverage(),
        coverage_line=_fake_coverage_line(),
        surface=surface,
        walkdown_line=_fake_walkdown_line(),
        price_warnings=[],
    )
    assert "amplifier" not in out.lower()
    assert "the cheapest and most expensive configurations were not compared" in out
    assert "only one" not in out.lower()


def test_render_ratio_pair_value_none_also_shows_neutral_sentence():
    # A caller that sets the key to `None` explicitly (rather than omitting
    # it) must render identically to an absent key, not raise or print a
    # blank line. (`_extremes_ratio` itself no longer returns `None` at all
    # after the FIX 3 change; this exercises terminal.py's own defensiveness
    # against a caller that still might.)
    surface = _fake_surface()
    surface["ratio_pair"] = None
    out = render(
        ingest_diag=_fake_ingest_diag(),
        coverage=_fake_coverage(),
        coverage_line=_fake_coverage_line(),
        surface=surface,
        walkdown_line=_fake_walkdown_line(),
        price_warnings=[],
    )
    assert "the cheapest and most expensive configurations were not compared" in out


def test_beat2_ratio_pair_unavailable_zero_configurations_prints_that_reason():
    # `_extremes_ratio` (cli.py) distinguishes the zero-configuration case
    # from the one-configuration case; terminal.py must print whichever one
    # it was actually given, verbatim.
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "unavailable": (
            "no workflow configuration met the minimum run count, so there "
            "is nothing to compare"
        )
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert (
        "no workflow configuration met the minimum run count, so there is "
        "nothing to compare"
    ) in text
    assert "amplifier" not in text.lower()


def test_beat2_ratio_pair_unavailable_one_configuration_prints_that_reason():
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "unavailable": (
            "only one workflow configuration met the minimum run count, so "
            "there is nothing to compare"
        )
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert (
        "only one workflow configuration met the minimum run count, so "
        "there is nothing to compare"
    ) in text


def test_beat2_ratio_pair_unavailable_no_accepted_run_prints_that_reason():
    # Two-or-more configurations ranked (this fixture's table has two rows),
    # but the ratio still could not be computed: an extreme configuration had
    # no accepted run. A distinct measured reason, never the one-
    # configuration sentence.
    surface = _fake_surface()
    assert len(surface["table"]) >= 2
    surface["ratio_pair"] = {
        "unavailable": (
            "the cheapest or most expensive configuration has no accepted "
            "run, so a per-accepted comparison is not defined"
        )
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "has no accepted run" in text
    assert "only one" not in text.lower()


def test_beat2_ratio_pair_unavailable_token_total_unknown_prints_that_reason():
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "unavailable": (
            "the token total for one of the two configurations is unknown, "
            "so the comparison would be guesswork"
        )
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "the token total for one of the two configurations is unknown" in text


def test_beat2_ratio_pair_unavailable_dollar_total_unknown_prints_that_reason():
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "unavailable": (
            "the dollar total for one of the two configurations is unknown, "
            "so the comparison would be guesswork"
        )
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "the dollar total for one of the two configurations is unknown" in text


def test_beat2_ratio_pair_with_missing_dollar_ratio_is_still_visible():
    # Flag 4 fix: SPEC section 5 requires token ratio AND dollar ratio side
    # by side; a ratio_pair that is present but only half-known must still
    # print a line (with the missing half as "n/a"), never disappear.
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "token_ratio": 3.1,
        "dollar_ratio": None,
        "amplifier": None,
        "label": "cheapest vs most expensive",
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "n/a" in text
    assert "3.1" in text
    assert "some runs unpriced" in text


def test_beat2_ratio_pair_with_missing_token_ratio_is_still_visible():
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "token_ratio": None,
        "dollar_ratio": 10.4,
        "amplifier": None,
        "label": "cheapest vs most expensive",
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "n/a" in text
    assert "10.4" in text
    assert "no token counts" in text


def test_beat2_ratio_pair_with_both_ratios_missing_is_still_visible():
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "token_ratio": None,
        "dollar_ratio": None,
        "amplifier": None,
        "label": "cheapest vs most expensive",
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert text.count("n/a") >= 2
    assert "nothing comparable" in text


def test_render_empty_exclusions_list():
    lines = beat3_exclusions(
        surface={"exclusions": []},
        coverage={"tiers": {"censored": 0}},
        ingest_diag={"skipped": {"total": 0}},
    )
    assert any("nothing excluded" in line for line in lines)


def test_render_totally_empty_inputs_does_not_raise():
    out = render(
        ingest_diag={},
        coverage={},
        coverage_line="",
        surface={},
        walkdown_line="",
        price_warnings=[],
    )
    assert PRIVACY_LINE in out


# ---------------------------------------------------------------------------
# 6. Exclusions beat names every entry with its count
# ---------------------------------------------------------------------------


def test_exclusions_beat_names_every_entry():
    surface = _fake_surface()
    lines = beat3_exclusions(
        surface=surface,
        coverage=_fake_coverage(),
        ingest_diag=_fake_ingest_diag(),
    )
    text = "\n".join(lines)
    for exc in surface["exclusions"]:
        assert str(exc["n"]) in text
        assert exc["detail"] in text
    # censored count (measured per bucket via n_censored, FIX 1) and
    # skipped-file count are also named
    assert "of which 420 were censored" in text
    assert "1,452" in text or "1452" in text


# ---------------------------------------------------------------------------
# 7. render_table alignment
# ---------------------------------------------------------------------------


def test_render_table_left_and_right_alignment():
    rows = [
        {"name": "a", "n": "3"},
        {"name": "bbbbb", "n": "1200"},
    ]
    columns = [("name", "name", "l"), ("n", "n", "r")]
    lines = render_table(rows, columns)
    assert len(lines) == 3  # header + 2 rows
    header, row1, row2 = lines

    name_w = max(len("name"), len("a"), len("bbbbb"))
    n_w = max(len("n"), len("3"), len("1200"))
    expected_header = "  ".join(["name".ljust(name_w), "n".rjust(n_w)])
    expected_row1 = "  ".join(["a".ljust(name_w), "3".rjust(n_w)])
    expected_row2 = "  ".join(["bbbbb".ljust(name_w), "1200".rjust(n_w)])

    assert header == expected_header
    assert row1 == expected_row1
    assert row2 == expected_row2
    # left-aligned column starts immediately at the value; right-aligned
    # column ends flush with the widest value in that column.
    assert row1.startswith("a")
    assert row2.startswith("bbbbb")
    assert row1.endswith("3".rjust(n_w))
    assert row2.endswith("1200")


def test_render_table_empty_rows():
    columns = [("a", "A", "l"), ("b", "B", "r")]
    lines = render_table([], columns)
    assert lines == ["A  B"]


def test_render_table_no_columns():
    assert render_table([{"a": 1}], []) == []


def test_render_table_auto_aligns_numbers_and_uses_unicode_cell_width():
    rows = [
        {"name": "界", "n": 3},
        {"name": "e\u0301", "n": 120},
    ]
    lines = render_table(
        rows,
        [TableColumn("name", "name"), TableColumn("n", "n")],
    )
    assert lines == ["name    n", "界      3", "e\u0301     120"]


def test_render_table_narrow_width_drops_lowest_priority_without_wrapping():
    rows = [{"name": "alpha", "count": 7, "note": "lowest first"}]
    columns = [
        TableColumn("name", "name", priority=100),
        TableColumn("count", "count", priority=50),
        TableColumn("note", "note", priority=0),
    ]

    lines = render_table(rows, columns, width=14)

    assert lines == ["name   count", "alpha      7"]
    assert all(len(line) <= 14 for line in lines)
    assert render_table(rows, columns, width=4) == ["name", "alp…"]


def test_configuration_table_narrow_width_uses_documented_drop_order():
    lines = beat2_configurations(_fake_surface(), "", [], width=50)
    header = next(line for line in lines if line.strip().startswith("configuration"))
    table_lines = [header] + [line for line in lines if "opus-5-xhigh" in line or "sonnet-5-medium" in line]

    assert "tiers" not in header
    assert "runs" not in header
    assert "accepted" not in header.split()
    assert "$/accepted" in header
    assert "80% band" in header
    assert all(len(line) <= 50 for line in table_lines)


def test_default_width_terminal_report_matches_golden():
    golden = Path(__file__).with_name("golden") / "terminal_default.txt"
    assert _full_render() + "\n" == golden.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Additional beat-level sanity checks
# ---------------------------------------------------------------------------


def test_beat1_mentions_harnesses_and_coverage_line():
    lines = beat1_what_i_read(_fake_ingest_diag(), _fake_coverage(), _fake_coverage_line())
    text = "\n".join(lines)
    assert "claude-code" in text
    assert "codex" in text
    assert "6,218" in text or "6218" in text
    assert _fake_coverage_line() in text


def test_beat1_names_the_time_window_and_what_it_excluded():
    # SPEC section 0: exclusions print. A default window that quietly keeps
    # most of the local estate out is exactly the kind of hidden exclusion
    # that makes a coverage number look better than it is.
    diag = dict(_fake_ingest_diag())
    diag["since_days"] = 14.0
    diag["files_outside_window"] = 5188
    text = "\n".join(beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line()))
    assert "last 14 days" in text
    assert "5,188" in text
    assert "--all" in text


def test_beat1_day_word_handles_integer_and_fractional_windows():
    diag = dict(_fake_ingest_diag())
    diag["since_days"] = 1.0
    lines = beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line())
    assert next(line for line in lines if "last" in line).endswith("last 1 day")

    diag["since_days"] = 2.0
    lines = beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line())
    assert next(line for line in lines if "last" in line).endswith("last 2 days")

    diag["since_days"] = 0.5
    lines = beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line())
    assert next(line for line in lines if "last" in line).endswith("last 0.5 days")


def test_beat1_count_words_handle_one():
    diag = {
        "files_seen": {"claude-code": 1, "total": 1},
        "records": {"claude-code": 1, "total": 1},
        "skipped": {"claude-code": 0, "total": 0},
        "cache_hits": 0,
        "ocp_documents": 1,
        "ocp_attempts": 1,
        "since_days": 1,
        "files_outside_window": 1,
        "limit": 1,
        "files_omitted_by_limit": {"claude-code": 1, "total": 1},
        "zero_token_synthetic": 1,
    }
    text = "\n".join(beat1_what_i_read(diag, {}, ""))
    for phrase in (
        "1 session file, 1 parsed",
        "1 OCP document(s), 1 attempt imported",
        "1 older file was not read",
        "1 more file was found but not read because of it "
        "(raise --limit, or drop it, to read it)",
        "1 parsed session carried",
    ):
        assert phrase in text


def test_cli_counted_noun_handles_one_and_many():
    from loopmath.cli import _counted_noun

    assert _counted_noun(1, "session file") == "1 session file"
    assert _counted_noun(2, "session file") == "2 session files"


def test_beat1_says_so_when_no_window_was_applied():
    diag = dict(_fake_ingest_diag())
    diag["since_days"] = None
    text = "\n".join(beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line()))
    assert "every log file" in text
    assert "were not read" not in text


def test_beat2_includes_walkdown_line_and_price_warnings():
    lines = beat2_configurations(_fake_surface(), _fake_walkdown_line(), _fake_price_warnings())
    text = "\n".join(lines)
    assert _fake_walkdown_line() in text
    for warning in _fake_price_warnings():
        assert warning in text
    assert "80% confidence band" in text


def test_beat4_privacy_is_exact():
    assert beat4_privacy() == [PRIVACY_LINE]


# ---------------------------------------------------------------------------
# 8. Flag 1: token mode never says dollars or prints $
# ---------------------------------------------------------------------------


def test_beat2_token_mode_has_no_dollar_sign():
    surface = _fake_surface()
    surface["cost_col"] = "out_tokens"
    lines = beat2_configurations(surface, _fake_walkdown_line(), _fake_price_warnings())
    text = "\n".join(lines)
    assert "$" not in text
    assert "output tokens per accepted run" in text
    assert "tokens/accepted" in text


def test_beat2_dollar_mode_says_dollars_and_uses_dollar_column():
    surface = _fake_surface()
    surface["cost_col"] = "usd"
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "dollars per accepted run" in text
    assert "$/accepted" in text
    assert "$12.34" in text


def test_beat2_unrecognised_cost_col_names_itself_plainly():
    surface = _fake_surface()
    surface["cost_col"] = "widgets"
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "$" not in text
    assert "widgets per accepted run" in text
    assert "widgets/accepted" in text


# ---------------------------------------------------------------------------
# 9. Flag 2: the confidence band label follows the actual `ci`
# ---------------------------------------------------------------------------


def test_beat2_ci_90_percent_renders_in_subtitle_and_header():
    surface = _fake_surface()
    surface["ci"] = 0.90
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "90% confidence band" in text
    assert "90% band" in text
    assert "80%" not in text


def test_beat2_ci_none_renders_width_not_recorded():
    surface = _fake_surface()
    surface["ci"] = None
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "confidence band (width not recorded)" in text
    assert "80%" not in text
    assert "90%" not in text


def test_beat2_ci_out_of_range_renders_width_not_recorded():
    surface = _fake_surface()
    surface["ci"] = 1.4  # not a fraction between 0 and 1: do not assert a width
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "confidence band (width not recorded)" in text


def test_beat2_ci_default_is_80_percent_when_key_absent():
    # surface.get("ci", 0.80): an ordinary surface that never mentions `ci`
    # (the common case; `cost_surface`'s own default is 0.80) still gets the
    # familiar 80% wording, not the width-not-recorded fallback.
    surface = _fake_surface()
    assert "ci" not in surface
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "80% confidence band" in text
    assert "80% band" in text


# ---------------------------------------------------------------------------
# 10. Flag 3: censored runs are never counted twice in beat 3
# ---------------------------------------------------------------------------


def test_beat3_censored_nests_under_unknown_acceptance_when_present():
    # FIX 1: the nested count comes from the exclusion entry's OWN measured
    # `n_censored`, never from the global coverage tier count (which may
    # include censored rows some other bucket already claimed).
    surface = {
        "exclusions": [
            {
                "reason": "unknown acceptance",
                "n": 882,
                "detail": "run has an unknown acceptance outcome, so it cannot inform a cost-per-accepted-run number",
                "n_censored": 420,
            },
        ],
    }
    coverage = {"tiers": {"censored": 420}}
    lines = beat3_exclusions(surface=surface, coverage=coverage, ingest_diag={})
    text = "\n".join(lines)

    # Named exactly once each: a reader must never be able to sum "882" and
    # "420" as if they were disjoint sets.
    assert text.count("882") == 1
    assert text.count("420") == 1

    top_idx = next(i for i, line in enumerate(lines) if "882" in line)
    nested_idx = next(i for i, line in enumerate(lines) if "420" in line)
    assert nested_idx == top_idx + 1
    assert "of which" in lines[nested_idx]
    assert "censored" in lines[nested_idx]

    top_indent = len(lines[top_idx]) - len(lines[top_idx].lstrip(" "))
    nested_indent = len(lines[nested_idx]) - len(lines[nested_idx].lstrip(" "))
    assert nested_indent > top_indent


def test_beat3_censored_stays_top_level_without_unknown_acceptance_exclusion():
    # FIX 1: this fallback now fires only when the surface returned NO
    # exclusions at all (there is then no bucket left to nest a censored
    # count under), not merely "no unknown-acceptance bucket present".
    surface = {"exclusions": []}
    coverage = {"tiers": {"censored": 420}}
    lines = beat3_exclusions(surface=surface, coverage=coverage, ingest_diag={})
    text = "\n".join(lines)
    assert text.count("420") == 1
    assert "of which" not in text
    assert any("runs censored" in line for line in lines)


def test_beat3_n_censored_nests_under_each_bucket_that_actually_holds_them():
    # FIX 1 (blocker): the reviewer's complaint. Censored runs can land in
    # more than one exclusion bucket (whichever reason claims a row first in
    # surface.py's first-match-wins ladder); each bucket must print its OWN
    # "of which" line from its OWN measured `n_censored`, never a single
    # global count nested under just one of them.
    surface = {
        "exclusions": [
            {
                "reason": "no model label",
                "n": 1526,
                "detail": "run has no model label, so it cannot be placed in any workflow configuration",
                "n_censored": 900,
            },
            {
                "reason": "unknown acceptance",
                "n": 2262,
                "detail": "run has an unknown acceptance outcome, so it cannot inform a cost-per-accepted-run number",
                "n_censored": 1010,
            },
            {
                "reason": "configuration below min_n",
                "n": 8,
                "detail": "fewer than 5 runs in this workflow configuration",
                "n_censored": 0,
            },
        ],
    }
    lines = beat3_exclusions(surface=surface, coverage={"tiers": {"censored": 1910}}, ingest_diag={})
    text = "\n".join(lines)

    # Both nested counts appear, each once, and neither equals the global
    # tier total (1,910) -- that total is never printed here at all, because
    # it would double count rows already split across two buckets.
    assert text.count("900") == 1
    assert text.count("1,010") == 1 or text.count("1010") == 1
    assert "1,910" not in text and "1910" not in text
    assert text.count("of which") == 2

    # Each nested count, read back off the rendered text itself (not the
    # input fixture), is strictly a subset of its own parent's count.
    def _leading_int(line: str) -> int:
        return int(line.strip().split(" ", 1)[0].replace(",", ""))

    def _first_int_after(line: str, marker: str) -> int:
        rest = line.split(marker, 1)[1]
        return int(rest.strip().split(" ", 1)[0].replace(",", ""))

    top_1526_idx = next(i for i, line in enumerate(lines) if "1,526" in line)
    nested_900_idx = next(i for i, line in enumerate(lines) if "900" in line)
    assert nested_900_idx == top_1526_idx + 1
    assert _first_int_after(lines[nested_900_idx], "of which") <= _leading_int(lines[top_1526_idx])

    top_2262_idx = next(i for i, line in enumerate(lines) if "2,262" in line)
    nested_1010_idx = next(i for i, line in enumerate(lines) if "1,010" in line or "1010" in line)
    assert nested_1010_idx == top_2262_idx + 1
    assert _first_int_after(lines[nested_1010_idx], "of which") <= _leading_int(lines[top_2262_idx])

    # The zero-`n_censored` bucket gets no nested line at all.
    below_min_n_idx = next(i for i, line in enumerate(lines) if "fewer than 5 runs" in line)
    assert below_min_n_idx == len(lines) - 1  # last entry, nothing nested after it


def test_beat3_n_censored_absent_key_prints_no_nested_line():
    # An older caller whose exclusion dicts predate this fix (no `n_censored`
    # key at all) must render with no nested line for that bucket, never
    # fall back to guessing from the global tier count.
    surface = {
        "exclusions": [
            {
                "reason": "unknown acceptance",
                "n": 2262,
                "detail": "run has an unknown acceptance outcome, so it cannot inform a cost-per-accepted-run number",
            },
        ],
    }
    lines = beat3_exclusions(surface=surface, coverage={"tiers": {"censored": 1910}}, ingest_diag={})
    text = "\n".join(lines)
    assert "of which" not in text
    assert "1,910" not in text and "1910" not in text


@pytest.mark.parametrize(
    ("reason", "detail", "singular", "plural"),
    [
        ("no model label", "run has no model label, so it cannot be placed in any workflow configuration", "1 run has no model label, so it cannot", "2 runs have no model label, so they cannot"),
        ("no usable cost value", "run could not be priced, so it has no dollar cost and is never counted as zero", "1 run could not be priced, so it has", "2 runs could not be priced, so they have no dollar cost and are never"),
        ("unknown acceptance", "run has an unknown acceptance outcome, so it cannot inform a cost-per-accepted-run number", "1 run has an unknown acceptance outcome", "2 runs have an unknown acceptance outcome, so they cannot"),
        ("configuration below min_n", "fewer than 5 runs in this workflow configuration", "1 run belongs to a workflow configuration with fewer than 5 runs", "2 runs belong to workflow configurations with fewer than 5 runs"),
        ("configuration below min_overlap_cells", "workflow configuration does not share at least 2 task-mix cells with another eligible configuration", "1 run belongs to a workflow configuration that does not share", "2 runs belong to workflow configurations that do not share"),
        ("run's task-mix cell not shared with another configuration", "run is in a task-mix cell no other comparable workflow configuration worked in", "1 run is in a task-mix cell", "2 runs are in a task-mix cell"),
    ],
)
def test_beat3_exclusion_grammar_tracks_count(reason, detail, singular, plural):
    def rendered(n):
        surface = {"exclusions": [{"reason": reason, "n": n, "detail": detail}]}
        return "\n".join(beat3_exclusions(surface, {}, {}))

    assert singular in rendered(1)
    assert plural in rendered(2)


# ---------------------------------------------------------------------------
# FIX 2: skip_reasons rendering
# ---------------------------------------------------------------------------


def test_beat3_skip_reasons_render_descending_order_under_the_skip_line():
    ingest_diag = {
        "skipped": {"total": 20},
        "skip_reasons": {"no assistant turns": 18, "unreadable or empty": 2},
    }
    lines = beat3_exclusions(surface={}, coverage={}, ingest_diag=ingest_diag)
    text = "\n".join(lines)
    assert "20 session files did not parse into a run" in text
    assert "(no assistant turns or unreadable)" not in text  # no more asserted cause
    assert "of which 18 had no assistant turns, 2 were unreadable or empty" in text

    top_idx = next(i for i, line in enumerate(lines) if "did not parse into a run" in line)
    nested_idx = next(i for i, line in enumerate(lines) if "of which" in line)
    assert nested_idx == top_idx + 1


def test_beat3_one_skipped_session_uses_singular_file():
    lines = beat3_exclusions(surface={}, coverage={}, ingest_diag={"skipped": {"total": 1}})
    assert lines == ["What I left out", "  1 session file did not parse into a run"]


def test_beat3_one_censored_subset_uses_was():
    surface = {
        "exclusions": [{
            "reason": "unknown acceptance",
            "n": 2,
            "detail": "run has an unknown acceptance outcome",
            "n_censored": 1,
        }]
    }
    lines = beat3_exclusions(surface=surface, coverage={}, ingest_diag={})
    assert lines[-1].strip() == "of which 1 was censored: instant retry under 60 s with no approval signal"


def test_beat3_one_fallback_censored_run_uses_singular_run():
    lines = beat3_exclusions(surface={}, coverage={"tiers": {"censored": 1}}, ingest_diag={})
    assert lines == [
        "What I left out",
        "  1 run censored: instant retry under 60 s with no approval signal",
    ]


def test_beat3_one_default_skip_reason_uses_was():
    ingest_diag = {
        "skipped": {"total": 1},
        "skip_reasons": {"unreadable or empty": 1},
    }
    lines = beat3_exclusions(surface={}, coverage={}, ingest_diag=ingest_diag)
    assert lines[-1].strip() == "of which 1 was unreadable or empty"


def test_beat3_skip_reasons_missing_prints_top_line_with_no_cause():
    ingest_diag = {"skipped": {"total": 20}}  # no skip_reasons key at all
    lines = beat3_exclusions(surface={}, coverage={}, ingest_diag=ingest_diag)
    text = "\n".join(lines)
    assert "20 session files did not parse into a run" in text
    assert "of which" not in text
    assert "no assistant turns" not in text
    assert "unreadable" not in text


def test_beat3_skip_reasons_empty_dict_prints_top_line_with_no_cause():
    ingest_diag = {"skipped": {"total": 20}, "skip_reasons": {}}
    lines = beat3_exclusions(surface={}, coverage={}, ingest_diag=ingest_diag)
    text = "\n".join(lines)
    assert "20 session files did not parse into a run" in text
    assert "of which" not in text


def test_beat3_skip_reasons_over_classification_cap_reason_renders():
    ingest_diag = {
        "skipped": {"total": 1200},
        "skip_reasons": {
            "no assistant turns": 1000,
            "not classified (skip count over the 1000-file classification cap)": 200,
        },
    }
    lines = beat3_exclusions(surface={}, coverage={}, ingest_diag=ingest_diag)
    text = "\n".join(lines)
    assert "1,000 had no assistant turns" in text
    assert "200 were not classified (skip count over the 1000-file classification cap)" in text


# ---------------------------------------------------------------------------
# FIX 4: the task-mix-cell definition line
# ---------------------------------------------------------------------------


def test_beat2_defines_task_mix_cell_right_after_the_spread_line():
    lines = beat2_configurations(_fake_surface(), _fake_walkdown_line(), [])
    text = "\n".join(lines)
    definition = (
        "A task-mix cell is one workspace paired with the main kind of file "
        "the run wrote; matching compares configurations only where they "
        "share a cell, which is a proxy for the kind of task worked on, not "
        "an identification of the task itself."
    )
    assert definition in text
    # PATCH 3: the old misdescription (a cell IS the same task, not a proxy
    # for one) must be gone, not just amended somewhere else in the line.
    assert "worked on the same kinds of task" not in text
    walkdown_idx = next(i for i, l in enumerate(lines) if _fake_walkdown_line() in l)
    definition_idx = next(i for i, l in enumerate(lines) if definition in l)
    assert definition_idx == walkdown_idx + 1


def test_beat2_no_task_mix_cell_definition_when_no_walkdown_line():
    # band_support_note legitimately mentions "task-mix cells" in its own
    # prose (surface.py's wording, unrelated to this fix), so check for the
    # definition sentence specifically rather than the bare substring.
    surface = _fake_surface()
    surface["band_note"] = None
    surface["band_support_note"] = None
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "task-mix cell" not in text
    assert "A task-mix cell is" not in text


def test_overlap_caveat_prints_under_the_walkdown():
    # Lead integration: when no task-mix cell holds two comparable workflow
    # configurations, surface.py falls back to an unmatched fit and reports
    # `overlap_caveat`. That sentence changes what every number in beat 2
    # means, so it has to print with them, not down in the exclusions beat.
    surface = _fake_surface()
    surface["overlap_caveat"] = (
        "No task-mix cell contains two comparable workflow configurations, so "
        "these numbers are not adjusted for task mix."
    )
    out = render(
        ingest_diag=_fake_ingest_diag(),
        coverage=_fake_coverage(),
        coverage_line=_fake_coverage_line(),
        surface=surface,
        walkdown_line=_fake_walkdown_line(),
        price_warnings=[],
    )
    assert "not adjusted for task mix" in out
    body = out.split("What I left out")[0]
    assert "not adjusted for task mix" in body


def test_no_overlap_caveat_key_is_silent():
    surface = _fake_surface()
    surface.pop("overlap_caveat", None)
    out = render(
        ingest_diag=_fake_ingest_diag(),
        coverage=_fake_coverage(),
        coverage_line=_fake_coverage_line(),
        surface=surface,
        walkdown_line=_fake_walkdown_line(),
        price_warnings=[],
    )
    assert "not adjusted for task mix" not in out


# ---------------------------------------------------------------------------
# 11. band_note / band_support_note: printed after the walkdown line and the
# overlap caveat, each on its own line, each only when present; the table
# marks exactly the out-of-band rows.
# ---------------------------------------------------------------------------


def test_beat2_band_note_and_support_note_print_after_walkdown_and_caveat():
    surface = _fake_surface()
    surface["overlap_caveat"] = (
        "No task-mix cell contains two comparable workflow configurations, so "
        "these numbers are not adjusted for task mix."
    )
    lines = beat2_configurations(surface, _fake_walkdown_line(), [])

    walkdown_idx = next(i for i, l in enumerate(lines) if _fake_walkdown_line() in l)
    caveat_idx = next(i for i, l in enumerate(lines) if surface["overlap_caveat"] in l)
    band_note_idx = next(i for i, l in enumerate(lines) if surface["band_note"] in l)
    support_idx = next(i for i, l in enumerate(lines) if surface["band_support_note"] in l)

    assert walkdown_idx < caveat_idx < band_note_idx < support_idx

    # same indentation convention as overlap_caveat (two leading spaces, own line)
    caveat_indent = len(lines[caveat_idx]) - len(lines[caveat_idx].lstrip(" "))
    band_note_indent = len(lines[band_note_idx]) - len(lines[band_note_idx].lstrip(" "))
    support_indent = len(lines[support_idx]) - len(lines[support_idx].lstrip(" "))
    assert band_note_indent == caveat_indent == support_indent
    assert lines[band_note_idx].strip() == surface["band_note"]
    assert lines[support_idx].strip() == surface["band_support_note"]


def test_beat2_band_notes_absent_when_none():
    surface = _fake_surface()
    surface["band_note"] = None
    surface["band_support_note"] = None
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "resampling" not in text.lower()
    assert "point estimate that falls outside" not in text.lower()


def test_beat2_band_note_prints_alone_when_support_note_absent():
    surface = _fake_surface()
    surface["band_support_note"] = None
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert surface["band_note"] in text
    assert "thinnest confidence band" not in text.lower()


def test_beat2_marks_out_of_band_rows_in_table():
    surface = _fake_surface()
    table = surface["table"].copy()
    table["estimate_outside_band"] = [True, False]
    surface["table"] = table

    lines = beat2_configurations(surface, "", [])
    opus_line = next(l for l in lines if "opus-5-xhigh" in l)
    sonnet_line = next(l for l in lines if "sonnet-5-medium" in l)

    assert "*" in opus_line
    assert "*" not in sonnet_line
    # the column still lines up: every data row and the header share one width
    header_line = next(l for l in lines if l.strip().startswith("configuration"))
    band_col_start = header_line.index("accepted") + len("accepted")
    # sanity: both rows are at least as long as the header up to that point
    assert len(opus_line) >= band_col_start
    assert len(sonnet_line) >= band_col_start


def test_beat2_no_marker_when_estimate_outside_band_column_absent():
    # Backward compatibility: a table built before this column existed (or a
    # caller that never set it) must render its configuration rows with no
    # marker at all, not raise. (band_note itself legitimately mentions the
    # marker in prose, so check the data rows, not the whole beat.)
    surface = _fake_surface()
    surface["band_note"] = None
    surface["band_support_note"] = None
    lines = beat2_configurations(surface, "", [])
    opus_line = next(l for l in lines if "opus-5-xhigh" in l)
    sonnet_line = next(l for l in lines if "sonnet-5-medium" in l)
    assert "*" not in opus_line
    assert "*" not in sonnet_line


# ---------------------------------------------------------------------------
# PATCH 2 (reviewer round, HIGH): tier composition column plus a plain-
# language caveat when acceptance is measured by a different instrument
# per configuration.
# ---------------------------------------------------------------------------


def test_beat2_prints_tier_composition_column_and_caveat_line():
    surface = _fake_surface()
    table = surface["table"].copy()
    table["tiers"] = ["31v/2h", "27h"]
    surface["table"] = table
    surface["tier_note"] = (
        "Acceptance is not measured the same way for every workflow "
        "configuration above: rows graded at the verified tier are "
        "accepted by construction, so a configuration with more verified "
        "rows tends to show a higher acceptance rate for reasons about the "
        "evidence available, not about the work itself. See the tiers "
        "column for each configuration's mix."
    )

    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)

    # The composition display: each configuration's own tier mix appears on
    # its own row.
    opus_line = next(l for l in lines if "opus-5-xhigh" in l)
    sonnet_line = next(l for l in lines if "sonnet-5-medium" in l)
    assert "31v/2h" in opus_line
    assert "27h" in sonnet_line

    header_line = next(l for l in lines if l.strip().startswith("configuration"))
    assert "tiers" in header_line

    # The caveat line: printed once, under the table.
    assert surface["tier_note"] in text
    assert "accepted by construction" in text


def test_beat2_tier_composition_absent_key_renders_n_a_not_raise():
    # Backward compatibility: a table with no "tiers" column at all (older
    # surface, or a hand-built fixture) renders "n/a" rather than raising.
    surface = _fake_surface()
    lines = beat2_configurations(surface, "", [])
    opus_line = next(l for l in lines if "opus-5-xhigh" in l)
    assert "n/a" in opus_line


def test_beat2_no_tier_caveat_line_when_note_absent():
    surface = _fake_surface()
    surface["tier_note"] = None
    text = "\n".join(beat2_configurations(surface, "", []))
    assert "accepted by construction" not in text


# ---------------------------------------------------------------------------
# Final reviewer fix round, FIX 1 (blocker): `_extremes_ratio` (cli.py) must
# restrict a configuration's token sum, dollar sum, AND accepted count to the
# SAME subset of rows (both known), never a subset for the numerator and the
# full accepted count for the denominator; and it must disclose the exact
# basis whenever that restriction actually dropped anything.
# ---------------------------------------------------------------------------


def _ratio_table(costs: dict) -> dict:
    return {"table": pd.DataFrame({"arm": list(costs), "cost_per_accepted": list(costs.values())})}


def test_extremes_ratio_restricts_numerator_and_denominator_to_the_same_rows():
    df = pd.DataFrame(
        [
            # "dear" configuration: 3 accepted runs, one missing its dollar cost.
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": 10.0},
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": 10.0},
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": float("nan")},
            # "cheap" configuration: 2 accepted runs, both fully known.
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
        ]
    )
    result = _ratio_table({"cheap.medium": 1.0, "dear.high": 10.0})
    pair = _extremes_ratio(df, result, price_mod)

    assert "unavailable" not in pair
    # Numerator (token/dollar sums) and denominator (accepted count) both
    # come from the 2 (of 3) accepted "dear" runs with a known dollar cost --
    # never 2000 tokens / 3 accepted, which would silently understate the
    # per-accepted token figure.
    assert pair["token_ratio"] == pytest.approx(10.0)  # (2000/2) / (200/2)
    assert pair["dollar_ratio"] == pytest.approx(10.0)  # (20/2) / (2/2)
    assert pair["amplifier"] == pytest.approx(1.0)

    assert pair.get("basis_note") is not None
    assert "2 of 3 runs considered (dear.high)" in pair["basis_note"]
    assert "cheap.medium" not in pair["basis_note"]  # cheap lost nothing
    assert "—" not in pair["basis_note"]


def test_extremes_ratio_basis_note_names_both_configurations_when_both_drop():
    df = pd.DataFrame(
        [
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": 10.0},
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": float("nan")},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": float("nan"), "usd": 1.0},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
        ]
    )
    result = _ratio_table({"cheap.medium": 1.0, "dear.high": 10.0})
    pair = _extremes_ratio(df, result, price_mod)

    assert "1 of 2 runs considered (dear.high)" in pair["basis_note"]
    assert "2 of 3 runs considered (cheap.medium)" in pair["basis_note"]


def test_extremes_ratio_basis_note_fires_on_a_rejected_row_missing_a_value():
    """FIX 1 (final reviewer round, blocker): a REJECTED attempt missing its

    dollar cost used to slip past `basis_note` entirely, because the old
    code compared ACCEPTED counts (unaffected by dropping a rejected row)
    instead of ROW counts. "dear.high" has 2 accepted runs (both fully
    known) plus 1 REJECTED run missing its dollar cost: the accepted count
    is unchanged by the drop (2 in, 2 out), but the row count is not (3 in,
    2 out), and the numerator sums over all 3 known-outcome rows, accepted
    and rejected alike, so the dropped rejected row silently understated the
    spend unless the row-level comparison catches it.
    """
    df = pd.DataFrame(
        [
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": 10.0},
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": 10.0},
            {"arm": "dear.high", "accepted": False, "total_tokens": 500.0, "usd": float("nan")},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
        ]
    )
    result = _ratio_table({"cheap.medium": 1.0, "dear.high": 10.0})
    pair = _extremes_ratio(df, result, price_mod)

    assert pair.get("basis_note") is not None
    assert "2 of 3 runs considered (dear.high)" in pair["basis_note"]
    assert "cheap.medium" not in pair["basis_note"]


def test_extremes_ratio_no_basis_note_when_nothing_dropped():
    df = pd.DataFrame(
        [
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": 10.0},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
        ]
    )
    result = _ratio_table({"cheap.medium": 1.0, "dear.high": 10.0})
    pair = _extremes_ratio(df, result, price_mod)
    assert "basis_note" not in pair


def test_extremes_ratio_unavailable_when_no_accepted_run_has_both_known_together():
    # Each side has SOME known values individually, just never together on
    # the same accepted run -- distinct from either side being wholly
    # unknown, and must not be misreported as one of those two causes.
    df = pd.DataFrame(
        [
            {"arm": "dear.high", "accepted": True, "total_tokens": 1000.0, "usd": float("nan")},
            {"arm": "dear.high", "accepted": True, "total_tokens": float("nan"), "usd": 5.0},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
        ]
    )
    result = _ratio_table({"cheap.medium": 1.0, "dear.high": 10.0})
    pair = _extremes_ratio(df, result, price_mod)
    assert pair == {
        "unavailable": (
            "no accepted run in one of the two configurations has both a "
            "known token count and a known dollar cost, so the comparison "
            "would be guesswork"
        )
    }


def test_extremes_ratio_still_reports_wholly_unknown_token_total():
    df = pd.DataFrame(
        [
            {"arm": "dear.high", "accepted": True, "total_tokens": float("nan"), "usd": 10.0},
            {"arm": "cheap.medium", "accepted": True, "total_tokens": 100.0, "usd": 1.0},
        ]
    )
    result = _ratio_table({"cheap.medium": 1.0, "dear.high": 10.0})
    pair = _extremes_ratio(df, result, price_mod)
    assert pair == {
        "unavailable": (
            "the token total for one of the two configurations is unknown, "
            "so the comparison would be guesswork"
        )
    }


def test_beat2_prints_basis_note_directly_under_amplifier_line():
    surface = _fake_surface()
    surface["ratio_pair"] = {
        "token_ratio": 3.1,
        "dollar_ratio": 10.4,
        "amplifier": 3.4,
        "label": "cheapest vs most expensive",
        "basis_note": (
            "computed on 40 of 62 runs (dear.high) where both the token count "
            "and the dollar cost are known"
        ),
    }
    lines = beat2_configurations(surface, "", [])
    text = "\n".join(lines)
    assert "computed on 40 of 62 runs (dear.high)" in text

    amp_idx = next(i for i, l in enumerate(lines) if "cheapest vs most expensive" in l)
    note_idx = next(i for i, l in enumerate(lines) if "computed on 40 of 62 runs" in l)
    assert note_idx == amp_idx + 1

    amp_indent = len(lines[amp_idx]) - len(lines[amp_idx].lstrip(" "))
    note_indent = len(lines[note_idx]) - len(lines[note_idx].lstrip(" "))
    assert note_indent > amp_indent


def test_beat2_no_basis_note_line_when_absent():
    lines = beat2_configurations(_fake_surface(), "", [])
    text = "\n".join(lines)
    assert "where both the token count and the dollar cost are known" not in text


# ---------------------------------------------------------------------------
# Final reviewer fix round, FIX 2 (blocker): `--limit` must name what it cut,
# in the same style as the existing `window` line.
# ---------------------------------------------------------------------------


def test_beat1_names_the_limit_and_what_it_cut():
    diag = dict(_fake_ingest_diag())
    diag["limit"] = 50
    diag["files_omitted_by_limit"] = {"claude-code": 3000, "codex": 1200, "total": 4200}
    text = "\n".join(beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line()))
    assert "--limit 50" in text
    assert (
        "4,200 more files were found but not read because of it "
        "(raise --limit, or drop it, to read them)"
    ) in text


def test_beat1_no_limit_line_when_limit_given_but_nothing_omitted():
    diag = dict(_fake_ingest_diag())
    diag["limit"] = 50
    diag["files_omitted_by_limit"] = {"claude-code": 0, "codex": 0, "total": 0}
    text = "\n".join(beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line()))
    assert "--limit" not in text


def test_beat1_no_limit_line_when_limit_key_absent():
    text = "\n".join(beat1_what_i_read(_fake_ingest_diag(), _fake_coverage(), _fake_coverage_line()))
    assert "--limit" not in text


# ---------------------------------------------------------------------------
# Coordination note: `zero_token_synthetic` is another module's key, printed
# only when present and greater than zero.
# ---------------------------------------------------------------------------


def test_beat1_prints_zero_token_synthetic_count_when_present():
    diag = _fake_ingest_diag()
    diag["zero_token_synthetic"] = 1526
    text = "\n".join(beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line()))
    assert "1,526" in text
    assert "zero-token" in text.lower()


def test_beat1_absent_zero_token_synthetic_key_prints_nothing():
    text = "\n".join(beat1_what_i_read(_fake_ingest_diag(), _fake_coverage(), _fake_coverage_line()))
    assert "zero-token" not in text.lower()


def test_beat1_zero_token_synthetic_zero_prints_nothing():
    diag = _fake_ingest_diag()
    diag["zero_token_synthetic"] = 0
    text = "\n".join(beat1_what_i_read(diag, _fake_coverage(), _fake_coverage_line()))
    assert "zero-token" not in text.lower()


# ---------------------------------------------------------------------------
# Final reviewer fix round, FIX 5 (accepted in part): runs with no recorded
# workspace are named in "What I left out", not silently absent.
# ---------------------------------------------------------------------------


def test_beat3_names_no_workspace_runs_when_present():
    surface = {
        "exclusions": [],
        "no_workspace_note": (
            "3 runs have no recorded workspace, so they cannot take part in "
            "task-mix matching"
        ),
    }
    lines = beat3_exclusions(surface=surface, coverage={}, ingest_diag={})
    text = "\n".join(lines)
    assert "3 runs have no recorded workspace" in text
    assert "nothing excluded" not in text


def test_beat3_no_workspace_note_absent_key_prints_nothing_extra():
    lines = beat3_exclusions(surface={"exclusions": []}, coverage={}, ingest_diag={})
    text = "\n".join(lines)
    assert "no recorded workspace" not in text
    assert "nothing excluded" in text


def test_beat1_tier_table_accounts_for_every_parsed_run():
    # SPEC section 0: the tiers must add up to what was read. `ungraded` was
    # computed and only visible behind --grading, so the default table summed
    # to less than the run count with nothing on screen explaining the gap.
    diag = dict(_fake_ingest_diag())
    coverage = {
        "tiers": {"verified": 3, "reported": 0, "heuristic": 4,
                  "asserted": 2, "censored": 5, "ungraded": 6},
        "n_total": 20,
    }
    text = "\n".join(beat1_what_i_read(diag, coverage, _fake_coverage_line()))
    assert "ungraded" in text
    counts = [int(n) for n in re.findall(r"^\s+\w+\s+(\d+)$", text, re.M)]
    assert sum(counts) == 20
