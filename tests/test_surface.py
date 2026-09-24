"""Tests for the task-mix-honest cost surface (SPEC section 6).

Records are built inline via `_rec`, a factory that stamps the fields
`build_frame` reads: model/effort/workspace/written_file_kinds/usd/grade.
No real corpus data appears here.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

import loopmath.surface as surface_mod
from loopmath.e0.estimate import spread_S
from loopmath.surface import (
    COST_DOLLARS,
    COST_TOKENS,
    build_frame,
    cost_surface,
    walkdown,
    walkdown_line,
)


def _rec(
    run_id,
    model,
    effort,
    workspace,
    usd,
    out_tokens,
    accepted,
    kinds=None,
    tier="verified",
    in_tokens=100,
    cache_read_tokens=0,
    cache_write_tokens=0,
    wall_s=120.0,
    ts="2026-08-20T10:00:00Z",
    harness="claude-code",
):
    """One `RunRecord.to_dict()`-shaped record, already graded and priced."""
    return {
        "run_id": run_id,
        "harness": harness,
        "model": model,
        "effort": effort,
        "tokens": {
            "in": in_tokens,
            "cache_read": cache_read_tokens,
            "cache_write": cache_write_tokens,
            "out": out_tokens,
        },
        "wall_s": wall_s,
        "ts": ts,
        "workspace": workspace,
        "written_file_kinds": kinds or {},
        "grade": {"accepted": accepted, "tier": tier, "signal": "test fixture"},
        "usd": usd,
    }


def _varied_records():
    """Two arms 4x apart in cost, spread over 4 shared task-mix cells.

    Jitter is seeded (a plain `random.Random`, independent of any bootstrap
    seed under test) so the fit has genuine row-to-row variance -- enough
    that a cell-cluster bootstrap actually produces a spread of draws, which
    the 80%-vs-90%-band test needs to see a real width difference.
    """
    rng = random.Random(7)
    cells = ["py", "md", "json", "yaml"]
    records = []
    i = 0
    for model, base_cost in (("cheap-model", 2.0), ("pricey-model", 8.0)):
        for cell in cells:
            for _ in range(10):
                jitter = 1.0 + (rng.random() - 0.5) * 0.3  # +/- 15%
                cost = base_cost * jitter
                i += 1
                records.append(
                    _rec(
                        run_id=f"r{i}",
                        model=model,
                        effort="medium",
                        workspace="proj",
                        usd=cost,
                        out_tokens=int(cost * 1000),
                        accepted=True,
                        kinds={cell: 1},
                    )
                )
    return records


# ---------------------------------------------------------------------------
# 1. build_frame
# ---------------------------------------------------------------------------


def test_build_frame_columns_arm_and_cell_formatting():
    records = [
        _rec("r1", "opus-5", "medium", "proj-a", 1.23, 500, True, kinds={"py": 3, "md": 1}),
        _rec("r2", "opus-5", None, "proj-a", 0.5, 100, True, kinds={}),
        _rec("r3", None, "medium", "proj-a", 0.1, 10, True),  # no model label: kept, arm null
        _rec("r4", "fable-5", "xhigh", "proj-b", None, 2000, None, kinds={"py": 2, "md": 2}),
    ]
    df = build_frame(records)

    assert list(df.columns) == [
        "run_id", "harness", "arm", "model", "effort", "workspace", "cell",
        "usd", "out_tokens", "total_tokens", "tokens_complete", "accepted",
        "tier", "proxy_known", "wall_s", "ts", "priced",
    ]
    # r3 has no model label but must be KEPT (SPEC full-honesty: nothing is
    # silently dropped before exclusion accounting gets a chance to run).
    assert set(df["run_id"]) == {"r1", "r2", "r3", "r4"}

    r1 = df.set_index("run_id").loc["r1"]
    assert r1["arm"] == "opus-5.medium"
    assert r1["cell"] == "proj-a|py"  # py:3 beats md:1

    r2 = df.set_index("run_id").loc["r2"]
    assert r2["arm"] == "opus-5.?"  # unknown effort
    assert r2["cell"] == "proj-a|no-writes"  # empty write histogram

    r3 = df.set_index("run_id").loc["r3"]
    assert pd.isna(r3["arm"])  # no model label -> arm is null, never the string "None"

    r4 = df.set_index("run_id").loc["r4"]
    assert bool(r4["priced"]) is False
    assert np.isnan(r4["usd"])  # unpriced row kept, not dropped
    assert bool(r4["proxy_known"]) is False  # accepted is None
    assert r4["cell"] == "proj-b|md"  # py:2 vs md:2 tie -> alphabetical: md


def test_no_model_record_survives_build_frame_and_lands_in_own_bucket():
    """FIX 1: a no-model row is kept by build_frame and counted, not dropped."""
    records = _varied_records() + [
        _rec("nomodel-1", None, "medium", "proj", 1.0, 100, True, kinds={"py": 1})
    ]
    df = build_frame(records)

    assert "nomodel-1" in set(df["run_id"])
    row = df.set_index("run_id").loc["nomodel-1"]
    assert pd.isna(row["arm"])

    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=20, seed=1)
    by_reason = {e["reason"]: e["n"] for e in surface["exclusions"]}
    assert by_reason["no model label"] == 1
    assert surface["exclusion_by_run_id"]["nomodel-1"] == "no model label"


def test_build_frame_kind_tie_break_alphabetical():
    records = [_rec("r1", "opus-5", "medium", "ws", 1.0, 100, True, kinds={"yaml": 2, "json": 2})]
    df = build_frame(records)
    assert df.iloc[0]["cell"] == "ws|json"  # tie: json < yaml alphabetically


def test_missing_or_invalid_token_stream_is_nan_not_zero():
    """FIX 2: a missing/invalid token stream must not be silently priced as 0."""
    records = [
        _rec("r1", "m", "medium", "ws", 1.0, 100, True),  # complete: in=100, cache_read=0, cache_write=0, out=100
        {  # r2: 'out' stream missing entirely
            "run_id": "r2",
            "harness": "claude-code",
            "model": "m",
            "effort": "medium",
            "tokens": {"in": 50, "cache_read": 0, "cache_write": 0},
            "wall_s": 10.0,
            "ts": "2026-08-20T10:00:00Z",
            "workspace": "ws",
            "written_file_kinds": {},
            "grade": {"accepted": True, "tier": "verified", "signal": "x"},
            "usd": 2.0,
        },
        _rec("r3", "m", "medium", "ws", 3.0, -5, True),  # negative out tokens: invalid
    ]
    df = build_frame(records).set_index("run_id")

    assert bool(df.loc["r1", "tokens_complete"]) is True
    assert df.loc["r1", "out_tokens"] == 100.0
    assert df.loc["r1", "total_tokens"] == 200.0  # 100 in + 0 cache_read + 0 cache_write + 100 out

    assert bool(df.loc["r2", "tokens_complete"]) is False
    assert np.isnan(df.loc["r2", "out_tokens"])
    assert np.isnan(df.loc["r2", "total_tokens"])

    assert bool(df.loc["r3", "tokens_complete"]) is False
    assert np.isnan(df.loc["r3", "out_tokens"])
    assert np.isnan(df.loc["r3", "total_tokens"])

    # On the token basis, the existing exclusion ladder (df[cost_col].isna())
    # must now catch these rows instead of pricing a missing count as zero.
    surface = cost_surface(df.reset_index(), cost_col=COST_TOKENS, n_boot=10, seed=1)
    assert surface["exclusion_by_run_id"]["r2"] == "no usable cost value"
    assert surface["exclusion_by_run_id"]["r3"] == "no usable cost value"


def test_total_tokens_sums_all_four_distinct_streams():
    """SPEC amendment 1: total_tokens is the sum of all four billable streams.

    The four values are chosen distinct (and non-additively-overlapping) so
    that dropping any one stream, or misreading one stream's value into
    another stream's slot, changes the summed total in a way this exact
    comparison would catch.
    """
    records = [
        {
            "run_id": "r1",
            "harness": "claude-code",
            "model": "m",
            "effort": "medium",
            "tokens": {"in": 1000, "cache_read": 200, "cache_write": 30, "out": 4},
            "wall_s": 10.0,
            "ts": "2026-08-20T10:00:00Z",
            "workspace": "ws",
            "written_file_kinds": {},
            "grade": {"accepted": True, "tier": "verified", "signal": "x"},
            "usd": 1.0,
        }
    ]
    df = build_frame(records).set_index("run_id")
    row = df.loc["r1"]

    assert bool(row["tokens_complete"]) is True
    assert row["out_tokens"] == 4.0  # out_tokens is the 'out' stream alone
    assert row["total_tokens"] == 1234.0  # 1000 + 200 + 30 + 4, not any 3-stream subset


def test_missing_cache_write_alone_makes_total_tokens_nan():
    """SPEC amendment 1: a single missing stream (cache_write here) still
    poisons total_tokens and tokens_complete, even though it leaves the
    secondary `out_tokens` basis untouched (item 3: out_tokens is the `out`
    stream alone, unaffected by the other three). Fed as its own cost basis
    (`cost_col="total_tokens"`, which `cost_surface` accepts generically --
    see its docstring: "or whatever a future cost_col names"), such a row
    lands in the "no usable cost value" bucket rather than being counted as
    a zero-cost run.
    """
    records = [
        {
            "run_id": "r1",
            "harness": "claude-code",
            "model": "m",
            "effort": "medium",
            "tokens": {"in": 100, "cache_read": 10, "out": 50},  # cache_write missing
            "wall_s": 10.0,
            "ts": "2026-08-20T10:00:00Z",
            "workspace": "ws",
            "written_file_kinds": {},
            "grade": {"accepted": True, "tier": "verified", "signal": "x"},
            "usd": 1.0,
        }
    ]
    df = build_frame(records)
    row = df.set_index("run_id").loc["r1"]

    assert bool(row["tokens_complete"]) is False
    assert np.isnan(row["total_tokens"])
    assert row["out_tokens"] == 50.0  # the 'out' stream alone is still usable

    # On the out-tokens-only basis (COST_TOKENS), out is present, so this row
    # is not excluded for lacking a usable cost value.
    out_basis = cost_surface(df, cost_col=COST_TOKENS, min_n=1, min_overlap_cells=1, n_boot=10, seed=1)
    assert "r1" not in out_basis["exclusion_by_run_id"]

    # On the four-stream total-tokens basis, the missing cache_write makes
    # the cost value unusable, so the row is excluded and never priced as 0.
    total_basis = cost_surface(df, cost_col="total_tokens", n_boot=10, seed=1)
    assert total_basis["exclusion_by_run_id"]["r1"] == "no usable cost value"
    by_reason = {e["reason"]: e["n"] for e in total_basis["exclusions"]}
    assert by_reason["no usable cost value"] == 1
    assert total_basis["n_rows"] == total_basis["n_rows_used"] + sum(
        e["n"] for e in total_basis["exclusions"]
    )


def test_row_accounting_invariant_holds_with_four_stream_incomplete_rows():
    """Row-level accounting (n_rows == n_rows_used + sum(exclusion counts))
    must still hold exactly once some rows are missing one of the four
    streams, not just when every row is token-complete.
    """
    records = list(_varied_records())
    # Three additional rows, each missing a different one of the four streams.
    records.append(
        {
            "run_id": "missing-in",
            "harness": "claude-code",
            "model": "cheap-model",
            "effort": "medium",
            "tokens": {"cache_read": 1, "cache_write": 1, "out": 100},
            "wall_s": 10.0,
            "ts": "2026-08-20T10:00:00Z",
            "workspace": "proj",
            "written_file_kinds": {"py": 1},
            "grade": {"accepted": True, "tier": "verified", "signal": "x"},
            "usd": 1.0,
        }
    )
    records.append(
        {
            "run_id": "missing-cache-read",
            "harness": "claude-code",
            "model": "cheap-model",
            "effort": "medium",
            "tokens": {"in": 100, "cache_write": 1, "out": 100},
            "wall_s": 10.0,
            "ts": "2026-08-20T10:00:00Z",
            "workspace": "proj",
            "written_file_kinds": {"py": 1},
            "grade": {"accepted": True, "tier": "verified", "signal": "x"},
            "usd": 1.0,
        }
    )
    records.append(
        {
            "run_id": "missing-cache-write",
            "harness": "claude-code",
            "model": "cheap-model",
            "effort": "medium",
            "tokens": {"in": 100, "cache_read": 1, "out": 100},
            "wall_s": 10.0,
            "ts": "2026-08-20T10:00:00Z",
            "workspace": "proj",
            "written_file_kinds": {"py": 1},
            "grade": {"accepted": True, "tier": "verified", "signal": "x"},
            "usd": 1.0,
        }
    )
    df = build_frame(records)
    surface = cost_surface(df, cost_col="total_tokens", n_boot=20, seed=1)

    for run_id in ("missing-in", "missing-cache-read", "missing-cache-write"):
        assert surface["exclusion_by_run_id"][run_id] == "no usable cost value"

    assert surface["n_rows"] == surface["n_rows_used"] + sum(
        e["n"] for e in surface["exclusions"]
    )
    excluded_ids = set(surface["exclusion_by_run_id"])
    used_ids = set(df["run_id"]) - excluded_ids
    assert len(used_ids) == surface["n_rows_used"]


# ---------------------------------------------------------------------------
# 2. cost_surface: ordering + bands contain the point estimate
# ---------------------------------------------------------------------------


def test_cost_surface_ordering_and_bands_contain_point():
    df = build_frame(_varied_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=500, seed=42, ci=0.80)
    table = surface["table"].set_index("arm")

    cheap = table.loc["cheap-model.medium", "cost_per_accepted"]
    pricey = table.loc["pricey-model.medium", "cost_per_accepted"]
    assert pricey > cheap  # the 4x-more-expensive configuration must stay pricier

    for _, row in table.iterrows():
        assert row["lo"] <= row["cost_per_accepted"] <= row["hi"]


def test_band_n_draws_present_and_bounds_n_boot_valid_s():
    """FIX 7: per-configuration draw support is surfaced, not discarded at merge."""
    df = build_frame(_varied_records())
    surface = cost_surface(df, n_boot=200, seed=7)
    table = surface["table"]

    assert "band_n_draws" in table.columns
    assert (table["band_n_draws"] > 0).all()  # every banded configuration has support

    assert surface["n_boot_valid_S"] <= surface["n_boot_valid"]
    assert surface["n_boot_valid_S"] > 0


# ---------------------------------------------------------------------------
# 3. 80% bands are narrower than 90% bands
# ---------------------------------------------------------------------------


def test_bands_are_80_percent_not_90():
    df = build_frame(_varied_records())
    s80 = cost_surface(df, n_boot=500, seed=42, ci=0.80)
    s90 = cost_surface(df, n_boot=500, seed=42, ci=0.90)

    width_80 = s80["S_hi"] - s80["S_lo"]
    width_90 = s90["S_hi"] - s90["S_lo"]
    assert width_80 < width_90


# ---------------------------------------------------------------------------
# 4. determinism
# ---------------------------------------------------------------------------


def test_bootstrap_is_deterministic_given_seed():
    df = build_frame(_varied_records())
    s1 = cost_surface(df, n_boot=300, seed=123)
    s2 = cost_surface(df, n_boot=300, seed=123)
    assert s1["S_lo"] == s2["S_lo"]
    assert s1["S_hi"] == s2["S_hi"]


# ---------------------------------------------------------------------------
# 5. exclusions account for every dropped row, exactly once
# ---------------------------------------------------------------------------


def test_exclusions_account_for_every_row():
    records = []
    # Two eligible, overlapping configurations: 2 rows/cell across 3 shared
    # cells each (min_n=5 satisfied, overlap-cell restriction satisfied).
    for cell in ("c1", "c2", "c3"):
        for j in range(2):
            records.append(_rec(f"a-{cell}-{j}", "model-a", "medium", "ws", 1.0, 100, True, kinds={cell: 1}))
            records.append(_rec(f"b-{cell}-{j}", "model-b", "medium", "ws", 1.0, 100, True, kinds={cell: 1}))
    # A thin configuration: below min_n, excluded entirely.
    for j in range(3):
        records.append(_rec(f"c-{j}", "model-c", "medium", "ws", 1.0, 100, True, kinds={"c1": 1}))
    # No-usable-cost rows (usd unknown): 4 of them.
    for j in range(2):
        records.append(_rec(f"a-unpriced-{j}", "model-a", "medium", "ws", None, 100, True, kinds={"c1": 1}))
        records.append(_rec(f"b-unpriced-{j}", "model-b", "medium", "ws", None, 100, True, kinds={"c2": 1}))
    # Unknown-acceptance rows: 3 of them.
    for j in range(2):
        records.append(_rec(f"a-unknown-{j}", "model-a", "medium", "ws", 1.0, 100, None, kinds={"c3": 1}))
    records.append(_rec("b-unknown-0", "model-b", "medium", "ws", 1.0, 100, None, kinds={"c1": 1}))

    df = build_frame(records)
    surface = cost_surface(df, min_n=5, min_overlap_cells=2, n_boot=50, seed=1)

    n_rows = surface["n_rows"]
    n_used = surface["n_rows_used"]
    exclusions = surface["exclusions"]
    total_excluded = sum(e["n"] for e in exclusions)

    assert n_rows == 22
    assert n_used == 12  # 6 rows of model-a + 6 of model-b
    assert n_rows == n_used + total_excluded

    by_reason = {e["reason"]: e["n"] for e in exclusions}
    assert by_reason["no model label"] == 0
    assert by_reason["no usable cost value"] == 4
    assert by_reason["unknown acceptance"] == 3
    assert by_reason["configuration below min_n"] == 3
    assert by_reason["configuration below min_overlap_cells"] == 0


def test_exclusions_are_disjoint_at_the_row_level():
    """FIX 9: prove disjointness with actual run ids, not a self-referential count.

    The old version of this test expanded the aggregate exclusion counts into
    a list and compared the list's length to the sum of the counts it was
    built from -- true by construction, regardless of whether any row was
    ever actually double-counted. This version checks the real thing: every
    run_id in the input lands in exactly one place (used, or exactly one
    named exclusion bucket), via `exclusion_by_run_id`, which `cost_surface`
    exposes at row granularity for exactly this kind of check.
    """
    records = []
    for cell in ("c1", "c2", "c3"):
        for j in range(2):
            records.append(_rec(f"a-{cell}-{j}", "model-a", "medium", "ws", 1.0, 100, True, kinds={cell: 1}))
            records.append(_rec(f"b-{cell}-{j}", "model-b", "medium", "ws", 1.0, 100, True, kinds={cell: 1}))
    for j in range(3):
        records.append(_rec(f"c-{j}", "model-c", "medium", "ws", 1.0, 100, True, kinds={"c1": 1}))
    for j in range(2):
        records.append(_rec(f"a-unpriced-{j}", "model-a", "medium", "ws", None, 100, True, kinds={"c1": 1}))
        records.append(_rec(f"b-unpriced-{j}", "model-b", "medium", "ws", None, 100, True, kinds={"c2": 1}))
    for j in range(2):
        records.append(_rec(f"a-unknown-{j}", "model-a", "medium", "ws", 1.0, 100, None, kinds={"c3": 1}))
    records.append(_rec("b-unknown-0", "model-b", "medium", "ws", 1.0, 100, None, kinds={"c1": 1}))

    df = build_frame(records)
    surface = cost_surface(df, min_n=5, min_overlap_cells=2, n_boot=50, seed=1)

    exclusion_by_run_id = surface["exclusion_by_run_id"]
    all_run_ids = set(df["run_id"])
    excluded_ids = set(exclusion_by_run_id)
    used_ids = all_run_ids - excluded_ids

    # Every excluded run_id maps to exactly one reason (dict invariant), and
    # that partition must reconcile with the aggregate counts and n_rows_used.
    assert len(excluded_ids) == sum(e["n"] for e in surface["exclusions"])
    assert len(used_ids) == surface["n_rows_used"]
    assert surface["n_rows"] == surface["n_rows_used"] + sum(e["n"] for e in surface["exclusions"])

    from collections import Counter

    reason_counts = Counter(exclusion_by_run_id.values())
    for e in surface["exclusions"]:
        assert reason_counts.get(e["reason"], 0) == e["n"]

    # Specific known run_ids land in the specific expected bucket, not just
    # "some" bucket -- this is what makes the check row-level instead of a
    # second aggregate count in disguise.
    assert exclusion_by_run_id["c-0"] == "configuration below min_n"
    assert exclusion_by_run_id["c-1"] == "configuration below min_n"
    assert exclusion_by_run_id["a-unpriced-0"] == "no usable cost value"
    assert exclusion_by_run_id["b-unpriced-1"] == "no usable cost value"
    assert exclusion_by_run_id["a-unknown-0"] == "unknown acceptance"
    assert exclusion_by_run_id["b-unknown-0"] == "unknown acceptance"
    assert "a-c1-0" in used_ids  # a plain used row is not in the exclusion map at all


# ---------------------------------------------------------------------------
# 5a. Final reviewer fix round, FIX 2 (blocker): a row can be dropped from
# `work` for two DIFFERENT reasons, and they must land in two DIFFERENT
# buckets, not both under "configuration below min_overlap_cells".
# ---------------------------------------------------------------------------


def test_row_excluded_for_its_own_unshared_cell_is_not_blamed_on_its_configuration():
    """Two distinct drop reasons, measured separately.

    "model-a" and "model-b" share 2 task-mix cells (cA1, cA2) with >= 2 rows
    each, which clears the overlap requirement for BOTH of them -- but
    "model-a" has one more row in "cA4", a cell nobody else worked in. That
    single row is not itself overlap-eligible even though its configuration
    is: it must land under the new row-level reason, never under
    "configuration below min_overlap_cells" (which would falsely claim
    "model-a" as a whole failed to share cells with another configuration).

    "model-c" has 5 rows, all in "cC5", a cell no other eligible
    configuration ever touches: its ENTIRE configuration fails the overlap
    requirement, so all 5 of its rows belong under the ORIGINAL
    "configuration below min_overlap_cells" reason.

    Old (buggy) behaviour called both of these "configuration below
    min_overlap_cells", which is false for "model-a"'s single stray row: its
    configuration did clear the overlap requirement, on its other 4 rows.
    """
    records = []
    for j in range(2):
        records.append(_rec(f"a-cA1-{j}", "model-a", "medium", "ws", 1.0, 100, True, kinds={"cA1": 1}))
        records.append(_rec(f"b-cA1-{j}", "model-b", "medium", "ws", 1.0, 100, True, kinds={"cA1": 1}))
    for j in range(2):
        records.append(_rec(f"a-cA2-{j}", "model-a", "medium", "ws", 1.0, 100, True, kinds={"cA2": 1}))
    for j in range(3):
        records.append(_rec(f"b-cA2-{j}", "model-b", "medium", "ws", 1.0, 100, True, kinds={"cA2": 1}))
    records.append(_rec("a-cA4-0", "model-a", "medium", "ws", 1.0, 100, True, kinds={"cA4": 1}))
    for j in range(5):
        records.append(_rec(f"c-cC5-{j}", "model-c", "medium", "ws", 1.0, 100, True, kinds={"cC5": 1}))

    df = build_frame(records)
    surface = cost_surface(df, min_n=5, min_overlap_cells=2, n_boot=20, seed=1)

    n_rows = surface["n_rows"]
    n_used = surface["n_rows_used"]
    exclusions = surface["exclusions"]
    by_reason = {e["reason"]: e for e in exclusions}

    assert n_rows == 15  # 5 (model-a) + 5 (model-b) + 5 (model-c)
    assert n_used == 9  # model-a's 4 shared rows + model-b's 5 shared rows

    # "model-a" cleared the overlap requirement (on cA1/cA2); only its lone
    # cA4 row is excluded, and only under the new, row-level reason.
    assert by_reason["run's task-mix cell not shared with another configuration"]["n"] == 1
    assert (
        by_reason["run's task-mix cell not shared with another configuration"]["detail"]
        == "run is in a task-mix cell no other comparable workflow configuration worked in"
    )
    assert by_reason["run's task-mix cell not shared with another configuration"]["n_censored"] == 0

    # "model-c" never cleared the overlap requirement at all: all 5 of its
    # rows are excluded, and only under the configuration-level reason.
    assert by_reason["configuration below min_overlap_cells"]["n"] == 5
    assert by_reason["configuration below min_overlap_cells"]["n_censored"] == 0

    # The buckets still sum to the row total, and nothing is double counted.
    assert n_rows == n_used + sum(e["n"] for e in exclusions)

    exclusion_by_run_id = surface["exclusion_by_run_id"]
    assert exclusion_by_run_id["a-cA4-0"] == "run's task-mix cell not shared with another configuration"
    for j in range(5):
        assert exclusion_by_run_id[f"c-cC5-{j}"] == "configuration below min_overlap_cells"
    assert "a-cA1-0" not in exclusion_by_run_id  # a plain used row is never excluded


# ---------------------------------------------------------------------------
# 5b. Reviewer fix round FIX 1: `n_censored` is measured PER exclusion
# bucket, never taken from the global censored tier count.
# ---------------------------------------------------------------------------


def test_exclusions_n_censored_partitions_by_bucket_not_globally():
    """A censored run can land in any of three buckets, first-match-wins.

    grade.py's R5 censoring rule only ever demotes a row whose acceptance
    outcome is already unknown (`accepted is None`), so a censored row can
    only ever be claimed by "no model label", "no usable cost value", or
    "unknown acceptance" -- never by the two below-min-n / below-overlap
    buckets, which only ever see rows with a KNOWN acceptance outcome. This
    proves each of the first three buckets reports its own true intersection
    with the censored tier, that the two configuration-size buckets always
    report zero, and that the per-bucket counts never double count a row.
    """
    records = []
    # Baseline eligible, overlapping configurations, untouched by this test:
    # keeps the fit itself non-degenerate (2 accepted rows/cell, 3 shared
    # cells, for both model-a and model-b).
    for cell in ("c1", "c2", "c3"):
        for j in range(2):
            records.append(_rec(f"a-{cell}-{j}", "model-a", "medium", "ws", 1.0, 100, True, kinds={cell: 1}))
            records.append(_rec(f"b-{cell}-{j}", "model-b", "medium", "ws", 1.0, 100, True, kinds={cell: 1}))
    # A configuration below min_n, with a KNOWN (non-censored) acceptance
    # outcome -- proves the below-min-n bucket's n_censored is 0 even when
    # the bucket itself is non-empty.
    for j in range(2):
        records.append(_rec(f"c-{j}", "model-c", "medium", "ws", 1.0, 100, True, kinds={"c1": 1}))

    # Censored, with no model label at all: claimed by "no model label",
    # first in the ladder, even though it also has accepted=None.
    for j in range(3):
        records.append(
            _rec(f"no-model-censored-{j}", None, None, "ws", 1.0, 100, None, kinds={"c1": 1}, tier="censored")
        )
    # Censored, with a model label but no usable cost value: claimed by "no
    # usable cost value", the next rung down.
    for j in range(2):
        records.append(
            _rec(
                f"no-cost-censored-{j}", "model-a", "medium", "ws", None, 100, None,
                kinds={"c1": 1}, tier="censored",
            )
        )
    # Censored, with both a model label and a cost value: only "unknown
    # acceptance" is left to claim these.
    for j in range(4):
        records.append(
            _rec(
                f"unknown-censored-{j}", "model-a", "medium", "ws", 1.0, 100, None,
                kinds={"c1": 1}, tier="censored",
            )
        )

    df = build_frame(records)
    surface = cost_surface(df, min_n=5, min_overlap_cells=2, n_boot=20, seed=1)
    by_reason = {e["reason"]: e for e in surface["exclusions"]}

    assert by_reason["no model label"]["n_censored"] == 3
    assert by_reason["no usable cost value"]["n_censored"] == 2
    assert by_reason["unknown acceptance"]["n_censored"] == 4
    assert by_reason["configuration below min_n"]["n"] == 2  # non-empty bucket ...
    assert by_reason["configuration below min_n"]["n_censored"] == 0  # ... but never censored
    assert by_reason["configuration below min_overlap_cells"]["n_censored"] == 0

    # The buckets still sum to the total censored count in the input, and no
    # row is ever double counted across two buckets.
    n_censored_in_input = sum(1 for r in records if r["grade"]["tier"] == "censored")
    assert n_censored_in_input == 9
    assert sum(e["n_censored"] for e in surface["exclusions"]) == n_censored_in_input

    # Each bucket's n_censored is a strict subset of that bucket's own n.
    for e in surface["exclusions"]:
        assert e["n_censored"] <= e["n"]


# ---------------------------------------------------------------------------
# 6. walkdown_line: readable, no em dash, no forbidden vocabulary
# ---------------------------------------------------------------------------


def test_walkdown_line_is_clean():
    df = build_frame(_varied_records())
    surface = cost_surface(df, n_boot=100, seed=5)
    line = walkdown_line(surface)

    assert "—" not in line  # no em dash
    forbidden = [
        "knowledge gradient", "posterior", "prior", "experimental design",
        "value of information", "bandit", "arms",
    ]
    lowered = line.lower()
    for word in forbidden:
        assert word not in lowered

    # sanity: the line actually renders numbers, not placeholders
    assert "spread:" in line
    assert "band" in line

    steps = walkdown(surface)
    assert [s["step"] for s in steps] == [
        "naive spread",
        "after matching on task mix",
        "after pooling thin configurations",
        "80% band",
    ]


# ---------------------------------------------------------------------------
# PATCH 1 (reviewer round, HIGH): a `.low` configuration must not be
# silently dropped from any of the three walkdown spreads.
# ---------------------------------------------------------------------------


def _low_included_records():
    """Three configurations, ~20x apart, cheapest is a `.low` effort.

    Mirrors the reviewer's exact reproduction: a table genuinely spanning
    20x that, with E0's `spread_S` default (`exclude_trivial=True`), prints
    only the ~2x gap between the two non-`.low` configurations because the
    cheapest one (ending in "low") gets silently dropped from the ratio.
    Small jitter keeps within-configuration variance low relative to the
    between-configuration gap, so James-Stein shrinkage does not compress
    the fitted spread back toward 2x either.
    """
    rng = random.Random(11)
    cells = ["py", "md", "json", "yaml"]
    records = []
    i = 0
    for model, effort, base_cost in (
        ("cheap-model", "low", 1.0),
        ("mid-model", "medium", 10.0),
        ("costly-model", "xhigh", 20.0),
    ):
        for cell in cells:
            for _ in range(5):
                jitter = 1.0 + (rng.random() - 0.5) * 0.1  # +/- 5%
                cost = base_cost * jitter
                i += 1
                records.append(
                    _rec(
                        run_id=f"t{i}",
                        model=model,
                        effort=effort,
                        workspace="proj",
                        usd=cost,
                        out_tokens=int(cost * 1000),
                        accepted=True,
                        kinds={cell: 1},
                    )
                )
    return records


def test_low_effort_configuration_is_not_silently_dropped_from_the_spread():
    df = build_frame(_low_included_records())
    surface = cost_surface(df, n_boot=100, seed=3)

    # Full coverage: exactly the reviewer's reproduction (n_rows_used ==
    # n_rows, no exclusions) even though the fix changes the printed spread.
    assert surface["n_rows_used"] == surface["n_rows"]
    assert all(e["n"] == 0 for e in surface["exclusions"])

    # The .low configuration must still be IN the fitted table (never
    # dropped from row accounting -- only spread_S's own internal ratio
    # filter was ever in question).
    assert "cheap-model.low" in set(surface["table"]["arm"])

    # Sanity check against the exact pre-fix behavior: computing the spread
    # the old way (spread_S's own default, exclude_trivial=True) on this
    # same table reproduces the reviewer's ~2.0x number.
    old_buggy_spread = spread_S(
        surface["table"][["arm", "cost_per_accepted"]].rename(columns={"cost_per_accepted": "R"})
    )
    assert old_buggy_spread == pytest.approx(2.0, rel=0.2)

    # The fixed headline numbers must reflect the true ~20x range, not the
    # ~2x figure you get by silently excluding the cheapest (.low) row.
    assert surface["S_naive"] > 10
    assert surface["S_matched"] > 10
    assert surface["S_honest"] > 10
    assert surface["S_honest"] != pytest.approx(old_buggy_spread)

    # The walkdown line a reader actually sees must carry the true spread.
    line = walkdown_line(surface)
    assert "2.0x" not in line


# ---------------------------------------------------------------------------
# 7. no overlap cells at all: loud caveat, not a silent look-alike table
# ---------------------------------------------------------------------------


def test_no_overlap_cells_sets_overlap_caveat_and_walkdown_says_so():
    """FIX 4: the frozen no-overlap fallback must not look identical to the honest case."""
    records = []
    # Two eligible configurations that never share a task-mix cell.
    for j in range(6):
        records.append(_rec(f"a-{j}", "model-a", "medium", "ws", 1.0, 100, True, kinds={"only-a": 1}))
        records.append(_rec(f"b-{j}", "model-b", "medium", "ws", 4.0, 400, True, kinds={"only-b": 1}))
    df = build_frame(records)
    surface = cost_surface(df, min_n=5, min_overlap_cells=2, n_boot=20, seed=3)

    assert surface["overlap_restricted"] is False
    assert isinstance(surface["overlap_caveat"], str)
    assert surface["overlap_caveat"]  # non-empty

    # FIX 3 (final reviewer round, blocker): `shrunken_arm_estimates` still
    # runs its full pipeline on this fallback path (joint fit, James-Stein
    # shrinkage, mix standardization), so the numbers shown are fitted,
    # shrunken cost-per-accepted estimates -- never "raw per-run costs", the
    # old (false) claim.
    assert "cost-per-accepted estimates fitted without matching" in surface["overlap_caveat"]
    assert "task-mix matching was not possible" in surface["overlap_caveat"]
    assert "raw per-run costs" not in surface["overlap_caveat"]
    assert "raw" not in surface["overlap_caveat"].lower()

    line = walkdown_line(surface)
    assert "not adjusted for task mix (no shared task-mix cells)" in line

    steps = walkdown(surface)
    matched_note = steps[1]["note"]
    honest_note = steps[2]["note"]
    assert "no matching on task mix was possible" in matched_note

    # FIX 4 (final reviewer round, blocker): pooling (James-Stein shrinkage
    # toward the group) DOES run on this unmatched fit -- only the task-mix
    # matching step is unavailable. The old "no pooling ... was possible"
    # claim asserted a step never ran when it did.
    assert "task-mix matching was unavailable" in honest_note
    assert "pooling of thin configurations still happened on this unmatched fit" in honest_note
    assert "no pooling of thin configurations was possible" not in honest_note


def test_overlap_caveat_is_none_when_overlap_exists():
    df = build_frame(_varied_records())  # shares 4 cells across both configurations
    surface = cost_surface(df, n_boot=20, seed=3)
    assert surface["overlap_restricted"] is True
    assert surface["overlap_caveat"] is None
    assert "not adjusted for task mix" not in walkdown_line(surface)


# ---------------------------------------------------------------------------
# 8. naive ledger preserves exact float dollar sums, renamed off "tokens"
# ---------------------------------------------------------------------------


def test_naive_ledger_preserves_float_dollar_sums_not_truncated():
    """FIX 5: naive_arm_estimates' int-cast ledger columns must not truncate dollars."""
    records = [
        _rec(f"r-{j}", "model-a", "medium", "ws", 3.70, 100, True, kinds={"c1": 1}) for j in range(6)
    ]
    df = build_frame(records)
    surface = cost_surface(df, min_n=5, min_overlap_cells=1, n_boot=5, seed=1)

    naive = surface["naive"].set_index("arm")
    assert "total_output_tokens" not in surface["naive"].columns
    assert "known_output_tokens" not in surface["naive"].columns
    assert "median_output_tokens" not in surface["naive"].columns

    row = naive.loc["model-a.medium"]
    assert row["total_cost"] == pytest.approx(3.70 * 6)
    assert row["known_cost"] == pytest.approx(3.70 * 6)
    assert row["median_cost"] == pytest.approx(3.70)
    # The bug under test: int(3.70 * 6) == int(22.2) == 22, a silent truncation.
    assert row["total_cost"] != 22


# ---------------------------------------------------------------------------
# 9. S_naive is computed on the same population as S_matched / S_honest
# ---------------------------------------------------------------------------


def test_s_naive_matches_shrunk_population_not_every_min_n_config():
    """FIX 6: the walkdown's first step must describe the same configurations as the rest."""
    records = []
    # Two overlapping, well-populated configurations sharing 3 cells.
    for cell in ("c1", "c2", "c3"):
        for j in range(3):
            records.append(_rec(f"a-{cell}-{j}", "model-a", "medium", "ws", 1.0, 100, True, kinds={cell: 1}))
            records.append(_rec(f"b-{cell}-{j}", "model-b", "medium", "ws", 4.0, 400, True, kinds={cell: 1}))
    # A third configuration that meets min_n but shares no overlap cell with
    # anyone (its own private cell): naive_arm_estimates includes it (only
    # checks min_n); shrunken_arm_estimates excludes it (overlap-restricted).
    for j in range(6):
        records.append(_rec(f"d-{j}", "model-d", "medium", "ws", 100.0, 10000, True, kinds={"only-d": 1}))

    df = build_frame(records)
    surface = cost_surface(df, min_n=5, min_overlap_cells=2, n_boot=20, seed=9)

    naive_arms = set(surface["naive"]["arm"])
    shrunk_arms = set(surface["table"]["arm"])
    assert "model-d.medium" in naive_arms
    assert "model-d.medium" not in shrunk_arms  # confirms the two populations really differ

    assert "S_naive_all_configs" in surface
    assert surface["S_naive"] != surface["S_naive_all_configs"]

    restricted_naive = surface["naive"][surface["naive"]["arm"].isin(shrunk_arms)]
    expected = spread_S(
        restricted_naive[["arm", "naive_tokens_per_accepted"]].rename(
            columns={"naive_tokens_per_accepted": "R"}
        )
    )
    assert surface["S_naive"] == expected
    assert surface["S_naive"] == pytest.approx(4.0)  # model-b (4) / model-a (1), d excluded
    assert surface["S_naive_all_configs"] == pytest.approx(100.0)  # model-d (100) / model-a (1)


# ---------------------------------------------------------------------------
# 10. walkdown basis noun tracks cost_col; never says "dollars" on tokens
# ---------------------------------------------------------------------------


def test_walkdown_token_basis_never_says_dollars():
    """FIX 8: the naive step's unit must track cost_col, not be hardcoded to dollars."""
    df = build_frame(_varied_records())
    surface = cost_surface(df, cost_col=COST_TOKENS, n_boot=50, seed=11)

    line = walkdown_line(surface)
    assert "dollars" not in line.lower()
    assert "output tokens" in line.lower()

    for step in walkdown(surface):
        assert "dollars" not in step["note"].lower()


# ---------------------------------------------------------------------------
# 11. band honesty: estimate_outside_band, band_note, band_support_note
#
# A percentile bootstrap on a ratio estimator can, and routinely does, place
# the point estimate outside its own band: resampling whole cells trims cell
# diversity, and by Jensen's inequality the mean of the resampled ratios sits
# above the once-computed ratio. These tests do not change that behavior
# (the band stays the same percentile band E0 always produced); they check
# that the surface names it honestly instead of leaving a row that looks like
# a bug to speak for itself.
# ---------------------------------------------------------------------------


def _outside_band_records():
    """Small, high-variance, two-cell fixture where the fixed bootstrap
    parameters used below routinely place a configuration's point estimate
    outside its own band -- the systematic effect this module now names. Two
    thin cells and a wide multiplicative jitter (independent, seeded jitter,
    like `_varied_records`'s) keep the cell-cluster bootstrap's resampled
    ratios volatile enough to reproduce it deterministically.
    """
    rng = random.Random(5)
    cells = ["c0", "c1"]
    records = []
    i = 0
    for model, base_cost in (("cheap-model", 2.0), ("volatile-model", 5.0)):
        for cell in cells:
            n = rng.randint(1, 5)
            for _ in range(n):
                jitter = 2 ** (rng.gauss(0, 1.0))
                cost = base_cost * jitter
                i += 1
                records.append(
                    _rec(
                        run_id=f"ob{i}",
                        model=model,
                        effort="medium",
                        workspace="proj",
                        usd=cost,
                        out_tokens=int(cost * 1000) + 1,
                        accepted=True,
                        kinds={cell: 1},
                    )
                )
    return records


def _full_support_records():
    """Six shared cells, five well-behaved rows each, deliberately low
    jitter: every configuration keeps at least two overlap cells in every
    resample, so every one of `n_boot` draws survives eligibility and backs
    the band. Used for the case where `band_n_draws_min == n_boot` and there
    is nothing for `band_support_note` to say.
    """
    rng = random.Random(7)
    cells = [f"c{i}" for i in range(6)]
    records = []
    i = 0
    for model, base_cost in (("cheap-model", 2.0), ("pricey-model", 8.0)):
        for cell in cells:
            for _ in range(5):
                jitter = 1.0 + (rng.random() - 0.5) * 0.3
                cost = base_cost * jitter
                i += 1
                records.append(
                    _rec(
                        run_id=f"fs{i}",
                        model=model,
                        effort="medium",
                        workspace="proj",
                        usd=cost,
                        out_tokens=int(cost * 1000) + 1,
                        accepted=True,
                        kinds={cell: 1},
                    )
                )
    return records


def test_estimate_outside_band_is_flagged_counted_and_explained():
    df = build_frame(_outside_band_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=500, seed=20260831, ci=0.80)
    table = surface["table"]

    assert "estimate_outside_band" in table.columns
    n_flagged = int(table["estimate_outside_band"].sum())
    assert n_flagged > 0  # the fixture is built to reproduce the effect
    assert surface["n_estimate_outside_band"] == n_flagged

    # The flag must agree row by row with the actual outside-band condition,
    # never the reverse of it.
    for _, row in table.iterrows():
        outside = (row["cost_per_accepted"] < row["lo"]) or (row["cost_per_accepted"] > row["hi"])
        assert bool(row["estimate_outside_band"]) == outside

    note = surface["band_note"]
    assert note is not None
    assert str(n_flagged) in note
    assert "ratio" in note.lower()
    assert "resampling" in note.lower()
    assert "*" in note  # names the table marker it is explaining


def test_estimate_outside_band_false_not_nan_when_band_missing():
    """A configuration with no surviving bootstrap draw (no band at all) must
    read False, never True and never NaN, for `estimate_outside_band`.
    """
    # min_overlap_cells=99 is unreachable, so shrunken_arm_estimates' fallback
    # never restricts to overlap and every resample can still refit, but no
    # arm draw is guaranteed to survive; use a plain min_n-only fit instead to
    # force a missing band deterministically: n_boot=0 draws means no arm ever
    # gets a band at all.
    df = build_frame(_varied_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=0, seed=1)
    table = surface["table"]

    assert not table.empty
    assert table["lo"].isna().all() and table["hi"].isna().all()
    assert table["estimate_outside_band"].isin([False]).all()
    assert not table["estimate_outside_band"].isna().any()
    assert surface["n_estimate_outside_band"] == 0
    assert surface["band_note"] is None


def test_all_estimates_inside_band_band_note_is_none():
    df = build_frame(_varied_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=500, seed=42, ci=0.80)
    table = surface["table"]

    assert not table["estimate_outside_band"].any()
    assert surface["n_estimate_outside_band"] == 0
    assert surface["band_note"] is None


def test_band_n_draws_min_max_reported_and_support_note_none_when_full():
    df = build_frame(_full_support_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=500, seed=42, ci=0.80)

    assert surface["band_n_draws_min"] == 500
    assert surface["band_n_draws_max"] == 500
    assert surface["band_support_note"] is None


def test_band_support_note_names_min_and_total_when_support_is_thin():
    df = build_frame(_outside_band_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=500, seed=20260831, ci=0.80)

    band_min = surface["band_n_draws_min"]
    assert band_min is not None
    assert band_min < surface["n_boot"]

    note = surface["band_support_note"]
    assert note is not None
    assert str(band_min) in note
    assert str(surface["n_boot"]) in note
    assert "minimum run count" in note
    assert "task-mix cells" in note


def test_band_notes_have_no_forbidden_vocabulary_or_em_dash():
    df = build_frame(_outside_band_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=500, seed=20260831, ci=0.80)

    for text in (surface["band_note"], surface["band_support_note"]):
        assert text is not None
        assert "—" not in text
        lowered = text.lower()
        for word in (
            "knowledge gradient", "posterior", "prior", "experimental design",
            "value of information", "bandit", "arms",
        ):
            assert word not in lowered


# ---------------------------------------------------------------------------
# Final reviewer fix round, FIX 3 (blocker): a resampled per-configuration
# estimate of exactly zero is a valid bootstrap draw, not an invalid one; a
# NaN/infinite/negative one is invalid and must be counted, never silently
# dropped from `n_draws` with no trace.
# ---------------------------------------------------------------------------


def test_bootstrap_bands_keeps_exact_zero_draws_not_silently_dropped(monkeypatch):
    """The old filter (`val > 0`) silently dropped a draw of exactly zero,
    shrinking that configuration's `n_draws` and biasing its band upward with
    no sign anything was excluded.

    Reaching a bit-exact 0.0 through the real OLS + James-Stein pipeline is
    not reliable: even an arm whose every cost is genuinely zero comes back
    from `shrunken_arm_estimates` as float noise at the 1e-13 to 1e-16 scale
    (positive or negative depending on the resample), not a literal 0.0 --
    that is an artifact of the frozen estimator's linear algebra, not
    something this fix should have to fight. So this controls
    `shrunken_arm_estimates`'s return value directly, to prove the FILTER
    inside `_bootstrap_bands` treats a literal 0.0 as valid, independent of
    whether the estimator itself can ever land exactly on it.
    """

    def _fake_shrunk(df, min_n, min_overlap_cells, cell_col):
        return pd.DataFrame({"arm": ["zero-arm", "base-arm"], "R": [0.0, 100.0]})

    monkeypatch.setattr(surface_mod, "shrunken_arm_estimates", _fake_shrunk)

    view = pd.DataFrame(
        {
            "arm": ["zero-arm", "base-arm"],
            "cell": ["c1", "c1"],
            "output_tokens": [0.0, 100.0],
            "accepted": [True, True],
            "proxy_known": [True, True],
        }
    )
    bands = surface_mod._bootstrap_bands(
        view, n_boot=50, seed=1, ci=0.80, min_n=1, min_overlap_cells=1
    )
    per_arm = bands["per_arm"].set_index("arm")

    assert per_arm.loc["zero-arm", "n_draws"] == 50  # every draw kept; the bug would make this 0
    assert per_arm.loc["zero-arm", "lo"] == 0.0
    assert per_arm.loc["zero-arm", "hi"] == 0.0
    assert per_arm.loc["zero-arm", "n_invalid_draws"] == 0


def test_bootstrap_bands_counts_invalid_draws_instead_of_silently_dropping():
    """An arm whose known-outcome rows are ALL rejected has acceptance rate
    exactly 0 (bit-exact: the mean of an all-`False` boolean array), so its
    resampled cost-per-accepted estimate is `R = level / 0` -- infinite, a
    genuinely unusable draw, distinct from a valid zero. The old code
    already excluded it (`np.isfinite` was already checked), but silently:
    nothing counted it. The fix must count it in `n_invalid_draws`.
    """
    rows = []
    for cell in ("c1", "c2"):
        for _ in range(5):
            rows.append(
                {
                    "arm": "rejected-arm",
                    "cell": cell,
                    "output_tokens": 100.0,
                    "accepted": False,
                    "proxy_known": True,
                }
            )
            rows.append(
                {
                    "arm": "base-arm",
                    "cell": cell,
                    "output_tokens": 100.0,
                    "accepted": True,
                    "proxy_known": True,
                }
            )
    view = pd.DataFrame(rows)

    bands = surface_mod._bootstrap_bands(
        view, n_boot=200, seed=1, ci=0.80, min_n=5, min_overlap_cells=2
    )
    per_arm = bands["per_arm"].set_index("arm")

    assert "rejected-arm" in per_arm.index
    row = per_arm.loc["rejected-arm"]
    assert row["n_draws"] == 0  # never a usable estimate
    assert row["n_invalid_draws"] > 0  # but counted, not silently absent
    # base-arm is unaffected: its own draws are untouched by rejected-arm's
    # invalid ones.
    assert per_arm.loc["base-arm", "n_invalid_draws"] == 0
    assert per_arm.loc["base-arm", "n_draws"] > 0


def test_band_n_invalid_draws_column_present_and_zero_when_well_behaved():
    """Public-API wiring check: on ordinary, well-behaved data, the new
    column exists and is zero everywhere, and the total is zero. Uses
    `_full_support_records` (every resample keeps every configuration
    banded) so `band_support_note` staying `None` is attributable to the
    zero invalid count, not to some other, unrelated thin-band cause.
    """
    df = build_frame(_full_support_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=200, seed=42, ci=0.80)
    table = surface["table"]

    assert "band_n_invalid_draws" in table.columns
    assert (table["band_n_invalid_draws"].fillna(0) == 0).all()
    assert surface["band_n_invalid_draws_total"] == 0
    # Unaffected: a zero invalid count must not, by itself, manufacture a
    # support note the pre-fix behavior never printed either.
    assert surface["band_support_note"] is None


def test_cost_surface_surfaces_nonzero_invalid_draws_in_support_note(monkeypatch):
    """Wiring check with a controlled invalid count injected at the
    `_bootstrap_bands` boundary: `cost_surface` must carry it into the
    per-configuration table AND mention it in `band_support_note`, even when
    the (real) bands underneath are otherwise full (`band_n_draws_min ==
    n_boot`), where the pre-fix support note would have stayed `None`.
    """
    df = build_frame(_full_support_records())
    real_bootstrap = surface_mod._bootstrap_bands

    def _patched(*args, **kwargs):
        bands = dict(real_bootstrap(*args, **kwargs))
        per_arm = bands["per_arm"].copy()
        per_arm.loc[per_arm["arm"] == "pricey-model.medium", "n_invalid_draws"] = 37
        bands["per_arm"] = per_arm
        return bands

    monkeypatch.setattr(surface_mod, "_bootstrap_bands", _patched)
    surface = surface_mod.cost_surface(df, cost_col=COST_DOLLARS, n_boot=500, seed=42, ci=0.80)
    table = surface["table"]

    row = table.set_index("arm").loc["pricey-model.medium"]
    assert int(row["band_n_invalid_draws"]) == 37
    assert surface["band_n_invalid_draws_total"] == 37
    assert surface["band_support_note"] is not None
    assert "37" in surface["band_support_note"]
    assert "—" not in surface["band_support_note"]


# ---------------------------------------------------------------------------
# Final reviewer fix round, FIX 5 (accepted in part): a run with no known
# workspace must not share a task-mix cell with any other such run.
# ---------------------------------------------------------------------------


def test_build_frame_no_workspace_rows_get_unique_unmatchable_cells():
    records = [
        _rec("r1", "opus-5", "medium", None, 1.0, 100, True, kinds={"py": 1}),
        _rec("r2", "opus-5", "medium", None, 1.0, 100, True, kinds={"py": 1}),
        _rec("r3", "opus-5", "medium", "proj", 1.0, 100, True, kinds={"py": 1}),
    ]
    df = build_frame(records)
    cells = df.set_index("run_id")["cell"]

    # Two workspace-less rows, identical in every other way, must never share
    # a cell -- the old `f"{workspace}|{kind}"` formatting gave them BOTH the
    # literal string "None|py".
    assert cells["r1"] != cells["r2"]
    assert "None" not in cells["r1"]
    assert "None" not in cells["r2"]
    assert cells["r1"] != cells["r3"]
    assert cells["r2"] != cells["r3"]


def test_cost_surface_names_no_workspace_rows_when_present():
    records = _varied_records() + [
        _rec("nw-1", "cheap-model", "medium", None, 2.0, 2000, True, kinds={"py": 1}),
        _rec("nw-2", "pricey-model", "medium", None, 8.0, 8000, True, kinds={"py": 1}),
    ]
    df = build_frame(records)
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=20, seed=1)

    assert surface["n_no_workspace"] == 2
    assert surface["no_workspace_note"] is not None
    assert "2" in surface["no_workspace_note"]
    assert "task-mix matching" in surface["no_workspace_note"]
    assert "—" not in surface["no_workspace_note"]


def test_cost_surface_no_workspace_note_absent_when_none():
    df = build_frame(_varied_records())
    surface = cost_surface(df, cost_col=COST_DOLLARS, n_boot=20, seed=1)

    assert surface["n_no_workspace"] == 0
    assert surface["no_workspace_note"] is None


def test_band_support_note_names_the_spread_bands_own_support():
    # The spread line prints its own 80% band, whose draw support was
    # computed and returned but never shown. SPEC section 0: support that
    # exists and is not printed is support that is hidden.
    from loopmath.surface import _band_support_note

    text = _band_support_note(393, 1000, 0, 992)
    assert "992 of 1000 draws" in text
    assert "—" not in text
    for term in ("knowledge gradient", "posterior", "bandit"):
        assert term not in text.lower()


def test_band_support_note_omits_the_spread_sentence_at_full_support():
    from loopmath.surface import _band_support_note

    text = _band_support_note(393, 1000, 0, 1000)
    assert "spread band" not in text
    assert "393 of 1000" in text


# ---------------------------------------------------------------------------
# PATCH 2 (reviewer round, HIGH): acceptance is measured by a different
# instrument (evidence tier) per configuration; disclose the tier
# composition and caveat it when it matters.
# ---------------------------------------------------------------------------


def _tier_mix_records():
    """Two configurations sharing task-mix cells, on two different tiers.

    `verified-model.medium` is 30/30 `verified` (accepted by construction,
    per grade.py); `heuristic-model.medium` is 30 `heuristic` rows with a
    genuine mix of accepted/rejected, so its acceptance rate is a different
    kind of measurement even though both configurations meet min_n and share
    cells.
    """
    cells = ["py", "md"]
    records = []
    i = 0
    for cell in cells:
        for _ in range(15):
            i += 1
            records.append(
                _rec(f"v{i}", "verified-model", "medium", "proj", 5.0, 5000, True,
                     kinds={cell: 1}, tier="verified")
            )
        for j in range(15):
            i += 1
            accepted = j % 3 != 0  # a genuine mix, not uniformly True
            records.append(
                _rec(f"h{i}", "heuristic-model", "medium", "proj", 5.0, 5000, accepted,
                     kinds={cell: 1}, tier="heuristic")
            )
    return records


def test_tier_composition_column_present_and_correct_per_configuration():
    df = build_frame(_tier_mix_records())
    surface = cost_surface(df, n_boot=20, seed=1)
    table = surface["table"].set_index("arm")

    assert table.loc["verified-model.medium", "tiers"] == "30v"
    assert table.loc["heuristic-model.medium", "tiers"] == "30h"


def test_tier_caveat_fires_when_verified_mixes_with_another_tier():
    df = build_frame(_tier_mix_records())
    surface = cost_surface(df, n_boot=20, seed=1)

    note = surface["tier_note"]
    assert note is not None
    lowered = note.lower()
    assert "verified" in lowered
    assert "accepted by construction" in lowered
    assert "—" not in note
    for term in ("knowledge gradient", "posterior", "prior", "experimental design",
                 "value of information", "bandit", "arms"):
        assert term not in lowered


def test_tier_caveat_absent_when_every_fitted_row_shares_one_tier():
    # _varied_records() (used throughout this file) stamps every row
    # "verified" by default -- a uniform tier population is measured by one
    # instrument throughout, so no confound is live and no caveat fires.
    df = build_frame(_varied_records())
    surface = cost_surface(df, n_boot=20, seed=1)
    assert surface["tier_note"] is None

    table = surface["table"].set_index("arm")
    assert table.loc["cheap-model.medium", "tiers"] == "40v"
    assert table.loc["pricey-model.medium", "tiers"] == "40v"
