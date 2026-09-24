"""Input-frame construction for the cost surface."""

from __future__ import annotations

import numpy as np
import pandas as pd


_FRAME_COLUMNS = [
    "run_id", "harness", "arm", "model", "effort", "workspace", "cell",
    "usd", "out_tokens", "total_tokens", "tokens_complete", "accepted", "tier",
    "proxy_known", "wall_s", "ts", "priced",
]

def _coerce_token_stream(value) -> float:
    """`value` as a float when it is a finite, non-negative number; NaN otherwise.

    A missing key, a non-numeric value, a negative value, or a non-finite
    value (inf/NaN) all come back as `float("nan")` -- "no usable count" --
    rather than being defaulted to 0. Zero tokens is a real, legitimate
    value; a missing or malformed count is not, and treating the two the
    same would understate any total built from this stream (SPEC sections
    0 and 5). `bool` is excluded even though it is technically an `int`
    subclass in Python, since a stray `True`/`False` in a token field is a
    data error, not a count.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    if not np.isfinite(value) or value < 0:
        return float("nan")
    return float(value)


def _primary_kind(written_file_kinds: dict) -> str:
    """The extension with the highest write count, ties broken alphabetically.

    `"no-writes"` when the histogram is empty. This, paired with the
    workspace, is the task-mix cell: the only shape of a task a metadata-only
    parse can see is where the writes landed and what kind of files they were.
    """
    if not written_file_kinds:
        return "no-writes"
    return sorted(written_file_kinds.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def build_frame(records: list[dict]) -> pd.DataFrame:
    """One row per graded, priced run. Every input record keeps a row.

    Columns: run_id, harness, arm, model, effort, workspace, cell, usd,
    out_tokens, total_tokens, tokens_complete, accepted, tier, proxy_known,
    wall_s, ts, priced.

    Input records are `RunRecord.to_dict()` dicts already run through
    `grade.grade_all` (so `record["grade"] = {"accepted", "tier", "signal"}`)
    and `price.price_all` (so `record["usd"]` is a float or None). Every key
    is read with `.get` since a record may be missing fields this module
    does not itself require.

    `arm` is `f"{model}.{effort}"` when both are known, `f"{model}.?"` when
    effort is unknown, and None when model is unknown -- a run with no model
    label cannot be grouped into any workflow configuration at all, but the
    row is kept (never dropped) so `cost_surface` can count and name it as
    the `"no model label"` exclusion instead of it silently vanishing before
    accounting even starts. (A frame built entirely from no-model records
    would give pandas an all-None `arm` column, which comes back as a
    float64 NaN column rather than an object column of `None`; `.isna()`
    catches both, so every consumer of `arm` in this module treats them the
    same. The one thing to never let happen is a null arm being stringified
    into the literal text "None" -- this function never calls `str()` on
    `arm`, so that cannot occur here.)

    Rows with an unpriced run (`usd is None`) are kept, with `usd` set to
    NaN, so a caller can count and name them; `cost_surface` is what excludes
    them from a dollar fit. The four token streams (`in`, `cache_read`,
    `cache_write`, `out`) are each coerced through `_coerce_token_stream`: a
    missing key or a non-numeric/negative value becomes NaN rather than a
    false zero, because a genuinely-zero token count and a missing one are
    different facts and conflating them understates any total built from
    them. `out_tokens` is therefore NaN whenever the `out` stream itself is
    missing or invalid; `total_tokens` is NaN whenever any of the four is.
    `tokens_complete` is True only when all four streams were usable, so a
    caller can tell "zero tokens" apart from "unknown token count" without
    inspecting NaN directly.
    """
    rows: list[dict] = []
    for i, rec in enumerate(records):
        model = rec.get("model")
        effort = rec.get("effort")
        if model is None:
            arm = None  # no model label: kept, but cannot be placed in any configuration
        else:
            arm = f"{model}.{effort}" if effort is not None else f"{model}.?"

        workspace = rec.get("workspace")
        kinds = rec.get("written_file_kinds") or {}
        if workspace is None:
            # FIX 5 (reviewer round, accepted in part): `f"{workspace}|..."`
            # with `workspace=None` stringifies to the literal text "None",
            # which would put every workspace-less run into the SAME
            # task-mix cell as every other workspace-less run, regardless of
            # what harness, model, or kind of file it touched -- letting a
            # metadata-only parse match runs that share nothing in common. A
            # cell keyed on this row's own position can never equal another
            # row's cell, so a workspace-less run is correctly treated as
            # unmatchable on task mix rather than falsely matchable against
            # an arbitrary other workspace-less run. `cost_surface` counts
            # these rows (`n_no_workspace`) and names them in the report
            # when the count is non-zero.
            cell = f"__no_workspace__:{i}"
        else:
            cell = f"{workspace}|{_primary_kind(kinds)}"

        usd = rec.get("usd")
        priced = usd is not None

        tokens = rec.get("tokens") if isinstance(rec.get("tokens"), dict) else {}
        in_tok = _coerce_token_stream(tokens.get("in"))
        cache_read_tok = _coerce_token_stream(tokens.get("cache_read"))
        cache_write_tok = _coerce_token_stream(tokens.get("cache_write"))
        out_tok = _coerce_token_stream(tokens.get("out"))
        tokens_complete = not (
            np.isnan(in_tok)
            or np.isnan(cache_read_tok)
            or np.isnan(cache_write_tok)
            or np.isnan(out_tok)
        )
        total_tok = (
            (in_tok + cache_read_tok + cache_write_tok + out_tok)
            if tokens_complete
            else float("nan")
        )

        grade = rec.get("grade") or {}
        accepted = grade.get("accepted")

        rows.append(
            {
                "run_id": rec.get("run_id"),
                "harness": rec.get("harness"),
                "arm": arm,
                "model": model,
                "effort": effort,
                "workspace": workspace,
                "cell": cell,
                "usd": float(usd) if priced else float("nan"),
                "out_tokens": out_tok,
                "total_tokens": total_tok,
                "tokens_complete": tokens_complete,
                "accepted": accepted,
                "tier": grade.get("tier"),
                "proxy_known": accepted is not None,
                "wall_s": rec.get("wall_s"),
                "ts": rec.get("ts"),
                "priced": priced,
            }
        )
    return pd.DataFrame(rows, columns=_FRAME_COLUMNS)
