"""Sweep token pricing and assembly support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .ingest.base import canonical_model
from .price import PriceTable, price_all, price_run, warning_lines

__all__ = [
    "DEFAULT_SWEEP_DIR", "PLAIN_VARIANT", "T7_VARIANT", "T1_T5_TASKS", "T7_TASKS",
    "E1_TASKS", "_TASK_VARIANT", "_TASK_PLANNER", "EFFORT_ORDER",
    "_SWEEP_TOKEN_FIELDS", "GPT56_LONG_CONTEXT_THRESHOLD", "_ATTEMPT_MODEL_PREFIX",
    "_parse_sweep_filename", "_map_sweep_tokens", "_price_attempt_tokens",
    "_attempt_model_name", "_price_run_total", "_validate_accepted",
    "_cost_price_note", "_long_context_note", "_acceptance_bias_note",
]


# The sweep folder is a config value, not a path in the package: None means
# "resolve per call" through research_paths.sweep_dir() (flag, then
# LOOPMATH_SWEEP_DIR, then research.sweep_dir in config.toml). Tests may set it.
DEFAULT_SWEEP_DIR: Path | None = None

# DATA DECISION B: the R1 slice is reviewer claude-opus-5-xhigh throughout,
# but t1-t5 and t7 each exist under only one variant of that family in this
# corpus snapshot -- see the module docstring.
PLAIN_VARIANT = "rev-claude-opus-5-xhigh"
T7_VARIANT = "plan-gpt-5.6-luna-low@rev-claude-opus-5-xhigh"

T1_T5_TASKS = ("t1-cli-tool", "t2-algorithm", "t3-bugfix", "t4-refactor", "t5-parser-api")
T7_TASKS = ("t7-dataflow", "t7-regex", "t7-rope", "t7-sched", "t7-sql")
E1_TASKS = T1_T5_TASKS + T7_TASKS

_TASK_VARIANT = {t: PLAIN_VARIANT for t in T1_T5_TASKS}
_TASK_VARIANT.update({t: T7_VARIANT for t in T7_TASKS})

_TASK_PLANNER = {t: "none" for t in T1_T5_TASKS}
_TASK_PLANNER.update({t: "gpt-5.6-luna-low" for t in T7_TASKS})

# Canonical effort order. The shared effort curve in the model is defined over
# exactly this order; the masking grammar validates efforts against it too.
EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")

_TABLE_COLUMNS = [
    "model", "effort", "task", "usd", "accepted",
    "run_id", "family", "task_group", "dev_attempts",
    "out_tokens", "total_tokens", "source_file",
    "usd_recorded", "usd_todo_rate", "usd_run_total",
    "planner", "variant",
]

# The sweep's own four token fields, one-to-one with price.py's four-stream
# contract (SPEC amendment; see `_map_sweep_tokens`).
_SWEEP_TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write")

# GPT-5.6's long-context surcharge (SPEC amendment): requests over roughly
# this many tokens are billed at a different rate that a per-attempt record
# cannot resolve (which side of the threshold a request's context fell on is
# not recorded). Documentation only -- `_long_context_note` does not use this
# value in a computation; it reuses `loopmath.price`'s own NOTE line (the module
# that actually owns the threshold and the price table) rather than trying
# to detect the crossing itself from a row's summed tokens, which is exactly
# the false precision this constant used to invite.
GPT56_LONG_CONTEXT_THRESHOLD = 272_000

# Decodes an attempt's compact `model` field (e.g. "opus5·xhigh",
# "fable·high") into the full model names used elsewhere in this table.
# Only needed for `usd_run_total`, which has to price planner and reviewer
# attempts too (the developer attempt's model is already known exactly from
# `arm.model`). Small, closed, audited vocabulary -- see the fix-round
# report; an unrecognized prefix returns None rather than guessing.
_ATTEMPT_MODEL_PREFIX = {
    "opus5": "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "fable": "claude-fable-5",
    "luna": "gpt-5.6-luna",
    "sol": "gpt-5.6-sol",
    "terra": "gpt-5.6-terra",
}


def _parse_sweep_filename(name: str) -> dict | None:
    """Split `<task>--<model>--<effort>@<variant>.run.json` into parts.

    Task and model names in this corpus use single hyphens only, so splitting
    the stem on `--` yields exactly three parts. The effort/variant boundary
    is the *first* `@`, since a plan-variant name can itself contain another
    `@` (e.g. `plan-gpt-5.6-luna-low@rev-claude-opus-5-xhigh`) that must stay
    part of the variant string, not be split further.
    """
    suffix = ".run.json"
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    parts = stem.split("--")
    if len(parts) != 3:
        return None
    task, model, rest = parts
    if "@" not in rest:
        return None
    effort, variant = rest.split("@", 1)
    return {"task": task, "model": model, "effort": effort, "variant": variant}


def _map_sweep_tokens(raw_tokens) -> dict[str, float] | None:
    """Map the sweep's four token fields one-to-one onto price.py's
    four-stream contract (SPEC amendment): `in = input`, `cache_read =
    cache_read`, `cache_write = cache_write`, `out = output`. Nothing is
    summed -- an earlier version of this mapping folded cache writes into
    `in`, which is wrong now that cache writes and cache reads are priced
    very differently (a cache write can run up to 20x a cache read on
    Anthropic; OpenAI charges nothing for either).

    Returns None -- a signal to exclude, never to price at zero -- when the
    block is missing, is not a dict, is missing one of the four sweep
    fields, or has a non-numeric, non-finite, or negative value in one of
    them. A field that is present and legitimately `0` (seen for
    `cache_write` in this corpus) is used normally.
    """
    if not isinstance(raw_tokens, dict):
        return None
    values: dict[str, float] = {}
    for field in _SWEEP_TOKEN_FIELDS:
        if field not in raw_tokens:
            return None
        try:
            v = float(raw_tokens[field])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(v) or v < 0:
            return None
        values[field] = v
    return {
        "in": values["input"],
        "cache_read": values["cache_read"],
        "cache_write": values["cache_write"],
        "out": values["output"],
    }


def _price_attempt_tokens(attempt: dict, model: str | None, table: PriceTable) -> dict:
    """Price one sweep attempt's own tokens against `table`.

    Returns the same shape `loopmath.price.price_run` returns (`usd`,
    `usd_breakdown`, `priced`, `todo`, `model`, `reason`), so every attempt
    (developer, planner, reviewer) is priced through one uniform path.
    """
    ext = attempt.get("ext") or {}
    mapped = _map_sweep_tokens(ext.get("tokens"))
    if mapped is None:
        return {
            "usd": None,
            "usd_breakdown": None,
            "priced": False,
            "todo": False,
            "model": model,
            "reason": "attempt has a missing or malformed tokens block",
        }
    return price_run({"model": model, "tokens": mapped}, table)


def _attempt_model_name(raw: str | None) -> str | None:
    """Decode a sweep attempt's compact `model` field into a full model
    name (`_ATTEMPT_MODEL_PREFIX`). Returns None on an unrecognized prefix.
    """
    if not raw:
        return None
    prefix = str(raw).split("·", 1)[0].strip().lower()
    return _ATTEMPT_MODEL_PREFIX.get(prefix)


def _price_run_total(record: dict, dev_usd: float, table: PriceTable) -> tuple[float | None, str | None]:
    """FLAG 1 / FLAG 2: sum priced tokens over every attempt in the run.

    Starts from `dev_usd` (the developer attempt(s), already priced by the
    caller) and adds every planner and reviewer attempt, each priced from
    its own tokens. GATE attempts (`actor == "harness"`, no `ext` at all)
    carry no model cost and are simply not part of the sum -- that is what
    a harness check step is, not a data defect.

    Returns `(usd_run_total, reason)`. `reason` is None on success. When any
    planner/reviewer attempt cannot be priced (unrecognized model label,
    missing/malformed tokens, or a model absent from the price table),
    returns `(None, reason)`: never a partial sum presented as a total.
    """
    total = dev_usd
    for task in record.get("tasks", []):
        if task.get("owner") not in ("planner", "reviewer"):
            continue
        for attempt in task.get("attempts", []):
            if attempt.get("actor") not in ("planner", "reviewer"):
                continue
            model = canonical_model(_attempt_model_name(attempt.get("model")))
            priced = _price_attempt_tokens(attempt, model, table)
            if not priced["priced"]:
                reason = priced.get("reason") or "planner/reviewer attempt could not be priced"
                return None, reason
            total += priced["usd"]
    return total, None


def _validate_accepted(raw) -> tuple[bool | None, bool]:
    """FLAG 3: strict acceptance validation.

    Returns `(value, ok)`. `ok` is False for anything that is not a real
    `bool`, the exact lowercased strings `"true"`/`"false"`, or the ints
    `0`/`1` -- including `None`. Never guesses: `bool()` on a non-empty
    string, or on `None`, is exactly the bug this replaces.
    """
    if isinstance(raw, bool):
        return raw, True
    if isinstance(raw, int):
        if raw in (0, 1):
            return bool(raw), True
        return None, False
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s == "true":
            return True, True
        if s == "false":
            return False, True
        return None, False
    return None, False


def _cost_price_note(df: pd.DataFrame, table: PriceTable) -> tuple[str, dict]:
    """DATA DECISION A's mandatory honesty note (plain language, printed by
    `run_fit`, also recorded in `df.attrs["assembly"]`): `usd` comes from
    token counts against the packaged price table, never from the sweep's
    own recorded cost. Under the amended (four-stream) price table, the two
    are no longer expected to disagree: the table was itself derived from
    real billing data (see `prices.toml`'s header), so on rows that carry a
    recorded cost, the token-derived figure is the same computation done
    twice. This note reports that as a *measurement*: the exact-match count
    and the worst residual are computed here at runtime from whichever rows
    in `df` carry both figures, never hardcoded (an earlier draft of this
    function reported a per-model ratio framed as "disagreement" -- that was
    true against the pre-amendment table's placeholder rates and is false
    now, so it is gone, not just softened). Also reports how many rows are
    priced from a placeholder rate.
    """
    as_of = table.as_of
    if df.empty:
        return (
            f"usd is computed from token counts against the packaged price "
            f"table (as of {as_of}), not from the sweep's own recorded cost. "
            f"No rows were assembled, so there is nothing to compare.",
            {},
        )

    has_both = df[df["usd_recorded"].notna()]
    todo_count = int(df["usd_todo_rate"].sum())

    stats: dict = {"n_compared": int(len(has_both))}
    if len(has_both):
        residuals = (has_both["usd"] - has_both["usd_recorded"]).abs()
        # A tenth of a cent absorbs float round-trip noise (TOML rate ->
        # float -> per-attempt product -> row sum) without hiding a real
        # mismatch; nothing else in this table is audited below the cent.
        tolerance = 0.001
        n_exact = int((residuals <= tolerance).sum())
        worst = float(residuals.max())
        stats["n_exact_match"] = n_exact
        stats["worst_residual"] = worst
        match_text = (
            f" On the {len(has_both)} rows where the sweep also recorded a cost, the "
            f"token-derived figure reproduces it (within a tenth of a cent) on {n_exact} "
            f"of them; the worst residual measured here is ${worst:.4f}."
        )
    else:
        match_text = " No rows in this table carried both figures to compare."

    note = (
        f"usd is computed from every row's own token counts against the packaged "
        f"price table (as of {as_of}), not taken from the sweep's own recorded "
        f"cost_usd." + match_text + f" {todo_count} of {len(df)} rows are priced "
        f"from a placeholder (todo) rate, not a confirmed one."
    )
    return note, stats


def _long_context_note(dev_price_records: list[dict], table: PriceTable) -> str:
    """SPEC amendment caveat: GPT-5.6 bills a long-context surcharge above
    roughly `GPT56_LONG_CONTEXT_THRESHOLD` tokens per request that this
    table cannot resolve from a per-attempt record.

    Reused, not restated: an earlier version of this function counted rows
    whose own `total_tokens` crossed the threshold, which claims knowledge
    this data does not support -- the threshold is about a single request's
    context size, not an attempt's summed tokens across however many calls
    it made. `loopmath.price.price_all` / `loopmath.price.warning_lines` (owned by
    another module) already say this correctly, at the granularity they can
    actually stand behind (how many priced attempts used a gpt-5.6-* model,
    not which ones crossed the threshold): this calls that same path over
    the developer-attempt records that fed this table's `usd` column and
    takes its NOTE line verbatim, so the caveat is worded once, by the
    module that owns the price table.
    """
    _, warnings = price_all(dev_price_records, table)
    lines = [ln for ln in warning_lines(warnings, table) if ln.startswith("NOTE:")]
    if lines:
        return lines[0]
    return (
        "No gpt-5.6-* developer attempt is priced into this table, so the "
        "long-context surcharge caveat (loopmath.price's NOTE line) does not apply here."
    )


def _acceptance_bias_note(dropped_zero_tokens_outcomes: dict, df: pd.DataFrame) -> str:
    """PATCH 4 disclosure: the `dev_zero_tokens` rows are dropped from the
    table entirely (cost is undefined for `log10(usd)` at 0), but their
    `accepted` outcome was real and known before that drop -- and this drop
    is not outcome-neutral. Measured, not asserted: this compares the
    rejection rate among the dropped rows to the rejection rate among the
    kept rows the acceptance head actually sees, and names the direction of
    the bias plainly rather than only counting the drop.

    Keeping these rows in the acceptance likelihood while excluding them
    only from the cost likelihood would need two different row sets feeding
    one PyMC model (one masked to rows with a usable cost, one not, but both
    sharing the same model/effort/task coords and the same held-out scoring
    path in `predict`/`transfer_test`); `build_model` does not support that
    split today, and bolting it on under this fix round risks introducing a
    new, harder-to-see bug into the cost head and the masking/transfer-test
    plumbing for a comparatively small (23-row) fix. So the corrected
    behavior here is disclosure, not re-inclusion: this note is threaded to
    every place acceptance is reported (`run_fit` prints it; `loopmath
    transfer-test` always calls `run_fit` first).
    """
    n_rejected = dropped_zero_tokens_outcomes.get("rejected", 0)
    n_accepted = dropped_zero_tokens_outcomes.get("accepted", 0)
    n_dropped = n_rejected + n_accepted
    if n_dropped == 0:
        return (
            "No rows were dropped for pricing to exactly $0.00 in this table, so "
            "there is no zero-token-row exclusion bias to report against the "
            "acceptance head."
        )

    dropped_reject_rate = n_rejected / n_dropped
    kept_reject_rate = (
        float((~df["accepted"]).mean()) if not df.empty else float("nan")
    )
    direction = (
        "inflates the modelled acceptance rate (makes it look higher than the "
        "full corpus would)"
        if dropped_reject_rate > kept_reject_rate
        else "deflates the modelled acceptance rate (makes it look lower than the "
        "full corpus would)"
    )
    return (
        f"{n_dropped} rows ({n_rejected} rejected, {n_accepted} accepted) were dropped "
        f"from this table for pricing to exactly $0.00 (cost is undefined for "
        f"log10(usd) there), and their acceptance outcome was dropped along with "
        f"them: {dropped_reject_rate * 100:.0f}% of the dropped rows were rejected, "
        f"against {kept_reject_rate * 100:.0f}% of the {len(df)} kept rows the "
        f"acceptance head actually fits. Removing a rejection-heavy set before the "
        f"acceptance head sees it {direction}."
    )
