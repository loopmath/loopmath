"""E1 sweep-record selection and table assembly."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import research_paths
from .ingest.base import canonical_model
from .price import PriceTable, load_prices
from .fit_pricing import (
    DEFAULT_SWEEP_DIR, E1_TASKS, _TASK_PLANNER, _TASK_VARIANT,
    _acceptance_bias_note, _cost_price_note, _long_context_note,
    _map_sweep_tokens, _parse_sweep_filename, _price_attempt_tokens,
    _price_run_total, _validate_accepted,
)

__all__ = ["_TABLE_COLUMNS", "assemble_table"]


_TABLE_COLUMNS = [
    "model", "effort", "task", "usd", "accepted",
    "run_id", "family", "task_group", "dev_attempts",
    "out_tokens", "total_tokens", "source_file",
    "usd_recorded", "usd_todo_rate", "usd_run_total",
    "planner", "variant",
]

def assemble_table(
    sweep_dir: str | Path | None = None,
    tasks: tuple[str, ...] = E1_TASKS,
    price_table: PriceTable | None = None,
) -> pd.DataFrame:
    """The E1 model table: rows = runs, cols = model, effort, task, usd, accepted.

    Extra columns kept for diagnostics: run_id, family, task_group,
    dev_attempts, out_tokens, total_tokens, source_file, usd_recorded,
    usd_todo_rate, usd_run_total, planner, variant.

    Selection (DATA DECISION B): a t1-t5 file is included when its filename
    variant equals `PLAIN_VARIANT` exactly; a t7-* file is included when its
    variant equals `T7_VARIANT` exactly. t1-t5 are never also pulled from
    the other planner-pinned variant that also carries them -- that would
    double the rows and confound planner with task. `planner` records "none"
    or "gpt-5.6-luna-low"; `variant` records the full matched variant
    string, so the difference is visible instead of hidden in the task
    effect.

    Cost rule (DATA DECISION A): `usd` is computed from every developer
    attempt's own token counts against `price_table` (the packaged table by
    default), summed across retries within the run. `usd_recorded` is the
    sweep's own recorded cost (summed the same way) when *every* developer
    attempt in the run carries one, else None; it is never mixed into `usd`.
    `usd_todo_rate` marks a row priced from a placeholder rate.
    `usd_run_total` additionally sums every planner/developer/reviewer
    attempt in the run (FLAG 1); it is None, and counted, for a run where
    any of those attempts cannot be priced (FLAG 2) -- never a partial sum.

    A row is excluded, and counted by name in `dropped`, when: its file is
    unreadable JSON (`unreadable`); it has no developer attempts at all
    (`no_dev_task`); its `accepted` field fails strict validation, FLAG 3
    (`invalid_accepted`); a developer attempt's tokens are missing or
    malformed (`dev_tokens_unusable`); the developer's model has no price
    table entry (`dev_model_unpriced`); or every developer attempt prices to
    exactly $0.00 (`dev_zero_tokens`) -- a verified condition in a minority
    of t7 rows where every token stream is reported as 0, which prices
    correctly under price.py's own rules but is undefined for log10(usd) and
    correlates heavily with rejection, so it looks like a harness logging
    gap on that variant rather than a genuine free success (see the fix-round
    report). Never priced at zero and silently kept, never guessed.

    Prints nothing. The drop accounting, cost-price note, planner note, and
    the GPT-5.6 long-context caveat are attached at `df.attrs["assembly"]`.
    """
    root = research_paths.sweep_dir(sweep_dir if sweep_dir is not None else DEFAULT_SWEEP_DIR)
    table = price_table if price_table is not None else load_prices()
    task_set = set(tasks)

    all_files = sorted(root.glob("*.run.json"))
    files_seen = len(all_files)

    selected: list[tuple[Path, dict]] = []
    for f in all_files:
        parts = _parse_sweep_filename(f.name)
        if parts is None:
            continue
        if parts["task"] not in task_set:
            continue
        wanted_variant = _TASK_VARIANT.get(parts["task"])
        if wanted_variant is None or parts["variant"] != wanted_variant:
            continue
        selected.append((f, parts))
    files_selected = len(selected)

    dropped = {
        "unreadable": 0,
        "no_dev_task": 0,
        "invalid_accepted": 0,
        "dev_tokens_unusable": 0,
        "dev_model_unpriced": 0,
        "dev_zero_tokens": 0,
    }
    run_total_nulled = {"count": 0, "reasons": {}}
    # PATCH 4 measurement: these rows are excluded from the table entirely
    # (cost is undefined for log10(usd) at 0), but their acceptance outcome
    # was still real and known before they were dropped. Tallied here so the
    # bias this introduces into the acceptance head can be reported by name
    # instead of silently inflating modelled acceptance. See the module
    # docstring's PATCH 4 note for why these rows are dropped wholesale
    # rather than kept for the acceptance likelihood alone.
    dropped_zero_tokens_outcomes = {"accepted": 0, "rejected": 0}
    rows: list[dict] = []
    # Every developer attempt that actually fed a kept row's `usd`, in the
    # shape `price.price_all` expects -- accumulated so `_long_context_note`
    # can reuse `loopmath.price`'s own long-context caveat instead of guessing
    # one from a row's summed tokens (see that function's docstring).
    dev_price_records: list[dict] = []

    for f, parts in selected:
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            dropped["unreadable"] += 1
            continue

        exp = (record.get("ext") or {}).get("experiment") or {}
        arm = exp.get("arm") or {}
        task = exp.get("task") or parts["task"]

        dev_attempts = [
            a
            for t in record.get("tasks", [])
            if t.get("id") == "DEV"
            for a in t.get("attempts", [])
        ]
        if not dev_attempts:
            dropped["no_dev_task"] += 1
            continue

        accepted, accepted_ok = _validate_accepted(exp.get("accepted"))
        if not accepted_ok:
            dropped["invalid_accepted"] += 1
            continue

        dev_model = canonical_model(arm.get("model"))
        if dev_model is None:
            dropped["dev_model_unpriced"] += 1
            continue

        priced_attempts = [_price_attempt_tokens(a, dev_model, table) for a in dev_attempts]
        bad = next((p for p in priced_attempts if not p["priced"]), None)
        if bad is not None:
            reason = bad.get("reason") or ""
            if reason.startswith("no price entry"):
                dropped["dev_model_unpriced"] += 1
            else:
                dropped["dev_tokens_unusable"] += 1
            continue

        usd = float(sum(p["usd"] for p in priced_attempts))
        if usd <= 0:
            # A verified data condition, not a pricing bug: a handful of t7
            # (plan-gpt-5.6-luna-low) rows have every developer attempt
            # reporting all four token streams as exactly 0, which prices
            # correctly to $0.00 under price.py's own "a stream that is
            # legitimately 0 is priced normally" rule. But log10(usd) is
            # undefined at 0, and these rows skew heavily toward `accepted:
            # false` compared to the corpus as a whole (16 of 23 rejected,
            # against roughly 7% rejected corpus-wide) -- a pattern that
            # looks like a harness logging failure on that variant, not a
            # genuine free success. Excluded and named rather than fed to a
            # log-cost model as a silent -inf, or silently dropped.
            dropped["dev_zero_tokens"] += 1
            dropped_zero_tokens_outcomes["accepted" if accepted else "rejected"] += 1
            continue
        usd_todo_rate = bool(table.is_todo(dev_model))

        recorded_vals = [(a.get("ext") or {}).get("cost_usd") for a in dev_attempts]
        usd_recorded = (
            float(sum(recorded_vals)) if all(v is not None for v in recorded_vals) else None
        )

        out_tokens = sum(
            int(((a.get("ext") or {}).get("tokens") or {}).get("output", 0) or 0)
            for a in dev_attempts
        )
        total_tokens = sum(
            int(((a.get("ext") or {}).get("tokens") or {}).get("total", 0) or 0)
            for a in dev_attempts
        )

        usd_run_total, run_total_reason = _price_run_total(record, usd, table)
        if usd_run_total is None:
            run_total_nulled["count"] += 1
            reasons = run_total_nulled["reasons"]
            reasons[run_total_reason] = reasons.get(run_total_reason, 0) + 1

        task_group = "t7" if str(task).startswith("t7-") else task
        planner = _TASK_PLANNER.get(parts["task"], "none")

        for a in dev_attempts:
            mapped = _map_sweep_tokens((a.get("ext") or {}).get("tokens"))
            if mapped is not None:
                dev_price_records.append({"model": dev_model, "tokens": mapped})

        rows.append({
            "model": arm.get("model"),
            "effort": arm.get("effort"),
            "task": task,
            "usd": usd,
            "accepted": bool(accepted),
            "run_id": (record.get("run") or {}).get("id"),
            "family": arm.get("family"),
            "task_group": task_group,
            "dev_attempts": int(exp.get("dev_attempts") or len(dev_attempts)),
            "out_tokens": int(out_tokens),
            "total_tokens": int(total_tokens),
            "source_file": f.name,
            "usd_recorded": usd_recorded,
            "usd_todo_rate": usd_todo_rate,
            "usd_run_total": usd_run_total,
            "planner": planner,
            "variant": parts["variant"],
        })

    df = pd.DataFrame(rows, columns=_TABLE_COLUMNS)
    if not df.empty:
        df["accepted"] = df["accepted"].astype(bool)
        df["usd"] = df["usd"].astype(float)
        df["usd_todo_rate"] = df["usd_todo_rate"].astype(bool)

    cost_note, cost_match_stats = _cost_price_note(df, table)
    planner_note = (
        "t7 rows were built with a pinned planner (gpt-5.6-luna-low); t1-t5 rows use "
        "the default shared plan and carry no pinned planner. A task effect on t7 may "
        "reflect that planner difference, not task difficulty by itself."
    )
    long_context_note = _long_context_note(dev_price_records, table)
    acceptance_bias_note = _acceptance_bias_note(dropped_zero_tokens_outcomes, df)

    df.attrs["assembly"] = {
        "files_seen": files_seen,
        "files_selected": files_selected,
        "rows": len(df),
        "dropped": dropped,
        "dropped_zero_tokens_outcomes": dropped_zero_tokens_outcomes,
        "usd_run_total_nulled": run_total_nulled,
        "usd_todo_rate_count": int(df["usd_todo_rate"].sum()) if not df.empty else 0,
        "price_table_as_of": table.as_of,
        "cost_price_note": cost_note,
        "cost_match_stats": cost_match_stats,
        "planner_note": planner_note,
        "long_context_note": long_context_note,
        "acceptance_bias_note": acceptance_bias_note,
    }
    return df
