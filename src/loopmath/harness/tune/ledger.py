"""The cost ledger: dollars for a session's tokens, and one row per session per model in `calls.jsonl`.

A session's tokens come per model from its stream (`stream.read_stream`), so a subagent on another model gets its
own row. Each row is priced against the price table with `loopmath.price.price_run` (the function
`harness_runs.price_tokens` uses), unrounded, and keeps Claude Code's own figure beside it:

    ts, tune, task_id, label, role, model, effort, input_tokens, cache_read_tokens, cache_write_tokens,
    output_tokens, cost_usd, cli_cost_usd, wall_s, exit, is_error

plus `stage`, `attempt` and `error_kind`, and `price_note` when the model has no row (cost_usd None, never 0) or
only a placeholder row. A row with no tokens costs 0 whatever its model: an API error that used none (a 529 after
the CLI's own retries) is a call priced 0, not a lost or unpriced one. `wall_s` is the session's on its first row
and 0 on the others, so a sum over rows is the sessions' wall time. A session that reported no tokens still gets
one row, on its own model, with zero tokens.

`summary` reads calls.jsonl back: totals by role and by model, and the rows whose price and Claude Code's figure
differ by more than 5%.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Sequence

from .contracts import SessionEnd, SessionRequest, TuneDir, Usage, now_iso

__all__ = ["FIELDS", "GAP", "PriceLedger", "row_cost", "summary"]

FIELDS = ("ts", "tune", "task_id", "label", "role", "model", "effort", "input_tokens", "cache_read_tokens",
          "cache_write_tokens", "output_tokens", "cost_usd", "cli_cost_usd", "wall_s", "exit", "is_error")
GAP = 0.05


def _tokens(u: Usage) -> dict:
    return {"in": int(u.input or 0), "cache_read": int(u.cache_read or 0), "cache_write": int(u.cache_write or 0),
            "out": int(u.output or 0)}


def row_cost(u: Usage, table: Any) -> tuple[float | None, str]:
    """(USD, note) for one model's tokens: None and the reason when the table has no row for it; a note when the
    row is a placeholder. No tokens is 0 for any model."""
    tokens = _tokens(u)
    if not any(tokens.values()):
        return 0.0, ""  # no tokens, no charge: a refused request is still a call
    if table is None:
        return None, "no price table"
    from ...price import price_run

    got = price_run({"model": u.model, "tokens": tokens}, table)
    if got.get("usd") is None:
        return None, str(got.get("reason") or f"no price entry for model '{u.model}'")
    return float(got["usd"]), ("placeholder price row" if got.get("todo") else "")


class PriceLedger:
    """`contracts.Ledger`. `prices` is a `loopmath.price.PriceTable`, a path to a price table, or None for the
    packaged table."""

    def __init__(self, tune: TuneDir, prices: Any = None):
        self.tune = tune
        if prices is None or isinstance(prices, (str, Path)):
            from ...price import load_prices

            prices = load_prices(prices)
        self.table = prices
        self._lock = threading.Lock()

    def price(self, usage: Sequence[Usage]) -> float | None:
        total = 0.0
        for u in usage:
            usd, _ = row_cost(u, self.table)
            if usd is None:
                return None
            total += usd
        return total

    def rows(self, req: SessionRequest, end: SessionEnd, task_id: str) -> list[dict]:
        usage = list(end.usage) or [Usage(model=req.limits.model)]
        ts = now_iso()
        out = []
        for i, u in enumerate(usage):
            usd, note = row_cost(u, self.table)
            row: dict[str, Any] = {
                "ts": ts, "tune": self.tune.id, "task_id": task_id, "label": req.label, "role": req.role,
                "model": u.model, "effort": req.limits.effort, "input_tokens": int(u.input or 0),
                "cache_read_tokens": int(u.cache_read or 0), "cache_write_tokens": int(u.cache_write or 0),
                "output_tokens": int(u.output or 0), "cost_usd": usd, "cli_cost_usd": u.cli_cost_usd,
                "wall_s": float(end.wall_s) if i == 0 else 0.0, "exit": end.exit, "is_error": bool(end.is_error),
                "stage": req.stage, "attempt": req.attempt, "error_kind": end.error_kind or "",
            }
            if req.meta.get("retry"):
                row["retry"] = req.meta["retry"]
            if note:
                row["price_note"] = note
            out.append(row)
        return out

    def record(self, req: SessionRequest, end: SessionEnd, task_id: str) -> list[dict]:
        out = self.rows(req, end, task_id)
        text = "".join(json.dumps(r, sort_keys=False) + "\n" for r in out)
        with self._lock:
            self.tune.calls.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tune.calls, "a", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
        return out


def _read_rows(path: Path) -> tuple[list[dict], int]:
    rows, bad = [], 0
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            rows.append(r)
        else:
            bad += 1
    return rows, bad


def summary(path: str | Path) -> dict:
    """Totals of a calls.jsonl: `{rows, bad_lines, cost_usd, cli_cost_usd, unpriced, by_role, by_model, gaps}`.
    `cost_usd` sums the priced rows; `unpriced` counts rows with no price. `gaps` lists the rows whose price and
    Claude Code's figure differ by more than `GAP` of the larger."""
    rows, bad = _read_rows(Path(path))
    tot: dict[str, Any] = {"rows": len(rows), "bad_lines": bad, "cost_usd": 0.0, "cli_cost_usd": 0.0, "unpriced": 0,
                           "by_role": {}, "by_model": {}, "gaps": []}
    for r in rows:
        cost, cli = r.get("cost_usd"), r.get("cli_cost_usd")
        for key, table in (("role", tot["by_role"]), ("model", tot["by_model"])):
            b = table.setdefault(str(r.get(key)), {"rows": 0, "cost_usd": 0.0, "cli_cost_usd": 0.0, "unpriced": 0,
                                                   "input_tokens": 0, "cache_read_tokens": 0,
                                                   "cache_write_tokens": 0, "output_tokens": 0})
            b["rows"] += 1
            for t in ("input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
                b[t] += int(r.get(t) or 0)
            if isinstance(cost, (int, float)):
                b["cost_usd"] += cost
            else:
                b["unpriced"] += 1
            if isinstance(cli, (int, float)):
                b["cli_cost_usd"] += cli
        if isinstance(cost, (int, float)):
            tot["cost_usd"] += cost
        else:
            tot["unpriced"] += 1
        if isinstance(cli, (int, float)):
            tot["cli_cost_usd"] += cli
        if isinstance(cost, (int, float)) and isinstance(cli, (int, float)):
            big = max(abs(cost), abs(cli))
            if big > 0 and abs(cost - cli) / big > GAP:
                tot["gaps"].append({"label": r.get("label"), "model": r.get("model"), "cost_usd": cost,
                                    "cli_cost_usd": cli, "gap": round(abs(cost - cli) / big, 4)})
    return tot
