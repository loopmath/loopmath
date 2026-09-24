"""Spend from recorded runs against the cap (design/0.1/00-overview.md section 4, 05-recommender.md section 4).

The budget is advisory. Spend is the dollars of runs recorded through loopmath
whose run started in the current period (calendar week from Monday, calendar
month, or all time for `none`). Logged history brought in by `onboard`
(configuration source `habit`) is shown apart and does not count toward the
cap: it is spend that happened before the user asked loopmath to track it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from . import runs as R
from .ids import parse_ts

PERIODS = ("week", "month", "none")


def period_start(period: str, now: datetime | None = None) -> datetime | None:
    now = (now or datetime.now().astimezone())
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        return midnight - timedelta(days=midnight.weekday())
    if period == "month":
        return midnight.replace(day=1)
    return None


def spend(store, period: str, now: datetime | None = None) -> dict[str, Any]:
    """{usd, tokens, runs, exploration_usd, history_usd, history_runs, attempts_not_costed, runs_not_costed, open_runs,
    since}.

    Dollars are the known dollars: an attempt without a dollar figure (an unpriced model, a clip with no
    requests) adds nothing and is counted in `attempts_not_costed`, so `usd` is a lower bound whenever
    that count is above zero.
    """
    since = period_start(period, now)
    usd = tokens = 0.0
    n = 0
    exploration = history = 0.0
    open_runs = not_costed = runs_not_costed = history_runs = 0
    for row in store.runs():
        started = parse_ts(row.get("started_at"))
        if since is not None and (started is None or started < since):
            continue
        if row.get("state") != R.FINISHED:
            open_runs += 1
            continue
        if "cost_usd_known" in row and "attempts_not_costed" in row:
            u = float(row.get("cost_usd_known") or 0.0)
            t = float(row.get("tokens") or 0)
            missing = int(row.get("attempts_not_costed") or 0)
        else:  # an index row written before these fields: read the run
            try:
                cost = R.run_cost(store._read(row["run"]))
            except Exception:
                continue
            u, t, missing = cost["usd_known"], float(cost["tokens"] or 0), cost["unpriced"]
        if row.get("source") == "habit":
            history += u
            history_runs += 1
            continue
        usd += u
        tokens += t
        n += 1
        not_costed += missing
        runs_not_costed += 1 if missing else 0
        if row.get("source") in ("exploration", "designed"):
            exploration += u
    return {"usd": round(usd, 6), "tokens": int(tokens), "runs": n, "exploration_usd": round(exploration, 6),
            "history_usd": round(history, 6), "history_runs": history_runs, "attempts_not_costed": not_costed, "runs_not_costed": runs_not_costed,
            "open_runs": open_runs, "since": since.isoformat() if since else None}


def budget_state(store, now: datetime | None = None) -> dict[str, Any]:
    """{cap_usd, period, spent: spend(...), remaining_usd, reached}; no cap means nothing is paused."""
    conf = store.config()
    cap = conf.get("budget.usd")
    period = conf.get("budget.period") or "month"
    s = spend(store, period, now)
    cap_f = float(cap) if isinstance(cap, (int, float)) and not isinstance(cap, bool) else None
    remaining = None if cap_f is None else round(cap_f - s["usd"], 6)
    return {"cap_usd": cap_f, "period": period, "spent": s, "remaining_usd": remaining,
            "reached": cap_f is not None and s["usd"] >= cap_f}
