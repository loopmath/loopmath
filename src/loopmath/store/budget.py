"""Spend from recorded runs against the cap (design/0.1/00-overview.md section 4, 05-recommender.md section 4).

The budget is advisory. Spend is the dollars of runs recorded through loopmath
whose run started in the current period (calendar week from Monday, calendar
month, or all time for `none`), plus what loopmath itself spent in the period
outside any run (onboard's labeller calls, `Store.add_spend`). Logged history
brought in by `onboard` (`runs.is_history`) is shown apart and does not count
toward the cap: it is spend that happened before the user asked loopmath to
track it. A loop run recorded with `--source habit` is the user's run and counts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..output import fmt_usd
from . import runs as R
from .ids import parse_ts, utc_iso

PERIODS = ("week", "month", "none")


def period_start(period: str, now: datetime | None = None) -> datetime | None:
    now = (now or datetime.now().astimezone())
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        return midnight - timedelta(days=midnight.weekday())
    if period == "month":
        return midnight.replace(day=1)
    return None


def is_history_row(store, row: dict[str, Any]) -> bool:
    """`runs.is_history` from an index row: the row's `history` flag (0.2.4); an older row without it is
    history only when its configuration source is `habit` and the run file says onboard wrote it."""
    if "history" in row:
        return bool(row["history"])
    if row.get("source") != "habit":
        return False
    try:
        return R.is_history(store._read(row["run"]))
    except Exception:
        return True  # unreadable: as before 0.2.4, a habit run is history


def labeling(store, since: datetime | None) -> dict[str, Any]:
    """{usd, tokens, calls, records, not_costed}: onboard's labeller spend recorded since `since` (all when None)."""
    usd = 0.0
    tokens = records = not_costed = calls = 0
    for rec in store.spend_records():
        if rec.get("kind") != "labeling":
            continue
        at = parse_ts(rec.get("at"))
        if since is not None and (at is None or at < since):
            continue
        records += 1
        u = rec.get("usd")
        if isinstance(u, (int, float)) and not isinstance(u, bool):
            usd += float(u)
        else:
            not_costed += 1
        t = rec.get("tokens")
        tokens += t if isinstance(t, int) and not isinstance(t, bool) else 0
        c = (rec.get("detail") or {}).get("calls") if isinstance(rec.get("detail"), dict) else None
        calls += c if isinstance(c, int) and not isinstance(c, bool) else 0
    return {"usd": round(usd, 6), "tokens": tokens, "calls": calls, "records": records, "not_costed": not_costed}


def spend(store, period: str, now: datetime | None = None) -> dict[str, Any]:
    """{usd, tokens, runs, runs_usd, labeling, exploration_usd, history_usd, history_runs, attempts_not_costed,
    runs_not_costed, open_runs, since}.

    `usd` and `tokens` are the period's runs plus onboard's labelling (`labeling`, its own part);
    `runs_usd` is the runs' part alone. Dollars are the known dollars: an attempt without a dollar
    figure (an unpriced model, a clip with no requests) adds nothing and is counted in
    `attempts_not_costed`, so `usd` is a lower bound whenever that count is above zero.
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
        if is_history_row(store, row):
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
    lab = labeling(store, since)
    return {"usd": round(usd + lab["usd"], 6), "tokens": int(tokens) + lab["tokens"], "runs": n,
            "runs_usd": round(usd, 6), "labeling": lab, "exploration_usd": round(exploration, 6),
            "history_usd": round(history, 6), "history_runs": history_runs, "attempts_not_costed": not_costed, "runs_not_costed": runs_not_costed,
            "open_runs": open_runs, "since": utc_iso(since) if since else None}


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


def since_text(since: str | None) -> str:
    """` since 2026-09-01` (the period's first day, local), or ` in all` for no period."""
    at = parse_ts(since)
    return f" since {at.astimezone().date().isoformat()}" if at is not None else " in all"


def spend_parts(s: dict[str, Any]) -> str:
    """`runs $2.74 over 3 runs, onboard labelling $0.17`: what the period's spend is made of."""
    lab = s.get("labeling") or {}
    text = f"runs {fmt_usd(s.get('runs_usd', s['usd']))} over {_count(s['runs'], 'run')}"
    if lab.get("records"):
        text += f", onboard labelling {fmt_usd(lab['usd'])}"
    return text


def history_line(s: dict[str, Any]) -> str:
    """The runs `onboard` brought in from your logs, shown apart: they are not counted toward the cap."""
    return (f"history from onboard{since_text(s.get('since'))}, not counted: {fmt_usd(s['history_usd'])} over "
            f"{_count(s['history_runs'], 'run')}")


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"
