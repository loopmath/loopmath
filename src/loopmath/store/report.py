"""`loopmath report`: calibration of receipts, cost per accepted change, exploration picks, recommendation moves.

design/0.1/02-commands.md section 5 and 05-recommender.md section 6: interval
coverage (near 80 percent when calibrated), mean log score of the outcome,
predicted against realized success by decile, predicted against realized cost,
cost per accepted change, exploration picks taken, and whether the
recommendation changed after exploration runs. Reads only.
"""

from __future__ import annotations

import html
import math
from datetime import datetime
from typing import Any

from . import runs as R
from .finish import Q_DEFAULT, evidence_for, rule_of
from .home import NotFound, Store, StoreError
from .ids import parse_ts, ulid_time

EXPLORATION_SOURCES = ("exploration", "designed")


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _get(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _rate(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 6) if xs else None


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    mid = len(ys) // 2
    return round(ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2, 6)


def calibration(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in receipts if isinstance(r.get("after"), dict) and isinstance(r.get("scored"), dict)]
    level = _num(_get(scored[0], "before", "cost", "usd", "level")) if scored else None
    cost_in = [bool(r["scored"]["cost_in_interval"]) for r in scored if isinstance(r["scored"].get("cost_in_interval"), bool)]
    tok_in = [bool(r["scored"]["tokens_in_interval"]) for r in scored if isinstance(r["scored"].get("tokens_in_interval"), bool)]
    logs = [v for v in (_num(r["scored"].get("log_score_z")) for r in scored) if v is not None]
    ratios = [v for v in (_num(r["scored"].get("cost_log_ratio")) for r in scored) if v is not None]
    surprise = [v for v in (_num(r["scored"].get("surprise")) for r in scored) if v is not None]
    pred_cost = [v for v in (_num(_get(r, "before", "cost", "usd", "mean")) for r in scored) if v is not None]
    real_cost = [v for v in (_num(_get(r, "after", "cost", "usd")) for r in scored) if v is not None]
    deciles: list[dict[str, Any]] = []
    bins: dict[int, list[tuple[float, float]]] = {}
    for r in scored:
        p = _num(_get(r, "before", "p_success", "mean"))
        z = _num(_get(r, "after", "z"))
        if p is None or z is None:
            continue
        bins.setdefault(min(int(p * 10), 9), []).append((p, z))
    for b in sorted(bins):
        pairs = bins[b]
        deciles.append({"bin": [b / 10, (b + 1) / 10], "n": len(pairs),
                        "predicted": round(sum(p for p, _ in pairs) / len(pairs), 4),
                        "realized": round(sum(z for _, z in pairs) / len(pairs), 4)})
    return {
        "receipts": len(receipts),
        "scored": len(scored),
        "waiting": len(receipts) - len(scored),
        "rescore_pending": sum(1 for r in receipts if r.get("rescore")),
        "level": level if level is not None else 0.8,
        "cost_coverage": _rate(cost_in),
        "tokens_coverage": _rate(tok_in),
        "mean_log_score": _mean(logs),
        "success_by_decile": deciles,
        "cost": {"predicted_mean_usd": _mean(pred_cost), "realized_mean_usd": _mean(real_cost),
                 "median_log_ratio": _median(ratios), "mean_surprise": _mean(surprise)},
    }


def _cfg_id(x: Any) -> str | None:
    if isinstance(x, str):
        return x if x.startswith("cfg_") else None
    if isinstance(x, dict):
        if isinstance(x.get("id"), str) and x["id"].startswith("cfg_"):
            return x["id"]
        for key in ("config", "candidate", "configuration"):
            got = _cfg_id(x.get(key))
            if got:
                return got
    return None


def _mtime(store: Store, rec_id: str) -> datetime | None:
    try:
        return datetime.fromtimestamp(store.rec_path(rec_id).stat().st_mtime).astimezone()
    except (OSError, StoreError):
        return None


def recommendation_moves(store: Store, since: datetime | None, finished: list[dict[str, Any]]) -> dict[str, Any]:
    """Per task group (type, repo), the default pick across stored recs in time order, and exploration runs between changes."""
    series: dict[tuple[str, str], list[tuple[datetime, str, str]]] = {}
    for rec_id, payload in store.recs():
        at = parse_ts(payload.get("created_at")) or ulid_time(rec_id) or _mtime(store, rec_id)
        if at is None or (since is not None and at < since):
            continue
        task = payload.get("task") if isinstance(payload.get("task"), dict) else {}
        task = task.get("task") if isinstance(task.get("task"), dict) else task
        key = (str(task.get("type") or "?"), str(task.get("repo") or "?"))
        pick = _cfg_id(payload.get("default_pick")) or _cfg_id(payload.get("usual"))
        if pick:
            series.setdefault(key, []).append((at, rec_id, pick))
    explore_times: dict[tuple[str, str], list[datetime]] = {}
    for row in finished:
        if row.get("source") in EXPLORATION_SOURCES:
            t = parse_ts(row.get("finished_at")) or parse_ts(row.get("started_at"))
            if t is not None:
                explore_times.setdefault((str(row.get("task_type")), str(row.get("repo"))), []).append(t)
    groups = []
    moved_after_exploration = 0
    for key, points in sorted(series.items()):
        points.sort()
        changes = []
        for (t0, _r0, c0), (t1, r1, c1) in zip(points, points[1:]):
            if c0 != c1:
                n_explore = sum(1 for t in explore_times.get(key, []) if t0 <= t <= t1)
                changes.append({"rec": r1, "at": t1.isoformat(), "from": c0, "to": c1, "exploration_runs_between": n_explore})
                if n_explore:
                    moved_after_exploration += 1
        groups.append({"type": key[0], "repo": key[1], "recs": len(points), "first": points[0][2], "last": points[-1][2],
                       "changes": changes})
    return {"groups": groups, "changed": sum(1 for g in groups if g["changes"]),
            "moved_after_exploration": moved_after_exploration}


def report_payload(store: Store, since: datetime | None = None) -> dict[str, Any]:
    rows = [r for r in store.runs(since=since)] if since else list(store.runs())
    finished_rows = [r for r in rows if r.get("state") == R.FINISHED]
    in_window = {r["run"] for r in rows}
    receipts = [r for r in store.receipts() if r.get("run") in in_window]
    by_run = {}
    for r in receipts:
        by_run.setdefault(r.get("run"), []).append(r)

    conf = store.config()
    q = {t: conf.q_for_tier(t) for t in Q_DEFAULT}
    per_source: dict[str, dict[str, Any]] = {}
    explore_runs = []
    total_usd = 0.0
    accepted = unknown = not_costed = history = 0
    for row in finished_rows:
        source = str(row.get("source") or "?")
        if source == "habit":
            history += 1
            continue  # history from onboard: not work loopmath recommended or recorded live
        try:
            doc = store._read(row["run"])
        except (NotFound, StoreError, ValueError):
            continue
        z = next((_num(_get(rc, "after", "z")) for rc in by_run.get(row["run"], []) if isinstance(rc.get("after"), dict)), None)
        if z is None:
            z = evidence_for(doc, rule_of(doc), q_by_tier=q).z
        cost = R.run_cost(doc)
        usd = cost["usd_known"]  # known dollars; a run missing some is counted in runs_not_costed
        missing = 0 if cost["complete"] else 1
        total_usd += usd
        not_costed += missing
        s = per_source.setdefault(source, {"runs": 0, "usd": 0.0, "accepted": 0, "unknown": 0, "runs_not_costed": 0})
        s["runs"] += 1
        s["usd"] = round(s["usd"] + usd, 6)
        s["runs_not_costed"] += missing
        if z is None:
            unknown += 1
            s["unknown"] += 1
        elif z >= 0.5:
            accepted += 1
            s["accepted"] += 1
        if source in EXPLORATION_SOURCES:
            rc = next(iter(by_run.get(row["run"], [])), None)
            explore_runs.append({"run": row["run"], "config": row.get("config"), "slate": row.get("slate"),
                                 "usd": cost["usd"], "usd_known": round(usd, 6), "z": z,
                                 "predicted_p_success": _num(_get(rc, "before", "p_success", "mean")) if rc else None})
    for s in per_source.values():
        s["usd_per_accepted"] = round(s["usd"] / s["accepted"], 6) if s["accepted"] else None
    return {
        "since": since.isoformat() if since else None,
        "runs": {"total": len(rows), "finished": len(finished_rows), "open": len(rows) - len(finished_rows),
                 "history": history},
        "calibration": calibration(receipts),
        "cost_per_accepted": {"usd": round(total_usd / accepted, 6) if accepted else None, "total_usd": round(total_usd, 6),
                              "accepted": accepted, "unknown": unknown, "runs_not_costed": not_costed,
                              "by_source": per_source},
        "exploration": {"taken": len(explore_runs), "accepted": sum(1 for e in explore_runs if e["z"] is not None and e["z"] >= 0.5),
                        "usd": round(sum(e["usd_known"] for e in explore_runs), 6),
                        "runs_not_costed": sum(1 for e in explore_runs if e["usd"] is None), "runs": explore_runs[-50:]},
        "recommendation": recommendation_moves(store, since, finished_rows),
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def report_lines(p: dict[str, Any]) -> list[str]:
    c = p["calibration"]
    history = p["runs"].get("history") or 0
    lines = [f"report{' since ' + p['since'][:10] if p['since'] else ''}: {p['runs']['finished']} finished run(s), "
             f"{p['runs']['open']} open"
             + (f"; {history} of them from onboard history, left out of cost per accepted change" if history else "")]
    if c["scored"]:
        lines.append(f"receipts: {c['scored']} scored, {c['waiting']} waiting; cost inside its {c['level']:.0%} interval "
                     f"{_pct(c['cost_coverage'])}, tokens {_pct(c['tokens_coverage'])}")
        if c["mean_log_score"] is not None:
            lines.append(f"mean log score of the outcome: {c['mean_log_score']:.3f} (0 is perfect; log 0.5 = -0.693 is a coin)")
        cost = c["cost"]
        if cost["predicted_mean_usd"] is not None and cost["realized_mean_usd"] is not None:
            lines.append(f"cost per run: predicted ${cost['predicted_mean_usd']:,.2f}, realized ${cost['realized_mean_usd']:,.2f}")
        for d in c["success_by_decile"][:10]:
            lines.append(f"  predicted {d['predicted']:.0%} success, realized {d['realized']:.0%} (n={d['n']})")
    else:
        lines.append(f"receipts: {c['receipts']} ({c['waiting']} waiting for their run to finish); none scored yet")
    k = p["cost_per_accepted"]
    if k["usd"] is not None:
        lines.append(f"cost per accepted change: ${k['usd']:,.2f} ({k['accepted']} accepted, {k['unknown']} unknown)"
                     + (f"; known dollars only, {k['runs_not_costed']} run(s) lack some" if k["runs_not_costed"] else ""))
    else:
        lines.append(f"cost per accepted change: n/a ({k['accepted']} accepted so far)")
    e = p["exploration"]
    lines.append(f"exploration picks taken: {e['taken']} (${e['usd']:,.2f}), {e['accepted']} accepted")
    m = p["recommendation"]
    if m["groups"]:
        lines.append(f"recommendation moved in {m['changed']} of {len(m['groups'])} task group(s); "
                     f"{m['moved_after_exploration']} change(s) followed exploration runs")
        for g in [g for g in m["groups"] if g["changes"]][:4]:
            ch = g["changes"][-1]
            lines.append(f"  {g['type']} in {g['repo']}: {ch['from']} to {ch['to']}")
    else:
        lines.append("recommendation: no stored recommendations " + (f"since {p['since'][:10]}" if p["since"] else "yet"))
    return lines[:25]


_CSS = ("body{font:14px/1.45 -apple-system,system-ui,sans-serif;margin:2rem auto;max-width:60rem;padding:0 1rem;color:#222}"
        "table{border-collapse:collapse;margin:.5rem 0 1.5rem}td,th{border-bottom:1px solid #ddd;padding:.25rem .6rem;"
        "text-align:left}th{font-weight:600}.n{text-align:right;font-variant-numeric:tabular-nums}")


def _table(head: list[str], rows: list[list[Any]]) -> str:
    h = "".join(f"<th>{html.escape(x)}</th>" for x in head)
    body = "".join("<tr>" + "".join(
        f"<td class=n>{html.escape(str(v))}</td>" if isinstance(v, (int, float)) else f"<td>{html.escape(str(v))}</td>"
        for v in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def report_html(p: dict[str, Any]) -> str:
    """A self-contained page: no scripts, no external resources."""
    parts = [f"<!doctype html><meta charset=utf-8><title>loopmath report</title><style>{_CSS}</style>",
             "<h1>loopmath report</h1>", "<p>" + "<br>".join(html.escape(x) for x in report_lines(p)) + "</p>"]
    c = p["calibration"]
    if c["success_by_decile"]:
        parts.append("<h2>Predicted against realized success</h2>")
        parts.append(_table(["predicted bin", "runs", "predicted", "realized"],
                            [[f"{d['bin'][0]:.0%} to {d['bin'][1]:.0%}", d["n"], f"{d['predicted']:.0%}", f"{d['realized']:.0%}"]
                             for d in c["success_by_decile"]]))
    by = p["cost_per_accepted"]["by_source"]
    if by:
        parts.append("<h2>Cost per accepted change by source</h2>")
        parts.append(_table(["source", "runs", "known dollars", "runs missing dollars", "accepted", "dollars per accepted"],
                            [[k, v["runs"], f"${v['usd']:,.2f}", v["runs_not_costed"], v["accepted"],
                              "n/a" if v["usd_per_accepted"] is None else f"${v['usd_per_accepted']:,.2f}"]
                             for k, v in sorted(by.items())]))
    ex = p["exploration"]["runs"]
    if ex:
        parts.append("<h2>Exploration runs</h2>")
        parts.append(_table(["run", "configuration", "dollars", "outcome", "predicted success"],
                            [[e["run"], e["config"] or "",
                              f"${e['usd']:,.2f}" if e["usd"] is not None else f"unknown (${e['usd_known']:,.2f} known)",
                              "unknown" if e["z"] is None else ("accepted" if e["z"] >= 0.5 else "not accepted"),
                              _pct(e["predicted_p_success"])] for e in ex]))
    groups = [g for g in p["recommendation"]["groups"] if g["changes"]]
    if groups:
        parts.append("<h2>Recommendation changes</h2>")
        parts.append(_table(["type", "repo", "from", "to", "at", "exploration runs between"],
                            [[g["type"], g["repo"], ch["from"], ch["to"], ch["at"], ch["exploration_runs_between"]]
                             for g in groups for ch in g["changes"]]))
    return "\n".join(parts) + "\n"
