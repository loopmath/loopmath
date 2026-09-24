"""`loopmath status`: fit age, background fit state, unfinished runs, spend against the cap, store size and health.

Reads only; never takes the store lock (readers never lock).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from . import runs as R
from .budget import budget_state
from .config import ConfigError
from .fitjob import fit_state
from .home import Store
from .ids import parse_ts
from .lock import is_locked, read_jsonl

STALE_OPEN_H = 24


def _size(home: Path) -> dict[str, Any]:
    """The store's bytes and files, and how many of the bytes are fits (D91: fits are the bulk)."""
    total = files = fits = 0
    fits_dir = os.path.join(str(home), "fits")
    for root, _dirs, names in os.walk(home):
        in_fits = root == fits_dir or root.startswith(fits_dir + os.sep)
        for name in names:
            try:
                size = os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue
            total += size
            fits += size if in_fits else 0
            files += 1
    return {"bytes": total, "files": files, "fits_bytes": fits}


def _count(folder: Path, pattern: str) -> int:
    return sum(1 for _ in folder.glob(pattern)) if folder.is_dir() else 0


def status_payload(store: Store, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now().astimezone()
    home = store.home
    health: list[dict[str, str]] = []

    def warn(code: str, message: str) -> None:
        health.append({"level": "warn", "code": code, "message": message})

    rows = store.index_rows()
    files = {p.name[: -len(".ocp.json")] for p in store.runs_dir.glob("*.ocp.json")} if store.runs_dir.is_dir() else set()
    raw_rows = len(read_jsonl(store.index_path))
    missing_row = sorted(files - set(rows))
    missing_file = sorted(set(rows) - files)
    if missing_row:
        warn("index_missing", f"{len(missing_row)} run file(s) not in index.jsonl; run `loopmath status --reindex`")
    if missing_file:
        warn("index_orphan", f"{len(missing_file)} index row(s) name a run file that is gone; run `loopmath status --reindex`")

    open_runs = []
    for run, row in rows.items():
        if row.get("state") == R.FINISHED or run in missing_file:
            continue
        started = parse_ts(row.get("started_at"))
        age_h = round((now - started).total_seconds() / 3600, 1) if started else None
        open_runs.append({"run": run, "started_at": row.get("started_at"), "age_h": age_h, "type": row.get("task_type"),
                          "repo": row.get("repo"), "config": row.get("config"), "slate": row.get("slate")})
    open_runs.sort(key=lambda r: str(r.get("started_at") or ""))
    stale = [r for r in open_runs if r["age_h"] is not None and r["age_h"] > STALE_OPEN_H]
    if stale:
        warn("open_runs_stale", f"{len(stale)} run(s) open for more than {STALE_OPEN_H} h; finish them or they never reach the fit")

    orphans = []
    if (home / "signals").is_dir():
        for p in (home / "signals").glob("*.jsonl"):
            run = p.name[: -len(".jsonl")]
            if rows.get(run, {}).get("state") == R.FINISHED or run not in files:
                orphans.append(run)
    if orphans:
        warn("signals_orphan", f"{len(orphans)} pending signal file(s) belong to no open run")

    temps = [p.name for sub in ("runs", "receipts", "recs", "fits", "") for p in (home / sub).glob(".*.tmp")
             if (home / sub).is_dir()]
    if temps:
        warn("temp_files", f"{len(temps)} leftover temp file(s) from an interrupted write (safe to delete)")

    rescore = sum(1 for r in store.receipts() if r.get("rescore"))
    if rescore:
        warn("receipts_rescore", f"{rescore} receipt(s) wait for re-scoring after a late signal")

    config: dict[str, Any] = {"path": str(store.config_path), "exists": store.config_path.is_file(), "ok": True}
    try:
        spend = budget_state(store, now)
    except ConfigError as exc:
        config["ok"] = False
        warn("config", f"config.toml: {exc}")
        spend = None
    except ValueError as exc:
        config["ok"] = False
        warn("config", f"config.toml cannot be read: {exc}")
        spend = None

    fit = fit_state(home)
    job = fit.get("job") or {}
    if job.get("status") == "failed" and not fit["running"]:
        warn("fit_failed", f"the last background fit failed: {job.get('error')}")
    if fit.get("partial"):
        warn("fit_partial", f"{len(fit['partial'])} unfinished fit folder(s) from an interrupted fit")
    if fit["latest"] is None:
        warn("no_fit", "no fit yet; run `loopmath onboard` or `loopmath fit`")

    counts = {"runs": len(files), "finished": sum(1 for r in rows.values() if r.get("state") == R.FINISHED),
              "open": len(open_runs), "index_rows": raw_rows, "receipts": _count(home / "receipts", "*.json"),
              "recs": _count(home / "recs", "rec_*.json"),
              "fits": sum(1 for p in (home / "fits").glob("fit_*") if p.is_dir()) if (home / "fits").is_dir() else 0}
    return {
        "home": str(home),
        "exists": home.is_dir(),
        "fit": fit,
        "open_runs": open_runs[:50],
        "counts": counts,
        "budget": spend,
        "size": _size(home) if home.is_dir() else {"bytes": 0, "files": 0, "fits_bytes": 0},
        "locked": is_locked(home / "lock") if (home / "lock").exists() else False,
        "config": config,
        "health": health,
        "ok": not health,
    }


def _age(seconds: int | None) -> str:
    if seconds is None:
        return "unknown age"
    if seconds < 3600:
        return f"{seconds // 60} min old"
    if seconds < 172800:
        return f"{seconds // 3600} h old"
    return f"{seconds // 86400} days old"


def _bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def status_lines(p: dict[str, Any]) -> list[str]:
    lines = [f"store: {p['home']}" + ("" if p["exists"] else " (empty: nothing recorded yet)")]
    fit = p["fit"]
    job = fit.get("job") or {}
    if fit["latest"]:
        lines.append(f"fit: {fit['latest']}, {_age(fit.get('age_s'))}")
    else:
        lines.append("fit: none yet")
    if fit["running"]:
        lines.append("background fit: running" + (", another queued" if fit["pending"] else ""))
    elif job.get("status"):
        lines.append(f"background fit: last {job['status']} at {job.get('finished_at') or job.get('started_at')}")
    c = p["counts"]
    lines.append(f"runs: {c['runs']} ({c['finished']} finished, {c['open']} open); receipts {c['receipts']}; recs {c['recs']}")
    for r in p["open_runs"][:5]:
        age = f"{r['age_h']} h" if r["age_h"] is not None else "unknown age"
        lines.append(f"  open {r['run']} ({r.get('type') or '?'} in {r.get('repo') or '?'}, {age})")
    if len(p["open_runs"]) > 5:
        lines.append(f"  and {len(p['open_runs']) - 5} more open")
    b = p.get("budget")
    if b:
        spent = b["spent"]
        if b["cap_usd"] is None:
            lines.append(f"spend this {b['period']}: ${spent['usd']:,.2f} ({spent['tokens']:,} tokens); no cap")
        else:
            lines.append(f"spend this {b['period']}: ${spent['usd']:,.2f} of ${b['cap_usd']:,.2f} ({spent['tokens']:,} tokens)"
                         + ("; cap reached" if b["reached"] else ""))
        if spent.get("attempts_not_costed"):
            lines.append(f"  known dollars only: {spent['attempts_not_costed']} attempt(s) have no dollars")
    lines.append(f"size: {_bytes(p['size']['bytes'])} in {p['size']['files']} files"
                 + (f", of which fits {_bytes(p['size']['fits_bytes'])} in {c['fits']} fit folder(s)" if c["fits"] else "")
                 + ("; store lock held right now" if p["locked"] else ""))
    if p["health"]:
        for h in p["health"][:8]:
            lines.append(f"warn: {h['message']}")
    else:
        lines.append("health: ok")
    return lines
