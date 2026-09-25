"""`loopmath doctor`: what loopmath can see on this machine (spec 02, section 5).

Six checks, each `ok`, `warn` or `fail`:
  logs     Claude Code and Codex log roots, files in total and in the scan window
  agents   `claude --version` and `codex --version`
  store    the store root exists and is writable
  fit      `fits/latest` present, and its age
  prices   every model seen in the scan window has a price
  skill    where the skill set is installed (and which Codex form), and whether it is whole and current

Nothing here opens a network connection. Log files are read through the
parse cache, read only; Codex auth files are never opened.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping

from ..belief.design import other_design
from . import install as _install

SCAN_DAYS = 30
VERSION_TIMEOUT_S = 10


def _check(name: str, status: str, summary: str, **detail) -> dict:
    return {"name": name, "status": status, "summary": summary, "detail": detail}


def log_roots(env: Mapping[str, str] | None = None) -> dict[str, Path]:
    return {"claude-code": _install.claude_dir(env) / "projects",
            "codex": _install.codex_dir(env) / "sessions"}


def _session_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.jsonl")) if root.is_dir() else []


def _recent(paths: list[Path], days: float, now: float) -> list[Path]:
    cutoff = now - days * 86400.0
    kept = []
    for p in paths:
        try:
            if p.stat().st_mtime >= cutoff:
                kept.append(p)
        except OSError:
            continue
    return kept


def check_logs(roots: dict[str, Path], files: dict[str, list[Path]], recent: dict[str, list[Path]]) -> dict:
    rows = {h: {"root": str(roots[h]), "exists": roots[h].is_dir(), "files": len(files[h]),
                "recent_files": len(recent[h])} for h in roots}
    found = [h for h in roots if files[h]]
    parts = [f"{h} {rows[h]['files']} files ({rows[h]['recent_files']} in {SCAN_DAYS} days)" if rows[h]["exists"]
             else f"{h} none at {roots[h]}" for h in roots]
    status = "ok" if found else "warn"
    return _check("logs", status, "; ".join(parts), roots=rows)


def _version(binary: str, run: Callable) -> dict:
    path = shutil.which(binary)
    if path is None:
        return {"found": False, "path": None, "version": None}
    try:
        out = run([path, "--version"], capture_output=True, text=True, timeout=VERSION_TIMEOUT_S)
        text = (out.stdout or out.stderr or "").strip().splitlines()
        return {"found": True, "path": path, "version": text[0] if text else None, "exit": out.returncode}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"found": True, "path": path, "version": None, "error": type(exc).__name__}


def check_agents(run: Callable = subprocess.run) -> dict:
    tools = {"claude-code": _version("claude", run), "codex": _version("codex", run)}
    parts = [f"{h} {t['version'] or ('found, version unknown' if t['found'] else 'not found')}" for h, t in tools.items()]
    status = "ok" if all(t["found"] for t in tools.values()) else "warn"
    return _check("agents", status, "; ".join(parts), tools=tools)


def check_store(home: Path) -> dict:
    created = False
    try:
        if not home.exists():
            home.mkdir(parents=True)
            created = True
        with tempfile.NamedTemporaryFile(dir=home, prefix=".doctor-", delete=True) as fh:
            fh.write(b"ok")
            fh.flush()
    except OSError as exc:
        return _check("store", "fail", f"{home} is not writable: {exc.strerror or exc}", home=str(home), writable=False)
    note = " (created)" if created else ""
    return _check("store", "ok", f"{home} writable{note}", home=str(home), writable=True, created=created)


def _fit_time(latest: Path) -> tuple[str | None, float]:
    meta = latest / "meta.json"
    if meta.is_file():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        for key in ("created_at", "at"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    return value, _dt.datetime.fromisoformat(value).timestamp()
                except ValueError:
                    pass
    stamp = latest.stat().st_mtime
    return _dt.datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds"), stamp


def _age(seconds: float) -> str:
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def check_fit(home: Path, now: float) -> dict:
    latest = home / "fits" / "latest"
    if not latest.exists():
        return _check("fit", "warn", "no fit yet: run `loopmath fit` (or `loopmath onboard`)", present=False)
    at, stamp = _fit_time(latest)
    age_s = max(0.0, now - stamp)
    try:
        meta = json.loads((latest / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    why = other_design(meta) if isinstance(meta, dict) else None
    if why is not None:  # the same check as `loopmath recommend`: this loopmath cannot use the fit
        return _check("fit", "warn", why, present=True, usable=False, fit=latest.resolve().name, at=at,
                      age_s=round(age_s))
    return _check("fit", "ok", f"{latest.resolve().name}, {_age(age_s)} old", present=True, usable=True,
                  fit=latest.resolve().name, at=at, age_s=round(age_s))


def seen_models(recent: dict[str, list[Path]]) -> tuple[Counter, dict]:
    """Model counts over the parsed records of `recent` files, through the parse cache."""
    from .. import ingest

    records, diag = ingest.parse_all(None, discovered=ingest._discovery_manifest(recent))
    return Counter(r.get("model") for r in records), {"records": len(records), "cache_hits": diag.get("cache_hits")}


def check_prices(models: Counter, parse: dict) -> dict:
    from .. import price

    try:
        table = price.load_prices()
    except (OSError, ValueError) as exc:
        return _check("prices", "fail", f"the packaged price table does not load: {exc}", error=str(exc))
    unpriced = {m: n for m, n in models.items() if m is not None and table.rate(m) is None}
    todo = {m: n for m, n in models.items() if m is not None and m not in unpriced and table.is_todo(m)}
    named = sum(n for m, n in models.items() if m is not None)
    detail = dict(as_of=table.as_of, models={str(m): n for m, n in models.items()}, unpriced=unpriced,
                  placeholder=todo, no_model_label=models.get(None, 0), **parse)
    if not named:
        return _check("prices", "ok", f"no models seen in {SCAN_DAYS} days; table as of {table.as_of}", **detail)
    if unpriced:
        listed = ", ".join(f"{m} ({n})" for m, n in sorted(unpriced.items(), key=lambda kv: -kv[1]))
        return _check("prices", "warn", f"no price for {listed}; table as of {table.as_of}", **detail)
    if todo:
        listed = ", ".join(sorted(todo))
        return _check("prices", "warn", f"placeholder prices for {listed}; table as of {table.as_of}", **detail)
    return _check("prices", "ok", f"{len([m for m in models if m])} models seen in {SCAN_DAYS} days, all priced "
                  f"(table as of {table.as_of})", **detail)


def _install_cmd(place: dict) -> str:
    return f"loopmath skill install --target {place['target']} --scope {place['scope']}"


def check_skill(cwd: Path | None, env: Mapping[str, str] | None) -> dict:
    """`ok` when the full set is installed and current for some target, with nothing old or partial
    left anywhere; else a warning with the exact install command."""
    places = _install.status(cwd=cwd, env=env)
    n = len(_install.SKILLS)
    seen = [p for p in places if p["any"]]
    if not seen:
        return _check("skill", "warn", "not installed: run `loopmath skill install`", places=places)
    problems = []
    for p in seen:
        where = f"{p['target']} {p['scope']}"
        if p["old"]:
            problems.append(f"{where} still has the 0.1 single skill: run `{_install_cmd(p)}`")
        elif p["missing"]:
            problems.append(f"{where} lacks {', '.join(p['missing'])}: run `{_install_cmd(p)}`")
        elif not p["current"]:
            problems.append(f"{where} is older than this loopmath: run `{_install_cmd(p)}`")
    if problems:
        return _check("skill", "warn", "; ".join(problems), places=places)
    parts = [f"{p['target']} {p['scope']} ({'AGENTS.md block' if p['method'] == 'agents_block' else 'skills'})"
             for p in seen]
    return _check("skill", "ok", f"{n} skills: " + ", ".join(parts), places=places)


def run_checks(home: Path, *, env: Mapping[str, str] | None = None, cwd: Path | None = None,
               now: float | None = None, run: Callable = subprocess.run,
               progress: Callable[[str], None] | None = None) -> dict:
    """Every check in order. `progress` hears one line before the recent logs are read,
    which is the slow part on a large history."""
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    roots = log_roots(env)
    files = {h: _session_files(r) for h, r in roots.items()}
    recent = {h: _recent(p, SCAN_DAYS, now) for h, p in files.items()}
    n_recent = sum(len(p) for p in recent.values())
    if progress is not None and n_recent:
        progress(f"checking {n_recent} log file{'' if n_recent == 1 else 's'} from the last {SCAN_DAYS} days...")
    models, parse = seen_models(recent)
    checks = [
        check_logs(roots, files, recent),
        check_agents(run),
        check_store(home),
        check_fit(home, now),
        check_prices(models, parse),
        check_skill(cwd, env),
    ]
    return {"ok": not any(c["status"] == "fail" for c in checks), "home": str(home), "checks": checks}
