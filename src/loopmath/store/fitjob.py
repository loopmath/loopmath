"""The background fit trigger (design/0.1/02-commands.md, sections 3 and 4).

`run finish` and late `outcome` signals call `spawn_fit(home)`, which starts
`loopmath.store.fitjob.main(["--home", HOME])` in a detached process and returns at once.
The job takes a non-blocking lock on `fits/.lock`, so one fit runs at a time.
A trigger that finds a fit running writes `fits/pending` instead; the running
job sees it when its fit ends and fits once more, so any number of triggers
during one fit cost exactly one more fit. The job's state is in
`fits/job.json` (for `loopmath status`) and its output in `fits/fit.log`.

Lane 5 builds the fit itself (`belief.fit.fit(home, ...)`); its `fit
--background` can call `spawn_fit`, and a foreground `fit` can take the same
lock with `fit_lock(home)`.

A refit the store starts on its own (after `run finish`, `run record`, a late
`outcome` or an import) leaves out what the last fit left out: its
`--without` sources and `--no-prior`, read from `fits/latest/meta.json` when
the job fits, so a queued refit follows a `fit --without` that ran meanwhile
(0.2.4). `--full` is not carried: background refits stay the quick fit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from .ids import now_iso, parse_ts
from .lock import atomic_write_json, is_locked, read_json, try_lock

MAX_RERUNS = 5
FIT_WAIT_S = 900  # a foreground `loopmath fit` holds lane 5's lock this long at most before the job gives up


def _fits(home: Path) -> Path:
    d = Path(home) / "fits"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lock_path(home: Path) -> Path:
    return Path(home) / "fits" / ".lock"


def fit_lock(home: Path):
    """Context manager yielding True when this process now holds the fit lock."""
    return try_lock(_fits(home) / ".lock")


def _mark_pending(home: Path, opts: dict[str, Any]) -> None:
    atomic_write_json(_fits(home) / "pending", {"at": now_iso(), "opts": opts})


def _job_state(home: Path, **fields: Any) -> None:
    path = _fits(home) / "job.json"
    try:
        cur = read_json(path)
    except (FileNotFoundError, ValueError):
        cur = {}
    if not isinstance(cur, dict):
        cur = {}
    cur.update(fields)
    atomic_write_json(path, cur)


def _opts(no_prior: bool, without: tuple[str, ...] | list[str], full: bool, inherit: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"no_prior": bool(no_prior), "without": list(without or ()), "full": bool(full)}
    if inherit:
        out["inherit"] = True
    return out


def exclusions(home: Path) -> dict[str, Any]:
    """What the latest fit left out, which background refits keep: {no_prior, without, fit}; empty lists and
    False when there is no fit or its meta.json cannot be read."""
    link = Path(home) / "fits" / "latest"
    try:
        target = link.resolve() if link.exists() else None
        meta = read_json(target / "meta.json") if target is not None else None
    except (OSError, ValueError):
        meta = None
    options = meta.get("options") if isinstance(meta, dict) and isinstance(meta.get("options"), dict) else {}
    without = [str(w) for w in options.get("without") or [] if isinstance(w, str) and w]
    return {"no_prior": bool(options.get("no_prior")), "without": without,
            "fit": target.name if options and target is not None else None}


def _resolve(home: Path, opts: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the fit's keyword options, what the job records): an `inherit` request takes the latest fit's
    exclusions now."""
    use = {"no_prior": bool(opts.get("no_prior")), "without": list(opts.get("without") or ()),
           "full": bool(opts.get("full"))}
    if not opts.get("inherit"):
        return use, dict(use)
    ex = exclusions(home)
    use.update(no_prior=ex["no_prior"], without=list(ex["without"]))
    return use, {**use, "as_fit": ex["fit"]}


def _package_root() -> str:
    import loopmath

    return str(Path(loopmath.__file__).resolve().parent.parent)


def spawn_fit(home: Path, *, no_prior: bool | None = None, without: tuple[str, ...] | list[str] | None = None,
              full: bool = False) -> dict[str, Any]:
    """Start a detached fit and return at once: {started, pid, log} or {started: False, queued: True, reason}.

    With neither `no_prior` nor `without` given (the store's own refits) the fit leaves out what the
    latest fit left out, decided when it runs; `fit --background` passes its flags, even empty ones."""
    home = Path(home)
    fits = _fits(home)
    inherit = no_prior is None and without is None
    opts = _opts(bool(no_prior), without or (), full, inherit=inherit)
    if is_locked(fits / ".lock"):
        _mark_pending(home, opts)
        return {"started": False, "queued": True, "reason": "a fit is running; it fits again when it ends"}
    # `-c` rather than `-m`: the package imports this module, and `-m` would run a second copy of it
    cmd = [sys.executable, "-c", "import sys; from loopmath.store.fitjob import main; sys.exit(main())",
           "--home", str(home)]
    if inherit:
        cmd.append("--as-last-fit")
    if no_prior:
        cmd.append("--no-prior")
    for w in without or ():
        cmd += ["--without", w]
    if full:
        cmd.append("--full")
    env = dict(os.environ)
    root = _package_root()
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_path = fits / "fit.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=str(home),
                                env=env, start_new_session=True, close_fds=True)
    return {"started": True, "pid": proc.pid, "log": str(log_path)}


def _default_fit(home: Path, **opts: Any) -> Any:
    from ..belief.fit import fit

    # lane 5's own fit lock: wait out a foreground fit rather than fail at once
    return fit(home, no_prior=opts["no_prior"], without=tuple(opts["without"]), full=opts["full"], wait_s=FIT_WAIT_S)


def _stamp(home: Path, fit_path: Path) -> dict[str, Any]:
    """Receipts scored since the last fit get `fit_after` (and `moved`); a failure is recorded, never raised."""
    from .finish import stamp_fit_after
    from .home import Store

    try:
        try:
            from ..belief.state import load
            belief = load(fit_path)
        except Exception:  # a fit lane 5 cannot read back: fit_after without `moved`
            belief = None
        out = stamp_fit_after(Store(home), fit_path.name, belief)
        return {"stamped": len(out["stamped"]), "moved": len(out["moved"]), "skipped": len(out["skipped"])}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def _rescore_flagged(home: Path) -> dict[str, Any]:
    """Receipts still flagged after a late signal are re-scored before the fit; a failure is recorded, never raised."""
    from .finish import rescore_flagged
    from .home import Store

    try:
        out = rescore_flagged(Store(home))
        return {"rescored": len(out["rescored"]), "failed": out["failed"]}
    except Exception as exc:
        traceback.print_exc()
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def run_job(home: Path, *, fit_fn: Callable[..., Any] | None = None, no_prior: bool = False,
            without: tuple[str, ...] | list[str] = (), full: bool = False, inherit: bool = False) -> dict[str, Any]:
    """Fit under the fit lock, then again while `fits/pending` appears. Never raises for a failed fit.
    `inherit`: leave out what the latest fit left out (`exclusions`), read before each fit."""
    home = Path(home)
    fn = fit_fn or _default_fit
    opts = _opts(no_prior, without, full, inherit=inherit)
    pending = _fits(home) / "pending"
    with fit_lock(home) as got:
        if not got:
            _mark_pending(home, opts)
            return {"ran": 0, "queued": True}
        ran = 0
        failed = 0
        while True:
            try:
                queued = read_json(pending)
                opts = dict(queued.get("opts") or opts) if isinstance(queued, dict) else opts
            except (FileNotFoundError, ValueError):
                pass
            try:
                os.unlink(pending)
            except FileNotFoundError:
                pass
            use, recorded = _resolve(home, opts)
            _job_state(home, pid=os.getpid(), status="running", started_at=now_iso(), finished_at=None,
                       error=None, opts=recorded, rescored=_rescore_flagged(home))
            try:
                out = fn(home, **use)
                ran += 1
                _job_state(home, status="done", finished_at=now_iso(), fit=Path(out).name if out else None,
                           receipts=_stamp(home, Path(out)) if out else None)
            except Exception as exc:  # the job must record any failure, never crash silently
                failed += 1
                traceback.print_exc()
                _job_state(home, status="failed", finished_at=now_iso(), error=f"{type(exc).__name__}: {exc}"[:500])
            if not pending.exists() or ran + failed >= MAX_RERUNS:
                break
        return {"ran": ran, "failed": failed, "queued": False}


def fit_state(home: Path) -> dict[str, Any]:
    """For `loopmath status`: running, pending, the last job record, the latest fit and its age."""
    home = Path(home)
    fits = home / "fits"
    try:
        job = read_json(fits / "job.json")
    except (FileNotFoundError, ValueError):
        job = None
    latest = None
    age_s = None
    link = fits / "latest"
    if link.exists():
        target = link.resolve()
        latest = target.name
        meta_at = None
        try:
            meta = read_json(target / "meta.json")
            meta_at = parse_ts(meta.get("created_at")) if isinstance(meta, dict) else None
        except (FileNotFoundError, ValueError):
            pass
        if meta_at is not None:
            from datetime import datetime

            age_s = int((datetime.now().astimezone() - meta_at).total_seconds())
        else:
            import time

            age_s = int(time.time() - target.stat().st_mtime)
    partial = sorted(p.name for p in fits.glob("*.partial")) if fits.is_dir() else []
    running = is_locked(fits / ".lock")
    return {"running": running, "pending": (fits / "pending").exists(), "job": job, "latest": latest,
            "age_s": age_s, "partial": partial if not running else []}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="loopmath-fitjob")
    p.add_argument("--home", required=True)
    p.add_argument("--no-prior", action="store_true")
    p.add_argument("--without", action="append", default=[])
    p.add_argument("--full", action="store_true")
    p.add_argument("--as-last-fit", action="store_true", help="leave out what the latest fit left out")
    a = p.parse_args(argv)
    print(f"{now_iso()} fit job {os.getpid()} starting", flush=True)
    out = run_job(Path(a.home), no_prior=a.no_prior, without=a.without, full=a.full, inherit=a.as_last_fit)
    print(f"{now_iso()} fit job {os.getpid()} ended: {out}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
