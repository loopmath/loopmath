"""Handler for `loopmath fit` (spec 02, lane 5).

`loopmath fit [--background] [--full] [--no-prior] [--without SOURCE ...] [--home PATH] [--json]`

- Foreground: builds the belief (belief.fit.fit), waiting up to 30 s for a running fit to
  finish (exit 4 after that), and prints a summary or `loopmath.fit/1` JSON.
- `--background`: detaches and returns at once through the store's fit trigger
  (`store.fitjob.spawn_fit`, lane 7), so a running fit queues one more fit instead of
  racing it.
- A `--without` name that matches no source exits 2 and lists the known names (D86); a fit
  with nothing to fit writes nothing, leaves `fits/latest` where it was and exits 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import output

LOCK_WAIT_S = 30.0
SCHEMA = "loopmath.fit/1"


def _spawn(home: Path, *, no_prior: bool, without: tuple[str, ...], full: bool) -> dict:
    from ..store.fitjob import spawn_fit

    return dict(spawn_fit(home, no_prior=no_prior, without=without, full=full))


def _unknown_without(home: Path, without: tuple[str, ...]) -> str | None:
    """The D86 error for `--without` names that match no source, checked before any fit starts.

    Names the bundle, `benchmark`, `user` or `shared` know pass at once; only a likely typo
    reads the source names of the stored and imported runs.
    """
    if not without:
        return None
    from .design import data_source, source_label
    from .fit import store_docs, unknown_source_message, unknown_sources
    from .priors import shared_docs

    unknown, _ = unknown_sources(without)
    if not unknown:
        return None
    stored = list(store_docs(home))
    seen = ({"user"} if stored else set()) | {data_source(doc) for doc in shared_docs(home)}
    unknown, known = unknown_sources(without, seen)
    if not unknown:
        return None
    labels = sorted({source_label(doc) for doc in stored} & set(unknown))
    hint = (f"; {', '.join(labels)} is only a label on runs in your store, which are always the source user "
            "(--without user leaves them out)") if labels else ""
    return unknown_source_message(unknown, known) + hint


def _summary(path: Path) -> dict:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    return {
        "fit": {"id": meta["fit"], "at": meta["created_at"], "age_s": 0, "n_runs": meta.get("n_runs", {})},
        "path": str(path),
        "seconds": meta.get("seconds"),
        "options": meta.get("options", {}),
        "runs_by_source": meta.get("runs_by_source", {}),
        "shipped_overlap": meta.get("shipped_overlap", {}),
        "user_labels": meta.get("user_labels", {}),
        "heads": {name: {"rows": h.get("n_rows"), "runs": h.get("n_runs"), "sigma": h.get("sigma"),
                         "rows_by_source": h.get("rows_by_source", {}),
                         "converged": (h.get("optimizer") or {}).get("converged")}
                  for name, h in (meta.get("heads") or {}).items()},
        "dropped": meta.get("dropped", {}),
        "not_model_attempts": meta.get("not_model_attempts", {}),
        "full": meta.get("full"),
    }


def _print_summary(s: dict) -> None:
    from ..priors import overlap_note
    from .fit import dropped_text

    f = s["fit"]
    n = f.get("n_runs") or {}
    print(f"fit {f['id']} written to {s['path']} in {s.get('seconds')} s; recommend and posterior now read it")
    by_src = ", ".join(f"{k} {v}" for k, v in sorted(s["runs_by_source"].items(), key=lambda kv: -kv[1]))
    labels = ", ".join(f"{k} {v}" for k, v in sorted((s.get("user_labels") or {}).items(), key=lambda kv: -kv[1]))
    detail = "; ".join(x for x in (by_src, f"yours labelled {labels}" if labels else "") if x)
    print(f"runs: {n.get('prior', 0)} prior, {n.get('user', 0)} yours" + (f" ({detail})" if detail else ""))
    note = overlap_note(s.get("shipped_overlap"))
    if note:
        print(note)
    heads = list(s["heads"].items())[:12]
    width = max([22] + [len(name) for name, _ in heads])
    for name, h in heads:
        conv = "" if h.get("converged") in (True, None) else "  (scales did not converge)"
        print(f"  {name:{width}s} {h['rows']:>7} rows  {h['runs']:>6} runs{conv}")
    checks = s.get("not_model_attempts") or {}
    if checks:
        names = ", ".join(k if len(checks) == 1 else f"{k} {v}" for k, v in sorted(checks.items(), key=lambda kv: -kv[1]))
        print(f"not model attempts: {sum(checks.values())} checks ({names}); not evidence, not dropped")
    dropped = s.get("dropped") or {}
    if dropped:
        print(dropped_text(dropped))
    full = s.get("full")
    if isinstance(full, dict) and not full.get("ran"):
        print(f"pymc check: {full.get('reason', 'not run')}")
    elif isinstance(full, dict):
        for name, h in (full.get("heads") or {}).items():
            print(f"pymc check, {name}: {100 * h['inside']:.0f}% of {h['nodes']} node means inside the PyMC "
                  f"interval (widths x{h['median_width_ratio']:.2f}), {100 * h['rows_inside']:.0f}% of "
                  f"{h['rows']} rows (x{h['rows_median_width_ratio']:.2f}), {h['divergences']} divergences "
                  f"(table in pymc_check.json)")


def fit(args: argparse.Namespace) -> int:
    home = output.home(getattr(args, "home", None))
    without = tuple(getattr(args, "without", None) or ())
    no_prior = bool(getattr(args, "no_prior", False))
    full = bool(getattr(args, "full", False))
    as_json = bool(getattr(args, "json", False))
    bad = _unknown_without(home, without)
    if bad:
        return output.fail(bad, output.EXIT_NOT_FOUND)
    if getattr(args, "background", False):
        started = _spawn(home, no_prior=no_prior, without=without, full=full)
        if as_json:
            output.emit_json(SCHEMA, {"background": True, **started})
        elif started.get("started"):
            print(f"fit started in the background (pid {started.get('pid')}); fits/latest moves when it is done")
            print(f"log: {started.get('log')}")
        else:
            print(f"fit queued: {started.get('reason', 'a fit is running')}")
        return output.EXIT_OK
    from .fit import FitBusy, NothingToFit, UnknownSource, fit as run_fit

    try:
        path = run_fit(home, no_prior=no_prior, without=without, full=full, wait_s=LOCK_WAIT_S)
    except FitBusy:
        return output.fail(f"another fit has held {home / 'fits'} for more than {int(LOCK_WAIT_S)} s; "
                           "try again later or use --background", output.EXIT_LOCKED)
    except UnknownSource as exc:
        return output.fail(str(exc), output.EXIT_NOT_FOUND)
    except NothingToFit as exc:
        return output.fail(str(exc), output.EXIT_USER)
    summary = _summary(path)
    if as_json:
        output.emit_json(SCHEMA, summary)
    else:
        _print_summary(summary)
    return output.EXIT_OK
