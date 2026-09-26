"""Summaries of bundled runs for the manifest and `loopmath prior show` (lane 11), and the starting
prior a user's answers begin from (lane 23P): `onboard`, the results page and `prior show` print its line."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path


def _verdicts(run: dict) -> dict[str, str]:
    return {s.get("name"): s.get("value") for s in run.get("signals") or [] if s.get("kind") == "verdict"}


def _accepted(run: dict) -> bool | None:
    """Accepted under the run's own rule, for counting only (the outcome function is lane 5's)."""
    rule = run.get("acceptance_rule") or {}
    verdicts = _verdicts(run)
    need = list(rule.get("requires") or [])
    ok_words = ("accept", "pass")
    if need:
        if any(verdicts.get(n) in ("reject", "fail") for n in need):
            return False
        if not all(verdicts.get(n) in ok_words for n in need):
            return None
    score = rule.get("score")
    if isinstance(score, dict):
        vals = [s.get("value") for s in run.get("signals") or []
                if s.get("kind") == "score" and s.get("name") == score.get("name") and s.get("value") is not None]
        if not vals:
            return None
        v, t = vals[-1], score.get("target")
        return v >= t if score.get("better", "higher") == "higher" else v <= t
    return True if need else None


def run_summary(docs: list[dict]) -> dict:
    """Counts by type, model, shape and outcome, plus total tokens and dollars."""
    types, models, shapes, outcomes = Counter(), Counter(), Counter(), Counter()
    usd = 0.0
    tokens = 0
    unpriced = 0
    unknown = 0
    for d in docs:
        run = d.get("run") or {}
        types[(run.get("task") or {}).get("type", "untyped")] += 1
        wf = ((run.get("configuration") or {}).get("workflow") or {})
        shapes[wf.get("id", "?")] += 1
        acc = _accepted(run)
        outcomes["accepted" if acc else ("not accepted" if acc is False else "unknown")] += 1
        for att in d.get("attempts") or []:
            model = (att.get("model") or {}).get("id")
            if model:
                models[model] += 1
            cost = att.get("cost") or {}
            if cost:
                tokens += sum(int(cost.get(k) or 0) for k in
                              ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens"))
                if "usd" in cost:
                    usd += float(cost["usd"])
                else:
                    unpriced += 1
            elif model:
                unknown += 1  # a model attempt with no cost block: its usage is unknown (a killed CLI)
    return {
        "runs": len(docs),
        "outcomes": dict(sorted(outcomes.items())),
        "types": dict(sorted(types.items())),
        "shapes": dict(sorted(shapes.items())),
        "attempts_by_model": dict(sorted(models.items())),
        "tokens": tokens,
        "usd": round(usd, 2),
        "unpriced_attempts": unpriced,
        "cost_unknown_attempts": unknown,
    }


def _built_date(value: object) -> str | None:
    """The manifest's `built_at` as a UTC date: `YYYY-MM-DD` as is, an ISO date and time converted to UTC."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text).isoformat()
    except ValueError:
        pass
    try:
        stamp = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    return (stamp.astimezone(dt.timezone.utc) if stamp.tzinfo else stamp).date().isoformat()


def _runs(by_source: dict[str, int]) -> str:
    total = sum(by_source.values())
    return f"{total:,} run{'' if total == 1 else 's'} (" + ", ".join(f"{k} {n:,}" for k, n in by_source.items()) + ")"


def _benchmark_names(benches: list[dict]) -> list[str]:
    """One name per benchmark: entries with one `url` (a success result and its token counts) are one benchmark,
    named by its success entry's title up to the first comma ("Terminal-Bench 4.0")."""
    groups: dict[str, list[dict]] = {}
    for b in benches:
        groups.setdefault(str(b.get("url") or b.get("id")), []).append(b)
    main = [next((b for b in g if b.get("kind") == "success"), g[0]) for g in groups.values()]
    return [str(b.get("title") or b.get("id")).split(",")[0].strip() for b in main]


def _listed(names: list[str]) -> str:
    """`A`, `A and B`, `A, B and C`; names that differ only in their last word say the rest once
    ("Terminal-Bench 4.0 and 2.1")."""
    heads = {n.rpartition(" ")[0] for n in names}
    if len(names) > 1 and len(heads) == 1 and "" not in heads:
        names = names[:1] + [n.rpartition(" ")[2] for n in names[1:]]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def starting_prior(directory: Path | None = None, benchmarks: Path | None = None) -> dict:
    """What a user's answers start from before any run of theirs: the shipped bundle and the benchmark file.

    `version` is the installed loopmath's (the prior ships inside the package); `bundle_built_by` is the
    manifest's `loopmath_version`. Runs per source come from the manifest, the benchmarks from
    `benchmarks.toml` (counted and named once per benchmark: the entries of one `url` are one benchmark; the
    entries' ids and titles are kept), `built` is `built_at` as a UTC date. `line` says it in one sentence.
    Reading never fails: a missing or unreadable file gives a line that says so.
    """
    from .. import __version__
    from . import manifest
    from .benchmarks import load_benchmarks

    try:
        data = manifest(directory)
    except (OSError, ValueError):
        data = {}
    by_source = {}
    for name, entry in (data.get("sources") or {}).items():
        runs = entry.get("runs", (entry.get("summary") or {}).get("runs"))
        by_source[name] = int(runs or 0)
    try:
        benches = load_benchmarks(benchmarks)["benchmark"]
    except (OSError, ValueError):
        benches = []
    names = _benchmark_names(benches)
    built = _built_date(data.get("built_at"))
    parts = [_runs(by_source) if by_source else "no shipped runs (no prior bundle in this install)"]
    parts.append(f"{len(names)} benchmark{'' if len(names) == 1 else 's'} ({_listed(names)})" if names
                 else "no benchmarks")
    if by_source:
        parts.append(f"built {built}" if built else "build date unknown")
    return {
        "version": __version__,
        "bundle_built_by": data.get("loopmath_version"),
        "runs": sum(by_source.values()),
        "runs_by_source": by_source,
        "benchmarks": {"count": len(names), "names": names, "entries": len(benches),
                       "ids": [str(b.get("id")) for b in benches],
                       "titles": [str(b.get("title") or "") for b in benches]},
        "built": built,
        "built_at": data.get("built_at"),
        "line": f"Starting prior: loopmath {__version__}, " + ", ".join(parts) + ".",
    }


def fit_prior(meta: dict, directory: Path | None = None, benchmarks: Path | None = None) -> dict:
    """The starting prior of a kept fit, for the results page: `starting_prior()`, unless the fit's meta.json
    says another loopmath version made it or it left shipped runs or the benchmarks out (`--no-prior`,
    `--without`). Then `fit` and `line` say what the fit started from: the meta's `runs_by_source` without the
    user's and shared runs (`shared` and `shared:<org>`), and whether it used the benchmarks; the rest stays the installed prior's.
    """
    got = starting_prior(directory, benchmarks)
    options = meta.get("options") or {}
    left_out = set(options.get("without") or []) & {*got["runs_by_source"], "benchmark"}
    version = meta.get("code_version") or got["version"]
    if not isinstance(meta.get("runs_by_source"), dict) or (
            version == got["version"] and not options.get("no_prior") and not left_out):
        return got
    used = {k: int(v) for k, v in meta["runs_by_source"].items() if k != "user" and k.split(":")[0] != "shared"}
    with_benchmarks = not options.get("no_prior") and "benchmark" not in left_out
    got["fit"] = {"version": version, "runs": sum(used.values()), "runs_by_source": used, "benchmarks": with_benchmarks}
    got["line"] = (f"Starting prior of this fit: loopmath {version}, {_runs(used) if used else 'no shipped runs'}, "
                   f"{'with' if with_benchmarks else 'no'} benchmarks. loopmath prior show lists the installed prior.")
    return got
