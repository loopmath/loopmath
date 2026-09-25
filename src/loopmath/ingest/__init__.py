"""Log discovery and parser dispatch.

`discover` walks the two harness log roots and yields candidate session files;
`parse_all` turns them into run records. Both are read-only on the log roots by
construction: nothing in this package opens a log file for writing.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    DEFAULT_CACHE_DIR,
    EFFORTS,
    MODEL_ALIASES,
    RunRecord,
    Tokens,
    cache_dir,
    canonical_effort,
    canonical_model,
    file_kind,
    iter_jsonl,
    parse_ts,
    repo_rel_dir,
    summarize_writes,
    ts_epoch,
    workspace_name,
)

__all__ = [
    "DEFAULT_CACHE_DIR",
    "EFFORTS",
    "MODEL_ALIASES",
    "RunRecord",
    "Tokens",
    "cache_dir",
    "canonical_effort",
    "canonical_model",
    "discover",
    "file_kind",
    "iter_jsonl",
    "parse_all",
    "parse_ts",
    "repo_rel_dir",
    "summarize_writes",
    "ts_epoch",
    "workspace_name",
]

CLAUDE_CODE_ROOT = Path.home() / ".claude" / "projects"
CODEX_ROOT = Path.home() / ".codex" / "sessions"


def discover(
    logs: str | Path | None = None, since_days: float | None = None
) -> dict[str, list[Path]]:
    """Find session files per harness.

    With no argument, both default roots are scanned. With `--logs PATH` (the
    SPEC section 6 escape hatch) that one directory is scanned for both shapes:
    files named `rollout-*.jsonl` are codex, other `*.jsonl` files are Claude
    Code. Returns `{"claude-code": [...], "codex": [...]}` with sorted paths.

    `since_days` keeps only files modified within that many days. The local log
    roots hold about ten gigabytes, so a full cold parse is minutes, not the
    seconds SPEC section 6 promises. The window is the honest fix: the report
    prints which window it used, and `--all` turns it off.
    """
    cutoff = None
    if since_days is not None:
        import time

        cutoff = time.time() - float(since_days) * 86400.0

    def _keep(paths: "list[Path]") -> "list[Path]":
        if cutoff is None:
            return paths
        kept = []
        for p in paths:
            try:
                if p.stat().st_mtime >= cutoff:
                    kept.append(p)
            except OSError:
                continue
        return kept

    if logs is None:
        cc = sorted(CLAUDE_CODE_ROOT.rglob("*.jsonl")) if CLAUDE_CODE_ROOT.is_dir() else []
        cx = sorted(CODEX_ROOT.rglob("*.jsonl")) if CODEX_ROOT.is_dir() else []
        return {"claude-code": _keep(cc), "codex": _keep(cx)}

    root = Path(logs).expanduser()
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(root.rglob("*.jsonl"))
    else:
        candidates = []
    cc, cx = [], []
    for p in candidates:
        (cx if p.name.startswith("rollout-") else cc).append(p)
    return {"claude-code": _keep(cc), "codex": _keep(cx)}


def _discovery_manifest(found: dict) -> tuple[tuple[str, Path, int | None, int | None], ...]:
    """Freeze canonical paths and their exact cache stamps in one immutable value."""
    manifest = []
    for harness in sorted(found):
        for raw_path in sorted((Path(path) for path in found[harness]), key=str):
            try:
                stat = raw_path.stat()
                mtime_ns, size = stat.st_mtime_ns, stat.st_size
            except OSError:
                mtime_ns, size = None, None
            manifest.append((str(harness), raw_path, mtime_ns, size))
    return tuple(manifest)


def _manifest_paths(discovered, harness: str) -> tuple[Path, ...]:
    """Return one harness's canonical paths from a frozen discovery manifest."""
    return tuple(path for name, path, _, _ in discovered if name == harness)


# FIX 4 (reviewer round, blocker): bumped from 1 to 2 alongside `_stamp`'s
# move to nanosecond mtime resolution. A cache file written under the old,
# whole-second stamp could otherwise be read back and matched against a
# path whose real mtime now differs only at sub-second resolution, serving
# a stale record with no sign anything was wrong. Bumping the version
# changes the cache file's name (`_cache_path` below), so an old-format
# cache is never opened under the new scheme; it is simply rebuilt (the
# cache is rebuildable by construction, so this costs one cold parse, not
# a migration).
CACHE_VERSION = 2

# Key under which the parser fingerprint is stored inside the cache file. It is
# not a session path, so lookups can never collide with a real entry.
_FINGERPRINT_KEY = "__parser_fingerprint__"


def parser_fingerprint() -> str:
    """Short content hash of the parsers, so a parser edit invalidates the cache.

    The cache is keyed on `(path, mtime, size)` of the LOG file, which answers
    "has this log changed" but not "would today's parser read it differently".
    Those come apart exactly when a parser is fixed, which is the moment a
    stale cached record is most dangerous: a bug fix would appear to change
    nothing because the wrong old answers were served straight back. Hashing
    the parser sources closes that, and costs one read of four small files.
    `codex_support.py` is included because it decides which rollout files
    form one Codex record.
    """
    import hashlib

    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in ("base.py", "claude_code.py", "codex.py", "codex_support.py"):
        try:
            h.update((here / name).read_bytes())
        except OSError:
            # An unreadable parser source is not a reason to fail a parse run;
            # fold in the name so the fingerprint still changes if it returns.
            h.update(name.encode())
    return h.hexdigest()[:16]


def _cache_path() -> Path:
    return cache_dir() / f"parsed-v{CACHE_VERSION}.json"


def load_cache() -> dict:
    """Load the parsed-run cache. Any problem yields an empty cache.

    The cache is rebuildable by construction: it holds only what the
    parsers derived from logs that are still on disk, so a corrupt or
    stale-format cache is discarded rather than repaired. A cache written by a
    different build of the parsers is discarded wholesale for the same reason:
    see `parser_fingerprint`.
    """
    import json

    p = _cache_path()
    if not p.is_file():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    if data.get(_FINGERPRINT_KEY) != parser_fingerprint():
        return {}
    return data


def save_cache(cache: dict) -> None:
    """Write the cache atomically. Failure to cache is never fatal."""
    import json

    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        payload = dict(cache)
        payload[_FINGERPRINT_KEY] = parser_fingerprint()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(p)
    except OSError:
        pass


def _stamp(path: Path) -> str:
    """`(mtime_ns, size)` fingerprint of a log file's on-disk state.

    FIX 4 (reviewer round, blocker): this used to truncate to whole-second
    resolution (`int(st.st_mtime)`). A file rewritten to the same size within
    the same wall-clock second stamped identically under that scheme, so the
    cache would silently hand back the STALE record for the NEW content with
    nothing in the output to say so. `st.st_mtime_ns` (nanosecond resolution)
    fixes that; `CACHE_VERSION` was bumped alongside it so a cache file
    written under the old, coarser stamp is never read back under this one.
    """
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


# `parse_all` used to count a skip without recording why; the "no assistant
# turns or unreadable" wording that beat 3 printed named two possible causes
# it never actually checked. This measures it instead: a skipped file is
# re-read once (through the same tolerant `iter_jsonl` every parser already
# uses) and sorted into exactly one of these buckets. Re-reading only the
# files that were already skipped bounds the added cost to the skip count,
# not the corpus size; `_SKIP_CLASSIFY_CAP` bounds it further for a corpus
# with an unusually large skip count.
_SKIP_CLASSIFY_CAP = 1000
_SKIP_CAP_REASON = "not classified (skip count over the 1000-file classification cap)"

# The one structural marker each harness's assistant turns carry: a Claude
# Code line's own `type` field, or a Codex top-level `type` of
# `"response_item"` (the envelope every model turn arrives in; see
# `loopmath.ingest.codex`). This does not distinguish an assistant response_item
# from any other kind: measuring only "at least one exists" is what the brief
# asks for, not a full re-parse of the session.
_ASSISTANT_MARKER = {"claude-code": "assistant", "codex": "response_item"}


def _classify_skip(path: Path, harness: str) -> str:
    """Best-effort, MEASURED reason one skipped session file did not parse.

    Only called for a file a parser (or a cached prior parse) already
    returned `None` for. Never raises: `iter_jsonl` already swallows both
    unreadable files and unparseable lines, so "zero parseable JSON objects"
    covers both an empty file and a file this process cannot read.
    """
    rows = iter_jsonl(path)
    if not rows:
        return "unreadable or empty"
    marker = _ASSISTANT_MARKER.get(harness)
    if marker is not None and any(row.get("type") == marker for row in rows):
        return "readable but produced no run record"
    return "no assistant turns"


def _parse_unit(job: "tuple[str, list[Path]]") -> "dict | None":
    """One unit's record dict with `_signals`, or None when it yields no record.
    Module level so a worker process can run it."""
    from . import claude_code, codex

    harness, unit = job
    try:
        rec = codex.parse_session(unit) if harness == "codex" else claude_code.parse_session(unit[0])
    except Exception:  # a single malformed log must not kill the run
        rec = None
    if rec is None:
        return None
    d = rec.to_dict()
    d["_signals"] = rec.grading_signals()
    return d


def parse_all(
    logs: str | Path | None = None,
    harnesses: "tuple[str, ...]" = ("claude-code", "codex"),
    limit: int | None = None,
    use_cache: bool = True,
    progress=None,
    since_days: float | None = None,
    discovered=None,
) -> "tuple[list[dict], dict]":
    """Parse every discovered session into SPEC section 3 run-record dicts.

    Returns `(records, diagnostics)`. Each record dict carries the SPEC-3 body
    plus a `_signals` key holding `RunRecord.grading_signals()`, so grade.py
    never has to reopen a log. Diagnostics report files seen, records produced,
    files skipped, continuation files grouped into logical records, and cache
    hits per harness; skips are printed, never hidden.

    The local log roots hold gigabytes, so a cold parse is minutes and a warm
    one is seconds. The cache is keyed by (path, mtime, size): an edited or
    appended log reparses, an untouched one does not.

    `discovered` may supply a previously frozen immutable discovery manifest
    containing canonical harness/path pairs and their captured mtime_ns/size.
    When supplied, this function uses those exact stamps for cache decisions,
    never stats those paths again, and never calls `discover`. Omitting it
    preserves the legacy behavior by discovering and freezing a manifest here.

    FIX 2 (reviewer round, blocker): `limit` truncates the per-harness path
    list to the first N files BEFORE anything below is parsed, so a limited
    run's diagnostics must not describe that truncated list as though it were
    the whole corpus. `diag["files_seen"]` means ATTEMPTED, not discovered:
    the count of files this call actually tried to parse, after any `limit`
    truncation. (The alternative reading, "discovered", is what
    `ingest.discover` already answers on its own -- `cli.analyze` calls it
    separately, before `limit` is applied, to print "scanning N session
    files" -- so `files_seen` keeping the "attempted" meaning is what makes
    the cache-hit ratio printed in `cli.analyze` ["N of files_seen from
    cache"] correct: every file counted in `files_seen` is one the cache
    lookup in this function actually ran against.) What a `limit` cut is
    reported separately and always: `diag["files_omitted_by_limit"]` (per
    harness and `"total"`) counts discovered-but-not-attempted files, present
    and 0 when no `limit` was given, never merged into or confused with
    `files_seen`. `diag["limit"]` records the raw `limit` value passed in
    (`None` when not given), so a caller does not have to thread it through
    separately to explain the omitted count.
    """
    from . import codex
    from .. import pool

    manifest = (
        _discovery_manifest(discover(logs, since_days=since_days))
        if discovered is None
        else discovered
    )
    found: dict[str, list[tuple[Path, int | None, int | None]]] = {}
    for harness, path, mtime_ns, size in manifest:
        found.setdefault(harness, []).append((path, mtime_ns, size))
    cache = load_cache() if use_cache else {}
    records: list[dict] = []
    diag: dict = {
        "files_seen": {},
        "records": {},
        "skipped": {},
        "grouped_continuation_files": {},
        "cache_hits": 0,
        "skip_reasons": {},
        # FIX 2: always present (0 per harness, and 0 total, when no `limit`
        # was given), never only added when it matters -- a caller must never
        # have to special-case its absence to know nothing was cut.
        "files_omitted_by_limit": {},
        "limit": limit,
        # Flag 6: count of parsed records whose `_signals["zero_token_synthetic"]`
        # is true (Claude Code sessions whose assistant lines are all the
        # harness's own `"<synthetic>"` bookkeeping turn with an all-zero
        # usage block). Counted here, at ingest time, over EVERY record this
        # call returns -- fresh parses and cache hits alike, since a cache hit
        # already carries `_signals` from when it was parsed -- so the key is
        # always present and correct regardless of cache state. grade.py's
        # `grade_all` is what actually excludes these from the graded
        # population; this key exists so ingest diagnostics can report the
        # count even before grading runs.
        "zero_token_synthetic": 0,
    }
    dirty = False
    skip_budget = _SKIP_CLASSIFY_CAP

    def _tally_skip(path: Path, harness: str) -> None:
        # Mutates `skip_budget` via the enclosing scope (not a module global)
        # so the 1000-file classification cap is shared across both harness
        # loops, not reset per harness.
        nonlocal skip_budget
        if skip_budget > 0:
            reason = _classify_skip(path, harness)
            skip_budget -= 1
        else:
            reason = _SKIP_CAP_REASON
        diag["skip_reasons"][reason] = diag["skip_reasons"].get(reason, 0) + 1

    # First pass: which units the cache answers and which need parsing. The misses
    # are parsed together (in worker processes when there are many), then the second
    # pass below walks every unit in order, exactly as a one-by-one parse would.
    plans = []
    misses: list[tuple[str, list[Path]]] = []
    miss_weights: list[int] = []
    for harness in harnesses:
        harness_entries = found.get(harness, [])
        entries = harness_entries[:limit] if limit is not None else harness_entries
        paths = [path for path, _, _ in entries]
        stamps = {path: (mtime_ns, size) for path, mtime_ns, size in entries}
        units = codex.group_session_paths(paths) if harness == "codex" else [[p] for p in paths]
        plans.append((harness_entries, paths, stamps, units))
        for unit in units:
            unit_stamps = [stamps[path] for path in unit]
            if any(mtime_ns is None or size is None for mtime_ns, size in unit_stamps):
                continue
            key = str(unit[0]) if len(unit) == 1 else "\0".join(str(path) for path in unit)
            stamp = "|".join(f"{mtime_ns}:{size}" for mtime_ns, size in unit_stamps)
            hit = cache.get(key)
            if use_cache and isinstance(hit, dict) and hit.get("stamp") == stamp:
                continue
            misses.append((harness, unit))
            miss_weights.append(sum(size for _, size in unit_stamps))
    parsed = dict(zip(
        ((h, tuple(u)) for h, u in misses),
        pool.ordered_map(_parse_unit, misses, weights=miss_weights,
                         done=(lambda i, n: progress("sessions", i, n)) if progress is not None else None),
    ))

    for harness, (harness_entries, paths, stamps, units) in zip(harnesses, plans):
        # FIX 2: what `limit` removed is counted here, at truncation time, not
        # inferred later from a difference of two other numbers a caller would
        # have to know to compute.
        diag["files_omitted_by_limit"][harness] = len(harness_entries) - len(paths)
        # A resumed Codex thread may continue in another rollout file with
        # the same persisted session id. It must be parsed as one logical
        # stream before attempt_rule v1 sees the records; otherwise one user
        # attempt is spuriously counted once per physical file (units, above).
        n_ok = 0
        n_skip = 0
        for i, unit in enumerate(units):
            key = str(unit[0]) if len(unit) == 1 else "\0".join(str(path) for path in unit)
            unit_stamps = [stamps[path] for path in unit]
            if any(mtime_ns is None or size is None for mtime_ns, size in unit_stamps):
                n_skip += len(unit)
                for path in unit:
                    _tally_skip(path, harness)
                continue
            stamp = "|".join(f"{mtime_ns}:{size}" for mtime_ns, size in unit_stamps)
            hit = cache.get(key)
            if use_cache and isinstance(hit, dict) and hit.get("stamp") == stamp:
                diag["cache_hits"] += len(unit)
                rec_d = hit.get("record")
                if rec_d is None:
                    n_skip += len(unit)
                    for path in unit:
                        _tally_skip(path, harness)
                    continue
                records.append(rec_d)
                n_ok += 1
                continue
            d = parsed[(harness, tuple(unit))]
            dirty = True
            if d is None:
                cache[key] = {"stamp": stamp, "record": None}
                n_skip += len(unit)
                for path in unit:
                    _tally_skip(path, harness)
                continue
            cache[key] = {"stamp": stamp, "record": d}
            records.append(d)
            n_ok += 1
        diag["files_seen"][harness] = len(paths)
        diag["records"][harness] = n_ok
        diag["skipped"][harness] = n_skip
        # A successful Codex parse can combine a resumed session's physical
        # files into one logical record. Name those extra files separately so
        # files_seen always reconciles with records, skips, and continuations.
        diag["grouped_continuation_files"][harness] = len(paths) - n_ok - n_skip
    diag["files_seen"]["total"] = sum(
        v for k, v in diag["files_seen"].items() if k != "total"
    )
    diag["files_omitted_by_limit"]["total"] = sum(
        v for k, v in diag["files_omitted_by_limit"].items() if k != "total"
    )
    diag["records"]["total"] = len(records)
    diag["skipped"]["total"] = sum(v for k, v in diag["skipped"].items() if k != "total")
    diag["grouped_continuation_files"]["total"] = sum(
        v for k, v in diag["grouped_continuation_files"].items() if k != "total"
    )
    diag["zero_token_synthetic"] = sum(
        1 for r in records if (r.get("_signals") or {}).get("zero_token_synthetic")
    )
    if use_cache and dirty:
        save_cache(cache)
    return records, diag
