"""Link cache for per-session scans.

`scan.py` (Claude transcripts) and `extract.py` (codex rollouts) look a scan up with
`link_cache_get(key)` before scanning and store it with `link_cache_put(key, value)`
afterwards. `key` is `("claude" | "codex", path, mtime_ns, size)` from
`scan.cache_key`; the value is the scan dict exactly as the scanner returned it.

Layout: one JSON file per session file under `<parsed-run cache dir>/links-v2/`, the
parsed-run cache directory `loopmath.ingest.cache_dir` already uses (`LOOPMATH_CACHE_DIR`,
else `<home>/cache` for a `--home` or `LOOPMATH_HOME` store, else ~/.loopmath).
The file name is a hash of the session path; the file holds
the full key, a scanner fingerprint and the encoded value. A hit requires the stored
key to equal the requested one (a rewritten session, new mtime or size, is a miss)
and the stored fingerprint to equal the current one. The fingerprint hashes the
scanner sources (`scan.py`, `bashwrites.py`, `codexio.py`, its split parser/scanner,
`launch.py`, this file)
and the identity of the functions the scan actually calls, resolved at call time
through their module attributes, so an edited scanner, or one replaced under test
(a monkeypatched `writes_from_command`), never gets an older scanner's answers. A
stale entry is overwritten in place by the next put, so the directory never grows
beyond one file per session file.

The cache never changes results: an entry is only ever the scanner's own output for
the same bytes of the same file under the same scanner; the value round-trips
through JSON with the one non-JSON type a scan carries (`frozenset` under
`bash[i]["launch"]`) restored through a typed envelope (a tag key, a version key and
the members) that no ordinary dictionary can collide with, since a dictionary that
carries the tag key is itself escaped; and every hit is a fresh object, so a caller
mutating a scan cannot leak into the next run. Any unreadable, malformed or
structurally wrong entry is a counted miss (never an exception), and a failed write
is counted and ignored: caching is never fatal and never a reason to skip a scan.
`stats()` reports hits, misses (and how many were malformed), puts and write
failures for the process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

CACHE_VERSION = 2
_KINDS = ("claude", "codex")
# The envelope for the values JSON cannot carry. A tagged object is a dict holding
# exactly `_TAG`, `_VER` and `items`, and `_TAG` is a key no scanner writes; an
# ordinary dict that does carry it (user data can be anything) is itself wrapped as
# an escaped `dict` envelope, so no plain scan dictionary is ever decoded as a set.
_TAG = "__dagr_cache_type__"
_VER = "__dagr_cache_v__"
_ENVELOPE_VERSION = 1
_ENVELOPE_TYPES = ("frozenset", "set", "dict")
_SCANNER_SOURCES = (
    "scan.py",
    "bashwrites.py",
    "codexio.py",
    "codexio_parse.py",
    "codexio_scan.py",
    "launch.py",
    "cache.py",
)

_stats = {"hits": 0, "misses": 0, "malformed": 0, "puts": 0, "put_failures": 0}
_source_hash: str | None = None
_root_made: set[str] = set()


class CacheFormatError(ValueError):
    """A stored entry is not the shape `encode` writes. Never raised out of the
    public functions: `link_cache_get` turns it into a counted miss."""


def stats() -> dict:
    """Counts for this process: `hits`, `misses` (`malformed` of them were entries
    that could not be decoded), `puts`, `put_failures`."""
    return dict(_stats)


def reset_stats() -> None:
    for k in _stats:
        _stats[k] = 0


def _sources_hash() -> str:
    """Content hash of the scanner sources, computed once per process."""
    global _source_hash
    if _source_hash is None:
        h = hashlib.sha256()
        here = Path(__file__).resolve().parent
        for name in _SCANNER_SOURCES:
            try:
                h.update((here / name).read_bytes())
            except OSError:
                h.update(name.encode())
        _source_hash = h.hexdigest()
    return _source_hash


def _scan_functions() -> list:
    """The callables a scan runs, looked up through the module attributes the
    scanners use, so a replacement (a test's monkeypatch) is seen."""
    import importlib  # lazy: both modules import this one; `loopmath.graph.extract` the
    # function shadows the module of that name on the package, hence import_module.

    scan = importlib.import_module(f"{__package__}.scan")
    extract = importlib.import_module(f"{__package__}.extract")
    return [scan._scan_claude_session, scan.detect_launch, scan.bashwrites.writes_from_command, extract.scan_codex_session]


def _identity(fn) -> str:
    code = getattr(fn, "__code__", None)
    if code is None:
        return f"{type(fn).__module__}.{type(fn).__qualname__}:{id(fn)}"
    return "|".join((
        str(getattr(fn, "__module__", "")),
        str(getattr(fn, "__qualname__", "")),
        code.co_filename,
        str(code.co_firstlineno),
        hashlib.sha256(code.co_code).hexdigest()[:16],
    ))


def scanner_fingerprint() -> str:
    """Short hash of the scanner sources plus the identity of the functions a scan
    calls right now. Two runs with the same fingerprint scan a file the same way."""
    h = hashlib.sha256(_sources_hash().encode())
    for fn in _scan_functions():
        h.update(_identity(fn).encode("utf-8", "replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def cache_root() -> Path:
    """The link cache directory, under the parsed-run cache directory loopmath uses."""
    from ..ingest.base import cache_dir  # lazy: keeps `loopmath.graph` importable on its own

    return cache_dir() / f"links-v{CACHE_VERSION}"


def _entry_path(key: tuple) -> Path:
    kind, path = key[0], str(key[1])
    digest = hashlib.sha256(path.encode("utf-8", "surrogateescape")).hexdigest()[:32]
    return cache_root() / f"{kind}-{digest}.json"


def _valid_key(key) -> bool:
    return (
        isinstance(key, tuple)
        and len(key) == 4
        and key[0] in _KINDS
        and isinstance(key[1], str)
        and all(isinstance(k, int) and not isinstance(k, bool) for k in key[2:])
    )


def _envelope(kind: str, items: list) -> dict:
    return {_TAG: kind, _VER: _ENVELOPE_VERSION, "items": items}


def encode(value):
    """`value` as plain JSON types: every `frozenset`/`set` becomes a tagged
    envelope holding its sorted members, and a dict that itself carries the tag key
    becomes an escaped `dict` envelope of `[key, value]` pairs, so `decode` restores
    the original exactly and no ordinary dictionary can be mistaken for a set."""
    if isinstance(value, dict):
        if _TAG in value:
            return _envelope("dict", [[str(k), encode(v)] for k, v in value.items()])
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, frozenset):
        return _envelope("frozenset", sorted((encode(v) for v in value), key=_sort_key))
    if isinstance(value, set):
        return _envelope("set", sorted((encode(v) for v in value), key=_sort_key))
    return value


def _sort_key(encoded) -> str:
    """Members of a set in a fixed order whatever their type (strings mostly; a set
    of frozensets encodes to envelopes), so equal sets encode to equal JSON."""
    return json.dumps(encoded, sort_keys=True, default=str)


def decode(value):
    """The inverse of `encode`. Raises `CacheFormatError` on anything that is not
    the shape `encode` writes (a tagged object with a wrong version, an unknown
    type, missing or extra keys, members that cannot sit in a set, a `dict`
    envelope whose pairs are not `[string, value]`)."""
    if isinstance(value, dict):
        if _TAG not in value:
            return {k: decode(v) for k, v in value.items()}
        if set(value) != {_TAG, _VER, "items"}:
            raise CacheFormatError(f"tagged object with keys {sorted(value)}")
        kind, ver, items = value[_TAG], value[_VER], value["items"]
        if ver != _ENVELOPE_VERSION or kind not in _ENVELOPE_TYPES or not isinstance(items, list):
            raise CacheFormatError(f"tagged object {kind!r} version {ver!r}")
        if kind == "dict":
            out = {}
            for pair in items:
                if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                    raise CacheFormatError("dict envelope pair is not [string, value]")
                out[pair[0]] = decode(pair[1])
            return out
        try:
            members = frozenset(decode(v) for v in items)
        except TypeError as exc:
            raise CacheFormatError(f"unhashable set member: {exc}") from exc
        return members if kind == "frozenset" else set(members)
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def _miss(malformed: bool = False):
    _stats["misses"] += 1
    if malformed:
        _stats["malformed"] += 1
    return None


def link_cache_get(key: tuple):
    """The scan stored for `key` by the same scanner, as a fresh object, or None.
    Never raises: an unreadable, malformed or structurally wrong entry is a counted
    miss (`stats()["malformed"]`)."""
    if not _valid_key(key):
        return _miss()
    try:
        with _entry_path(key).open("r", encoding="utf-8") as fh:
            entry = json.load(fh)
    except FileNotFoundError:
        return _miss()
    except (OSError, ValueError, RecursionError):
        return _miss(malformed=True)
    if not isinstance(entry, dict) or "value" not in entry or "key" not in entry:
        return _miss(malformed=True)
    if entry["key"] != list(key) or entry.get("scanner") != scanner_fingerprint():
        return _miss()  # a stale entry (the file was rewritten, or the scanner changed): plain miss
    try:
        value = decode(entry["value"])
    except (CacheFormatError, TypeError, ValueError, RecursionError, AttributeError, KeyError):
        return _miss(malformed=True)
    if not isinstance(value, dict):
        return _miss(malformed=True)
    _stats["hits"] += 1
    return value


def link_cache_put(key: tuple, value: dict) -> None:
    """Store `value` under `key`, atomically (a temp file, then rename). Failure to
    store is counted, never raised. The stored copy is independent of the caller's
    object."""
    if not _valid_key(key) or not isinstance(value, dict):
        _stats["put_failures"] += 1
        return None
    p = _entry_path(key)
    try:
        if str(p.parent) not in _root_made:
            p.parent.mkdir(parents=True, exist_ok=True)
            _root_made.add(str(p.parent))
        payload = {"key": list(key), "scanner": scanner_fingerprint(), "value": encode(copy.deepcopy(value))}
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        tmp.replace(p)
        _stats["puts"] += 1
    except (OSError, TypeError, ValueError):
        _stats["put_failures"] += 1
    return None
