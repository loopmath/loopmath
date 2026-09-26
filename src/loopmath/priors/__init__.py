"""The shipped prior bundle (our data as OCP v0.3) and benchmark priors (spec 04 section 5).

The read API for the belief model (lane 5):

- `bundle_docs(without=(), sources=None)`: every bundled OCP v0.3 run document,
  each with `run.task.source.kind` set to its source node (`sweep`, `e0`,
  `rq1`, `repo_history`); `fit --without SOURCE` passes `without`.
- `bundle_entries()`: the same documents as `(source, document)` pairs, so the fit
  knows where each run came from whatever its label says (spec 04 section 1).
- `shipped_overlap(run_ids)` and `overlap_note(counts)`: how many of the user's runs
  are also in a shipped source; `run import`, `fit` and `status` say it once. A
  stored run and its shipped copy are matched on `run_id_of`, the run id less a
  start stamp (`stable_run_id`), since the shipped rq1 ids lose theirs (23B).
- `manifest()`, `sources()`, `bundle_dir()`: what the bundle holds and where.

Reading never touches the network or the store; the bundle is package data.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

BUNDLE_DIR = Path(__file__).resolve().parent / "bundle"


def bundle_dir() -> Path:
    return BUNDLE_DIR


def manifest(directory: Path | None = None) -> dict:
    """The bundle manifest, or an empty one when no bundle has been built."""
    path = Path(directory or BUNDLE_DIR) / "manifest.json"
    if not path.is_file():
        return {"schema": "loopmath.prior.manifest/1", "sources": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def sources(directory: Path | None = None) -> list[str]:
    return sorted(manifest(directory).get("sources", {}))


def bundle_entries(sources: Iterable[str] | None = None,
                   directory: Path | None = None) -> Iterator[tuple[str, dict]]:
    """(source, run document) for every bundled run, source by source in manifest order.

    The source is the manifest's name for the file, whatever the document's label says.
    """
    base = Path(directory or BUNDLE_DIR)
    wanted = set(sources) if sources is not None else None
    for name, entry in manifest(base).get("sources", {}).items():
        if wanted is not None and name not in wanted:
            continue
        with gzip.open(base / entry["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    doc = json.loads(line)
                    task = doc.setdefault("run", {}).setdefault("task", {})
                    task.setdefault("source", {}).setdefault("kind", name)
                    yield name, doc


def bundle_docs(without: Iterable[str] = (), sources: Iterable[str] | None = None,
                directory: Path | None = None) -> Iterator[dict]:
    """Bundled run documents, source by source in manifest order, runs sorted by id."""
    skip = set(without)
    wanted = set(sources) if sources is not None else None
    names = [n for n in manifest(directory).get("sources", {}) if n not in skip and (wanted is None or n in wanted)]
    for _name, doc in bundle_entries(names, directory):
        yield doc


_STAMP = re.compile(r"-\d{8}-\d{6}(?=-|$)")  # lane 10's run start, YYYYMMDD-HHMMSS local time


def stable_run_id(run_id: str) -> str:
    """A run id less its `-YYYYMMDD-HHMMSS` start stamp; an id without one is unchanged.

    The shipped rq1 runs carry lane 10's ids without the stamp (a real date and time
    does not ship, 23B) while an RQ1 store keeps it, so a stored run and its shipped
    copy compare on this key. It is the id itself, not a hash: the stamp space is
    small enough that a hash of the stamped id would give the time away.
    """
    return _STAMP.sub("", run_id)


def run_id_of(doc: dict) -> str:
    """The key a stored run and its shipped copy are matched on, read without parsing the run.

    The run id the fit keys on (`belief.design.parse_run`), less a start stamp
    (`stable_run_id`); the fit and `shipped_overlap` both compare on it.
    """
    run = doc.get("run") or {}
    task = run.get("task") or {}
    return stable_run_id(str(run.get("id") or task.get("id") or (run.get("labels") or {}).get("task") or "unknown"))


def shipped_ids(directory: Path | None = None) -> dict[str, str]:
    """Run id (`run_id_of`) to its shipped source, for every bundled run."""
    out: dict[str, str] = {}
    for name, doc in bundle_entries(directory=directory):
        out.setdefault(run_id_of(doc), name)
    return out


def shipped_overlap(run_ids: Iterable[str], directory: Path | None = None) -> dict[str, int]:
    """How many of `run_ids` (the user's runs) are also in each shipped source.

    A fit uses the store's copy of such a run and leaves the shipped copy out. The
    ids compare as `run_id_of` does, without a start stamp.
    """
    ids = shipped_ids(directory)
    keys = [stable_run_id(r) for r in set(run_ids)]
    return dict(Counter(ids[k] for k in keys if k in ids))


def overlap_note(counts: dict[str, int] | None) -> str | None:
    """One line for `run import`, `fit` and `status`, or None when no run overlaps."""
    overlap = {k: int(v) for k, v in (counts or {}).items() if v}
    if not overlap:
        return None
    total = sum(overlap.values())
    if len(overlap) == 1:
        where = f"the shipped {next(iter(overlap))} prior"
    else:
        where = "the shipped prior (" + ", ".join(f"{k} {v}" for k, v in sorted(overlap.items(), key=lambda kv: -kv[1])) + ")"
    if total == 1:
        return f"1 of your runs is also in {where} (same run): fits use your copy"
    return f"{total} of your runs are also in {where} (same runs): fits use your copies"


iter_runs = bundle_docs
