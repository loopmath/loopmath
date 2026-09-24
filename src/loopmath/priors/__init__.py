"""The shipped prior bundle (our data as OCP v0.3) and benchmark priors (spec 04 section 5).

Owner: lane 11. The read API for the belief model (lane 5):

- `bundle_docs(without=(), sources=None)`: every bundled OCP v0.3 run document,
  each with `run.task.source.kind` set to its source node (`sweep`, `e0`,
  `rq1`, `repo_history`); `fit --without SOURCE` passes `without` (D37).
- `manifest()`, `sources()`, `bundle_dir()`: what the bundle holds and where.

Reading never touches the network or the store; the bundle is package data.
"""

from __future__ import annotations

import gzip
import json
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


def bundle_docs(without: Iterable[str] = (), sources: Iterable[str] | None = None,
                directory: Path | None = None) -> Iterator[dict]:
    """Bundled run documents, source by source in manifest order, runs sorted by id."""
    base = Path(directory or BUNDLE_DIR)
    skip = set(without)
    wanted = set(sources) if sources is not None else None
    for name, entry in manifest(base).get("sources", {}).items():
        if name in skip or (wanted is not None and name not in wanted):
            continue
        with gzip.open(base / entry["file"], "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    doc = json.loads(line)
                    task = doc.setdefault("run", {}).setdefault("task", {})
                    task.setdefault("source", {}).setdefault("kind", name)
                    yield doc


iter_runs = bundle_docs
