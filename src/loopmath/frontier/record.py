"""Atomic frontier records in the ordinary loopmath store."""
from __future__ import annotations
import json
import re
from pathlib import Path
from ..store.home import NotFound, StoreError
from ..store.commands import UserError
from ..store.ids import new_id
from ..store.lock import atomic_write_json
from . import FRONTIER_SCHEMA


def frontiers_dir(store):
    return Path(store.home) / "frontiers"


def frontier_path(store, frontier):
    if not isinstance(frontier, str) or not re.fullmatch(r"frn_[A-Za-z0-9]+", frontier):
        raise UserError("a frontier id is frn_ followed by letters and digits")
    return frontiers_dir(store) / f"{frontier}.json"


def load(store, frontier):
    path = frontier_path(store, frontier)
    if not path.is_file():
        raise NotFound(f"no frontier {frontier}; loopmath frontier init makes one")
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise StoreError(f"{path}: not JSON: {exc}") from None
    if not isinstance(rec, dict) or rec.get("schema") != FRONTIER_SCHEMA:
        raise StoreError(f"{path}: not a {FRONTIER_SCHEMA} record")
    return rec


def save(store, rec):
    frontiers_dir(store).mkdir(parents=True, exist_ok=True)
    return atomic_write_json(frontier_path(store, rec["frontier"]), rec)


def latest(store):
    found = sorted(p.stem for p in frontiers_dir(store).glob("frn_*.json"))
    if not found:
        raise NotFound("no frontiers; loopmath frontier init makes one")
    return found[-1]
