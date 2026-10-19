"""The harness record: what an agent's harness tree holds, as a digest and a list of named factors.

`spec/harness.md`. A record sits on an OCP run as `run.ext["dev.loopmath.harness"]`; the belief model reads its
factors (`belief/design.py`), so evidence pools across trees that share parts. Any tool can write one: an unknown
factor kind or source is kept and modelled, never refused.

    record_from_tree(path, source="hand", *, parent=None, tool=None, layout="auto") -> dict
    digest(path, layout="auto") -> "sha256:<hex>"
    factors(record) -> tuple of "<kind>:<name>", sorted, without repeats
    resolve_tree(value, folders=(), cwd=None) -> the tree `--harness PATH|auto|none` names, or None
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from .layouts import LAYOUTS, scan, tree_digest

SCHEMA = "loopmath.harness/0.1"
EXT_KEY = "dev.loopmath.harness"
BARE_KEY = "harness"  # read too, never written (`spec/harness.md` section 1)
_KIND = re.compile(r"[A-Za-z0-9-]+")

CHOICES = ("none", "auto")  # the config key `harness.record`; the flag also takes a PATH

__all__ = ["BARE_KEY", "CHOICES", "EXT_KEY", "LAYOUTS", "SCHEMA", "digest", "factors", "git_top", "normalize_record",
           "record_from_tree", "resolve_tree", "run_record"]


def digest(path: str | Path, layout: str = "auto") -> str:
    """The digest of the tree's harness files (`spec/harness.md` section 3)."""
    return tree_digest(scan(Path(path), layout).files)


def record_from_tree(path: str | Path, source: str = "hand", *, parent: str | None = None,
                     tool: dict | None = None, layout: str = "auto") -> dict:
    """The record of the harness tree at `path`. Reads files only; never runs one or follows a symlink."""
    found = scan(Path(path), layout)
    rec: dict[str, Any] = {"schema": SCHEMA, "digest": tree_digest(found.files), "parent": parent,
                           "source": str(source or "hand"), "layouts": sorted(found.layouts),
                           "factors": sorted(found.factors)}
    if tool is not None:
        rec["tool"] = dict(tool)
    return rec


def _factor(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFC", value).strip()
    if not text or "\n" in text or "\r" in text:
        return None
    kind, sep, name = text.partition(":")
    name = name.strip()
    if sep and name and _KIND.fullmatch(kind):
        return f"{kind.lower()}:{name}"
    return f"other:{text}"


def factors(record: Any) -> tuple[str, ...]:
    """The record's factors: cleaned, sorted and unique. A string without a kind is `other:<string>`; a value
    that is not a non-empty one-line string is dropped. Any kind and any schema version is read."""
    if not isinstance(record, dict) or not isinstance(record.get("factors"), list):
        return ()
    return tuple(sorted({f for f in map(_factor, record["factors"]) if f is not None}))


def normalize_record(record: dict) -> dict:
    """The record with its factors cleaned (`factors`); every other field kept as it is."""
    out = dict(record)
    out["factors"] = list(factors(record))
    return out


def run_record(doc: Any) -> dict | None:
    """The harness record on an OCP run document: `run.ext["dev.loopmath.harness"]`, else a bare
    `run.ext["harness"]`; None when there is none."""
    run = doc.get("run") if isinstance(doc, dict) else None
    ext = run.get("ext") if isinstance(run, dict) else None
    if not isinstance(ext, dict):
        return None
    for key in (EXT_KEY, BARE_KEY):
        rec = ext.get(key)
        if isinstance(rec, dict):
            return normalize_record(rec)
    return None


def describe(record: dict, limit: int = 6) -> str:
    """One line for the CLI (`run finish`, `run record`, `runs RUN`): the digest's first 12 hex digits, the source,
    the layouts read when the record says, and the named parts."""
    fs = factors(record)
    shown = ", ".join(fs[:limit]) + (f" and {len(fs) - limit} more" if len(fs) > limit else "")
    words = [str(record.get("source") or "hand")]
    if "layouts" in record:
        words.append(", ".join(map(str, record.get("layouts") or [])) or "no harness files")
    words.append(f"{len(fs)} part{'' if len(fs) == 1 else 's'}: {shown}" if fs else "no named parts")
    return f"{str(record.get('digest') or 'no digest')[:19]} ({'; '.join(words)})"


def git_top(folder: str | Path) -> Path:
    """The top level of the git working tree holding `folder` (the nearest folder with a `.git` entry, as a
    worktree's `.git` file counts too), else `folder` itself. Reads the file system only; never runs git."""
    start = Path(folder).expanduser().resolve()
    return _git_top(start) or start


def _git_top(start: Path) -> Path | None:
    for d in (start, *start.parents):
        if os.path.lexists(d / ".git"):
            return d
    return None


class NoTree(ValueError):
    """`auto` found no single tree: no git working tree holds the folder, or the attempts ran in two trees.
    `detail` says which, without the advice the message adds."""

    def __init__(self, detail: str):
        super().__init__(f"--harness auto: {detail}; pass --harness PATH, or --harness none")
        self.detail = detail


def resolve_tree(value: str | None, folders=(), cwd: str | Path | None = None) -> Path | None:
    """The tree a `--harness` value names: None for `none` (or no value); for `auto`, the git top level of the
    folders the run's attempts ran in, else of `cwd`; else the path itself. NoTree (a ValueError) when `auto`
    finds two trees or no git working tree; ValueError when the path is not a folder."""
    if value is None or value == "none":
        return None
    if value == "auto":
        starts = sorted({Path(f).expanduser().resolve() for f in folders if f}, key=str)
        where = "the run's attempts ran in" if starts else "the command runs in"
        here = Path(cwd if cwd is not None else Path.cwd()).expanduser().resolve()
        found = {s: _git_top(s) for s in starts or [here]}
        outside = [s for s, top in found.items() if top is None]
        if outside:
            noun = "folders" if len(outside) > 1 else "folder"
            raise NoTree(f"no git working tree holds the {noun} {where} ({', '.join(map(str, outside))})")
        tops = sorted(set(found.values()), key=str)
        if len(tops) > 1:
            raise NoTree(f"the run's attempts ran in {len(tops)} trees ({', '.join(map(str, tops))})")
        return tops[0]
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ValueError(f"--harness: not a folder: {value}")
    return path.resolve()
