"""Edits: atomic changes to a base harness tree, each one subfolder of the edits folder, laid over the base.

Inside an edit's folder, a file `PATH.append` appends its text to the base's PATH (making it if missing), a file
`PATH.remove` removes each of its lines from the base's PATH (each must be there), and any other file is copied to
PATH. A tree for a subset of edits is the base, then every copy, then every removal, then every append in edit
name order, so the order of the edits never changes the result. Two edits that copy to one path are refused at
`screen start`. `.git`, `.DS_Store` and `__pycache__` are skipped (`.claude/` is not); symbolic links are refused.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from ..store.commands import UserError
from ..store.home import NotFound

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
APPEND, REMOVE = ".append", ".remove"
MAX_EDITS = 31
SKIP = {".git", ".hg", ".svn", ".DS_Store", "__pycache__"}


def _files(root: Path) -> list[Path]:
    """The files under `root` but `SKIP`, sorted by relative path; a symbolic link is refused."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
        for d in dirnames:
            if (Path(dirpath) / d).is_symlink():
                raise UserError(f"{Path(dirpath) / d}: a symbolic link; an edit or base holds plain files")
        for f in sorted(filenames):
            if f in SKIP:
                continue
            p = Path(dirpath) / f
            if p.is_symlink():
                raise UserError(f"{p}: a symbolic link; an edit or base holds plain files")
            out.append(p)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def digest(root: Path) -> str:
    """sha256 over every file's relative path and bytes (`SKIP` left out)."""
    h = hashlib.sha256()
    for p in _files(root):
        h.update(p.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


def _op(rel: str) -> tuple[str, str]:
    """(kind, target path) of one file in an edit."""
    for suffix, kind in ((APPEND, "append"), (REMOVE, "remove")):
        if rel.endswith(suffix) and len(rel) > len(suffix) and not rel.endswith("/" + suffix):
            return kind, rel[: -len(suffix)]
    return "copy", rel


def load(base: Path, folder: Path) -> list[dict[str, Any]]:
    """The edits in `folder`, sorted by name: [{"name", "path", "digest", "ops": [{"kind", "file", "target"}]}].
    Refuses an empty folder, an edit with no files, two copies to one path, and a removal of a line the base's
    file does not have."""
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise NotFound(f"--edits {folder}: no folder there; give a folder with one subfolder per edit")
    edits = []
    for sub in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not NAME_RE.match(sub.name):
            raise UserError(f"edit {sub.name!r}: a name is letters, digits and . _ + -, starting with a letter or "
                            f"digit")
        files = _files(sub)
        if not files:
            raise UserError(f"edit {sub.name} in {folder} has no files")
        ops = []
        for p in files:
            rel = p.relative_to(sub).as_posix()
            kind, target = _op(rel)
            ops.append({"kind": kind, "file": rel, "target": target})
        edits.append({"name": sub.name, "path": str(sub.resolve()), "digest": digest(sub), "ops": ops})
    if not edits:
        raise UserError(f"no edits in {folder}: each edit is a subfolder holding the files it adds or changes "
                        f"(PATH, PATH.append or PATH.remove)")
    if len(edits) > MAX_EDITS:
        raise UserError(f"{len(edits)} edits in {folder}: a screen takes at most {MAX_EDITS}")
    copies: dict[str, str] = {}
    for e in edits:
        for op in e["ops"]:
            if op["kind"] != "copy":
                continue
            if op["target"] in copies:
                raise UserError(f"edits {copies[op['target']]} and {e['name']} both write {op['target']}; an edit "
                                f"that adds to a file the other also writes can use {op['target']}{APPEND}")
            copies[op["target"]] = e["name"]
        for op in e["ops"]:
            if op["kind"] == "remove":
                have = _lines(Path(base) / op["target"])
                for line in _lines(Path(e["path"]) / op["file"]):
                    if line.strip() and line not in have:
                        raise UserError(f"edit {e['name']} removes a line {op['target']} in the base does not hold: "
                                        f"{line!r}")
    return edits


def _lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []


def build(base: Path, edits: list[dict[str, Any]], on: Iterable[str], out: Path) -> Path:
    """Write the base with the edits named in `on` laid over it into `out`, a new folder."""
    on = set(on)
    chosen = [e for e in edits if e["name"] in on]
    unknown = on - {e["name"] for e in edits}
    if unknown:
        raise UserError(f"no edit {', '.join(sorted(unknown))}; the edits are {', '.join(e['name'] for e in edits)}")
    out = Path(out)
    shutil.copytree(base, out, ignore=shutil.ignore_patterns(*SKIP), symlinks=True)
    for kind in ("copy", "remove", "append"):
        for e in chosen:
            for op in e["ops"]:
                if op["kind"] != kind:
                    continue
                src, dst = Path(e["path"]) / op["file"], out / op["target"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                if kind == "copy":
                    shutil.copyfile(src, dst)
                elif kind == "remove":
                    drop = [line for line in _lines(src) if line.strip()]
                    kept = _lines(dst)
                    for line in drop:
                        if line in kept:
                            kept.remove(line)
                    dst.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
                else:
                    text = dst.read_text(encoding="utf-8") if dst.exists() else ""
                    if text and not text.endswith("\n"):
                        text += "\n"
                    dst.write_text(text + src.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def subset_key(on: Iterable[str]) -> str:
    """A short stable name for a subset of edits, for its tree's folder."""
    joined = "\n".join(sorted(on))
    return "none" if not joined else hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def built(base: Path, edits: list[dict[str, Any]], on: Iterable[str], cache: Path) -> Path:
    """The tree for `on` under `cache`, built once: into a temporary folder, then renamed into place."""
    on = sorted(on)
    target = Path(cache) / subset_key(on)
    if target.is_dir():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    tmp = Path(cache) / f".tmp-{subset_key(on)}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    build(base, edits, on, tmp)
    try:
        os.rename(tmp, target)
    except OSError:  # another process built it first
        shutil.rmtree(tmp, ignore_errors=True)
    return target
