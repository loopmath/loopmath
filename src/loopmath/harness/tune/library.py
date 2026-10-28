"""The role library: lessons the reflector wrote, filed by domain, for later editors to read.

    <library>/<domain>/lessons.jsonl     one JSON object per line, appended, never rewritten

`library.path` (default `$LOOPMATH_HOME/harness/library`) is the root; `library.domain` (default: the repo's
folder name) picks the folder. A line is

    {"v": 1, "id": "lesson-<12 hex>", "ts": ..., "domain": ..., "lesson": ..., "applies_when": ...,
     "evidence": ..., "confidence": "low" | "medium" | "high", "edits": [{"slot", "effect"}], "tune": ...,
     "task": ..., "round": ..., "why": ..., "kept": [...]}

with the engine's meta keys (tune, task, round, why, kept, and any others) beside the lesson. The id hashes the
domain, the tune, the task, the round and the lesson text, so a resumed tune that adds the same lesson again
appends nothing. A torn last line is skipped on read. Editors get the domain's lessons, newest first, as one
markdown file (`render`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...store.lock import append_line, read_jsonl
from .contracts import Config, Lesson, now_iso, record

__all__ = ["JsonlLibrary", "LESSONS_FILE", "domain_key", "read_lessons", "render", "lesson_id", "home_root"]

LESSONS_FILE = "lessons.jsonl"
RENDER_LIMIT = 20
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_RESERVED = ("v", "id", "ts", "domain", "lesson", "applies_when", "evidence", "confidence", "edits")


def home_root(home: Path | None = None) -> Path:
    """`$LOOPMATH_HOME/harness/library/`."""
    if home is None:
        env = os.environ.get("LOOPMATH_HOME")
        home = Path(env).expanduser() if env else Path.home() / ".loopmath"
    return Path(home) / "harness" / "library"


def domain_key(domain: str) -> str:
    """A domain as a folder name: letters, digits, `.`, `_` and `-`, at most 80 characters."""
    key = _UNSAFE.sub("-", unicodedata.normalize("NFKD", str(domain)).encode("ascii", "ignore").decode())
    key = key.strip(".-_")[:80].strip(".-_")
    return key or "default"


def lesson_id(domain: str, lesson: str, meta: Mapping[str, Any]) -> str:
    parts = [domain_key(domain), str(meta.get("tune", "")), str(meta.get("task", "")), str(meta.get("round", "")),
             " ".join(lesson.split())]
    return "lesson-" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]


def read_lessons(folder: Path | None) -> list[dict]:
    """The lessons in a domain folder, oldest first; lines without a text lesson are skipped."""
    if folder is None:
        return []
    return [r for r in read_jsonl(Path(folder) / LESSONS_FILE)
            if isinstance(r.get("lesson"), str) and r["lesson"].strip()]


def _line(text: Any) -> str:
    return " ".join(str(text or "").split())


def render(lessons: Sequence[Mapping[str, Any]], limit: int = RENDER_LIMIT) -> str:
    """The lessons, newest first and at most `limit`, as markdown for an editor's working folder."""
    picked = list(lessons)[::-1][:max(0, limit)]
    out = ["# Lessons from earlier tasks", "",
           f"{len(picked)} of {len(lessons)} lessons, newest first. Each comes from tuning another task.", ""]
    for i, r in enumerate(picked, 1):
        out.append(f"{i}. {_line(r.get('lesson'))}")
        if r.get("applies_when"):
            out.append(f"   - Applies when: {_line(r['applies_when'])}")
        if r.get("evidence"):
            out.append(f"   - Evidence: {_line(r['evidence'])}")
        src = ", ".join(f"{k} {_line(r[k])}" for k in ("task", "tune") if r.get(k))
        out.append(f"   - Confidence: {_line(r.get('confidence') or 'low')}" + (f"; from {src}" if src else ""))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


class JsonlLibrary:
    """The `Library` protocol over `<root>/<domain>/lessons.jsonl`."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).expanduser() if root else home_root()

    @classmethod
    def from_config(cls, cfg: Config) -> "JsonlLibrary":
        return cls(cfg.get("library.path") or None)

    def path(self, domain: str) -> Path:
        return self.root / domain_key(domain)

    def lessons(self, domain: str) -> list[dict]:
        return read_lessons(self.path(domain))

    def folder(self, domain: str) -> Path | None:
        """The domain's folder when it holds at least one lesson, else None."""
        return self.path(domain) if self.lessons(domain) else None

    def add(self, domain: str, lesson: Lesson, meta: Mapping[str, Any]) -> str | None:
        text = lesson.lesson
        if not isinstance(text, str) or not text.strip():
            return None
        lid = lesson_id(domain, text, meta)
        folder = self.path(domain)
        if any(r.get("id") == lid for r in read_jsonl(folder / LESSONS_FILE)):
            return lid
        row: dict[str, Any] = {"v": 1, "id": lid, "ts": now_iso(), "domain": domain_key(domain),
                               "lesson": text.strip(), "applies_when": lesson.applies_when or "",
                               "evidence": lesson.evidence or "", "confidence": lesson.confidence or "low",
                               "edits": record(list(lesson.edits or []))}
        for k, v in meta.items():
            if k not in _RESERVED:
                row[str(k)] = record(v)
        json.dumps(row)  # refuse a row that is not plain JSON before anything is written
        append_line(folder / LESSONS_FILE, row)
        return lid
