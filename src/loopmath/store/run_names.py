"""Run ids to run file names.

A run id lives inside its document (`run.id`); the store reads ids from the documents and the
index, never from file names. The file name is the id with every character outside
`[A-Za-z0-9._-]` written as `%XX` per UTF-8 byte, so `%` itself is escaped too. The map is one
to one and reversible: `a:b` is stored as `a%3Ab`, `a_b` stays `a_b`, and an id that is a safe
name already (every id loopmath mints for its own runs) is its own file name, so stores written
before this keep their file names. Adapters and `graph --format run` mint ids such as
`pi-adapt:0ec5...` and `loopmath-graph:WORKSPACE`, which `run import` stores this way.
"""

from __future__ import annotations

import string
import unicodedata
from urllib.parse import unquote

SAFE = frozenset(string.ascii_letters + string.digits + "._-")
FIRST = frozenset(string.ascii_letters + string.digits)
MAX_NAME = 200
RULE = "it starts with a letter or digit, has no control characters and fits in 200 characters as a file name"


def file_stem(run: object) -> str | None:
    """The file name (without `.ocp.json`) for run id `run`, or None when the id cannot be stored:
    empty, a first character that is not an ASCII letter or digit, a control or unassigned
    character, or a name over 200 characters once escaped."""
    if not isinstance(run, str) or not run or run[0] not in FIRST:
        return None
    if any(unicodedata.category(c)[0] == "C" for c in run):
        return None
    stem = "".join(c if c in SAFE else "".join(f"%{b:02X}" for b in c.encode("utf-8")) for c in run)
    return stem if len(stem) <= MAX_NAME else None


def run_id(stem: str) -> str:
    """The run id a file name stem stands for (the inverse of `file_stem`)."""
    return unquote(stem, errors="replace")


def file_name(run: object) -> str:
    """`<stem>.ocp.json` for readers that build a run file path themselves (`Store.run_path` raises
    instead). An id that cannot be stored gives a name the store never writes, so the read finds nothing."""
    stem = file_stem(run)
    return f"{stem}.ocp.json" if stem is not None else ".not-a-run-id.ocp.json"
