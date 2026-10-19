"""The run log `loopmath.harness-runs/0.1`: one JSON object per rollout, one line each.

The spelling is spec/harness-runs.md (and spec/harness-runs-0.1.schema.json). Only `schema` and
`task_id` are required. Unknown fields are kept. A reader refuses a whole file on its first bad line.
"""

from __future__ import annotations

import contextlib
import difflib
import errno
import fcntl
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterator

from ..types import TASK_TYPE_IDS

SCHEMA = "loopmath.harness-runs/0.1"
STREAMS = ("input", "cache_read", "cache_write", "output")
KNOWN = ("schema", "id", "task_id", "task", "arm", "pair", "experiment", "status", "harness", "agent", "model",
         "effort", "outcome", "cost", "started_at", "ended_at", "tool", "ext")
STATUSES = ("done", "failed")
SCALES = ("linear", "log", "fraction")
BETTER = ("higher", "lower")
TOOL_KEYS = ("name", "version", "run_id", "task_id", "stage")
TASK_KEYS = ("type", "repo", "title", "base_commit", "source")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _opt_str(row: dict, key: str, problems: list[str], prefix: str = "") -> None:
    if key in row and row[key] is not None and not isinstance(row[key], str):
        problems.append(f"{prefix}{key}: a string or null, got {type(row[key]).__name__}")


def _check_outcome(out: Any, problems: list[str]) -> None:
    if out is None:
        return
    if not isinstance(out, dict):
        problems.append("outcome: an object or null")
        return
    has_score, has_verdict = "score" in out, "accepted" in out
    if has_score == has_verdict:
        problems.append("outcome: exactly one of score or accepted")
        return
    if has_score:
        if out["score"] is not None and not _number(out["score"]):
            problems.append("outcome.score: a number or null")
        for key in ("name", "judge"):
            if key in out and not (isinstance(out[key], str) and out[key]):
                problems.append(f"outcome.{key}: a non-empty string")
        if "scale" in out and out["scale"] not in SCALES:
            problems.append(f"outcome.scale: one of {', '.join(SCALES)}, got {out['scale']!r}")
        elif out.get("scale") == "fraction" and _number(out["score"]) and not 0 <= out["score"] <= 1:
            problems.append(f"outcome.score: a fraction lies in [0, 1], got {out['score']!r}")
        if "better" in out and out["better"] not in BETTER:
            problems.append(f"outcome.better: one of {', '.join(BETTER)}, got {out['better']!r}")
    else:
        if not isinstance(out["accepted"], bool):
            problems.append("outcome.accepted: true or false")
        if "by" in out and not isinstance(out["by"], str):
            problems.append("outcome.by: a string")
    if "observed_at" in out and not (isinstance(out["observed_at"], str) and TIMESTAMP.match(out["observed_at"])):
        problems.append("outcome.observed_at: an RFC 3339 time with an offset")


def _check_cost(cost: Any, problems: list[str]) -> None:
    if cost is None:
        return
    if not isinstance(cost, dict):
        problems.append("cost: an object or null")
        return
    if cost.get("usd") is not None and not (_number(cost["usd"]) and cost["usd"] >= 0):
        problems.append("cost.usd: a number >= 0 or null")
    tokens = cost.get("tokens")
    if tokens is not None:
        if not isinstance(tokens, dict):
            problems.append("cost.tokens: an object {input, cache_read, cache_write, output} or null")
        else:
            for key, value in tokens.items():
                if key not in STREAMS:
                    problems.append(f"cost.tokens.{key}: not a stream (known: {', '.join(STREAMS)})")
                elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    problems.append(f"cost.tokens.{key}: an integer >= 0")
    for key in ("recovered", "complete"):
        if key in cost and not isinstance(cost[key], bool):
            problems.append(f"cost.{key}: true or false")
    if cost.get("complete") is False and cost.get("usd") is not None:
        problems.append("cost.usd: must be null when cost.complete is false (a partial sum goes in cost.partial_usd)")
    if cost.get("partial_usd") is not None and not (_number(cost["partial_usd"]) and cost["partial_usd"] >= 0):
        problems.append("cost.partial_usd: a number >= 0 or null")
    for key in ("usd_source", "unpriced", "incomplete"):
        _opt_str(cost, key, problems, "cost.")
    for key in ("files", "skipped_lines"):
        if key in cost and (isinstance(cost[key], bool) or not isinstance(cost[key], int) or cost[key] < 0):
            problems.append(f"cost.{key}: an integer >= 0")


def _check_harness(value: Any, problems: list[str]) -> None:
    if value is None or (isinstance(value, str) and value):
        return
    if isinstance(value, dict):
        schema = value.get("schema")
        if not (isinstance(schema, str) and schema.startswith("loopmath.harness/")):
            problems.append("harness: a record needs schema loopmath.harness/0.x")
        if not (isinstance(value.get("digest"), str) and value["digest"]):
            problems.append("harness: a record needs a digest")
        if not isinstance(value.get("factors"), list):
            problems.append("harness: a record needs a factors list")
        return
    problems.append("harness: a loopmath.harness record, a tree path, or null")


def task_type_problem(value: str) -> str | None:
    """Why `value` is not a loopmath task type, naming the types and the closest one; None when it is one."""
    if value in TASK_TYPE_IDS:
        return None
    close = difflib.get_close_matches(value.strip().lower().replace("-", "_").replace(" ", "_"), TASK_TYPE_IDS, n=1)
    return (f"{value!r} is not a loopmath task type (one of {', '.join(TASK_TYPE_IDS)})"
            + (f"; did you mean {close[0]!r}?" if close else ""))


def experiment_problem(value: str) -> str | None:
    """Why `value` cannot be an experiment (it is a path, spec/harness-runs.md section 2); None when it can."""
    if "/" in value or "\\" in value or value.startswith("~") or value in (".", ".."):
        return f"{value!r} is a path; give a stable name that is not one, so moving a folder keeps it"
    return None


def check_row(row: Any) -> list[str]:
    """Every problem with one row, each naming its field; empty when the row is valid."""
    if not isinstance(row, dict):
        return ["row: a JSON object"]
    problems: list[str] = []
    schema = row.get("schema")
    if schema is None:
        problems.append(f"schema: missing (this reader reads {SCHEMA})")
        return problems
    if schema != SCHEMA:
        problems.append(f"schema: {schema!r} is not supported (this reader reads {SCHEMA})")
        return problems
    if not (isinstance(row.get("task_id"), str) and row["task_id"].strip()):
        problems.append("task_id: a non-empty string is required")
    for key in ("id", "arm", "pair", "experiment", "agent", "model", "effort"):
        _opt_str(row, key, problems)
    if isinstance(row.get("experiment"), str) and (why := experiment_problem(row["experiment"])):
        problems.append(f"experiment: {why}")
    if row.get("status") is not None and row["status"] not in STATUSES:
        problems.append(f"status: one of {', '.join(STATUSES)}, got {row['status']!r}")
    task = row.get("task")
    if task is not None:
        if not isinstance(task, dict):
            problems.append("task: an object")
        else:
            for key in TASK_KEYS:
                _opt_str(task, key, problems, "task.")
            if isinstance(task.get("type"), str) and (why := task_type_problem(task["type"])):
                problems.append(f"task.type: {why}")
    _check_harness(row.get("harness"), problems)
    _check_outcome(row.get("outcome"), problems)
    _check_cost(row.get("cost"), problems)
    for key in ("started_at", "ended_at"):
        value = row.get(key)
        if value is not None and not (isinstance(value, str) and TIMESTAMP.match(value)):
            problems.append(f"{key}: an RFC 3339 time with an offset, or null")
    tool = row.get("tool")
    if tool is not None:
        if not isinstance(tool, dict):
            problems.append("tool: an object")
        else:
            for key in TOOL_KEYS:
                _opt_str(tool, key, problems, "tool.")
    if "ext" in row and not isinstance(row["ext"], dict):
        problems.append("ext: an object")
    return problems


def row(task_id: str, **fields: Any) -> dict[str, Any]:
    """A checked row with `schema` set. Raises ValueError naming every bad field."""
    out = {"schema": SCHEMA, "task_id": task_id, **fields}
    problems = check_row(out)
    if problems:
        raise ValueError("; ".join(problems))
    return out


def row_id(r: dict[str, Any]) -> str:
    """The row's stable id: its `id`, else the tool's name, the experiment and the tool's run id, else a hash
    of the row. Native run ids are unique only within one experiment, so the experiment is part of the id."""
    if isinstance(r.get("id"), str) and r["id"]:
        return r["id"]
    tool = r.get("tool") if isinstance(r.get("tool"), dict) else {}
    if tool.get("name") and tool.get("run_id"):
        if r.get("experiment"):
            return f"{tool['name']}:{r['experiment']}:{tool['run_id']}"
        return f"{tool['name']}:{tool['run_id']}"
    canon = json.dumps({k: v for k, v in r.items() if k != "id"}, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


# The log's (device, inode) -> the descriptor this thread holds its lock on. flock is per open file, so a
# second os.open of the same log in this thread would wait on itself; appends under `locked` reuse the held one.
_local = threading.local()


def _held() -> dict[tuple[int, int], int]:
    h = getattr(_local, "held", None)
    if h is None:
        h = _local.held = {}
    return h


def _line(r: dict[str, Any]) -> bytes:
    problems = check_row(r)
    if problems:
        raise ValueError("; ".join(problems))
    return (json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def _write(fd: int, path: Path, line: bytes) -> None:
    """One checked os.write of a whole line, then fsync, on a descriptor whose lock the caller holds."""
    start = os.fstat(fd).st_size
    if start and os.pread(fd, 1, start - 1) != b"\n":
        raise ValueError(f"{path}: its last line has no newline (a torn write); not appending after it")
    done = 0
    try:
        done = os.write(fd, line)
        if done != len(line):
            raise OSError(errno.EIO, f"wrote {done} of {len(line)} bytes of a row, then cut it back", str(path))
        os.fsync(fd)
    except OSError:
        if done:
            with contextlib.suppress(OSError):   # if the cut fails, the torn tail stops the next append
                if os.fstat(fd).st_size == start + done:   # re-checked under the lock: only this line is cut
                    os.ftruncate(fd, start)
        raise


class Locked:
    """The handle `locked` yields: the log's path and an `append` that writes under the lock it holds."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self._fd = fd

    def append(self, r: dict[str, Any]) -> None:
        _write(self._fd, self.path, _line(r))

    def read(self) -> list[dict[str, Any]]:
        return read(self.path)


@contextlib.contextmanager
def locked(path: str | Path) -> Iterator[Locked]:
    """Hold the log's exclusive lock (an fcntl.flock, waiting for it) and yield a handle whose `append` writes
    under it, so a caller can read, check and append with no other append in between. Every `append` takes
    the same lock; inside `locked` in the same thread, `append(path, row)` and a nested `locked` reuse it."""
    path = Path(path)
    fd = os.open(str(path), os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        st = os.fstat(fd)
        key, held = (st.st_dev, st.st_ino), _held()
        if key in held:
            yield Locked(path, held[key])
            return
        fcntl.flock(fd, fcntl.LOCK_EX)
        held[key] = fd
        try:
            yield Locked(path, fd)
        finally:
            del held[key]
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def append(path: str | Path, r: dict[str, Any]) -> None:
    """Append one checked row as one line, under the log's lock: a single os.write on an O_APPEND descriptor,
    checked, then fsync.

    One write per line keeps concurrent appends from interleaving. A write that stops short (or an fsync that
    fails) is cut back to the file's old end and raised as OSError, so no torn line is left behind, and a file
    whose last line already has no newline is refused rather than joined to. The lock covers the write, the
    fsync and the cut, so a cut never takes another writer's row."""
    line = _line(r)
    with locked(path) as log:
        _write(log._fd, log.path, line)


def read(path: str | Path) -> list[dict[str, Any]]:
    """Every row of a file, checked. Raises ValueError('<file>:<line>: ...') on the first bad line.

    Lines are split on newline only: U+2028, U+2029 and U+0085 stay inside a row's strings, as append wrote them."""
    return [row for _, row in read_numbered(path)]


def read_numbered(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    """`read`, with each row's line number, for messages that name `<file>:<line>`."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: cannot read: {exc}") from None
    if text.startswith("\ufeff"):
        raise ValueError(f"{path}:1: the file starts with a byte order mark; save it as UTF-8 without one")
    rows: list[tuple[int, dict[str, Any]]] = []
    for n, line in enumerate(text.split("\n"), 1):
        if not line.strip(" \t\r"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{n}: not JSON ({exc.msg})") from None
        problems = check_row(value)
        if problems:
            raise ValueError(f"{path}:{n}: " + "; ".join(problems))
        rows.append((n, value))
    return rows
