"""fcntl locking and atomic writes (design/0.1/03-interfaces.md, section 4).

Writers take `store_lock` around every read-modify-write of a store file and
every index append; readers never lock. Every file is replaced whole: written
to a temp file in the same folder, fsynced, renamed over the target, and the
folder fsynced, so a reader sees the old file or the new one, never a torn one.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator

LOCK_TIMEOUT_S = 30.0
_POLL_S = 0.05


class StoreLocked(RuntimeError):
    """The store lock was held by another writer for longer than the timeout (exit code 4)."""


# Lock path -> nesting depth for this thread, so a Store method that calls
# another under the lock does not wait on itself. flock is per open file, so a
# second os.open in the same process would block.
_local = threading.local()


def _depths() -> dict[str, int]:
    d = getattr(_local, "depths", None)
    if d is None:
        d = _local.depths = {}
    return d


@contextlib.contextmanager
def store_lock(home: Path, timeout: float | None = None) -> Iterator[None]:
    """Exclusive lock on `$HOME/lock`, waiting up to `timeout` seconds (default 30)."""
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    path = str((home / "lock").resolve())
    depths = _depths()
    if depths.get(path):
        depths[path] += 1
        try:
            yield
        finally:
            depths[path] -= 1
        return
    limit = LOCK_TIMEOUT_S if timeout is None else timeout
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        deadline = time.monotonic() + limit
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise
                if time.monotonic() >= deadline:
                    raise StoreLocked(f"the store at {home} stayed locked for more than {limit:g} s") from None
                time.sleep(_POLL_S)
        depths[path] = 1
        try:
            yield
        finally:
            depths.pop(path, None)
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextlib.contextmanager
def try_lock(path: Path) -> Iterator[bool]:
    """Non-blocking exclusive lock on `path`; yields False when another holder has it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def is_locked(path: Path) -> bool:
    """True when another holder has an flock on `path` (False when the file does not exist)."""
    if not Path(path).exists():
        return False
    with try_lock(path) as got:
        return not got


def _fsync_dir(folder: Path) -> None:
    try:
        fd = os.open(str(folder), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Temp file in the same folder, fsync, rename over `path`, fsync the folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path: Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False, default=_default) + "\n"


def atomic_write_json(path: Path, obj: Any) -> Path:
    return atomic_write_text(path, dump_json(obj))


def append_line(path: Path, obj: Any) -> None:
    """Append one JSON line with a single write and fsync. The caller holds the store lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_default) + "\n"
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: Path) -> list[dict]:
    """Every complete JSON object line; a torn or unreadable line is skipped."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")
