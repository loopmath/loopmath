"""Output conventions shared by every 0.1.0 command (design/0.1/02-commands.md, section 1).

- `--json`: exactly one object on stdout, `schema` as its first key.
- `--html [PATH]`: without PATH, `$LOOPMATH_HOME/views/<command>-<YYYYmmdd-HHMMSS>.html`.
- Exit codes below.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_USER = 1
EXIT_NOT_FOUND = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_LOCKED = 4
EXIT_NO_FIT = 5

HTML_DEFAULT = "__default__"  # argparse const for a bare --html


def home(override: str | None = None) -> Path:
    """The store root: --home, else LOOPMATH_HOME, else ~/.loopmath."""
    if override:
        return Path(override).expanduser()
    env = os.environ.get("LOOPMATH_HOME")
    return Path(env).expanduser() if env else Path.home() / ".loopmath"


def emit_json(schema: str, payload: dict[str, Any], stream=None) -> None:
    obj = {"schema": schema}
    obj.update({k: v for k, v in payload.items() if k != "schema"})
    (stream or sys.stdout).write(json.dumps(obj, indent=1, ensure_ascii=False, default=_default) + "\n")


def _default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def html_path(command: str, arg: str | None, home_override: str | None = None) -> Path | None:
    """Resolve the --html argument. None when --html was not given."""
    if arg is None:
        return None
    if arg != HTML_DEFAULT:
        return Path(arg).expanduser()
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder, n = home(home_override) / "views", 1
    path = folder / f"{command}-{stamp}.html"
    while path.exists():  # a second page in the same second must not replace the first
        n += 1
        path = folder / f"{command}-{stamp}-{n}.html"
    return path


def write_html(path: Path, page: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(page, encoding="utf-8")
    os.replace(tmp, path)
    print(str(path))
    return path


def not_implemented(lane: int, command: str) -> int:
    print(f"loopmath {command}: not implemented yet: lane {lane:02d}", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def fail(message: str, code: int = EXIT_USER) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code
