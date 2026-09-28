"""Output conventions shared by every 0.1.0 command (design/0.1/02-commands.md, section 1).

- `--json`: exactly one object on stdout, `schema` as its first key.
- `--html [PATH]`: without PATH, `$LOOPMATH_HOME/views/<command>-<YYYYmmdd-HHMMSS>.html`.
- Exit codes below.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
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


def fmt_usd(value: Any) -> str:
    """Dollars for a terminal line. An exact zero is `$0`; a cost above zero and under a cent keeps one
    significant digit (`$0.002`, never `$0.00`); from a cent up, `$1,234.56`. None or a non-number is `n/a`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return "n/a"
    sign, v = ("-", -float(value)) if value < 0 else ("", float(value))
    if v == 0:
        return "$0"
    if v < 0.01:
        digits = max(2, -math.floor(math.log10(v)))
        small = round(v, digits)
        if small >= 10.0 ** (1 - digits):  # 0.00096 rounds up to $0.001, still one digit
            digits -= 1
        if small < 0.01:
            return f"{sign}${small:.{digits}f}"
        v = small  # 0.0096 rounds up to a cent
    return f"{sign}${v:,.2f}"


def _as_datetime(value: Any) -> _dt.datetime | None:
    if isinstance(value, _dt.datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            dt = _dt.datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError:
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.astimezone()  # a naive time is local time


def fmt_time(value: Any, *, seconds: bool = False) -> str:
    """A stored time for a terminal line: local time with its zone, `2026-09-27 19:56 PDT`.

    Stored and JSON times stay ISO strings; this is for people only. A value that does not read as a
    time is returned as given; None is `n/a`."""
    dt = _as_datetime(value)
    if dt is None:
        return "n/a" if value is None or value == "" else str(value)
    local = dt.astimezone()
    zone = local.strftime("%Z") or local.strftime("%z")
    return local.strftime("%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M") + (f" {zone}" if zone else "")


def not_implemented(lane: int, command: str) -> int:
    print(f"loopmath {command}: not implemented yet: lane {lane:02d}", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


def fail(message: str, code: int = EXIT_USER) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code
