"""The daily update check (0.3.5): is a newer loopmath on PyPI?

`cli.main` calls `after_command` once a command has finished. At most once a day it asks
pypi.org for loopmath's latest version: one GET of `URL` with `loopmath/<version>` as the
User-Agent, nothing else sent, `TIMEOUT_S` to answer. What it heard goes to `update.json` in the
store's cache folder (`ingest.cache_dir()`, so LOOPMATH_CACHE_DIR wins): `checked_at`, `latest`
(the last version pypi.org gave) and `error` (why the last ask failed, or null). An answer is
reused for 24 h and a failure for 6 h, so an offline machine waits out the timeout at most four
times a day. When a newer version is known and stderr is a terminal, one line on stderr says so,
with the upgrade command. Nothing here writes to stdout, and no error here changes a command's
output or exit code.

loopmath never upgrades itself: `loopmath update` and `status --json` (`update`) give the command,
and the user, or their agent once the user agrees, runs it. The check is off when
LOOPMATH_NO_UPDATE_CHECK is set to anything but empty, or when CI is set; `loopmath update`, which
the user asks for by name, still asks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import __version__

URL = "https://pypi.org/pypi/loopmath/json"
TIMEOUT_S = 2.0
OK_TTL_S = 24 * 3600
FAIL_TTL_S = 6 * 3600
CACHE_NAME = "update.json"
MAX_BYTES = 8 << 20  # the JSON lists every release's files; far below this
CHANGES_URL = "https://github.com/loopmath/loopmath/blob/v{version}/CHANGELOG.md"
THEN = "loopmath skill install"
SCHEMA = "loopmath.update/1"

# PEP 440, as the `packaging` library spells it (loopmath does not depend on packaging).
_PEP440 = re.compile(r"""
    v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>[-_.]?(?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)[-_.]?(?P<pre_n>[0-9]+)?)?
    (?P<post>(?:-(?P<post_n1>[0-9]+))|(?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n2>[0-9]+)?))?
    (?P<dev>[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
""", re.VERBOSE | re.IGNORECASE)
_PRE_RANK = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "c": 2, "rc": 2, "pre": 2, "preview": 2}


def version_key(text: Any) -> tuple | None:
    """A sort key in PEP 440 order (a local label is ignored), or None for a string that is not a version."""
    m = _PEP440.fullmatch(text.strip()) if isinstance(text, str) else None
    if m is None:
        return None
    release = tuple(int(part) for part in m["release"].split("."))
    while len(release) > 1 and release[-1] == 0:
        release = release[:-1]
    if m["pre_l"]:
        pre: tuple = (1, _PRE_RANK[m["pre_l"].lower()], int(m["pre_n"] or 0))
    else:
        pre = (0,) if m["dev"] and not m["post"] else (2,)  # 1.0.dev1 sorts before 1.0a1
    post = (1, int(m["post_n1"] or m["post_n2"] or 0)) if m["post"] else (0,)
    dev = (0, int(m["dev_n"] or 0)) if m["dev"] else (1,)
    return int(m["epoch"] or 0), release, pre, post, dev


def is_prerelease(text: str) -> bool:
    m = _PEP440.fullmatch(text.strip())
    return bool(m and (m["pre_l"] or m["dev"]))


def newer(latest: Any, installed: str = __version__) -> bool:
    """True when `latest` is a later version than `installed`. A pre-release counts only when
    `installed` is one too."""
    a, b = version_key(latest), version_key(installed)
    if a is None or b is None:
        return False
    if is_prerelease(latest) and not is_prerelease(installed):
        return False
    return a > b


def enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return not env.get("LOOPMATH_NO_UPDATE_CHECK") and "CI" not in env


def cache_path() -> Path:
    from .ingest.base import cache_dir

    return cache_dir() / CACHE_NAME


def fetch(timeout: float = TIMEOUT_S) -> str:
    """The latest version on pypi.org (`info.version`). Raises on any failure."""
    import urllib.request

    request = urllib.request.Request(URL, headers={"User-Agent": f"loopmath/{__version__}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read(MAX_BYTES))
    version = data.get("info", {}).get("version") if isinstance(data, dict) else None
    if version_key(version) is None:
        raise ValueError(f"pypi.org gave no version loopmath can read: {version!r}")
    return version


def _ask(timeout: float) -> tuple[str | None, str | None]:
    """(latest, None) or (None, why), within `timeout` in all: urlopen's own timeout does not
    cover the name lookup, so the request runs in a thread that is left behind if it is late."""
    box: dict[str, str] = {}

    def run() -> None:
        try:
            box["latest"] = fetch(timeout)
        except BaseException as exc:  # noqa: BLE001 - every failure is a failed check
            box["error"] = (f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__)[:300]

    worker = threading.Thread(target=run, name="loopmath-update-check", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return None, f"no answer from pypi.org within {timeout:g} s"
    return box.get("latest"), box.get("error")


def _iso(at: float) -> str:
    return datetime.fromtimestamp(int(at), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _instant(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp() if isinstance(value, str) else None
    except ValueError:
        return None


def read_cache(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or _instant(data.get("checked_at")) is None:
        return None
    latest = data.get("latest")
    return {"checked_at": data["checked_at"], "latest": latest if version_key(latest) else None,
            "error": data.get("error") if isinstance(data.get("error"), str) else None}


def fresh(state: dict, now: float) -> bool:
    """An answer is good for 24 h and a failure for 6 h; a time ahead of the clock is not trusted."""
    age = now - (_instant(state.get("checked_at")) or float("-inf"))
    return 0 <= age < (FAIL_TTL_S if state.get("error") else OK_TTL_S)


def _write(path: Path, state: dict) -> None:
    """Temp file beside `path`, then rename over it."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def check(*, refresh: bool = False, now: float | None = None, path: Path | None = None) -> dict:
    """What loopmath knows of the latest version: `checked_at`, `latest`, `error`, and `cached` (true
    when read from `update.json` without asking). Asks pypi.org only when the cache is stale or
    `refresh` is set, and never when the cache folder cannot be written: an unwritable folder would
    otherwise mean a request on every command."""
    now = time.time() if now is None else now
    path = cache_path() if path is None else path
    old = read_cache(path)
    if old is not None and not refresh and fresh(old, now):
        return {**old, "cached": True}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(path.parent, os.W_OK)
    except OSError:
        writable = False
    if not writable:
        return {**(old or {"checked_at": None, "latest": None}), "error": f"cannot write {path}", "cached": False}
    latest, error = _ask(TIMEOUT_S)
    state = {"checked_at": _iso(now), "latest": latest or (old or {}).get("latest"), "error": error}
    try:
        _write(path, state)
    except OSError:
        pass  # the answer stands; the next command asks again
    return {**state, "cached": False}


def installer(prefix: str | None = None) -> str:
    """How this loopmath was installed: `uv` (uv tool), `pipx`, or `pip` (anything else)."""
    root = Path(sys.prefix if prefix is None else prefix)
    if (root / "uv-receipt.toml").is_file():
        return "uv"
    if (root / "pipx_metadata.json").is_file():
        return "pipx"
    return "pip"


def upgrade_command(kind: str | None = None) -> str:
    kind = installer() if kind is None else kind
    if kind == "uv":
        return "uv tool upgrade loopmath"
    if kind == "pipx":
        return "pipx upgrade loopmath"
    return f"{shlex.quote(sys.executable)} -m pip install -U loopmath"


def details(state: dict, kind: str | None = None) -> dict:
    """`state` for people and agents: the versions, whether an upgrade is available, and its commands."""
    kind = installer() if kind is None else kind
    upgrade = upgrade_command(kind)
    latest = state.get("latest")
    return {"latest": latest, "installed": __version__, "available": newer(latest),
            "command": f"{upgrade} && {THEN}", "checked_at": state.get("checked_at"),
            "installer": kind, "upgrade": upgrade, "then": THEN,
            "changes": CHANGES_URL.format(version=latest) if latest else None}


def status_object() -> dict | None:
    """`status --json`'s `update`: null when the check is off or pypi.org has never answered."""
    if not enabled():
        return None
    try:
        info = details(check())
    except Exception:  # noqa: BLE001 - the check never fails a command
        return None
    if info["latest"] is None:
        return None
    return {k: info[k] for k in ("latest", "installed", "available", "command", "checked_at")}


def notice(info: dict) -> str | None:
    if not info.get("available"):
        return None
    return f"loopmath {info['latest']} is out (you have {info['installed']}): {info['upgrade']}, then {THEN}"


def _command_word(argv: list[str]) -> str | None:
    return next((a for a in argv if not a.startswith("-")), None)


def after_command(argv: list[str], stream=None) -> None:
    """After a command (`cli.main`): the daily check, then one line on stderr when a newer
    loopmath is out and stderr is a terminal. Silent on any error, Ctrl-C included."""
    try:
        if not enabled() or _command_word(argv) == "update":  # `update` has just said it all
            return
        line = notice(details(check()))
        stream = sys.stderr if stream is None else stream
        if line and stream.isatty():
            print(line, file=stream)
    except (Exception, KeyboardInterrupt):  # noqa: BLE001
        pass


def command(args: argparse.Namespace) -> int:
    """`loopmath update [--refresh] [--json]`: the installed and latest versions and the upgrade
    commands. It never upgrades. Exit 0 when the check worked (up to date, or a newer one shown),
    1 when it failed."""
    from .output import emit_json, fmt_time

    state = check(refresh=args.refresh)
    info = details(state)
    error = state.get("error") or (None if info["latest"] else "pypi.org gave no version")
    if args.json:
        emit_json(SCHEMA, {"ok": error is None, **info, "cached": state["cached"], "error": error,
                           **({"exit": 1} if error else {})})
        return 1 if error else 0
    if error:
        when = f" (at {fmt_time(state['checked_at'])}; `loopmath update --refresh` asks again)" if state["cached"] else ""
        print(f"error: could not get the latest version from pypi.org: {error}{when}", file=sys.stderr)
        return 1
    checked = f"pypi.org, checked {fmt_time(state['checked_at'])}"
    if not info["available"]:
        if version_key(__version__) > version_key(info["latest"]):  # a build from source, or a pre-release
            print(f"loopmath {__version__} is newer than the latest on pypi.org, {info['latest']} ({checked})")
        else:
            print(f"loopmath {__version__} is the latest ({checked})")
        return 0
    print(f"loopmath {info['latest']} is out (you have {__version__}; {checked})")
    print(f"what changed: {info['changes']}")
    print(f"upgrade: {info['upgrade']}")
    print(f"then: {THEN}")
    print("loopmath never upgrades itself; run these yourself, or have your agent run them")
    return 0
