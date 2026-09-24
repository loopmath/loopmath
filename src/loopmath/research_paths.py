"""Where the research verbs find their data: config values, not hardcoded paths.

`loopmath research fit|transfer-test` read the sweep run records, and
`loopmath analyze-e0` reads the E0 corpus. Both folders live outside the
package. Each is resolved in this order:

1. the command's flag (`--sweep-dir`, `--corpus`);
2. the environment (`LOOPMATH_SWEEP_DIR`, `LOOPMATH_E0_CORPUS`);
3. the store's config (`research.sweep_dir`, `research.e0_corpus` in
   `$LOOPMATH_HOME/config.toml`, set with `loopmath config set`).

With none of them, the caller gets `ResearchPathError` naming all three.
The config file is only read here, never written.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from .output import home

SWEEP = ("sweep_dir", "LOOPMATH_SWEEP_DIR", "--sweep-dir", "sweep run records")
E0_CORPUS = ("e0_corpus", "LOOPMATH_E0_CORPUS", "--corpus", "E0 corpus")


class ResearchPathError(ValueError):
    """No folder was given for a research verb's data."""


def _config_value(key: str, store: Path) -> str | None:
    path = store / "config.toml"
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = (data.get("research") or {}).get(key)
    return value if isinstance(value, str) and value else None


def resolve(which: tuple[str, str, str, str], explicit: str | Path | None = None,
            store: Path | None = None) -> Path:
    key, env_name, flag, what = which
    if explicit is not None:
        return Path(explicit).expanduser()
    env_value = os.environ.get(env_name)
    if env_value:
        return Path(env_value).expanduser()
    store = home() if store is None else store
    configured = _config_value(key, store)
    if configured:
        return Path(configured).expanduser()
    raise ResearchPathError(
        f"no folder for the {what}: pass {flag} PATH, set {env_name}, "
        f"or run `loopmath config set research.{key} PATH`"
    )


def sweep_dir(explicit: str | Path | None = None, store: Path | None = None) -> Path:
    return resolve(SWEEP, explicit, store)


def e0_corpus(explicit: str | Path | None = None, store: Path | None = None) -> Path:
    return resolve(E0_CORPUS, explicit, store)
