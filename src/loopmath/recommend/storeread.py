"""What the recommender reads from the store, and the one file it writes (`recs/<rec>.json`).

Readers never take the store lock (spec 03 section 4). Config is read from
`config.toml`, run history from `runs/index.jsonl`, and configurations by id from
earlier recommendations and stored run documents. Lane 7 owns the store; this
module only reads its documented layout, and writes `recs/` atomically as spec 05
section 6 asks of `recommend`.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from ..types import Configuration

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
HISTORY_DAYS = 90


def ulid() -> str:
    """26-character ULID: 48-bit millisecond time and 80 random bits, Crockford base32."""
    value = (int(time.time() * 1000) << 80) | secrets.randbits(80)
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def now_local() -> _dt.datetime:
    return _dt.datetime.now().astimezone()


def iso(ts: _dt.datetime) -> str:
    return ts.isoformat(timespec="seconds")


def parse_ts(text: str | None) -> _dt.datetime | None:
    if not text:
        return None
    try:
        ts = _dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.astimezone()


# ---------------------------------------------------------------- config
class Conf:
    """Dotted-key access over `config.toml` ("rescue.kind", "usual.feature")."""

    def __init__(self, data: dict[str, Any] | None = None):
        self.data = data or {}

    @classmethod
    def load(cls, home: Path) -> "Conf":
        path = home / "config.toml"
        if not path.exists():
            return cls({})
        with path.open("rb") as fh:
            return cls(tomllib.load(fh))

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.data:
            return self.data[key]
        node: Any = self.data
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def float(self, key: str) -> float | None:
        v = self.get(key)
        if v in (None, ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"config {key} must be a number; got {v!r}") from None


# ---------------------------------------------------------------- history
def index_rows(home: Path) -> Iterator[dict[str, Any]]:
    path = home / "runs" / "index.jsonl"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


# Designed runs (loopmath-exp slates) are the planner's choice, not the user's habit: they feed the fit but not
# the usual or the default usual's models, with no fallback to them.
NOT_HABIT_SOURCES = frozenset({"designed"})


def run_rows(home: Path) -> list[dict[str, Any]]:
    """One index row per run, in the order of each run's last row.

    The store appends a row per state change and the last row per run wins
    (lane 7, `store/home.py`), so counting raw rows would count a run once per
    change. Rows without a run id are kept as they are.
    """
    last: dict[Any, tuple[int, dict[str, Any]]] = {}
    for i, row in enumerate(index_rows(home)):
        last[row.get("run") or ("", i)] = (i, row)
    return [row for _, row in sorted(last.values(), key=lambda x: x[0])]


def own_runs(home: Path | None) -> int:
    """The user's recorded runs (N3, 0.2.3): runs in the store whose last index row is finished, the ones a fit
    learns costs from. Shipped prior runs live in the bundle, not the store, so they never count."""
    if home is None:
        return 0
    return sum(1 for row in run_rows(Path(home)) if row.get("run") and row.get("state") == "finished")


def usual_from_history(home: Path, task_type: str, repo: str, *, now: _dt.datetime | None = None,
                       days: int = HISTORY_DAYS) -> tuple[str | None, str | None]:
    """(config id, level) seen in the most runs for (type, repo) in the last `days`, else for the type.

    Runs without a start time count; later runs win ties; designed runs do not count.
    """
    now = now or now_local()
    since = now - _dt.timedelta(days=days)
    by_repo: Counter = Counter()
    by_type: Counter = Counter()
    last: dict[str, int] = {}
    for i, row in enumerate(run_rows(home)):
        cfg = row.get("config")
        if not cfg or row.get("task_type") != task_type or row.get("source") in NOT_HABIT_SOURCES:
            continue
        ts = parse_ts(row.get("started_at"))
        if ts is not None and ts < since:
            continue
        by_type[cfg] += 1
        if row.get("repo") == repo:
            by_repo[cfg] += 1
        last[cfg] = i
    for counts, level in ((by_repo, "repo"), (by_type, "type")):
        if counts:
            best = max(counts, key=lambda c: (counts[c], last.get(c, -1)))
            return best, level
    return None, None


def usual_from_config(conf: Conf, task_type: str, repo: str) -> str | None:
    """`[usual.<type>]` keyed by repo, then `"*"`; values are configuration ids."""
    table = conf.get(f"usual.{task_type}")
    if isinstance(table, str):
        return table
    if isinstance(table, dict):
        for key in (repo, "*"):
            v = table.get(key)
            if isinstance(v, str) and v:
                return v
    return None


def model_habits(home: Path, limit: int = 500) -> list[tuple[str, str, str]]:
    """(harness, model, effort) settings of recent runs, most used first, from stored run documents."""
    counts: Counter = Counter()
    rows = [r for r in run_rows(home) if r.get("source") not in NOT_HABIT_SOURCES][-limit:]
    for row in rows:
        doc = read_json(home / "runs" / f"{row.get('run')}.ocp.json")
        conf = (doc or {}).get("run", {}).get("configuration") if isinstance(doc, dict) else None
        cfg = config_from_any(conf) if conf else None
        if cfg is None:
            continue
        for s in cfg.settings.values():
            counts[(s.harness, s.model, s.effort)] += 1
    return [k for k, _ in counts.most_common()]


RECORDED_LIMIT = 60


def recorded_configs(home: Path, task_type: str, repo: str, *,
                     limit: int = RECORDED_LIMIT) -> tuple[list[tuple[Configuration, int]], str | None]:
    """([(configuration, runs)], level): the configurations the store's runs used for (type, repo), else for
    the type, most runs first, then the most recent (spec 05 section 1).

    Every source counts, designed included: a recorded workflow is a candidate even when it is not the user's
    habit (D74 keeps it out of the usual only). One run document is read per configuration.
    """
    by_repo: Counter = Counter()
    by_type: Counter = Counter()
    last: dict[str, int] = {}
    run_of: dict[str, str] = {}
    for i, row in enumerate(run_rows(home)):
        cfg = row.get("config")
        if not cfg or row.get("task_type") != task_type:
            continue
        by_type[cfg] += 1
        if row.get("repo") == repo:
            by_repo[cfg] += 1
        last[cfg] = i
        if row.get("run"):
            run_of[cfg] = str(row["run"])
    for counts, level in ((by_repo, "repo"), (by_type, "type")):
        if not counts:
            continue
        out = []
        for cfg_id in sorted(counts, key=lambda c: (-counts[c], -last.get(c, -1))):
            doc = read_json(home / "runs" / f"{run_of.get(cfg_id)}.ocp.json") if cfg_id in run_of else None
            conf = doc.get("run", {}).get("configuration") if isinstance(doc, dict) else None
            cfg = config_from_any(conf) if conf else None
            if cfg is not None:
                out.append((cfg, counts[cfg_id]))
            if len(out) >= limit:
                break
        if out:
            return out, level
    return [], None


# ---------------------------------------------------------------- configurations by id
def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _walk_configs(obj: Any) -> Iterator[dict]:
    if isinstance(obj, dict):
        if isinstance(obj.get("id"), str) and obj["id"].startswith("cfg_") and "workflow" in obj and "settings" in obj:
            yield obj
        for v in obj.values():
            yield from _walk_configs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_configs(v)


def find_config(home: Path, cfg_id: str, *, max_recs: int = 200) -> Configuration | None:
    """A configuration by id, from recent recommendations, then stored run documents."""
    recs = home / "recs"
    if recs.is_dir():
        files = sorted(recs.glob("rec_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_recs]
        for path in files:
            for d in _walk_configs(read_json(path)):
                if d["id"] == cfg_id:
                    cfg = config_from_any(d)
                    if cfg is not None:
                        return cfg
    for row in index_rows(home):
        if row.get("config") != cfg_id:
            continue
        doc = read_json(home / "runs" / f"{row.get('run')}.ocp.json")
        conf = doc.get("run", {}).get("configuration") if isinstance(doc, dict) else None
        cfg = config_from_any(conf) if conf else None
        if cfg is not None:
            return cfg
    return None


def config_from_any(d: dict | None) -> Configuration | None:
    """A `Configuration` from its `to_dict()` form or from OCP v0.3 `run.configuration`; None when unreadable.

    Lane 4's `workflows.ocp.configuration_from_any` decodes it: nothing behavioral is
    dropped (gate rules, artifact kinds, rescue objects, unknown fields) and the id is
    recomputed from the content. A different stored id is kept in `extra["declared_id"]`,
    so a configuration's id always names what it runs.
    """
    if not isinstance(d, dict) or "settings" not in d or "workflow" not in d:
        return None
    from ..workflows.ocp import configuration_from_any

    try:
        c = configuration_from_any(d)
    except (KeyError, TypeError, ValueError, LookupError):
        return None
    declared = d.get("id")
    if declared and declared != c.id and "declared_id" not in c.extra:
        # lane 4 keeps it for OCP objects only; the types form follows the same rule here
        c = Configuration(c.id, c.workflow, c.settings, {**c.extra, "declared_id": declared})
    return c


# ---------------------------------------------------------------- recs
def atomic_write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_default)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _default(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def save_rec(home: Path, rec_id: str, payload: dict[str, Any]) -> Path:
    return atomic_write_json(home / "recs" / f"{rec_id}.json", payload)


def load_rec(home: Path, rec_id: str) -> dict[str, Any] | None:
    obj = read_json(home / "recs" / f"{rec_id}.json")
    return obj if isinstance(obj, dict) else None
