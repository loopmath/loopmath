"""What the OpenEvolve and ShinkaEvolve readers share: one harness-runs row per LLM call whose model was recorded
(an OpenEvolve child, a ShinkaEvolve patch query), with one `mutator` stage, and the outcome "this call wrote a child
that beat its parent".

The rows use the run log's `stages` (spec/harness-runs.md): the role is `mutator`, the piece `mutate`, the agent
the tool's name and the model the one that wrote the child. A verdict outcome says whether the child's score beat
its parent's by more than `min_delta`; `delta`, `parent_score` and `child_score` ride along as numbers. Nothing
the tool wrote as text (code, diffs, prompts, replies, feedback, logs) is copied: ext holds ids, numbers and
loopmath's own labels.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

ROLE = "mutator"
PIECE = "mutate"
EXPERIMENT_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class FormatError(ValueError):
    """An input the reader does not recognise, named with what it found."""


class NeedsFlag(ValueError):
    """The input is readable, but a flag must say what it does not: the message names the flag."""


@dataclass(frozen=True)
class TablePrice:
    """A price table the caller loaded, as a reader uses it: `usd(model, tokens)` gives the dollars for one
    model's tokens (keys `in`, `cache_read`, `cache_write` and `out`), or None when the table has no row for the
    model; `path` and `as_of` name the table in the notes. The readers never load or price a table themselves."""

    usd: Callable[[str, dict[str, int]], float | None]
    path: str
    as_of: str


@dataclass
class Result:
    rows: list[dict[str, Any]]
    format: str
    skipped: list[dict[str, str]] = field(default_factory=list)   # [{run_id, reason}]
    counts: dict[str, int] = field(default_factory=dict)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    accounting: dict[str, float] = field(default_factory=dict)   # dollars: total, attributed, unattributed


def number(value: Any) -> float | None:
    """A finite float, or None for anything else (bools are flags, not scores)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def iso(epoch: Any) -> str | None:
    """Epoch seconds as RFC 3339 UTC, or None."""
    value = number(epoch)
    if value is None or value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def experiment_id(prefix: str, root_id: str) -> str:
    """A stable experiment id from the run's first program: `<prefix>-` and 16 hex."""
    return f"{prefix}-" + hashlib.sha256(root_id.encode("utf-8")).hexdigest()[:16]


def check_label(label: str | None) -> None:
    if label is not None and not EXPERIMENT_LABEL.match(label):
        raise NeedsFlag(f"--experiment {label!r}: letters, digits, '.', '_' or '-', at most 64")


def improved(child: float | None, parent: float | None, min_delta: float) -> tuple[bool, float | None]:
    """Whether the child beat its parent by more than `min_delta`, and the delta (None when either is unknown)."""
    if child is None or parent is None:
        return False, None
    delta = child - parent
    return delta > min_delta, delta


def mutator_row(*, tool: str, ns: str, experiment: str, child_id: str, task_id: str, task: dict[str, Any],
                model: str, effort: str | None, accepted: bool, delta: float | None, parent_score: float | None,
                child_score: float | None, metric: str | None, cost: dict[str, Any] | None, status: str,
                started: str | None, ended: str | None, stage_started: str | None, stage_ended: str | None,
                stage: str, note: dict[str, Any]) -> dict[str, Any]:
    """One harness-runs row (spec/harness-runs.md) for one child: a single mutator stage."""
    from ..harness_runs.rows import SCHEMA

    outcome: dict[str, Any] = {"accepted": accepted, "by": tool}
    if metric:
        outcome["metric"] = metric
    for key, value in (("delta", delta), ("parent_score", parent_score), ("child_score", child_score)):
        if value is not None:
            outcome[key] = value
    st: dict[str, Any] = {"role": ROLE, "piece": PIECE, "status": status}
    if cost:
        st["cost"] = cost
    if stage_started:
        st["started_at"] = stage_started
    if stage_ended:
        st["ended_at"] = stage_ended
    row: dict[str, Any] = {"schema": SCHEMA, "id": f"{tool}:{experiment}:{child_id}", "task_id": task_id,
                           "task": task, "experiment": experiment, "status": status, "agent": tool, "model": model,
                           "outcome": outcome, "stages": [st],
                           "tool": {"name": tool, "run_id": child_id, "task_id": task_id, "stage": stage},
                           "ext": {ns: {k: v for k, v in note.items() if v is not None}}}
    if effort:
        row["effort"] = effort
    if started:
        row["started_at"] = started
    if ended:
        row["ended_at"] = ended
    return row


def task_of(task_id: str, task_type: str | None, repo: str | None) -> dict[str, Any]:
    return {"type": task_type or "research", "repo": repo or task_id}


def experiments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["experiment"]] = counts.get(r["experiment"], 0) + 1
    return [{"experiment": e, "runs": n} for e, n in sorted(counts.items())]


def plural(n: int, word: str, many: str | None = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"
