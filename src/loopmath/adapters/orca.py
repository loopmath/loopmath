"""Read Orca's local orchestration database as OCP v0.2.

Evidence is deliberately narrow. A ``spawn`` edge is ``verified`` because its
endpoints come directly from ``tasks.parent_id``. The effective agent, model,
and effort in ``worker_dispatches.start_options`` are ``reported`` launch
configuration. The adapter preserves only the typed ``escalation``,
``worker_done``, ``merge_ready``, and ``decision_gate`` messages as events;
their enum values are verified store records, while their success/readiness
meaning remains a report by Orca or its worker.

Worker state and record timestamps are also persisted facts, but the OCP status
and outcome they produce are reported projections rather than acceptance
evidence. In particular, uncertain start/stop states remain ``working``.

The dispatch-to-vendor-session join remains open. Neither the read-only SQLite
schema nor the inspected profile, hook, and CLI context surfaces provided a
stable historical join. Consequently attempts omit ``session`` and ``origin``,
and all relations stay keyed by Orca's native run, task, and dispatch ids. No
heuristic relation is emitted, and prompt, task, result, and message text is
never read into the OCP document.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .base import Adapter, Selection
from .registry import register


_DEFAULT_STORE = Path.home() / "Library" / "Application Support" / "orca" / "orchestration.db"
_NAMESPACE = "dev.dagr.adapter.orca"
_MESSAGE_TYPES = ("decision_gate", "escalation", "merge_ready", "worker_done")
_DIAGNOSTIC_REASONS = (
    "start_options_missing",
    "start_options_invalid_type",
    "start_options_invalid_json",
    "start_options_json_not_object",
    "message_payload_missing",
    "message_payload_invalid_type",
    "message_payload_invalid_json",
    "message_payload_json_not_object",
)
_ATTEMPT_STATUS = {
    "starting": "working",
    "ready": "working",
    "start_unknown": "working",
    "failed": "failed",
    "succeeded": "done",
    "stopping": "working",
    "stop_unknown": "working",
    "stopped": "canceled",
    "abandoned": "settled_unverified",
}
_TERMINAL = {"done", "failed", "rejected", "canceled", "settled_unverified", "lost"}


@dataclass(frozen=True)
class _Run:
    store: Path
    id: str
    created_at: str | None
    updated_at: str | None
    workspaces: tuple[str, ...]


def _empty_diagnostics() -> dict[str, int]:
    return {reason: 0 for reason in _DIAGNOSTIC_REASONS}


def _diagnostic_records(counts: Mapping[str, int]) -> list[dict[str, Any]]:
    return [
        {"reason": reason, "count": counts[reason]}
        for reason in _DIAGNOSTIC_REASONS
    ]


def _print_diagnostics(counts: Mapping[str, int]) -> None:
    rendered = ", ".join(
        f"{reason}={counts[reason]}" for reason in _DIAGNOSTIC_REASONS
    )
    print(f"orca diagnostics: {rendered}", file=sys.stderr)


def _connect(path: Path) -> sqlite3.Connection:
    """Open one SQLite store without granting the connection write access."""

    uri = path.expanduser().resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _resolve_store(path: Path) -> Path:
    store = path.expanduser()
    if store.is_dir():
        raise ValueError(
            f"Orca selected store {store} is a directory; expected an "
            "orchestration.db SQLite file. SQL-script fixture directories must be "
            "materialized with fixture_selection()."
        )
    if not store.is_file():
        raise ValueError(
            f"Orca selected store {store} does not exist or is not a regular file"
        )
    return store.resolve()


@contextmanager
def _read_connection(
    path: Path,
) -> Iterator[tuple[Path, sqlite3.Connection]]:
    store = _resolve_store(path)
    try:
        with closing(_connect(store)) as connection:
            yield store, connection
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(
            f"Orca selected store {store} is not a readable orchestration SQLite "
            f"database: {exc}"
        ) from None


def _timestamp(value: Any) -> tuple[str, datetime] | None:
    """Normalize Orca's SQLite UTC timestamps to RFC 3339."""

    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat().replace("+00:00", "Z"), utc


def _bound(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    parsed = _timestamp(value)
    if parsed is None:
        raise ValueError(f"Orca selection {name} must be an RFC 3339 timestamp")
    return parsed[1]


def _json_object(
    value: Any, field: str, diagnostics: dict[str, int]
) -> dict[str, Any]:
    if value is None or (isinstance(value, str) and not value.strip()):
        diagnostics[f"{field}_missing"] += 1
        return {}
    if not isinstance(value, str):
        diagnostics[f"{field}_invalid_type"] += 1
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        diagnostics[f"{field}_invalid_json"] += 1
        return {}
    if not isinstance(parsed, dict):
        diagnostics[f"{field}_json_not_object"] += 1
        return {}
    return parsed


def _effective_launch(
    start_options: Any, diagnostics: dict[str, int]
) -> tuple[str | None, str | None, str | None]:
    options = _json_object(start_options, "start_options", diagnostics)
    launch = options.get("launch")
    launch = launch if isinstance(launch, dict) else {}
    effective = launch.get("effective")
    effective = effective if isinstance(effective, dict) else {}

    def field(name: str) -> str | None:
        value = effective.get(name, options.get(name))
        return value if isinstance(value, str) and value else None

    return field("agent"), field("model"), field("effort")


def _run_rows(store: Path) -> tuple[_Run, ...]:
    with _read_connection(store) as (resolved_store, connection):
        workspaces: dict[str, set[str]] = {}
        for row in connection.execute(
            """
            SELECT DISTINCT t.run_id, w.worktree_id
            FROM tasks AS t
            JOIN dispatch_contexts AS d ON d.task_id = t.id AND d.run_id = t.run_id
            JOIN worker_dispatches AS w ON w.dispatch_id = d.id
            WHERE w.worktree_id IS NOT NULL
            """
        ):
            workspaces.setdefault(row["run_id"], set()).add(row["worktree_id"])
        rows = connection.execute(
            "SELECT id, created_at, updated_at FROM runs ORDER BY created_at, id"
        ).fetchall()
    runs = []
    for row in rows:
        created = _timestamp(row["created_at"])
        updated = _timestamp(row["updated_at"])
        runs.append(
            _Run(
                store=resolved_store,
                id=row["id"],
                created_at=created[0] if created else None,
                updated_at=updated[0] if updated else None,
                workspaces=tuple(sorted(workspaces.get(row["id"], ()))),
            )
        )
    return tuple(runs)


def _selected_runs(stores: Iterable[Path], selection: Selection) -> tuple[_Run, ...]:
    since = _bound(selection.since, "since")
    until = _bound(selection.until, "until")
    wanted_sessions = set(selection.session_ids)
    wanted_workspaces = set(selection.workspaces)
    selected: list[_Run] = []
    for store in stores:
        for run in _run_rows(store):
            created = _timestamp(run.created_at)
            if wanted_sessions and run.id not in wanted_sessions:
                continue
            if wanted_workspaces and wanted_workspaces.isdisjoint(run.workspaces):
                continue
            if since is not None and (created is None or created[1] < since):
                continue
            if until is not None and (created is None or created[1] > until):
                continue
            selected.append(run)
    selected.sort(key=lambda run: (run.created_at or "", run.id, str(run.store)))
    if selection.limit is not None:
        selected = selected[: selection.limit]
    return tuple(selected)


def _document_run_id(runs: tuple[_Run, ...]) -> str:
    if len(runs) == 1:
        return runs[0].id
    if not runs:
        return "orca-empty-selection"
    material = "\n".join(run.id for run in runs)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"orca-selection-{digest}"


def _unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate Orca {label} id {value!r} across selected stores")
        seen.add(value)


@register
class OrcaAdapter(Adapter):
    """Adapter for Orca's undocumented, read-only SQLite orchestration store."""

    name = "orca"

    def discover(self) -> tuple[Path, ...]:
        return (_DEFAULT_STORE,) if _DEFAULT_STORE.is_file() else ()

    def fixture_selection(
        self, fixture_dir: Path, /
    ) -> AbstractContextManager[Selection]:
        source = fixture_dir.expanduser().resolve() / "orchestration.sql"

        @contextmanager
        def materialized() -> Iterator[Selection]:
            with tempfile.TemporaryDirectory(prefix="loopmath-orca-fixture-") as temp:
                store = Path(temp) / "orchestration.db"
                with closing(sqlite3.connect(store)) as connection:
                    connection.executescript(source.read_text(encoding="utf-8"))
                    connection.commit()
                store.chmod(0o444)
                yield Selection(stores=(store,))

        return materialized()

    def sessions(self) -> tuple[Mapping[str, Any], ...]:
        sessions = [
            {
                "id": run.id,
                "store": str(run.store),
                "created_at": run.created_at,
                "updated_at": run.updated_at,
                "workspaces": run.workspaces,
            }
            for store in self.discover()
            for run in _run_rows(store)
        ]
        sessions.sort(key=lambda item: (item["created_at"] or "", item["id"], item["store"]))
        return tuple(sessions)

    def emit(self, selection: Selection) -> dict[str, Any]:
        stores = selection.stores or tuple(self.discover())
        runs = _selected_runs(stores, selection)
        _unique((run.id for run in runs), "run")

        groups: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        diagnostics = _empty_diagnostics()
        node_ids: set[str] = set()
        attempt_nodes: dict[str, str] = {}

        for run in runs:
            groups.append({"id": run.id})
            with _read_connection(run.store) as (_resolved_store, connection):
                task_rows = connection.execute(
                    """
                    SELECT id, parent_id
                    FROM tasks
                    WHERE run_id = ?
                    ORDER BY created_at, id
                    """,
                    (run.id,),
                ).fetchall()
                local_task_ids = {row["id"] for row in task_rows}
                for row in task_rows:
                    node_id = row["id"]
                    if node_id in node_ids:
                        raise ValueError(
                            f"duplicate Orca task id {node_id!r} across selected stores"
                        )
                    node_ids.add(node_id)
                    nodes.append({"id": node_id, "kind": "task", "group": run.id})
                for row in task_rows:
                    parent_id = row["parent_id"]
                    if parent_id is None:
                        continue
                    if parent_id not in local_task_ids:
                        raise ValueError(
                            f"Orca task {row['id']!r} names unknown parent {parent_id!r}"
                        )
                    edges.append(
                        {
                            "from": parent_id,
                            "to": row["id"],
                            "kind": "spawn",
                            "tier": "verified",
                            "evidence": "Orca tasks.parent_id",
                        }
                    )

                dispatch_rows = connection.execute(
                    """
                    SELECT w.dispatch_id, d.task_id, w.state, w.start_options,
                           w.created_at, w.updated_at
                    FROM worker_dispatches AS w
                    JOIN dispatch_contexts AS d ON d.id = w.dispatch_id
                    JOIN tasks AS t ON t.id = d.task_id
                    WHERE d.run_id = ? AND t.run_id = ?
                    ORDER BY d.task_id, w.created_at, w.dispatch_id
                    """,
                    (run.id, run.id),
                ).fetchall()
                ordinals: dict[str, int] = {}
                for row in dispatch_rows:
                    attempt_id = row["dispatch_id"]
                    if attempt_id in attempt_nodes:
                        raise ValueError(
                            f"duplicate Orca dispatch id {attempt_id!r} across selected stores"
                        )
                    task_id = row["task_id"]
                    if task_id not in local_task_ids:
                        raise ValueError(
                            f"Orca dispatch {attempt_id!r} names unknown task {task_id!r}"
                        )
                    ordinals[task_id] = ordinals.get(task_id, 0) + 1
                    try:
                        status = _ATTEMPT_STATUS[row["state"]]
                    except KeyError:
                        raise ValueError(
                            f"unknown Orca worker_dispatches.state {row['state']!r}"
                        ) from None
                    attempt: dict[str, Any] = {
                        "id": attempt_id,
                        "node": task_id,
                        "n": ordinals[task_id],
                        "status": status,
                    }
                    agent, model, effort = _effective_launch(
                        row["start_options"], diagnostics
                    )
                    if agent is not None:
                        attempt["harness"] = agent
                    if model is not None:
                        attempt["model"] = {"raw": model, "tier": "reported"}
                    if effort is not None:
                        attempt["effort"] = effort
                    started = _timestamp(row["created_at"])
                    if started is not None:
                        attempt["started_at"] = started[0]
                    ended = _timestamp(row["updated_at"])
                    if status in _TERMINAL:
                        if ended is not None:
                            attempt["ended_at"] = ended[0]
                        attempt["outcome"] = {
                            "result": status,
                            "evidence": "reported",
                            "receipt": f"Orca worker_dispatches.state={row['state']!r}",
                        }
                    attempts.append(attempt)
                    attempt_nodes[attempt_id] = task_id

                placeholders = ", ".join("?" for _ in _MESSAGE_TYPES)
                message_rows = connection.execute(
                    f"""
                    SELECT id, type, payload, sequence, created_at
                    FROM messages
                    WHERE run_id = ? AND type IN ({placeholders})
                    ORDER BY created_at, sequence, id
                    """,
                    (run.id, *_MESSAGE_TYPES),
                ).fetchall()
                for row in message_rows:
                    at = _timestamp(row["created_at"])
                    if at is None:
                        raise ValueError(
                            f"Orca message {row['id']!r} has no valid created_at timestamp"
                        )
                    event: dict[str, Any] = {"at": at[0], "type": row["type"]}
                    payload = _json_object(
                        row["payload"], "message_payload", diagnostics
                    )
                    dispatch_id = payload.get("dispatchId")
                    task_id = payload.get("taskId")
                    if isinstance(dispatch_id, str) and dispatch_id in attempt_nodes:
                        event["attempt"] = dispatch_id
                        event["node"] = attempt_nodes[dispatch_id]
                    elif isinstance(task_id, str) and task_id in local_task_ids:
                        event["node"] = task_id
                    events.append(event)

        edges.sort(key=lambda edge: (edge["from"], edge["to"], edge["kind"]))
        attempts.sort(key=lambda attempt: (attempt["node"], attempt["n"], attempt["id"]))
        events.sort(key=lambda event: (event["at"], event["type"], event.get("attempt", "")))
        _print_diagnostics(diagnostics)
        return {
            "ocp": "0.2",
            "producer": {
                "name": "dagr-adapter-orca",
                "framework": "orca",
                "source_contract": "orca/orchestration.db",
                "capabilities": {
                    "groups": True, "events": True, "artifacts": False, "edges_dep": False,
                    "edges_spawn": True, "edges_launch": False, "edges_artifact": False,
                    "cost_usd": False, "cost_tokens": False, "outcome_evidence": True,
                },
            },
            "privacy": {
                "profile": "metadata_only",
                "note": "Task, result, and message text omitted.",
            },
            "run": {"id": _document_run_id(runs)},
            "groups": groups,
            "nodes": nodes,
            "edges": edges,
            "attempts": attempts,
            "events": events,
            "ext": {
                _NAMESPACE: {"diagnostics": _diagnostic_records(diagnostics)}
            },
        }
