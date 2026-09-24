"""Read Atrium's hook-fed SQLite timeline as OCP v0.2.

Atrium records explicit pane/session and source/target-pane message pairs plus
typed subagent-stop events whose subagent id may be empty. The stored facts are
``verified``, but runtime pane ids are not stable OCP task-node ids and the
closed OCP edge vocabulary has no neutral message or runtime-correlation edge.
This adapter therefore preserves them as typed metadata-only events under the
documented ``dev.dagr.adapter.atrium`` extension and emits no core graph edge. It
does not infer a spawn relation from a subagent-stop record or infer acceptance
from segment closure, session end, or subagent stop.

The source has no token counts, cost, model, effort, stable task node, or
acceptance result. Its locally inspected data is stale since 2026-08-14 and
its pane-to-session coverage is sparse. SQL projection excludes prompt,
message, title, body, transcript, actor, and task text before rows reach
Python; none is retained or emitted.
"""

from __future__ import annotations

import hashlib
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


_DEFAULT_ROOT = Path.home() / ".atrium"
_STORE_NAMES = ("tasks.db", "timeline.db")
_NAMESPACE = "dev.dagr.adapter.atrium"
_TIMELINE_KINDS = ("agent-message", "session-end", "subagent-stop")
_DIAGNOSTIC_REASONS = (
    "timeline_kind_unsupported",
    "segment_adapter_session_id_missing",
    "segment_adapter_session_id_invalid_type",
    "filter_session",
    "filter_workspace",
    "filter_before_since",
    "filter_after_until",
    "filter_limit",
)


@dataclass(frozen=True)
class _Relation:
    store: Path
    workspace_id: str
    at: str
    event_type: str
    pane_id: str
    related_kind: str | None
    related_id: str | None


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
    print(f"atrium diagnostics: {rendered}", file=sys.stderr)


def _connect(path: Path) -> sqlite3.Connection:
    """Open an Atrium SQLite store without granting write access."""

    uri = path.expanduser().resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


@contextmanager
def _read_connection(path: Path) -> Iterator[sqlite3.Connection]:
    try:
        with closing(_connect(path)) as connection:
            yield connection
    except (OSError, sqlite3.Error) as exc:
        raise ValueError(
            f"Atrium selected store {path} is not a readable SQLite database: {exc}"
        ) from None


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Atrium relation has no stable string {field}")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _timestamp(value: Any, field: str) -> tuple[str, datetime]:
    raw = _string(value, field).strip()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"Atrium {field} must be an RFC 3339 timestamp") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat().replace("+00:00", "Z"), utc


def _bound(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    try:
        return _timestamp(value, name)[1]
    except ValueError:
        raise ValueError(f"Atrium selection {name} must be an RFC 3339 timestamp") from None


def _store_paths(
    stores: Iterable[Path], *, require_selected: bool = False
) -> tuple[Path, ...]:
    paths: dict[str, Path] = {}
    for raw_store in stores:
        store = raw_store.expanduser()
        if store.is_dir():
            candidates = tuple(store / name for name in _STORE_NAMES)
            usable = tuple(candidate for candidate in candidates if candidate.is_file())
            if require_selected and not usable:
                raise ValueError(
                    f"Atrium selected store directory {store} contains no usable "
                    "SQLite database; expected tasks.db or timeline.db. SQL-script "
                    "fixture directories must be materialized with fixture_selection()."
                )
        else:
            candidates = (store,)
            usable = tuple(candidate for candidate in candidates if candidate.is_file())
            if require_selected and not usable:
                raise ValueError(
                    f"Atrium selected store {store} does not exist or is not a "
                    "regular file"
                )
        for candidate in usable:
            resolved = candidate.resolve()
            paths[str(resolved)] = resolved
    return tuple(paths[key] for key in sorted(paths))


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row["name"]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    }


def _timeline_relations(
    connection: sqlite3.Connection, store: Path
) -> list[_Relation]:
    rows = connection.execute(
        """
        SELECT workspace_id, kind, pane_id, created_at,
               CASE kind
                 WHEN 'agent-message' THEN
                   CASE WHEN json_valid(metadata_json)
                        THEN json_extract(metadata_json, '$.fromPaneId') END
                 ELSE pane_id
               END AS source_pane_id,
               CASE kind
                 WHEN 'agent-message' THEN 'target-pane'
                 WHEN 'session-end' THEN 'session'
                 WHEN 'subagent-stop' THEN 'subagent'
               END AS related_kind,
               CASE kind
                 WHEN 'agent-message' THEN
                   CASE WHEN json_valid(metadata_json)
                        THEN json_extract(metadata_json, '$.targetPaneId') END
                 WHEN 'session-end' THEN
                   CASE WHEN json_valid(metadata_json)
                        THEN json_extract(metadata_json, '$.sessionId') END
                 WHEN 'subagent-stop' THEN
                   CASE WHEN json_valid(metadata_json)
                        THEN json_extract(metadata_json, '$.subagent') END
               END AS related_id
        FROM timeline
        WHERE kind IN (?, ?, ?)
        ORDER BY created_at, kind, workspace_id, source_pane_id, related_id
        """,
        _TIMELINE_KINDS,
    ).fetchall()
    relations: list[_Relation] = []
    for row in rows:
        at = _timestamp(row["created_at"], "timeline.created_at")[0]
        relations.append(
            _Relation(
                store=store,
                workspace_id=_string(row["workspace_id"], "workspace_id"),
                at=at,
                event_type=_string(row["kind"], "kind"),
                pane_id=_string(row["source_pane_id"], "pane_id"),
                related_kind=(
                    _string(row["related_kind"], "related_kind")
                    if _optional_string(row["related_id"]) is not None
                    else None
                ),
                related_id=_optional_string(row["related_id"]),
            )
        )
    return relations


def _segment_relations(
    connection: sqlite3.Connection, store: Path
) -> list[_Relation]:
    rows = connection.execute(
        """
        SELECT workspace_id, pane_id, adapter_session_id, started_at
        FROM task_run_segments
        WHERE typeof(adapter_session_id) = 'text'
          AND trim(adapter_session_id) != ''
        ORDER BY started_at, workspace_id, pane_id, adapter_session_id
        """
    ).fetchall()
    relations: list[_Relation] = []
    for row in rows:
        at = _timestamp(row["started_at"], "task_run_segments.started_at")[0]
        relations.append(
            _Relation(
                store=store,
                workspace_id=_string(row["workspace_id"], "workspace_id"),
                at=at,
                event_type="task-run-segment",
                pane_id=_string(row["pane_id"], "pane_id"),
                related_kind="session",
                related_id=_string(row["adapter_session_id"], "adapter_session_id"),
            )
        )
    return relations


def _relations(
    stores: Iterable[Path], *, require_selected: bool = False
) -> tuple[tuple[_Relation, ...], dict[str, int]]:
    relations: list[_Relation] = []
    diagnostics = _empty_diagnostics()
    for store in _store_paths(stores, require_selected=require_selected):
        with _read_connection(store) as connection:
            tables = _tables(connection)
            recognized = False
            if "timeline" in tables:
                placeholders = ", ".join("?" for _ in _TIMELINE_KINDS)
                row = connection.execute(
                    f"""
                    SELECT COUNT(*) AS n
                    FROM timeline
                    WHERE kind IS NULL
                       OR typeof(kind) != 'text'
                       OR kind NOT IN ({placeholders})
                    """,
                    _TIMELINE_KINDS,
                ).fetchone()
                diagnostics["timeline_kind_unsupported"] += int(row["n"])
                relations.extend(_timeline_relations(connection, store))
                recognized = True
            if "task_run_segments" in tables:
                row = connection.execute(
                    """
                    SELECT
                      SUM(CASE
                            WHEN adapter_session_id IS NULL
                              OR (typeof(adapter_session_id) = 'text'
                                  AND trim(adapter_session_id) = '')
                            THEN 1 ELSE 0
                          END) AS missing,
                      SUM(CASE
                            WHEN adapter_session_id IS NOT NULL
                             AND typeof(adapter_session_id) != 'text'
                            THEN 1 ELSE 0
                          END) AS invalid_type
                    FROM task_run_segments
                    """
                ).fetchone()
                diagnostics["segment_adapter_session_id_missing"] += int(
                    row["missing"] or 0
                )
                diagnostics["segment_adapter_session_id_invalid_type"] += int(
                    row["invalid_type"] or 0
                )
                relations.extend(_segment_relations(connection, store))
                recognized = True
            if not recognized:
                raise ValueError(
                    f"Atrium selected store {store} has no supported relation table; "
                    "expected timeline or task_run_segments"
                )
    relations.sort(
        key=lambda row: (
            row.at,
            row.event_type,
            row.workspace_id,
            row.pane_id,
            row.related_kind or "",
            row.related_id or "",
            str(row.store),
        )
    )
    return tuple(relations), diagnostics


def _selected_relations(
    relations: tuple[_Relation, ...], selection: Selection
) -> tuple[tuple[_Relation, ...], dict[str, int]]:
    since = _bound(selection.since, "since")
    until = _bound(selection.until, "until")
    wanted_sessions = set(selection.session_ids)
    wanted_workspaces = set(selection.workspaces)
    diagnostics = _empty_diagnostics()
    eligible_session_workspaces: set[str] | None = None
    if wanted_sessions:
        eligible_session_workspaces = {
            row.workspace_id
            for row in relations
            if row.workspace_id in wanted_sessions
            or (row.related_kind == "session" and row.related_id in wanted_sessions)
        }

    selected: list[_Relation] = []
    for row in relations:
        parsed_at = _timestamp(row.at, "event timestamp")[1]
        if (
            eligible_session_workspaces is not None
            and row.workspace_id not in eligible_session_workspaces
        ):
            diagnostics["filter_session"] += 1
            continue
        if wanted_workspaces and row.workspace_id not in wanted_workspaces:
            diagnostics["filter_workspace"] += 1
            continue
        if since is not None and parsed_at < since:
            diagnostics["filter_before_since"] += 1
            continue
        if until is not None and parsed_at > until:
            diagnostics["filter_after_until"] += 1
            continue
        selected.append(row)
    if selection.limit is not None:
        scope_order = sorted(
            {row.workspace_id for row in selected},
            key=lambda workspace_id: (
                min(row.at for row in selected if row.workspace_id == workspace_id),
                workspace_id,
            ),
        )
        allowed = set(scope_order[: selection.limit])
        diagnostics["filter_limit"] += sum(
            row.workspace_id not in allowed for row in selected
        )
        selected = [row for row in selected if row.workspace_id in allowed]
    return tuple(selected), diagnostics


def _document_run_id(relations: tuple[_Relation, ...]) -> str:
    if not relations:
        return "atrium-empty-selection"
    material = "\n".join(
        "\0".join(
            (
                row.workspace_id,
                row.at,
                row.event_type,
                row.pane_id,
                row.related_kind or "",
                row.related_id or "",
            )
        )
        for row in relations
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"atrium-selection-{digest}"


@register
class AtriumAdapter(Adapter):
    """Adapter for Atrium's stale, metadata-only local SQLite stores."""

    name = "atrium"

    def discover(self) -> tuple[Path, ...]:
        return _store_paths((_DEFAULT_ROOT,))

    def fixture_selection(
        self, fixture_dir: Path, /
    ) -> AbstractContextManager[Selection]:
        source = fixture_dir.expanduser().resolve()

        @contextmanager
        def materialized() -> Iterator[Selection]:
            with tempfile.TemporaryDirectory(prefix="loopmath-atrium-fixture-") as temp:
                root = Path(temp)
                stores: list[Path] = []
                for name in ("tasks", "timeline"):
                    store = root / f"{name}.db"
                    script = source / f"{name}.sql"
                    with closing(sqlite3.connect(store)) as connection:
                        connection.executescript(script.read_text(encoding="utf-8"))
                        connection.commit()
                    store.chmod(0o444)
                    stores.append(store)
                yield Selection(stores=tuple(stores))

        return materialized()

    def sessions(self) -> tuple[Mapping[str, Any], ...]:
        relations, _diagnostics = _relations(self.discover())
        sessions: list[Mapping[str, Any]] = []
        for workspace_id in sorted({row.workspace_id for row in relations}):
            scoped = [row for row in relations if row.workspace_id == workspace_id]
            sessions.append(
                {
                    "id": workspace_id,
                    "stores": tuple(sorted({str(row.store) for row in scoped})),
                    "workspace": workspace_id,
                    "created_at": min(row.at for row in scoped),
                    "updated_at": max(row.at for row in scoped),
                }
            )
        sessions.sort(key=lambda item: (item["created_at"], item["id"]))
        return tuple(sessions)

    def emit(self, selection: Selection) -> dict[str, Any]:
        stores = selection.stores or tuple(self.discover())
        source_relations, diagnostics = _relations(
            stores, require_selected=bool(selection.stores)
        )
        relations, filter_diagnostics = _selected_relations(source_relations, selection)
        for reason in _DIAGNOSTIC_REASONS:
            diagnostics[reason] += filter_diagnostics[reason]
        _print_diagnostics(diagnostics)
        events = []
        for row in relations:
            extension = {
                "workspace_id": row.workspace_id,
                "pane_id": row.pane_id,
                "tier": "verified",
            }
            if row.related_id is not None and row.related_kind is not None:
                extension["related_kind"] = row.related_kind
                extension["related_id"] = row.related_id
            events.append(
                {
                    "at": row.at,
                    "type": row.event_type,
                    "ext": {_NAMESPACE: extension},
                }
            )
        return {
            "ocp": "0.2",
            "producer": {
                "name": "dagr-adapter-atrium",
                "framework": "atrium",
                "source_contract": "atrium/timeline+task-run-segments",
                "capabilities": {
                    "groups": False, "events": True, "artifacts": False, "edges_dep": False,
                    "edges_spawn": False, "edges_launch": False, "edges_artifact": False,
                    "cost_usd": False, "cost_tokens": False, "outcome_evidence": False,
                },
            },
            "privacy": {
                "profile": "metadata_only",
                "note": "Prompt, message, title, body, transcript, actor, and task text omitted.",
            },
            "run": {"id": _document_run_id(relations)},
            "nodes": [],
            "edges": [],
            "attempts": [],
            "events": events,
            "ext": {
                _NAMESPACE: {"diagnostics": _diagnostic_records(diagnostics)}
            },
        }
