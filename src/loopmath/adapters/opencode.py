"""OpenCode session exports to OCP v0.2.

One OpenCode session is one OCP node and one attempt.  The adapter sums the
five billable token streams from assistant-message usage and takes USD from
the session-level ``cost`` field.  ``session.parentID`` is an explicit source
relation: when both endpoints are selected it becomes a ``spawn`` edge at the
``verified`` evidence tier.  No inferred relation is emitted.

The default store is inventoried read-only through ``opencode.db`` and each
selected session is read through the supported ``opencode export <sessionID>``
boundary.  Explicit JSON export files or directories of exports are supported
for portable input and tests.  An explicitly selected non-default SQLite file
is read directly as a compatibility fallback.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any, Mapping, Sequence

from .base import Adapter, Selection
from . import opencode_store as _opencode_store
from .opencode_store import (
    _TOKEN_FIELDS,
    _SessionSource,
    _assistant_infos,
    _as_mapping,
    _connect_read_only,
    _cost_record,
    _database_sessions,
    _decode_export,
    _default_database,
    _entity_id,
    _export_session,
    _format_timestamp,
    _json_object,
    _messages_from_database,
    _message_models,
    _nested_nonnegative_int,
    _nonnegative_number,
    _parse_bound,
    _read_export,
    _short,
    _store_sessions,
    _timestamp,
)
from .registry import register


class _OpenCodeModule(_ModuleType):
    """Keep moved helpers responsive to rebinding through this facade."""

    _support = _opencode_store
    _support_rebindings = frozenset(
        {
            "Any",
            "Mapping",
            "Path",
            "Sequence",
            "_TOKEN_FIELDS",
            "_SessionSource",
            "_as_mapping",
            "_connect_read_only",
            "_nested_nonnegative_int",
            "_nonnegative_number",
            "_database_sessions",
            "_decode_export",
            "_json_object",
            "_read_export",
            "_timestamp",
            "closing",
            "dataclass",
            "datetime",
            "defaultdict",
            "json",
            "math",
            "sha256",
            "sqlite3",
            "subprocess",
            "timezone",
        }
    )

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        module_type = type(self)
        if name in module_type._support_rebindings:
            setattr(module_type._support, name, value)


_sys.modules[__name__].__class__ = _OpenCodeModule
del _ModuleType, _OpenCodeModule, _opencode_store, _sys


_EXT = "dev.dagr.adapter.opencode"

def _session_created(session: _SessionSource) -> datetime | None:
    return _timestamp(_as_mapping(session.info.get("time")).get("created"))


def _selected_sessions(
    sessions: Sequence[_SessionSource], selection: Selection
) -> tuple[list[_SessionSource], dict[str, int]]:
    since = _parse_bound(selection.since, "--since")
    until = _parse_bound(selection.until, "--until")
    wanted_ids = set(selection.session_ids)
    wanted_workspaces = set(selection.workspaces)
    counters: defaultdict[str, int] = defaultdict(int)
    selected = []

    for session in sessions:
        info = session.info
        session_id = info.get("id")
        if wanted_ids and session_id not in wanted_ids:
            counters["selection_excluded_by_session_id"] += 1
            continue
        workspace = info.get("directory")
        if wanted_workspaces:
            if not isinstance(workspace, str) or not workspace:
                counters["selection_unknown_workspace"] += 1
            elif workspace not in wanted_workspaces:
                counters["selection_excluded_by_workspace"] += 1
                continue
        created = _session_created(session)
        if since is not None or until is not None:
            if created is None:
                counters["selection_unknown_created_at"] += 1
            elif since is not None and created < since:
                counters["selection_excluded_by_time"] += 1
                continue
            elif until is not None and created > until:
                counters["selection_excluded_by_time"] += 1
                continue
        selected.append(session)

    selected.sort(
        key=lambda item: (
            -(_session_created(item).timestamp() if _session_created(item) else float("-inf")),
            str(item.info["id"]),
        )
    )
    if selection.limit is not None:
        if selection.limit < 1:
            raise ValueError("OpenCode selection limit must be at least 1")
        counters["selection_excluded_by_limit"] += max(
            0, len(selected) - selection.limit
        )
        selected = selected[: selection.limit]
    return selected, dict(counters)


def _model_fields(
    assistants: Sequence[Mapping[str, Any]], info: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]], int]:
    usage = _message_models(assistants)
    attributed = sum(item["requests"] for item in usage)
    unattributed = len(assistants) - attributed
    if len(usage) == 1:
        only = usage[0]
        if unattributed:
            return None, None, usage, unattributed
        model = {
            "raw": only["model"],
            "id": only["model"],
            "provider": only["provider"],
            "tier": "verified",
        }
        effort = only["variants"][0] if len(only["variants"]) == 1 else None
        return model, effort, usage if len(only["variants"]) > 1 else [], 0
    if len(usage) > 1:
        return None, None, usage, unattributed

    configured = _as_mapping(info.get("model"))
    model_id = configured.get("id")
    provider = configured.get("providerID")
    if isinstance(model_id, str) and model_id:
        model = {"raw": model_id, "id": model_id, "tier": "reported"}
        if isinstance(provider, str) and provider:
            model["provider"] = provider
        variant = configured.get("variant")
        effort = variant if isinstance(variant, str) and variant else None
        return model, effort, [], unattributed
    return None, None, [], unattributed


def _workspace_group(workspace: str) -> str:
    return "workspace:" + sha256(workspace.encode("utf-8")).hexdigest()[:24]


def _emit_document(
    sessions: Sequence[_SessionSource], all_session_ids: set[str], counters: dict[str, int]
) -> dict[str, Any]:
    ordered = sorted(sessions, key=lambda item: str(item.info["id"]))
    source_ids = [str(item.info["id"]) for item in ordered]
    selected_ids = set(source_ids)
    node_ids = {session_id: _entity_id("", session_id) for session_id in source_ids}
    attempt_ids = {
        session_id: _entity_id("attempt:", session_id) for session_id in source_ids
    }

    workspaces = sorted(
        {
            workspace
            for session in ordered
            for workspace in (session.info.get("directory"),)
            if isinstance(workspace, str) and workspace
        }
    )
    groups = [
        {
            "id": _workspace_group(workspace),
            "title": f"OpenCode workspace {index}",
        }
        for index, workspace in enumerate(workspaces, start=1)
    ]

    nodes = []
    attempts = []
    events: list[dict[str, Any]] = []
    starts: list[datetime] = []
    archived_times: list[datetime] = []

    for session in ordered:
        info = session.info
        source_id = str(info["id"])
        times = _as_mapping(info.get("time"))
        started = _timestamp(times.get("created"))
        updated = _timestamp(times.get("updated"))
        archived = _timestamp(times.get("archived"))
        state = "settled_unverified" if archived is not None else "working"
        node: dict[str, Any] = {
            "id": node_ids[source_id],
            "kind": "unknown",
            "title": _short(f"OpenCode session {source_id}", 500),
            "state": state,
        }
        workspace = info.get("directory")
        if isinstance(workspace, str) and workspace:
            node["group"] = _workspace_group(workspace)
        nodes.append(node)

        assistants = _assistant_infos(session)
        model, effort, model_usage, unattributed_models = _model_fields(assistants, info)
        cost, incomplete_usage = _cost_record(assistants, info)
        attempt: dict[str, Any] = {
            "id": attempt_ids[source_id],
            "node": node_ids[source_id],
            "n": 1,
            "harness": "opencode",
            "status": state,
            "cost": cost,
            "session": _short(source_id, 500),
        }
        actor = _short(info.get("agent"), 200)
        if actor is None and assistants:
            actor = _short(assistants[-1].get("agent"), 200)
        if actor is not None:
            attempt["actor"] = actor
        if model is not None:
            attempt["model"] = model
        if effort is not None:
            attempt["effort"] = _short(effort, 100)
        if isinstance(workspace, str) and workspace:
            attempt["origin"] = {
                "launched_by": None,
                "workspace": _short(workspace, 500),
                "external": None,
                "how": None,
                "tier": "verified",
                "evidence": "OpenCode session.directory",
            }
        if started is not None:
            attempt["started_at"] = _format_timestamp(started)
            starts.append(started)
            events.append(
                {
                    "at": _format_timestamp(started),
                    "type": "attempt_started",
                    "node": node_ids[source_id],
                    "attempt": attempt_ids[source_id],
                }
            )
        if updated is not None and updated != started and updated != archived:
            events.append(
                {
                    "at": _format_timestamp(updated),
                    "type": "session_updated",
                    "node": node_ids[source_id],
                    "attempt": attempt_ids[source_id],
                }
            )
        if archived is not None:
            attempt["ended_at"] = _format_timestamp(archived)
            attempt["outcome"] = {
                "result": "settled_unverified",
                "evidence": "heuristic",
                "reason": "OpenCode archived the session without an acceptance signal",
            }
            archived_times.append(archived)
            events.append(
                {
                    "at": _format_timestamp(archived),
                    "type": "attempt_settled",
                    "node": node_ids[source_id],
                    "attempt": attempt_ids[source_id],
                }
            )
        attempt_ext: dict[str, Any] = {}
        if model_usage:
            attempt_ext["model_usage"] = model_usage
        if unattributed_models:
            attempt_ext["model_requests_unattributed"] = unattributed_models
        if incomplete_usage:
            attempt_ext["usage_incomplete_messages"] = incomplete_usage
        if attempt_ext:
            attempt["ext"] = {_EXT: attempt_ext}
        attempts.append(attempt)

    edges = []
    for session in ordered:
        child_id = str(session.info["id"])
        parent_id = session.info.get("parentID")
        if parent_id is None:
            continue
        if not isinstance(parent_id, str) or not parent_id or parent_id == child_id:
            counters["parent_relations_unresolved"] = counters.get(
                "parent_relations_unresolved", 0
            ) + 1
        elif parent_id in selected_ids:
            edges.append(
                {
                    "from": node_ids[parent_id],
                    "to": node_ids[child_id],
                    "kind": "spawn",
                    "tier": "verified",
                    "from_attempt": attempt_ids[parent_id],
                    "to_attempt": attempt_ids[child_id],
                    "evidence": "OpenCode session.parentID matched the selected parent session id",
                }
            )
        elif parent_id in all_session_ids:
            counters["parent_relations_omitted"] = counters.get(
                "parent_relations_omitted", 0
            ) + 1
        else:
            counters["parent_relations_unresolved"] = counters.get(
                "parent_relations_unresolved", 0
            ) + 1

    digest = sha256("\0".join(source_ids).encode("utf-8")).hexdigest()[:24]
    run: dict[str, Any] = {
        "id": f"opencode:{digest}" if source_ids else "opencode:empty",
        "title": f"OpenCode session inventory ({len(source_ids)} selected)",
    }
    if len(workspaces) == 1:
        run["workspace"] = _short(workspaces[0], 500)
    if starts:
        run["started_at"] = _format_timestamp(min(starts))
    if ordered and len(archived_times) == len(ordered):
        run["ended_at"] = _format_timestamp(max(archived_times))
    nonzero_counters = {key: value for key, value in sorted(counters.items()) if value}
    if nonzero_counters:
        run["ext"] = {_EXT: nonzero_counters}

    versions = {
        str(session.info["version"])
        for session in ordered
        if isinstance(session.info.get("version"), str) and session.info["version"]
    }
    source_formats = {session.source_format for session in ordered}
    if source_formats == {"database"}:
        source_contract = "opencode SQLite session/message schema"
    elif source_formats == {"database", "export"}:
        source_contract = "opencode export + SQLite session/message schema"
    else:
        source_contract = "opencode export"
    producer: dict[str, Any] = {
        "name": "dagr-adapter-opencode",
        "framework": "opencode",
        "source_contract": source_contract,
        "capabilities": {
            "groups": True, "events": True, "artifacts": False, "edges_dep": False,
            "edges_spawn": True, "edges_launch": False, "edges_artifact": False,
            "cost_usd": True, "cost_tokens": True, "outcome_evidence": True,
        },
    }
    if len(versions) == 1:
        producer["framework_version"] = _short(versions.pop(), 100)

    events.sort(key=lambda event: (event["at"], event["type"], event["attempt"]))
    return {
        "ocp": "0.2",
        "producer": producer,
        "privacy": {
            "profile": "metadata_only",
            "note": "Prompt, response, reasoning, tool, title and diff text are omitted",
        },
        "run": run,
        "groups": groups,
        "nodes": nodes,
        "edges": sorted(edges, key=lambda edge: (edge["from"], edge["to"])),
        "attempts": attempts,
        "artifacts": [],
        "events": events,
    }


@register
class OpenCodeAdapter(Adapter):
    """Read OpenCode exports and emit metadata-only OCP v0.2.

    The only relation emitted is ``spawn``.  Its tier is ``verified`` because
    the source session's explicit ``parentID`` must equal the selected parent
    session's id.  The adapter emits no heuristic or reported edges.
    """

    name = "opencode"

    def discover(self) -> Sequence[Path]:
        database = _default_database()
        return (database,) if database.is_file() else ()

    def sessions(self) -> Sequence[Mapping[str, Any]]:
        result = []
        for store in self.discover():
            for session in _store_sessions(store):
                record: dict[str, Any] = {
                    "id": str(session.info["id"]),
                    "store": str(store),
                }
                workspace = session.info.get("directory")
                if isinstance(workspace, str) and workspace:
                    record["workspace"] = workspace
                parent_id = session.info.get("parentID")
                if isinstance(parent_id, str) and parent_id:
                    record["parent_id"] = parent_id
                created = _session_created(session)
                if created is not None:
                    record["created_at"] = _format_timestamp(created)
                result.append(record)
        return tuple(sorted(result, key=lambda item: item["id"]))

    def emit(self, selection: Selection) -> dict[str, Any]:
        stores = selection.stores or tuple(self.discover())
        sessions = []
        for store in stores:
            sessions.extend(_store_sessions(Path(store).expanduser()))

        by_id: dict[str, _SessionSource] = {}
        for session in sessions:
            session_id = str(session.info["id"])
            if session_id in by_id:
                raise ValueError(f"duplicate OpenCode session id across stores: {session_id!r}")
            by_id[session_id] = session

        selected, counters = _selected_sessions(tuple(by_id.values()), selection)
        default_database = _default_database().resolve()
        materialized = []
        for session in selected:
            if session.messages is not None:
                materialized.append(session)
            elif session.database is not None and session.database.resolve() == default_database:
                materialized.append(_export_session(session))
            else:
                materialized.append(_messages_from_database(session))
        return _emit_document(materialized, set(by_id), counters)
