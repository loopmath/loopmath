"""Pi session JSONL to OCP v0.2.

One Pi session is one OCP node with one attempt. Usage-bearing entries are
measured requests: their token and USD fields are summed into the attempt's
core cost record, while ``dev.dagr.adapter.pi.usage_records`` retains the exact
per-entry breakdown that OCP v0.2 cannot express. Model labels on assistant
messages are verified; thinking effort is reported by Pi's explicit
``thinking_level_change`` entries.

Pi records no terminal session signal. Nodes and attempts therefore remain
``working`` and omit terminal outcomes and end times.

Pi ``parentId`` values describe entry ancestry inside a conversation, not an
agent-to-agent relationship. ``parentSession`` describes a session fork, for
which OCP v0.2 has no edge kind. This is a format-level constraint, not a
search result: the adapter emits no edges because the Pi format records no OCP
relation, not because the selected sessions happened to contain none.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys as _sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType as _ModuleType
from typing import Any, Iterable, Mapping, Sequence

from .base import Adapter, Selection
from .pi_support import (
    _bound,
    _cost,
    _entry_state,
    _jsonl_paths,
    _load_selected,
    _nonnegative_int,
    _nonnegative_number,
    _parse_timestamp,
    _read_session,
    _selection_stats,
    _session_locator,
    _session_sort_key,
    _stable_id,
    _SUPPORTED_SESSION_VERSIONS,
    _sum_if_complete,
    _Session,
    _uniform_attempt_model,
    _uniform_effort,
    _USAGE_ENTRY_TYPES,
    _usage_records,
    _usage_source,
)
from .registry import register


_DEFAULT_STORE = Path.home() / ".pi" / "agent" / "sessions"
_EXT = "dev.dagr.adapter.pi"


class _PiModule(_ModuleType):
    def __setattr__(
        self,
        name: str,
        value: object,
        _support=_sys.modules[f"{__package__}.pi_support"],
        _bridged=frozenset(
            {
                "Any",
                "Iterable",
                "Mapping",
                "Path",
                "Selection",
                "Sequence",
                "_SUPPORTED_SESSION_VERSIONS",
                "_Session",
                "_USAGE_ENTRY_TYPES",
                "_bound",
                "_entry_state",
                "_jsonl_paths",
                "_nonnegative_int",
                "_nonnegative_number",
                "_parse_timestamp",
                "_read_session",
                "_selection_stats",
                "_session_sort_key",
                "_sum_if_complete",
                "_usage_source",
                "datetime",
                "hashlib",
                "json",
                "math",
                "timezone",
            }
        ),
    ) -> None:
        super().__setattr__(name, value)
        if name in _bridged:
            setattr(_support, name, value)


_sys.modules[__name__].__class__ = _PiModule
del _ModuleType, _PiModule, _sys


@register
class PiAdapter(Adapter):
    """Emit no edges because Pi records no OCP relation, not because none were found."""

    name = "pi"

    def discover(self) -> Sequence[Path]:
        return (_DEFAULT_STORE,) if _DEFAULT_STORE.is_dir() else ()

    def sessions(self) -> Sequence[Mapping[str, Any]]:
        sessions, _ = _load_selected(Selection(), self.discover())
        return tuple(
            {
                "id": session.session_id,
                "workspace": session.workspace,
                "timestamp": session.started_at,
                "path": str(session.path),
            }
            for session in sessions
        )

    def emit(self, selection: Selection) -> dict[str, Any]:
        sessions, stats = _load_selected(selection, self.discover())

        groups: list[dict[str, Any]] = []
        group_by_workspace: dict[str, str] = {}
        for workspace in sorted({s.workspace for s in sessions if s.workspace is not None}):
            group_id = _stable_id("pi-workspace:", workspace)
            group_by_workspace[workspace] = group_id
            groups.append({"id": group_id, "title": "Pi workspace"})

        nodes: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        for session in sessions:
            node_id = _stable_id("pi-session:", session.session_id)
            attempt_id = _stable_id("pi-attempt:", session.session_id)
            node: dict[str, Any] = {
                "id": node_id,
                "kind": "unknown",
                "title": "Pi coding session",
                "state": "working",
                "labels": {"harness": "pi"},
            }
            if session.workspace is not None:
                node["group"] = group_by_workspace[session.workspace]
            nodes.append(node)

            usage_records, incomplete_usage_records = _usage_records(session)
            attempt: dict[str, Any] = {
                "id": attempt_id,
                "node": node_id,
                "n": 1,
                "actor": "pi-agent",
                "harness": "pi",
                "status": "working",
                "session": _session_locator(session.session_id),
                "ext": {
                    _EXT: {
                        "usage_records": usage_records,
                        "incomplete_usage_records": incomplete_usage_records,
                        "malformed_records": session.malformed_records,
                    }
                },
            }
            if session.started_at is not None:
                attempt["started_at"] = session.started_at
                events.append(
                    {
                        "at": session.started_at,
                        "type": "attempt_started",
                        "node": node_id,
                        "attempt": attempt_id,
                    }
                )
            model = _uniform_attempt_model(usage_records)
            if model is not None:
                attempt["model"] = model
            effort = _uniform_effort(usage_records)
            if effort is not None:
                attempt["effort"] = effort
            cost = _cost(usage_records)
            if cost is not None:
                attempt["cost"] = cost
            attempts.append(attempt)

        workspaces = sorted({s.workspace for s in sessions if s.workspace is not None})
        identity = "\0".join(s.session_id for s in sessions)
        run: dict[str, Any] = {
            "id": "pi-adapt:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            "title": "Pi session import",
            "ext": {_EXT: {"selection": stats}},
        }
        starts = [s.started_dt for s in sessions if s.started_dt is not None]
        if starts:
            earliest = min(starts)
            run["started_at"] = earliest.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            )
        if len(workspaces) == 1 and len(workspaces[0]) <= 500:
            run["workspace"] = workspaces[0]

        events.sort(
            key=lambda event: (
                _parse_timestamp(event["at"])[0]
                or datetime.max.replace(tzinfo=timezone.utc),
                event["attempt"],
            )
        )
        return {
            "ocp": "0.2",
            "producer": {
                "name": "loopmath/adapter-pi",
                "framework": "pi",
                "source_contract": "Pi session JSONL v1-v3",
                "capabilities": {
                    "groups": True, "events": True, "artifacts": False, "edges_dep": False,
                    "edges_spawn": False, "edges_launch": False, "edges_artifact": False,
                    "cost_usd": True, "cost_tokens": True, "outcome_evidence": False,
                },
            },
            "privacy": {
                "profile": "metadata_only",
                "note": "Prompt, response, tool, compaction, and summary text is excluded",
            },
            "run": run,
            "groups": groups,
            "nodes": nodes,
            "edges": [],
            "attempts": attempts,
            "artifacts": [],
            "events": events,
        }
