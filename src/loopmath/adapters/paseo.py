"""Read Paseo's per-agent JSON records as OCP v0.2.

Evidence is deliberately narrow. Paseo reports ``provider``, ``cwd``,
``config.model``, and ``config.thinkingOptionId``; the adapter therefore emits
the harness configuration, model, effort, and origin workspace as ``reported``
metadata. A closed agent becomes ``settled_unverified`` because closure is not
evidence that the work was accepted.

The vendor-session join is mechanically supported when
``persistence.sessionId`` equals both available native corroborators,
``persistence.nativeHandle`` and ``runtimeInfo.sessionId``. The matching value
is emitted as ``attempt.session``. Read-only inspection on 2026-09-02 also
matched every local value to a Claude or Codex session filename without reading
transcripts. This is a verified identity correlation, not a graph relation.
Paseo records no distinct launcher or parent endpoint, so this adapter emits no
``launch`` or ``spawn`` edge and uses no heuristic relation.

Paseo has no token counts, cost, turn or tool structure, acceptance result, or
parent relation. Titles, system prompts, feature descriptions, and persistence
metadata outside the fields above are discarded. Document-level diagnostics
count source values that cannot be represented and every selection exclusion;
they never retain the malformed or unknown source value.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .base import Adapter, Selection
from .registry import register


_DEFAULT_STORE = Path.home() / ".paseo" / "agents"
_NAMESPACE = "dev.dagr.adapter.paseo"
_TERMINAL = {"failed", "canceled", "settled_unverified"}
_STATUS = {
    "working": "working",
    "error": "failed",
    "failed": "failed",
    "canceled": "canceled",
    "cancelled": "canceled",
    "closed": "settled_unverified",
    "completed": "settled_unverified",
    "stopped": "settled_unverified",
}
_RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_DIAGNOSTIC_REASONS = (
    "record_unreadable",
    "record_invalid_json",
    "record_json_not_object",
    "record_not_agent",
    "record_last_status_missing",
    "record_last_status_invalid_type",
    "record_last_status_unknown",
    "session_id_missing_or_invalid",
    "session_corroborator_missing_or_invalid",
    "session_corroborator_mismatch",
    "timestamp_missing",
    "timestamp_invalid_type",
    "timestamp_invalid_value",
    "timestamp_timezone_missing",
    "filter_session",
    "filter_workspace",
    "filter_created_at_unknown",
    "filter_before_since",
    "filter_after_until",
    "filter_limit",
)


@dataclass(frozen=True)
class _Agent:
    path: Path
    id: str
    provider: str | None
    workspace: str | None
    model: str | None
    effort: str | None
    created_at: str | None
    created_time: datetime | None
    updated_at: str | None
    updated_time: datetime | None
    status: str | None
    source_status: str | None
    status_issue: str | None
    native_session: str | None
    session_issue: str | None
    timestamp_issues: tuple[str, ...]


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
    print(f"paseo diagnostics: {rendered}", file=sys.stderr)


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _timestamp(value: Any) -> tuple[tuple[str, datetime] | None, str | None]:
    """Normalize a Paseo timestamp to an RFC 3339 UTC timestamp."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    if not isinstance(value, str):
        return None, "invalid_type"
    raw = value.strip()
    try:
        parseable = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        parsed = datetime.fromisoformat(parseable)
    except ValueError:
        return None, "invalid_value"
    if parsed.tzinfo is None:
        return None, "timezone_missing"
    if _RFC3339.fullmatch(raw) is None:
        return None, "invalid_value"
    utc = parsed.astimezone(timezone.utc)
    return (utc.isoformat().replace("+00:00", "Z"), utc), None


def _bound(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    parsed, _issue = _timestamp(value)
    if parsed is None:
        raise ValueError(f"Paseo selection {name} must be an RFC 3339 timestamp")
    return parsed[1]


def _status(value: Any) -> tuple[str | None, str | None, str | None]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, None, "missing"
    if not isinstance(value, str):
        return None, None, "invalid_type"
    source = value.strip()
    normalized = source.lower().replace("-", "_")
    status = _STATUS.get(normalized)
    if status is None:
        return None, None, "unknown"
    return status, source, None


def _native_session(
    persistence: Mapping[str, Any], runtime: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    session_id = _string(persistence.get("sessionId"))
    if session_id is None:
        return None, "id_missing_or_invalid"
    native_handle = _string(persistence.get("nativeHandle"))
    runtime_session = _string(runtime.get("sessionId"))
    if native_handle is None or runtime_session is None:
        return None, "corroborator_missing_or_invalid"
    if native_handle != session_id or runtime_session != session_id:
        return None, "corroborator_mismatch"
    return session_id, None


def _agent_paths(stores: Iterable[Path]) -> tuple[Path, ...]:
    paths: dict[str, Path] = {}
    for raw_store in stores:
        store = raw_store.expanduser()
        if store.is_file() and store.suffix.lower() == ".json":
            candidates = (store,)
        elif store.is_dir():
            candidates = tuple(store.rglob("*.json"))
        else:
            candidates = ()
        for candidate in candidates:
            resolved = candidate.resolve()
            paths[str(resolved)] = resolved
    return tuple(paths[key] for key in sorted(paths))


def _read_agent(path: Path) -> tuple[_Agent | None, str | None]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None, "record_unreadable"
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return None, "record_invalid_json"
    if not isinstance(record, dict):
        return None, "record_json_not_object"

    agent_id = _string(record.get("id"))
    if agent_id is None:
        return None, "record_not_agent"

    config = record.get("config")
    config = config if isinstance(config, dict) else {}
    persistence = record.get("persistence")
    persistence = persistence if isinstance(persistence, dict) else {}
    runtime = record.get("runtimeInfo")
    runtime = runtime if isinstance(runtime, dict) else {}

    native_session, session_issue = _native_session(persistence, runtime)
    status, source_status, status_issue = _status(record.get("lastStatus"))
    created, created_issue = _timestamp(record.get("createdAt"))
    updated, updated_issue = _timestamp(record.get("updatedAt"))
    return (
        _Agent(
            path=path,
            id=agent_id,
            provider=_string(record.get("provider")),
            workspace=_string(record.get("cwd")),
            model=_string(config.get("model")),
            effort=_string(config.get("thinkingOptionId")),
            created_at=created[0] if created else None,
            created_time=created[1] if created else None,
            updated_at=updated[0] if updated else None,
            updated_time=updated[1] if updated else None,
            status=status,
            source_status=source_status,
            status_issue=status_issue,
            native_session=native_session,
            session_issue=session_issue,
            timestamp_issues=tuple(
                issue for issue in (created_issue, updated_issue) if issue is not None
            ),
        ),
        None,
    )


def _agents(
    stores: Iterable[Path],
) -> tuple[tuple[_Agent, ...], dict[str, int]]:
    diagnostics = _empty_diagnostics()
    agents: list[_Agent] = []
    for path in _agent_paths(stores):
        agent, issue = _read_agent(path)
        if issue is not None:
            diagnostics[issue] += 1
            continue
        if agent is None:
            raise AssertionError("readable Paseo agent record has no parsed agent")
        agents.append(agent)
    agents.sort(key=lambda agent: (agent.created_at or "", agent.id, str(agent.path)))
    seen: set[str] = set()
    for agent in agents:
        if agent.id in seen:
            raise ValueError(f"duplicate Paseo agent id {agent.id!r} across selected stores")
        seen.add(agent.id)
        if agent.status_issue is not None:
            diagnostics[f"record_last_status_{agent.status_issue}"] += 1
        if agent.session_issue is not None:
            diagnostics[f"session_{agent.session_issue}"] += 1
        for issue in agent.timestamp_issues:
            diagnostics[f"timestamp_{issue}"] += 1
    return tuple(agents), diagnostics


def _selected_agents(
    agents: tuple[_Agent, ...],
    selection: Selection,
    diagnostics: dict[str, int],
) -> tuple[_Agent, ...]:
    since = _bound(selection.since, "since")
    until = _bound(selection.until, "until")
    wanted_sessions = set(selection.session_ids)
    wanted_workspaces = set(selection.workspaces)
    selected: list[_Agent] = []
    for agent in agents:
        if wanted_sessions and agent.id not in wanted_sessions:
            diagnostics["filter_session"] += 1
            continue
        if wanted_workspaces and agent.workspace not in wanted_workspaces:
            diagnostics["filter_workspace"] += 1
            continue
        if agent.status is None:
            continue
        if (since is not None or until is not None) and agent.created_time is None:
            diagnostics["filter_created_at_unknown"] += 1
            continue
        if since is not None and agent.created_time is not None and agent.created_time < since:
            diagnostics["filter_before_since"] += 1
            continue
        if until is not None and agent.created_time is not None and agent.created_time > until:
            diagnostics["filter_after_until"] += 1
            continue
        selected.append(agent)
    if selection.limit is not None:
        diagnostics["filter_limit"] += max(0, len(selected) - selection.limit)
        selected = selected[: selection.limit]
    return tuple(selected)


def _document_run_id(agents: tuple[_Agent, ...]) -> str:
    if len(agents) == 1:
        return agents[0].id
    if not agents:
        return "paseo-empty-selection"
    material = "\n".join(agent.id for agent in agents)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"paseo-selection-{digest}"


@register
class PaseoAdapter(Adapter):
    """Adapter for Paseo's local, read-only per-agent JSON store."""

    name = "paseo"

    def discover(self) -> tuple[Path, ...]:
        return (_DEFAULT_STORE,) if _DEFAULT_STORE.is_dir() else ()

    def sessions(self) -> tuple[Mapping[str, Any], ...]:
        agents, _diagnostics = _agents(self.discover())
        return tuple(
            {
                "id": agent.id,
                "store": str(agent.path),
                "created_at": agent.created_at,
                "updated_at": agent.updated_at,
                "workspace": agent.workspace,
                "provider": agent.provider,
                "model": agent.model,
                "effort": agent.effort,
                "session": agent.native_session,
            }
            for agent in agents
        )

    def emit(self, selection: Selection) -> dict[str, Any]:
        stores = selection.stores or tuple(self.discover())
        source_agents, diagnostics = _agents(stores)
        agents = _selected_agents(source_agents, selection, diagnostics)
        _print_diagnostics(diagnostics)
        nodes: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []

        for agent in agents:
            status = agent.status
            if status is None:
                raise AssertionError("selected Paseo agent has no representable status")
            nodes.append({"id": agent.id, "kind": "task", "state": status})
            attempt: dict[str, Any] = {
                "id": f"paseo-attempt-{agent.id}",
                "node": agent.id,
                "n": 1,
                "status": status,
            }
            if agent.provider is not None:
                attempt["harness"] = agent.provider
            if agent.model is not None:
                attempt["model"] = {"raw": agent.model, "tier": "reported"}
            if agent.effort is not None:
                attempt["effort"] = agent.effort
            if agent.created_at is not None:
                attempt["started_at"] = agent.created_at
            if agent.workspace is not None:
                attempt["origin"] = {
                    "launched_by": None,
                    "workspace": agent.workspace,
                    "external": None,
                    "how": None,
                    "tier": "reported",
                    "evidence": "Paseo cwd",
                }
            if agent.native_session is not None:
                attempt["session"] = agent.native_session
            if status in _TERMINAL:
                if agent.updated_at is not None:
                    attempt["ended_at"] = agent.updated_at
                attempt["outcome"] = {
                    "result": status,
                    "evidence": "reported",
                    "receipt": f"Paseo lastStatus={agent.source_status!r}",
                }
            attempts.append(attempt)

        run: dict[str, Any] = {"id": _document_run_id(agents)}
        workspaces = {agent.workspace for agent in agents if agent.workspace is not None}
        if len(workspaces) == 1:
            run["workspace"] = next(iter(workspaces))
        starts = [
            (agent.created_at, agent.created_time)
            for agent in agents
            if agent.created_at is not None and agent.created_time is not None
        ]
        if starts:
            run["started_at"] = min(starts, key=lambda value: value[1])[0]
        if agents and all(
            agent.status in _TERMINAL and agent.updated_at is not None
            and agent.updated_time is not None
            for agent in agents
        ):
            ends = [
                (agent.updated_at, agent.updated_time)
                for agent in agents
                if agent.updated_at is not None and agent.updated_time is not None
            ]
            run["ended_at"] = max(ends, key=lambda value: value[1])[0]

        return {
            "ocp": "0.2",
            "producer": {
                "name": "dagr-adapter-paseo",
                "framework": "paseo",
                "source_contract": "paseo/agent-json",
                "capabilities": {
                    "groups": False, "events": False, "artifacts": False, "edges_dep": False,
                    "edges_spawn": False, "edges_launch": False, "edges_artifact": False,
                    "cost_usd": False, "cost_tokens": False, "outcome_evidence": True,
                },
            },
            "privacy": {
                "profile": "metadata_only",
                "note": "Titles, prompts, feature descriptions, and transcript text omitted.",
            },
            "run": run,
            "nodes": nodes,
            "edges": [],
            "attempts": attempts,
            "ext": {
                _NAMESPACE: {"diagnostics": _diagnostic_records(diagnostics)}
            },
        }
