"""Cross-session links and selection helpers for the OMP adapter."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping

from .omp_parse import (
    _entity_id,
    _message,
    _milliseconds_timestamp,
    _nonempty_string,
    _record_timestamp,
)
from .omp_types import _NativeSession, _SpawnFacts, _TaskFacts


def _terminal_end(session: _NativeSession) -> datetime | None:
    last_message_record = next(
        (record for record in reversed(session.records) if _message(record) is not None),
        None,
    )
    if last_message_record is None:
        return None
    message = _message(last_message_record)
    if (
        message is None
        or message.get("role") != "assistant"
        or message.get("stopReason") == "toolUse"
    ):
        return None
    return _milliseconds_timestamp(message.get("completedAt")) or _record_timestamp(
        last_message_record
    )


def _session_init_signature(session: _NativeSession) -> tuple[str, str] | None:
    if session.session_init is None:
        return None
    task = _nonempty_string(session.session_init.get("task"))
    agent = _nonempty_string(session.session_init.get("agent"))
    return (task, agent) if task is not None and agent is not None else None


def _spawn_facts(
    sessions: tuple[_NativeSession, ...], facts: Mapping[str, _TaskFacts]
) -> _SpawnFacts:
    edges: list[Mapping[str, str]] = []
    child_sessions = 0
    matched_children = 0
    zero_candidate_children = 0
    ambiguous_children = 0
    multiple_job_children = 0
    missing_signature_children = 0
    terminal_ends = {session.session_id: _terminal_end(session) for session in sessions}
    for child in sessions:
        if child.session_init is None:
            continue
        child_sessions += 1
        signature = _session_init_signature(child)
        if signature is None:
            missing_signature_children += 1
            zero_candidate_children += 1
            continue
        if child.started_at is None:
            zero_candidate_children += 1
            continue
        child_task, child_agent = signature
        candidates: list[tuple[str, str]] = []
        for parent in sessions:
            if parent.session_id == child.session_id or parent.started_at is None:
                continue
            parent_end = terminal_ends[parent.session_id]
            for job in facts[parent.session_id].jobs:
                if (
                    job.task != child_task
                    or job.agent != child_agent
                    or job.called_at is None
                ):
                    continue
                if not (parent.started_at <= job.called_at <= child.started_at):
                    continue
                if parent_end is not None and child.started_at > parent_end:
                    continue
                candidates.append((parent.session_id, job.job_id))
        if not candidates:
            zero_candidate_children += 1
            continue
        if len(candidates) != 1:
            multiple_job_children += 1
            if len({parent_id for parent_id, _ in candidates}) > 1:
                ambiguous_children += 1
            continue
        parent_id, _ = candidates[0]
        matched_children += 1
        edges.append(
            {
                "from": _entity_id("omp-session", parent_id),
                "to": _entity_id("omp-session", child.session_id),
                "kind": "spawn",
                "tier": "heuristic",
                "evidence": (
                    "One OMP task job and child session_init match exactly by task and "
                    "agent within overlapping timestamps; OMP records no parent-session key"
                ),
            }
        )
    return _SpawnFacts(
        edges=tuple(sorted(edges, key=lambda edge: (edge["from"], edge["to"]))),
        child_sessions=child_sessions,
        matched_children=matched_children,
        zero_candidate_children=zero_candidate_children,
        ambiguous_children=ambiguous_children,
        multiple_job_children=multiple_job_children,
        missing_signature_children=missing_signature_children,
    )


def _run_id(sessions: tuple[_NativeSession, ...]) -> str:
    if not sessions:
        return "omp:empty"
    native_ids = json.dumps(
        sorted(session.session_id for session in sessions),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return "omp:" + hashlib.sha256(native_ids.encode("utf-8")).hexdigest()[:20]


def _selection_filter(
    sessions: list[_NativeSession],
    name: str,
    active: bool,
    source: Any,
    matches: Any,
) -> tuple[list[_NativeSession], dict[str, str | bool | int]]:
    unknown = 0
    excluded = 0
    retained: list[_NativeSession] = []
    if not active:
        return sessions, {"name": name, "active": False, "unknown": 0, "excluded": 0}
    for session in sessions:
        value = source(session)
        if value is None:
            unknown += 1
            retained.append(session)
        elif matches(value):
            retained.append(session)
        else:
            excluded += 1
    return retained, {
        "name": name,
        "active": True,
        "unknown": unknown,
        "excluded": excluded,
    }
