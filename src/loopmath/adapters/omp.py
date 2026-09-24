"""OMP 18.1.5 session-store adapter.

OMP records an explicit entry tree through ``id`` and ``parentId``. Its task
tool calls, tool results, progress/result jobs, and ``session_init`` child
markers are also explicit source facts. OMP does not, however, persist a
parent-session id on the child. A cross-session ``spawn`` edge is therefore
emitted only for one exact task-and-agent job match across all candidate
parents, and its evidence tier is always ``heuristic``.

The emitted document uses the ``metadata_only`` privacy profile. Prompt,
completion, system-prompt, and task text are used only while matching records
in memory and are never copied into OCP output.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys as _sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType as _ModuleType
from typing import Any, Iterable, Mapping

from . import omp_facts as _omp_facts
from . import omp_links as _omp_links
from . import omp_parse as _omp_parse
from . import omp_types as _omp_types
from .base import Adapter, Selection, Usage
from .omp_facts import (
    _effort,
    _message_role_counts,
    _model,
    _parent_link_counts,
    _task_facts,
    _usage_facts,
)
from .omp_links import (
    _run_id,
    _selection_filter,
    _session_init_signature,
    _spawn_facts,
    _terminal_end,
)
from .omp_parse import (
    _bounded,
    _entity_id,
    _format_timestamp,
    _load_sessions,
    _message,
    _milliseconds_timestamp,
    _nonempty_string,
    _parse_timestamp,
    _read_session,
    _record_timestamp,
    _session_paths,
    _source_finding,
    _source_finding_rows,
    _strict_json,
)
from .omp_types import (
    _EXT_NAMESPACE,
    _KNOWN_MESSAGE_ROLES,
    _TOKEN_FIELDS,
    _LoadResult,
    _NativeSession,
    _SpawnFacts,
    _TaskFacts,
    _TaskJob,
    _UsageFacts,
    _UsageGroup,
)
from .registry import register


DEFAULT_STORE = Path.home() / ".omp" / "agent" / "sessions"


@register
class OMPAdapter(Adapter):
    """Convert OMP 18.1.5 JSONL v3 session stores to OCP v0.2."""

    name = "omp"

    def discover(self) -> tuple[Path, ...]:
        """Return the verified default OMP session-store root when present."""

        return (DEFAULT_STORE,) if DEFAULT_STORE.is_dir() else ()

    def sessions(self) -> tuple[Mapping[str, Any], ...]:
        """Return deterministic descriptors for sessions in the default store."""

        descriptors: list[Mapping[str, Any]] = []
        for session in _load_sessions(self.discover()).sessions:
            descriptor: dict[str, Any] = {
                "id": session.session_id,
                "path": str(session.path),
                "workspace": session.workspace,
                "format_version": session.format_version,
                "is_subagent": session.session_init is not None,
            }
            if session.started_at is not None:
                descriptor["started_at"] = _format_timestamp(session.started_at)
            descriptors.append(descriptor)
        return tuple(descriptors)

    def emit(self, selection: Selection) -> dict[str, Any]:
        """Emit one metadata-only OCP v0.2 document for the selected sessions."""

        if selection.limit is not None and selection.limit < 1:
            raise ValueError("selection.limit must be at least 1")
        since = _parse_timestamp(selection.since) if selection.since is not None else None
        until = _parse_timestamp(selection.until) if selection.until is not None else None
        if selection.since is not None and since is None:
            raise ValueError("selection.since must be an RFC 3339 timestamp with an offset")
        if selection.until is not None and until is None:
            raise ValueError("selection.until must be an RFC 3339 timestamp with an offset")
        if since is not None and until is not None and since > until:
            raise ValueError("selection.since must not be after selection.until")

        stores = selection.stores or tuple(self.discover())
        loaded = _load_sessions(stores)
        sessions = list(loaded.sessions)
        filter_rows: list[dict[str, str | bool | int]] = []
        session_ids = set(selection.session_ids)
        workspaces = set(selection.workspaces)
        sessions, row = _selection_filter(
            sessions,
            "session_id",
            bool(session_ids),
            lambda session: session.session_id,
            lambda value: value in session_ids,
        )
        filter_rows.append(row)
        sessions, row = _selection_filter(
            sessions,
            "workspace",
            bool(workspaces),
            lambda session: session.workspace,
            lambda value: value in workspaces,
        )
        filter_rows.append(row)
        sessions, row = _selection_filter(
            sessions,
            "since",
            since is not None,
            lambda session: session.started_at,
            lambda value: value >= since,
        )
        filter_rows.append(row)
        sessions, row = _selection_filter(
            sessions,
            "until",
            until is not None,
            lambda session: session.started_at,
            lambda value: value <= until,
        )
        filter_rows.append(row)
        limit_omitted = 0
        if selection.limit is not None and len(sessions) > selection.limit:
            limit_omitted = len(sessions) - selection.limit
            sessions = sessions[: selection.limit]
        selected = tuple(sessions)
        task_facts = {session.session_id: _task_facts(session) for session in selected}
        usage_facts = {session.session_id: _usage_facts(session) for session in selected}
        spawn = _spawn_facts(selected, task_facts)

        nodes: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        ends: list[datetime] = []
        for session in selected:
            node_id = _entity_id("omp-session", session.session_id)
            attempt_id = _entity_id("omp-attempt", session.session_id)
            terminal_end = _terminal_end(session)
            state = "settled_unverified" if terminal_end is not None else "working"
            title_id = (
                session.session_id
                if len(session.session_id) <= 440
                else hashlib.sha256(session.session_id.encode("utf-8")).hexdigest()
            )
            nodes.append(
                {
                    "id": node_id,
                    "kind": "task",
                    "title": f"OMP session {title_id}",
                    "state": state,
                }
            )

            linked_parents, unresolved_parents, malformed_parents = _parent_link_counts(
                session
            )
            tasks = task_facts[session.session_id]
            usages = usage_facts[session.session_id]
            usage_by_model = [
                {
                    "provider": group.provider,
                    "model": group.model,
                    "calls": group.calls,
                    "usage_complete_calls": group.complete_calls,
                    "usage_unknown_calls": group.unknown_calls,
                    "known_input_tokens": group.known_tokens.get("input_tokens", 0),
                    "known_cached_input_tokens": group.known_tokens.get(
                        "cached_input_tokens", 0
                    ),
                    "known_cache_creation_tokens": group.known_tokens.get(
                        "cache_creation_tokens", 0
                    ),
                    "known_output_tokens": group.known_tokens.get("output_tokens", 0),
                    "unknown_reasons": [
                        {"reason": reason, "count": count}
                        for reason, count in sorted(group.unknown_reasons.items())
                        if count
                    ],
                }
                for group in usages.groups
            ]
            pricing_results = [
                (group, self.price_usage(group.model, group.usage))
                for group in usages.groups
            ]
            pricing_by_model = [
                {
                    "provider": group.provider,
                    "model": group.model,
                    "priced": result.priced,
                    "provisional": result.provisional,
                    "reason": result.reason,
                    "usd": result.usd,
                    "estimate_usd": result.estimate_usd,
                }
                for group, result in pricing_results
            ]
            origin_evidence = (
                "OMP session header reports cwd but no durable launch or external-origin fact"
                if session.workspace is not None
                else "OMP records no usable cwd, launch, or external-origin fact"
            )
            attempt_ext: dict[str, Any] = {
                "message_role_counts": _message_role_counts(session),
                "record_parent_links": linked_parents,
                "unresolved_parent_links": unresolved_parents,
                "malformed_parent_links": malformed_parents,
                "task_calls": tasks.calls,
                "linked_task_results": tasks.linked_results,
                "linked_task_jobs": tasks.linked_jobs,
                "task_jobs": tasks.task_jobs,
                "task_findings": [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(tasks.findings.items())
                    if count
                ],
                "model_calls": usages.calls,
                "usage_complete_calls": usages.complete_calls,
                "usage_unknown_calls": usages.unknown_calls,
                "usage_findings": [
                    {"reason": reason, "count": count}
                    for reason, count in sorted(usages.findings.items())
                    if count
                ],
                "usage_by_model": usage_by_model,
                "pricing_by_model": pricing_by_model,
            }
            attempt: dict[str, Any] = {
                "id": attempt_id,
                "node": node_id,
                "n": 1,
                "actor": "omp",
                "harness": "omp",
                "cause": {"type": "initial"},
                "status": state,
                "origin": {
                    "launched_by": None,
                    "workspace": _bounded(session.workspace, 500),
                    "external": None,
                    "how": None,
                    "tier": "reported",
                    "evidence": origin_evidence,
                },
                "role": {"value": None, "tier": "reported", "evidence": None},
                "phase": {"value": None, "tier": "reported", "evidence": None},
                "ext": {_EXT_NAMESPACE: attempt_ext},
            }
            if len(session.session_id) <= 500:
                attempt["session"] = session.session_id
            if session.started_at is not None:
                attempt["started_at"] = _format_timestamp(session.started_at)
            model = _model(session)
            if model is not None:
                attempt["model"] = model
            effort = _effort(session)
            if effort is not None:
                attempt["effort"] = effort
            if usages.usage is not None:
                settled = bool(pricing_results) and all(
                    result.priced for _, result in pricing_results
                )
                usd = None
                if settled:
                    usd = math.fsum(
                        result.usd
                        for _, result in pricing_results
                        if result.usd is not None
                    )
                cost = usages.usage.to_ocp(usd=usd)
                cost["requests"] = usages.calls
                attempt["cost"] = cost

            if session.session_init is not None:
                agent = _nonempty_string(session.session_init.get("agent"))
                if agent is not None and len(agent) <= 200:
                    attempt["actor"] = agent
                attempt["origin"] = {
                    "launched_by": None,
                    "workspace": _bounded(session.workspace, 500),
                    "external": None,
                    "how": "OMP task subagent; parent session not recorded",
                    "tier": "reported",
                    "evidence": (
                        "OMP session_init reports a task subagent but no parent or external origin"
                    ),
                }
                attempt["role"] = {
                    "value": "subagent",
                    "tier": "verified",
                    "evidence": "OMP session_init record",
                }
            elif tasks.calls:
                attempt["role"] = {
                    "value": "lead",
                    "tier": "heuristic",
                    "evidence": "OMP task activity suggests an orchestration lead role",
                }

            if terminal_end is not None:
                ends.append(terminal_end)
                attempt["ended_at"] = _format_timestamp(terminal_end)
                attempt["outcome"] = {
                    "result": "settled_unverified",
                    "evidence": "heuristic",
                    "reason": "OMP recorded a completed assistant turn but no acceptance signal",
                }
            attempts.append(attempt)

        producer: dict[str, Any] = {
            "name": "loopmath/adapter-omp",
            "framework": "omp",
            "framework_version": "18.1.5",
            "source_contract": "omp/session-jsonl-v3",
            "capabilities": {
                "groups": False, "events": False, "artifacts": False, "edges_dep": False,
                "edges_spawn": True, "edges_launch": False, "edges_artifact": False,
                "cost_usd": True, "cost_tokens": True, "outcome_evidence": True,
            },
        }
        producer_version = self.producer_version()
        if producer_version is not None:
            producer["version"] = producer_version
        run_ext = {
            "source_files": {
                "candidates": loaded.candidates,
                "loaded": len(loaded.sessions),
            },
            "source_findings": _source_finding_rows(loaded.findings),
            "selection_filters": filter_rows,
            "limit_omitted": limit_omitted,
            "spawn_matching": {
                "child_sessions": spawn.child_sessions,
                "matched_children": spawn.matched_children,
                "zero_candidate_children": spawn.zero_candidate_children,
                "ambiguous_children": spawn.ambiguous_children,
                "multiple_job_children": spawn.multiple_job_children,
                "missing_signature_children": spawn.missing_signature_children,
            },
        }
        doc: dict[str, Any] = {
            "ocp": "0.2",
            "producer": producer,
            "privacy": {
                "profile": "metadata_only",
                "note": "Transcript, prompt, completion, system-prompt, and task text omitted",
            },
            "run": {"id": _run_id(selected), "ext": {_EXT_NAMESPACE: run_ext}},
            "nodes": nodes,
            "edges": list(spawn.edges),
            "attempts": attempts,
        }
        starts = [
            session.started_at
            for session in selected
            if session.started_at is not None
        ]
        if selected and len(starts) == len(selected):
            doc["run"]["started_at"] = _format_timestamp(min(starts))
        if selected and len(ends) == len(selected):
            doc["run"]["ended_at"] = _format_timestamp(max(ends))
        unique_workspaces = {
            session.workspace for session in selected if session.workspace is not None
        }
        if len(unique_workspaces) == 1:
            workspace = _bounded(next(iter(unique_workspaces)), 500)
            if workspace is not None:
                doc["run"]["workspace"] = workspace
        return doc


class _OMPModule(_ModuleType):
    def __setattr__(
        self,
        name: str,
        value: Any,
        _targets={
            "Any": (_omp_types, _omp_parse, _omp_links),
            "Mapping": (_omp_types, _omp_parse, _omp_links),
            "Path": (_omp_types, _omp_parse),
            "Usage": (_omp_types, _omp_facts),
            "_NativeSession": (_omp_types, _omp_parse, _omp_facts, _omp_links),
            "_TaskJob": (_omp_types, _omp_facts),
            "_UsageGroup": (_omp_types, _omp_facts),
            "datetime": (_omp_types, _omp_parse, _omp_links),
            "Counter": (_omp_parse, _omp_facts),
            "Iterable": (_omp_parse,),
            "_LoadResult": (_omp_parse,),
            "_nonempty_string": (_omp_parse, _omp_facts, _omp_links),
            "_parse_timestamp": (_omp_parse,),
            "_read_session": (_omp_parse,),
            "_record_timestamp": (_omp_parse, _omp_facts, _omp_links),
            "_session_paths": (_omp_parse,),
            "_source_finding": (_omp_parse,),
            "_strict_json": (_omp_parse,),
            "hashlib": (_omp_parse, _omp_links),
            "json": (_omp_parse, _omp_links),
            "math": (_omp_parse,),
            "timezone": (_omp_parse,),
            "_KNOWN_MESSAGE_ROLES": (_omp_facts,),
            "_TOKEN_FIELDS": (_omp_facts,),
            "_TaskFacts": (_omp_facts, _omp_links),
            "_UsageFacts": (_omp_facts,),
            "_message": (_omp_facts, _omp_links),
            "_SpawnFacts": (_omp_links,),
            "_entity_id": (_omp_links,),
            "_milliseconds_timestamp": (_omp_links,),
            "_session_init_signature": (_omp_links,),
            "_terminal_end": (_omp_links,),
        },
    ) -> None:
        for target in _targets.get(name, ()):
            setattr(target, name, value)
        super().__setattr__(name, value)


_omp_module = _sys.modules[__name__]
_omp_module.__class__ = _OMPModule
del _OMPModule, _ModuleType, _omp_facts, _omp_links, _omp_module
del _omp_parse, _omp_types, _sys
