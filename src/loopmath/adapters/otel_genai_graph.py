"""Trace selection and graph relations for the OTLP/JSON GenAI adapter."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Sequence

from .base import Selection
from .otel_genai_types import (
    _AGENT_KEYS,
    _PARENT_AGENT_KEYS,
    _RFC3339,
    _SESSION_KEYS,
    _WORKSPACE_KEYS,
    _NodeKey,
    _NodeSource,
    _Omissions,
    _Span,
    _Trace,
    _alias_string,
    _bounded_source_string,
    _harness,
    _has_alias,
)


def _traces(spans: Sequence[_Span], omissions: _Omissions) -> list[_Trace]:
    grouped: dict[str, list[_Span]] = {}
    for span in spans:
        grouped.setdefault(span.trace_id, []).append(span)
    traces = []
    for trace_id, members in grouped.items():
        ordered = tuple(sorted(members, key=lambda span: (span.start_ns, span.span_id)))
        aliases = tuple(
            sorted(
                {
                    alias
                    for span in ordered
                    if (alias := _alias_string(span, _SESSION_KEYS, omissions))
                    is not None
                }
            )
        )
        workspaces = tuple(
            sorted(
                {
                    workspace
                    for span in ordered
                    if (
                        workspace := _alias_string(span, _WORKSPACE_KEYS, omissions)
                    )
                    is not None
                }
            )
        )
        traces.append(
            _Trace(
                trace_id=trace_id,
                spans=ordered,
                aliases=aliases,
                workspaces=workspaces,
                start_ns=min(span.start_ns for span in ordered),
                end_ns=max(span.end_ns for span in ordered),
            )
        )
    return sorted(traces, key=lambda trace: (trace.start_ns, trace.trace_id))


def _bound_ns(value: str | None, label: str) -> int | None:
    if value is None:
        return None
    match = _RFC3339.fullmatch(value)
    if match is None:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?", value):
            raise ValueError(f"--{label} must include a timezone: {value!r}")
        raise ValueError(f"--{label} must be an RFC3339 timestamp: {value!r}")
    year, month, day, hour, minute, second = map(int, match.groups()[:6])
    fraction = match.group(7) or ""
    if len(fraction) > 9:
        if any(digit != "0" for digit in fraction[9:]):
            raise ValueError(
                f"--{label} has precision finer than one nanosecond: {value!r}"
            )
        fraction = fraction[:9]
    fraction_ns = int(fraction.ljust(9, "0") or "0")
    zone = match.group(8)
    offset_minutes = 0
    if zone != "Z":
        sign = 1 if zone[0] == "+" else -1
        offset_hour, offset_minute = map(int, zone[1:].split(":"))
        if offset_hour > 23 or offset_minute > 59:
            raise ValueError(f"--{label} must be an RFC3339 timestamp: {value!r}")
        offset_minutes = sign * (offset_hour * 60 + offset_minute)
    try:
        local = datetime(year, month, day, hour, minute, second)
        utc = local - timedelta(minutes=offset_minutes)
    except (OverflowError, ValueError):
        raise ValueError(f"--{label} must be an RFC3339 timestamp: {value!r}") from None
    epoch = datetime(1970, 1, 1)
    delta = utc - epoch
    whole_seconds = delta.days * 86_400 + delta.seconds
    return whole_seconds * 1_000_000_000 + fraction_ns


def _select_traces(
    traces: Sequence[_Trace], selection: Selection, omissions: _Omissions
) -> list[_Trace]:
    since_ns = _bound_ns(selection.since, "since")
    until_ns = _bound_ns(selection.until, "until")
    session_ids = set(selection.session_ids)
    workspaces = set(selection.workspaces)
    selected = []
    for trace in traces:
        identities = {trace.trace_id, *trace.aliases}
        keep = not session_ids or not identities.isdisjoint(session_ids)
        keep = keep and (
            not workspaces or not set(trace.workspaces).isdisjoint(workspaces)
        )
        keep = keep and (since_ns is None or trace.start_ns >= since_ns)
        keep = keep and (until_ns is None or trace.start_ns <= until_ns)
        if keep:
            selected.append(trace)
        else:
            omissions.add("filtered_trace")
    if selection.limit is not None and len(selected) > selection.limit:
        omissions.add("filtered_trace", len(selected) - selection.limit)
        selected = selected[: selection.limit]
    return selected


def _main_identity(
    span: _Span, harness: str, fallback: str, omissions: _Omissions
) -> str:
    if harness == "codex":
        return _alias_string(
            span,
            ("conversation.id", "session.id", "thread.id", "gen_ai.conversation.id"),
            omissions,
        ) or fallback
    return _alias_string(span, _SESSION_KEYS, omissions) or fallback


def _source_session(source: _NodeSource, omissions: _Omissions) -> str | None:
    keys = (
        ("conversation.id", "session.id", "thread.id", "gen_ai.conversation.id")
        if source.key.harness == "codex"
        else _SESSION_KEYS
    )
    values = {
        value
        for span in source.spans
        if (value := _alias_string(span, keys, omissions)) is not None
    }
    if len(values) > 1:
        omissions.add("conflicting_attribute")
        return None
    value = next(iter(values)) if values else None
    return _bounded_source_string(value, 500, omissions, hash_opaque=True)


def _owners(
    traces: Sequence[_Trace], omissions: _Omissions
) -> tuple[
    dict[_NodeKey, _NodeSource],
    dict[tuple[str, str], _NodeKey],
    dict[tuple[str, str], _Span],
]:
    nodes: dict[_NodeKey, _NodeSource] = {}
    owners: dict[tuple[str, str], _NodeKey] = {}
    all_spans = {
        (span.trace_id, span.span_id): span for trace in traces for span in trace.spans
    }
    harnesses = {key: _harness(span) for key, span in all_spans.items()}

    def resolve(span: _Span, trail: set[tuple[str, str]]) -> _NodeKey | None:
        span_key = span.trace_id, span.span_id
        if span_key in owners:
            return owners[span_key]
        harness = harnesses[span_key]
        if harness is None:
            return None
        has_agent = _has_alias(span, _AGENT_KEYS)
        agent = _alias_string(span, _AGENT_KEYS, omissions)
        if has_agent and agent is None:
            omissions.add("missing_identity")
            return None
        if agent is not None:
            owner = _NodeKey(span.trace_id, harness, "agent", agent)
            owners[span_key] = owner
            return owner
        if span_key not in trail and span.parent_span_id is not None:
            parent_key = span.trace_id, span.parent_span_id
            parent = all_spans.get(parent_key)
            if parent is not None and harnesses.get(parent_key) == harness:
                parent_owner = resolve(parent, {*trail, span_key})
                if parent_owner is not None:
                    owners[span_key] = parent_owner
                    return parent_owner
        owner = _NodeKey(
            span.trace_id,
            harness,
            "session",
            _main_identity(span, harness, span.span_id, omissions),
        )
        owners[span_key] = owner
        return owner

    for trace in traces:
        for span in trace.spans:
            owner = resolve(span, set())
            if owner is None:
                omissions.add("unsupported_signal")
                continue
            nodes.setdefault(owner, _NodeSource(owner, [])).spans.append(span)
    for source in nodes.values():
        source.spans.sort(key=lambda span: (span.start_ns, span.span_id))
    return nodes, owners, all_spans


def _agent_relations(
    nodes: Mapping[_NodeKey, _NodeSource],
    owners: Mapping[tuple[str, str], _NodeKey],
    all_spans: Mapping[tuple[str, str], _Span],
    omissions: _Omissions,
) -> list[tuple[_NodeKey, _NodeKey, str]]:
    agents = {
        (key.trace_id, key.harness, key.identity): key
        for key in nodes
        if key.identity_kind == "agent"
    }
    relations: dict[_NodeKey, tuple[_NodeKey, str]] = {}
    explicit_children: set[_NodeKey] = set()
    for child, source in sorted(nodes.items()):
        explicit_spans = [
            span for span in source.spans if _has_alias(span, _PARENT_AGENT_KEYS)
        ]
        if not explicit_spans:
            continue
        explicit_children.add(child)
        if child.identity_kind != "agent":
            omissions.add("unresolved_parent_agent_id")
            continue
        parent_ids = {
            parent
            for span in explicit_spans
            if (parent := _alias_string(span, _PARENT_AGENT_KEYS, omissions))
            is not None
        }
        if len(parent_ids) != 1:
            omissions.add("unresolved_parent_agent_id")
            continue
        parent = agents.get((child.trace_id, child.harness, next(iter(parent_ids))))
        if parent is None or parent == child:
            omissions.add("unresolved_parent_agent_id")
        else:
            relations[child] = (
                parent,
                "Claude parent_agent_id matched agent_id within one OTLP trace",
            )

    for child, source in sorted(nodes.items()):
        if (
            child.harness != "claude-code"
            or child.identity_kind != "agent"
            or child in explicit_children
        ):
            continue
        candidates: set[_NodeKey] = set()
        for span in source.spans:
            current = span
            visited: set[str] = set()
            while current.parent_span_id and current.parent_span_id not in visited:
                visited.add(current.parent_span_id)
                ancestor = all_spans.get((child.trace_id, current.parent_span_id))
                if ancestor is None:
                    break
                ancestor_owner = owners.get((ancestor.trace_id, ancestor.span_id))
                tool_name = (
                    _alias_string(ancestor, ("tool_name",), omissions) or ""
                ).lower()
                if (
                    ancestor_owner is not None
                    and ancestor_owner != child
                    and ancestor_owner.harness == "claude-code"
                    and ancestor.name == "claude_code.tool"
                    and tool_name in {"agent", "task"}
                ):
                    candidates.add(ancestor_owner)
                    break
                current = ancestor
        if len(candidates) == 1:
            relations[child] = (
                next(iter(candidates)),
                "Claude Agent/Task span ancestry identified the direct subagent",
            )
        elif len(candidates) > 1:
            omissions.add("unresolved_parent_agent_id")

    return sorted(
        [
            (parent, child, evidence)
            for child, (parent, evidence) in relations.items()
        ],
        key=lambda item: (item[0], item[1], item[2]),
    )


def _launch_relations(
    owners: Mapping[tuple[str, str], _NodeKey],
    all_spans: Mapping[tuple[str, str], _Span],
    omissions: _Omissions,
) -> list[tuple[_NodeKey, _NodeKey]]:
    candidates: dict[_NodeKey, set[_NodeKey]] = {}
    roots: dict[_NodeKey, list[tuple[int, str]]] = {}
    for key, span in all_spans.items():
        owner = owners.get(key)
        if owner is None:
            continue
        parent_owner = (
            owners.get((span.trace_id, span.parent_span_id))
            if span.parent_span_id is not None
            else None
        )
        if parent_owner is None or parent_owner.harness != owner.harness:
            roots.setdefault(owner, []).append((span.start_ns, span.span_id))
    canonical_roots = {owner: min(members)[1] for owner, members in roots.items()}
    for key, child in sorted(all_spans.items()):
        child_owner = owners.get(key)
        if child_owner is None or child.parent_span_id is None:
            continue
        parent_key = child.trace_id, child.parent_span_id
        parent_owner = owners.get(parent_key)
        if parent_owner is None:
            omissions.add("unresolved_parent_span_id")
            continue
        if parent_owner.harness == child_owner.harness:
            continue
        if canonical_roots.get(child_owner) != child.span_id:
            omissions.add("unresolved_parent_span_id")
            continue
        parent = all_spans[parent_key]
        same_harness_parent = _harness(parent) == _harness(child)
        if same_harness_parent:
            continue
        if parent.name not in {
            "claude_code.tool.execution",
            "codex.tool.execution",
            "gen_ai.tool.execution",
        }:
            omissions.add("unresolved_parent_span_id")
            continue
        # The cross-harness parent is the root boundary for this harness
        # component. Inner spans with same-harness parents never reach here.
        candidates.setdefault(child_owner, set()).add(parent_owner)
    relations = []
    for child, parents in sorted(candidates.items()):
        if len(parents) == 1:
            relations.append((next(iter(parents)), child))
        else:
            omissions.add("unresolved_parent_span_id")
    return relations


def _paths(stores: Sequence[Path]) -> tuple[Path, ...]:
    if not stores:
        raise ValueError(
            "otel-genai requires at least one explicit --store OTLP JSON file"
        )
    paths = tuple(
        sorted({Path(store).expanduser().resolve() for store in stores}, key=str)
    )
    for path in paths:
        if not path.is_file():
            raise ValueError(f"OTLP JSON store is not a regular file: {path}")
    return paths
