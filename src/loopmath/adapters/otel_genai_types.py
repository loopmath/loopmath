"""Shared types and scalar helpers for the OTLP/JSON GenAI adapter."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping
from urllib.parse import quote


EXT = "dev.dagr.adapter.otel-genai"
_TRACE_ID = re.compile(r"[0-9a-fA-F]{32}")
_SPAN_ID = re.compile(r"[0-9a-fA-F]{16}")
_RFC3339 = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d+))?(Z|[+-]\d{2}:\d{2})$"
)

_SESSION_KEYS = (
    "session.id",
    "conversation.id",
    "gen_ai.conversation.id",
    "thread.id",
)
_WORKSPACE_KEYS = (
    "workspace",
    "process.working_directory",
    "cwd",
)
_AGENT_KEYS = ("agent_id", "gen_ai.agent.id")
_PARENT_AGENT_KEYS = ("parent_agent_id", "gen_ai.agent.parent.id")
_MODEL_KEYS = ("gen_ai.request.model", "model")
_EFFORT_KEYS = (
    "effort",
    "gen_ai.request.reasoning.level",
    "model_reasoning_effort",
    "codex.turn.reasoning_effort",
)


@dataclass(frozen=True)
class _Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_ns: int
    end_ns: int
    attrs: Mapping[str, Any]
    resource: Mapping[str, Any]
    scope_name: str
    status: Mapping[str, Any]

    def attr(self, keys: Iterable[str]) -> Any:
        for key in keys:
            if key in self.attrs:
                return self.attrs[key]
            if key in self.resource:
                return self.resource[key]
        return None


@dataclass(frozen=True)
class _Trace:
    trace_id: str
    spans: tuple[_Span, ...]
    aliases: tuple[str, ...]
    workspaces: tuple[str, ...]
    start_ns: int
    end_ns: int


@dataclass(frozen=True, order=True)
class _NodeKey:
    trace_id: str
    harness: str
    identity_kind: str
    identity: str


@dataclass
class _NodeSource:
    key: _NodeKey
    spans: list[_Span]


class _Omissions:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def add(self, reason: str, count: int = 1) -> None:
        if count > 0:
            self.counts[reason] += count

    def records(self) -> list[dict[str, Any]]:
        return [
            {"reason": reason, "count": count}
            for reason, count in sorted(self.counts.items())
            if count > 0
        ]


def _string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return (
            int(value)
            if math.isfinite(value) and value >= 0 and value.is_integer()
            else None
        )
    if isinstance(value, str):
        if re.fullmatch(r"[0-9]+", value) is None:
            return None
        return int(value)
    return None


def _nonnegative_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def _first_string(span: _Span, keys: Iterable[str]) -> str | None:
    return _string(span.attr(keys))


def _alias_string(
    span: _Span, keys: Iterable[str], omissions: _Omissions
) -> str | None:
    values: set[str] = set()
    for key in keys:
        present = key in span.attrs or key in span.resource
        if not present:
            continue
        value = _string(span.attr((key,)))
        if value is None:
            omissions.add("dropped_source_data")
            continue
        values.add(value)
    if len(values) > 1:
        omissions.add("conflicting_attribute")
        return None
    return next(iter(values)) if values else None


def _has_alias(span: _Span, keys: Iterable[str]) -> bool:
    return any(key in span.attrs or key in span.resource for key in keys)


def _bounded_source_string(
    value: str | None,
    limit: int,
    omissions: _Omissions,
    *,
    hash_opaque: bool = False,
) -> str | None:
    if value is None or len(value) <= limit:
        return value
    omissions.add("oversize_source_string")
    if hash_opaque:
        return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"
    return None


def _harness(span: _Span) -> str | None:
    service = (_first_string(span, ("service.name",)) or "").lower()
    scope = span.scope_name.lower()
    if span.name.startswith("claude_code.") or "claude" in service or "claude" in scope:
        return "claude-code"
    if span.name.startswith("codex.") or "codex" in service or "codex" in scope:
        return "codex"
    return None


def _timestamp(nanoseconds: int) -> str:
    seconds, remainder = divmod(nanoseconds, 1_000_000_000)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    base = stamp.strftime("%Y-%m-%dT%H:%M:%S")
    if remainder % 1_000:
        return f"{base}.{remainder:09d}Z"
    return f"{base}.{remainder // 1_000:06d}Z"


def _node_id(key: _NodeKey, omissions: _Omissions) -> str:
    harness = "claude" if key.harness == "claude-code" else key.harness
    candidate = (
        f"otel:{key.trace_id}:{harness}:{key.identity_kind}:"
        f"{quote(key.identity, safe='-._~')}"
    )
    if len(candidate) <= 196:
        return candidate
    omissions.add("oversize_source_string")
    digest = hashlib.sha256(
        "\x00".join((key.trace_id, key.harness, key.identity_kind, key.identity)).encode()
    ).hexdigest()
    return f"otel:sha256:{digest}"
