"""Shared data models and constants for the OMP adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .base import Usage


_EXT_NAMESPACE = "dev.dagr.adapter.omp"
_KNOWN_MESSAGE_ROLES = {"assistant", "system", "toolResult", "user"}
_TOKEN_FIELDS = {
    "input": "input_tokens",
    "cacheRead": "cached_input_tokens",
    "cacheWrite": "cache_creation_tokens",
    "output": "output_tokens",
}


@dataclass(frozen=True)
class _TaskJob:
    job_id: str
    task: str
    agent: str
    called_at: datetime | None


@dataclass(frozen=True)
class _TaskFacts:
    calls: int
    linked_results: int
    linked_jobs: int
    task_jobs: int
    jobs: tuple[_TaskJob, ...]
    findings: Mapping[str, int]


@dataclass(frozen=True)
class _UsageGroup:
    provider: str | None
    model: str | None
    calls: int
    complete_calls: int
    unknown_calls: int
    usage: Usage | None
    known_tokens: Mapping[str, int]
    unknown_reasons: Mapping[str, int]


@dataclass(frozen=True)
class _UsageFacts:
    calls: int
    complete_calls: int
    unknown_calls: int
    usage: Usage | None
    groups: tuple[_UsageGroup, ...]
    findings: Mapping[str, int]


@dataclass(frozen=True)
class _NativeSession:
    path: Path
    session_id: str
    workspace: str | None
    format_version: int
    started_at: datetime | None
    records: tuple[Mapping[str, Any], ...]
    session_init: Mapping[str, Any] | None


@dataclass(frozen=True)
class _LoadResult:
    sessions: tuple[_NativeSession, ...]
    candidates: int
    findings: Mapping[tuple[str, str], int]


@dataclass(frozen=True)
class _SpawnFacts:
    edges: tuple[Mapping[str, str], ...]
    child_sessions: int
    matched_children: int
    zero_candidate_children: int
    ambiguous_children: int
    multiple_job_children: int
    missing_signature_children: int
