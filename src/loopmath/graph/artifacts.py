"""Compatibility facade for artifact recovery and graph joining.

Classification, read-only git recovery, and event joining are separate cohesive
implementations. This module retains every established public graph API.
"""

from __future__ import annotations

import subprocess
from typing import Callable

from .artifact_git import (
    GIT_LOG_ARGS as _GIT_LOG_ARGS,
    GIT_REASONS,
    GIT_RULE_COMMIT_CALL,
    GIT_RULE_CONTAINS,
    git_fallback_writes as _git_fallback_writes,
    git_from_disk as _git_from_disk,
    parse_git_log,
)
from .artifact_join import build_artifacts as _build_artifacts
from .artifact_kinds import KINDS, artifact_kind, heredoc_hints, hint_for_write, kind_from_hint
from .schema import Artifact, GraphEdge, GraphNode

# Kept mutable at this path for existing callers and test doubles.
GIT_EXE = "git"
GIT_LOG_ARGS = _GIT_LOG_ARGS


def git_from_disk(cwd: str) -> tuple[str, str] | str:
    """Read a worktree's log, using compatibility-configurable git settings."""
    return _git_from_disk(cwd, git_exe=GIT_EXE, git_log_args=GIT_LOG_ARGS)


def git_fallback_writes(
    scans: dict[str, dict],
    nodes: dict[str, GraphNode],
    *,
    git: Callable[[str], tuple[str, str] | str] | None = None,
    meta: dict | None = None,
) -> dict[str, list[dict]]:
    """Return git-recovered writes while preserving the original call surface."""
    return _git_fallback_writes(scans, nodes, git=git or git_from_disk, meta=meta)


def build_artifacts(
    scans: dict[str, dict],
    nodes: dict[str, GraphNode],
    *,
    meta: dict | None = None,
    git: Callable[[str], tuple[str, str] | str] | None = None,
):
    """Build artifacts and edges while preserving the original call surface."""
    return _build_artifacts(scans, nodes, meta=meta, git=git or git_from_disk)


__all__ = (
    "GIT_EXE", "GIT_LOG_ARGS", "GIT_REASONS", "GIT_RULE_COMMIT_CALL",
    "GIT_RULE_CONTAINS", "KINDS", "Artifact", "GraphEdge", "GraphNode",
    "artifact_kind", "build_artifacts",
    "git_fallback_writes", "git_from_disk", "heredoc_hints", "hint_for_write",
    "kind_from_hint", "parse_git_log",
)
