"""Parsing and formatting helpers for the bb adapter."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_PROVIDERS = frozenset(("claude-code", "codex"))

_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    """Open one store without creating it or permitting writes."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"bb store does not exist: {resolved}")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _normalized_uuid(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate.lower() if _UUID_RE.fullmatch(candidate) else None


def _iso_from_ms(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1000.0, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _bound_ms(value: str | None, label: str) -> int | None:
    if value is None:
        return None
    if _RFC3339_RE.fullmatch(value) is None:
        raise ValueError(f"bb {label} must be an RFC3339 timestamp with an offset")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"bb {label} is not a valid RFC3339 timestamp: {value!r}") from exc
    return int(parsed.timestamp() * 1000)


def _workspace_group(workspace: str) -> str:
    digest = hashlib.sha256(workspace.encode("utf-8")).hexdigest()[:20]
    return f"bb-workspace:{digest}"


def _attempt_id(node_id: str) -> str:
    return f"{node_id}.a1"


def _thread_node_id(thread_id: str) -> str:
    return f"bb:{thread_id}"


def _read_store(path: Path) -> tuple[list[dict[str, Any]], list[tuple[str, str | None]]]:
    """Return thread metadata and queued-message endpoints, never their text."""

    with _connect_read_only(path) as connection:
        rows = connection.execute(
            """
            SELECT t.id, t.provider_id, t.model_override,
                   t.reasoning_level_override, t.parent_thread_id,
                   t.source_thread_id, t.origin_kind, t.created_at,
                   t.updated_at, e.path AS workspace
            FROM threads AS t
            LEFT JOIN environments AS e ON e.id = t.environment_id
            ORDER BY t.created_at, t.id
            """
        ).fetchall()
        event_rows = connection.execute(
            """
            SELECT thread_id, provider_thread_id,
                   CASE WHEN json_valid(data)
                        THEN json_extract(data, '$.providerThreadId')
                        ELSE NULL END AS json_provider_thread_id,
                   type
            FROM events
            WHERE provider_thread_id IS NOT NULL
               OR (json_valid(data)
                   AND json_type(data, '$.providerThreadId') IS NOT NULL
                   AND json_type(data, '$.providerThreadId') <> 'null')
               OR type = 'thread/tokenUsage/updated'
            ORDER BY thread_id, sequence
            """
        ).fetchall()
        queued = connection.execute(
            """
            SELECT thread_id, sender_thread_id
            FROM queued_thread_messages
            ORDER BY thread_id, created_at, id
            """
        ).fetchall()

    vendor_ids: dict[str, set[str]] = defaultdict(set)
    invalid_vendor_ids: dict[str, int] = defaultdict(int)
    token_snapshots: dict[str, int] = defaultdict(int)
    for event in event_rows:
        # Current stores populate the dedicated column.  JSON remains the
        # fallback for legacy rows; a disagreeing JSON value is deliberately
        # ignored when the column is present.
        raw_id = event["provider_thread_id"]
        if raw_id is None:
            raw_id = event["json_provider_thread_id"]
        thread_id = str(event["thread_id"])
        if raw_id is not None:
            vendor_id = _normalized_uuid(raw_id)
            if vendor_id is None:
                invalid_vendor_ids[thread_id] += 1
            else:
                vendor_ids[thread_id].add(vendor_id)
        if event["type"] == "thread/tokenUsage/updated":
            token_snapshots[thread_id] += 1

    threads: list[dict[str, Any]] = []
    for row in rows:
        thread_id = str(row["id"])
        ids = sorted(vendor_ids.get(thread_id, ()))
        invalid_ids = invalid_vendor_ids.get(thread_id, 0)
        created_at_ms = int(row["created_at"])
        updated_at_ms = int(row["updated_at"])
        if updated_at_ms < created_at_ms:
            raise ValueError(
                "bb thread updated_at must not be earlier than created_at"
            )
        threads.append(
            {
                "store": path.expanduser().resolve(),
                "id": thread_id,
                "provider": str(row["provider_id"]),
                "model": str(row["model_override"]) if row["model_override"] else None,
                "effort": (
                    str(row["reasoning_level_override"])
                    if row["reasoning_level_override"]
                    else None
                ),
                "parent": str(row["parent_thread_id"]) if row["parent_thread_id"] else None,
                "source": str(row["source_thread_id"]) if row["source_thread_id"] else None,
                "origin_kind": str(row["origin_kind"]) if row["origin_kind"] else None,
                "created_at_ms": created_at_ms,
                "updated_at_ms": updated_at_ms,
                "workspace": str(row["workspace"]) if row["workspace"] else None,
                "vendor_session_id": (
                    ids[0] if len(ids) == 1 and invalid_ids == 0 else None
                ),
                "vendor_session_ids": tuple(ids),
                "vendor_session_id_count": len(ids),
                "invalid_vendor_session_id_count": invalid_ids,
                "ignored_token_snapshots": token_snapshots.get(thread_id, 0),
            }
        )
    queue_endpoints = [
        (str(row["thread_id"]), str(row["sender_thread_id"]) if row["sender_thread_id"] else None)
        for row in queued
    ]
    return threads, queue_endpoints


def _attempt(
    node_id: str,
    *,
    actor: str,
    harness: str,
    started_at: str | None,
    wall_s: float | None,
    model: str | None = None,
    model_tier: str | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    attempt: dict[str, Any] = {
        "id": _attempt_id(node_id),
        "node": node_id,
        "n": 1,
        "actor": actor,
        "harness": harness,
        "cause": {"type": "initial"},
        "status": "settled_unverified",
        "outcome": {
            "result": "settled_unverified",
            "evidence": "heuristic",
            "reason": "the source has no acceptance result",
        },
    }
    if model:
        attempt["model"] = {"raw": model}
        if model_tier:
            attempt["model"]["tier"] = model_tier
    if effort:
        attempt["effort"] = effort
    if started_at is not None:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        start = start.astimezone(timezone.utc)
        attempt["started_at"] = (
            start.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        )
        if wall_s is not None:
            end = datetime.fromtimestamp(start.timestamp() + wall_s, timezone.utc)
            attempt["ended_at"] = (
                end.isoformat(timespec="milliseconds").replace("+00:00", "Z")
            )
    return attempt


def _bb_join_issue(
    thread: dict[str, Any],
    vendor_owners: dict[tuple[str, str], list[str]],
) -> str | None:
    """Classify only correlation failures visible in bb's own store."""

    if thread["invalid_vendor_session_id_count"]:
        return "invalid_bb_correlate"
    if thread["vendor_session_id_count"] == 0:
        return "missing_bb_correlate"
    if thread["vendor_session_id_count"] != 1:
        return "conflicting_bb_correlate"
    if thread["provider"] not in SUPPORTED_PROVIDERS:
        return "unsupported_bb_provider"
    session_id = thread["vendor_session_id"]
    if session_id is None:
        raise AssertionError("one valid bb correlate must be available")
    if len(vendor_owners.get((thread["provider"], session_id), ())) != 1:
        return "duplicate_bb_ownership"
    return None


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
