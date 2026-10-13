"""Session parsing, selection, and usage helpers for the Pi adapter."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .base import Selection


_USAGE_ENTRY_TYPES = {"compaction", "branch_summary"}
_SUPPORTED_SESSION_VERSIONS = {1, 2, 3}


@dataclass(frozen=True)
class _Session:
    path: Path
    session_id: str
    workspace: str | None
    started_at: str | None
    started_dt: datetime | None
    entries: tuple[dict[str, Any], ...]
    malformed_records: int


def _jsonl_paths(stores: Iterable[Path]) -> tuple[Path, ...]:
    found: dict[str, Path] = {}
    for raw_store in stores:
        store = raw_store.expanduser()
        candidates: Iterable[Path]
        if store.is_file():
            candidates = (store,) if store.suffix == ".jsonl" else ()
        elif store.is_dir():
            candidates = store.rglob("*.jsonl")
        else:
            candidates = ()
        for path in candidates:
            key = str(path.resolve(strict=False))
            found.setdefault(key, path)
    return tuple(found[key] for key in sorted(found))


def _parse_timestamp(value: Any) -> tuple[datetime | None, str | None]:
    if not isinstance(value, str):
        return None, None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None, None
    if parsed.tzinfo is None:
        return None, None
    normalized = parsed.isoformat(timespec="milliseconds")
    if parsed.utcoffset() == timezone.utc.utcoffset(parsed):
        normalized = normalized.replace("+00:00", "Z")
    return parsed, normalized


def _bound(value: str | None, label: str) -> datetime | None:
    if value is None:
        return None
    parsed, _ = _parse_timestamp(value)
    if parsed is None:
        raise ValueError(f"Pi adapter {label} must be an RFC 3339 timestamp with an offset")
    return parsed


def _read_session(path: Path) -> tuple[_Session | None, str | None]:
    header: Mapping[str, Any] | None = None
    entries: list[dict[str, Any]] = []
    malformed = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    malformed += 1
                    continue
                if not isinstance(value, dict):
                    malformed += 1
                    continue
                if value.get("type") == "session" and header is None:
                    header = value
                elif value.get("type") != "session":
                    entries.append(value)
    except (OSError, UnicodeDecodeError):
        return None, "unreadable_or_missing_id"

    if header is None:
        return None, "unreadable_or_missing_id"
    version = header.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _SUPPORTED_SESSION_VERSIONS
    ):
        return None, "unsupported_or_missing_version"
    if not isinstance(header.get("id"), str) or not header["id"]:
        return None, "unreadable_or_missing_id"
    started_dt, started_at = _parse_timestamp(header.get("timestamp"))
    workspace = header.get("cwd")
    return (
        _Session(
            path=path,
            session_id=header["id"],
            workspace=workspace if isinstance(workspace, str) and workspace else None,
            started_at=started_at,
            started_dt=started_dt,
            entries=tuple(entries),
            malformed_records=malformed,
        ),
        None,
    )


def _session_sort_key(session: _Session) -> tuple[datetime, str, str]:
    latest = datetime.max.replace(tzinfo=timezone.utc)
    return (session.started_dt or latest, session.session_id, str(session.path))


def _stable_id(prefix: str, value: str) -> str:
    direct = prefix + value
    if len(direct) <= 200:
        return direct
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
    return prefix + digest


def _session_locator(value: str) -> str:
    if len(value) <= 500:
        return value
    return "sha256:" + hashlib.sha256(
        value.encode("utf-8", errors="replace")
    ).hexdigest()


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _nonnegative_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value < 0 or not math.isfinite(float(value)):
        return None
    return value


def _entry_state(
    entry: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[str | None, str | None, str | None]:
    """Return configured provider, model, and effort on ``entry``'s branch."""
    provider = model = effort = None
    parent_id = entry.get("parentId")
    seen: set[str] = set()
    while isinstance(parent_id, str) and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            break
        if parent.get("type") == "model_change" and model is None:
            raw_provider = parent.get("provider")
            raw_model = parent.get("modelId")
            provider = raw_provider if isinstance(raw_provider, str) else None
            model = raw_model if isinstance(raw_model, str) else None
        elif parent.get("type") == "thinking_level_change" and effort is None:
            raw_effort = parent.get("thinkingLevel")
            effort = raw_effort if isinstance(raw_effort, str) else None
        if model is not None and effort is not None:
            break
        parent_id = parent.get("parentId")
    return provider, model, effort


def _usage_source(
    entry: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], Mapping[str, Any]] | None:
    typ = entry.get("type")
    if typ == "message" and isinstance(entry.get("message"), dict):
        message = entry["message"]
        if isinstance(message.get("usage"), dict):
            role = message.get("role")
            if role == "assistant":
                return "assistant_message", message, message["usage"]
            if role == "toolResult":
                return "tool_result", message, message["usage"]
    if typ in _USAGE_ENTRY_TYPES and isinstance(entry.get("usage"), dict):
        return str(typ), entry, entry["usage"]
    return None


def _usage_records(session: _Session) -> tuple[list[dict[str, Any]], int]:
    by_id = {
        entry["id"]: entry
        for entry in session.entries
        if isinstance(entry.get("id"), str)
    }
    records: list[dict[str, Any]] = []
    incomplete = 0
    for entry in session.entries:
        source = _usage_source(entry)
        if source is None:
            continue
        kind, carrier, usage = source
        configured_provider, configured_model, effort = _entry_state(entry, by_id)

        raw_provider = carrier.get("provider")
        requested_model = carrier.get("model")
        response_model = carrier.get("responseModel")
        provider = raw_provider if isinstance(raw_provider, str) else configured_provider
        if kind == "tool_result":
            # Tool usage does not identify its own provider, model, or effort.
            # The surrounding conversation state is not evidence about the
            # implementation of the tool, so leave all three unknown.
            provider = model = model_tier = effort = None
        elif isinstance(response_model, str) and response_model:
            model = response_model
            model_tier = "verified"
        elif isinstance(requested_model, str) and requested_model:
            model = requested_model
            model_tier = "verified"
        else:
            model = configured_model
            model_tier = "reported" if configured_model is not None else None

        entry_id = entry.get("id")
        _, at = _parse_timestamp(entry.get("timestamp"))
        record: dict[str, Any] = {"kind": kind}
        if isinstance(entry_id, str):
            record["id"] = entry_id
        if at is not None:
            record["at"] = at
        if provider is not None:
            record["provider"] = provider
        if model is not None:
            record["model"] = model
        if model_tier is not None:
            record["model_tier"] = model_tier
        if effort is not None:
            record["effort"] = effort
            record["effort_tier"] = "reported"

        token_fields = {
            "input": "input",
            "cache_read": "cacheRead",
            "cache_write": "cacheWrite",
            "cache_write_1h": "cacheWrite1h",
            "output": "output",
            "reasoning": "reasoning",
            "total": "totalTokens",
        }
        tokens = {
            target: value
            for target, source_key in token_fields.items()
            if (value := _nonnegative_int(usage.get(source_key))) is not None
        }
        if tokens:
            record["tokens"] = tokens

        raw_cost = usage.get("cost")
        costs: dict[str, int | float] = {}
        if isinstance(raw_cost, dict):
            for target, source_key in (
                ("input", "input"),
                ("cache_read", "cacheRead"),
                ("cache_write", "cacheWrite"),
                ("output", "output"),
                ("total", "total"),
            ):
                value = _nonnegative_number(raw_cost.get(source_key))
                if value is not None:
                    costs[target] = value
        if costs:
            record["usd"] = costs

        required_tokens = {
            "input",
            "cache_read",
            "cache_write",
            "output",
            "reasoning",
        }
        if not required_tokens.issubset(tokens) or "total" not in costs:
            incomplete += 1
        records.append(record)
    return records, incomplete


def _sum_if_complete(
    records: Sequence[Mapping[str, Any]], block: str, member: str
) -> int | float | None:
    values: list[int | float] = []
    for record in records:
        nested = record.get(block)
        if not isinstance(nested, dict) or member not in nested:
            return None
        value = nested[member]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(value)
    return sum(values) if values else None


def _cost(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    cost: dict[str, Any] = {"requests": len(records), "basis": "measured"}
    for source, target in (
        ("input", "input_tokens"),
        ("cache_read", "cached_input_tokens"),
        ("cache_write", "cache_creation_tokens"),
        ("output", "output_tokens"),
        ("reasoning", "reasoning_tokens"),
    ):
        value = _sum_if_complete(records, "tokens", source)
        if isinstance(value, int):
            cost[target] = value

    one_hour = 0
    five_minute = 0
    buckets_complete = True
    for record in records:
        tokens = record.get("tokens")
        if not isinstance(tokens, dict):
            buckets_complete = False
            break
        cache_write = tokens.get("cache_write")
        cache_write_1h = tokens.get("cache_write_1h")
        if (
            not isinstance(cache_write, int)
            or isinstance(cache_write, bool)
            or not isinstance(cache_write_1h, int)
            or isinstance(cache_write_1h, bool)
            or cache_write_1h > cache_write
        ):
            buckets_complete = False
            break
        one_hour += cache_write_1h
        five_minute += cache_write - cache_write_1h
    if buckets_complete:
        cost["cache_creation_1h_tokens"] = one_hour
        cost["cache_creation_5m_tokens"] = five_minute

    usd = _sum_if_complete(records, "usd", "total")
    if isinstance(usd, (int, float)):
        cost["usd"] = usd
    return cost


def _uniform_attempt_model(records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    pairs = [(record.get("provider"), record.get("model")) for record in records]
    if not pairs or any(
        not isinstance(provider, str) or not isinstance(model, str)
        for provider, model in pairs
    ):
        return None
    if len(set(pairs)) != 1:
        return None
    provider, model = pairs[0]
    tiers = {record.get("model_tier") for record in records}
    if not tiers.issubset({"verified", "reported"}) or None in tiers:
        return None
    return {
        "raw": model,
        "id": model,
        "provider": provider,
        "tier": "reported" if "reported" in tiers else "verified",
    }


def _uniform_effort(records: Sequence[Mapping[str, Any]]) -> str | None:
    efforts = {record.get("effort") for record in records}
    if len(efforts) == 1:
        effort = next(iter(efforts))
        return effort if isinstance(effort, str) else None
    return None


def _selection_stats() -> dict[str, Any]:
    return {
        "files_seen": 0,
        "sessions_parsed": 0,
        "sessions_selected": 0,
        "retained_unknown_workspace": 0,
        "retained_unknown_timestamp": 0,
        "omitted": {
            "unreadable_or_missing_id": 0,
            "unsupported_or_missing_version": 0,
            "duplicate_session_id": 0,
            "id_filter": 0,
            "workspace_filter": 0,
            "time_filter": 0,
            "limit": 0,
        },
    }


def _load_selected(
    selection: Selection, defaults: Sequence[Path]
) -> tuple[list[_Session], dict[str, Any]]:
    if selection.limit is not None and (
        not isinstance(selection.limit, int)
        or isinstance(selection.limit, bool)
        or selection.limit < 1
    ):
        raise ValueError("Pi adapter --limit must be at least 1")
    since = _bound(selection.since, "--since")
    until = _bound(selection.until, "--until")
    if since is not None and until is not None and since > until:
        raise ValueError("Pi adapter --since must not be later than --until")

    paths = _jsonl_paths(selection.stores or defaults)
    stats = _selection_stats()
    stats["files_seen"] = len(paths)
    parsed: list[_Session] = []
    for path in paths:
        session, omission = _read_session(path)
        if session is None:
            assert omission is not None
            stats["omitted"][omission] += 1
        else:
            parsed.append(session)
    stats["sessions_parsed"] = len(parsed)

    selected: list[_Session] = []
    seen_ids: set[str] = set()
    wanted_ids = set(selection.session_ids)
    wanted_workspaces = set(selection.workspaces)
    for session in sorted(parsed, key=_session_sort_key):
        if session.session_id in seen_ids:
            stats["omitted"]["duplicate_session_id"] += 1
            continue
        seen_ids.add(session.session_id)
        if wanted_ids and session.session_id not in wanted_ids:
            stats["omitted"]["id_filter"] += 1
            continue
        if wanted_workspaces:
            if session.workspace is None:
                stats["retained_unknown_workspace"] += 1
            elif session.workspace not in wanted_workspaces:
                stats["omitted"]["workspace_filter"] += 1
                continue
        if since is not None or until is not None:
            if session.started_dt is None:
                stats["retained_unknown_timestamp"] += 1
            elif (since is not None and session.started_dt < since) or (
                until is not None and session.started_dt > until
            ):
                stats["omitted"]["time_filter"] += 1
                continue
        selected.append(session)

    if selection.limit is not None and len(selected) > selection.limit:
        stats["omitted"]["limit"] = len(selected) - selection.limit
        selected = selected[: selection.limit]
    stats["sessions_selected"] = len(selected)
    return selected, stats
