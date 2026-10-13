"""Session-store parsing and scalar helpers for the OMP adapter."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .omp_types import _LoadResult, _NativeSession


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def _format_timestamp(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    timespec = "milliseconds" if utc.microsecond else "seconds"
    return utc.isoformat(timespec=timespec).replace("+00:00", "Z")


def _milliseconds_timestamp(value: object) -> datetime | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)):
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None


def _record_timestamp(record: Mapping[str, Any]) -> datetime | None:
    return _parse_timestamp(record.get("timestamp"))


def _message(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = record.get("message")
    return (
        value
        if record.get("type") == "message" and isinstance(value, dict)
        else None
    )


def _nonempty_string(value: object) -> str | None:
    return value if isinstance(value, str) and bool(value) else None


def _strict_json(line: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(line, parse_constant=reject_constant)


def _source_finding(
    findings: Counter[tuple[str, str]], reason: str, effect: str, count: int = 1
) -> None:
    findings[(reason, effect)] += count


def _source_finding_rows(
    findings: Mapping[tuple[str, str], int],
) -> list[dict[str, str | int]]:
    return [
        {"reason": reason, "effect": effect, "count": count}
        for (reason, effect), count in sorted(findings.items())
        if count
    ]


def _read_session(
    path: Path, findings: Counter[tuple[str, str]]
) -> _NativeSession | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        _source_finding(findings, "decode_failure", "file_excluded")
        return None
    except OSError:
        _source_finding(findings, "read_failure", "file_excluded")
        return None

    if len(lines) < 2:
        _source_finding(findings, "bad_header", "file_excluded")
        return None
    try:
        title = _strict_json(lines[0])
        header = _strict_json(lines[1])
    except (json.JSONDecodeError, ValueError):
        _source_finding(findings, "bad_header", "file_excluded")
        return None
    if not (
        isinstance(title, dict)
        and title.get("type") == "title"
        and type(title.get("v")) is int
        and title.get("v") == 1
        and isinstance(header, dict)
        and header.get("type") == "session"
    ):
        _source_finding(findings, "bad_header", "file_excluded")
        return None
    version = header.get("version")
    if type(version) is not int or version != 3:
        _source_finding(findings, "unsupported_version", "file_excluded")
        return None
    session_id = _nonempty_string(header.get("id"))
    if session_id is None:
        _source_finding(findings, "invalid_session_id", "file_excluded")
        return None

    started_at = _parse_timestamp(header.get("timestamp"))
    if started_at is None:
        _source_finding(findings, "invalid_header_timestamp", "field_unknown")
    workspace_value = header.get("cwd")
    workspace = workspace_value if isinstance(workspace_value, str) else None
    if workspace is None:
        _source_finding(findings, "workspace_unknown", "field_unknown")

    records: list[Mapping[str, Any]] = []
    record_ids: set[str] = set()
    for line in lines[2:]:
        if not line.strip():
            _source_finding(findings, "malformed_json", "record_excluded")
            continue
        try:
            value = _strict_json(line)
        except (json.JSONDecodeError, ValueError):
            _source_finding(findings, "malformed_json", "record_excluded")
            continue
        if not isinstance(value, dict):
            _source_finding(findings, "non_object_record", "record_excluded")
            continue
        record_type = _nonempty_string(value.get("type"))
        if record_type is None:
            _source_finding(findings, "malformed_record_type", "record_excluded")
            continue
        record_id = _nonempty_string(value.get("id"))
        if record_id is None:
            _source_finding(findings, "invalid_record_id", "record_excluded")
            continue
        if record_id in record_ids:
            _source_finding(findings, "duplicate_record_id", "record_excluded")
            continue
        record_ids.add(record_id)
        if _record_timestamp(value) is None:
            _source_finding(findings, "invalid_record_timestamp", "field_unknown")
        parent_id = value.get("parentId")
        if parent_id is not None and not isinstance(parent_id, str):
            _source_finding(findings, "malformed_parent_id", "field_unknown")
        records.append(value)

    session_init = next(
        (record for record in records if record.get("type") == "session_init"),
        None,
    )
    return _NativeSession(
        path=path,
        session_id=session_id,
        workspace=workspace,
        format_version=3,
        started_at=started_at,
        records=tuple(records),
        session_init=session_init,
    )


def _session_paths(
    stores: Iterable[Path], findings: Counter[tuple[str, str]]
) -> tuple[Path, ...]:
    paths: dict[str, Path] = {}
    for raw_store in stores:
        candidate = Path(raw_store).expanduser().absolute()
        try:
            if candidate.is_file() and candidate.suffix == ".jsonl":
                discovered = (candidate,)
            elif candidate.is_dir():
                discovered = tuple(
                    path for path in candidate.rglob("*.jsonl") if path.is_file()
                )
            else:
                _source_finding(findings, "store_path_unavailable", "path_omitted")
                continue
        except OSError:
            _source_finding(findings, "store_scan_failure", "path_omitted")
            continue
        for path in discovered:
            key = str(path.absolute())
            if key in paths:
                _source_finding(findings, "duplicate_source_path", "path_omitted")
            else:
                paths[key] = path.absolute()
    return tuple(paths[key] for key in sorted(paths))


def _load_sessions(stores: Iterable[Path]) -> _LoadResult:
    findings: Counter[tuple[str, str]] = Counter()
    paths = _session_paths(stores, findings)
    by_id: dict[str, _NativeSession] = {}
    for path in paths:
        session = _read_session(path, findings)
        if session is None:
            continue
        if session.session_id in by_id:
            _source_finding(findings, "duplicate_session_id", "file_excluded")
            continue
        by_id[session.session_id] = session

    def sort_key(session: _NativeSession) -> tuple[float, str, str]:
        timestamp = (
            session.started_at.timestamp()
            if session.started_at is not None
            else float("inf")
        )
        return timestamp, session.session_id, str(session.path)

    return _LoadResult(
        sessions=tuple(sorted(by_id.values(), key=sort_key)),
        candidates=len(paths),
        findings=dict(findings),
    )


def _entity_id(prefix: str, session_id: str) -> str:
    candidate = f"{prefix}:{session_id}"
    if len(candidate) <= 200:
        return candidate
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _bounded(value: str | None, maximum: int) -> str | None:
    return value if value is not None and len(value) <= maximum else None
