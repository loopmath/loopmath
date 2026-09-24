"""Read OpenCode export and SQLite session stores without mutation."""

from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
import subprocess
from typing import Any, Mapping, Sequence


_TOKEN_FIELDS = {
    "input_tokens": ("input",),
    "cached_input_tokens": ("cache", "read"),
    "cache_creation_tokens": ("cache", "write"),
    "output_tokens": ("output",),
    "reasoning_tokens": ("reasoning",),
}


@dataclass(frozen=True)
class _SessionSource:
    info: Mapping[str, Any]
    messages: tuple[Mapping[str, Any], ...] | None
    database: Path | None = None
    source_format: str = "export"


def _default_database() -> Path:
    return Path.home() / ".local" / "share" / "opencode" / "opencode.db"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return _as_mapping(decoded)


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = (
            float(value) / 1000
            if abs(float(value)) >= 100_000_000_000
            else float(value)
        )
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    return None


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _parse_bound(value: str | None, label: str) -> datetime | None:
    if value is None:
        return None
    parsed = _timestamp(value)
    if parsed is None:
        raise ValueError(f"OpenCode {label} must be an RFC 3339 timestamp with an offset")
    return parsed


def _short(value: Any, maximum: int) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if len(value) <= maximum:
        return value
    digest = sha256(value.encode("utf-8")).hexdigest()[:16]
    return value[: maximum - 18] + ":" + digest


def _entity_id(prefix: str, source_id: str) -> str:
    candidate = f"{prefix}{source_id}"
    if len(candidate) <= 200:
        return candidate
    return f"{prefix}{sha256(source_id.encode('utf-8')).hexdigest()}"


def _nested_nonnegative_int(value: Any, path: tuple[str, ...]) -> int | None:
    current = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        return None
    return current


def _nonnegative_number(value: Any) -> int | float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return value


def _read_export(path: Path) -> _SessionSource:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read OpenCode export {path}: {exc}") from exc
    return _decode_export(raw, str(path))


def _decode_export(raw: Any, source: str) -> _SessionSource:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("info"), Mapping):
        raise ValueError(f"OpenCode export {source} has no object-valued info record")
    messages = raw.get("messages")
    if not isinstance(messages, list) or any(
        not isinstance(item, Mapping) or not isinstance(item.get("info"), Mapping)
        for item in messages
    ):
        raise ValueError(f"OpenCode export {source} has no well-formed messages array")
    session_id = raw["info"].get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(f"OpenCode export {source} has no session id")
    return _SessionSource(dict(raw["info"]), tuple(messages))


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _database_sessions(path: Path) -> list[_SessionSource]:
    try:
        with closing(_connect_read_only(path)) as connection:
            rows = connection.execute(
                """
                SELECT id, directory, parent_id, version, cost, model,
                       time_created, time_updated, time_archived
                FROM session
                ORDER BY id
                """
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError(f"cannot inventory OpenCode database {path}: {exc}") from exc

    sessions = []
    for (
        session_id,
        directory,
        parent_id,
        version,
        cost,
        model,
        created,
        updated,
        archived,
    ) in rows:
        info: dict[str, Any] = {
            "id": session_id,
            "directory": directory,
            "version": version,
            "cost": cost,
            "model": dict(_json_object(model)),
            "time": {"created": created, "updated": updated},
        }
        if parent_id is not None:
            info["parentID"] = parent_id
        if archived is not None:
            info["time"]["archived"] = archived
        sessions.append(_SessionSource(info, None, path, "database"))
    return sessions


def _messages_from_database(session: _SessionSource) -> _SessionSource:
    assert session.database is not None
    session_id = session.info["id"]
    try:
        with closing(_connect_read_only(session.database)) as connection:
            rows = connection.execute(
                """
                SELECT id, data
                FROM message
                WHERE session_id = ?
                ORDER BY time_created, id
                """,
                (session_id,),
            ).fetchall()
    except sqlite3.Error as exc:
        raise ValueError(
            f"cannot read session {session_id!r} from OpenCode database {session.database}: {exc}"
        ) from exc

    messages: list[Mapping[str, Any]] = []
    for message_id, data in rows:
        try:
            decoded = json.loads(data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"OpenCode message {message_id!r} in {session.database} is not valid JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"OpenCode message {message_id!r} in {session.database} is not a JSON object"
            )
        info = dict(decoded)
        info.setdefault("id", message_id)
        info.setdefault("sessionID", session_id)
        messages.append({"info": info, "parts": []})
    return _SessionSource(dict(session.info), tuple(messages), source_format="database")


def _export_session(session: _SessionSource) -> _SessionSource:
    session_id = str(session.info["id"])
    try:
        completed = subprocess.run(
            ["opencode", "export", "--pure", "--sanitize", session_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "the opencode executable is required to read the discovered default store"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"opencode export timed out for session {session_id!r}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise RuntimeError(f"opencode export failed for session {session_id!r}{suffix}")
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"opencode export returned invalid JSON for session {session_id!r}"
        ) from exc
    exported = _decode_export(raw, f"session {session_id!r}")
    if exported.info["id"] != session_id:
        raise RuntimeError(
            f"opencode export returned session {exported.info['id']!r} for requested {session_id!r}"
        )

    # --sanitize intentionally redacts directory text. Directory is a core
    # OCP workspace locator, so restore only this metadata field from the
    # read-only inventory row. No prompt, response, tool, title or diff text
    # from the export is ever emitted.
    info = dict(exported.info)
    if isinstance(session.info.get("directory"), str):
        info["directory"] = session.info["directory"]
    return _SessionSource(info, exported.messages, source_format="export")


def _store_sessions(path: Path) -> list[_SessionSource]:
    if path.is_file() and path.suffix.lower() == ".json":
        return [_read_export(path)]
    database = path / "opencode.db" if path.is_dir() else path
    if database.is_file() and (
        database.name == "opencode.db"
        or database.suffix.lower() in {".db", ".sqlite", ".sqlite3"}
    ):
        return _database_sessions(database)
    if path.is_dir():
        return [_read_export(candidate) for candidate in sorted(path.glob("*.json"))]
    raise ValueError(f"OpenCode store does not exist or is unsupported: {path}")


def _assistant_infos(session: _SessionSource) -> list[Mapping[str, Any]]:
    result = []
    for message in session.messages or ():
        info = _as_mapping(message.get("info"))
        if info.get("role") == "assistant":
            result.append(info)
    return result


def _cost_record(
    assistants: Sequence[Mapping[str, Any]], info: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    cost: dict[str, Any] = {"requests": len(assistants), "basis": "measured"}
    incomplete: dict[str, int] = {}
    for output_name, source_path in _TOKEN_FIELDS.items():
        values = [
            _nested_nonnegative_int(item.get("tokens"), source_path)
            for item in assistants
        ]
        if all(value is not None for value in values):
            cost[output_name] = sum(value for value in values if value is not None)
        else:
            incomplete[output_name] = sum(value is None for value in values)
    usd = _nonnegative_number(info.get("cost"))
    if usd is None:
        raise ValueError(
            f"OpenCode session {info.get('id')!r} has no valid nonnegative cost"
        )
    cost["usd"] = usd
    return cost, incomplete


def _message_models(
    assistants: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in assistants:
        provider = item.get("providerID")
        model = item.get("modelID")
        if isinstance(provider, str) and provider and isinstance(model, str) and model:
            grouped[(provider, model)].append(item)

    records = []
    for (provider, model), messages in sorted(grouped.items()):
        record: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "variants": sorted(
                {
                    variant
                    for message in messages
                    for variant in (message.get("variant"), message.get("mode"))
                    if isinstance(variant, str) and variant
                }
            ),
            "requests": len(messages),
        }
        for output_name, source_path in _TOKEN_FIELDS.items():
            values = [
                _nested_nonnegative_int(message.get("tokens"), source_path)
                for message in messages
            ]
            if all(value is not None for value in values):
                record[output_name] = sum(
                    value for value in values if value is not None
                )
        costs = [_nonnegative_number(message.get("cost")) for message in messages]
        if all(value is not None for value in costs):
            record["usd"] = sum(value for value in costs if value is not None)
        records.append(record)
    return records
