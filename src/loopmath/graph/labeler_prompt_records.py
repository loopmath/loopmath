"""Typed JSONL, human-text, and timestamp evidence extraction."""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .labeler_prompt_result import EvidenceResult, ReceiptBook, add_receipt, assert_receipt_keys
from .labeler_prompt_validation import _MISSING, _typed_field, _valid_timestamp


_EVIDENCE_TEXT_LIMIT = 1600
_EVIDENCE_TEXT_HALF = 800
_SessionReadSnapshot = tuple[
    tuple[tuple[int, dict], ...],
    tuple[tuple[str, tuple[EvidenceResult[object], ...]], ...],
]
_SESSION_READ_SNAPSHOTS: ContextVar[
    dict[tuple[object, ...], _SessionReadSnapshot] | None
] = ContextVar("labeler_session_read_snapshots", default=None)


@contextmanager
def _session_read_snapshot_scope() -> Iterator[None]:
    """Reuse each positional transcript snapshot for one v4 projection."""
    active = _SESSION_READ_SNAPSHOTS.get()
    if active is not None:
        yield
        return
    token = _SESSION_READ_SNAPSHOTS.set({})
    try:
        yield
    finally:
        _SESSION_READ_SNAPSHOTS.reset(token)


def _materialize_session_read_snapshot(
    snapshot: _SessionReadSnapshot,
) -> tuple[list[tuple[int, dict]], ReceiptBook]:
    records, receipt_rows = copy.deepcopy(snapshot)
    return list(records), {unit: list(receipts) for unit, receipts in receipt_rows}


def _bounded_text(text: str) -> tuple[str, dict]:
    """Return deterministic raw prefix and suffix text plus truncation accounting."""
    original = len(text)
    if original <= _EVIDENCE_TEXT_LIMIT:
        return text, {
            "original_chars": original,
            "shown_chars": original,
            "not_shown_chars": 0,
            "truncated": False,
        }
    value = text[:_EVIDENCE_TEXT_HALF] + text[-_EVIDENCE_TEXT_HALF:]
    return value, {
        "original_chars": original,
        "shown_chars": len(value),
        "truncated": True,
        "not_shown_chars": original - len(value),
    }


def _bounded_text_result(
    text: object, *, field: str, source_key: tuple, book: ReceiptBook
) -> tuple[str | None, dict]:
    """Cap one validated text atom and receipt the cap decision."""
    key = source_key + (field,)
    if not isinstance(text, str):
        add_receipt(
            book,
            EvidenceResult.exclude(
                field=field,
                reason="invalid_type",
                accounting_unit="text_cap",
                source_key=key,
            ),
        )
        return None, {
            "original_chars": None,
            "shown_chars": None,
            "not_shown_chars": None,
            "truncated": None,
            "reason": "invalid_type",
        }
    value, metadata = _bounded_text(text)
    add_receipt(
        book,
        EvidenceResult.include(
            value,
            field=field,
            reason="head_tail_cap" if metadata["truncated"] else "within_limit",
            accounting_unit="text_cap",
            source_key=key,
        ),
    )
    return value, metadata


def _locator_result(item: object, collection_index: int) -> EvidenceResult[Path]:
    key = collection_index
    if not isinstance(item, dict):
        return EvidenceResult.exclude(field="session_path", reason="node_not_object", accounting_unit="session_locator", source_key=key)
    features = item.get("features", _MISSING)
    if features is _MISSING:
        reason = "features_missing"
    elif features is None:
        reason = "features_null"
    elif not isinstance(features, dict):
        reason = "features_invalid_type"
    else:
        skeleton = features.get("skeleton", _MISSING)
        if skeleton is _MISSING:
            reason = "skeleton_missing"
        elif skeleton is None:
            reason = "skeleton_null"
        elif not isinstance(skeleton, dict):
            reason = "skeleton_invalid_type"
        else:
            path = skeleton.get("session_path", _MISSING)
            if path is _MISSING:
                reason = "session_path_missing"
            elif path is None:
                reason = "session_path_null"
            elif not isinstance(path, str):
                reason = "session_path_invalid_type"
            elif not path:
                reason = "session_path_empty"
            else:
                return EvidenceResult.include(
                    Path(path), field="session_path", reason="valid", accounting_unit="session_locator", source_key=key
                )
    return EvidenceResult.exclude(field="session_path", reason=reason, accounting_unit="session_locator", source_key=key)


def _read_session_receipts_uncached(
    locator: EvidenceResult[Path], collection_index: int
) -> tuple[list[tuple[int, dict]], ReceiptBook]:
    """Read one transcript and retain one typed disposition for every input line."""
    book: ReceiptBook = {}
    add_receipt(book, locator)
    line_keys: set[tuple] = set()
    if locator.excluded:
        add_receipt(book, EvidenceResult.exclude(
            field="session_availability", reason=locator.reason,
            accounting_unit="session_availability", source_key=collection_index,
        ))
        assert_receipt_keys(book["session_locator"], "session_locator", {collection_index})
        assert_receipt_keys(book["session_availability"], "session_availability", {collection_index})
        return [], book
    path = locator.value
    assert path is not None
    if not path.is_file():
        add_receipt(book, EvidenceResult.exclude(
            field="session_availability", reason="session_file_missing",
            accounting_unit="session_availability", source_key=collection_index,
        ))
        assert_receipt_keys(book["session_locator"], "session_locator", {collection_index})
        assert_receipt_keys(book["session_availability"], "session_availability", {collection_index})
        return [], book
    records: list[tuple[int, dict]] = []
    read_reason = "session_file_readable"
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                line_key = (collection_index, line_number)
                line_keys.add(line_key)
                if not line.strip():
                    result = EvidenceResult.exclude(
                        field="session_jsonl_line", reason="blank_line",
                        accounting_unit="session_jsonl_line", source_key=line_key,
                    )
                else:
                    try:
                        value = json.loads(line)
                    except ValueError:
                        result = EvidenceResult.exclude(
                            field="session_jsonl_line", reason="invalid_json",
                            accounting_unit="session_jsonl_line", source_key=line_key,
                        )
                    else:
                        if isinstance(value, dict):
                            result = EvidenceResult.include(
                                value, field="session_jsonl_line", reason="parsed_object",
                                accounting_unit="session_jsonl_line", source_key=line_key,
                            )
                            records.append((line_number, value))
                        else:
                            result = EvidenceResult.exclude(
                                field="session_jsonl_line", reason="not_object",
                                accounting_unit="session_jsonl_line", source_key=line_key,
                            )
                add_receipt(book, result)
    except UnicodeError:
        read_reason = (
            "session_file_partial_decode_error"
            if book.get("session_jsonl_line")
            else "session_file_decode_error"
        )
    except OSError:
        read_reason = (
            "session_file_partial_read"
            if book.get("session_jsonl_line")
            else "session_file_unreadable"
        )
    if read_reason == "session_file_readable":
        availability = EvidenceResult.include(
            path, field="session_availability", reason=read_reason,
            accounting_unit="session_availability", source_key=collection_index,
        )
    else:
        availability = EvidenceResult.exclude(
            field="session_availability", reason=read_reason,
            accounting_unit="session_availability", source_key=collection_index,
        )
    add_receipt(book, availability)
    assert_receipt_keys(book["session_locator"], "session_locator", {collection_index})
    assert_receipt_keys(book["session_availability"], "session_availability", {collection_index})
    assert_receipt_keys(book.get("session_jsonl_line", []), "session_jsonl_line", line_keys)
    return records, book


def _read_session_receipts(
    item: object, collection_index: int
) -> tuple[list[tuple[int, dict]], ReceiptBook]:
    """Read or reuse one immutable positional snapshot of a transcript."""
    locator = _locator_result(item, collection_index)
    locator_identity = locator.value if locator.included else locator.reason
    key = (collection_index, locator.included, locator_identity)
    snapshots = _SESSION_READ_SNAPSHOTS.get()
    if snapshots is not None and key in snapshots:
        return _materialize_session_read_snapshot(snapshots[key])

    records, book = _read_session_receipts_uncached(locator, collection_index)
    if snapshots is None:
        return records, book
    snapshot = (
        tuple(records),
        tuple(
            (unit, tuple(receipts))
            for unit, receipts in sorted(book.items())
        ),
    )
    snapshots[key] = snapshot
    return _materialize_session_read_snapshot(snapshot)


def _read_session_records(item: dict) -> tuple[list[dict], Counter]:
    """Compatibility projection of typed transcript receipts into legacy counters."""
    located_records, book = _read_session_receipts(item, 0)
    counts: Counter = Counter()
    for result in book.get("session_jsonl_line", []):
        counts["source_lines"] += 1
        if result.included:
            counts["source_records"] += 1
        else:
            counts[f"line_excluded:{result.reason}"] += 1
    for result in book.get("session_availability", []):
        if result.excluded:
            counts[f"unavailable:{result.reason}"] += 1
    return [record for _line_number, record in located_records], counts


def _content_text_result(
    content: object,
    *,
    field: str,
    source_key: tuple,
    book: ReceiptBook,
) -> EvidenceResult[str]:
    unit = "evidence_field"
    key = source_key + (field,)
    if isinstance(content, str):
        if content.strip():
            result = EvidenceResult.include(content, field=field, reason="text", accounting_unit=unit, source_key=key)
        else:
            result = EvidenceResult.exclude(field=field, reason="empty_text", accounting_unit=unit, source_key=key)
        add_receipt(book, result)
        return result
    if content is _MISSING:
        reason = "missing"
    elif content is None:
        reason = "null"
    elif not isinstance(content, list):
        reason = "invalid_type"
    else:
        parts: list[str] = []
        for block_index, block in enumerate(content):
            block_key = source_key + (field, block_index)
            if not isinstance(block, dict):
                add_receipt(book, EvidenceResult.exclude(
                    field=f"{field}[]", reason="block_not_object",
                    accounting_unit="human_text_block", source_key=block_key,
                ))
                continue
            if block.get("type") not in ("text", "input_text", "output_text"):
                add_receipt(book, EvidenceResult.exclude(
                    field=f"{field}[]", reason="block_not_text",
                    accounting_unit="human_text_block", source_key=block_key,
                ))
                continue
            text_result = _typed_field(
                block, "text", field=f"{field}[].text", expected=str,
                valid=lambda value: bool(value.strip()), accounting_unit=unit,
                source_key=block_key + ("text",),
            )
            add_receipt(book, text_result)
            if text_result.included:
                parts.append(text_result.value)
                add_receipt(book, EvidenceResult.include(
                    text_result.value, field=f"{field}[]", reason="visible_text",
                    accounting_unit="human_text_block", source_key=block_key,
                ))
            else:
                add_receipt(book, EvidenceResult.exclude(
                    field=f"{field}[]", reason=text_result.reason,
                    accounting_unit="human_text_block", source_key=block_key,
                ))
        if parts:
            result = EvidenceResult.include(
                "\n".join(parts), field=field, reason="visible_text", accounting_unit=unit, source_key=key
            )
            add_receipt(book, result)
            return result
        reason = "no_visible_text"
    result = EvidenceResult.exclude(field=field, reason=reason, accounting_unit=unit, source_key=key)
    add_receipt(book, result)
    return result


def _content_text(content: object) -> str | None:
    book: ReceiptBook = {}
    return _content_text_result(content, field="content", source_key=(0,), book=book).value


def _human_span_result(record: object, record_key: tuple, book: ReceiptBook) -> EvidenceResult[tuple[str, str, object]]:
    unit = "human_text"
    if not isinstance(record, dict):
        result = EvidenceResult.exclude(field="human_text", reason="record_not_object", accounting_unit=unit, source_key=record_key)
        add_receipt(book, result)
        return result
    message = record.get("message", _MISSING)
    if isinstance(message, dict):
        role = message.get("role", _MISSING)
        if role not in ("user", "assistant"):
            reason = "message_role_missing" if role is _MISSING else "message_role_invalid"
        elif role == "user" and record.get("isMeta") is True:
            reason = "meta_user_record"
        else:
            text = _content_text_result(
                message.get("content", _MISSING), field="message.content", source_key=record_key, book=book
            )
            if text.included:
                timestamp = _timestamp_result(
                    record.get("timestamp", _MISSING), field="human_text.at",
                    source_key=record_key, book=book,
                    accounting_unit="human_timestamp",
                )
                result = EvidenceResult.include(
                    (role, text.value, timestamp), field="human_text",
                    reason="visible_message_text", accounting_unit=unit, source_key=record_key,
                )
                add_receipt(book, result)
                return result
            reason = f"message_content_{text.reason}"
    elif record.get("type", _MISSING) == "response_item":
        payload = record.get("payload", _MISSING)
        if not isinstance(payload, dict):
            reason = "response_payload_missing" if payload is _MISSING else "response_payload_invalid"
        elif payload.get("type", _MISSING) != "message":
            reason = "response_payload_not_message"
        else:
            role = payload.get("role", _MISSING)
            if role not in ("user", "assistant"):
                reason = "response_role_missing" if role is _MISSING else "response_role_invalid"
            else:
                text = _content_text_result(
                    payload.get("content", _MISSING), field="payload.content", source_key=record_key, book=book
                )
                if text.included:
                    timestamp = _timestamp_result(
                        record.get("timestamp", _MISSING), field="human_text.at",
                        source_key=record_key, book=book,
                        accounting_unit="human_timestamp",
                    )
                    result = EvidenceResult.include(
                        (role, text.value, timestamp), field="human_text",
                        reason="visible_response_text", accounting_unit=unit, source_key=record_key,
                    )
                    add_receipt(book, result)
                    return result
                reason = f"response_content_{text.reason}"
    else:
        reason = "unsupported_record_shape"
    result = EvidenceResult.exclude(field="human_text", reason=reason, accounting_unit=unit, source_key=record_key)
    add_receipt(book, result)
    return result


def _human_span(record: dict) -> tuple[str, str, object] | None:
    book: ReceiptBook = {}
    return _human_span_result(record, (0, 0), book).value


def _timestamp_result(
    value: object,
    *,
    field: str,
    source_key: tuple,
    book: ReceiptBook,
    accounting_unit: str = "evidence_field",
) -> EvidenceResult[str]:
    container = {} if value is _MISSING else {"timestamp": value}
    result = _typed_field(
        container, "timestamp", field=field, expected=str, valid=_valid_timestamp,
        accounting_unit=accounting_unit, source_key=source_key + (field,),
    )
    add_receipt(book, result)
    return result


def _text_span_result(
    text: str,
    *,
    at: object,
    speaker: str,
    field: str,
    source_key: tuple,
    book: ReceiptBook,
) -> dict:
    value, accounting = _bounded_text_result(
        text, field=f"{field}.value", source_key=source_key, book=book
    )
    assert value is not None
    timestamp = (
        at
        if isinstance(at, EvidenceResult)
        else _timestamp_result(at, field=f"{field}.at", source_key=source_key, book=book)
    )
    return {"at": timestamp.value, "speaker": speaker, "value": value, **accounting}


def _text_span(text: str, *, at: object, speaker: str) -> dict:
    book: ReceiptBook = {}
    return _text_span_result(text, at=at, speaker=speaker, field="text", source_key=(0,), book=book)
