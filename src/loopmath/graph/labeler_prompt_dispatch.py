"""Typed workflow dispatch fields, child joins, and runtime-id selection."""

from __future__ import annotations

from pathlib import Path

from .labeler_prompt_validation import _MISSING, _typed_field
from .labeler_prompt_records import _bounded_text_result, _timestamp_result
from .labeler_prompt_result import (
    EvidenceResult,
    ReceiptBook,
    add_receipt,
    assert_receipt_keys,
)


def _field_result(
    container: dict,
    key: str,
    *,
    field: str,
    source_key: tuple,
    book: ReceiptBook,
) -> EvidenceResult[str]:
    result = _typed_field(
        container,
        key,
        field=field,
        expected=str,
        valid=bool,
        accounting_unit="evidence_field",
        source_key=source_key + (field,),
    )
    add_receipt(book, result)
    return result


def _dispatch_text_result(
    value: object, *, field: str, source_key: tuple, book: ReceiptBook
) -> tuple[str | None, dict]:
    container = {} if value is _MISSING else {"value": value}
    result = _field_result(
        container, "value", field=field, source_key=source_key, book=book
    )
    if result.included:
        return _bounded_text_result(
            result.value, field=f"{field}.value", source_key=source_key, book=book
        )
    return None, {
        "original_chars": None,
        "shown_chars": None,
        "not_shown_chars": None,
        "truncated": None,
        "reason": result.reason,
    }


def _dispatch_text(value: object) -> tuple[str | None, dict]:
    """Bound valid dispatch text; preserve absent or invalid values as unknown."""
    book: ReceiptBook = {}
    return _dispatch_text_result(
        value, field="dispatch.text", source_key=(0,), book=book
    )


def _child_path_result(
    child: object, *, source_key: tuple, book: ReceiptBook
) -> EvidenceResult[str]:
    field = "dispatch.child_session_path"
    key = source_key + (field,)
    reason = None
    if not isinstance(child, dict):
        reason = "child_unavailable"
    else:
        features = child.get("features", _MISSING)
        if features is _MISSING:
            reason = "child_features_missing"
        elif features is None:
            reason = "child_features_null"
        elif not isinstance(features, dict):
            reason = "child_features_invalid_type"
        else:
            skeleton = features.get("skeleton", _MISSING)
            if skeleton is _MISSING:
                reason = "child_skeleton_missing"
            elif skeleton is None:
                reason = "child_skeleton_null"
            elif not isinstance(skeleton, dict):
                reason = "child_skeleton_invalid_type"
            else:
                result = _typed_field(
                    skeleton,
                    "session_path",
                    field=field,
                    expected=str,
                    valid=bool,
                    accounting_unit="evidence_field",
                    source_key=key,
                )
                add_receipt(book, result)
                return result
    result = EvidenceResult.exclude(
        field=field,
        reason=reason,
        accounting_unit="evidence_field",
        source_key=key,
    )
    add_receipt(book, result)
    return result


def _runtime_child_fields(
    child: object, tool_input: dict, source_key: tuple, book: ReceiptBook
) -> tuple[str | None, str | None]:
    if isinstance(child, dict):
        child_id = _field_result(
            child,
            "id",
            field="dispatch.child_session_id",
            source_key=source_key,
            book=book,
        )
    else:
        child_id = EvidenceResult.exclude(
            field="dispatch.child_session_id",
            reason="child_unavailable",
            accounting_unit="evidence_field",
            source_key=source_key + ("dispatch.child_session_id",),
        )
        add_receipt(book, child_id)
    path = _child_path_result(child, source_key=source_key, book=book)
    resume = _field_result(
        tool_input,
        "resume",
        field="dispatch.resume",
        source_key=source_key,
        book=book,
    )
    path_stem = Path(path.value).stem if path.included else ""
    if path_stem:
        runtime_id = path_stem
        selected = EvidenceResult.include(
            runtime_id,
            field="dispatch.runtime_child_id",
            reason="session_path_stem",
            accounting_unit="runtime_child_selection",
            source_key=source_key,
        )
    elif resume.included:
        runtime_id = resume.value
        selected = EvidenceResult.include(
            runtime_id,
            field="dispatch.runtime_child_id",
            reason="resume",
            accounting_unit="runtime_child_selection",
            source_key=source_key,
        )
    else:
        runtime_id = None
        selected = EvidenceResult.exclude(
            field="dispatch.runtime_child_id",
            reason="no_valid_session_path_or_resume",
            accounting_unit="runtime_child_selection",
            source_key=source_key,
        )
    add_receipt(book, selected)
    return child_id.value, runtime_id


def _message_reason(message: object) -> str:
    if message is _MISSING:
        return "message_missing"
    if message is None:
        return "message_null"
    if not isinstance(message, dict):
        return "message_invalid_type"
    content = message.get("content", _MISSING)
    if content is _MISSING:
        return "message_content_missing"
    if content is None:
        return "message_content_null"
    if not isinstance(content, list):
        return "message_content_invalid_type"
    return "message_content_empty"


def _task_dispatches_with_receipts(
    records: list[tuple[int, dict]],
    parent_id: str,
    children: dict[str, list[dict]],
    collection_index: int,
    book: ReceiptBook,
    *,
    selected_child_keys: set[int] | None = None,
    ambiguous_child_keys: set[int] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    candidate_keys: set[tuple] = set()
    for line_number, record in records:
        record_key = (collection_index, line_number)
        message = record.get("message", _MISSING)
        content = message.get("content", _MISSING) if isinstance(message, dict) else _MISSING
        if not isinstance(content, list) or not content:
            candidate_key = record_key + ("record",)
            candidate_keys.add(candidate_key)
            add_receipt(
                book,
                EvidenceResult.exclude(
                    field="dispatch",
                    reason=_message_reason(message),
                    accounting_unit="dispatch_candidate",
                    source_key=candidate_key,
                ),
            )
            continue
        for block_index, block in enumerate(content):
            block_key = record_key + ("block", block_index)
            candidate_keys.add(block_key)
            if not isinstance(block, dict):
                reason = "block_not_object"
            elif block.get("type", _MISSING) != "tool_use":
                reason = "block_not_tool_use"
            elif block.get("name", _MISSING) not in ("Task", "Agent"):
                reason = "tool_not_task_or_agent"
            else:
                _field_result(
                    block,
                    "name",
                    field="dispatch.tool_name",
                    source_key=block_key,
                    book=book,
                )
                raw_input = block.get("input", _MISSING)
                if isinstance(raw_input, dict):
                    tool_input = raw_input
                    input_result = EvidenceResult.include(
                        raw_input,
                        field="dispatch.input",
                        reason="object",
                        accounting_unit="evidence_field",
                        source_key=block_key + ("dispatch.input",),
                    )
                else:
                    tool_input = {}
                    input_reason = (
                        "missing"
                        if raw_input is _MISSING
                        else "null"
                        if raw_input is None
                        else "invalid_type"
                    )
                    input_result = EvidenceResult.exclude(
                        field="dispatch.input",
                        reason=input_reason,
                        accounting_unit="evidence_field",
                        source_key=block_key + ("dispatch.input",),
                    )
                add_receipt(book, input_result)
                prompt_value, prompt_meta = _dispatch_text_result(
                    tool_input.get("prompt", _MISSING),
                    field="dispatch.prompt",
                    source_key=block_key,
                    book=book,
                )
                description_value, description_meta = _dispatch_text_result(
                    tool_input.get("description", _MISSING),
                    field="dispatch.description",
                    source_key=block_key,
                    book=book,
                )
                tool_use_id = _field_result(
                    block,
                    "id",
                    field="dispatch.tool_use_id",
                    source_key=block_key,
                    book=book,
                )
                matches = children.get(tool_use_id.value, []) if tool_use_id.included else []
                if len(matches) == 1:
                    if selected_child_keys is not None:
                        selected_child_keys.add(matches[0]["node_index"])
                    child = matches[0]["item"]
                    join = EvidenceResult.include(
                        child,
                        field="dispatch.child",
                        reason="verified_tool_use_id",
                        accounting_unit="child_join",
                        source_key=block_key,
                    )
                else:
                    if len(matches) > 1 and ambiguous_child_keys is not None:
                        ambiguous_child_keys.update(
                            match["node_index"] for match in matches
                        )
                    child = None
                    join_reason = (
                        f"tool_use_id_{tool_use_id.reason}"
                        if tool_use_id.excluded
                        else "no_verified_child_match"
                        if not matches
                        else "ambiguous_verified_child_match"
                    )
                    join = EvidenceResult.exclude(
                        field="dispatch.child",
                        reason=join_reason,
                        accounting_unit="child_join",
                        source_key=block_key,
                    )
                add_receipt(book, join)
                child_id, runtime_child_id = _runtime_child_fields(
                    child, tool_input, block_key, book
                )
                parent_session_id = _field_result(
                    {"value": parent_id}, "value",
                    field="dispatch.parent_session_id", source_key=block_key,
                    book=book,
                ).value
                at = _timestamp_result(
                    record.get("timestamp", _MISSING), field="dispatch.at",
                    source_key=block_key, book=book,
                ).value
                row = {
                    "at": at,
                    "parent_session_id": parent_session_id,
                    "child_session_id": child_id,
                    "runtime_child_id": runtime_child_id,
                    "description": description_value,
                    "description_accounting": description_meta,
                    "prompt_excerpt": prompt_value,
                    "prompt_accounting": prompt_meta,
                    "_source_record_key": record_key,
                    "_source_key": block_key,
                }
                rows.append(row)
                add_receipt(
                    book,
                    EvidenceResult.include(
                        row,
                        field="dispatch",
                        reason="task_or_agent_call",
                        accounting_unit="dispatch_candidate",
                        source_key=block_key,
                    ),
                )
                continue
            add_receipt(
                book,
                EvidenceResult.exclude(
                    field="dispatch",
                    reason=reason,
                    accounting_unit="dispatch_candidate",
                    source_key=block_key,
                ),
            )
    assert_receipt_keys(
        book.get("dispatch_candidate", []), "dispatch_candidate", candidate_keys
    )
    dispatch_keys = {row["_source_key"] for row in rows}
    assert_receipt_keys(book.get("child_join", []), "child_join", dispatch_keys)
    assert_receipt_keys(
        book.get("runtime_child_selection", []),
        "runtime_child_selection",
        dispatch_keys,
    )
    return rows


def _task_dispatches(
    records: list[dict], parent_id: str, children: dict[str, dict], collection_index: int
) -> list[dict]:
    book: ReceiptBook = {}
    normalized = {key: [{"item": child}] for key, child in children.items()}
    return _task_dispatches_with_receipts(
        list(enumerate(records, 1)), parent_id, normalized, collection_index, book
    )
