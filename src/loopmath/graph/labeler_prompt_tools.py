"""Typed disposition-tool extraction and output joins for prompt evidence."""

from __future__ import annotations

import json
import re
from collections import Counter

from .labeler_prompt_validation import _MISSING, _typed_field
from .labeler_prompt_records import (
    _bounded_text_result,
    _content_text_result,
    _timestamp_result,
)
from .labeler_prompt_result import EvidenceResult, ReceiptBook, add_receipt, assert_receipt_keys


_DISPOSITION_TOOL_RE = re.compile(
    r"(?:tools/review\.sh|(?:^|[\s/])review(?:er)?(?:[\s./_-]|$)|(?:^|[\s/])gate(?:[\s./_-]|$)|"
    r"(?:^|[\s/])checkpoint(?:[\s./_-]|$)|\bgit\s+(?:(?:-C|--git-dir|--work-tree)\s+\S+\s+)*"
    r"(?:merge|cherry-pick)\b)",
    re.IGNORECASE,
)


def _string_result(
    value: object, *, field: str, source_key: tuple, book: ReceiptBook
) -> EvidenceResult[str]:
    container = {} if value is _MISSING else {"value": value}
    result = _typed_field(
        container, "value", field=field, expected=str, valid=bool,
        accounting_unit="evidence_field", source_key=source_key + (field,),
    )
    add_receipt(book, result)
    return result


def _command_value(value: object, *, source_key: tuple, book: ReceiptBook) -> EvidenceResult[str]:
    field = "tool_events.command"
    key = source_key + (field,)
    reason = None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            if value.strip():
                result = EvidenceResult.include(
                    value, field=field, reason="raw_text", accounting_unit="evidence_field", source_key=key
                )
                add_receipt(book, result)
                return result
            reason = "empty_text"
        else:
            value = decoded
    if isinstance(value, dict):
        value = value.get("cmd", value.get("command", _MISSING))
    if isinstance(value, list):
        if value and all(isinstance(part, str) for part in value):
            value = " ".join(value)
        else:
            reason, value = "invalid_command_list", None
    if isinstance(value, str) and value.strip():
        result = EvidenceResult.include(
            value, field=field, reason="valid", accounting_unit="evidence_field", source_key=key
        )
    else:
        if reason is None:
            if value is _MISSING:
                reason = "missing"
            elif value is None:
                reason = "null"
            elif not isinstance(value, str):
                reason = "invalid_type"
            else:
                reason = "empty_text"
        result = EvidenceResult.exclude(
            field=field, reason=reason, accounting_unit="evidence_field", source_key=key
        )
    add_receipt(book, result)
    return result


def _command_from_arguments(arguments: object) -> str | None:
    book: ReceiptBook = {}
    return _command_value(arguments, source_key=(0,), book=book).value


def _tool_output(
    raw: object,
    call_id: object,
    *,
    record_key: tuple,
    output_key: tuple,
    outputs: dict[str, list[tuple[str, tuple, tuple]]],
    book: ReceiptBook,
) -> None:
    output = _content_text_result(
        raw, field="tool_events.result", source_key=output_key, book=book
    )
    output_call_id = _string_result(
        call_id, field="tool_events.output_call_id", source_key=output_key, book=book
    )
    if output.included and output_call_id.included:
        outputs.setdefault(output_call_id.value, []).append((output.value, record_key, output_key))
        result = EvidenceResult.include(
            output.value, field="tool_output", reason="valid",
            accounting_unit="tool_output_candidate", source_key=output_key,
        )
    else:
        reason = output.reason if output.excluded else f"call_id_{output_call_id.reason}"
        result = EvidenceResult.exclude(
            field="tool_output", reason=reason,
            accounting_unit="tool_output_candidate", source_key=output_key,
        )
    add_receipt(book, result)


def _claude_tools(
    record: dict,
    record_key: tuple,
    calls: list[tuple],
    outputs: dict[str, list[tuple[str, tuple, tuple]]],
    book: ReceiptBook,
) -> tuple[set[tuple], set[tuple]]:
    output_keys: set[tuple] = set()
    block_keys: set[tuple] = set()
    message = record.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), list):
        return output_keys, block_keys
    for block_index, block in enumerate(message["content"]):
        block_key = record_key + ("block", block_index)
        block_keys.add(block_key)
        if not isinstance(block, dict):
            result = EvidenceResult.exclude(
                field="tool_block", reason="block_not_object",
                accounting_unit="tool_block", source_key=block_key,
            )
        elif block.get("type") == "tool_use" and block.get("name") == "Bash":
            result = EvidenceResult.include(
                block, field="tool_block", reason="bash_tool_use",
                accounting_unit="tool_block", source_key=block_key,
            )
            call_id = _string_result(
                block.get("id", _MISSING), field="tool_events.call_id",
                source_key=block_key, book=book,
            )
            tool_name = _string_result(
                block.get("name", _MISSING), field="tool_events.tool",
                source_key=block_key, book=book,
            )
            tool_input = block.get("input", _MISSING)
            command = tool_input.get("command", _MISSING) if isinstance(tool_input, dict) else tool_input
            calls.append((block_key, call_id, tool_name, command, record_key, record))
        elif block.get("type") == "tool_result":
            result = EvidenceResult.include(
                block, field="tool_block", reason="tool_result",
                accounting_unit="tool_block", source_key=block_key,
            )
            output_keys.add(block_key)
            _tool_output(
                block.get("content", _MISSING), block.get("tool_use_id", _MISSING),
                record_key=record_key, output_key=block_key, outputs=outputs, book=book,
            )
        else:
            result = EvidenceResult.exclude(
                field="tool_block", reason="not_supported_tool_block",
                accounting_unit="tool_block", source_key=block_key,
            )
        add_receipt(book, result)
    return output_keys, block_keys


def _codex_tool(
    record: dict,
    record_key: tuple,
    calls: list[tuple],
    outputs: dict[str, list[tuple[str, tuple, tuple]]],
    book: ReceiptBook,
) -> tuple[set[tuple], set[tuple]]:
    if record.get("type") != "response_item":
        return set(), set()
    payload = record.get("payload")
    if not isinstance(payload, dict):
        add_receipt(book, EvidenceResult.exclude(
            field="tool_payload", reason="payload_invalid_type",
            accounting_unit="tool_payload", source_key=record_key,
        ))
        return set(), {record_key}
    payload_key = record_key + ("payload",)
    payload_type = payload.get("type")
    if payload_type in ("function_call", "custom_tool_call"):
        result = EvidenceResult.include(
            payload, field="tool_payload", reason="tool_call",
            accounting_unit="tool_payload", source_key=record_key,
        )
        call_id = _string_result(
            payload.get("call_id", _MISSING), field="tool_events.call_id",
            source_key=payload_key, book=book,
        )
        tool_name = _string_result(
            payload.get("name", _MISSING), field="tool_events.tool",
            source_key=payload_key, book=book,
        )
        raw_command = payload.get("arguments", _MISSING)
        if raw_command is _MISSING:
            raw_command = payload.get("input", _MISSING)
        calls.append((payload_key, call_id, tool_name, raw_command, record_key, record))
        keys: set[tuple] = set()
    elif payload_type in ("function_call_output", "custom_tool_call_output"):
        result = EvidenceResult.include(
            payload, field="tool_payload", reason="tool_output",
            accounting_unit="tool_payload", source_key=record_key,
        )
        _tool_output(
            payload.get("output", _MISSING), payload.get("call_id", _MISSING),
            record_key=record_key, output_key=payload_key, outputs=outputs, book=book,
        )
        keys = {payload_key}
    else:
        result = EvidenceResult.exclude(
            field="tool_payload", reason="not_supported_tool_payload",
            accounting_unit="tool_payload", source_key=record_key,
        )
        keys = set()
    add_receipt(book, result)
    return keys, {record_key}


def _tool_calls_with_receipts(
    records: list[tuple[int, dict]], collection_index: int, book: ReceiptBook
) -> list[dict]:
    calls: list[tuple] = []
    outputs: dict[str, list[tuple[str, tuple, tuple]]] = {}
    output_keys: set[tuple] = set()
    block_keys: set[tuple] = set()
    payload_keys: set[tuple] = set()
    for line_number, record in records:
        record_key = (collection_index, line_number)
        claude_outputs, claude_blocks = _claude_tools(
            record, record_key, calls, outputs, book
        )
        codex_outputs, codex_payloads = _codex_tool(
            record, record_key, calls, outputs, book
        )
        output_keys.update(claude_outputs | codex_outputs)
        block_keys.update(claude_blocks)
        payload_keys.update(codex_payloads)
    call_id_counts = Counter(
        call_id.value for _key, call_id, _tool, _command, _record_key, _record in calls
        if call_id.included
    )
    consumed_outputs: set[tuple] = set()
    output_exclusion_reasons = {key: "unmatched_tool_output" for key in output_keys}
    included: list[dict] = []
    for call_key, call_id, tool_name, raw_command, record_key, record in calls:
        command = _command_value(raw_command, source_key=call_key, book=book)
        if command.excluded or not _DISPOSITION_TOOL_RE.search(command.value):
            reason = f"command_{command.reason}" if command.excluded else "not_disposition_related"
            add_receipt(book, EvidenceResult.exclude(
                field="tool_event", reason=reason,
                accounting_unit="tool_event_candidate", source_key=call_key,
            ))
            continue
        timestamp = _timestamp_result(
            record.get("timestamp", _MISSING),
            field="tool_events.at", source_key=call_key, book=book,
        )
        bounded_command, command_meta = _bounded_text_result(
            command.value, field="tool_events.command.value", source_key=call_key, book=book
        )
        assert bounded_command is not None
        row = {
            "at": timestamp.value,
            "tool": tool_name.value,
            "command": bounded_command,
            "command_accounting": command_meta,
            "_source_record_keys": [record_key],
            "_source_key": call_key,
        }
        matches = outputs.get(call_id.value, []) if call_id.included else []
        if call_id.excluded:
            reason = f"call_id_{call_id.reason}"
        elif call_id_counts[call_id.value] > 1:
            reason = "ambiguous_duplicate_tool_call_id"
        elif not matches:
            reason = "no_matching_tool_output"
        elif len(matches) > 1:
            reason = "multiple_matching_tool_outputs"
        elif matches[0][2] in consumed_outputs:
            reason = "tool_output_already_consumed"
        else:
            reason = None
        if reason in (
            "ambiguous_duplicate_tool_call_id",
            "multiple_matching_tool_outputs",
        ):
            for _raw_output, _record_key, output_key in matches:
                output_exclusion_reasons[output_key] = reason
        if reason is None:
            raw_output, output_record_key, output_key = matches[0]
            consumed_outputs.add(output_key)
            output, output_meta = _bounded_text_result(
                raw_output, field="tool_events.result.value", source_key=call_key, book=book
            )
            assert output is not None
            row.update({"result": output, "result_accounting": output_meta})
            row["_source_record_keys"].append(output_record_key)
            join = EvidenceResult.include(
                raw_output, field="tool_events.result", reason="matched_call_id",
                accounting_unit="tool_output_join", source_key=call_key,
            )
        else:
            reason_text = {
                "no_matching_tool_output": "no matching tool output record",
                "multiple_matching_tool_outputs": "multiple matching tool output records",
                "ambiguous_duplicate_tool_call_id": "duplicate tool call id is ambiguous",
                "tool_output_already_consumed": "matching tool output record was already consumed",
            }.get(reason, f"tool call id is unavailable: {call_id.reason}")
            row.update({"result": None, "result_reason": reason_text})
            join = EvidenceResult.exclude(
                field="tool_events.result", reason=reason,
                accounting_unit="tool_output_join", source_key=call_key,
            )
        add_receipt(book, join)
        add_receipt(book, EvidenceResult.include(
            row, field="tool_event", reason="disposition_related",
            accounting_unit="tool_event_candidate", source_key=call_key,
        ))
        included.append(row)
    output_candidates = book.get("tool_output_candidate", [])
    for candidate in output_candidates:
        if candidate.source_key in consumed_outputs:
            consumption = EvidenceResult.include(
                candidate.value,
                field="tool_output",
                reason="matched_once",
                accounting_unit="tool_output_consumption",
                source_key=candidate.source_key,
            )
        else:
            reason = (
                output_exclusion_reasons.get(candidate.source_key, "unmatched_tool_output")
                if candidate.included
                else f"output_candidate_{candidate.reason}"
            )
            consumption = EvidenceResult.exclude(
                field="tool_output",
                reason=reason,
                accounting_unit="tool_output_consumption",
                source_key=candidate.source_key,
            )
        add_receipt(book, consumption)
    assert_receipt_keys(book.get("tool_event_candidate", []), "tool_event_candidate", {call[0] for call in calls})
    assert_receipt_keys(book.get("tool_output_candidate", []), "tool_output_candidate", output_keys)
    assert_receipt_keys(
        book.get("tool_output_consumption", []), "tool_output_consumption", output_keys
    )
    assert_receipt_keys(book.get("tool_block", []), "tool_block", block_keys)
    assert_receipt_keys(book.get("tool_payload", []), "tool_payload", payload_keys)
    shown = {result.source_key for result in book.get("tool_event_candidate", []) if result.included}
    assert_receipt_keys(book.get("tool_output_join", []), "tool_output_join", shown)
    matched_joins = sum(result.included for result in book.get("tool_output_join", []))
    consumed = sum(result.included for result in book.get("tool_output_consumption", []))
    assert matched_joins == consumed == len(consumed_outputs)
    return included


def _tool_calls(records: list[dict]) -> list[dict]:
    book: ReceiptBook = {}
    return _tool_calls_with_receipts(list(enumerate(records, 1)), 0, book)
