"""Raw transcript selection and per-session evidence accounting."""

from __future__ import annotations

import re
from collections import Counter

from .labeler_prompt_records import (
    _EVIDENCE_TEXT_HALF,
    _EVIDENCE_TEXT_LIMIT,
    _bounded_text,
    _content_text,
    _human_span,
    _human_span_result,
    _read_session_receipts,
    _read_session_records,
    _text_span,
    _text_span_result,
)
from .labeler_prompt_tools import (
    _DISPOSITION_TOOL_RE,
    _command_from_arguments,
    _tool_calls,
    _tool_calls_with_receipts,
)
from .labeler_prompt_result import (
    EvidenceResult,
    ReceiptBook,
    aggregate_receipts,
    accounting_from_receipts,
    add_receipt,
    assert_receipt_keys,
    extend_receipts,
)


_SESSION_LIST_LIMIT = 8
_SESSION_LIST_HALF = 4
_DECISION_LIST_LIMIT = 64
_DECISION_LIST_HALF = 32
_COMMAND_WRAPPER_RE = re.compile(r"^<command-name>.*?</command-name>", re.DOTALL)
_DECISION_TEXT_RE = re.compile(
    r"\b(?:send(?:ing|s|t)?[ -]?back|rework|repair(?:ed|ing)?|fix(?:ed|es|ing)?|review(?:ed|ing)?|"
    r"verdict|blocker(?:s)?|flag(?:s|ged)?|patch[ -]?round|resume(?:d|s|ing)?|"
    r"reject(?:ed|ion)?|approv(?:e|ed|al|ing)|accept(?:ed|ance|ing)?|merge(?:d|ing)?|gate|"
    r"checkpoint|pass(?:ed|ing)?|fail(?:ed|ure|ing)?|reopen(?:ed|ing)?)\b",
    re.IGNORECASE,
)
_SYSTEM_USER_TEXT_RE = re.compile(
    r"^(?:<task-notification>|<system-reminder>|<local-command-|"
    r"This session is being continued from a previous conversation that ran out of context\.)"
)
_SESSION_ACCOUNTING_UNITS = (
    "closing_text_selection",
    "evidence_field",
    "followup_candidate",
    "followup_selection",
    "human_text",
    "human_text_block",
    "human_timestamp",
    "parsed_session_record",
    "session_availability",
    "session_jsonl_line",
    "session_locator",
    "text_cap",
    "tool_block",
    "tool_event_candidate",
    "tool_event_selection",
    "tool_output_candidate",
    "tool_output_consumption",
    "tool_output_join",
    "tool_payload",
)


def _bounded_rows(rows: list[dict], *, limit: int, half: int) -> tuple[list[dict], int]:
    if len(rows) <= limit:
        return rows, 0
    return rows[:half] + rows[-half:], len(rows) - limit


def _cap_rows(
    rows: list[dict],
    *,
    limit: int,
    half: int,
    accounting_unit: str,
    field: str,
    cap_reason: str,
    book: ReceiptBook,
) -> list[dict]:
    shown, omitted = _bounded_rows(rows, limit=limit, half=half)
    shown_keys = {row["_source_key"] for row in shown}
    for row in rows:
        key = row["_source_key"]
        if key in shown_keys:
            result = EvidenceResult.include(
                row, field=field, reason="selected", accounting_unit=accounting_unit, source_key=key
            )
        else:
            result = EvidenceResult.exclude(
                field=field, reason=cap_reason, accounting_unit=accounting_unit, source_key=key
            )
        add_receipt(book, result)
    assert omitted == len(rows) - len(shown)
    assert_receipt_keys(
        book.get(accounting_unit, []), accounting_unit,
        {row["_source_key"] for row in rows},
    )
    return shown


def _user_text_exclusion(text: str) -> str | None:
    if _COMMAND_WRAPPER_RE.match(text):
        return "command_wrapper"
    if _SYSTEM_USER_TEXT_RE.match(text):
        return "generated_system_text"
    return None


def _reason_maps(counts: Counter) -> tuple[dict[str, int], dict[str, int]]:
    """Separate non-record input failures from parsed-record dispositions."""
    line_exclusions = {
        key[len("line_excluded:") :]: value
        for key, value in sorted(counts.items())
        if key.startswith("line_excluded:")
    }
    unavailable = {
        key[len("unavailable:") :]: value
        for key, value in sorted(counts.items())
        if key.startswith("unavailable:")
    }
    return line_exclusions, unavailable


def _session_record_fallback_reason(record: dict) -> str:
    """Name why a parsed record supplied no shown per-session evidence."""
    message = record.get("message")
    if isinstance(message, dict):
        if record.get("isMeta") is True and message.get("role") == "user":
            return "meta_user_record"
        content = message.get("content")
        if isinstance(content, list):
            if any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
                return "tool_output_without_included_disposition_call"
            if any(
                isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Bash"
                for block in content
            ):
                return "tool_not_disposition_related"
        return "message_without_visible_session_evidence"
    if record.get("type") == "response_item":
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return "invalid_response_item_payload"
        payload_type = payload.get("type")
        if payload_type in ("function_call", "custom_tool_call"):
            return "tool_not_disposition_related"
        if payload_type in ("function_call_output", "custom_tool_call_output"):
            return "tool_output_without_included_disposition_call"
        if payload_type == "message":
            return "response_message_without_visible_text"
        return "unsupported_response_item_type"
    return "unsupported_record_shape"


def _record_accounting(
    records: list[dict], included: set[int], reasons: dict[int, str], *, fallback
) -> tuple[int, int, dict[str, int]]:
    """Give every parsed source record one mutually exclusive disposition."""
    excluded = Counter()
    for record_index, record in enumerate(records):
        if record_index not in included:
            excluded[reasons.get(record_index) or fallback(record)] += 1
    assert len(records) == len(included) + sum(excluded.values())
    return len(included), sum(excluded.values()), dict(sorted(excluded.items()))


def _line_and_availability_metadata(book: ReceiptBook) -> tuple[int, int, dict, dict]:
    lines = book.get("session_jsonl_line", [])
    line_exclusions = Counter(result.reason for result in lines if result.excluded)
    unavailable = Counter(
        result.reason for result in book.get("session_availability", []) if result.excluded
    )
    return (
        len(lines),
        sum(result.included for result in lines),
        dict(sorted(line_exclusions.items())),
        dict(sorted(unavailable.items())),
    )


def _field_metadata(accounting: dict) -> tuple[dict, dict]:
    combined: dict[str, Counter] = {}
    for unit in ("evidence_field", "human_timestamp"):
        for field, reasons in accounting.get(unit, {}).get("excluded_fields_by_reason", {}).items():
            combined.setdefault(field, Counter()).update(reasons)
    fields = {
        field: dict(sorted(reasons.items())) for field, reasons in sorted(combined.items())
    }
    timestamps = {field: reasons for field, reasons in fields.items() if field.endswith(".at")}
    return fields, timestamps


def _timestamp_alias(
    aliases: dict[str, Counter], field: str, result: EvidenceResult[str]
) -> None:
    if result.excluded:
        aliases.setdefault(field, Counter())[result.reason] += 1


def _session_evidence_collection(item: object, collection_index: int) -> tuple[dict, dict, ReceiptBook]:
    records, book = _read_session_receipts(item, collection_index)
    human: list[tuple[tuple[int, int], str, str, object]] = []
    for line_number, record in records:
        record_key = (collection_index, line_number)
        span = _human_span_result(record, record_key, book)
        if span.included:
            role, text, at = span.value
            human.append((record_key, role, text, at))
    record_keys = {(collection_index, line_number) for line_number, _record in records}
    assert_receipt_keys(book.get("human_text", []), "human_text", record_keys)
    visible_human_keys = {
        result.source_key + ("human_text.at",)
        for result in book.get("human_text", []) if result.included
    }
    assert_receipt_keys(
        book.get("human_timestamp", []), "human_timestamp", visible_human_keys
    )

    assistants = [(key, text, at) for key, role, text, at in human if role == "assistant"]
    record_reasons: dict[tuple[int, int], str] = {}
    timestamp_aliases: dict[str, Counter] = {}
    if assistants:
        closing_key, closing_text, closing_at = assistants[-1]
        _timestamp_alias(timestamp_aliases, "closing_text.at", closing_at)
        closing = _text_span_result(
            closing_text, at=closing_at, speaker="assistant", field="closing_text",
            source_key=closing_key, book=book,
        ) | {"_source_record_key": closing_key, "_source_key": closing_key}
        for record_key, _text, _at in assistants:
            if record_key == closing_key:
                result = EvidenceResult.include(
                    closing, field="closing_text", reason="latest_assistant_text",
                    accounting_unit="closing_text_selection", source_key=record_key,
                )
            else:
                record_reasons[record_key] = "earlier_assistant_text"
                result = EvidenceResult.exclude(
                    field="closing_text", reason="earlier_assistant_text",
                    accounting_unit="closing_text_selection", source_key=record_key,
                )
            add_receipt(book, result)
    else:
        closing = {"value": None, "reason": "no visible assistant closing text"}
        add_receipt(book, EvidenceResult.exclude(
            field="closing_text", reason="no_visible_assistant_closing_text",
            accounting_unit="closing_text_selection", source_key=(collection_index, "absent"),
        ))
    closing_keys = (
        {record_key for record_key, _text, _at in assistants}
        if assistants else {(collection_index, "absent")}
    )
    assert_receipt_keys(book["closing_text_selection"], "closing_text_selection", closing_keys)

    users: list[tuple[tuple[int, int], str, object]] = []
    for record_key, role, text, at in human:
        if role != "user":
            continue
        reason = _user_text_exclusion(text)
        if reason is not None:
            record_reasons[record_key] = reason
            add_receipt(book, EvidenceResult.exclude(
                field="followup_user_text", reason=reason,
                accounting_unit="followup_candidate", source_key=record_key,
            ))
            continue
        users.append((record_key, text, at))
    followup_candidates: list[dict] = []
    for user_index, (record_key, text, at) in enumerate(users):
        if user_index == 0:
            record_reasons[record_key] = "first_user_text"
            add_receipt(book, EvidenceResult.exclude(
                field="followup_user_text", reason="first_user_text",
                accounting_unit="followup_candidate", source_key=record_key,
            ))
            continue
        row = _text_span_result(
            text, at=at, speaker="user", field="followup_user_text", source_key=record_key, book=book
        ) | {"_source_record_key": record_key, "_source_key": record_key}
        _timestamp_alias(timestamp_aliases, "followup_user_text.at", at)
        add_receipt(book, EvidenceResult.include(
            row, field="followup_user_text", reason="later_user_text",
            accounting_unit="followup_candidate", source_key=record_key,
        ))
        followup_candidates.append(row)
    assert_receipt_keys(
        book.get("followup_candidate", []), "followup_candidate",
        {record_key for record_key, role, _text, _at in human if role == "user"},
    )
    followups = _cap_rows(
        followup_candidates, limit=_SESSION_LIST_LIMIT, half=_SESSION_LIST_HALF,
        accounting_unit="followup_selection", field="followup_user_text",
        cap_reason="followup_list_cap", book=book,
    )
    for result in book.get("followup_selection", []):
        if result.excluded:
            record_reasons[result.source_key] = result.reason

    tool_candidates = _tool_calls_with_receipts(records, collection_index, book)
    tools = _cap_rows(
        tool_candidates, limit=_SESSION_LIST_LIMIT, half=_SESSION_LIST_HALF,
        accounting_unit="tool_event_selection", field="tool_events",
        cap_reason="tool_event_list_cap", book=book,
    )
    for result in book.get("tool_event_selection", []):
        if result.excluded:
            row = next(row for row in tool_candidates if row["_source_key"] == result.source_key)
            for record_key in row["_source_record_keys"]:
                record_reasons[record_key] = result.reason

    closing_values = [closing] if closing.get("value") is not None else []
    candidate_values = closing_values + followup_candidates + tool_candidates
    shown_values = closing_values + followups + tools
    included_record_keys = {value["_source_record_key"] for value in closing_values + followups}
    for value in tools:
        included_record_keys.update(value["_source_record_keys"])
    for line_number, record in records:
        key = (collection_index, line_number)
        if key in included_record_keys:
            result = EvidenceResult.include(
                record, field="session_record", reason="contributed_visible_evidence",
                accounting_unit="parsed_session_record", source_key=key,
            )
        else:
            reason = record_reasons.get(key) or _session_record_fallback_reason(record)
            result = EvidenceResult.exclude(
                field="session_record", reason=reason,
                accounting_unit="parsed_session_record", source_key=key,
            )
        add_receipt(book, result)
    assert_receipt_keys(book.get("parsed_session_record", []), "parsed_session_record", record_keys)

    original_characters = shown_characters = truncated_values = 0
    for value in candidate_values:
        if "original_chars" in value:
            original_characters += value["original_chars"]
            truncated_values += int(value["truncated"])
        for accounting_key in ("command_accounting", "result_accounting"):
            accounting = value.get(accounting_key)
            if isinstance(accounting, dict):
                original_characters += accounting["original_chars"]
                truncated_values += int(accounting["truncated"])
    for value in shown_values:
        if "shown_chars" in value:
            shown_characters += value["shown_chars"]
        for accounting_key in ("command_accounting", "result_accounting"):
            accounting = value.get(accounting_key)
            if isinstance(accounting, dict):
                shown_characters += accounting["shown_chars"]

    accounting = accounting_from_receipts(book)
    for unit in _SESSION_ACCOUNTING_UNITS:
        accounting.setdefault(unit, aggregate_receipts([], unit))
    accounting = dict(sorted(accounting.items()))
    record_accounting = accounting["parsed_session_record"]
    source_lines, source_records, line_exclusions, unavailable = _line_and_availability_metadata(book)
    fields, timestamps = _field_metadata(accounting)
    for field, reasons in timestamp_aliases.items():
        timestamps[field] = dict(sorted(reasons.items()))
    metadata = {
        "accounting_unit": "parsed_session_record",
        "source_lines": source_lines,
        "source_records": source_records,
        "included_records": record_accounting["included"],
        "excluded_records": record_accounting["excluded"],
        "excluded_by_reason": record_accounting["excluded_by_reason"],
        "source_line_exclusions_by_reason": line_exclusions,
        "source_unavailable_by_reason": unavailable,
        "original_characters": original_characters,
        "shown_characters": shown_characters,
        "truncated_values": truncated_values,
        "followup_user_text": {"available": len(followup_candidates), "included": len(followups)},
        "tool_events": {"available": len(tool_candidates), "included": len(tools)},
        "field_exclusions_by_reason": fields,
        "timestamp_fields_by_reason": timestamps,
        "accounting": accounting,
    }
    for value in closing_values + followup_candidates:
        value.pop("_source_record_key", None)
        value.pop("_source_key", None)
    for value in tool_candidates:
        value.pop("_source_record_keys", None)
        value.pop("_source_key", None)
    return {"closing_text": closing, "followup_user_text": followups, "tool_events": tools, "metadata": metadata}, metadata, book


def _session_evidence_result(item: dict) -> tuple[dict, dict]:
    evidence, metadata, _book = _session_evidence_collection(item, 0)
    return evidence, metadata


def session_evidence(item: dict) -> dict:
    """Project raw, label-free transcript observations for one v4 node."""
    return _session_evidence_result(item)[0]


def _sum_accounting(metadata_rows: list[dict]) -> dict:
    units = sorted(
        set(_SESSION_ACCOUNTING_UNITS)
        | {unit for metadata in metadata_rows for unit in metadata.get("accounting", {})}
    )
    combined: dict[str, dict] = {}
    for unit in units:
        rows = [metadata["accounting"][unit] for metadata in metadata_rows if unit in metadata.get("accounting", {})]
        if any(row["accounting_unit"] != unit for row in rows):
            raise ValueError("cannot aggregate unlike evidence accounting units")
        excluded = Counter()
        included = Counter()
        fields: dict[str, Counter] = {}
        for row in rows:
            excluded.update(row["excluded_by_reason"])
            included.update(row["included_by_reason"])
            for field, reasons in row["excluded_fields_by_reason"].items():
                fields.setdefault(field, Counter()).update(reasons)
        combined[unit] = {
            "accounting_unit": unit,
            "source_items": sum(row["source_items"] for row in rows),
            "included": sum(row["included"] for row in rows),
            "excluded": sum(row["excluded"] for row in rows),
            "included_by_reason": dict(sorted(included.items())),
            "excluded_by_reason": dict(sorted(excluded.items())),
            "excluded_fields_by_reason": {
                field: dict(sorted(reasons.items())) for field, reasons in sorted(fields.items())
            },
        }
        assert combined[unit]["source_items"] == combined[unit]["included"] + combined[unit]["excluded"]
    return combined


def _session_evidence_summary(metadata_rows: list[dict]) -> dict:
    excluded: Counter = Counter()
    line_exclusions: Counter = Counter()
    unavailable: Counter = Counter()
    field_exclusions: dict[str, Counter] = {}
    timestamps: dict[str, Counter] = {}
    for metadata in metadata_rows:
        excluded.update(metadata["excluded_by_reason"])
        line_exclusions.update(metadata["source_line_exclusions_by_reason"])
        unavailable.update(metadata["source_unavailable_by_reason"])
        for field, reasons in metadata.get("field_exclusions_by_reason", {}).items():
            field_exclusions.setdefault(field, Counter()).update(reasons)
        for field, reasons in metadata.get("timestamp_fields_by_reason", {}).items():
            timestamps.setdefault(field, Counter()).update(reasons)
    result = {
        "accounting_unit": "parsed_session_record",
        "node_collections": len(metadata_rows),
        "source_lines": sum(metadata["source_lines"] for metadata in metadata_rows),
        "source_records": sum(metadata["source_records"] for metadata in metadata_rows),
        "included_records": sum(metadata["included_records"] for metadata in metadata_rows),
        "excluded_records": sum(metadata["excluded_records"] for metadata in metadata_rows),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "source_line_exclusions_by_reason": dict(sorted(line_exclusions.items())),
        "source_unavailable_by_reason": dict(sorted(unavailable.items())),
        "original_characters": sum(metadata["original_characters"] for metadata in metadata_rows),
        "shown_characters": sum(metadata["shown_characters"] for metadata in metadata_rows),
        "truncated_values": sum(metadata["truncated_values"] for metadata in metadata_rows),
        "field_exclusions_by_reason": {
            field: dict(sorted(reasons.items())) for field, reasons in sorted(field_exclusions.items())
        },
        "timestamp_fields_by_reason": {
            field: dict(sorted(reasons.items())) for field, reasons in sorted(timestamps.items())
        },
        "accounting": _sum_accounting(metadata_rows),
    }
    assert result["source_records"] == result["included_records"] + result["excluded_records"]
    return result
