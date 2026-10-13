"""Dataset-wide workflow dispatch and disposition evidence projections."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from .labeler_prompt_dispatch import (
    _dispatch_text,
    _task_dispatches,
    _task_dispatches_with_receipts,
)
from .labeler_prompt_validation import _MISSING, _timestamp_sort_value
from .labeler_prompt_records import (
    _human_span_result,
    _read_session_receipts,
    _text_span_result,
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
from .labeler_prompt_session import (
    _DECISION_LIST_HALF,
    _DECISION_LIST_LIMIT,
    _DECISION_TEXT_RE,
    _cap_rows,
    _field_metadata,
    _line_and_availability_metadata,
    _timestamp_alias,
    _user_text_exclusion,
)

_WORKFLOW_ACCOUNTING_UNITS = (
    "child_join",
    "dataset_node_item",
    "decision_candidate",
    "decision_selection",
    "dispatch_candidate",
    "evidence_field",
    "human_text",
    "human_text_block",
    "human_timestamp",
    "parsed_session_record",
    "runtime_child_selection",
    "session_availability",
    "session_jsonl_line",
    "session_locator",
    "text_cap",
    "verified_child_candidate",
    "verified_child_selection",
)


def _shape_reason(prefix: str, value: object) -> str:
    if value is _MISSING:
        return f"{prefix}_missing"
    if value is None:
        return f"{prefix}_null"
    return f"{prefix}_invalid_type"


def _verified_child_candidates(nodes: list[object], book: ReceiptBook) -> dict[str, dict[str, list[dict]]]:
    children: dict[str, dict[str, list[dict]]] = {}
    for node_index, item in enumerate(nodes):
        key = node_index
        reason = None
        if not isinstance(item, dict):
            reason = "node_not_object"
        else:
            features = item.get("features", _MISSING)
            if not isinstance(features, dict):
                reason = _shape_reason("features", features)
            else:
                parent = features.get("parent", _MISSING)
                skeleton = features.get("skeleton", _MISSING)
                if not isinstance(parent, dict):
                    reason = _shape_reason("parent", parent)
                elif parent.get("tier", _MISSING) != "verified":
                    tier = parent.get("tier", _MISSING)
                    reason = (
                        "parent_tier_not_verified"
                        if isinstance(tier, str)
                        else _shape_reason("parent_tier", tier)
                    )
                elif not isinstance(parent.get("value", _MISSING), str) or not parent.get("value"):
                    parent_value = parent.get("value", _MISSING)
                    reason = (
                        "parent_id_empty"
                        if parent_value == ""
                        else _shape_reason("parent_id", parent_value)
                    )
                elif not isinstance(skeleton, dict):
                    reason = _shape_reason("skeleton", skeleton)
                else:
                    spawn = skeleton.get("spawn", _MISSING)
                    if not isinstance(spawn, dict):
                        reason = _shape_reason("spawn", spawn)
                    elif not isinstance(spawn.get("tool_use_id", _MISSING), str) or not spawn.get("tool_use_id"):
                        tool_use_id_value = spawn.get("tool_use_id", _MISSING)
                        reason = (
                            "tool_use_id_empty"
                            if tool_use_id_value == ""
                            else _shape_reason("tool_use_id", tool_use_id_value)
                        )
                    else:
                        parent_id = parent["value"]
                        tool_use_id = spawn["tool_use_id"]
                        value = {"item": item, "node_index": node_index}
                        children.setdefault(parent_id, {}).setdefault(tool_use_id, []).append(value)
                        add_receipt(book, EvidenceResult.include(
                            value, field="verified_child", reason="verified_parent_and_tool_use_id",
                            accounting_unit="verified_child_candidate", source_key=key,
                        ))
                        continue
        add_receipt(book, EvidenceResult.exclude(
            field="verified_child", reason=reason,
            accounting_unit="verified_child_candidate", source_key=key,
        ))
    assert_receipt_keys(
        book.get("verified_child_candidate", []), "verified_child_candidate", set(range(len(nodes)))
    )
    return children


def _verified_children(nodes: list[dict]) -> dict[str, dict[str, dict]]:
    """Compatibility view containing only unambiguous verified child joins."""
    book: ReceiptBook = {}
    candidates = _verified_child_candidates(nodes, book)
    return {
        parent_id: {
            tool_use_id: matches[0]["item"]
            for tool_use_id, matches in by_tool.items()
            if len(matches) == 1
        }
        for parent_id, by_tool in candidates.items()
    }


def _workflow_node_result(
    item: object, node_index: int, verified_parent_ids: set[str]
) -> EvidenceResult[tuple[str, dict]]:
    unit = "dataset_node_item"
    if not isinstance(item, dict):
        return EvidenceResult.exclude(field="workflow_node", reason="node_not_object", accounting_unit=unit, source_key=node_index)
    node_id = item.get("id", _MISSING)
    if node_id is _MISSING:
        return EvidenceResult.exclude(field="workflow_node", reason="node_id_missing", accounting_unit=unit, source_key=node_index)
    if node_id is None:
        return EvidenceResult.exclude(field="workflow_node", reason="node_id_null", accounting_unit=unit, source_key=node_index)
    if not isinstance(node_id, str):
        return EvidenceResult.exclude(field="workflow_node", reason="node_id_invalid_type", accounting_unit=unit, source_key=node_index)
    if not node_id:
        return EvidenceResult.exclude(field="workflow_node", reason="node_id_empty", accounting_unit=unit, source_key=node_index)
    verified_parent = node_id in verified_parent_ids
    features = item.get("features", _MISSING)
    if not isinstance(features, dict):
        if verified_parent:
            return EvidenceResult.include((node_id, item), field="workflow_node", reason="verified_parent", accounting_unit=unit, source_key=node_index)
        reason = _shape_reason("features", features)
        return EvidenceResult.exclude(field="workflow_node", reason=reason, accounting_unit=unit, source_key=node_index)
    skeleton = features.get("skeleton", _MISSING)
    if not isinstance(skeleton, dict):
        if verified_parent:
            return EvidenceResult.include((node_id, item), field="workflow_node", reason="verified_parent", accounting_unit=unit, source_key=node_index)
        reason = _shape_reason("skeleton", skeleton)
        return EvidenceResult.exclude(field="workflow_node", reason=reason, accounting_unit=unit, source_key=node_index)
    role = skeleton.get("role", _MISSING)
    lead = role == "lead"
    if lead and verified_parent:
        reason = "lead_and_verified_parent"
    elif lead:
        reason = "lead"
    elif verified_parent:
        reason = "verified_parent"
    else:
        if role is _MISSING:
            exclusion = "role_missing"
        elif role is None:
            exclusion = "role_null"
        elif not isinstance(role, str):
            exclusion = "role_invalid_type"
        else:
            exclusion = "not_lead_or_verified_parent"
        return EvidenceResult.exclude(field="workflow_node", reason=exclusion, accounting_unit=unit, source_key=node_index)
    return EvidenceResult.include((node_id, item), field="workflow_node", reason=reason, accounting_unit=unit, source_key=node_index)


def _workflow_record_fallback_reason(record: dict) -> str:
    """Name why a parsed orchestrator record supplied no shown workflow evidence."""
    message = record.get("message")
    if isinstance(message, dict):
        if record.get("isMeta") is True and message.get("role") == "user":
            return "meta_user_record"
        return "message_without_dispatch_or_visible_text"
    if record.get("type") == "response_item":
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return "invalid_response_item_payload"
        if payload.get("type") == "message":
            return "response_message_without_visible_text"
        return "unsupported_response_item_type"
    return "unsupported_record_shape"


def _sort_key(row: dict) -> tuple:
    parsed = _timestamp_sort_value(row["at"])
    return (parsed is None, parsed or datetime.max.replace(tzinfo=timezone.utc), row["_source_key"])


def _workflow_evidence_collection(nodes: list[object]) -> tuple[dict, dict, ReceiptBook]:
    book: ReceiptBook = {}
    child_map = _verified_child_candidates(nodes, book)
    parent_ids = set(child_map)
    node_results = [_workflow_node_result(item, index, parent_ids) for index, item in enumerate(nodes)]
    for result in node_results:
        add_receipt(book, result)
    assert_receipt_keys(book.get("dataset_node_item", []), "dataset_node_item", set(range(len(nodes))))
    dispatches: list[dict] = []
    decisions: list[dict] = []
    selected_child_keys: set[int] = set()
    ambiguous_child_keys: set[int] = set()
    source_records: dict[tuple[int, int], dict] = {}
    record_reasons: dict[tuple[int, int], str] = {}
    timestamp_aliases: dict[str, Counter] = {}
    for node_result in node_results:
        if node_result.excluded:
            continue
        collection_index = node_result.source_key
        parent_id, item = node_result.value
        records, read_book = _read_session_receipts(item, collection_index)
        extend_receipts(book, read_book)
        for line_number, record in records:
            source_records[(collection_index, line_number)] = record
        rows = _task_dispatches_with_receipts(
            records,
            parent_id,
            child_map.get(parent_id, {}),
            collection_index,
            book,
            selected_child_keys=selected_child_keys,
            ambiguous_child_keys=ambiguous_child_keys,
        )
        dispatches.extend(rows)
        for line_number, record in records:
            record_key = (collection_index, line_number)
            span = _human_span_result(record, record_key, book)
            if span.excluded:
                add_receipt(book, EvidenceResult.exclude(
                    field="decision_text", reason=span.reason,
                    accounting_unit="decision_candidate", source_key=record_key,
                ))
                continue
            speaker, raw_text, at = span.value
            if speaker == "user":
                reason = _user_text_exclusion(raw_text)
                if reason is not None:
                    record_reasons[record_key] = reason
                    add_receipt(book, EvidenceResult.exclude(
                        field="decision_text", reason=reason,
                        accounting_unit="decision_candidate", source_key=record_key,
                    ))
                    continue
            if not _DECISION_TEXT_RE.search(raw_text):
                record_reasons[record_key] = "text_without_disposition_language"
                add_receipt(book, EvidenceResult.exclude(
                    field="decision_text", reason="text_without_disposition_language",
                    accounting_unit="decision_candidate", source_key=record_key,
                ))
                continue
            row = _text_span_result(
                raw_text, at=at, speaker=speaker, field="decision_text", source_key=record_key, book=book
            ) | {"session_id": parent_id, "_source_record_key": record_key, "_source_key": record_key}
            _timestamp_alias(timestamp_aliases, "decision_text.at", at)
            decisions.append(row)
            add_receipt(book, EvidenceResult.include(
                row, field="decision_text", reason="disposition_language",
                accounting_unit="decision_candidate", source_key=record_key,
            ))
    verified_child_keys: set[int] = set()
    for by_tool_use_id in child_map.values():
        for matches in by_tool_use_id.values():
            for match in matches:
                key = match["node_index"]
                verified_child_keys.add(key)
                if key in selected_child_keys:
                    result = EvidenceResult.include(
                        match["item"],
                        field="verified_child",
                        reason="selected_verified_child",
                        accounting_unit="verified_child_selection",
                        source_key=key,
                    )
                else:
                    reason = (
                        "ambiguous_verified_child_match"
                        if key in ambiguous_child_keys
                        else "unreferenced_verified_child_candidate"
                    )
                    result = EvidenceResult.exclude(
                        field="verified_child",
                        reason=reason,
                        accounting_unit="verified_child_selection",
                        source_key=key,
                    )
                add_receipt(book, result)
    assert not selected_child_keys & ambiguous_child_keys
    assert selected_child_keys | ambiguous_child_keys <= verified_child_keys
    assert_receipt_keys(
        book.get("verified_child_selection", []),
        "verified_child_selection",
        verified_child_keys,
    )
    assert_receipt_keys(
        book.get("decision_candidate", []), "decision_candidate", set(source_records)
    )
    visible_human_keys = {
        result.source_key + ("human_text.at",)
        for result in book.get("human_text", []) if result.included
    }
    assert_receipt_keys(
        book.get("human_timestamp", []), "human_timestamp", visible_human_keys
    )
    dispatches.sort(key=_sort_key)
    decisions.sort(key=_sort_key)
    shown_decisions = _cap_rows(
        decisions, limit=_DECISION_LIST_LIMIT, half=_DECISION_LIST_HALF,
        accounting_unit="decision_selection", field="decision_text",
        cap_reason="decision_text_list_cap", book=book,
    )
    for result in book.get("decision_selection", []):
        if result.excluded:
            record_reasons[result.source_key] = result.reason
    included_record_keys = {row["_source_record_key"] for row in dispatches + shown_decisions}
    for record_key, record in source_records.items():
        if record_key in included_record_keys:
            result = EvidenceResult.include(
                record, field="workflow_record", reason="contributed_visible_evidence",
                accounting_unit="parsed_session_record", source_key=record_key,
            )
        else:
            result = EvidenceResult.exclude(
                field="workflow_record", reason=record_reasons.get(record_key) or _workflow_record_fallback_reason(record),
                accounting_unit="parsed_session_record", source_key=record_key,
            )
        add_receipt(book, result)
    assert_receipt_keys(
        book.get("parsed_session_record", []), "parsed_session_record", set(source_records)
    )

    original_characters = shown_characters = truncated_values = 0
    dispatch_text_issues: dict[str, Counter] = {}
    for row in dispatches:
        for field, key in (("description", "description_accounting"), ("prompt", "prompt_accounting")):
            value = row[key]
            reason = value.get("reason")
            if isinstance(reason, str):
                dispatch_text_issues.setdefault(field, Counter())[reason] += 1
            else:
                original_characters += value["original_chars"]
                shown_characters += value["shown_chars"]
                truncated_values += int(value["truncated"])
    for row in decisions:
        original_characters += row["original_chars"]
        truncated_values += int(row["truncated"])
    shown_characters += sum(row["shown_chars"] for row in shown_decisions)

    accounting = accounting_from_receipts(book)
    for unit in _WORKFLOW_ACCOUNTING_UNITS:
        accounting.setdefault(unit, aggregate_receipts([], unit))
    accounting = dict(sorted(accounting.items()))
    node_accounting = accounting["dataset_node_item"]
    record_accounting = accounting.get("parsed_session_record", {
        "source_items": 0, "included": 0, "excluded": 0, "excluded_by_reason": {},
    })
    source_lines, source_record_count, line_exclusions, unavailable = _line_and_availability_metadata(book)
    fields, timestamps = _field_metadata(accounting)
    for field, reasons in timestamp_aliases.items():
        timestamps[field] = dict(sorted(reasons.items()))
    metadata = {
        "accounting_unit": "parsed_session_record",
        "orchestrator_sessions": node_accounting["included"],
        "source_lines": source_lines,
        "source_records": source_record_count,
        "included_records": record_accounting["included"],
        "excluded_records": record_accounting["excluded"],
        "excluded_by_reason": record_accounting["excluded_by_reason"],
        "source_line_exclusions_by_reason": line_exclusions,
        "source_unavailable_by_reason": unavailable,
        "original_characters": original_characters,
        "shown_characters": shown_characters,
        "truncated_values": truncated_values,
        "dispatches": {"available": len(dispatches), "included": len(dispatches)},
        "decision_text": {"available": len(decisions), "included": len(shown_decisions)},
        "dataset_node_accounting_unit": "dataset_node_item",
        "dataset_node_items": node_accounting["source_items"],
        "included_nodes": node_accounting["included"],
        "excluded_nodes": node_accounting["excluded"],
        "included_nodes_by_reason": node_accounting["included_by_reason"],
        "excluded_nodes_by_reason": node_accounting["excluded_by_reason"],
        "field_exclusions_by_reason": fields,
        "timestamp_fields_by_reason": timestamps,
        "accounting": accounting,
    }
    assert metadata["dataset_node_items"] == metadata["included_nodes"] + metadata["excluded_nodes"] == len(nodes)
    assert metadata["source_records"] == metadata["included_records"] + metadata["excluded_records"]
    if dispatch_text_issues:
        metadata["dispatch_text_fields_by_reason"] = {
            field: dict(sorted(reasons.items())) for field, reasons in sorted(dispatch_text_issues.items())
        }
    for row in dispatches + decisions:
        row.pop("_source_record_key", None)
        row.pop("_source_key", None)
    return {"dispatches": dispatches, "decision_text": shown_decisions, "metadata": metadata}, metadata, book


def _workflow_evidence_result(nodes: list[dict]) -> tuple[dict, dict]:
    evidence, metadata, _book = _workflow_evidence_collection(nodes)
    return evidence, metadata


def workflow_evidence(nodes: list[dict]) -> dict:
    """Project raw dispatch and disposition wording from verified workflow parents."""
    return _workflow_evidence_result(nodes)[0]
