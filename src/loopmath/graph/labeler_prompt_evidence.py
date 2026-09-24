"""Node views and versioned attempt-evidence projections for labeler prompts."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from .labeler_prompt_node import _feat, _node_view_result, node_view
from .labeler_prompt_result import (
    EvidenceResult,
    ReceiptBook,
    aggregate_receipts,
    accounting_from_receipts,
    add_receipt,
    assert_receipt_keys,
    extend_receipts,
)
from .labeler_prompt_validation import (
    _MISSING,
    _TIMESTAMP_RE,
    _timestamp_sort_value,
    _typed_field,
    _valid_timestamp,
)


def _normalization(counts: Counter, field: str, reason: str, kind: str) -> None:
    """Count one value changed to null, naming both its field and reason."""
    counts[f"normalized_field:{field}:{reason}"] += 1
    counts[f"{kind}_field:{field}"] += 1


def _field(
    container: dict,
    key: str,
    counts: Counter,
    *,
    field: str | None = None,
    expected: type,
    valid=lambda _value: True,
    receipts: ReceiptBook | None = None,
    source_key: object | None = None,
) -> tuple[object | None, str | None]:
    """A projected value and, when normalized, the exact reason it became null."""
    name = field or key
    result = _typed_field(
        container,
        key,
        field=name,
        expected=expected,
        valid=valid,
        accounting_unit="attempt_field",
        source_key=source_key if source_key is not None else (name, id(container)),
    )
    if receipts is not None:
        add_receipt(receipts, result)
    if result.excluded:
        kind = "invalid" if result.reason.startswith("invalid") else "missing"
        _normalization(counts, name, result.reason, kind)
        return None, result.reason
    return result.value, None


def _cause_fields(
    attempt: dict,
    counts: Counter,
    *,
    receipts: ReceiptBook | None = None,
    source_prefix: tuple = (),
) -> dict:
    raw = attempt.get("cause", _MISSING)
    reason = None
    if raw is _MISSING:
        reason = "missing"
    elif raw is None:
        reason = "null"
    elif not isinstance(raw, dict):
        reason = "invalid_type"
    if reason is not None:
        kind = "invalid" if reason == "invalid_type" else "missing"
        _normalization(counts, "cause", reason, kind)
        child_reason = f"cause_{reason}"
        _normalization(counts, "cause.type", child_reason, kind)
        _normalization(counts, "cause.ref", child_reason, kind)
        if receipts is not None:
            for field, field_reason in (("cause", reason), ("cause.type", child_reason), ("cause.ref", child_reason)):
                add_receipt(receipts, EvidenceResult.exclude(
                    field=field,
                    reason=field_reason,
                    accounting_unit="attempt_field",
                    source_key=source_prefix + (field,),
                ))
        return {"type": None, "ref": None}
    if receipts is not None:
        add_receipt(receipts, EvidenceResult.include(
            raw,
            field="cause",
            reason="valid",
            accounting_unit="attempt_field",
            source_key=source_prefix + ("cause",),
        ))
    cause_type, _ = _field(
        raw, "type", counts, field="cause.type", expected=str, valid=bool,
        receipts=receipts, source_key=source_prefix + ("cause.type",),
    )
    cause_ref, _ = _field(
        raw, "ref", counts, field="cause.ref", expected=str, valid=bool,
        receipts=receipts, source_key=source_prefix + ("cause.ref",),
    )
    return {"type": cause_type, "ref": cause_ref}


def _attempt_evidence_result(
    item: dict,
    version: str = "v2",
    *,
    _node_index: int = 0,
    _keep_source_keys: bool = False,
) -> tuple[list[dict], Counter]:
    """The label-free contract evidence for one session and its exclusion counts.

    This deliberately projects a versioned allowlist instead of copying an attempt.
    Contract attempts also carry scored labels and their sources, which must never
    enter a model prompt. V2 retains its original row shape. V3 retains only the
    session id and times. Every malformed projected value is normalized to null and
    counted by field and reason.
    """
    rows, counts, _receipts = _attempt_collection(item, version, _node_index)
    if not _keep_source_keys:
        for row in rows:
            row.pop("_source_key", None)
    return rows, counts


def _attempt_collection(item: object, version: str, node_index: int) -> tuple[list[dict], Counter, ReceiptBook]:
    if version not in ("v2", "v3"):
        raise ValueError(f"attempt evidence is not defined for prompt version {version!r}")
    rows: list[dict] = []
    counts: Counter = Counter()
    receipts: ReceiptBook = {}
    raw_attempts = item.get("attempts", _MISSING) if isinstance(item, dict) else _MISSING
    container_key = (node_index, "attempts")
    if raw_attempts is _MISSING or raw_attempts is None:
        counts["nodes_without_attempts"] += 1
        reason = "node_not_object" if not isinstance(item, dict) else "missing" if raw_attempts is _MISSING else "null"
        add_receipt(receipts, EvidenceResult.exclude(
            field="attempts", reason=reason, accounting_unit="attempt_container", source_key=container_key,
        ))
        assert_receipt_keys(receipts["attempt_container"], "attempt_container", {container_key})
        return rows, counts, receipts
    counts["nodes_with_attempts_field"] += 1
    if not isinstance(raw_attempts, list):
        counts["attempt_containers_invalid"] += 1
        counts["excluded:attempts_not_list"] += 1
        add_receipt(receipts, EvidenceResult.exclude(
            field="attempts", reason="attempts_not_list", accounting_unit="attempt_container", source_key=container_key,
        ))
        assert_receipt_keys(receipts["attempt_container"], "attempt_container", {container_key})
        return rows, counts, receipts
    add_receipt(receipts, EvidenceResult.include(
        raw_attempts, field="attempts", reason="list", accounting_unit="attempt_container", source_key=container_key,
    ))
    if not raw_attempts:
        counts["nodes_with_empty_attempts"] += 1
    for attempt_index, attempt in enumerate(raw_attempts):
        attempt_key = (node_index, attempt_index)
        counts["attempt_records"] += 1
        if not isinstance(attempt, dict):
            counts["attempt_records_excluded"] += 1
            counts["excluded:attempt_not_object"] += 1
            add_receipt(receipts, EvidenceResult.exclude(
                field="attempt", reason="attempt_not_object", accounting_unit="attempt_record", source_key=attempt_key,
            ))
            continue
        add_receipt(receipts, EvidenceResult.include(
            attempt, field="attempt", reason="attempt_object", accounting_unit="attempt_record", source_key=attempt_key,
        ))
        field_key = lambda name: attempt_key + (name,)
        session_id, _ = _field(item, "id", counts, field="session_id", expected=str, valid=bool,
                               receipts=receipts, source_key=field_key("session_id"))
        started_at, _ = _field(attempt, "started_at", counts, expected=str, valid=_valid_timestamp,
                               receipts=receipts, source_key=field_key("started_at"))
        ended_at, _ = _field(attempt, "ended_at", counts, expected=str, valid=_valid_timestamp,
                             receipts=receipts, source_key=field_key("ended_at"))
        row = {"session_id": session_id}
        if version == "v2":
            task, task_issue = _field(attempt, "task", counts, expected=str, valid=bool,
                                      receipts=receipts, source_key=field_key("task"))
            number, number_issue = _field(attempt, "n", counts, expected=int, valid=lambda value: value > 0,
                                          receipts=receipts, source_key=field_key("n"))
            attempt_id = None
            if task_issue is None and number_issue is None:
                attempt_id = f"{task}·a{number}"
                add_receipt(receipts, EvidenceResult.include(
                    attempt_id, field="attempt_id", reason="derived", accounting_unit="attempt_field",
                    source_key=field_key("attempt_id"),
                ))
            else:
                issues = "+".join(
                    f"{name}_{issue}" for name, issue in (("task", task_issue), ("n", number_issue)) if issue is not None
                )
                kind = "invalid" if any(issue and issue.startswith("invalid") for issue in (task_issue, number_issue)) else "missing"
                _normalization(counts, "attempt_id", issues, kind)
                add_receipt(receipts, EvidenceResult.exclude(
                    field="attempt_id", reason=issues, accounting_unit="attempt_field", source_key=field_key("attempt_id"),
                ))
            result, _ = _field(attempt, "result", counts, expected=str, valid=bool,
                               receipts=receipts, source_key=field_key("result"))
            row.update({
                "task": task, "attempt_number": number, "attempt_id": attempt_id,
                "started_at": started_at, "ended_at": ended_at, "result": result,
                "cause": _cause_fields(attempt, counts, receipts=receipts, source_prefix=attempt_key),
            })
        else:
            row.update({"started_at": started_at, "ended_at": ended_at})
        row["_source_key"] = attempt_key
        rows.append(row)
        counts["attempt_records_included"] += 1
    assert_receipt_keys(receipts["attempt_container"], "attempt_container", {container_key})
    assert_receipt_keys(
        receipts.get("attempt_record", []), "attempt_record",
        {(node_index, attempt_index) for attempt_index in range(len(raw_attempts))},
    )
    field_names = ("session_id", "started_at", "ended_at") if version == "v3" else (
        "session_id", "started_at", "ended_at", "task", "n", "attempt_id", "result",
        "cause", "cause.type", "cause.ref",
    )
    expected_fields = {
        (node_index, attempt_index, field)
        for attempt_index, attempt in enumerate(raw_attempts)
        if isinstance(attempt, dict)
        for field in field_names
    }
    assert_receipt_keys(receipts.get("attempt_field", []), "attempt_field", expected_fields)
    return rows, counts, receipts


def attempt_evidence(item: dict, version: str = "v2") -> list[dict]:
    """The projected rows; `build_prompt` surfaces all exclusions in its metadata."""
    return _attempt_evidence_result(item, version)[0]


def _attempt_ledger_result(
    nodes: list[dict], version: str = "v2", *, include_accounting: bool = False
) -> tuple[list[dict], dict]:
    """A stable ledger plus an honest accounting of every attempted input record."""
    rows: list[dict] = []
    counts: Counter = Counter()
    receipts: ReceiptBook = {}
    for node_index, item in enumerate(nodes):
        item_rows, item_counts, item_receipts = _attempt_collection(item, version, node_index)
        rows.extend(item_rows)
        counts.update(item_counts)
        extend_receipts(receipts, item_receipts)

    def key(row: dict) -> tuple:
        parsed = _timestamp_sort_value(row["started_at"])
        invalid_key = row["_source_key"]
        if version == "v2":
            tie = (str(row["session_id"]), str(row["task"]), str(row["attempt_number"]))
        else:
            tie = (str(row["session_id"]), str(row["ended_at"]))
        return (parsed is None, parsed or datetime.max.replace(tzinfo=timezone.utc), tie, invalid_key)

    ledger = sorted(rows, key=key)
    for row in ledger:
        row.pop("_source_key", None)
    normalized: dict[str, dict[str, int]] = {}
    for counter, count in sorted(counts.items()):
        if not counter.startswith("normalized_field:"):
            continue
        _prefix, field, reason = counter.split(":", 2)
        normalized.setdefault(field, {})[reason] = count
    metadata = {
        "nodes": len(nodes),
        "nodes_with_attempts_field": counts["nodes_with_attempts_field"],
        "nodes_without_attempts": counts["nodes_without_attempts"],
        "nodes_with_empty_attempts": counts["nodes_with_empty_attempts"],
        "attempt_containers_invalid": counts["attempt_containers_invalid"],
        "attempt_records": counts["attempt_records"],
        "attempt_records_included": counts["attempt_records_included"],
        "attempt_records_excluded": counts["attempt_records_excluded"],
        "excluded_by_reason": {k[len("excluded:") :]: v for k, v in sorted(counts.items()) if k.startswith("excluded:")},
        "missing_fields": {k[len("missing_field:") :]: v for k, v in sorted(counts.items()) if k.startswith("missing_field:")},
        "invalid_fields": {k[len("invalid_field:") :]: v for k, v in sorted(counts.items()) if k.startswith("invalid_field:")},
        "normalized_fields": normalized,
    }
    assert counts["attempt_records"] == counts["attempt_records_included"] + counts["attempt_records_excluded"]
    if include_accounting:
        accounting = accounting_from_receipts(receipts)
        for unit in ("attempt_container", "attempt_record", "attempt_field"):
            accounting.setdefault(unit, aggregate_receipts([], unit))
        metadata["accounting"] = dict(sorted(accounting.items()))
    return ledger, metadata
