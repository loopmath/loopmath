"""OTLP/JSON decoding for the GenAI adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .otel_genai_types import (
    _SPAN_ID,
    _TRACE_ID,
    _Omissions,
    _Span,
    _timestamp,
)


def _decode_value(value: Any, omissions: _Omissions) -> Any:
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return str(value["stringValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        raw = value["intValue"]
        if isinstance(raw, bool):
            parsed = None
        elif isinstance(raw, int):
            parsed = raw
        elif isinstance(raw, str) and re.fullmatch(r"-?[0-9]+", raw):
            parsed = int(raw)
        else:
            parsed = None
        if parsed is None:
            omissions.add("dropped_source_data")
        return parsed
    if "doubleValue" in value:
        try:
            return float(value["doubleValue"])
        except (TypeError, ValueError):
            omissions.add("dropped_source_data")
            return None
    if "bytesValue" in value:
        return str(value["bytesValue"])
    if "arrayValue" in value:
        values = value.get("arrayValue", {}).get("values", [])
        return [_decode_value(item, omissions) for item in values]
    if "kvlistValue" in value:
        values = value.get("kvlistValue", {}).get("values", [])
        return _decode_attributes(values, omissions)
    omissions.add("dropped_source_data")
    return None


def _decode_attributes(
    attributes: Any, omissions: _Omissions
) -> dict[str, Any] | None:
    decoded: dict[str, Any] = {}
    if not isinstance(attributes, list):
        omissions.add("malformed_record")
        return None
    for item in attributes:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            omissions.add("dropped_source_data")
            continue
        key = item["key"]
        value = _decode_value(item.get("value"), omissions)
        if key in decoded and decoded[key] != value:
            omissions.add("conflicting_attribute")
            return None
        decoded[key] = value
    return decoded


def _file_records(path: Path, omissions: _Omissions) -> list[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read OTLP JSON file {path}: {exc}") from exc
    if not text.strip():
        return []
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        records: list[Mapping[str, Any]] = []
        nonblank = [
            (number, line)
            for number, line in enumerate(text.splitlines(), 1)
            if line.strip()
        ]
        for index, (line_number, line) in enumerate(nonblank):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                if records and index == len(nonblank) - 1:
                    omissions.add("truncated_final_record")
                    break
                raise ValueError(
                    f"invalid OTLP JSON in {path} line {line_number}: {exc.msg}"
                ) from None
            if not isinstance(value, dict):
                raise ValueError(
                    f"OTLP JSON in {path} line {line_number} is not an object"
                )
            records.append(value)
        return records
    if not isinstance(whole, dict):
        raise ValueError(f"OTLP JSON in {path} is not an object")
    return [whole]


def _valid_id(value: Any, pattern: re.Pattern[str]) -> str | None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        return None
    folded = value.lower()
    return folded if int(folded, 16) != 0 else None


def _nanoseconds(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        parsed = int(value)
    else:
        return None
    return parsed if parsed >= 0 else None


def _read_spans(
    paths: Sequence[Path], omissions: _Omissions
) -> tuple[list[_Span], int]:
    spans: dict[tuple[str, str], _Span] = {}
    saw_span_shape = False
    for path in paths:
        for record in _file_records(path, omissions):
            if "resourceSpans" not in record:
                if record:
                    raise ValueError(f"OTLP JSON file {path} does not contain traces")
                continue
            resource_spans = record.get("resourceSpans")
            if not isinstance(resource_spans, list):
                raise ValueError(f"OTLP JSON file {path} has non-list resourceSpans")
            for resource_span in resource_spans:
                if not isinstance(resource_span, dict):
                    omissions.add("malformed_record")
                    continue
                resource = _decode_attributes(
                    resource_span.get("resource", {}).get("attributes", []), omissions
                )
                if resource is None:
                    continue
                scope_spans = resource_span.get("scopeSpans", [])
                if not isinstance(scope_spans, list):
                    omissions.add("malformed_record")
                    continue
                for scope_span in scope_spans:
                    if not isinstance(scope_span, dict):
                        omissions.add("malformed_record")
                        continue
                    scope = scope_span.get("scope", {})
                    scope_name = (
                        str(scope.get("name") or "") if isinstance(scope, dict) else ""
                    )
                    raw_spans = scope_span.get("spans", [])
                    if not isinstance(raw_spans, list):
                        omissions.add("malformed_record")
                        continue
                    for raw in raw_spans:
                        if not isinstance(raw, dict):
                            omissions.add("malformed_record")
                            continue
                        saw_span_shape = True
                        trace_id = _valid_id(raw.get("traceId"), _TRACE_ID)
                        span_id = _valid_id(raw.get("spanId"), _SPAN_ID)
                        if trace_id is None or span_id is None:
                            omissions.add("invalid_id")
                            continue
                        parent = raw.get("parentSpanId")
                        parent_span_id = None
                        if parent not in (None, ""):
                            parent_span_id = _valid_id(parent, _SPAN_ID)
                            if parent_span_id is None:
                                omissions.add("invalid_id")
                        start_ns = _nanoseconds(raw.get("startTimeUnixNano"))
                        end_ns = _nanoseconds(raw.get("endTimeUnixNano"))
                        if start_ns is None or end_ns is None or end_ns < start_ns:
                            omissions.add("malformed_record")
                            continue
                        try:
                            _timestamp(start_ns)
                            _timestamp(end_ns)
                        except (OverflowError, OSError, ValueError):
                            omissions.add("malformed_record")
                            continue
                        name = raw.get("name")
                        if not isinstance(name, str) or not name:
                            omissions.add("malformed_record")
                            continue
                        attrs = _decode_attributes(raw.get("attributes", []), omissions)
                        if attrs is None:
                            continue
                        status = raw.get("status", {})
                        if not isinstance(status, dict):
                            status = {}
                        span = _Span(
                            trace_id=trace_id,
                            span_id=span_id,
                            parent_span_id=parent_span_id,
                            name=name,
                            start_ns=start_ns,
                            end_ns=end_ns,
                            attrs=attrs,
                            resource=resource,
                            scope_name=scope_name,
                            status=status,
                        )
                        key = trace_id, span_id
                        previous = spans.get(key)
                        if previous is not None:
                            if previous != span:
                                raise ValueError(
                                    f"conflicting duplicate OTLP span {trace_id}/{span_id}"
                                )
                            continue
                        spans[key] = span
    if saw_span_shape and not spans:
        raise ValueError(
            "OTLP input contained span records but none had valid trace structure"
        )
    ordered = sorted(
        spans.values(), key=lambda span: (span.trace_id, span.start_ns, span.span_id)
    )
    return ordered, len(ordered)
