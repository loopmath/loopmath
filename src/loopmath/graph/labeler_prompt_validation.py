"""Shared typed field and timestamp validation for prompt evidence."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from .labeler_prompt_result import EvidenceResult


_MISSING = object()
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _typed_field(
    container: dict,
    key: str,
    *,
    field: str,
    expected: type,
    valid: Callable[[object], bool],
    accounting_unit: str,
    source_key: object,
) -> EvidenceResult:
    value = container.get(key, _MISSING)
    if value is _MISSING:
        return EvidenceResult.exclude(
            field=field, reason="missing", accounting_unit=accounting_unit,
            source_key=source_key,
        )
    if value is None:
        return EvidenceResult.exclude(
            field=field, reason="null", accounting_unit=accounting_unit,
            source_key=source_key,
        )
    if type(value) is not expected:
        return EvidenceResult.exclude(
            field=field, reason="invalid_type", accounting_unit=accounting_unit,
            source_key=source_key,
        )
    if not valid(value):
        return EvidenceResult.exclude(
            field=field, reason="invalid_value", accounting_unit=accounting_unit,
            source_key=source_key,
        )
    return EvidenceResult.include(
        value, field=field, reason="valid", accounting_unit=accounting_unit,
        source_key=source_key,
    )


def _valid_timestamp(value: str) -> bool:
    if not _TIMESTAMP_RE.fullmatch(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _timestamp_sort_value(value: object) -> datetime | None:
    """Return a real instant only for a validated timestamp."""
    if not isinstance(value, str) or not _valid_timestamp(value):
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
