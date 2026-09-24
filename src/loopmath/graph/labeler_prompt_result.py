"""Typed atomic receipts for the labeler prompt evidence pipeline."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar, cast


T = TypeVar("T")
ReceiptBook = dict[str, list["EvidenceResult[object]"]]


@dataclass(frozen=True)
class PromptSnapshot:
    """Immutable prompt bytes and metadata bytes from one evidence projection."""

    text: str
    metadata_json: str

    @property
    def metadata(self) -> dict:
        return json.loads(self.metadata_json)


def prompt_snapshot(text: str, metadata: dict) -> PromptSnapshot:
    return PromptSnapshot(
        text=text,
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )


@dataclass(frozen=True)
class EvidenceResult(Generic[T]):
    """One included or excluded evidence atom with its stable identity.

    Results stay internal to prompt construction.  The JSON prompt receives only
    aggregate counts, never source keys or the receipt objects themselves.
    """

    value: T | None
    field: str
    reason: str
    accounting_unit: str
    source_key: Hashable
    included: int
    excluded: int

    def __post_init__(self) -> None:
        labels = (self.field, self.reason, self.accounting_unit)
        if not all(isinstance(label, str) and label for label in labels):
            raise ValueError(
                "evidence field, reason, and accounting unit must be explicit strings"
            )
        try:
            hash(self.source_key)
        except TypeError as exc:
            raise TypeError("evidence source key must be hashable") from exc
        if self.included not in (0, 1) or self.excluded not in (0, 1):
            raise ValueError("atomic evidence counts must be zero or one")
        if self.included + self.excluded != 1:
            raise ValueError("atomic evidence must be exactly one of included or excluded")
        if self.included and self.value is None:
            raise ValueError("included evidence must have a non-null value")
        if self.excluded and self.value is not None:
            raise ValueError("excluded evidence must have value None")

    @classmethod
    def include(
        cls,
        value: T,
        *,
        field: str,
        reason: str,
        accounting_unit: str,
        source_key: Hashable,
    ) -> "EvidenceResult[T]":
        return cls(value, field, reason, accounting_unit, source_key, 1, 0)

    @classmethod
    def exclude(
        cls,
        *,
        field: str,
        reason: str,
        accounting_unit: str,
        source_key: Hashable,
    ) -> "EvidenceResult[T]":
        return cls(None, field, reason, accounting_unit, source_key, 0, 1)


def add_receipt(book: ReceiptBook, result: EvidenceResult[object]) -> None:
    """Store an atom under its unit without combining unlike denominators."""
    book.setdefault(result.accounting_unit, []).append(result)


def extend_receipts(book: ReceiptBook, other: ReceiptBook) -> None:
    """Merge books by unit; each later aggregation still sees one unit only."""
    for unit, results in other.items():
        if any(result.accounting_unit != unit for result in results):
            raise ValueError(f"receipt book entry {unit!r} contains a different unit")
        book.setdefault(unit, []).extend(results)


def aggregate_receipts(
    results: list[EvidenceResult[object]], accounting_unit: str
) -> dict:
    """Summarize one exact unit and assert its identity and count invariants."""
    if any(result.accounting_unit != accounting_unit for result in results):
        raise ValueError(f"cannot aggregate mixed evidence units as {accounting_unit!r}")
    keys = [result.source_key for result in results]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate source key in {accounting_unit!r} receipts")
    included = sum(result.included for result in results)
    excluded = sum(result.excluded for result in results)
    if len(results) != included + excluded:
        raise AssertionError("evidence accounting identity failed")
    included_reasons = Counter(result.reason for result in results if result.included)
    excluded_reasons = Counter(result.reason for result in results if result.excluded)
    excluded_fields: dict[str, Counter[str]] = {}
    for result in results:
        if result.excluded:
            excluded_fields.setdefault(result.field, Counter())[result.reason] += 1
    return {
        "accounting_unit": accounting_unit,
        "source_items": len(results),
        "included": included,
        "excluded": excluded,
        "included_by_reason": dict(sorted(included_reasons.items())),
        "excluded_by_reason": dict(sorted(excluded_reasons.items())),
        "excluded_fields_by_reason": {
            field: dict(sorted(reasons.items()))
            for field, reasons in sorted(excluded_fields.items())
        },
    }


def assert_receipt_keys(
    results: list[EvidenceResult[object]], accounting_unit: str, expected_keys: set[Hashable]
) -> None:
    """Prove that an established raw population has exactly one receipt per key."""
    if any(result.accounting_unit != accounting_unit for result in results):
        raise ValueError(f"cannot check mixed evidence units as {accounting_unit!r}")
    actual = [result.source_key for result in results]
    if len(actual) != len(set(actual)) or set(actual) != expected_keys:
        raise AssertionError(
            f"{accounting_unit} receipt keys do not equal the raw positional keys"
        )


def accounting_from_receipts(book: ReceiptBook) -> dict[str, dict]:
    """Render every receipt unit independently and in deterministic order."""
    return {
        unit: aggregate_receipts(cast(list[EvidenceResult[object]], book[unit]), unit)
        for unit in sorted(book)
    }
