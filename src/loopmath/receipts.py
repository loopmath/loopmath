"""Validation for E2 spend receipt ledgers.

An ``e2-receipts.jsonl`` row has this deliberately small schema::

    {"timestamp": "2026-08-31T12:00:00Z", "task": "T10",
     "arm": "X", "spend_usd": 0.25, "cap_usd": 2.00}

Unknown keys are rejected.  ``FROZEN.txt`` is a conventional SHA-256 checksum
file: one line containing ``<64 lowercase-or-uppercase hex digits>  <name>``.
The name is resolved beside FROZEN.txt and must not escape that directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


_RECEIPT_KEYS = {"timestamp", "task", "arm", "spend_usd", "cap_usd"}
_FROZEN_RE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


class ReceiptViolation(ValueError):
    """The first integrity violation found in a receipt ledger."""


@dataclass(frozen=True)
class VerificationSummary:
    receipts: int
    task_arms: int
    spend_usd: Decimal
    prereg: str

    def message(self) -> str:
        return (
            f"OK: {self.receipts} receipts, {self.task_arms} task-arms, "
            f"${self.spend_usd:.2f} spend; prereg hash verified ({self.prereg})"
        )


def _timestamp(value: object, line_number: int) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        raise ReceiptViolation(f"line {line_number}: timestamp must be an RFC 3339 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ReceiptViolation(f"line {line_number}: invalid timestamp {value!r}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReceiptViolation(
            f"line {line_number}: timestamp must include a UTC offset"
        )
    return parsed


def _money(value: object, field: str, line_number: int) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptViolation(f"line {line_number}: {field} must be a non-negative number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        raise ReceiptViolation(
            f"line {line_number}: {field} must be a non-negative number"
        ) from None
    if not amount.is_finite() or amount < 0:
        raise ReceiptViolation(f"line {line_number}: {field} must be a non-negative number")
    return amount


def _read_frozen(frozen_path: Path) -> tuple[str, Path]:
    try:
        lines = frozen_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ReceiptViolation(f"FROZEN.txt: cannot read {frozen_path}: {exc.strerror}") from None
    if len(lines) != 1 or not (match := _FROZEN_RE.fullmatch(lines[0])):
        raise ReceiptViolation(
            "FROZEN.txt: expected one '<sha256>  <prereg file>' line"
        )

    expected, name = match.groups()
    prereg = frozen_path.parent / name
    # A checksum file must not turn verification into an arbitrary file read.
    try:
        prereg.resolve().relative_to(frozen_path.parent.resolve())
    except ValueError:
        raise ReceiptViolation("FROZEN.txt: prereg file must be in its directory") from None
    if not prereg.is_file():
        raise ReceiptViolation(f"FROZEN.txt: prereg file not found: {name}")
    return expected.lower(), prereg


def _verify_prereg(frozen_path: Path) -> str:
    expected, prereg = _read_frozen(frozen_path)
    try:
        actual = hashlib.sha256(prereg.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReceiptViolation(
            f"FROZEN.txt: cannot read prereg file {prereg.name}: {exc.strerror}"
        ) from None
    if actual != expected:
        raise ReceiptViolation(f"prereg hash mismatch for {prereg.name}")
    return prereg.name


def verify_receipts(
    ledger_path: str | Path,
    frozen_path: str | Path | None = None,
) -> VerificationSummary:
    """Verify *ledger_path* and its companion checksum, stopping at first fault."""

    ledger = Path(ledger_path)
    frozen = Path(frozen_path) if frozen_path is not None else ledger.with_name("FROZEN.txt")

    try:
        source = ledger.open(encoding="utf-8")
    except OSError as exc:
        raise ReceiptViolation(f"cannot read ledger {ledger}: {exc.strerror}") from None

    count = 0
    total = Decimal("0")
    previous: tuple[datetime, int, str] | None = None
    spent: dict[tuple[str, str], Decimal] = {}
    caps: dict[tuple[str, str], Decimal] = {}

    with source:
        for line_number, raw in enumerate(source, 1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ReceiptViolation(
                    f"line {line_number}: invalid JSON ({exc.msg})"
                ) from None
            if not isinstance(row, dict):
                raise ReceiptViolation(f"line {line_number}: receipt must be a JSON object")
            missing = sorted(_RECEIPT_KEYS - row.keys())
            extra = sorted(row.keys() - _RECEIPT_KEYS)
            if missing:
                raise ReceiptViolation(
                    f"line {line_number}: missing required field {missing[0]!r}"
                )
            if extra:
                raise ReceiptViolation(f"line {line_number}: unknown field {extra[0]!r}")
            for field in ("task", "arm"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ReceiptViolation(
                        f"line {line_number}: {field} must be a non-empty string"
                    )

            stamp = _timestamp(row["timestamp"], line_number)
            if previous is not None and stamp < previous[0]:
                raise ReceiptViolation(
                    f"line {line_number}: timestamp {row['timestamp']} is before line "
                    f"{previous[1]} timestamp {previous[2]}"
                )
            previous = stamp, line_number, row["timestamp"]

            amount = _money(row["spend_usd"], "spend_usd", line_number)
            cap = _money(row["cap_usd"], "cap_usd", line_number)
            key = row["task"], row["arm"]
            if key in caps and cap != caps[key]:
                raise ReceiptViolation(
                    f"line {line_number}: cap changed for task-arm {key[0]}/{key[1]} "
                    f"from ${caps[key]:.2f} to ${cap:.2f}"
                )
            caps[key] = cap
            spent[key] = spent.get(key, Decimal("0")) + amount
            if spent[key] > cap:
                raise ReceiptViolation(
                    f"line {line_number}: task-arm {key[0]}/{key[1]} spend "
                    f"${spent[key]:.2f} exceeds cap ${cap:.2f}"
                )
            count += 1
            total += amount

    prereg = _verify_prereg(frozen)
    return VerificationSummary(count, len(spent), total, prereg)
