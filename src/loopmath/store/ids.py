"""Ids and timestamps for store records (design/0.1/02-commands.md, section 1).

Ids are a prefix plus a ULID: 48 bits of milliseconds and 80 random bits in
Crockford base32, so they sort by creation time and never collide across
processes. Timestamps are ISO 8601 in UTC with a `Z` (0.2.4): onboard wrote UTC and the loop
local time before, so one store held both forms. Older values stay as written and are read by
instant (`parse_ts`); terminal lines show local time with the zone (`output.fmt_time`), and
`--since` without an offset is local time.
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_last = [0, 0]  # (ms, random) of the previous id in this process, for monotonic ids
_guard = threading.Lock()


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def new_ulid() -> str:
    """26 characters; later ids from one process sort after earlier ones."""
    with _guard:
        ms = time.time_ns() // 1_000_000
        if ms <= _last[0]:
            ms = _last[0]
            rand = (_last[1] + 1) & ((1 << 80) - 1)
        else:
            rand = int.from_bytes(os.urandom(10), "big")
        _last[0], _last[1] = ms, rand
    return _encode(ms, 10) + _encode(rand, 16)


def new_id(prefix: str) -> str:
    """`run_01J...`, `att_01J...`; `prefix` is given without the underscore."""
    return f"{prefix}_{new_ulid()}"


def ulid_time(value: str) -> datetime | None:
    """The creation time inside a `prefix_<ulid>` id (local time), or None."""
    tail = value.rsplit("_", 1)[-1] if isinstance(value, str) else ""
    if len(tail) != 26:
        return None
    ms = 0
    for ch in tail[:10].upper():
        i = _CROCKFORD.find(ch)
        if i < 0:
            return None
        ms = ms * 32 + i
    try:
        return datetime.fromtimestamp(ms / 1000).astimezone()
    except (OverflowError, OSError, ValueError):
        return None


def now_iso() -> str:
    """Now, in UTC with a `Z`, second precision: `2026-09-28T14:05:09Z`."""
    return utc_iso(datetime.now(timezone.utc).replace(microsecond=0))  # type: ignore[return-value]


def utc_iso(value: datetime | str | None) -> str | None:
    """A time (aware datetime, or an ISO string in any offset) as UTC with a `Z`, whole seconds unless it
    has a fraction; None for None, and an unreadable string back as given (JSON keeps what it cannot read)."""
    if value is None:
        return None
    dt = value if isinstance(value, datetime) else parse_ts(value)
    if dt is None:
        return value if isinstance(value, str) else None
    if dt.tzinfo is None:
        dt = dt.astimezone()
    dt = dt.astimezone(timezone.utc)
    spec = "milliseconds" if dt.microsecond else "seconds"
    return dt.replace(tzinfo=None).isoformat(timespec=spec) + "Z"


def fit_id(at: datetime | None = None) -> str:
    """`fit_YYYYmmddHHMMSS` in local time."""
    return "fit_" + (at or datetime.now().astimezone()).strftime("%Y%m%d%H%M%S")


def parse_ts(value: str | None) -> datetime | None:
    """ISO 8601 (a trailing Z allowed) to an aware datetime; None when absent or unreadable.

    A naive timestamp is read as local time.
    """
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt


def instant(value: str | None) -> float:
    """Seconds since the epoch, for sorting times written with any offset; -inf when absent or unreadable."""
    at = parse_ts(value)
    return at.timestamp() if at is not None else float("-inf")


_SINCE_RE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([a-z]+)\s*", re.IGNORECASE)
_SINCE_UNITS = {"h": 3600, "d": 86400, "w": 7 * 86400}


class SinceError(ValueError):
    """An unreadable `--since` (exit code 1)."""

    exit_code = 1


class AmbiguousSince(SinceError):
    """`--since 3m`: months or minutes? Refused (exit code 2)."""

    exit_code = 2


def parse_since(text: str | None, now: datetime | None = None) -> datetime | None:
    """The one `--since` reader for report, onboard, runs and share.

    `36h`, `90d` or `12w` (a whole or decimal number, any case, a space allowed) before `now`, or an ISO
    date or time (local time when it has no offset). None when absent. `m` raises `AmbiguousSince`
    (exit 2); anything else unreadable raises `SinceError` (exit 1). Both are ValueErrors carrying
    `exit_code`, so a handler can `fail(str(exc), getattr(exc, "exit_code", 1))`.
    """
    if text is None or not str(text).strip():
        return None
    raw = str(text).strip()
    m = _SINCE_RE.fullmatch(raw)
    if m and m.group(2).lower() == "m":
        raise AmbiguousSince(f"--since {raw}: ambiguous: use 90d, 12w or a date")
    hint = f"--since {raw}: use 36h, 90d, 12w or a date such as 2026-09-01"
    if m and m.group(2).lower() in _SINCE_UNITS:
        try:  # a length past year 1 (99999999999d) overflows timedelta or the date
            span = timedelta(seconds=float(m.group(1)) * _SINCE_UNITS[m.group(2).lower()])
            return (now or datetime.now().astimezone()) - span
        except OverflowError:
            raise SinceError(hint) from None
    dt = parse_ts(raw)
    if dt is None:
        raise SinceError(hint)
    return dt


def since_days(text: str | None, now: datetime | None = None) -> float | None:
    """`parse_since` as the number of days back from `now` (onboard's window); None when absent.

    A window that is not in the past raises `SinceError`.
    """
    now = now or datetime.now().astimezone()
    start = parse_since(text, now)
    if start is None:
        return None
    days = (now - start).total_seconds() / 86400
    if days <= 0:
        raise SinceError(f"--since {str(text).strip()} is not a window in the past")
    return days


def normalize_ts(value: str | None) -> str | None:
    """A caller's timestamp checked and kept as given; raises ValueError when unreadable."""
    if value is None:
        return None
    if parse_ts(value) is None:
        raise ValueError(f"not an ISO 8601 timestamp: {value!r}")
    return value
