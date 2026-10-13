"""loopmath host-service implementations for extractable ADE adapters.

This module is intentionally outside :mod:`loopmath.adapters`: it is the only
adapter path allowed to know about loopmath's pricing table and vendor-log
ingesters.  Adapter modules receive the stdlib-only values and callables from
``loopmath.adapters.base`` instead of importing those implementation modules.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import __version__
from .adapters.base import (
    AdapterServices,
    PricingResult,
    Usage,
    VendorSession,
    VendorSessionResult,
)
from .ingest import claude_code, codex
from .price import load_prices, price_run


_CLAUDE_CODE_ROOT = Path.home() / ".claude" / "projects"
_CODEX_ROOT = Path.home() / ".codex" / "sessions"
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _normalized_correlate(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate.lower() if _UUID_RE.fullmatch(candidate) else None


def _codex_session_id(path: Path) -> str | None:
    """Read the first structural Codex session-id record from one log."""

    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.lstrip().startswith("{"):
                    continue
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(record, dict) or record.get("type") != "session_meta":
                    continue
                payload = record.get("payload")
                if not isinstance(payload, dict):
                    return None
                return _normalized_correlate(
                    payload.get("session_id") or payload.get("id")
                )
    except OSError:
        return None
    return None


def _as_record(value: object) -> dict[str, Any] | None:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        return None
    record = to_dict()
    return record if isinstance(record, dict) else None


def _usage(record: dict[str, Any]) -> Usage | None:
    tokens = record.get("tokens")
    if not isinstance(tokens, dict):
        return None
    values: list[int] = []
    for key in ("in", "cache_read", "cache_write", "out"):
        value = tokens.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        values.append(value)
    return Usage(
        input_tokens=values[0],
        cached_input_tokens=values[1],
        cache_creation_tokens=values[2],
        output_tokens=values[3],
        basis="measured",
    )


def _vendor_session(
    kind: str, correlate: str, record: dict[str, Any]
) -> VendorSession | None:
    prefix = "cc" if kind == "claude-code" else "cx"
    run_id = record.get("run_id")
    if run_id != f"{prefix}_{correlate}":
        return None

    started_at = record.get("ts")
    if not isinstance(started_at, str) or not started_at:
        started_at = None

    wall_s = record.get("wall_s")
    if (
        isinstance(wall_s, bool)
        or not isinstance(wall_s, (int, float))
        or not math.isfinite(float(wall_s))
        or wall_s < 0
    ):
        normalized_wall_s = None
    else:
        normalized_wall_s = float(wall_s)

    model = record.get("model")
    if not isinstance(model, str) or not model:
        model = None
    effort = record.get("effort")
    if not isinstance(effort, str) or not effort:
        effort = None

    return VendorSession(
        kind=kind,
        correlate=correlate,
        run_id=run_id,
        started_at=started_at,
        wall_s=normalized_wall_s,
        model=model,
        model_tier=("verified" if kind == "claude-code" else "reported")
        if model is not None
        else None,
        effort=effort,
        usage=_usage(record),
        match_evidence="one correlate matched exactly one parseable vendor session",
    )


class _VendorSessionResolver:
    def __init__(self, claude_code_root: Path, codex_root: Path) -> None:
        self._roots = {
            "claude-code": claude_code_root.expanduser(),
            "codex": codex_root.expanduser(),
        }
        self._indexes: dict[str, dict[str, list[Path]]] = {}
        self._resolved: dict[tuple[str, str], VendorSessionResult] = {}

    def _index(self, kind: str) -> dict[str, list[Path]]:
        previous = self._indexes.get(kind)
        if previous is not None:
            return previous

        index: dict[str, list[Path]] = defaultdict(list)
        root = self._roots[kind]
        if root.is_dir():
            for path in sorted(root.rglob("*.jsonl")):
                if kind == "claude-code":
                    if "subagents" in path.parts:
                        continue
                    correlate = _normalized_correlate(path.stem)
                else:
                    correlate = _codex_session_id(path)
                if correlate is not None:
                    index[correlate].append(path)
        result = dict(index)
        self._indexes[kind] = result
        return result

    def __call__(self, kind: str, correlate: str) -> VendorSessionResult:
        if kind not in self._roots:
            return VendorSessionResult(None, "unsupported_kind")
        normalized = _normalized_correlate(correlate)
        if normalized is None:
            return VendorSessionResult(None, "invalid_correlate")
        key = (kind, normalized)
        if key in self._resolved:
            return self._resolved[key]

        paths = self._index(kind).get(normalized, ())
        if not paths:
            result = VendorSessionResult(None, "not_found")
        elif kind == "claude-code":
            matches: list[VendorSession] = []
            for path in paths:
                try:
                    parsed = claude_code.parse_session(path)
                    record = _as_record(parsed)
                    session = (
                        _vendor_session(kind, normalized, record)
                        if record is not None
                        else None
                    )
                except Exception:
                    session = None
                if session is not None:
                    matches.append(session)
            if len(matches) > 1:
                result = VendorSessionResult(None, "ambiguous")
            elif len(matches) == 1:
                result = VendorSessionResult(matches[0], "resolved")
            else:
                result = VendorSessionResult(None, "unparseable")
        else:
            try:
                parsed = codex.parse_session(list(paths)) if paths else None
                record = _as_record(parsed)
                session = (
                    _vendor_session(kind, normalized, record)
                    if record is not None
                    else None
                )
            except Exception:
                session = None
            result = VendorSessionResult(
                session,
                "resolved" if session is not None else "unparseable",
            )

        self._resolved[key] = result
        return result


class _UsagePricer:
    def __init__(self, price_path: str | Path | None) -> None:
        self._price_path = price_path
        self._table = None

    def __call__(self, model: str | None, usage: Usage | None) -> PricingResult:
        if usage is None:
            return PricingResult(
                usage=None,
                usd=None,
                priced=False,
                reason="vendor session has no complete token measurements",
            )
        if self._table is None:
            self._table = load_prices(self._price_path)
        priced = price_run(
            {
                "model": model,
                "tokens": {
                    "in": usage.input_tokens,
                    "cache_read": usage.cached_input_tokens,
                    "cache_write": usage.cache_creation_tokens,
                    "out": usage.output_tokens,
                },
            },
            self._table,
        )
        is_priced = priced.get("priced") is True
        is_provisional = is_priced and priced.get("todo") is True
        usd = priced.get("usd")
        valid_usd = (
            not isinstance(usd, bool)
            and isinstance(usd, (int, float))
            and math.isfinite(float(usd))
            and usd >= 0
        )
        if is_provisional and valid_usd:
            return PricingResult(
                usage=usage,
                usd=None,
                priced=False,
                reason="price-table entry is provisional",
                provisional=True,
                estimate_usd=float(usd),
            )
        if is_priced and valid_usd:
            return PricingResult(
                usage=usage,
                usd=float(usd),
                priced=True,
                reason=None,
            )
        return PricingResult(
            usage=usage,
            usd=None,
            priced=False,
            reason=str(
                priced.get("reason")
                or (
                    "pricing service returned no valid USD"
                    if is_priced
                    else "unpriced"
                )
            ),
        )


def default_adapter_services(
    *,
    claude_code_root: str | Path | None = None,
    codex_root: str | Path | None = None,
    price_path: str | Path | None = None,
    producer_version: str = __version__,
) -> AdapterServices:
    """Build loopmath's lazily evaluated host services for one adapter instance."""

    resolver = _VendorSessionResolver(
        Path(claude_code_root) if claude_code_root is not None else _CLAUDE_CODE_ROOT,
        Path(codex_root) if codex_root is not None else _CODEX_ROOT,
    )
    return AdapterServices(
        price_usage=_UsagePricer(price_path),
        vendor_session=resolver,
        producer_version=producer_version,
    )
