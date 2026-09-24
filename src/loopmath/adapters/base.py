"""Common contract implemented by every ADE-to-OCP adapter."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Mapping, Sequence


@dataclass(frozen=True)
class Selection:
    """Common filters forwarded by :command:`loopmath adapt`.

    Empty tuples mean "all". ``since`` and ``until`` are inclusive RFC 3339
    bounds. Adapters must apply a filter only when the source has the field
    needed to evaluate it; any resulting unknown or omitted counts belong in
    the adapter's documented OCP extension namespace.
    """

    stores: tuple[Path, ...] = ()
    session_ids: tuple[str, ...] = ()
    workspaces: tuple[str, ...] = ()
    since: str | None = None
    until: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class Usage:
    """OCP-ready token measurements for one vendor attempt.

    The field names deliberately match OCP v0.2 ``attempt.cost``.  Values are
    exact non-negative integer measurements; missing or malformed source usage
    is represented by ``None`` at the containing :class:`VendorSession`, never
    by a zero-filled ``Usage``.
    """

    input_tokens: int
    cached_input_tokens: int
    cache_creation_tokens: int
    output_tokens: int
    basis: str = "measured"

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_tokens",
            "output_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.basis not in {"measured", "allocated"}:
            raise ValueError("basis must be 'measured' or 'allocated'")

    def to_ocp(self, *, usd: float | None = None) -> dict[str, Any]:
        """Return a fresh OCP v0.2 ``attempt.cost`` dictionary."""

        cost: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "output_tokens": self.output_tokens,
            "basis": self.basis,
        }
        if usd is not None:
            if isinstance(usd, bool) or not isinstance(usd, (int, float)):
                raise ValueError("usd must be a finite non-negative number")
            normalized = float(usd)
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError("usd must be a finite non-negative number")
            cost["usd"] = normalized
        return cost


@dataclass(frozen=True)
class PricingResult:
    """Normalized result of asking the host to price :class:`Usage`."""

    usage: Usage | None
    usd: float | None
    priced: bool
    reason: str | None
    provisional: bool = False
    estimate_usd: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.priced, bool):
            raise TypeError("priced must be a bool")
        if not isinstance(self.provisional, bool):
            raise TypeError("provisional must be a bool")
        if self.priced:
            if (
                self.usage is None
                or self.usd is None
                or self.provisional
                or self.estimate_usd is not None
                or self.reason is not None
            ):
                raise ValueError(
                    "a priced result requires settled usd and no provisional fields"
                )
            # Reuse Usage's OCP-number validation for the dollar value.
            self.usage.to_ocp(usd=self.usd)
        elif self.provisional:
            if self.usage is None or self.usd is not None or self.estimate_usd is None:
                raise ValueError(
                    "a provisional result requires usage and estimate_usd but no usd"
                )
            if self.reason != "price-table entry is provisional":
                raise ValueError(
                    "a provisional result requires the standard provisional reason"
                )
            self.usage.to_ocp(usd=self.estimate_usd)
        else:
            if self.usd is not None or self.estimate_usd is not None:
                raise ValueError("an unpriced result must not carry monetary values")
            if not isinstance(self.reason, str) or not self.reason:
                raise ValueError("an unpriced result requires a reason")

    def to_ocp(self) -> dict[str, Any] | None:
        """Return OCP ``attempt.cost``, retaining tokens when USD is unknown."""

        if self.usage is None:
            return None
        return self.usage.to_ocp(usd=self.usd if self.priced else None)


@dataclass(frozen=True)
class VendorSession:
    """One uniquely resolved vendor session in adapter-safe form.

    ``correlate`` is the exact opaque value supplied by the adapter after host
    normalization.  ``match_evidence`` describes only the host-side match;
    the adapter remains responsible for evidence joining its own source to the
    correlate.  No source path, parser record, or transcript content crosses
    this boundary.
    """

    kind: str
    correlate: str
    run_id: str
    started_at: str | None
    wall_s: float | None
    model: str | None
    model_tier: str | None
    effort: str | None
    usage: Usage | None
    match_evidence: str

    def __post_init__(self) -> None:
        for name in ("kind", "correlate", "run_id", "match_evidence"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.started_at is not None and not isinstance(self.started_at, str):
            raise TypeError("started_at must be a string or None")
        if self.wall_s is not None:
            if isinstance(self.wall_s, bool) or not isinstance(
                self.wall_s, (int, float)
            ):
                raise TypeError("wall_s must be a finite non-negative number or None")
            if not math.isfinite(float(self.wall_s)) or self.wall_s < 0:
                raise ValueError("wall_s must be a finite non-negative number or None")
        for name in ("model", "effort"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if self.model_tier not in {None, "verified", "reported", "heuristic"}:
            raise ValueError("model_tier must be an OCP evidence tier or None")
        if self.model is None and self.model_tier is not None:
            raise ValueError("model_tier requires a model")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise TypeError("usage must be Usage or None")


VendorSessionReason = Literal[
    "resolved",
    "service_unavailable",
    "unsupported_kind",
    "invalid_correlate",
    "not_found",
    "unparseable",
    "ambiguous",
]
VENDOR_SESSION_REASONS: tuple[VendorSessionReason, ...] = (
    "resolved",
    "service_unavailable",
    "unsupported_kind",
    "invalid_correlate",
    "not_found",
    "unparseable",
    "ambiguous",
)


@dataclass(frozen=True)
class VendorSessionResult:
    """Result of resolving a vendor correlate without losing failure cause."""

    session: VendorSession | None
    reason: VendorSessionReason

    def __post_init__(self) -> None:
        if self.reason not in VENDOR_SESSION_REASONS:
            raise ValueError(f"unknown vendor-session reason: {self.reason!r}")
        if self.reason == "resolved" and self.session is None:
            raise ValueError("a resolved result requires a session")
        if self.reason != "resolved" and self.session is not None:
            raise ValueError("an unresolved result must not carry a session")


PriceUsageHook = Callable[[str | None, Usage | None], PricingResult]
VendorSessionHook = Callable[[str, str], VendorSessionResult]


@dataclass(frozen=True)
class AdapterServices:
    """Optional host callables and producer version injected into an adapter."""

    price_usage: PriceUsageHook | None = None
    vendor_session: VendorSessionHook | None = None
    producer_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("price_usage", "vendor_session"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        if self.producer_version is not None and (
            not isinstance(self.producer_version, str) or not self.producer_version
        ):
            raise ValueError("producer_version must be a non-empty string or None")


class Adapter(ABC):
    """Base class for one ADE store-to-OCP v0.2 converter.

    Registered adapter classes must set a canonical ``name`` and be
    constructible without arguments.  The host may inject
    :class:`AdapterServices`; without them the adapter stays usable with the
    explicit unavailable behavior documented on the helper methods below.
    Methods are read-only with respect to the source store. ``discover`` and
    ``sessions`` return deterministic sequences; a session mapping always has
    a stable string ``id`` and may expose other metadata useful to callers.
    """

    name: ClassVar[str]

    def __init__(self, services: AdapterServices | None = None, /) -> None:
        self._services = services if services is not None else AdapterServices()

    def configure_services(self, services: AdapterServices, /) -> None:
        """Inject host services after no-argument adapter construction.

        The CLI uses this path so a registered adapter that implements its own
        no-argument ``__init__`` remains compatible.
        """

        if not isinstance(services, AdapterServices):
            raise TypeError("services must be AdapterServices")
        self._services = services

    def _host_services(self) -> AdapterServices:
        # A legacy subclass may implement a no-argument __init__ without
        # calling super(); retain explicit unavailable behavior in that case.
        services = getattr(self, "_services", None)
        return services if isinstance(services, AdapterServices) else AdapterServices()

    def price_usage(self, model: str | None, usage: Usage | None, /) -> PricingResult:
        """Price normalized usage, or return an explicit unpriced result.

        A missing service never discards valid token measurements.  Host hook
        exceptions and malformed return types remain visible to the caller.
        """

        if model is not None and not isinstance(model, str):
            raise TypeError("model must be a string or None")
        if usage is not None and not isinstance(usage, Usage):
            raise TypeError("usage must be Usage or None")
        hook = self._host_services().price_usage
        if hook is None:
            return PricingResult(
                usage=usage,
                usd=None,
                priced=False,
                reason="host pricing service is unavailable",
            )
        result = hook(model, usage)
        if not isinstance(result, PricingResult):
            raise TypeError("host pricing hook must return PricingResult")
        if result.usage != usage:
            raise ValueError("host pricing hook must preserve normalized usage")
        return result

    def resolve_vendor_session(
        self, kind: str, correlate: str, /
    ) -> VendorSessionResult:
        """Resolve one vendor correlate with a stable machine-readable reason.

        A resolved session repeats the requested kind and carries the
        host-normalized correlate.  Parser and pricing implementation types
        never cross this boundary.
        """

        if not isinstance(kind, str) or not isinstance(correlate, str):
            raise TypeError("kind and correlate must be strings")
        hook = self._host_services().vendor_session
        if hook is None:
            return VendorSessionResult(None, "service_unavailable")
        result = hook(kind, correlate)
        if not isinstance(result, VendorSessionResult):
            raise TypeError("host vendor-session hook must return VendorSessionResult")
        if result.session is not None and result.session.kind != kind:
            raise ValueError("host vendor-session hook returned a different kind")
        return result

    def producer_version(self) -> str | None:
        """Return the explicitly injected host producer version, if any."""

        return self._host_services().producer_version

    def fixture_selection(
        self, fixture_dir: Path, /
    ) -> AbstractContextManager[Selection]:
        """Yield the selection used to emit one synthetic fixture.

        Overrides may materialize temporary stores whose lifetime extends
        through the adapter's ``emit`` call. The default passes the fixture
        directory through as the only selected store.
        """

        return nullcontext(Selection(stores=(fixture_dir,)))

    @abstractmethod
    def discover(self) -> Sequence[Path]:
        """Return the existing default store paths found on this machine."""

    @abstractmethod
    def sessions(self) -> Sequence[Mapping[str, Any]]:
        """List session descriptors from the discovered default stores."""

    @abstractmethod
    def emit(self, selection: Selection) -> dict[str, Any]:
        """Return one conforming OCP v0.2 document for ``selection``."""
