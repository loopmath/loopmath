"""Plain-language rendering for a proposed workflow experiment.

This module is deliberately standalone so report code can use it without
pulling in data loading, fitting, or reporting dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite


@dataclass(frozen=True)
class ExperimentCandidate:
    """The values needed to explain one proposed comparison.

    ``upside_per_accepted`` and ``price`` are dollar amounts. ``reuse_volume``
    is the number of accepted results expected in a typical month.
    """

    configuration_a: str
    configuration_b: str
    upside_per_accepted: float
    price: float
    reuse_volume: int


def _money(value: float | Decimal) -> str:
    """Format a finite dollar amount with cents and grouped thousands."""
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("dollar amounts must be finite numbers") from None
    if not amount.is_finite():
        raise ValueError("dollar amounts must be finite numbers")
    return f"${amount:,.2f}"


def _validate(candidate: ExperimentCandidate) -> None:
    if not candidate.configuration_a.strip() or not candidate.configuration_b.strip():
        raise ValueError("configuration names must not be empty")
    if candidate.configuration_a == candidate.configuration_b:
        raise ValueError("configuration names must be different")

    for value, label in (
        (candidate.upside_per_accepted, "upside_per_accepted"),
        (candidate.price, "price"),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
            raise TypeError(f"{label} must be a number")
        if not isfinite(float(value)) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")

    if isinstance(candidate.reuse_volume, bool) or not isinstance(candidate.reuse_volume, int):
        raise TypeError("reuse_volume must be an integer")
    if candidate.reuse_volume < 0:
        raise ValueError("reuse_volume must be non-negative")


def render(candidate: ExperimentCandidate) -> str:
    """Render the report's plain-language pitch for ``candidate``.

    The monthly figure is an estimate: dollar upside per accepted result
    multiplied by the expected number of accepted results reused each month.
    The function performs no I/O and does not mutate its input.
    """
    _validate(candidate)
    monthly_upside = Decimal(str(candidate.upside_per_accepted)) * candidate.reuse_volume

    return "\n".join(
        [
            "The one experiment worth buying",
            (
                f"What to test: compare {candidate.configuration_a} with "
                f"{candidate.configuration_b}."
            ),
            f"What it costs: {_money(candidate.price)}.",
            (
                f"What it could save monthly: {_money(monthly_upside)}, based on "
                f"{_money(candidate.upside_per_accepted)} per accepted result reused "
                f"{candidate.reuse_volume:,} times each month."
            ),
        ]
    )

