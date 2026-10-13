"""Find concrete holes in task-cell and grading coverage.

The inputs are deliberately small summaries rather than run records:

``surface``
    A mapping from an internal arm name to the task cells in which it was
    observed.  Cell values may be any iterable (duplicates do not matter).

``coverage``
    A mapping from an internal arm name to ``{evidence_tier: count}``.

Keeping these inputs as maps makes gap detection usable before a cost model
can be fitted.  In particular, a configuration with no observations still
has an entry in one or both maps and can therefore be named rather than
silently disappearing from a fitted table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping, TypeAlias


_STRONG_TIERS = frozenset({"verified", "reported"})


@dataclass(frozen=True, slots=True)
class UnvisitedArm:
    """A workflow configuration with no observed task cell or grade."""

    arm: str
    description: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "description",
            f"Workflow configuration {self.arm} has no observed runs.",
        )


@dataclass(frozen=True, slots=True)
class ConfoundedPair:
    """Two observed configurations with too little shared task-cell coverage."""

    arm_a: str
    arm_b: str
    shared_cells: int
    min_overlap_cells: int
    description: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "description",
            (
                f"Workflow configurations {self.arm_a} and {self.arm_b} share "
                f"{self.shared_cells} {_plural(self.shared_cells, 'task cell')}, below "
                f"the required {self.min_overlap_cells}."
            ),
        )


@dataclass(frozen=True, slots=True)
class ObserverLimited:
    """A configuration graded only with evidence below the two strongest tiers."""

    arm: str
    tiers: tuple[str, ...]
    n_grades: int
    description: str = field(init=False)

    def __post_init__(self) -> None:
        evidence = ", ".join(self.tiers)
        object.__setattr__(
            self,
            "description",
            f"Workflow configuration {self.arm} has grades only from {evidence} evidence.",
        )


Gap: TypeAlias = UnvisitedArm | ConfoundedPair | ObserverLimited


def _counted_tiers(tier_counts: Mapping[str, int]) -> dict[str, int]:
    """Return positive, integer tier counts and reject ambiguous summaries."""
    counted: dict[str, int] = {}
    for tier, count in tier_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("evidence-tier counts must be non-negative integers")
        if count:
            counted[str(tier)] = count
    return counted


def _cell_set(cells: Iterable[str]) -> set[str]:
    """Materialize one configuration's cells with useful input errors."""
    if isinstance(cells, (str, bytes)):
        raise TypeError("task cells must be an iterable of cell names, not one string")
    return {str(cell) for cell in cells}


def _plural(count: int, noun: str) -> str:
    return noun if count == 1 else f"{noun}s"


def detect_gaps(
    surface: Mapping[str, Iterable[str]],
    coverage: Mapping[str, Mapping[str, int]],
    *,
    min_overlap_cells: int = 2,
) -> list[Gap]:
    """Detect unvisited, task-cell-confounded, and observer-limited gaps.

    Every key present in either input is considered a candidate workflow
    configuration.  A candidate is unvisited only when it has neither a task
    cell nor a grade.  Pairwise overlap is measured between visited
    candidates, and a pair is confounded when the number of shared task cells
    is below ``min_overlap_cells``.  A visited candidate is observer-limited
    when it has at least one grade but none at the ``verified`` or
    ``reported`` tiers.

    Results are deterministic: each kind is ordered by configuration name,
    with unvisited gaps first, then pairs, then observer limitations.
    Descriptions are one sentence each and use only reader-facing workflow
    language.
    """
    if isinstance(min_overlap_cells, bool) or not isinstance(min_overlap_cells, int):
        raise TypeError("min_overlap_cells must be an integer")
    if min_overlap_cells < 1:
        raise ValueError("min_overlap_cells must be at least 1")

    arms = sorted(set(surface) | set(coverage))
    cells_by_arm = {arm: _cell_set(surface.get(arm, ())) for arm in arms}
    tiers_by_arm = {arm: _counted_tiers(coverage.get(arm, {})) for arm in arms}

    visited = [
        arm for arm in arms
        if cells_by_arm[arm] or sum(tiers_by_arm[arm].values()) > 0
    ]
    gaps: list[Gap] = []

    for arm in arms:
        if arm not in visited:
            gaps.append(UnvisitedArm(arm=arm))

    for arm_a, arm_b in combinations(visited, 2):
        shared_cells = len(cells_by_arm[arm_a] & cells_by_arm[arm_b])
        if shared_cells < min_overlap_cells:
            gaps.append(
                ConfoundedPair(
                    arm_a=arm_a,
                    arm_b=arm_b,
                    shared_cells=shared_cells,
                    min_overlap_cells=min_overlap_cells,
                )
            )

    for arm in visited:
        tier_counts = tiers_by_arm[arm]
        n_grades = sum(tier_counts.values())
        if n_grades and not (_STRONG_TIERS & tier_counts.keys()):
            tiers = tuple(sorted(tier_counts))
            gaps.append(
                ObserverLimited(
                    arm=arm,
                    tiers=tiers,
                    n_grades=n_grades,
                )
            )

    return gaps


__all__ = [
    "ConfoundedPair",
    "Gap",
    "ObserverLimited",
    "UnvisitedArm",
    "detect_gaps",
]
