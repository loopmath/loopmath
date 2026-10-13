"""Explain comparison eligibility using the surface's existing exclusion counts."""
from collections import Counter


COMPARISON = (
    "configuration below min_overlap_cells",
    "run's task-mix cell not shared with another configuration",
    "configuration below min_n",
)
EARLY = ("unknown acceptance", "no usable cost value", "no model label")


def unavailable_lines(surface: dict, n_configs: int = 0) -> list[str]:
    if n_configs == 1:
        return ["only one workflow configuration is eligible, so there is nothing to compare"]
    counts = Counter()
    for item in surface.get("exclusions") or []:
        counts[item.get("reason")] += max(0, int(item.get("n") or 0))
    candidates = sum(counts[r] for r in COMPARISON) + int(surface.get("n_rows_used") or 0)
    choices = [r for r in COMPARISON if counts[r] > 0]
    population = f"{candidates:,} runs with known outcomes, usable cost and model labels"
    if not choices:
        choices = [r for r in EARLY if counts[r] > 0]
        population = f"{int(surface.get('n_rows') or sum(counts.values())):,} runs read"
    if not choices:
        return ["no workflow configuration is eligible, so there is nothing to compare"]
    # A tie follows the fixed presentation order above, never input ordering.
    reason = max(choices, key=counts.__getitem__)
    description, remedy = {
        COMPARISON[0]: ("their configurations lack enough shared task-mix cells (overlap)",
                        "read more comparable history with shared task-mix cells (--all or a larger --since)"),
        COMPARISON[1]: ("their task-mix cells are not shared with another configuration",
                        "read more comparable history with shared task-mix cells (--all or a larger --since)"),
        COMPARISON[2]: (f"their configurations fall below the minimum run count (--min-n {surface.get('min_n', 5)})",
                        "read more history (--all or a larger --since), or lower the minimum (--min-n 3)"),
        EARLY[0]: ("they have unknown acceptance outcomes", "see which evidence counted as acceptance (--grading)"),
        EARLY[1]: ("they have no usable cost", "check missing prices or token counts for the selected cost basis"),
        EARLY[2]: ("they have no model label", "check that the imported history records model names"),
    }[reason]
    lines = [f"no workflow configuration is eligible; the largest exclusion is {counts[reason]:,} of {population}: {description}"]
    early = [f"{counts[r]:,} {r}" for r in EARLY if counts[r]]
    if reason in COMPARISON and early:
        lines.append("before comparison: " + ", ".join(early))
    lines.append("to get a table: " + remedy)
    return lines
