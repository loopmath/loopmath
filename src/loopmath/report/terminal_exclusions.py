"""Skip-reason wording for terminal report exclusions."""

from __future__ import annotations

from .terminal_format import _fmt_int


_UNKNOWN_ACCEPTANCE_REASON = "unknown acceptance"
_SKIP_REASON_VERB = {"no assistant turns": "had"}
_DEFAULT_SKIP_REASON_VERB = "were"


def _counted_exclusion_detail(n: int, reason: str, detail: str) -> str:
    """Make a row-level exclusion detail agree with its printed count."""
    singular = n == 1
    if reason == "configuration below min_n":
        threshold = detail.removesuffix(" in this workflow configuration")
        noun = (
            "run belongs to a workflow configuration"
            if singular
            else "runs belong to workflow configurations"
        )
        return f"{noun} with {threshold}"
    if reason == "configuration below min_overlap_cells":
        condition = detail.removeprefix("workflow configuration ")
        if not singular:
            condition = condition.replace("does not ", "do not ", 1)
        noun = (
            "run belongs to a workflow configuration that"
            if singular
            else "runs belong to workflow configurations that"
        )
        return f"{noun} {condition}"
    if singular:
        return detail
    plural = (
        detail.replace("run has ", "runs have ", 1)
        .replace("run could ", "runs could ", 1)
        .replace("run is ", "runs are ", 1)
    )
    return (
        plural.replace(", so it has ", ", so they have ")
        .replace(", so it is ", ", so they are ")
        .replace(", so it cannot ", ", so they cannot ")
        .replace(" and is never ", " and are never ")
    )


def _skip_reasons_line(skip_reasons: dict) -> str | None:
    """"of which N <verb> <reason>, ..." in descending count order, or None.

    `skip_reasons` comes straight from `ingest_diag["skip_reasons"]`
    (`loopmath.ingest.parse_all`'s measured classification of every skipped
    file); an empty or missing dict means no cause was measured, so this
    returns None rather than asserting one. Ties in count are broken by the
    reason text so the line is deterministic run to run.
    """
    if not skip_reasons:
        return None
    ordered = sorted(
        ((reason, n) for reason, n in skip_reasons.items() if n),
        key=lambda kv: (-kv[1], kv[0]),
    )
    if not ordered:
        return None
    parts = [
        f"{_fmt_int(n)} {_SKIP_REASON_VERB.get(reason, 'was' if n == 1 else _DEFAULT_SKIP_REASON_VERB)} {reason}"
        for reason, n in ordered
    ]
    return "of which " + ", ".join(parts)
