"""One acceptance-rule resolver for the fit, `run finish` and its receipts, the share export and the runs view.

Before 0.2.7 the four readers each parsed `run.acceptance_rule` their own way. They now all call
`resolve_rule` and `evidence`, which give the fit's answers:

- `stored_rule(doc)` is the rule the fit has always read (`design.rule_from_doc`): a rule with no name is the
  default, `window_days: 0` is 14 days and an empty `excludes_events` is revert and incident.
- The review case: a run onboarded from the logs, or stored by `run import` from 0.2.7, whose stored rule is
  absent or the default (named `tests`, requiring exactly `tests`, no score), that has no `tests` verdict and
  has a `review` or `accepted` verdict, is read by the review rule. So is a run that `loopmath outcome --rule
  review` moved, whose stored rule carries the review mark.
- The review rule's reading: the latest `review` or `accepted` verdict, by observed time. Accept, approve and
  pass are a success, reject and fail are not; any other value gives no outcome, with the reason
  `unknown:review=<value>`. Late events, the window and q work as for any rule.

Only a rule with the review mark (`ext["dev.loopmath.rule"] = {"reading": "review"}`) is read this way, so a
user's own rule named `review` keeps its reading.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

from ..types import DEFAULT_RULE, AcceptanceRule, Evidence
from .design import rule_from_doc
from .outcome import _last, outcome_evidence

RULE_EXT = "dev.loopmath.rule"
IMPORT_EXT = "dev.loopmath.import"
STORED_BY_IMPORT = "run import"
REVIEW_COMMAND = "loopmath outcome --rule review"
REVIEW_NAMES = ("review", "accepted")
REVIEW_PASS = ("accept", "approve", "pass")
REVIEW_FAIL = ("reject", "fail")
REVIEW_RULE = AcceptanceRule(
    name="review", definition="the latest review or accepted verdict is accept, approve or pass",
    requires=("review",), extra={"ext": {RULE_EXT: {"reading": "review"}}})


def stored_rule(doc: dict) -> AcceptanceRule:
    """The rule recorded on the run as the fit reads it. Raises ValueError or TypeError on a malformed score."""
    return rule_from_doc(doc)


def is_review_rule(rule: AcceptanceRule | dict | None) -> bool:
    """A rule that carries the review mark (a rule object, or a rule as stored on a run)."""
    if isinstance(rule, AcceptanceRule):
        ext = rule.extra.get("ext")
    elif isinstance(rule, dict):
        ext = rule.get("ext")
    else:
        return False
    mark = ext.get(RULE_EXT) if isinstance(ext, dict) else None
    return isinstance(mark, dict) and mark.get("reading") == "review"


def review_rule_dict() -> dict[str, Any]:
    """The review rule as `outcome --rule review` stores it on a run."""
    return {k: v for k, v in REVIEW_RULE.to_dict().items() if v is not None}


def _verdicts(doc: dict) -> list[dict]:
    signals = (doc.get("run") or {}).get("signals") or []
    return [s for s in signals if isinstance(s, dict) and s.get("kind") == "verdict"]


def is_marked_import(doc: dict) -> bool:
    ext = (doc.get("run") or {}).get("ext")
    mark = ext.get(IMPORT_EXT) if isinstance(ext, dict) else None
    return isinstance(mark, dict) and mark.get("stored_by") == STORED_BY_IMPORT


def has_default_rule(doc: dict) -> bool:
    """The stored rule is absent, or named `tests` and requiring exactly `tests`, with no score."""
    try:
        rule = stored_rule(doc)
    except (TypeError, ValueError):
        return False
    return rule.name == "tests" and tuple(rule.requires) == ("tests",) and rule.score is None


def review_only(doc: dict) -> bool:
    """No `tests` verdict, and a `review` or `accepted` one."""
    names = {s.get("name") for s in _verdicts(doc)}
    return "tests" not in names and bool(names & set(REVIEW_NAMES))


def in_review_case(doc: dict) -> bool:
    from ..store.runs import is_history

    return (is_history(doc) or is_marked_import(doc)) and has_default_rule(doc) and review_only(doc)


def could_use_review_rule(doc: dict, rule: AcceptanceRule, z: float | None) -> bool:
    """A run `loopmath outcome --rule review` could help, as `runs` lists it: finished, read under another rule
    with no outcome, with no `tests` verdict and a `review` or `accepted` one."""
    from ..store.runs import is_finished

    return z is None and not is_review_rule(rule) and is_finished(doc) and review_only(doc)


def resolve_rule(doc: dict) -> AcceptanceRule:
    """The rule every reader applies to this run. Raises as `stored_rule` does on a malformed score."""
    raw = (doc.get("run") or {}).get("acceptance_rule")
    if is_review_rule(raw) or in_review_case(doc):
        return REVIEW_RULE
    return stored_rule(doc)


def resolve_rule_or_default(doc: dict) -> AcceptanceRule:
    """`resolve_rule`, with the default rule for a malformed stored rule (receipts and views; the fit drops such a
    run instead)."""
    try:
        return resolve_rule(doc)
    except (TypeError, ValueError):
        return DEFAULT_RULE


def evidence(doc: dict, rule: AcceptanceRule, *, now: datetime,
             q_by_tier: dict[str, float] | None = None) -> Evidence:
    """The run's outcome under `rule`: the review reading for the review rule, `outcome_evidence` otherwise."""
    if not is_review_rule(rule):
        return outcome_evidence(doc, rule, now=now, q_by_tier=q_by_tier)
    run = doc.get("run") or {}
    signals = [s for s in run.get("signals") or [] if isinstance(s, dict)]
    named = [s for s in signals if s.get("kind") == "verdict" and s.get("name") in REVIEW_NAMES]
    latest = _last(named)
    value = str(latest.get("value") or "").lower() if latest is not None else None
    reading = "accept" if value in REVIEW_PASS else "reject" if value in REVIEW_FAIL else None
    kept = [s for s in signals if not (s.get("kind") == "verdict" and s.get("name") in REVIEW_NAMES)]
    if reading is not None:
        kept.append({**latest, "name": "review", "value": reading})
    ev = outcome_evidence({**doc, "run": {**run, "signals": kept}}, rule, now=now, q_by_tier=q_by_tier)
    if latest is not None and reading is None:
        reasons = tuple(f"unknown:review={value}" if r == "missing:review" else r for r in ev.reasons)
        ev = dataclasses.replace(ev, reasons=reasons)
    return ev
