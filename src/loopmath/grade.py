"""Evidence-tier acceptance labeling for parsed run records (SPEC section 4).

Honest about what this module can and cannot see: it never opens a log file
and never re-derives anything from prompt or diff text. It works on the
`_signals` dict a parser already extracted (`RunRecord.grading_signals()`),
the SPEC section 3 record body (`workspace`, `touched_dirs`, `ts`, `wall_s`),
and optional `diff_snapshots` supplied by a caller that captured file content
at session end. Every tier below is a heuristic over that evidence, not proof: a
`verified` run has a machine-checked signal (a test flip, a gate record, an
explicit check exit) in the session; a `heuristic` run only has a clean exit
and the absence of visible rework, which is a weaker claim than it sounds
(a human could still have silently discarded the work outside the window
this module can see). `grading_report` and `coverage_line` print every tier,
including the ones with the least evidence, on purpose: SPEC section 0's full
honesty rule means we never let a caller mistake an `asserted` run for a
`verified` one.

attempt_rule v1 (SPEC section 3) is also implemented here because it shares
the same session-ordering and touched-directory-overlap machinery the R5
censoring rule and the "heuristic" tier's follow-up-fix-session check need.

The follow-up-session check remains directory-level because `ingest/base.py`'s
privacy contract stores only `touched_dirs`. The diff-survival check becomes
file-level when optional snapshots are present: Git follows renames when it
can, with a path-and-content fallback otherwise. Without snapshots, survival
retains the directory-level compatibility heuristic. Consequently, the
follow-up check can still over-trigger whenever two unrelated pieces of work
touch the same directory within `LOOKBACK_S`, reading as "the same work,
reworked" when it is not.
`LOOKBACK_S` itself is a chosen parameter, not a discovered one; `grading_report`
prints both the window and how many records the follow-up test demoted so
this tradeoff is visible in the output, not just in this comment.

Three signals this ladder consumes -- `gate_record`, `reviewer_verdict`,
`contract_run_json` -- are structurally unreachable through either live
parser today, and that is checked, not assumed: a read-only scan of the real
Claude Code and Codex session logs for a typed result envelope, a reviewer-
verdict shape (including a structured PR-review state, e.g. a `gh` JSON
response embedded in tool output), or any other structured completion record
found none. All three describe a SEPARATE artifact, loopmath's own contract-v3
`run.json` (SPEC section 3): a session log can be seen writing a file whose
path ends in `.dagr/run.json` (a real, observed shape, via the same write
metadata the parsers already extract), which is as far as reading only the
session log can honestly go, but the gate list, reviewer verdict, and
explicit attempt number all live INSIDE that file's contents, and opening a
second file is a new discovery source, out of scope for tonight's patch by
the same reasoning that keeps `ingest/__init__.py` scanning only `*.jsonl`.
So `reported` reads as 0 (and `verified`'s gate-record path never fires)
because the evidence lives in a file this pass does not open, not because no
review or gating work happened; `grading_report` says so explicitly (see the
line below the coverage line) so a reader does not mistake the zero for a
measurement of the work.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from loopmath.ingest.base import ts_epoch

from . import grade_support as _grade_support
from .grade_support import (
    CENSOR_MAX_WALL_S,
    LOOKBACK_S,
    _FOLLOWUP_REWORKED_SIGNAL,
    _Session,
    _TIERS,
    _git_rename_target,
    _follow_up_fix_session,
    _get_signals,
    _grouped_by_workspace,
    _record_diff_survives,
    _repository_root,
    _touched,
    assign_attempts,
    diff_survives,
)


_SUPPORT_REBINDINGS = frozenset(
    {
        "LOOKBACK_S",
        "Path",
        "subprocess",
        "ts_epoch",
        "_Session",
        "_follow_up_fix_session",
        "_get_signals",
        "_git_rename_target",
        "_grouped_by_workspace",
        "_record_diff_survives",
        "_repository_root",
        "_touched",
        "assign_attempts",
        "diff_survives",
    }
)


class _GradeModule(ModuleType):
    """Keep re-exported support globals responsive to grade-module rebinding."""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in _SUPPORT_REBINDINGS:
            setattr(_grade_support, name, value)


sys.modules[__name__].__class__ = _GradeModule


def _heuristic_unevaluable_reason(record: dict, context: dict | None) -> str | None:
    """None when the heuristic tier's preconditions hold; otherwise, why not.

    The heuristic tier claims positive evidence ("no follow-up fix session,
    work survived"), and that claim can only be made when the evidence is
    actually observable. So it may only fire when ALL of: a grading context
    was supplied (so later sessions in the workspace are visible), the
    record has a non-null `ts`, and the record has a non-empty
    `touched_dirs` (with no touched directories there is no "same files" to
    check for a re-touch).
    """
    if context is None:
        return "clean exit, but survival could not be checked: no later sessions visible"
    if record.get("ts") is None:
        return "clean exit, but survival could not be checked: no timestamp recorded"
    if not record.get("touched_dirs"):
        return "clean exit, but survival could not be checked: no directories recorded"
    return None


def grade_run(record: dict, context: dict | None = None) -> dict:
    """Grade one record. `context` optionally carries same-workspace neighbors.

    `context = {"sessions": [_Session, ...], "index": i}` lets the heuristic
    tier see later sessions in the same workspace. The heuristic tier may
    only fire when that context is present AND the record has a `ts` AND the
    record has a non-empty `touched_dirs` (see `_heuristic_unevaluable_reason`):
    without all three, the "no follow-up fix session, work survived" claim
    cannot be checked, so the tier does not fire, the record falls through to
    `asserted` / `ungraded` as appropriate, and `signal` records why.
    """
    sig = _get_signals(record)

    tier = "ungraded"
    accepted: bool | None = None
    signal = "no acceptance signal in log"

    heuristic_reason = (
        _heuristic_unevaluable_reason(record, context) if sig["exit_ok"] is True else None
    )

    # --- Tier 1: verified. Strongest wins if several verified signals fire. ---
    if sig["tests_red_to_green"]:
        tier, accepted, signal = "verified", True, "tests went red to green in session"
    elif sig["gate_record"]:
        tier, accepted, signal = "verified", True, "gate record"
    elif sig["check_exit_zero"]:
        tier, accepted, signal = "verified", True, "check exited 0"

    # --- Tier 2: reported. ---
    elif sig["reviewer_verdict"]:
        verdict = sig["reviewer_verdict"]
        tier = "reported"
        accepted = verdict == "approved"
        signal = f"reviewer verdict recorded: {verdict}"

    # --- Tier 3: heuristic. Only when exit_ok True AND preconditions hold. ---
    elif sig["exit_ok"] is True and heuristic_reason is None:
        tier = "heuristic"
        sessions = context["sessions"]
        index = context["index"]
        followed_up = _follow_up_fix_session(index, sessions)
        survived = _record_diff_survives(record)
        if followed_up:
            accepted = False
            signal = _FOLLOWUP_REWORKED_SIGNAL
        elif survived is False:
            accepted = False
            signal = "clean exit, but diff was reverted"
        else:
            accepted = True
            signal = "clean exit, no follow-up fix session, work survived"

    # --- Tier 4: asserted. ---
    elif sig["agent_claims_success"]:
        tier, accepted, signal = "asserted", None, "agent asserted success, nothing corroborates"

    # --- Tier 5: ungraded (defaults above already set) ---

    # exit_ok True but the heuristic tier's preconditions weren't met: the
    # record still lands on whatever the rest of the ladder gives it
    # (asserted or ungraded), but the signal explains why heuristic could
    # not fire instead of leaving that fact unstated.
    if heuristic_reason is not None and tier in ("asserted", "ungraded"):
        signal = heuristic_reason

    # --- R5 censoring: checked after the ladder, only demotes accepted=None. ---
    wall_s = record.get("wall_s")
    if accepted is None and wall_s is not None and wall_s < CENSOR_MAX_WALL_S:
        tier = "censored"
        accepted = None
        signal = "instant retry under 60 s with no approval signal"

    return {"accepted": accepted, "tier": tier, "signal": signal}


def grade_all(records: list[dict]) -> tuple[list[dict], dict]:
    """Assign attempts, grade every record, attach record["grade"].

    Returns (records, coverage) where coverage is the SPEC section 4 mandatory
    summary: tier counts, structural coverage percentage, and the acceptance
    denominator (graded records only; censored/asserted/ungraded excluded).

    Flag 6: zero-token synthetic sessions (`_signals["zero_token_synthetic"]`,
    set by claude_code.py from the literal `model: "<synthetic>"` marker) are
    removed from the population before grading, not merely excluded from the
    printed acceptance denominator: they are the harness's own bookkeeping
    turns, never a priced or gradeable unit of work, and measured at 1,526 of
    4,418 parsed records (34.5%) in the real estate -- almost all of the
    censored tier's own population. Left in, they inflate `n_total` (the
    denominator both `pct_structural` and the mandatory coverage line's "of M
    runs" describe) and supply most of the R5-censored count for a reason
    that has nothing to do with instant retries. `coverage["n_synthetic_excluded"]`
    carries the count so it is never silently subtracted; the returned
    `records` list also excludes them, so nothing downstream (pricing, the
    cost surface) counts a $0.00 bookkeeping row as a run either.
    """
    real_records = []
    n_synthetic_excluded = 0
    for r in records:
        if _get_signals(r)["zero_token_synthetic"]:
            n_synthetic_excluded += 1
        else:
            real_records.append(r)

    out = assign_attempts(real_records)
    groups = _grouped_by_workspace(out)

    # Grade in workspace/ts context so the heuristic tier can see follow-up
    # sessions, but write results back into `out` at the original index.
    graded_by_index: dict = {}
    n_unevaluable_heuristic = 0
    for _ws, sessions in groups.items():
        ordered = sorted(sessions, key=lambda s: (s.epoch is None, s.epoch))
        for i, sess in enumerate(ordered):
            context = {"sessions": ordered, "index": i}
            g = grade_run(sess.record, context=context)
            graded_by_index[sess.index] = g

            sig = _get_signals(sess.record)
            if sig["exit_ok"] is True and _heuristic_unevaluable_reason(sess.record, context):
                n_unevaluable_heuristic += 1

    tiers = {t: 0 for t in _TIERS}
    n_accepted = 0
    n_rejected = 0
    n_unknown = 0
    n_graded = 0

    for i, rec in enumerate(out):
        g = graded_by_index[i]
        rec["grade"] = g
        tiers[g["tier"]] += 1
        if g["accepted"] is None:
            n_unknown += 1
        elif g["accepted"] is True:
            n_accepted += 1
            n_graded += 1
        else:
            n_rejected += 1
            n_graded += 1

    n_total = len(out)
    # Full honesty (SPEC section 0): every record lands in exactly one tier,
    # so the five printed tiers plus "ungraded" must always sum to n_total.
    # No record may go missing from the count.
    assert sum(tiers.values()) == n_total, "tier counts do not cover every record"
    pct_structural = 100.0 * n_graded / n_total if n_total else 0.0
    denominator = n_accepted + n_rejected  # graded records only, per SPEC section 4

    coverage = {
        "n_total": n_total,
        "n_graded": n_graded,
        "pct_structural": pct_structural,
        "tiers": tiers,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "n_unknown": n_unknown,
        "denominator": denominator,
        "unevaluable_heuristic": n_unevaluable_heuristic,
        "n_synthetic_excluded": n_synthetic_excluded,
        "n_total_parsed": n_total + n_synthetic_excluded,
    }
    return out, coverage


def coverage_line(coverage: dict) -> str:
    """The mandatory SPEC section 4 coverage string.

    Exact shape: "graded 1560 of 1716 runs (90.9% structural); tiers:
    verified 1102 / reported 412 / heuristic 46 / asserted 121 / censored 35"

    This line's shape is quoted literally in SPEC section 4 with exactly
    five tiers and must not change. Everything the line does not show
    (ungraded records, how many had a clean exit whose survival could not be
    checked, and how many zero-token synthetic sessions were removed before
    either number here was computed -- Flag 6) is printed separately by
    `uncounted_line`. `n_total` here (the "of M runs" denominator) is already
    the CORRECTED population `grade_all` produces: zero-token synthetic
    sessions are excluded before this dict is built, not subtracted from a
    raw parsed count after the fact, so this line never needs to say "minus
    N" to be accurate -- it already describes the smaller, honest population.
    """
    t = coverage["tiers"]
    return (
        f"graded {coverage['n_graded']} of {coverage['n_total']} runs "
        f"({coverage['pct_structural']:.1f}% structural); tiers: "
        f"verified {t['verified']} / reported {t['reported']} / "
        f"heuristic {t['heuristic']} / asserted {t['asserted']} / "
        f"censored {t['censored']}"
    )


def uncounted_line(coverage: dict) -> str | None:
    """One line naming everything the coverage line does not show, or None when there is nothing."""
    n_ungraded = coverage["tiers"]["ungraded"]
    n_unevaluable = coverage.get("unevaluable_heuristic", 0)
    n_synthetic = coverage.get("n_synthetic_excluded", 0)
    if n_ungraded == 0 and n_unevaluable == 0 and n_synthetic == 0:
        return None
    parts = []
    if n_ungraded:
        parts.append(f"{n_ungraded} runs carried no acceptance signal at all")
    if n_unevaluable:
        parts.append(f"{n_unevaluable} more had a clean exit whose survival could not be checked")
    line = "ungraded: " + "; ".join(parts) if parts else ""
    if n_synthetic:
        # Flag 6: named separately from "ungraded" because these were removed
        # from the population entirely (see `grade_all`), not merely graded
        # with no signal; `n_total_parsed` (present whenever this fires) is
        # the raw parsed count before that removal, so a reader can see both
        # numbers and reconstruct the correction.
        n_total_parsed = coverage.get("n_total_parsed")
        before = f" (of {n_total_parsed} runs parsed)" if n_total_parsed is not None else ""
        synth_line = (
            f"{n_synthetic} zero-token synthetic sessions excluded from the graded "
            f"population{before}: harness bookkeeping turns (model \"<synthetic>\", "
            "all-zero usage), not priced work"
        )
        line = f"{line}; {synth_line}" if line else synth_line
    return line


def grading_report(records: list[dict], coverage: dict) -> list[str]:
    """Lines for `loopmath analyze --grading`: which rule graded what (SPEC section 3).

    First line is the mandatory coverage line. Next, when there is anything
    it does not show, the uncounted line; then a line naming the follow-up
    lookback window and how many heuristic runs it demoted (SPEC section 0's
    full honesty rule: a chosen parameter and its effect stay visible in the
    output, not buried in the code). The rest name which attempt_clause and
    grade tier each record landed on, so a reader can see the rule trail
    without re-reading logs.
    """
    lines = [coverage_line(coverage)]

    unc = uncounted_line(coverage)
    if unc is not None:
        lines.append(unc)

    demoted = sum(
        1 for rec in records if rec.get("grade", {}).get("signal") == _FOLLOWUP_REWORKED_SIGNAL
    )
    lines.append(
        f"follow-up window: {LOOKBACK_S:.0f}s, directory-level; "
        f"demoted {demoted} heuristic run(s) whose directories a later session re-touched"
    )

    # Flag 3: `reported` reads as 0, and the `gate_record` path inside
    # `verified` never fires, because neither live parser can set
    # `gate_record`, `reviewer_verdict`, or `contract_run_json` from a session
    # log alone (see the module docstring above for what was checked in the
    # real logs). Printed unconditionally, not gated on whether it happens to
    # be zero this run, so a reader never mistakes the absence of these
    # signals for an absence of gating or review work.
    lines.append(
        "note: gate_record, reviewer_verdict, and contract_run_json are structurally "
        "unreachable from session logs alone (they live inside a separate contract-v3 "
        "run.json this pass does not open); a 'reported' count of 0 reflects that gap, "
        "not the absence of review or gating work"
    )

    for rec in records:
        grade = rec.get("grade", {})
        clause = rec.get("attempt_clause", "continuation")
        rule = rec.get("attempt_rule", "v1")
        lines.append(
            f"{rec.get('run_id', '?')}: attempt {rec.get('attempt', 1)} "
            f"({rule}/{clause}) -> {grade.get('tier', 'ungraded')} "
            f"accepted={grade.get('accepted')}: {grade.get('signal', '')}"
        )
    return lines
