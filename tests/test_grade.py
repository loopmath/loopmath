"""Tests for loopmath.grade: evidence-tier acceptance labeling (SPEC section 4)."""

from __future__ import annotations

import subprocess

import pytest

from loopmath.grade import (
    CENSOR_MAX_WALL_S,
    LOOKBACK_S,
    assign_attempts,
    coverage_line,
    diff_survives,
    grade_all,
    grade_run,
    grading_report,
    uncounted_line,
)


def _record(
    run_id="r1",
    workspace="w",
    ts="2026-08-30T10:00:00Z",
    wall_s=400.0,
    touched_dirs=None,
    signals=None,
    attempt=1,
):
    """Small helper factory building a RunRecord.to_dict()-shaped dict."""
    return {
        "run_id": run_id,
        "harness": "codex",
        "model": "gpt-5.6-sol",
        "effort": "medium",
        "tokens": {"in": 1, "cache_read": 1, "cache_write": 1, "out": 1},
        "wall_s": wall_s,
        "ts": ts,
        "workspace": workspace,
        "attempt": attempt,
        "attempt_rule": "v1",
        "touched_dirs": touched_dirs if touched_dirs is not None else ["src"],
        "written_file_kinds": {"py": 1},
        "session_path": "x",
        "_signals": signals or {},
    }


# --------------------------------------------------------------------------
# Tier ladder
# --------------------------------------------------------------------------


def test_verified_tier_tests_red_to_green():
    rec = _record(signals={"tests_red_to_green": True})
    g = grade_run(rec)
    assert g == {
        "accepted": True,
        "tier": "verified",
        "signal": "tests went red to green in session",
    }


def test_verified_tier_gate_record():
    rec = _record(signals={"gate_record": True})
    g = grade_run(rec)
    assert g["tier"] == "verified"
    assert g["accepted"] is True
    assert g["signal"] == "gate record"


def test_verified_tier_check_exit_zero():
    rec = _record(signals={"check_exit_zero": True})
    g = grade_run(rec)
    assert g["tier"] == "verified"
    assert g["accepted"] is True
    assert g["signal"] == "check exited 0"


def test_reported_tier_approved():
    rec = _record(signals={"reviewer_verdict": "approved"})
    g = grade_run(rec)
    assert g["tier"] == "reported"
    assert g["accepted"] is True
    assert g["signal"] == "reviewer verdict recorded: approved"


def test_reported_tier_rejected():
    rec = _record(signals={"reviewer_verdict": "rejected"})
    g = grade_run(rec)
    assert g["tier"] == "reported"
    assert g["accepted"] is False
    assert g["signal"] == "reviewer verdict recorded: rejected"


def test_heuristic_tier_clean_exit_survives():
    # Heuristic requires a grading context (Flag 2): grade via grade_all so
    # the record is graded with its (empty) list of later sessions visible.
    rec = _record(signals={"exit_ok": True})
    out, _coverage = grade_all([rec])
    g = out[0]["grade"]
    assert g["tier"] == "heuristic"
    assert g["accepted"] is True
    assert g["signal"] == "clean exit, no follow-up fix session, work survived"


def test_asserted_tier():
    rec = _record(signals={"agent_claims_success": True})
    g = grade_run(rec)
    assert g["tier"] == "asserted"
    assert g["accepted"] is None
    assert g["signal"] == "agent asserted success, nothing corroborates"


def test_ungraded_tier():
    rec = _record(wall_s=400.0, signals={})
    g = grade_run(rec)
    assert g["tier"] == "ungraded"
    assert g["accepted"] is None
    assert g["signal"] == "no acceptance signal in log"


def test_ladder_order_verified_beats_everything():
    # tests_red_to_green True, but also a rejected reviewer verdict and an
    # agent success claim present: verified must win.
    rec = _record(
        signals={
            "tests_red_to_green": True,
            "reviewer_verdict": "rejected",
            "agent_claims_success": True,
        }
    )
    g = grade_run(rec)
    assert g["tier"] == "verified"
    assert g["accepted"] is True


def test_ladder_order_reported_beats_heuristic_and_asserted():
    rec = _record(
        signals={
            "reviewer_verdict": "approved",
            "exit_ok": True,
            "agent_claims_success": True,
        }
    )
    g = grade_run(rec)
    assert g["tier"] == "reported"
    assert g["accepted"] is True


def test_ladder_order_heuristic_beats_asserted():
    # Needs a grading context too (Flag 2), same as test_heuristic_tier_clean_exit_survives.
    rec = _record(signals={"exit_ok": True, "agent_claims_success": True})
    out, _coverage = grade_all([rec])
    g = out[0]["grade"]
    assert g["tier"] == "heuristic"
    assert g["accepted"] is True


# --------------------------------------------------------------------------
# Flag 2: heuristic tier preconditions (context / ts / touched_dirs)
# --------------------------------------------------------------------------


def test_heuristic_does_not_fire_without_context():
    rec = _record(signals={"exit_ok": True})
    g = grade_run(rec)  # no context supplied at all
    assert g["tier"] != "heuristic"
    assert g["signal"] == "clean exit, but survival could not be checked: no later sessions visible"


def test_heuristic_does_not_fire_without_ts():
    rec = _record(ts=None, signals={"exit_ok": True})
    out, coverage = grade_all([rec])
    g = out[0]["grade"]
    assert g["tier"] != "heuristic"
    assert g["signal"] == "clean exit, but survival could not be checked: no timestamp recorded"
    assert coverage["unevaluable_heuristic"] == 1


def test_heuristic_does_not_fire_with_empty_touched_dirs():
    rec = _record(touched_dirs=[], signals={"exit_ok": True})
    out, coverage = grade_all([rec])
    g = out[0]["grade"]
    assert g["tier"] != "heuristic"
    assert g["signal"] == "clean exit, but survival could not be checked: no directories recorded"
    assert coverage["unevaluable_heuristic"] == 1


def test_heuristic_unevaluable_records_are_not_double_counted_when_evaluable():
    # A normal, fully-evidenced heuristic record must not be counted as
    # unevaluable.
    rec = _record(signals={"exit_ok": True})
    _out, coverage = grade_all([rec])
    assert coverage["unevaluable_heuristic"] == 0


# --------------------------------------------------------------------------
# R5 censoring
# --------------------------------------------------------------------------


def test_censors_short_no_signal_run():
    rec = _record(wall_s=12.0, signals={})
    g = grade_run(rec)
    assert g["tier"] == "censored"
    assert g["accepted"] is None
    assert g["signal"] == "instant retry under 60 s with no approval signal"


def test_does_not_censor_short_run_with_check_exit_zero():
    rec = _record(wall_s=12.0, signals={"check_exit_zero": True})
    g = grade_run(rec)
    assert g["tier"] == "verified"
    assert g["accepted"] is True


def test_does_not_censor_short_run_with_rejected_verdict():
    # accepted is False (real evidence), not None, so it must not be censored
    # even though wall_s is under the threshold.
    rec = _record(wall_s=12.0, signals={"reviewer_verdict": "rejected"})
    g = grade_run(rec)
    assert g["tier"] == "reported"
    assert g["accepted"] is False


def test_censor_threshold_is_exclusive_boundary():
    rec_at = _record(wall_s=CENSOR_MAX_WALL_S, signals={})
    rec_under = _record(wall_s=CENSOR_MAX_WALL_S - 0.001, signals={})
    assert grade_run(rec_at)["tier"] == "ungraded"  # not < threshold
    assert grade_run(rec_under)["tier"] == "censored"


def test_censored_records_excluded_from_denominator():
    records = [
        _record(run_id="a", wall_s=12.0, signals={}),  # censored
        _record(run_id="b", wall_s=400.0, signals={"check_exit_zero": True}),  # verified
        _record(run_id="c", wall_s=400.0, signals={"agent_claims_success": True}),  # asserted
    ]
    _, coverage = grade_all(records)
    assert coverage["tiers"]["censored"] == 1
    assert coverage["tiers"]["verified"] == 1
    assert coverage["tiers"]["asserted"] == 1
    # only the verified record is graded/accepted; censored and asserted are excluded
    assert coverage["n_graded"] == 1
    assert coverage["denominator"] == 1
    assert coverage["n_accepted"] == 1
    assert coverage["n_rejected"] == 0
    assert coverage["n_unknown"] == 2


# --------------------------------------------------------------------------
# assign_attempts / attempt_rule v1
# --------------------------------------------------------------------------


def test_assign_attempts_three_clauses_and_continuation():
    records = [
        # 1: short, no signal -> first record, always attempt 1
        _record(run_id="s1", ts="2026-08-30T10:00:00Z", wall_s=30.0, touched_dirs=["a"]),
        # 2: wall_s >= 60 -> new attempt via "wall"
        _record(run_id="s2", ts="2026-08-30T10:01:00Z", wall_s=90.0, touched_dirs=["b"]),
        # 3: timed out -> new attempt via "timeout"
        _record(
            run_id="s3",
            ts="2026-08-30T10:02:00Z",
            wall_s=30.0,
            touched_dirs=["c"],
            signals={"timed_out": True},
        ),
        # 4: THIS record's own exit_ok is True and its own check_exit_zero is
        # False (an OBSERVED failure, not merely absent), and a LATER record
        # (s5) re-touches its dir ("c") within the lookback window ->
        # new attempt via "reedit".
        _record(
            run_id="s4",
            ts="2026-08-30T10:03:00Z",
            wall_s=30.0,
            touched_dirs=["c"],
            signals={"exit_ok": True, "check_exit_zero": False},
        ),
        # 5: short, re-touches s4's dir ("c") but has no exit_ok signal of its
        # own -> continuation of attempt 4, not a new "reedit" attempt itself.
        _record(run_id="s5", ts="2026-08-30T10:04:00Z", wall_s=10.0, touched_dirs=["c"]),
    ]
    out = assign_attempts(records)
    by_id = {r["run_id"]: r for r in out}

    assert by_id["s1"]["attempt"] == 1
    assert by_id["s2"]["attempt"] == 2
    assert by_id["s2"]["attempt_clause"] == "wall"
    assert by_id["s3"]["attempt"] == 3
    assert by_id["s3"]["attempt_clause"] == "timeout"
    assert by_id["s4"]["attempt"] == 4
    assert by_id["s4"]["attempt_clause"] == "reedit"
    assert by_id["s5"]["attempt"] == 4
    assert by_id["s5"]["attempt_clause"] == "continuation"
    for r in out:
        assert r["attempt_rule"] == "v1"


def test_assign_attempts_reedit_fires_on_exact_three_conditions():
    # (a) this record's exit_ok is True, (b) its check_exit_zero is an
    # OBSERVED failure (explicit False, not merely absent), (c) a later
    # record in the same workspace, within LOOKBACK_S, touches an
    # overlapping directory.
    records = [
        _record(run_id="s1", ts="2026-08-30T10:00:00Z", wall_s=30.0, touched_dirs=["c"]),
        _record(
            run_id="s2",
            ts="2026-08-30T10:01:00Z",
            wall_s=30.0,
            touched_dirs=["c"],
            signals={"exit_ok": True, "check_exit_zero": False},
        ),
        _record(run_id="s3", ts="2026-08-30T10:05:00Z", wall_s=30.0, touched_dirs=["c"]),
    ]
    out = assign_attempts(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["s2"]["attempt_clause"] == "reedit"
    assert by_id["s2"]["attempt"] == 2


def test_assign_attempts_reedit_not_triggered_when_exit_ok_is_none():
    # Condition (a) fails: this record's exit_ok is None (missing), which is
    # never treated as positive evidence for clause 3.
    records = [
        _record(run_id="s1", ts="2026-08-30T10:00:00Z", wall_s=30.0, touched_dirs=["c"]),
        _record(run_id="s2", ts="2026-08-30T10:01:00Z", wall_s=30.0, touched_dirs=["c"], signals={}),
        _record(run_id="s3", ts="2026-08-30T10:02:00Z", wall_s=30.0, touched_dirs=["c"]),
    ]
    out = assign_attempts(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["s2"]["attempt_clause"] == "continuation"


def test_assign_attempts_reedit_not_triggered_when_check_exit_zero_absent():
    # Condition (b) fails a different way than the True case below: this
    # record's check_exit_zero is absent entirely (None, "never observed"),
    # not an observed failure (False). A missing signal must never be treated
    # as a negative one -- clause 3 requires re-editing "after a FAILED
    # check", and no check was ever seen here at all, so it must not fire
    # even though (a) and (c) both hold. This is the regression this fix
    # exists to pin: the old rule used `not check_exit_zero`, which is truthy
    # for both `False` and `None` and so fired here too.
    records = [
        _record(run_id="s1", ts="2026-08-30T10:00:00Z", wall_s=30.0, touched_dirs=["c"]),
        _record(
            run_id="s2",
            ts="2026-08-30T10:01:00Z",
            wall_s=30.0,
            touched_dirs=["c"],
            signals={"exit_ok": True},  # check_exit_zero absent -> None
        ),
        _record(run_id="s3", ts="2026-08-30T10:05:00Z", wall_s=30.0, touched_dirs=["c"]),
    ]
    out = assign_attempts(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["s2"]["attempt_clause"] == "continuation"
    assert by_id["s2"]["attempt"] == 1


def test_assign_attempts_reedit_not_triggered_when_check_exit_zero_true():
    # Condition (b) fails: this record's own check_exit_zero is True, so its
    # check demonstrably passed, contradicting "after a failed check" even
    # though a later record does re-touch the same directory.
    records = [
        _record(run_id="s1", ts="2026-08-30T10:00:00Z", wall_s=30.0, touched_dirs=["c"]),
        _record(
            run_id="s2",
            ts="2026-08-30T10:01:00Z",
            wall_s=30.0,
            touched_dirs=["c"],
            signals={"exit_ok": True, "check_exit_zero": True},
        ),
        _record(run_id="s3", ts="2026-08-30T10:02:00Z", wall_s=30.0, touched_dirs=["c"]),
    ]
    out = assign_attempts(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["s2"]["attempt"] == 1
    assert by_id["s2"]["attempt_clause"] == "continuation"


def test_assign_attempts_reedit_not_triggered_outside_lookback():
    # Condition (c) fails: the later overlapping record sits outside
    # LOOKBACK_S.
    assert LOOKBACK_S < 2 * 3600
    records = [
        _record(run_id="s1", ts="2026-08-30T10:00:00Z", wall_s=30.0, touched_dirs=["c"]),
        _record(
            run_id="s2",
            ts="2026-08-30T10:01:00Z",
            wall_s=30.0,
            touched_dirs=["c"],
            signals={"exit_ok": True},
        ),
        _record(run_id="s3", ts="2026-08-30T12:01:00Z", wall_s=30.0, touched_dirs=["c"]),
    ]
    out = assign_attempts(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["s2"]["attempt"] == 1
    assert by_id["s2"]["attempt_clause"] == "continuation"


def test_assign_attempts_separate_workspaces_independent():
    records = [
        _record(run_id="s1", workspace="w1", ts="2026-08-30T10:00:00Z", wall_s=90.0),
        _record(run_id="s2", workspace="w2", ts="2026-08-30T10:00:00Z", wall_s=90.0),
    ]
    out = assign_attempts(records)
    # each is the first (and only) record in its own workspace -> attempt 1
    assert all(r["attempt"] == 1 for r in out)


def test_contract_run_json_sets_explicit_rule():
    rec = _record(
        run_id="s1",
        attempt=5,
        signals={"contract_run_json": "/some/run.json"},
    )
    out = assign_attempts([rec])
    assert out[0]["attempt_rule"] == "explicit"
    assert out[0]["attempt"] == 5
    assert out[0]["attempt_clause"] == "explicit"


# --------------------------------------------------------------------------
# coverage_line
# --------------------------------------------------------------------------


def test_coverage_line_exact_shape():
    coverage = {
        "n_total": 1716,
        "n_graded": 1560,
        "pct_structural": 100.0 * 1560 / 1716,
        "tiers": {
            "verified": 1102,
            "reported": 412,
            "heuristic": 46,
            "asserted": 121,
            "censored": 35,
            "ungraded": 0,
        },
        "n_accepted": 1400,
        "n_rejected": 160,
        "n_unknown": 156,
        "denominator": 1560,
    }
    line = coverage_line(coverage)
    assert line == (
        "graded 1560 of 1716 runs (90.9% structural); tiers: "
        "verified 1102 / reported 412 / heuristic 46 / asserted 121 / censored 35"
    )
    # sanity: the tiers that make up n_graded actually sum to it
    assert coverage["tiers"]["verified"] + coverage["tiers"]["reported"] + coverage[
        "tiers"
    ]["heuristic"] == coverage["n_graded"]


def test_grade_all_coverage_percentages_consistent():
    records = [
        _record(run_id="v1", wall_s=400.0, signals={"check_exit_zero": True}),
        _record(run_id="v2", wall_s=400.0, signals={"tests_red_to_green": True}),
        _record(run_id="r1", wall_s=400.0, signals={"reviewer_verdict": "approved"}),
        _record(run_id="h1", workspace="w2", wall_s=400.0, signals={"exit_ok": True}),
        _record(run_id="a1", wall_s=400.0, signals={"agent_claims_success": True}),
        _record(run_id="u1", wall_s=400.0, signals={}),
        _record(run_id="c1", wall_s=12.0, signals={}),
    ]
    out, coverage = grade_all(records)
    assert coverage["n_total"] == 7
    assert coverage["tiers"]["verified"] == 2
    assert coverage["tiers"]["reported"] == 1
    assert coverage["tiers"]["heuristic"] == 1
    assert coverage["tiers"]["asserted"] == 1
    assert coverage["tiers"]["ungraded"] == 1
    assert coverage["tiers"]["censored"] == 1
    assert coverage["n_graded"] == 4  # verified(2) + reported(1) + heuristic(1)
    assert coverage["denominator"] == coverage["n_graded"]
    expected_pct = 100.0 * 4 / 7
    assert abs(coverage["pct_structural"] - expected_pct) < 1e-9
    # h1 has ts + touched_dirs, so its heuristic grading was fully evidenced.
    assert coverage["unevaluable_heuristic"] == 0
    assert sum(coverage["tiers"].values()) == coverage["n_total"]
    # every record got a "grade" attached
    assert all("grade" in r for r in out)
    line = coverage_line(coverage)
    assert line.startswith(f"graded {coverage['n_graded']} of {coverage['n_total']} runs")


# --------------------------------------------------------------------------
# follow-up fix session flips heuristic acceptance
# --------------------------------------------------------------------------


def test_follow_up_fix_session_flips_heuristic_to_rejected():
    records = [
        _record(
            run_id="h1",
            ts="2026-08-30T10:00:00Z",
            wall_s=400.0,
            touched_dirs=["src/loopmath"],
            signals={"exit_ok": True},
        ),
        # a later session in the same workspace reworks the same directory
        # within the lookback window
        _record(
            run_id="h2",
            ts="2026-08-30T10:30:00Z",
            wall_s=400.0,
            touched_dirs=["src/loopmath"],
            signals={"exit_ok": True},
        ),
    ]
    out, _coverage = grade_all(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["h1"]["grade"]["tier"] == "heuristic"
    assert by_id["h1"]["grade"]["accepted"] is False
    assert (
        by_id["h1"]["grade"]["signal"]
        == "clean exit, but a later session reworked the same directories"
    )
    # h2 has no later session touching its dirs, so it survives
    assert by_id["h2"]["grade"]["accepted"] is True


def test_no_follow_up_when_outside_lookback_window():
    records = [
        _record(
            run_id="h1",
            ts="2026-08-30T10:00:00Z",
            wall_s=400.0,
            touched_dirs=["src/loopmath"],
            signals={"exit_ok": True},
        ),
        _record(
            run_id="h2",
            ts="2026-08-30T12:00:00Z",  # 2 hours later, beyond LOOKBACK_S
            wall_s=400.0,
            touched_dirs=["src/loopmath"],
            signals={"exit_ok": True},
        ),
    ]
    out, _coverage = grade_all(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["h1"]["grade"]["accepted"] is True


def test_no_follow_up_when_different_directories():
    records = [
        _record(
            run_id="h1",
            ts="2026-08-30T10:00:00Z",
            wall_s=400.0,
            touched_dirs=["src/loopmath"],
            signals={"exit_ok": True},
        ),
        _record(
            run_id="h2",
            ts="2026-08-30T10:05:00Z",
            wall_s=400.0,
            touched_dirs=["tests"],
            signals={"exit_ok": True},
        ),
    ]
    out, _coverage = grade_all(records)
    by_id = {r["run_id"]: r for r in out}
    assert by_id["h1"]["grade"]["accepted"] is True


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    ("case", "accepted"),
    [
        ("kept", True),
        ("reverted", False),
        ("renamed_kept", True),
        ("renamed_reverted", False),
    ],
)
def test_diff_survival_controls_heuristic_tier_in_git_repo(tmp_path, case, accepted):
    repo = tmp_path / case
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    original = "alpha\nbeta\ngamma\ndelta\n"
    edited = "alpha\nbeta edited\ngamma\ndelta\n"
    source = repo / "notes.txt"
    source.write_text(original)
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-qm", "baseline")

    source.write_text(edited)
    if case.startswith("renamed"):
        _git(repo, "mv", "notes.txt", "moved.txt")
        source = repo / "moved.txt"
    if case.endswith("reverted") or case == "reverted":
        source.write_text(original)

    rec = _record(
        workspace=str(repo),
        touched_dirs=["."],
        signals={"exit_ok": True},
    )
    rec["diff_snapshots"] = {"notes.txt": edited}
    out, _coverage = grade_all([rec])
    grade = out[0]["grade"]
    assert grade["tier"] == "heuristic"
    assert grade["accepted"] is accepted
    assert grade["signal"] == (
        "clean exit, no follow-up fix session, work survived"
        if accepted
        else "clean exit, but diff was reverted"
    )


def test_diff_survival_falls_back_to_path_and_content_outside_git(tmp_path):
    original = tmp_path / "before.txt"
    moved = tmp_path / "nested" / "after.txt"
    moved.parent.mkdir()
    expected = "the run's edited content\n"
    moved.write_text(expected)

    assert diff_survives(original, expected, workspace=tmp_path) is True
    moved.write_text("the pre-run content\n")
    assert diff_survives(original, expected, workspace=tmp_path) is False


# --------------------------------------------------------------------------
# grading_report
# --------------------------------------------------------------------------


def test_grading_report_starts_with_coverage_line_and_lists_records():
    records = [
        _record(run_id="v1", wall_s=400.0, signals={"check_exit_zero": True}),
        _record(run_id="a1", workspace="w2", wall_s=400.0, signals={"agent_claims_success": True}),
    ]
    out, coverage = grade_all(records)
    lines = grading_report(out, coverage)
    assert lines[0] == coverage_line(coverage)
    # No ungraded / unevaluable-heuristic records here, so no uncounted line,
    # but the follow-up window line and the gate/reviewer/contract-run_json
    # unreachability note (Flag 3) are always printed.
    assert lines[1].startswith("follow-up window:")
    assert f"{LOOKBACK_S:.0f}s" in lines[1]
    assert lines[2].startswith("note:")
    assert "gate_record" in lines[2]
    assert len(lines) == 5
    assert "v1" in lines[3] or "v1" in lines[4]
    assert any("verified" in line for line in lines[3:])
    assert any("asserted" in line for line in lines[3:])


def test_grading_report_includes_uncounted_line_and_demoted_count():
    records = [
        # h1's dirs get re-touched by h2 within the lookback window -> demoted.
        _record(
            run_id="h1",
            ts="2026-08-30T10:00:00Z",
            wall_s=400.0,
            touched_dirs=["src/loopmath"],
            signals={"exit_ok": True},
        ),
        _record(
            run_id="h2",
            ts="2026-08-30T10:30:00Z",
            wall_s=400.0,
            touched_dirs=["src/loopmath"],
            signals={"exit_ok": True},
        ),
        _record(run_id="u1", wall_s=400.0, signals={}),
    ]
    out, coverage = grade_all(records)
    lines = grading_report(out, coverage)
    assert lines[0] == coverage_line(coverage)
    unc = uncounted_line(coverage)
    assert unc is not None
    assert unc in lines
    followup_line = next(line for line in lines if line.startswith("follow-up window:"))
    assert "demoted 1" in followup_line


# --------------------------------------------------------------------------
# uncounted_line
# --------------------------------------------------------------------------


def test_uncounted_line_none_when_nothing_uncounted():
    coverage = {
        "tiers": {
            "verified": 5,
            "reported": 2,
            "heuristic": 1,
            "asserted": 0,
            "censored": 0,
            "ungraded": 0,
        },
        "unevaluable_heuristic": 0,
    }
    assert uncounted_line(coverage) is None


def test_uncounted_line_names_both_counts_when_nonzero():
    coverage = {
        "tiers": {
            "verified": 0,
            "reported": 0,
            "heuristic": 0,
            "asserted": 0,
            "censored": 0,
            "ungraded": 214,
        },
        "unevaluable_heuristic": 37,
    }
    line = uncounted_line(coverage)
    assert line is not None
    assert "214" in line
    assert "37" in line


# --------------------------------------------------------------------------
# tiers + ungraded must sum to n_total (Flag 4)
# --------------------------------------------------------------------------


def test_tiers_plus_ungraded_sum_to_n_total():
    records = [
        _record(run_id="v1", wall_s=400.0, signals={"check_exit_zero": True}),
        _record(run_id="r1", wall_s=400.0, signals={"reviewer_verdict": "approved"}),
        _record(run_id="h1", workspace="w2", wall_s=400.0, signals={"exit_ok": True}),
        _record(run_id="a1", wall_s=400.0, signals={"agent_claims_success": True}),
        _record(run_id="u1", wall_s=400.0, signals={}),
        _record(run_id="c1", wall_s=12.0, signals={}),
    ]
    _, coverage = grade_all(records)
    assert sum(coverage["tiers"].values()) == coverage["n_total"]


# --------------------------------------------------------------------------
# Flag 6: zero-token synthetic sessions excluded from the graded population
# --------------------------------------------------------------------------


def test_zero_token_synthetic_excluded_from_population_and_counted():
    records = [
        _record(run_id="real1", wall_s=400.0, signals={"check_exit_zero": True}),
        _record(
            run_id="synth1",
            wall_s=0.0,
            signals={"zero_token_synthetic": True},
        ),
        _record(
            run_id="synth2",
            wall_s=0.0,
            signals={"zero_token_synthetic": True},
        ),
    ]
    out, coverage = grade_all(records)
    # The synthetic records are removed entirely, not merely tagged: they do
    # not appear in the returned records and do not inflate n_total.
    assert {r["run_id"] for r in out} == {"real1"}
    assert coverage["n_total"] == 1
    assert coverage["n_synthetic_excluded"] == 2
    assert coverage["n_total_parsed"] == 3
    line = uncounted_line(coverage)
    assert line is not None
    assert "2 zero-token synthetic sessions excluded" in line
    assert "3 runs parsed" in line


def test_zero_token_synthetic_absent_key_defaults_to_not_excluded():
    rec = _record(signals={"check_exit_zero": True})
    out, coverage = grade_all([rec])
    assert coverage["n_synthetic_excluded"] == 0
    assert len(out) == 1


def test_missing_signals_key_defaults_safely():
    rec = {
        "run_id": "r1",
        "harness": "codex",
        "model": "gpt-5.6-sol",
        "effort": "medium",
        "tokens": {"in": 1, "cache_read": 1, "cache_write": 1, "out": 1},
        "wall_s": 400.0,
        "ts": "2026-08-30T10:00:00Z",
        "workspace": "w",
        "attempt": 1,
        "attempt_rule": "v1",
        "touched_dirs": ["src"],
        "written_file_kinds": {"py": 1},
        "session_path": "x",
        # no "_signals" key at all
    }
    g = grade_run(rec)
    assert g["tier"] == "ungraded"
    assert g["accepted"] is None
