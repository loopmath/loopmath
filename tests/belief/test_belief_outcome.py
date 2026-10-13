"""outcome_evidence: the one place the model reads success (spec 03 section 5)."""

from __future__ import annotations

from datetime import datetime

from loopmath.belief.outcome import outcome_evidence
from loopmath.types import AcceptanceRule, ScoreTarget

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
TESTS = AcceptanceRule("tests", "tests pass")
BOTH = AcceptanceRule("tests+referee", "tests and referee", requires=("tests", "referee"))
PERF = AcceptanceRule("perf>=2400", "perf reaches 2400", requires=(), score=ScoreTarget("perf", 2400.0, "higher"))
LATENCY = AcceptanceRule("p99<=120", "p99 at most 120", requires=(), score=ScoreTarget("p99", 120.0, "lower"))


def _doc(*signals, ended="2026-09-20T11:00:00-07:00", ext=None):
    run = {"id": "run_x", "ended_at": ended, "signals": list(signals)}
    if ext:
        run["ext"] = ext
    return {"run": run}


def _v(name, value, at="2026-09-20T11:00:00-07:00", tier="verified"):
    return {"kind": "verdict", "name": name, "value": value, "observed_at": at, "tier": tier}


def _s(name, value, at="2026-09-20T11:00:00-07:00", tier="verified"):
    return {"kind": "score", "name": name, "value": value, "observed_at": at, "tier": tier}


def _e(name, at, tier="reported"):
    return {"kind": "event", "name": name, "value": None, "observed_at": at, "tier": tier}


def test_verdicts_pass_fail_and_missing():
    ev = outcome_evidence(_doc(_v("tests", "pass")), TESTS, now=NOW)
    assert (ev.z, ev.q, ev.tier) == (1.0, 0.98, "verified")
    ev = outcome_evidence(_doc(_v("tests", "fail")), TESTS, now=NOW)
    assert ev.z == 0.0 and "failed:tests" in ev.reasons
    ev = outcome_evidence(_doc(), TESTS, now=NOW)
    assert ev.z is None and "missing:tests" in ev.reasons
    ev = outcome_evidence(_doc(_v("tests", "error")), TESTS, now=NOW)
    assert ev.z is None


def test_the_last_verdict_by_time_counts():
    doc = _doc(_v("tests", "pass", "2026-09-20T12:00:00-07:00"), _v("tests", "fail", "2026-09-20T10:00:00-07:00"))
    assert outcome_evidence(doc, TESTS, now=NOW).z == 1.0


def test_several_verdicts_are_a_conjunction_with_the_weakest_tier():
    doc = _doc(_v("tests", "pass"), _v("referee", "accept", tier="heuristic"))
    ev = outcome_evidence(doc, BOTH, now=NOW)
    assert (ev.z, ev.tier, ev.q) == (1.0, "heuristic", 0.8)
    ev = outcome_evidence(_doc(_v("tests", "pass"), _v("referee", "reject")), BOTH, now=NOW)
    assert ev.z == 0.0
    ev = outcome_evidence(_doc(_v("tests", "pass")), BOTH, now=NOW)
    assert ev.z is None
    ev = outcome_evidence(_doc(_v("tests", "fail")), BOTH, now=NOW)
    assert ev.z == 0.0  # one failing part decides even with another missing


def test_score_targets_in_both_directions_and_null_scores():
    assert outcome_evidence(_doc(_s("perf", 2500.0)), PERF, now=NOW).z == 1.0
    ev = outcome_evidence(_doc(_s("perf", 2300.0)), PERF, now=NOW)
    assert ev.z == 0.0 and "below_target:perf" in ev.reasons
    assert outcome_evidence(_doc(_s("p99", 100.0)), LATENCY, now=NOW).z == 1.0
    assert outcome_evidence(_doc(_s("p99", 130.0)), LATENCY, now=NOW).z == 0.0
    assert outcome_evidence(_doc(_s("perf", None)), PERF, now=NOW).z is None


def test_late_events_inside_the_window_flip_success():
    doc = _doc(_v("tests", "pass"), _e("revert", "2026-09-22T09:00:00-07:00"))
    ev = outcome_evidence(doc, TESTS, now=NOW)
    assert ev.z == 0.0 and "late:revert" in ev.reasons and ev.tier == "reported"
    outside = _doc(_v("tests", "pass"), _e("revert", "2026-10-20T09:00:00-07:00"))
    assert outcome_evidence(outside, TESTS, now=datetime.fromisoformat("2026-10-21T00:00:00-07:00")).z == 1.0
    future = _doc(_v("tests", "pass"), _e("incident", "2026-09-24T09:00:00-07:00"))
    assert outcome_evidence(future, TESTS, now=NOW).z == 1.0  # not yet observed at `now`


def test_scores_are_reported_whatever_the_rule():
    ev = outcome_evidence(_doc(_v("tests", "pass"), _s("perf", 2100.0), _s("perf", 2200.0, "2026-09-20T12:00:00-07:00"),
                               _s("runtime_s", 40)), TESTS, now=NOW)
    assert ev.scores == {"perf": 2200.0, "runtime_s": 40.0}


def test_shared_runs_carry_the_senders_evidence():
    share = {"org": "acme", "evidence": {"z": 1.0, "q": 0.95, "tier": "reported"}, "scores": {"perf": 2500}}
    ev = outcome_evidence(_doc(_v("tests", "fail"), ext={"dev.loopmath.share": share}), TESTS, now=NOW)
    assert (ev.z, ev.q, ev.tier, ev.scores, ev.reasons) == (1.0, 0.95, "reported", {"perf": 2500.0}, ("shared",))
    # without carried evidence the run's own signals decide
    ev = outcome_evidence(_doc(_v("tests", "fail"), ext={"dev.loopmath.share": {"org": "acme"}}), TESTS, now=NOW)
    assert ev.z == 0.0
