"""One rule resolver for the fit, `run finish`, the share export and the runs view (0.2.7).

An onboarded run (or one `run import` marked) under the default rule, with no `tests` verdict and a `review`
or `accepted` verdict, is read by the review rule: a success when its latest review or accepted verdict is
accept, approve or pass. Every other run reads as before, and the four readers agree on every run.
Synthetic documents only.
"""

from __future__ import annotations

import copy
from datetime import datetime

import pytest

from loopmath.belief.design import parse_run, rule_from_doc
from loopmath.belief.outcome import outcome_evidence
from loopmath.types import DEFAULT_RULE
from tests.belief import simdata

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")


def _verdict(name, value, at="2026-09-20T11:30:00-07:00", tier="reported", i=0):
    return {"id": f"sig_{name}_{i}", "kind": "verdict", "name": name, "value": value, "observed_at": at, "tier": tier}


_BASE = simdata.simulate(1, seed=41, source="live", prefix="run_r")[0][0]


def _doc(*signals, onboarded=True, rule=None, mark=False):
    """A finished simulated run with these signals; onboarded is the task source onboard writes."""
    doc = copy.deepcopy(_BASE)
    doc["run"]["signals"] = list(signals)
    doc["run"]["task"]["source"] = {"kind": "history" if onboarded else "live"}
    if rule is not None:
        doc["run"]["acceptance_rule"] = rule
    if mark:
        doc["run"].setdefault("ext", {})["dev.loopmath.import"] = {"stored_by": "run import"}
    return doc


def _read(doc):
    from loopmath.belief import rules

    rule = rules.resolve_rule(doc)
    return rule, rules.evidence(doc, rule, now=NOW)


def test_an_onboarded_review_only_run_is_read_by_the_review_rule():
    rule, ev = _read(_doc(_verdict("review", "accept")))
    assert rule.name == "review" and rule.requires == ("review",)
    assert ev.z == 1.0


@pytest.mark.parametrize("value,z", [("accept", 1.0), ("Approve", 1.0), ("pass", 1.0), ("reject", 0.0), ("fail", 0.0)])
def test_the_review_reading(value, z):
    assert _read(_doc(_verdict("review", value)))[1].z == z


@pytest.mark.parametrize("value", ["changes_requested", "error"])
def test_any_other_value_gives_no_row_and_names_the_value(value):
    _, ev = _read(_doc(_verdict("review", value)))
    assert ev.z is None
    assert f"unknown:review={value}" in ev.reasons


def test_review_and_accepted_are_one_reading_the_latest():
    doc = _doc(_verdict("accepted", "accept", at="2026-09-20T11:30:00-07:00"),
               _verdict("review", "reject", at="2026-09-20T12:00:00-07:00", i=1))
    assert _read(doc)[1].z == 0.0
    doc = _doc(_verdict("review", "reject", at="2026-09-20T11:30:00-07:00"),
               _verdict("accepted", "accept", at="2026-09-20T12:00:00-07:00", i=1))
    assert _read(doc)[1].z == 1.0


def test_a_tests_verdict_decides_alone():
    doc = _doc(_verdict("tests", "pass"), _verdict("review", "reject", i=1))
    rule, ev = _read(doc)
    assert rule.name == "tests" and ev.z == 1.0


def test_a_live_run_keeps_its_stored_rule():
    rule, ev = _read(_doc(_verdict("review", "accept"), onboarded=False))
    assert rule.name == "tests" and ev.z is None


def test_a_run_marked_by_run_import_is_in_the_case():
    rule, ev = _read(_doc(_verdict("review", "accept"), onboarded=False, mark=True))
    assert rule.name == "review" and ev.z == 1.0


def test_a_users_own_rule_named_review_keeps_todays_reading():
    own = {"name": "review", "definition": "my review", "requires": ["review"]}
    doc = _doc(_verdict("review", "approve"), rule=own)
    rule, ev = _read(doc)
    assert ev == outcome_evidence(doc, rule_from_doc(doc), now=NOW)


def test_runs_outside_the_case_read_exactly_as_before():
    from loopmath.belief import rules

    docs, _ = simdata.simulate(30, seed=41, source="live", prefix="run_s")
    docs += [_doc(_verdict("tests", "fail")), _doc(onboarded=False)]
    for doc in docs:
        assert rules.evidence(doc, rules.resolve_rule(doc), now=NOW) == outcome_evidence(doc, rule_from_doc(doc), now=NOW)


def _four_readers(doc):
    from loopmath.share.export import run_rule
    from loopmath.store.finish import evidence_for, rule_of
    from loopmath.views.runs import outcome

    fit = parse_run(doc, now=NOW).evidence
    finish = evidence_for(doc, rule_of(doc), now=NOW)
    share = evidence_for(doc, run_rule(doc), now=NOW)
    view = outcome(doc, doc["run"].get("signals") or [], NOW)
    return fit.z, finish.z, share.z, view["z"]


@pytest.mark.parametrize("signals,z", [
    ([_verdict("review", "accept")], 1.0),
    ([_verdict("review", "reject")], 0.0),
    ([_verdict("accepted", "approve")], 1.0),
    ([_verdict("review", "changes_requested")], None),
    ([_verdict("tests", "pass"), _verdict("review", "reject", i=1)], 1.0),
])
def test_the_four_readers_give_the_same_z(signals, z):
    assert _four_readers(_doc(*copy.deepcopy(signals))) == (z, z, z, z)


def test_window_zero_is_read_as_fourteen_days_by_every_reader():
    late = {"id": "sig_ev", "kind": "event", "name": "revert", "observed_at": "2026-09-22T10:00:00-07:00",
            "tier": "reported"}
    doc = _doc(_verdict("tests", "pass"), late, onboarded=False,
               rule={"name": "tests", "requires": ["tests"], "window_days": 0})
    assert _four_readers(doc) == (0.0, 0.0, 0.0, 0.0)


def test_empty_exclusions_are_read_as_revert_and_incident_by_every_reader():
    late = {"id": "sig_ev", "kind": "event", "name": "incident", "observed_at": "2026-09-21T10:00:00-07:00",
            "tier": "reported"}
    doc = _doc(_verdict("tests", "pass"), late, onboarded=False,
               rule={"name": "tests", "requires": ["tests"], "excludes_events": []})
    assert _four_readers(doc) == (0.0, 0.0, 0.0, 0.0)


def test_a_rule_with_no_name_is_the_default_for_every_reader():
    doc = _doc(_verdict("tests", "pass"), onboarded=False, rule={"requires": ["review"]})
    assert _four_readers(doc) == (1.0, 1.0, 1.0, 1.0)
    assert rule_from_doc(doc) == DEFAULT_RULE
