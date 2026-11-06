"""Check verdicts in rows that name a flow (spec/harness-runs.md section 10): a stage's `checks` reports the
flow's checks, each item one verdict signal at that stage's attempt, which the flow's acceptance rule reads. Rows
whose stages report no check give the same documents as before."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

from loopmath import flows
from loopmath import harness_runs as HR
from loopmath.belief.design import rule_from_doc
from loopmath.belief.outcome import outcome_evidence
from loopmath.harness_runs.ocp import NS
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id, flow_digest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hr_flow import (AT, FLOW_FILE, KEPT, SIMULATE, _findings, _ids, _kept, _validator,  # noqa: E402
                          example_rows)

NOW = datetime.fromisoformat("2026-11-04T09:00:00-08:00")
JUDGED = "2026-11-03T11:07:00-08:00"
FAIL = [{"id": "extract", "verdict": "fail"}]
PASS = [{"id": "extract", "verdict": "pass"}]
APPROVE = [{"id": "critic", "verdict": "pass"}]


def _stage(piece, role, rnd=1, checks=None, ended=None) -> dict:
    stage = {"role": role, "piece": piece, "round": rnd, "cost": {"usd": 0.1}}
    if checks is not None:
        stage["checks"] = copy.deepcopy(checks)
    if ended is not None:
        stage["ended_at"] = ended
    return stage


def _stages(first=FAIL, again=PASS, critic=APPROVE, ended=(None, None)) -> list[dict]:
    """Extract fails its claims check in round 1, passes it after the send-back, and the critic approves."""
    return [_stage("scout", "worker"), _stage("extract", "worker", 1, first, ended[0]),
            _stage("extract", "worker", 2, again, ended[1]), _stage("synthesize", "worker"),
            _stage("critic", "reviewer", 1, critic)]


def _checked(stages=None, **over) -> dict:
    """A rollout of the base arm whose judge scores quality 0.8, over the rule's 0.7."""
    row = {"schema": HR.SCHEMA, "id": "q07-r3", "task_id": "q07", "arm": "base", "status": "done",
           "flow": {"id": "lit-brief", "version": 1}, "stages": _stages() if stages is None else stages,
           "outcome": {"judges": [{"id": "j-claude", "score": 0.8, "observed_at": JUDGED}]},
           "started_at": "2026-11-03T11:00:00-08:00", "ended_at": AT}
    row.update(over)
    return row


@pytest.fixture(scope="module")
def flow():
    return flows.load(FLOW_FILE)


def _convert(row, flow) -> dict:
    return HR.to_ocp(row, flow=flow, observed_default=AT)


def _verdicts(doc) -> list[dict]:
    return [s for s in doc["run"]["signals"] if s["kind"] == "verdict"]


def _accepted(doc):
    ev = outcome_evidence(doc, rule_from_doc(doc), now=NOW)
    return ev.z, ev.reasons


def _edited(tmp_path, edit) -> flows.FlowFolder:
    """The example flow with `edit` applied to its flow object, its configuration id and digest recomputed."""
    doc = json.loads(FLOW_FILE.read_text(encoding="utf-8"))
    edit(doc["flow"])
    configuration = doc["flow"]["configuration"]
    configuration["id"] = config_id(configuration["workflow"], configuration["settings"])
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    path = Path(tmp_path) / "flow.ocp.json"
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return flows.load(path)


def _checks_form(checks):
    """An edit writing the base workflow's control as `checks` in place of its gates."""
    def edit(flow):
        control = flow["configuration"]["workflow"]["control"]
        for name in ("gates", "repair", "ext"):
            del control[name]
        control["checks"] = checks
    return edit


# No checks, no verdicts ---------------------------------------------------------------------------

def test_stages_that_report_no_check_give_the_documents_of_rows_without_the_member(flow):
    for row in example_rows():
        doc = _convert(row, flow)
        assert _verdicts(doc) == []
        for value in (None, []):
            same = copy.deepcopy(row)
            for stage in same["stages"]:
                stage["checks"] = value
            assert _convert(same, flow) == doc
            assert HR.flow_row_findings(same, flow) == []


def test_only_the_stages_that_report_checks_give_verdicts(flow):
    stages = _stages(first=[], again=None, critic=APPROVE)
    doc = _convert(_checked(stages), flow)
    rid = doc["run"]["id"]
    [verdict] = _verdicts(doc)
    assert verdict["id"] == f"sig_{rid}_check_0"  # an empty list takes no number
    assert verdict["at_attempt"] == doc["attempts"][4]["id"]
    assert (verdict["name"], verdict["value"]) == ("critic", "pass")
    assert _accepted(doc) == (None, ("missing:claims-check",))  # unreported is unknown, not passed
    assert _findings(flow, doc) == [[], []]


# Refusals -----------------------------------------------------------------------------------------

def test_a_check_one_stage_reports_twice_is_refused_by_name(flow):
    stages = _stages(first=FAIL + [{"id": "extract", "verdict": "pass"}])
    problem = "stages.1.checks.1.id: 'extract' is reported at stages.1.checks.0 too; a stage reports each check once"
    assert HR.check_row(_checked(stages)) == [problem]
    with pytest.raises(ValueError) as caught:
        _convert(_checked(stages), flow)
    assert str(caught.value) == problem
    assert HR.flow_row_findings(_checked(stages), flow) == []
    assert HR.check_row(_checked()) == []  # one id in two stages is two reports


@pytest.mark.parametrize("value", [[], FAIL, None, "critic", [{"id": 3, "verdict": "maybe"}]])
@pytest.mark.parametrize("flow_member", ["absent", None, "lit-brief"])
def test_checks_without_flow_are_unknown_members(value, flow_member):
    row = {"schema": HR.SCHEMA, "id": "q07-r1", "task_id": "q07", "outcome": {"score": 0.7},
           "stages": [{"role": "worker"}, {"role": "reviewer", "checks": copy.deepcopy(value)}]}
    if flow_member != "absent":
        row["flow"] = flow_member
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))
    doc = HR.to_ocp(row, observed_default=AT)
    assert doc["attempts"][1]["ext"][NS]["extra"] == {"checks": value}  # kept whole, null too
    assert [s["kind"] for s in doc["run"]["signals"]] == ["score"]


# An id the flow lacks -----------------------------------------------------------------------------

def test_an_id_the_flow_lacks_converts_as_its_own_name_and_is_a_finding(flow):
    row = _checked(_stages(again=[{"id": "cites-resolve", "verdict": "pass"}]))
    doc = _convert(row, flow)
    verdicts = _verdicts(doc)
    assert [(s["name"], s["source"]) for s in verdicts] == [
        ("claims-check", {"kind": "orchestrator", "ref": "extract"}),
        ("cites-resolve", {"kind": "orchestrator", "ref": "cites-resolve"}),
        ("critic", {"kind": "orchestrator", "ref": "critic"})]
    assert HR.flow_row_findings(row, flow) == [
        "stages.2.checks.0.id: 'cites-resolve' names no check or gate piece of flow 'lit-brief' arm 'base' "
        "(checks: critic, extract)"]
    assert _findings(flow, doc) == [[], []] and conformance.validate_doc(doc) == []


def test_flow_row_findings_is_empty_for_rows_it_does_not_judge(flow):
    assert HR.flow_row_findings(_checked(), flow) == []
    unknown = [{"id": "nope", "verdict": "pass"}]
    for row in (_checked(_stages(again=unknown), arm="opus-only"),            # an arm the flow lacks
                _checked(_stages(again=unknown), flow={"id": "other", "version": 1}),
                _checked(_stages(again=unknown + unknown))):                  # check_row refuses it
        assert HR.flow_row_findings(row, flow) == []
    plain = {"schema": HR.SCHEMA, "task_id": "q07", "outcome": {"score": 0.7}}
    assert HR.flow_row_findings(plain, flow) == []


# The acceptance rule reads the verdicts ------------------------------------------------------------

def test_a_check_failed_then_passed_after_the_send_back_meets_the_flows_rule(flow):
    doc = _convert(_checked(), flow)
    atts = doc["attempts"]
    assert [(atts[n]["node"], atts[n]["round"]) for n in (1, 2, 4)] == [("extract", 1), ("extract", 2), ("critic", 1)]
    assert [(s["name"], s["value"], s["at_attempt"]) for s in _verdicts(doc)] == [
        ("claims-check", "fail", atts[1]["id"]), ("claims-check", "pass", atts[2]["id"]),
        ("critic", "pass", atts[4]["id"])]
    assert {(s["tier"], s["observed_at"]) for s in _verdicts(doc)} == {("reported", AT)}
    assert [s["kind"] for s in doc["run"]["signals"]] == ["score", "verdict", "verdict", "verdict"]
    assert [(e["at"], e["detail"]) for e in doc["events"] if e["type"] == "signal_observed"] == [
        (AT, "verdict claims-check"), (AT, "verdict claims-check"), (AT, "verdict critic"), (JUDGED, "score quality")]
    assert _accepted(doc) == (1.0, ())
    assert _findings(flow, doc) == [[], []]
    assert _accepted(_convert(_checked(_stages(again=None)), flow)) == (0.0, ("failed:claims-check",))


def test_verdicts_observed_at_one_time_keep_the_stage_order(flow):
    # neither extract stage ended at a time of its own: both verdicts take the run's, and the later stage counts
    assert _accepted(_convert(_checked(_stages(first=PASS, again=FAIL)), flow)) == (0.0, ("failed:claims-check",))
    both = ("2026-11-03T11:03:00-08:00",) * 2
    doc = _convert(_checked(_stages(ended=both)), flow)
    assert [s["observed_at"] for s in _verdicts(doc)[:2]] == list(both)
    assert _accepted(doc) == (1.0, ())


def test_a_stage_end_time_orders_the_verdicts_before_the_stage_order(flow):
    in_order = ("2026-11-03T11:02:00-08:00", "2026-11-03T11:03:00-08:00")
    doc = _convert(_checked(_stages(ended=in_order)), flow)
    assert [s["observed_at"] for s in _verdicts(doc)] == [*in_order, AT]
    assert _accepted(doc) == (1.0, ())
    # the round 1 verdict observed later is the last one, wherever its stage sits
    late_first = ("2026-11-03T11:04:00-08:00", "2026-11-03T11:03:00-08:00")
    assert _accepted(_convert(_checked(_stages(ended=late_first)), flow)) == (0.0, ("failed:claims-check",))


def test_the_quality_score_still_counts_beside_the_check(flow):
    low = _checked(outcome={"judges": [{"id": "j-claude", "score": 0.6, "observed_at": JUDGED}]})
    assert _accepted(_convert(low, flow)) == (0.0, ("below_target:quality",))


# Names --------------------------------------------------------------------------------------------

def test_a_check_the_rule_requires_keeps_its_id_as_name(tmp_path):
    ff = _edited(tmp_path, lambda f: f["acceptance_rule"].update(requires=["extract"]))
    doc = _convert(_checked(), ff)
    assert [(s["name"], s["source"]["ref"]) for s in _verdicts(doc)] == [
        ("extract", "extract"), ("extract", "extract"), ("critic", "critic")]
    assert _accepted(doc) == (1.0, ())
    assert _findings(ff, doc) == [[], []]


def test_the_checks_form_names_two_checks_after_one_piece_apart(tmp_path):
    ff = _edited(tmp_path, _checks_form([
        {"id": "claims-check", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
        {"id": "cites-resolve", "after": "extract", "rule": "command:cites"},
        {"id": "critic-approves", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}]))
    both = [{"id": "claims-check", "verdict": "pass"}, {"id": "cites-resolve", "verdict": "fail"}]
    row = _checked(_stages(first=both, again=None, critic=[{"id": "critic-approves", "verdict": "pass"}]))
    doc = _convert(row, ff)
    assert [(s["name"], s["value"], s["source"]["ref"]) for s in _verdicts(doc)] == [
        ("claims-check", "pass", "claims-check"), ("cites", "fail", "cites-resolve"),
        ("critic-approves", "pass", "critic-approves")]
    assert {s["at_attempt"] for s in _verdicts(doc)[:2]} == {doc["attempts"][1]["id"]}
    assert HR.flow_row_findings(row, ff) == [] and _findings(ff, doc) == [[], []]
    # in the checks form a piece is not a check: `extract` names none here
    gate_named = _checked(_stages(again=None, critic=None))
    assert HR.flow_row_findings(gate_named, ff) == [
        "stages.1.checks.0.id: 'extract' names no check or gate piece of flow 'lit-brief' arm 'base' "
        "(checks: cites-resolve, claims-check, critic-approves)"]


def test_a_gate_moved_to_a_check_that_keeps_its_piece_id_keeps_its_name(tmp_path, flow):
    ff = _edited(tmp_path, _checks_form([
        {"id": "extract", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
        {"id": "critic", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}]))
    moved, gated = _convert(_checked(), ff), _convert(_checked(), flow)

    def named(doc):
        return [(s["name"], s["value"], s["source"]) for s in _verdicts(doc)]

    assert named(moved) == named(gated)
    assert _accepted(moved) == (1.0, ()) and HR.flow_row_findings(_checked(), ff) == []


CLAIMS = {"id": "claims-check", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"}
CRITIC = {"id": "critic-approves", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}


def _claims(verdict) -> list[dict]:
    return [{"id": "claims-check", "verdict": verdict}]


@pytest.mark.parametrize(("check", "judge_score", "name"), [
    ({"id": "quality", "after": "extract", "rule": "review_approve"}, None, "check:quality"),
    ({"id": "quality-review", "after": "extract", "rule": "command:quality"}, None, "quality-review"),
    ({"id": "faithfulness", "after": "extract", "rule": "command:faithfulness"}, "faithfulness",
     "check:faithfulness"),
])
def test_a_check_named_like_a_score_of_the_flow_takes_another_name(tmp_path, check, judge_score, name):
    def edit(flow):
        _checks_form([CLAIMS, check, CRITIC])(flow)
        if judge_score is not None:
            flow["judges"][1]["score"] = judge_score

    ff = _edited(tmp_path, edit)
    row = _checked(_stages(first=_claims("pass") + [{"id": check["id"], "verdict": "pass"}], again=None,
                           critic=[{"id": "critic-approves", "verdict": "pass"}]))
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))
    assert HR.flow_row_findings(row, ff) == []
    doc = _convert(row, ff)
    assert [(s["name"], s["source"]["ref"]) for s in _verdicts(doc)] == [
        ("claims-check", "claims-check"), (name, check["id"]), ("critic-approves", "critic-approves")]
    assert _findings(ff, doc) == [[], []]
    assert _accepted(doc) == (1.0, ())


def test_an_unknown_id_named_like_the_score_takes_another_name_too(flow):
    row = _checked(_stages(again=[{"id": "quality", "verdict": "pass"}]))
    doc = _convert(row, flow)
    assert [(s["name"], s["source"]["ref"]) for s in _verdicts(doc)][1] == ("check:quality", "quality")
    assert len(HR.flow_row_findings(row, flow)) == 1
    assert _findings(flow, doc) == [[], []]


def test_a_200_character_check_named_like_the_score_stays_within_200(tmp_path):
    long = "q" * 200

    def edit(flow):
        flow["acceptance_rule"]["score"]["name"] = flow["objective"]["score"] = long
        _checks_form([CLAIMS, dict(CRITIC, id=long)])(flow)

    ff = _edited(tmp_path, edit)
    doc = _convert(_checked(_stages(first=_claims("fail"), again=_claims("pass"),
                                    critic=[{"id": long, "verdict": "pass"}])), ff)
    want = f"check:{long}"
    tail = "-" + hashlib.sha256(want.encode()).hexdigest()[:12]
    assert _verdicts(doc)[2]["name"] == want[:200 - len(tail)] + tail
    assert {s["name"] for s in doc["run"]["signals"] if s["kind"] == "score"} == {long}
    assert _findings(ff, doc) == [[], []] and _accepted(doc) == (1.0, ())


# Ids ----------------------------------------------------------------------------------------------

def test_check_ids_from_a_run_id_at_the_cap_are_whole(flow):
    rid = "run_" + "q" * (HR.MAX_KEPT_RUN_ID - 4)
    doc = _convert(_checked(id=rid), flow)
    ids = _ids(doc)
    assert all(len(i) <= 200 for i in ids) and len(set(ids)) == len(ids)
    assert [s["id"] for s in _verdicts(doc)] == [f"sig_{rid}_check_{k}" for k in range(3)]
    assert conformance.validate_many([flow.doc, doc, _convert(_checked(id=rid[:-1] + "r"), flow)]) == [[], [], []]


@pytest.mark.parametrize("rid", [KEPT, "run_" + "q" * (HR.MAX_KEPT_RUN_ID - 4), "q07-r3"])
def test_a_score_named_like_a_check_signal_leaves_every_signal_its_own_id(rid, flow):
    row = _checked(id=rid, outcome={"name": "check_0", "score": 0.78, "scale": "fraction", "judge": "j-claude",
                                    "judges": [{"id": "j-claude", "score": 0.8, "observed_at": JUDGED}]})
    doc = _convert(row, flow)
    ids = [s["id"] for s in doc["run"]["signals"]]
    assert len(set(ids)) == len(ids) == 5 and all(len(i) <= 200 for i in ids)
    assert ids[0].endswith("_check_0")
    tail = "-" + hashlib.sha256(ids[0].encode()).hexdigest()[:12]
    assert ids[2] == ids[0][:200 - len(tail)] + tail  # the first check takes the score's id with a suffix
    assert ids[3].endswith("_check_1") and ids[4].endswith("_check_2")
    assert _findings(flow, doc) == [[], []]


# Pass rates from the signals alone ----------------------------------------------------------------

def test_pass_rates_per_check_and_round_come_from_the_signals(flow):
    again = [_stage("scout", "worker"), _stage("extract", "worker", 1, PASS), _stage("synthesize", "worker"),
             _stage("critic", "reviewer", 1, [{"id": "critic", "verdict": "fail"}]), _stage("synthesize", "worker", 2),
             _stage("critic", "reviewer", 2, APPROVE)]
    docs = HR.to_ocp_rows([_checked(), _checked(again, id="q07-r4")], flow=flow, observed_default=AT)
    rates: dict[tuple[str, int | None], list[int]] = {}
    for doc in docs:
        rounds = {a["id"]: a["round"] for a in doc["attempts"]}
        for s in doc["run"]["signals"]:
            if s["kind"] == "verdict" and s["source"]["kind"] == "orchestrator":
                for key in ((s["source"]["ref"], None), (s["source"]["ref"], rounds[s["at_attempt"]])):
                    seen = rates.setdefault(key, [0, 0])
                    seen[0], seen[1] = seen[0] + 1, seen[1] + (s["value"] == "pass")
    assert rates == {("extract", None): [3, 2], ("extract", 1): [2, 1], ("extract", 2): [1, 1],
                     ("critic", None): [3, 2], ("critic", 1): [2, 1], ("critic", 2): [1, 1]}
    assert not [s for d in docs for s in d["run"]["signals"]
                if s["kind"] != "verdict" and s["source"]["kind"] == "orchestrator"]


# Unknown members ----------------------------------------------------------------------------------

def test_unknown_check_members_are_kept_with_the_signal(flow):
    row = _checked(_stages(critic=[{"id": "critic", "verdict": "pass", "note": "two claims unsupported",
                                    "seconds": 4}]))
    doc = _convert(row, flow)
    assert _verdicts(doc)[2]["ext"] == {NS: {"note": "two claims unsupported", "seconds": 4}}
    assert "ext" not in _verdicts(doc)[0]
    assert doc["privacy"]["profile"] == "full"
    assert doc["run"]["ext"][NS]["retained_text"] == ["stages.4.checks.0.note"]
    assert all("extra" not in a["ext"][NS] for a in doc["attempts"])  # checks is a known stage member
    plain = _convert(_checked(), flow)
    assert plain["privacy"]["profile"] == "metadata_only" and "retained_text" not in plain["run"]["ext"][NS]


# check_row and the published schema agree ---------------------------------------------------------

def _with_checks(value) -> dict:
    row = _kept()
    row["stages"][1]["checks"] = value
    return row


GOOD = [
    [],
    None,
    [{"id": "extract", "verdict": "pass"}, {"id": "critic", "verdict": "fail"}],
    [{"id": "extract", "verdict": "pass", "note": "kept", "ext": {"example.note": 1}}],
    [{"id": "x" * 200, "verdict": "fail"}],
]

BAD = [  # (the stage's checks, the member check_row names)
    ("extract", "stages.1.checks"),
    ({"id": "extract", "verdict": "pass"}, "stages.1.checks"),
    (["extract"], "stages.1.checks.0"),
    ([{"verdict": "pass"}], "stages.1.checks.0.id"),
    ([{"id": "", "verdict": "pass"}], "stages.1.checks.0.id"),
    ([{"id": "x" * 201, "verdict": "pass"}], "stages.1.checks.0.id"),
    ([{"id": 3, "verdict": "pass"}], "stages.1.checks.0.id"),
    ([{"id": "extract"}], "stages.1.checks.0.verdict"),
    ([{"id": "extract", "verdict": None}], "stages.1.checks.0.verdict"),
    ([{"id": "extract", "verdict": "accept"}], "stages.1.checks.0.verdict"),
    ([{"id": "extract", "verdict": "PASS"}], "stages.1.checks.0.verdict"),
    ([{"id": "extract", "verdict": True}], "stages.1.checks.0.verdict"),
]

# Refused by check_row only: the schema cannot compare one item's id with another's
CODE_ONLY = [
    ([{"id": "extract", "verdict": "pass"}, {"id": "extract", "verdict": "fail"}],
     "stages.1.checks.1.id: 'extract' is reported at stages.1.checks.0 too"),
    ([{"id": "extract", "verdict": "pass"}, {"id": "critic", "verdict": "pass"}, {"id": "extract", "verdict": "pass"}],
     "stages.1.checks.2.id: 'extract' is reported at stages.1.checks.0 too"),
]


@pytest.mark.parametrize("value", GOOD)
def test_check_row_and_the_schema_accept_the_checks_shapes(value):
    row = _with_checks(copy.deepcopy(value))
    assert HR.check_row(row) == [] and not list(_validator().iter_errors(row))


@pytest.mark.parametrize(("value", "named"), BAD)
def test_check_row_and_the_schema_refuse_the_same_checks(value, named):
    row = _with_checks(copy.deepcopy(value))
    problems = HR.check_row(row)
    assert any(p.startswith(f"{named}:") for p in problems), problems
    assert list(_validator().iter_errors(row))


@pytest.mark.parametrize(("value", "named"), CODE_ONLY)
def test_a_repeated_check_is_refused_by_check_row_only(value, named):
    row = _with_checks(copy.deepcopy(value))
    problems = HR.check_row(row)
    assert any(p.startswith(named) for p in problems), problems
    assert not list(_validator().iter_errors(row))


# Simulated runs -----------------------------------------------------------------------------------

def test_a_simulated_row_with_checks_validates_with_the_flow(flow):
    row = _checked(ext={SIMULATE: {"seed": 7, "truth": "0123456789ab"}})
    doc = _convert(row, flow)
    assert {a["cost"]["basis"] for a in doc["attempts"]} == {"simulated"}
    assert len(_verdicts(doc)) == 3 and _accepted(doc) == (1.0, ())
    assert _findings(flow, doc) == [[], []]
