"""Checks (OCP 0.4 `control.checks`) in the workflow file format, the workflow types and the emitter, and the
store's version rule (E005)."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from loopmath.ocp import conformance, emit
from loopmath.ocp.canonical import GATE_RULES_KEY, config_id
from loopmath.types import Check, Configuration, Control, Gate, Piece, Setting, Workflow
from loopmath.workflows import ocp as wocp
from loopmath.workflows.candidates import twin_key
from loopmath.workflows.format import (WorkflowFormatError, catalog, dump_workflow, loads_workflow, loads_workflow_file,
                                       validate_configuration, validate_workflow)
from loopmath.workflows.ids import make_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "spec" / "examples"
BASE = json.loads((ROOT / "tests" / "fixtures" / "flows" / "to_dict.base.json").read_text(encoding="utf-8"))
RULE = {"name": "tests+review", "definition": "tests pass and the reviewer approves", "requires": ["tests"]}
AT = "2026-11-03T10:00:00-08:00"
SETTINGS = {"implement": Setting("api", "claude-opus-5-5", "high"), "review": Setting("api", "gpt-6-astra", "high")}

CHECKS = """\
id = "review_twice"
version = 1
title = "Implement, then two checks after review"
artifacts = ["patch", "notes"]
edges = ["implement -> patch", "patch -> review", "review -> notes"]

[[pieces]]
id = "implement"
role = "implementer"

[[pieces]]
id = "review"
role = "reviewer"

[control]
budget_rounds = 3

[[control.checks]]
id = "review-approves"
after = "review"
rule = "review_approve"
on_fail = "implement"

[[control.checks]]
id = "tests-pass"
after = "review"
rule = "tests_pass"
severity = "hard"
"""

# The workflow part of the lit-brief design file the round-2 check read (ocp-check/roundtrip.out.json)
LIT_BRIEF = """\
id = "lit-brief"
version = 1
title = "Question to a one-page brief"
edges = ["question -> scout", "scout -> sources", "sources -> extract", "extract -> claims",
         "claims -> synthesize", "question -> synthesize", "synthesize -> brief", "brief -> critic",
         "claims -> critic", "critic -> verdict"]
artifacts = ["question", "sources", "claims", "brief", "verdict"]

[[pieces]]
id = "scout"
role = "worker"
[[pieces]]
id = "extract"
role = "worker"
[[pieces]]
id = "synthesize"
role = "worker"
[[pieces]]
id = "critic"
role = "reviewer"

[control]
budget_rounds = 2
[[control.gates]]
after = "extract"
rule = "command:claims-check"
on_fail = "extract"
[[control.gates]]
after = "critic"
rule = "review_approve"
on_fail = "synthesize"

[settings]
scout      = { harness = "api", model = "claude-sonnet-5-5", effort = "default" }
extract    = { harness = "api", model = "claude-opus-5-5",   effort = "high" }
synthesize = { harness = "api", model = "gpt-6-astra",       effort = "high" }
critic     = { harness = "api", model = "claude-haiku-5-5",  effort = "default" }
"""
LIT_BRIEF_ID = "cfg_9bc4f6e876f9"
CHECKS_FORM = ("v0.4/two-checks.ocp.json", "golden/v0.4/E217-pass.ocp.json", "golden/v0.4/E217-fail.ocp.json")
MIXED = "control.checks is used with"


def _read(rel: str) -> dict:
    return json.loads((EXAMPLES / rel).read_text(encoding="utf-8"))


def _errors(doc: dict) -> list[tuple[str, str]]:
    return [(f.code, f.message) for f in conformance.validate_doc(doc) if f.level == "error"]


def _e217(doc: dict) -> list[tuple[str, str]]:
    return [(f.code, f.message) for f in conformance.validate_doc(doc) if f.code == "E217"]


def _run_doc(configuration) -> dict:
    return emit.new_run_doc(run_id="run_checks01", task={"id": "tsk_checks01", "type": "feature"},
                            configuration=configuration, acceptance_rule=RULE,
                            provenance={"kind": "designed", "chooser": "user"}, started_at=AT)


def _review(gates=(), checks=()) -> Workflow:
    return Workflow(id="ir", version=1, title="Implement, then review",
                    pieces=(Piece("implement", "implementer"), Piece("review", "reviewer")),
                    artifacts=("patch", "notes"),
                    edges=(("implement", "patch"), ("patch", "review"), ("review", "notes")),
                    control=Control(gates=tuple(gates), checks=tuple(checks), budget_rounds=2))


def _replace_text(old: str, new: str) -> str:
    assert old in CHECKS
    return CHECKS.replace(old, new)


# ------------------------------------------------------------------ the file format

def test_checks_read_validate_dump_and_read_again():
    wf = loads_workflow(CHECKS)
    assert wf.control.gates == ()
    assert [(k.id, k.after, k.rule, k.on_fail, k.extra) for k in wf.control.checks] == [
        ("review-approves", "review", "review_approve", "implement", {}),
        ("tests-pass", "review", "tests_pass", None, {"severity": "hard"})]
    assert validate_workflow(wf) == []
    text = dump_workflow(wf)
    assert text.count("[[control.checks]]") == 2 and "[[control.gates]]" not in text
    assert loads_workflow(text) == wf and dump_workflow(loads_workflow(text)) == text
    assert "on_fail" not in wf.control.checks[1].to_dict()  # absent stays absent


def test_a_file_without_checks_dumps_as_before():
    for wf in catalog().values():
        assert wf.control.checks == () and "checks" not in dump_workflow(wf)


GATE = '\n[[control.gates]]\nafter = "review"\non_fail = "implement"\n'
NO_CHECKS = CHECKS.split("\n[[control.checks]]")[0]
MIXED_FILES = {"both given": CHECKS + GATE, "checks empty": NO_CHECKS + "checks = []\n" + GATE,
         "gates empty": CHECKS.replace("budget_rounds = 3\n", "budget_rounds = 3\ngates = []\n"),
         "both empty": NO_CHECKS + "checks = []\ngates = []\n"}


@pytest.mark.parametrize("name", list(MIXED_FILES))
def test_a_file_with_gates_and_checks_is_refused(name):
    with pytest.raises(WorkflowFormatError) as caught:
        loads_workflow(MIXED_FILES[name])
    assert caught.value.errors == ["control: both gates and checks are given; a workflow states its checks as "
                                   "[[control.checks]] or as [[control.gates]], not both"]


def test_a_workflow_with_gates_and_checks_is_refused():
    wf = loads_workflow(CHECKS)
    gate = Gate(id="g_review", after="review", rule="review_approve", on_fail="implement")
    wf = dataclasses.replace(wf, control=dataclasses.replace(wf.control, gates=(gate,)))
    assert validate_workflow(wf) == [
        "a workflow states its checks as [[control.checks]] or as [[control.gates]], not both"]


USUAL = {"kind": "configuration", "ref": "usual"}


def test_an_empty_checks_list_is_the_checks_form_with_no_checks():
    wf = loads_workflow(NO_CHECKS + "checks = []\n")
    assert wf.control.checks == () and wf.control.gates == () and wf.control.extra == {"checks": []}
    assert validate_workflow(wf) == []
    text = dump_workflow(wf)
    assert "checks = []" in text and loads_workflow(text) == wf and dump_workflow(wf, empty_gates=True) == text
    assert emit.workflow_to_ocp(wf)["control"] == {"checks": [], "budget": 3, "rescue": USUAL}
    doc = _run_doc(Configuration("", wf, SETTINGS))
    assert doc["ocp"] == "0.4" and _errors(doc) == [] and emit.validate_strict(doc) == []
    assert wocp.workflow_from_ocp(emit.workflow_to_ocp(wf)).control == wf.control


def test_an_empty_gates_list_is_the_gates_form_and_written_only_when_asked():
    wf = loads_workflow(NO_CHECKS + "gates = []\n")
    assert wf == loads_workflow(NO_CHECKS)
    assert emit.workflow_to_ocp(wf)["control"] == {"gates": [], "repair": {}, "budget": 3, "rescue": USUAL}
    assert _run_doc(Configuration("", wf, SETTINGS))["ocp"] == "0.3"
    assert "gates = []" not in dump_workflow(wf)  # the file format's text, as before checks existed
    text = dump_workflow(wf, empty_gates=True)
    assert "gates = []" in text and loads_workflow(text) == wf
    assert "gates = []" not in dump_workflow(loads_workflow(CHECKS), empty_gates=True)


@pytest.mark.parametrize(("old", "new", "error"), [
    ('id = "tests-pass"', 'id = "review-approves"', "check id 'review-approves' appears twice"),
    ('rule = "tests_pass"', 'rule = "review_approve"',
     "check 'tests-pass': two checks after 'review' have rule 'review_approve'"),
    ('after = "review"\nrule = "tests_pass"', 'after = "notes"\nrule = "tests_pass"',
     "check 'tests-pass': after 'notes' is not a piece"),
    ('severity = "hard"', 'on_fail = "nobody"', "check 'tests-pass': on_fail 'nobody' is not a piece"),
], ids=["id-twice", "rule-twice", "after", "on-fail-piece"])
def test_each_check_error_by_name(old, new, error):
    assert validate_workflow(loads_workflow(_replace_text(old, new))) == [error]


def test_a_check_sending_downstream_is_refused():
    wf = _review(checks=(Check("first", "implement", "tests_pass", on_fail="review"),))
    assert validate_workflow(wf) == ["check 'first': on_fail 'review' is not upstream of 'implement'"]


def test_a_check_without_a_rule():
    with pytest.raises(WorkflowFormatError, match=r"control\.checks\[1\]\.rule: expected a non-empty string"):
        loads_workflow(_replace_text('rule = "tests_pass"\n', ""))
    assert validate_workflow(_review(checks=(Check("bare", "review", ""),))) == ["check 'bare' has no rule"]


def test_a_check_on_fail_given_empty_is_refused():
    with pytest.raises(WorkflowFormatError, match=r"control\.checks\[1\]\.on_fail: expected a non-empty string"):
        loads_workflow(_replace_text('severity = "hard"', 'on_fail = ""'))


# ------------------------------------------------------------------ the emitter forms

@pytest.mark.parametrize("name", list(catalog()))
def test_catalog_gates_keep_the_gates_form_and_0_3(name):
    wf = catalog()[name]
    control = emit.workflow_to_ocp(wf)["control"]
    assert "checks" not in control and control["gates"] == [g.after for g in wf.control.gates]
    assert _run_doc(Configuration("", wf, {}))["ocp"] == "0.3"


def test_a_checks_control_is_written_alone_and_gives_0_4():
    wf = loads_workflow(CHECKS)
    control = emit.workflow_to_ocp(wf)["control"]
    assert control == {"checks": [
        {"id": "review-approves", "after": "review", "rule": "review_approve", "on_fail": "implement"},
        {"id": "tests-pass", "after": "review", "rule": "tests_pass", "severity": "hard"}],
        "budget": 3, "rescue": {"kind": "configuration", "ref": "usual"}}
    doc = _run_doc(Configuration("", wf, SETTINGS))
    assert doc["ocp"] == "0.4" and _errors(doc) == [] and emit.validate_strict(doc) == []
    back = wocp.workflow_from_ocp(emit.workflow_to_ocp(wf))
    assert back.control.checks == wf.control.checks and back.control.gates == ()


@pytest.mark.parametrize("gate", [
    Gate("claims", "review", "review_approve", on_fail="implement"),
    Gate("g_review", "review", "review_approve", on_fail="implement", extra={"severity": "hard"}),
], ids=["non-default-id", "extra"])
def test_a_gate_the_gates_form_cannot_carry_is_written_as_a_check(gate):
    plain = _review(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),))
    wf = _review(gates=(gate,))
    control = emit.workflow_to_ocp(wf)["control"]
    assert control["checks"] == [{"id": gate.id, "after": "review", "rule": "review_approve", "on_fail": "implement",
                                  **gate.extra}]
    assert not {"gates", "repair"} & set(control)
    assert make_config(wf, SETTINGS).id == make_config(plain, SETTINGS).id  # the id does not move
    assert _run_doc(Configuration("", wf, SETTINGS))["ocp"] == "0.4"
    assert _run_doc(Configuration("", plain, SETTINGS))["ocp"] == "0.3"


def test_two_gates_after_one_piece_are_written_as_checks():
    wf = _review(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),
                        Gate("g_tests", "review", "tests_pass")))
    control = emit.workflow_to_ocp(wf)["control"]
    assert [c["id"] for c in control["checks"]] == ["g_review", "g_tests"] and "gates" not in control
    assert "on_fail" not in control["checks"][1]
    assert _errors(_run_doc(Configuration("", wf, SETTINGS))) == []


def test_a_gate_rule_beside_checks_is_not_written():
    wf = loads_workflow(CHECKS)
    wf = dataclasses.replace(wf, control=dataclasses.replace(
        wf.control, extra={"ext": {GATE_RULES_KEY: {"review": "tests_pass"}, "x.kept": 1}}))
    assert emit.workflow_to_ocp(wf)["control"]["ext"] == {"x.kept": 1}


def test_a_mixed_control_keeps_the_mix_and_e217_names_it():
    wf = _review(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),),
                 checks=(Check("tests-pass", "review", "tests_pass"),))
    control = emit.workflow_to_ocp(wf)["control"]
    assert control["gates"] == ["review"] and control["repair"] == {"review": "implement"}
    assert control["checks"] == [{"id": "tests-pass", "after": "review", "rule": "tests_pass"}]
    doc = _run_doc(Configuration("", wf, SETTINGS))
    assert doc["ocp"] == "0.4"
    assert _e217(doc) == [("E217", "control.checks is used with gates, repair; a workflow states its checks as "
                                   "checks or as gates, not both")]


@pytest.mark.parametrize("rel", CHECKS_FORM)
def test_checks_form_documents_emit_their_own_workflow(rel):
    """The stored `ocp` value equals the source workflow, its E217 findings are the source's (code and message),
    and none names a mix."""
    doc = _read(rel)
    record = wocp.configuration_from_ocp(doc["run"]["configuration"])
    emitted = wocp.configuration_to_ocp(record)
    assert emitted["workflow"] == doc["run"]["configuration"]["workflow"]
    assert emitted == BASE[f"ocp/{rel}"]["ocp"]
    again = copy.deepcopy(doc)
    again["run"]["configuration"]["workflow"] = emitted["workflow"]
    want = [] if "fail" not in rel else [("E217", "two checks after 'review' have rule 'review_approve'")]
    assert _e217(doc) == _e217(again) == want
    assert not any(message.startswith(MIXED) for _, message in _e217(again))
    assert _run_doc(record)["ocp"] == "0.4"


# ------------------------------------------------------------------ the round-2 regression

def test_round_2_the_design_file_gives_the_same_configuration():
    f = loads_workflow_file(LIT_BRIEF, "lit-brief")
    cfg = make_config(f.workflow, f.settings)
    assert cfg.id == LIT_BRIEF_ID and validate_configuration(cfg) == []
    ocp_cfg = wocp.configuration_to_ocp(cfg, source="usual")
    control = ocp_cfg["workflow"]["control"]
    assert config_id(ocp_cfg["workflow"], ocp_cfg["settings"]) == LIT_BRIEF_ID
    assert control["gates"] == ["extract", "critic"]
    assert control["repair"] == {"extract": "extract", "critic": "synthesize"}
    assert control["ext"] == {GATE_RULES_KEY: {"extract": "command:claims-check"}}
    back = wocp.configuration_from_ocp(ocp_cfg)
    f2 = loads_workflow_file(dump_workflow(back.workflow, back.settings), "back")
    cfg2 = make_config(f2.workflow, f2.settings)
    assert cfg2.id == LIT_BRIEF_ID
    assert cfg2.workflow.pieces == cfg.workflow.pieces and sorted(cfg2.workflow.edges) == sorted(cfg.workflow.edges)
    assert cfg2.workflow.control.gates == cfg.workflow.control.gates
    assert {k: (s.harness, s.model, s.effort, s.context_policy, s.options) for k, s in cfg2.settings.items()} == \
           {k: (s.harness, s.model, s.effort, s.context_policy, s.options) for k, s in cfg.settings.items()}
    assert wocp.configuration_to_ocp(cfg2, source="usual") == ocp_cfg


def test_round_2_a_gate_id_and_extra_are_kept_with_the_config_id():
    text = LIT_BRIEF.replace('after = "extract"', 'id = "claims"\nafter = "extract"\nseverity = "hard"')
    f = loads_workflow_file(text, "claims")
    cfg = make_config(f.workflow, f.settings)
    assert [(g.id, g.after, g.extra) for g in cfg.workflow.control.gates] == [
        ("claims", "extract", {"severity": "hard"}), ("g_critic", "critic", {})]
    back = wocp.configuration_from_ocp(wocp.configuration_to_ocp(cfg))
    assert [(k.id, k.after, k.rule, k.on_fail, k.extra) for k in back.workflow.control.checks] == [
        ("claims", "extract", "command:claims-check", "extract", {"severity": "hard"}),
        ("g_critic", "critic", "review_approve", "synthesize", {})]
    assert cfg.id == back.id == LIT_BRIEF_ID


def test_round_2_two_checks_after_extract_read_validate_and_round_trip():
    head, tail = LIT_BRIEF.split("[[control.gates]]", 1)
    text = head + ('[[control.checks]]\nid = "claims"\nafter = "extract"\nrule = "command:claims-check"\n'
                   'on_fail = "extract"\n[[control.checks]]\nid = "schema"\nafter = "extract"\n'
                   'rule = "command:schema-check"\non_fail = "extract"\n[[control.checks]]\nid = "g_critic"\n'
                   'after = "critic"\nrule = "review_approve"\non_fail = "synthesize"\n') + \
        tail[tail.index("[settings]"):]
    f = loads_workflow_file(text, "two-checks")
    cfg = make_config(f.workflow, f.settings)
    assert validate_configuration(cfg) == []
    ocp_cfg = wocp.configuration_to_ocp(cfg, source="usual")
    assert [c["id"] for c in ocp_cfg["workflow"]["control"]["checks"]] == ["claims", "schema", "g_critic"]
    doc = _run_doc(cfg)
    assert doc["ocp"] == "0.4"
    assert not [code for code, _ in _errors(doc) if code in ("E010", "E217")] and _errors(doc) == []
    back = wocp.configuration_from_ocp(ocp_cfg)
    assert back.id == cfg.id and back.workflow.control.checks == cfg.workflow.control.checks
    assert wocp.configuration_to_ocp(back, source="usual") == ocp_cfg
    f2 = loads_workflow_file(dump_workflow(back.workflow, back.settings), "back")
    assert make_config(f2.workflow, f2.settings).id == cfg.id


# ------------------------------------------------------------------ twin_key

def test_checks_only_differences_are_not_twins():
    one = make_config(_review(checks=(Check("a", "review", "review_approve", on_fail="implement"),)), SETTINGS)
    two = make_config(_review(checks=(Check("a", "review", "review_approve", on_fail="implement"),
                                      Check("b", "review", "tests_pass"))), SETTINGS)
    stop = make_config(_review(checks=(Check("a", "review", "review_approve"),)), SETTINGS)
    assert len({twin_key(one), twin_key(two), twin_key(stop)}) == 3


@pytest.mark.parametrize("name", list(catalog()))
def test_a_configuration_without_checks_keeps_its_key(name):
    wf = catalog()[name]
    key = twin_key(make_config(wf, {}))
    assert key[3] == tuple((g.after, g.rule, g.on_fail, json.dumps(g.extra, sort_keys=True))
                           for g in wf.control.gates)


def test_a_check_and_the_gate_it_restates_are_twins():
    gate = make_config(_review(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),)), SETTINGS)
    check = make_config(_review(checks=(Check("g_review", "review", "review_approve", on_fail="implement"),)),
                        SETTINGS)
    assert gate.id == check.id and twin_key(gate) == twin_key(check)


# ------------------------------------------------------------------ the store's version rule (E005)

@pytest.mark.parametrize("rel", ["v0.3/solo.ocp.json", "v0.4/lit-brief.run-1.ocp.json", "v0.4/lit-brief.run-2.ocp.json",
                                 "v0.4/two-checks.ocp.json"])
def test_the_store_takes_0_3_and_0_4_runs_with_a_flow_or_checks(rel):
    assert [f for f in emit.validate_strict(_read(rel)) if f["level"] == "error"] == []


E005 = [
    ("flow", "the store takes run documents; this is a flow document (kind 'flow'), which the flow commands read"),
    ("neither", "the store takes OCP 0.4 run documents that name a flow (run.flow) or state control.checks; this "
                "one does neither, so write it as OCP 0.3"),
    ("0.2", "the store takes OCP 0.3 documents, got '0.2' (run 'loopmath ocp migrate')"),
    ("0.5", "the store takes OCP 0.3 documents, got '0.5' (run 'loopmath ocp migrate')"),
    ("none", "the store takes OCP 0.3 documents, got None"),
]


@pytest.mark.parametrize(("case", "message"), E005, ids=[c for c, _ in E005])
def test_e005_names_what_the_store_refuses(case, message):
    if case == "flow":
        doc = _read("v0.4/lit-brief.flow.ocp.json")
    else:
        doc = _read("v0.3/solo.ocp.json")
        if case == "none":
            del doc["ocp"]
        else:
            doc["ocp"] = {"neither": "0.4"}.get(case, case)
    found = emit.validate_strict(doc)
    assert found[0] == {"level": "error", "code": "E005", "path": "$['ocp']", "message": message}
    assert [f for f in found if f["code"] == "E005"] == found[:1]
    assert emit.is_flow_document(doc) is (case == "flow") and not emit.run_uses_v04(doc)
