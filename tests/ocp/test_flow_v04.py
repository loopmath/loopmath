"""OCP 0.4 flow documents: kind, the flow object, its digest and rules E212 to E216."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys

import pytest

from loopmath import flows
from loopmath.cli import main
from loopmath.flows.plan import margin_problem, margin_table
from loopmath.ocp import conformance
from loopmath.ocp.canonical import FLOW_DIGEST_MEMBERS, config_id, flow_digest

from ._common import EXAMPLES, ROOT, SPEC, codes, example, load
from .test_checks_v04 import ID_FILES

FLOW_EXAMPLE = EXAMPLES / "v0.4" / "lit-brief.flow.ocp.json"
FLOW = "$['flow']"
RUN_MEMBERS = ("run", "nodes", "edges", "attempts", "artifacts", "events", "groups")


def flow_doc(digest=False):
    """The example; without its digest unless asked, so an edit does not also give E212."""
    doc = load(FLOW_EXAMPLE)
    if not digest:
        del doc["flow"]["digest"]
    return doc


def _rehash(doc):
    configuration = doc["flow"]["configuration"]
    configuration["id"] = config_id(configuration["workflow"], configuration["settings"])
    return doc


def _edited(edit):
    """The example without its digest, with edit applied to its flow object and the configuration id recomputed."""
    doc = flow_doc()
    edit(doc["flow"])
    return _rehash(doc)


def _checks_form(flow):
    """The base workflow with its two gates written as checks; the id stays the gates form's."""
    control = flow["configuration"]["workflow"]["control"]
    for name in ("gates", "repair", "ext"):
        del control[name]
    control["checks"] = [
        {"id": "claims-check", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
        {"id": "critic-approves", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"},
    ]
    return flow


# The example ---------------------------------------------------------------------------------------

def test_the_example_has_no_findings_and_its_stored_ids_hold():
    assert conformance.validate_file(FLOW_EXAMPLE) == []
    flow = load(FLOW_EXAMPLE)["flow"]
    configuration = flow["configuration"]
    assert configuration["id"] == "cfg_9bc4f6e876f9"
    assert config_id(configuration["workflow"], configuration["settings"]) == configuration["id"]
    assert flow["digest"] == flow_digest(flow)


def test_the_stored_id_set_reads_every_flow_file():
    flow_files = [p for p in sorted(EXAMPLES.rglob("*.ocp.json")) if load(p).get("kind") == "flow"]
    assert len(flow_files) == 11  # the example, and one pass and one fail golden for E212 to E216
    assert set(flow_files) <= set(ID_FILES)


def test_the_checks_form_of_the_base_workflow_keeps_its_id():
    flow = _checks_form(flow_doc()["flow"])
    configuration = flow["configuration"]
    assert config_id(configuration["workflow"], configuration["settings"]) == "cfg_9bc4f6e876f9"


# E010: the document kinds ---------------------------------------------------------------------------

@pytest.mark.parametrize("member", RUN_MEMBERS)
def test_a_flow_document_carrying_a_run_member_fails_e010(member):
    doc = flow_doc(digest=True)
    doc[member] = {} if member == "run" else []
    findings = conformance.validate_doc(doc)
    assert codes(findings) == ["E010"]
    assert {f.path for f in findings} == {f"$['{member}']"}
    assert any("a flow document carries no run" in f.message for f in findings)


def test_a_flow_document_without_flow_fails_e010():
    doc = flow_doc(digest=True)
    del doc["flow"]
    findings = conformance.validate_doc(doc)
    assert [(f.code, f.path) for f in findings] == [("E010", "$")]
    assert "'flow' is a required property" in findings[0].message


def test_an_unknown_kind_fails_e010():
    doc = flow_doc(digest=True)
    doc["kind"] = "plan"
    assert [(f.code, f.path) for f in conformance.validate_doc(doc)] == [("E010", "$['kind']")]


def test_kind_flow_at_0_3_fails(capsys, tmp_path):
    doc = flow_doc(digest=True)
    doc["ocp"] = "0.3"
    assert "E010" in codes(f for f in conformance.validate_doc(doc) if f.level == "error")
    path = tmp_path / "lit-brief.flow.ocp.json"
    path.write_text(json.dumps(doc, indent=2))
    assert main(["ocp", "validate", str(path)]) == 1
    assert capsys.readouterr().out.startswith("FAIL")


def test_kind_in_a_0_3_run_document_is_an_unknown_field():
    doc = example("implement_review")
    tagged = copy.deepcopy(doc)
    tagged["kind"] = "flow"
    assert conformance.validate_doc(tagged) == conformance.validate_doc(doc) == []


def test_kind_flow_in_a_0_3_run_document_fails_once_raised():
    doc = example("implement_review")
    doc["kind"] = "flow"
    doc["ocp"] = "0.4"
    assert "E010" in codes(f for f in conformance.validate_doc(doc) if f.level == "error")


def test_a_run_document_with_kind_run_validates_as_without_kind():
    doc = example("implement_review")
    doc["ocp"] = "0.4"
    with_kind = copy.deepcopy(doc)
    with_kind["kind"] = "run"
    assert conformance.validate_doc(with_kind) == conformance.validate_doc(doc) == []
    for record in (doc, with_kind):
        del record["nodes"]
    assert conformance.validate_doc(with_kind) == conformance.validate_doc(doc)
    assert ("E010", "$") in [(f.code, f.path) for f in conformance.validate_doc(with_kind)]


def test_a_flow_document_with_the_flow_capability_false_gives_e180():
    doc = flow_doc(digest=True)
    doc["producer"]["capabilities"]["flow"] = False
    assert [(f.code, f.path) for f in conformance.validate_doc(doc)] == [
        ("E180", "$['producer']['capabilities']['flow']")]


def test_a_flow_configuration_with_the_configuration_capability_false_gives_e180():
    doc = flow_doc(digest=True)
    doc["producer"]["capabilities"]["configuration"] = False
    assert [(f.code, f.path) for f in conformance.validate_doc(doc)] == [
        ("E180", "$['producer']['capabilities']['configuration']")]


# flow.configuration is checked as run.configuration ------------------------------------------------

def test_a_wrong_flow_configuration_id_gives_e190():
    doc = flow_doc()
    doc["flow"]["configuration"]["id"] = "cfg_000000000000"
    assert [(f.code, f.path) for f in conformance.validate_doc(doc)] == [
        ("E190", f"{FLOW}['configuration']['id']")]


def test_a_piece_without_a_setting_gives_e191():
    doc = flow_doc()
    del doc["flow"]["configuration"]["settings"]["scout"]
    assert [(f.code, f.path) for f in conformance.validate_doc(_rehash(doc))] == [
        ("E191", f"{FLOW}['configuration']['settings']")]


def test_a_repeated_check_id_gives_e217():
    doc = flow_doc()
    _checks_form(doc["flow"])["configuration"]["workflow"]["control"]["checks"][1]["id"] = "claims-check"
    assert [(f.code, f.path) for f in conformance.validate_doc(_rehash(doc))] == [
        ("E217", f"{FLOW}['configuration']['workflow']['control']['checks'][1]['id']")]


# E213 to E216 ---------------------------------------------------------------------------------------

def _pieces(flow):
    return flow["allow"]["pieces"]


def _merge(flow, **fields):
    flow["allow"]["merges"][0].update(fields)


def _no_control(flow):
    """The base workflow without control: no gate and no check."""
    del flow["configuration"]["workflow"]["control"]
    return flow


def _second_merge(flow):
    merge = copy.deepcopy(flow["allow"]["merges"][0])
    merge["into"] = {"id": "extract_synthesize_2", "role": "worker"}
    flow["allow"]["merges"].append(merge)


def _judge(flow, index, **fields):
    flow["judges"][index].update(fields)


def _objective(flow, **fields):
    flow["objective"] = fields


def _drop(record, *names):
    for name in names:
        del record[name]


ALLOW = f"{FLOW}['allow']"
CASES = {
    "E213 piece not in the workflow": (lambda f: _pieces(f).update(writer={}), f"{ALLOW}['pieces']['writer']"),
    "E213 width without select": (lambda f: _pieces(f)["scout"].pop("select"), f"{ALLOW}['pieces']['scout']['widths']"),
    "E213 merge of one piece": (lambda f: _merge(f, replace=["extract"]), f"{ALLOW}['merges'][0]['replace']"),
    "E213 merge of an unknown piece": (lambda f: _merge(f, replace=["extract", "writer"]),
                                       f"{ALLOW}['merges'][0]['replace'][1]"),
    "E213 merge of no chain": (lambda f: _merge(f, replace=["scout", "synthesize"]),
                               f"{ALLOW}['merges'][0]['replace'][1]"),
    "E213 merge into a piece id": (lambda f: _merge(f, into={"id": "critic", "role": "worker"}),
                                   f"{ALLOW}['merges'][0]['into']['id']"),
    "E213 merge into an artifact id": (lambda f: _merge(f, into={"id": "claims", "role": "worker"}),
                                       f"{ALLOW}['merges'][0]['into']['id']"),
    "E213 merge id twice": (_second_merge, f"{ALLOW}['merges'][1]['id']"),
    "E213 remove of no gate": (lambda f: f["allow"].update(checks={"remove": ["synthesize"]}),
                               f"{ALLOW}['checks']['remove'][0]"),
    "E213 remove of no check id": (lambda f: _checks_form(f)["allow"].update(checks={"remove": ["extract"]}),
                                   f"{ALLOW}['checks']['remove'][0]"),
    "E213 remove with no base control": (lambda f: _no_control(f)["allow"].update(checks={"remove": ["extract"]}),
                                         f"{ALLOW}['checks']['remove'][0]"),
    "E214 judge id twice": (lambda f: _judge(f, 1, id="j-claude"), f"{FLOW}['judges'][1]['id']"),
    "E214 rate above 1": (lambda f: _judge(f, 2, rate=1.5), f"{FLOW}['judges'][2]['rate']"),
    "E214 rate below 0": (lambda f: _judge(f, 2, rate=-0.2), f"{FLOW}['judges'][2]['rate']"),
    "E214 linear without range": (lambda f: _judge(f, 0, scale="linear"), f"{FLOW}['judges'][0]"),
    "E214 linear range falling": (lambda f: _judge(f, 0, scale="linear", range=[5, 1]),
                                  f"{FLOW}['judges'][0]['range']"),
    "E214 linear range of one number": (lambda f: _judge(f, 0, scale="linear", range=[1]),
                                        f"{FLOW}['judges'][0]['range']"),
    "E215 margin below 0": (lambda f: f["objective"].update(margin=-0.1), f"{FLOW}['objective']['margin']"),
    "E215 margin missing": (lambda f: f["objective"].pop("margin"), f"{FLOW}['objective']"),
    "E215 score missing": (lambda f: f["objective"].pop("score"), f"{FLOW}['objective']"),
    "E215 cost cap of 0": (lambda f: _objective(f, kind="best_under_cost", score="quality", cost_cap_usd=0),
                           f"{FLOW}['objective']['cost_cap_usd']"),
    "E215 cost cap missing": (lambda f: _objective(f, kind="best_under_cost", score="quality"),
                              f"{FLOW}['objective']"),
    "E215 objective score is a required verdict": (lambda f: f["objective"].update(score="claims-check"),
                                                   f"{FLOW}['objective']['score']"),
    "E215 judge score is a required verdict": (lambda f: _judge(f, 0, score="claims-check"),
                                               f"{FLOW}['judges'][0]['score']"),
    "E216 id twice": (lambda f: f["inputs"]["ids"].__setitem__(1, "q01"), f"{FLOW}['inputs']['ids'][1]"),
    "E216 holdout id twice": (lambda f: f["inputs"]["holdout"].__setitem__(1, "q13"),
                              f"{FLOW}['inputs']['holdout'][1]"),
    "E216 holdout outside ids": (lambda f: f["inputs"]["holdout"].__setitem__(7, "q21"),
                                 f"{FLOW}['inputs']['holdout'][7]"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_each_flow_rule_is_reported_at_its_member(case):
    edit, path = CASES[case]
    assert [(f.code, f.path) for f in conformance.validate_doc(_edited(edit))] == [(case.split()[0], path)]


@pytest.mark.parametrize("edit", [
    pytest.param(lambda f: f["allow"].update(checks={"remove": ["extract", "critic"]}), id="remove-gates"),
    pytest.param(lambda f: _checks_form(f)["allow"].update(checks={"remove": ["claims-check"]}), id="remove-check"),
    pytest.param(lambda f: _pieces(f).update(scout={"widths": [1]}), id="width-1-without-select"),
    pytest.param(lambda f: _judge(f, 0, scale="linear", range=[1, 5]), id="linear-range"),
    pytest.param(lambda f: _judge(f, 2, rate=0), id="rate-0"),
    pytest.param(lambda f: _judge(f, 2, rate=1), id="rate-1"),
    pytest.param(lambda f: f["objective"].update(margin=1e300), id="margin-1e300"),
    pytest.param(lambda f: _objective(f, kind="cost_per_accepted"), id="cost-per-accepted"),
    pytest.param(lambda f: _objective(f, kind="a_kind_of_its_own"), id="open-kind"),
    pytest.param(lambda f: f["inputs"].pop("ids"), id="holdout-without-ids"),
    pytest.param(lambda f: _drop(f, "title", "hypothesis", "judges", "allow"), id="optional-members"),
])
def test_valid_variants_have_no_findings(edit):
    assert conformance.validate_doc(_edited(edit)) == []


def test_a_referenced_base_workflow_is_resolved_for_e213():
    doc = flow_doc()
    configuration = doc["flow"]["configuration"]
    workflow = configuration["workflow"]
    configuration["workflow"] = {"ref": "lit-brief", "version": 1}
    _pieces(doc["flow"]).update(writer={})

    def resolve(ref, version):
        return copy.deepcopy(workflow) if (ref, version) == ("lit-brief", 1) else None

    assert [(f.code, f.path) for f in conformance.validate_doc(doc, resolve=resolve)] == [
        ("E213", f"{ALLOW}['pieces']['writer']")]


def test_an_unresolved_base_workflow_skips_only_the_parts_that_read_it():
    doc = flow_doc()
    doc["flow"]["configuration"]["workflow"] = {"ref": "not-in-any-catalog", "version": 1}
    _pieces(doc["flow"]).update(writer={})
    _pieces(doc["flow"])["scout"].pop("select")
    _second_merge(doc["flow"])
    doc["flow"]["allow"]["checks"] = {"remove": ["nothing"]}
    assert [(f.code, f.path) for f in conformance.validate_doc(doc)] == [
        ("E213", f"{ALLOW}['pieces']['scout']['widths']"),
        ("E213", f"{ALLOW}['merges'][1]['id']"),
    ]


def test_flow_rules_are_not_read_in_a_run_document():
    doc = example("implement_review")
    doc["ocp"] = "0.4"
    flow = flow_doc()["flow"]
    flow["judges"][1]["id"] = "j-claude"
    doc["flow"] = flow
    assert "E214" not in codes(conformance.validate_doc(doc))


# E215: numbers a float can hold -------------------------------------------------------------------------

HUGE = 10**400  # an integer JSON can write and no float can hold
RULE = f"{FLOW}['acceptance_rule']"


def _target(flow, **fields):
    flow["acceptance_rule"]["score"].update(fields)


def _window(flow, days):
    flow["acceptance_rule"]["window_days"] = days


@pytest.mark.parametrize("edit, member, message", [
    pytest.param(lambda f: f["objective"].update(margin=HUGE), "margin",
                 f"objective 'cheapest_within_margin' has margin {HUGE!r}; it is a number above 0 that a float "
                 "can hold", id="margin"),
    pytest.param(lambda f: _objective(f, kind="best_under_cost", score="quality", cost_cap_usd=HUGE), "cost_cap_usd",
                 f"objective 'best_under_cost' has cost_cap_usd {HUGE!r}; it is a number above 0 that a float can "
                 "hold", id="cost cap"),
    pytest.param(lambda f: _objective(f, kind="best_under_cost", score="quality", cost_cap_usd=-HUGE),
                 "cost_cap_usd", f"objective 'best_under_cost' has cost_cap_usd {-HUGE!r}; it is a number above 0 "
                 "that a float can hold", id="negative cost cap"),
    pytest.param(lambda f: f["objective"].update(cost_cap_usd=HUGE), "cost_cap_usd",
                 f"objective 'cheapest_within_margin' has cost_cap_usd {HUGE!r}; it is a number a float can hold",
                 id="a member the kind does not need"),
    pytest.param(lambda f: _objective(f, kind=3, margin=HUGE), "margin",
                 f"the objective has margin {HUGE!r}; it is a number a float can hold", id="a kind that is no string"),
])
def test_an_objective_number_no_float_holds_is_refused_once(edit, member, message):
    found = [(f.path, f.message) for f in conformance.validate_doc(_edited(edit)) if f.code == "E215"]
    assert found == [(f"{FLOW}['objective']['{member}']", message)]


@pytest.mark.parametrize("edit, member, message", [
    pytest.param(lambda f: _target(f, target=HUGE, scale="linear"), "['score']['target']",
                 f"the rule's score target {HUGE!r} is not a number a float can hold", id="target"),
    pytest.param(lambda f: _target(f, target=HUGE), "['score']['target']",
                 f"the rule's score target {HUGE!r} is not a number a float can hold", id="fraction target"),
    pytest.param(lambda f: _target(f, target=1.5), "['score']['target']",
                 "the rule's score has scale 'fraction' but target 1.5 is outside [0, 1]", id="fraction 1.5"),
    pytest.param(lambda f: _target(f, target=-0.1), "['score']['target']",
                 "the rule's score has scale 'fraction' but target -0.1 is outside [0, 1]", id="fraction -0.1"),
    pytest.param(lambda f: _window(f, 1000000000), "['window_days']",
                 "the rule's window_days 1000000000 is above 999999999, the most days a window can be", id="window"),
    pytest.param(lambda f: _window(f, 1000000000.0), "['window_days']",
                 "the rule's window_days 1000000000.0 is above 999999999, the most days a window can be",
                 id="window as a whole float"),
    pytest.param(lambda f: _window(f, HUGE), "['window_days']",
                 f"the rule's window_days {HUGE!r} is above 999999999, the most days a window can be",
                 id="huge window"),
])
def test_a_flow_rule_number_out_of_bounds_is_refused_once(edit, member, message):
    assert [(f.code, f.path, f.message) for f in conformance.validate_doc(_edited(edit))] == [
        ("E215", f"{RULE}{member}", message)]


@pytest.mark.parametrize("edit", [
    pytest.param(lambda f: f["objective"].update(margin=1e308), id="margin 1e308"),
    pytest.param(lambda f: _objective(f, kind="best_under_cost", score="quality", cost_cap_usd=2**1000),
                 id="cost cap 2**1000"),
    pytest.param(lambda f: _target(f, target=1e308, scale="linear"), id="linear target 1e308"),
    pytest.param(lambda f: _target(f, target=1e308, scale="log"), id="log target 1e308"),
    pytest.param(lambda f: _target(f, target=0), id="fraction target 0"),
    pytest.param(lambda f: _target(f, target=1), id="fraction target 1"),
    pytest.param(lambda f: (f["acceptance_rule"]["score"].pop("scale"), _target(f, target=1.5)),
                 id="no scale target 1.5"),
    pytest.param(lambda f: _window(f, 0), id="window 0"),
    pytest.param(lambda f: _window(f, 999999999), id="window at the limit"),
    pytest.param(lambda f: _window(f, 999999999.0), id="window at the limit as a whole float"),
])
def test_flow_numbers_a_float_holds_pass(edit):
    assert conformance.validate_doc(_edited(edit)) == []


@pytest.mark.parametrize("days", [-1, 1.5, 1e9 + 0.5])
def test_a_window_of_another_shape_is_the_schemas_alone(days):
    pytest.importorskip("jsonschema")
    assert codes(conformance.validate_doc(_edited(lambda f: _window(f, days)))) == ["E010"]


@pytest.mark.parametrize("edit, member", [
    pytest.param(lambda f: f["objective"].update(margin=HUGE), r"\['objective'\]\['margin'\]", id="margin"),
    pytest.param(lambda f: _target(f, target=HUGE), r"\['acceptance_rule'\]\['score'\]\['target'\]", id="target"),
])
def test_a_flow_file_with_a_number_no_float_holds_does_not_load(edit, member, tmp_path):
    path = tmp_path / "flow.ocp.json"
    path.write_text(json.dumps(_edited(edit)), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"does not validate: E215 at \$\['flow'\]{member}$"):
        flows.load(path)


# E215: a margin the planner takes ----------------------------------------------------------------------

MARGIN = f"{FLOW}['objective']['margin']"
ASSUMED = " (at a first plan's assumed spreads)"
LIMIT = "check pairs at sd_pair 0.197, above the limit of 1000000"


def _edge(k):
    """The least margin the planner takes at `k` judges and the float below it, by a bisection of its reason that
    stops at two adjacent floats."""
    low, high = 1e-6, 1.0
    assert margin_problem(low, k=k) is not None and margin_problem(high, k=k) is None
    for _ in range(100):
        middle = (low + high) / 2
        if middle in (low, high):
            break
        low, high = (low, middle) if margin_problem(middle, k=k) is None else (middle, high)
    assert math.nextafter(low, math.inf) == high
    return high, low


def _margin_found(margin, edit=None, **objective):
    """The findings of the example with this margin, its flow changed by `edit` and its objective replaced by
    `objective` when given."""
    def change(flow):
        if edit is not None:
            edit(flow)
        if objective:
            _objective(flow, **objective)
        flow["objective"]["margin"] = margin

    return [(f.code, f.path, f.message) for f in conformance.validate_doc(_edited(change))]


@pytest.mark.parametrize("margin, reason", [
    (0, "margin 0: a margin is a number above 0"),
    (0.0, "margin 0.0: a margin is a number above 0"),
    (-0.1, "margin -0.1: a margin is a number above 0"),
    (1e-160, f"margin 1e-160: needs more than 1e308 {LIMIT}"),
    (5e-324, f"margin 5e-324: needs more than 1e308 {LIMIT}"),
])
def test_a_margin_the_planner_refuses_is_refused_in_its_words(margin, reason):
    assert margin_problem(margin, k=2) == reason
    assert _margin_found(margin) == [
        ("E215", MARGIN, f"objective 'cheapest_within_margin': {reason}" + (ASSUMED if margin > 0 else ""))]


def test_the_least_margin_the_planner_takes_reads_and_the_float_below_it_does_not():
    edge, below = _edge(2)
    assert margin_table(margins=(edge,), k=2)[0]["check_pairs"] == 1000000  # the limit is inclusive
    assert _margin_found(edge) == []
    assert _margin_found(below) == [
        ("E215", MARGIN, f"objective 'cheapest_within_margin': margin {below!r}: needs 1000001 {LIMIT}{ASSUMED}")]


def test_a_margin_that_needs_one_pair_reads():
    assert margin_table(margins=(1e300,), k=2)[0]["check_pairs"] == 1
    assert _margin_found(1e300) == []


@pytest.mark.parametrize("edit, k", [
    pytest.param(None, 2, id="two model judges and a person"),
    pytest.param(lambda f: f["judges"].append({**f["judges"][2], "id": "person-2"}), 2, id="a second person"),
    pytest.param(lambda f: f["judges"].pop(1), 1, id="one model judge"),
    pytest.param(lambda f: f["judges"].__delitem__(slice(0, 2)), 1, id="the person alone"),
    pytest.param(lambda f: f.pop("judges"), 1, id="no judges"),
])
def test_the_margin_is_read_at_the_judges_that_are_not_person_judges(edit, k):
    margin = 5.2e-4  # taken at two such judges, refused at one
    assert _edge(2)[0] < margin < _edge(1)[0]
    assert _margin_found(margin, edit) == ([] if k == 2 else [
        ("E215", MARGIN, "objective 'cheapest_within_margin': margin 0.00052: needs 1115787 check pairs at sd_pair "
         f"0.2209, above the limit of 1000000{ASSUMED}")])


def test_a_command_judge_is_one_of_the_judges_the_margin_is_read_at():
    margin = 5.2e-4  # taken at two judges that are not person judges, refused at one
    assert _edge(2)[0] < margin < _edge(1)[0]
    assert _margin_found(margin, lambda f: f["judges"].__setitem__(1, {"id": "j-tests", "kind": "command"})) == []


KINDS = {
    "cheapest_within_margin": {"kind": "cheapest_within_margin", "score": "quality"},
    "cost_per_accepted": {"kind": "cost_per_accepted"},
    "best_under_cost": {"kind": "best_under_cost", "score": "quality", "cost_cap_usd": 10},
    "an open kind": {"kind": "a_kind_of_its_own"},
}


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("margin, reason", [
    (0, "margin 0: a margin is a number above 0"),
    (-0.1, "margin -0.1: a margin is a number above 0"),
    (1e-160, f"margin 1e-160: needs more than 1e308 {LIMIT}{ASSUMED}"),
])
def test_a_margin_of_any_kind_is_the_planners(kind, margin, reason):
    objective = KINDS[kind]
    assert _margin_found(margin, **objective) == [("E215", MARGIN, f"objective {objective['kind']!r}: {reason}")]
    assert _margin_found(0.1, **objective) == []


@pytest.mark.parametrize("kind", ["cost_per_accepted", "best_under_cost", "an open kind"])
def test_a_kind_that_needs_no_margin_reads_without_one(kind):
    assert conformance.validate_doc(_edited(lambda f: _objective(f, **KINDS[kind]))) == []


@pytest.mark.parametrize("margin", [HUGE, -HUGE], ids=["10**400", "-10**400"])
def test_a_margin_no_float_holds_is_refused_once_in_the_float_bound_words(margin):
    """The planner refuses it too, in its own words; validation reads the float bound first, so the one finding is
    the bound's."""
    why = margin_problem(margin, k=2)
    assert why == f"margin {margin!r}: a margin is a number above 0"
    message = f"objective 'cheapest_within_margin' has margin {margin!r}; it is a number above 0 that a float can hold"
    assert _margin_found(margin) == [("E215", MARGIN, message)]
    assert message != f"objective 'cheapest_within_margin': {why}"


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _refused_margin_file(tmp_path):
    path = tmp_path / "flow.ocp.json"
    path.write_text(json.dumps(_edited(lambda f: f["objective"].update(margin=1e-160))), encoding="utf-8")
    return path


COLD = {
    "the checker first": "from loopmath.ocp import conformance\nassert 'loopmath.flows.plan' not in sys.modules\n",
    "the planner first": "import loopmath.flows.plan\nfrom loopmath.ocp import conformance\n",
}


@pytest.mark.parametrize("order", list(COLD))
def test_validation_reads_the_planner_in_either_import_order(order, tmp_path):
    """In a fresh interpreter: the checker loads without the planner, and either order validates."""
    code = ("import json, sys\n" + COLD[order]
            + "found = [[f.code, f.path] for f in conformance.validate_file(sys.argv[1])]\n"
            + "print(json.dumps([found, 'loopmath.flows.plan' in sys.modules]))\n")
    done = subprocess.run([sys.executable, "-c", code, str(_refused_margin_file(tmp_path))], capture_output=True,
                          text=True, env=_env(), cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [[["E215", MARGIN]], True]


def test_the_spec_checker_script_refuses_the_margin(tmp_path):
    done = subprocess.run([sys.executable, str(SPEC / "ocp_conformance.py"), str(_refused_margin_file(tmp_path))],
                          capture_output=True, text=True, env=_env(), cwd=tmp_path)
    assert done.returncode == 1, done.stderr
    assert (f"error E215 {MARGIN}: objective 'cheapest_within_margin': margin 1e-160: needs more than 1e308 "
            f"{LIMIT}{ASSUMED}") in done.stdout


# Digest --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    pytest.param(lambda f: f.update(title="Another title"), id="title"),
    pytest.param(lambda f: f.update(version=2), id="version"),
    pytest.param(lambda f: f.update(id="lit-brief-2"), id="id"),
    pytest.param(lambda f: f.update(ext={"dev.loopmath.hypothesis_edits": []}), id="ext"),
    pytest.param(lambda f: f.update(digest="flw_000000000000"), id="digest"),
    pytest.param(lambda f: f["configuration"].update(source="pinned"), id="configuration-source"),
])
def test_the_digest_leaves_out(edit):
    flow = load(FLOW_EXAMPLE)["flow"]
    before = flow_digest(flow)
    edit(flow)
    assert flow_digest(flow) == before


@pytest.mark.parametrize("edit", [
    pytest.param(lambda f: f["objective"].update(margin=0.05), id="objective"),
    pytest.param(lambda f: f["judges"][2].update(rate=0.25), id="judge"),
    pytest.param(lambda f: f["inputs"].update(n=21), id="inputs"),
    pytest.param(lambda f: f.update(hypothesis="the critic matters"), id="hypothesis"),
    pytest.param(lambda f: f["allow"].update(budgets=[1, 2]), id="allow"),
    pytest.param(lambda f: f["acceptance_rule"].update(requires=[]), id="acceptance-rule"),
    pytest.param(lambda f: f["configuration"].update(id="cfg_000000000000"), id="configuration-id"),
])
def test_the_digest_covers(edit):
    flow = load(FLOW_EXAMPLE)["flow"]
    before = flow_digest(flow)
    edit(flow)
    assert flow_digest(flow) != before


def test_the_digest_hashes_members_as_written_with_null_for_an_absent_one():
    flow = load(FLOW_EXAMPLE)["flow"]
    _drop(flow, "hypothesis", "judges", "allow")
    body = {
        "acceptance_rule": flow["acceptance_rule"],
        "allow": None,
        "configuration": "cfg_9bc4f6e876f9",
        "hypothesis": None,
        "inputs": flow["inputs"],
        "judges": None,
        "objective": flow["objective"],
    }
    assert sorted(body) == sorted(FLOW_DIGEST_MEMBERS)
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert flow_digest(flow) == "flw_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def test_a_stale_digest_gives_e212():
    doc = flow_doc(digest=True)
    doc["flow"]["objective"]["margin"] = 0.05
    assert [(f.code, f.path) for f in conformance.validate_doc(doc)] == [("E212", f"{FLOW}['digest']")]
