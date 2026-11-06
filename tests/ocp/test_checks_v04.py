"""OCP 0.4 `control.checks`: several checks after one piece, the id rule and E217."""

from __future__ import annotations

import copy
import json

import pytest

from loopmath.ocp import canonical, conformance, emit
from loopmath.ocp.canonical import GATE_RULES_KEY, canonical_workflow, config_id

from ._common import EXAMPLES, ROOT, V03_EXAMPLES, codes, example, load

V04_EXAMPLES = EXAMPLES / "v0.4"
V04_GOLDEN = EXAMPLES / "golden" / "v0.4"
EXPECTED = json.loads((V04_GOLDEN / "expected.json").read_text())
CATALOG = ROOT / "src" / "loopmath" / "workflows" / "catalog"
CONTROL = "$['run']['configuration']['workflow']['control']"

# Each catalog workflow's id with empty settings, computed before control.checks existed
CATALOG_IDS = {
    "best_of_n": "cfg_ff6aa63161a0",
    "implement_review": "cfg_5d855fb7fd9e",
    "plan_implement": "cfg_fe0d95846906",
    "plan_implement_review": "cfg_f707697a2ae1",
    "solo": "cfg_09a7b08e6a01",
    "swarm": "cfg_f5e0c2804274",
}
ROLE_DEFAULT = {"reviewer": "review_approve", "referee": "referee_pick", "tester": "tests_pass"}

# A stored configuration id: run.configuration.id, or flow.configuration.id in a flow document
ID_FILES = [p for p in sorted(EXAMPLES.rglob("*.ocp.json"))
            if any(isinstance(((load(p).get(part) or {}).get("configuration") or {}).get("id"), str)
                   for part in ("run", "flow"))]


def _expected_codes(path):
    expected = path.parent / "expected.json"
    return set(json.loads(expected.read_text())[path.name]) if expected.is_file() else set()


def _id(doc):
    configuration = doc["run"]["configuration"]
    return config_id(configuration["workflow"], configuration["settings"], resolve=conformance.catalog_resolver)


def _rehash(doc):
    doc["run"]["configuration"]["id"] = _id(doc)
    return doc


def _twin(workflow, ext=None):
    """The checks form of a gates-form workflow: one check per gate, then no gates, repair or gate rules."""
    twin = copy.deepcopy(workflow)
    for piece in twin.get("pieces", []):
        if isinstance(piece.get("workflow"), dict) and "pieces" in piece["workflow"]:
            piece["workflow"] = _twin(piece["workflow"], ext)
    control = twin.setdefault("control", {})
    roles = {p["id"]: p.get("role") for p in twin.get("pieces", [])}
    rules = (control.get("ext") or {}).get(GATE_RULES_KEY) or {}
    repair = control.get("repair") or {}
    checks = []
    for piece in control.pop("gates", []):
        check = {"id": f"c-{piece}", "after": piece,
                 "rule": rules.get(piece) or ROLE_DEFAULT.get(roles.get(piece), "review_approve"),
                 "on_fail": repair.get(piece)}
        if ext is not None:
            check["id"] = f"other-{piece}"
            check["ext"] = ext
        checks.append(check)
    control.pop("repair", None)
    if "ext" in control:
        control["ext"].pop(GATE_RULES_KEY, None)
        if not control["ext"]:
            del control["ext"]
    control["checks"] = checks
    return twin


def _gated_workflows():
    """Every shipped workflow with gates: the v0.3 examples and the catalog."""
    out = []
    for path in sorted(V03_EXAMPLES.glob("*.ocp.json")):
        configuration = load(path)["run"]["configuration"]
        if configuration["workflow"].get("control", {}).get("gates"):
            out.append(pytest.param(configuration["workflow"], configuration["settings"], id=path.name))
    for name in sorted(CATALOG_IDS):
        workflow = conformance.catalog_resolver(name)
        if workflow["control"].get("gates"):
            out.append(pytest.param(workflow, {}, id=f"catalog/{name}"))
    return out


def _synthetic_gated_workflows():
    """Gates the shipped files lack: a non-default rule, a tester, any other role, and a nested workflow."""
    base = example("implement_review")["run"]["configuration"]["workflow"]
    rule = copy.deepcopy(base)
    rule["control"]["ext"] = {GATE_RULES_KEY: {"review": "command:lint"}, "example.other": 1}
    tester = copy.deepcopy(base)
    tester["pieces"][1]["role"] = "tester"
    other = copy.deepcopy(base)
    other["pieces"][1]["role"] = "docs"
    other["control"]["ext"] = {GATE_RULES_KEY: {"review": "tests_pass"}}
    nested = copy.deepcopy(base)
    nested["pieces"].append({"id": "inner", "workflow": copy.deepcopy(rule)})
    return [pytest.param(w, id=name) for name, w in
            (("gate-rule", rule), ("tester", tester), ("other-role", other), ("nested", nested))]


# Existing ids --------------------------------------------------------------------------------------

def test_the_id_sets_cover_the_new_files():
    names = {str(p.relative_to(EXAMPLES)) for p in ID_FILES}
    assert {"v0.4/two-checks.ocp.json", "golden/v0.4/E217-pass.ocp.json", "golden/v0.4/E217-fail.ocp.json"} <= names
    assert len(ID_FILES) >= 27  # 24 earlier files with an id, and the three above
    assert {"v0.4/lit-brief.run-1.ocp.json", "v0.4/lit-brief.run-2.ocp.json"} | {
        f"golden/v0.4/{rule}-{kind}.ocp.json" for rule in ("E171", "E218", "W219") for kind in ("pass", "fail")
    } <= names
    assert len(_gated_workflows()) == 9  # 5 v0.3 examples, 4 catalog workflows


@pytest.mark.parametrize("path", ID_FILES, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_every_stored_configuration_id_holds_unless_its_golden_expects_e190(path):
    found = "E190" in codes(conformance.validate_file(path))
    assert found == ("E190" in _expected_codes(path))


def test_catalog_workflow_ids_are_unchanged():
    assert sorted(p.stem for p in CATALOG.glob("*.toml")) == sorted(CATALOG_IDS)
    for name, cid in CATALOG_IDS.items():
        assert config_id(conformance.catalog_resolver(name), {}) == cid, name


def test_the_role_defaults_have_one_definition():
    assert emit.DEFAULT_GATE_RULE is canonical.DEFAULT_GATE_RULE
    assert canonical.DEFAULT_GATE_RULE == ROLE_DEFAULT


# Gates form and checks form ------------------------------------------------------------------------

@pytest.mark.parametrize(("workflow", "settings"), _gated_workflows())
def test_a_gates_workflow_and_its_checks_twin_have_one_id(workflow, settings):
    twin = _twin(workflow)
    assert "gates" not in twin["control"] and twin["control"]["checks"]
    assert config_id(twin, settings) == config_id(workflow, settings)
    assert config_id(_twin(workflow, ext={"example.note": "kept out"}), settings) == config_id(workflow, settings)


@pytest.mark.parametrize("workflow", _synthetic_gated_workflows())
def test_twins_keep_the_id_for_rules_roles_and_nested_workflows(workflow):
    assert config_id(_twin(workflow), {}) == config_id(workflow, {})


@pytest.mark.parametrize("name", ["best_of_n", "full-fields", "implement_review", "plan_implement_review", "swarm"])
def test_a_v03_example_with_its_checks_twin_validates_at_v04_with_its_stored_id(name):
    doc = example(name)
    doc["ocp"] = "0.4"
    workflow = doc["run"]["configuration"]["workflow"]
    doc["run"]["configuration"]["workflow"] = _twin(workflow)
    assert conformance.validate_doc(doc) == []


def test_a_gates_workflow_naming_its_default_rule_keeps_its_0_3_id_and_its_twin_drops_the_entry():
    """gate_rules usually names only rules other than the default; an explicit default is hashed as written."""
    configuration = example("implement_review")["run"]["configuration"]
    workflow, settings = configuration["workflow"], configuration["settings"]
    explicit = copy.deepcopy(workflow)
    explicit["control"]["ext"] = {GATE_RULES_KEY: {"review": "review_approve"}}
    twin = _twin(explicit)
    assert [c["rule"] for c in twin["control"]["checks"]] == ["review_approve"]
    assert config_id(explicit, settings) == "cfg_6bb7a39f1532"
    assert config_id(twin, settings) == "cfg_ab15440a2d9c"
    assert config_id(twin, settings) == config_id(workflow, settings)


def test_gates_next_to_checks_do_not_enter_the_id():
    doc = load(V04_EXAMPLES / "two-checks.ocp.json")
    control = doc["run"]["configuration"]["workflow"]["control"]
    before = _id(doc)
    control.update({"gates": ["implement"], "repair": {"implement": "review"},
                    "ext": {GATE_RULES_KEY: {"implement": "tests_pass"}}})
    assert _id(doc) == before


def test_an_empty_checks_list_reads_as_no_gates():
    workflow = example("solo")["run"]["configuration"]["workflow"]
    twin = copy.deepcopy(workflow)
    twin["control"] = {"checks": [], "budget": workflow["control"]["budget"],
                       "rescue": workflow["control"]["rescue"]}
    assert config_id(twin, {}) == config_id(workflow, {})


# Two checks after one piece ------------------------------------------------------------------------

def _with_checks(doc, checks):
    out = copy.deepcopy(doc)
    out["run"]["configuration"]["workflow"]["control"]["checks"] = checks
    return out


def test_two_checks_after_one_piece_validate_and_get_their_own_id():
    doc = load(V04_EXAMPLES / "two-checks.ocp.json")
    assert doc["ocp"] == "0.4"
    assert conformance.validate_file(V04_EXAMPLES / "two-checks.ocp.json") == []
    first, second = doc["run"]["configuration"]["workflow"]["control"]["checks"]
    assert first["after"] == second["after"] == "review"
    both, only_first, only_second = _id(doc), _id(_with_checks(doc, [first])), _id(_with_checks(doc, [second]))
    assert len({both, only_first, only_second}) == 3
    # One review_approve check is the gate implement_review states in the 0.3 form
    assert only_first == example("implement_review")["run"]["configuration"]["id"]


def test_check_order_ids_and_ext_leave_the_id_unchanged():
    doc = load(V04_EXAMPLES / "two-checks.ocp.json")
    first, second = doc["run"]["configuration"]["workflow"]["control"]["checks"]
    renamed = [dict(first, id="a", ext={"example.note": 1}), dict(second, id="b")]
    for checks in ([second, first], renamed):
        assert _id(_with_checks(doc, checks)) == _id(doc)


def test_checks_sort_by_after_rule_then_on_fail_with_null_first():
    workflow = load(V04_EXAMPLES / "two-checks.ocp.json")["run"]["configuration"]["workflow"]
    workflow["control"]["checks"] = [
        {"id": "w", "after": "review", "rule": "review_approve", "on_fail": "implement"},
        {"id": "x", "after": "review", "rule": "command:lint", "on_fail": "implement", "ext": {"example.note": 1}},
        {"id": "y", "after": "review", "rule": "command:lint"},
        {"id": "z", "after": "implement", "rule": "tests_pass", "on_fail": None},
    ]
    control = canonical_workflow(workflow)["control"]
    assert control == {
        "budget": 3,
        "checks": [
            {"after": "implement", "on_fail": None, "rule": "tests_pass"},
            {"after": "review", "on_fail": None, "rule": "command:lint"},
            {"after": "review", "on_fail": "implement", "rule": "command:lint"},
            {"after": "review", "on_fail": "implement", "rule": "review_approve"},
        ],
        "rescue": {"kind": "configuration", "ref": "usual"},
    }
    reversed_order = copy.deepcopy(workflow)
    reversed_order["control"]["checks"].reverse()
    assert config_id(reversed_order, {}) == config_id(workflow, {})


# E217 ----------------------------------------------------------------------------------------------

def test_every_v04_golden_is_listed():
    assert {p.name for p in V04_GOLDEN.glob("*.ocp.json")} == set(EXPECTED) == {
        f"{rule}-{kind}.ocp.json"
        for rule in ("E171", "E212", "E213", "E214", "E215", "E216", "E217", "E218", "W219", "W220")
        for kind in ("pass", "fail")}


# Checked together with the flow example, as their codes compare a run with its flow; E218-fail and W219-fail
# pass when checked alone
WITH_FLOW = ("E218", "W219")
FLOW_EXAMPLE = V04_EXAMPLES / "lit-brief.flow.ocp.json"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_v04_golden_gives_exactly_its_expected_codes(name, capsys):
    rule, kind = name.removesuffix(".ocp.json").split("-")
    path = V04_GOLDEN / name
    argv = [FLOW_EXAMPLE, path] if rule in WITH_FLOW else [path]
    *before, found = conformance.validate_files(argv)
    assert before == [[]] * (len(argv) - 1)
    assert codes(found) == EXPECTED[name]
    assert EXPECTED[name] == ([] if kind == "pass" else [rule])
    if rule in WITH_FLOW:
        assert conformance.validate_file(path) == []
    error = any(f.level == "error" for f in found)
    assert error == (kind == "fail" and rule.startswith("E"))
    capsys.readouterr()
    assert conformance.main([str(p) for p in argv]) == (1 if error else 0)
    assert (f" {rule} " in capsys.readouterr().out) == (kind == "fail")


def _passing():
    return load(V04_GOLDEN / "E217-pass.ocp.json")


def _check(**fields):
    return {"id": "review-approves", "after": "review", "rule": "review_approve", "on_fail": "implement", **fields}


def _nested_bad_check(doc):
    workflow = doc["run"]["configuration"]["workflow"]
    inner = copy.deepcopy(workflow)
    inner["control"]["checks"] = [_check(after="nowhere")]
    workflow["pieces"].append({"id": "inner", "workflow": inner})
    return doc


def _mixed(doc, **control):
    doc["run"]["configuration"]["workflow"]["control"].update(control)
    return doc


CASES = {
    "gates": (lambda d: _mixed(d, gates=["review"]), [CONTROL]),
    "empty-repair": (lambda d: _mixed(d, repair={}), [CONTROL]),  # a repair map without gates is also E193
    "gate-rules": (lambda d: _mixed(d, ext={GATE_RULES_KEY: {"review": "command:lint"}}), [CONTROL]),
    "duplicate-id": (lambda d: _with_checks(d, [_check(), _check(rule="command:lint")]),
                     [f"{CONTROL}['checks'][1]['id']"]),
    "unknown-after": (lambda d: _with_checks(d, [_check(after="lint")]), [f"{CONTROL}['checks'][0]['after']"]),
    "unknown-on-fail": (lambda d: _with_checks(d, [_check(on_fail="plan")]), [f"{CONTROL}['checks'][0]['on_fail']"]),
    "same-after-and-rule": (lambda d: _with_checks(d, [_check(), _check(id="again", on_fail=None)]),
                            [f"{CONTROL}['checks'][1]"]),
    "nested": (_nested_bad_check,
               ["$['run']['configuration']['workflow']['pieces'][2]['workflow']['control']['checks'][0]['after']"]),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_e217_reports_each_rule_at_its_member(case):
    build, paths = CASES[case]
    doc = _rehash(build(_passing()))
    findings = conformance.validate_doc(doc)
    assert codes(findings) == ["E217"], findings
    assert [f.path for f in findings] == paths


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_same_workflow_in_a_v03_document_gives_no_e217(case):
    build, _ = CASES[case]
    doc = build(_passing())
    doc["ocp"] = "0.3"
    assert conformance.validate_doc(_rehash(doc)) == []


def test_a_null_or_absent_on_fail_is_fine():
    for check in (_check(on_fail=None), {k: v for k, v in _check().items() if k != "on_fail"}):
        assert conformance.validate_doc(_rehash(_with_checks(_passing(), [check]))) == []


def test_mixing_names_every_0_3_member_present():
    doc = _rehash(_mixed(_passing(), gates=["review"], repair={"review": "implement"},
                         ext={GATE_RULES_KEY: {"review": "command:lint"}}))
    [finding] = conformance.validate_doc(doc)
    assert finding.code == "E217"
    assert "gates, repair, dev.loopmath.gate_rules" in finding.message
