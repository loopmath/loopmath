"""Catalog, TOML format and validator (lane 04)."""

from __future__ import annotations

import dataclasses

import pytest

from loopmath.types import Control, Gate, Piece, Setting, Workflow
from loopmath.workflows.format import (WorkflowFormatError, catalog, dump_workflow, load_workflow,
                                       loads_workflow, loads_workflow_file, user_workflows, validate_workflow,
                                       workflow_warnings)
from loopmath.workflows.shapes import CATALOG_IDS, ShapeParams, build_shape, catalog_shapes, nearest_catalog, shape_params


def test_catalog_has_the_six_shapes_in_order_and_equals_the_builder():
    cat = catalog()
    assert tuple(cat) == CATALOG_IDS
    assert cat == catalog_shapes()


@pytest.mark.parametrize("name", CATALOG_IDS)
def test_catalog_shapes_are_valid_acyclic_and_round_trip(name):
    wf = catalog()[name]
    assert validate_workflow(wf) == []
    assert workflow_warnings(wf) == []
    assert loads_workflow(dump_workflow(wf)) == wf
    assert shape_params(wf) is not None


def test_catalog_shapes_have_the_expected_pieces_and_gates():
    cat = catalog()
    roles = {k: [(p.id, p.role, p.width) for p in wf.pieces] for k, wf in cat.items()}
    assert roles["solo"] == [("implement", "implementer", 1)]
    assert roles["best_of_n"] == [("implement", "implementer", 3), ("select", "referee", 1)]
    assert roles["swarm"] == [("plan", "planner", 1), ("work", "worker", 3), ("review", "reviewer", 1)]
    review = cat["implement_review"].control
    assert review.budget_rounds == 3
    assert review.gates == (Gate("g_review", "review", "review_approve", on_fail="implement"),)
    assert cat["best_of_n"].control.gates == (Gate("g_select", "select", "referee_pick"),)
    assert cat["solo"].control == Control()
    for wf in cat.values():
        assert "issue" in wf.artifacts and "repo" in wf.artifacts and "diff" in wf.artifacts


def test_composed_shapes_and_nearest_catalog():
    assert build_shape(ShapeParams(middle="team", width=4)).id == "team"
    assert build_shape(ShapeParams(plan=True, middle="team")).id == "plan_team"
    assert build_shape(ShapeParams(middle="best_of_n", review=True, budget_rounds=2)).id == "best_of_n_review"
    assert nearest_catalog(ShapeParams(middle="team")) == "swarm"  # tie with best_of_n: same middle wins
    assert nearest_catalog(ShapeParams(plan=True, middle="team")) == "swarm"
    assert nearest_catalog(ShapeParams(middle="team", review=True)) == "swarm"
    assert nearest_catalog(ShapeParams(middle="best_of_n", review=True)) == "best_of_n"
    assert nearest_catalog(ShapeParams(plan=True, review=True)) == "plan_implement_review"


def _tdd_text(**overrides) -> str:
    base = {
        "edges": '["issue -> write_tests", "write_tests -> tests", ["tests", "implement"], "implement -> diff", '
                 '"diff -> check", "check -> test_record"]',
        "control": 'budget_rounds = 2\nrescue = "person"\n\n[[control.gates]]\nafter = "check"\non_fail = "implement"',
    }
    base.update(overrides)
    return f"""id = "tdd"
version = 2
title = "Test first"
owner = "me"
edges = {base["edges"]}

[[pieces]]
id = "write_tests"
role = "tester"

[[pieces]]
id = "implement"
role = "implementer"

[[pieces]]
id = "check"
role = "tester"

[[artifacts]]
id = "issue"

[[artifacts]]
id = "tests"
kind = "spec"

[[artifacts]]
id = "diff"

[[artifacts]]
id = "test_record"

[control]
{base["control"]}

[settings.implement]
harness = "claude-code"
model = "claude-opus-5-5"
effort = "high"
"""


def test_user_workflow_file_reads_defaults_and_round_trips():
    f = loads_workflow_file(_tdd_text())
    wf = f.workflow
    assert wf.id == "tdd" and wf.version == 2 and wf.extra["owner"] == "me"
    assert wf.extra["artifact_kinds"] == {"issue": "issue", "tests": "spec", "diff": "diff", "test_record": "test_record"}
    assert ("tests", "implement") in wf.edges
    assert wf.control.gates == (Gate("g_check", "check", "tests_pass", on_fail="implement"),)
    assert wf.control.rescue == "person" and wf.control.budget_rounds == 2
    assert validate_workflow(wf) == []
    assert f.settings == {"implement": Setting("claude-code", "claude-opus-5-5", "high")}
    assert f.configuration() is None  # two pieces have no setting
    again = loads_workflow_file(dump_workflow(wf, f.settings))
    assert again.workflow == wf and again.settings == f.settings


@pytest.mark.parametrize("edges,control,message", [
    ('["issue -> write_tests", "write_tests -> tests", "tests -> implement", "implement -> diff", "diff -> check", '
     '"check -> test_record", "test_record -> write_tests"]', 'budget_rounds = 1', "cycle"),
    ('["issue -> nowhere"]', 'budget_rounds = 1', "is not a piece or an artifact"),
    ('["write_tests -> implement"]', 'budget_rounds = 1', "joins two pieces"),
    ('["issue -> implement", "issue -> implement"]', 'budget_rounds = 1', "appears twice"),
    ('["issue -> implement"]', 'budget_rounds = 1\n\n[[control.gates]]\nafter = "ghost"', "is not a piece"),
    ('["issue -> implement", "implement -> diff", "diff -> check"]',
     'budget_rounds = 2\n\n[[control.gates]]\nafter = "implement"\non_fail = "check"', "is not upstream"),
    ('["issue -> implement"]', 'budget_rounds = 1\nrescue = "pray"', "control.rescue"),
    ('["issue -> implement"]',
     'budget_rounds = 1\n\n[[control.gates]]\nafter = "check"\n\n[[control.gates]]\nafter = "check"\nid = "g2"',
     "has two gates"),
])
def test_validator_names_each_structural_error(edges, control, message):
    wf = loads_workflow(_tdd_text(edges=edges, control=control))
    errors = validate_workflow(wf)
    assert any(message in e for e in errors), errors


@pytest.mark.parametrize("text,message", [
    ("id = ", "not valid TOML"),
    ('id = "x"\nversion = 0\n[[pieces]]\nid = "a"\nrole = "worker"\n', "version"),
    ('id = "x"\n', "at least one"),
    ('id = "x"\n[[pieces]]\nid = "a"\nrole = "worker"\nwidth = 0\n', "width"),
    ('id = "x"\nedges = ["a"]\n[[pieces]]\nid = "a"\nrole = "worker"\n', "from -> to"),
    ('id = "x"\n[[pieces]]\nid = "a"\nrole = "worker"\n[control]\nbudget_rounds = 0\n', "budget_rounds"),
])
def test_unreadable_files_raise_with_every_error(text, message):
    with pytest.raises(WorkflowFormatError) as info:
        loads_workflow(text)
    assert any(message in e for e in info.value.errors), info.value.errors


def test_validator_checks_ids_duplicates_and_collisions():
    wf = Workflow("bad id", 1, "", (Piece("a", "worker"), Piece("a", "worker"), Piece("x", "")), ("x",), ())
    errors = validate_workflow(wf)
    assert any("id 'bad id'" in e for e in errors)
    assert any("piece id 'a' appears twice" in e for e in errors)
    assert any("'x' is both a piece and an artifact" in e for e in errors)
    assert any("piece 'x' has no role" in e for e in errors)


def test_warnings_flag_probably_unmeant_workflows():
    wf = loads_workflow('id = "w"\nedges = ["issue -> a"]\nartifacts = ["issue", "notes"]\n'
                        '[[pieces]]\nid = "a"\nrole = "wizard"\n'
                        '[control]\nbudget_rounds = 1\n[[control.gates]]\nafter = "a"\nrule = "command:ok"\non_fail = "a"\n')
    warnings = workflow_warnings(wf)
    assert any("role 'wizard'" in w for w in warnings)
    assert any("produces no artifact" in w for w in warnings)
    assert any("'notes' is on no edge" in w for w in warnings)
    assert any("budget_rounds is 1" in w for w in warnings)


def test_user_workflows_in_the_store(tmp_path):
    folder = tmp_path / "workflows"
    folder.mkdir()
    (folder / "tdd.toml").write_text(_tdd_text())
    (folder / "broken.toml").write_text("id = ")
    (folder / "solo.toml").write_text(dump_workflow(catalog()["solo"]))
    found = {uw.path.name: uw for uw in user_workflows(tmp_path)}
    assert found["tdd.toml"].errors == [] and found["tdd.toml"].file.workflow.id == "tdd"
    assert found["broken.toml"].file is None and "not valid TOML" in found["broken.toml"].errors[0]
    assert any("catalog shape" in e for e in found["solo.toml"].errors)
    assert user_workflows(tmp_path / "missing") == []
    assert load_workflow(folder / "tdd.toml").id == "tdd"


def test_dump_keeps_piece_and_gate_extras():
    wf = catalog()["implement_review"]
    pieces = (dataclasses.replace(wf.pieces[0], extra={"note": "keep"}),) + wf.pieces[1:]
    gate = dataclasses.replace(wf.control.gates[0], extra={"why": "x"})
    wf2 = dataclasses.replace(wf, pieces=pieces, control=dataclasses.replace(wf.control, gates=(gate,)))
    assert loads_workflow(dump_workflow(wf2)) == wf2


def test_rescue_tables_are_read_validated_and_round_trip():
    wf = loads_workflow('id = "w"\nedges = ["issue -> a", "a -> diff"]\nartifacts = ["issue", "diff"]\n'
                        '[[pieces]]\nid = "a"\nrole = "implementer"\n'
                        '[control]\nrescue = {kind = "person", cost_usd = 40.0}\n')
    assert wf.control.rescue == {"kind": "person", "cost_usd": 40.0}
    assert validate_workflow(wf) == [] and workflow_warnings(wf) == []
    assert loads_workflow(dump_workflow(wf)) == wf
    odd = dataclasses.replace(wf, control=dataclasses.replace(wf.control, rescue={"kind": "workflow", "workflow": "solo"}))
    assert validate_workflow(odd) == []
    assert any("rescue kind 'workflow'" in w for w in workflow_warnings(odd))
    kindless = dataclasses.replace(wf, control=dataclasses.replace(wf.control, rescue={"ref": "usual"}))
    assert any("needs a kind" in e for e in validate_workflow(kindless))


def test_role_warning_names_the_word_estimates_use():
    # `coder` folds onto implementer (normalize_role, which lane 05's canonical_role calls); only unknown words stay
    wf = catalog()["implement_review"]
    renamed = dataclasses.replace(wf, pieces=(dataclasses.replace(wf.pieces[0], role="coder"),
                                              dataclasses.replace(wf.pieces[1], role="auditor")))
    assert workflow_warnings(renamed) == [
        "piece 'implement': role 'coder' is read as 'implementer' by estimates",
        "piece 'review': role 'auditor' is not one of planner, implementer, reviewer, tester, referee, worker, "
        "integrator; allowed, but estimates group pieces by role name"]
