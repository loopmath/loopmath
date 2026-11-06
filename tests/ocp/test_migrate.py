"""Migration to v0.3 (spec 01 section 4): goldens, idempotence, derivations, contract v3."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from loopmath.cli import main
from loopmath.ocp import conformance, contractv3
from loopmath.ocp.migrate import INFERRED_KEY, MIGRATION_KEY, MigrationError, migrate_doc

from ._common import EXAMPLES, MIGRATE_GOLDEN, codes, example, load


def _sweep_dir():
    """LOOPMATH_SWEEP_DIR, as for the research verbs: the sweep run records (`*.run.json`), or the
    results folder above them (its dagr/ child). None when unset. Read only."""
    value = os.environ.get("LOOPMATH_SWEEP_DIR")
    if not value:
        return None
    path = Path(value).expanduser()
    if not any(path.glob("*.run.json")) and (path / "dagr").is_dir():
        path = path / "dagr"
    return path


SWEEP = _sweep_dir()
INPUTS = sorted(p for p in MIGRATE_GOLDEN.glob("*.json") if ".expected." not in p.name)


def expected_path(path):
    return MIGRATE_GOLDEN / f"{path.name.split('.')[0]}.expected.ocp.json"


def test_every_input_has_a_golden():
    assert {p.name for p in INPUTS} == {"contract-v3.run.json", "labeled-v02.ocp.json", "minimal-v01.ocp.json",
                                        "swarm-v02.ocp.json"}
    for path in INPUTS:
        assert expected_path(path).is_file(), path


@pytest.mark.parametrize("path", INPUTS, ids=lambda p: p.name)
def test_migration_matches_its_golden_and_is_valid(path):
    doc = load(path)
    before = copy.deepcopy(doc)
    out = migrate_doc(doc, infer=None)
    assert doc == before  # the input is never mutated
    assert out == load(expected_path(path))
    assert out["ocp"] == "0.3"
    assert [f for f in conformance.validate_doc(out) if f.level == "error"] == []


@pytest.mark.parametrize("path", INPUTS, ids=lambda p: p.name)
def test_migration_is_idempotent(path):
    once = migrate_doc(load(path), infer=None)
    assert migrate_doc(once, infer=None) == once


@pytest.mark.parametrize("name", ["minimal.ocp.json", "review-loop-v01.ocp.json", "swarm-v02.ocp.json"])
def test_every_shipped_older_example_migrates_cleanly(name):
    out = migrate_doc(load(EXAMPLES / name))
    assert [f for f in conformance.validate_doc(out) if f.level == "error"] == []


def test_shipped_migration_example_is_the_migrate_output():
    """spec/examples/review-loop-v03.ocp.json is `loopmath ocp migrate` of review-loop-v01.ocp.json."""
    out = migrate_doc(load(EXAMPLES / "review-loop-v01.ocp.json"))
    assert out == load(EXAMPLES / "review-loop-v03.ocp.json")
    # No errors; W180 only, since a 0.1 producer declares no capabilities.
    assert [f.code for f in conformance.validate_doc(out)] == ["W180"]


def test_v03_documents_pass_through_unchanged():
    doc = example("full-fields")
    assert migrate_doc(doc) == doc


def test_derived_fields():
    out = load(expected_path(MIGRATE_GOLDEN / "labeled-v02.ocp.json"))
    run = out["run"]
    assert run["task"] == {"id": "tsk_widget_since", "type": "feature", "repo": "example/widget",
                           "labeled_by": {"how": "inferred", "tier": "heuristic"}}
    assert run["labels"]["team"] == "tools"  # labels are kept
    assert run["provenance"] == {"kind": "logged", "chooser": "habit"}
    assert run["configuration"] == {"source": "habit"}
    assert run["ext"][MIGRATION_KEY] == {"from": "0.2"}
    vertex = {n["id"]: n.get("vertex") for n in out["nodes"]}
    assert vertex == {"impl": "dev", "rev": "reviewer", "mixed": None}  # attempts disagree: no vertex


def test_task_id_falls_back_to_the_run_id():
    doc = load(MIGRATE_GOLDEN / "labeled-v02.ocp.json")
    del doc["run"]["labels"]["task"]
    assert migrate_doc(doc, infer=None)["run"]["task"]["id"] == "golden-labeled-run"
    del doc["run"]["labels"]
    assert "task" not in migrate_doc(doc, infer=None)["run"]


def test_v01_edges_get_the_reported_tier():
    out = load(expected_path(MIGRATE_GOLDEN / "minimal-v01.ocp.json"))
    assert out["edges"] and all(e["tier"] == "reported" for e in out["edges"])


def test_contract_v3_goes_through_the_contract_converter():
    out = load(expected_path(MIGRATE_GOLDEN / "contract-v3.run.json"))
    assert out["producer"]["source_contract"] == "dagr/3"
    assert out["run"]["ext"][MIGRATION_KEY] == {"from": "dagr/3"}
    assert codes(conformance.validate_doc(out)) == ["W180", "W182"]  # contract ext keys stay dev.dagr. (readable)


def test_contract_converter_reads_the_dagr_version_key():
    """The rename bug of spec 01 section 3: the contract's version key is 'dagr'."""
    doc = load(MIGRATE_GOLDEN / "contract-v3.run.json")
    doc["dagr"] = 2
    assert contractv3.convert(doc)["producer"]["source_contract"] == "dagr/2"
    del doc["dagr"]
    assert contractv3.convert(doc)["producer"]["source_contract"] == "dagr/1"
    with pytest.raises(ValueError):
        contractv3.convert({"ocp": "0.2"})


def test_inferred_configuration_is_marked_heuristic():
    configuration = copy.deepcopy(example("solo")["run"]["configuration"])
    doc = load(MIGRATE_GOLDEN / "labeled-v02.ocp.json")
    out = migrate_doc(doc, infer=lambda d: (configuration, 0.62))
    got = out["run"]["configuration"]
    assert got["source"] == "habit" and got["id"] == configuration["id"]
    assert got["ext"][INFERRED_KEY] == {"confidence": 0.62, "tier": "heuristic"}
    assert [f for f in conformance.validate_doc(out) if f.level == "error"] == []


def test_inference_reads_the_document_and_the_original_contract_file(monkeypatch):
    """Infer gets the migrated document, or a contract run file before convert (with ext.experiment)."""
    seen = []
    migrate_doc(load(MIGRATE_GOLDEN / "labeled-v02.ocp.json"), infer=lambda d: seen.append(d))
    contract = load(MIGRATE_GOLDEN / "contract-v3.run.json")
    contract.setdefault("ext", {})["experiment"] = {"sweep": "s"}
    migrate_doc(contract, infer=lambda d: seen.append(d))
    assert seen[0]["ocp"] == "0.3" and seen[1] == contract

    from loopmath.workflows import infer as infer_module

    given, real = [], infer_module.infer_workflow
    monkeypatch.setattr(infer_module, "infer_workflow", lambda d: given.append(d) or real(d))
    migrate_doc(load(MIGRATE_GOLDEN / "minimal-v01.ocp.json"))
    assert isinstance(given[0], dict) and given[0]["ocp"] == "0.3"


@pytest.mark.skipif(SWEEP is None or not SWEEP.is_dir(), reason="LOOPMATH_SWEEP_DIR is not set to the sweep run records")
def test_first_sweep_run_file_infers_plan_implement_review():
    """D51, the workflows module's measurement: 0.60 solo from from_ocp(doc), 0.98 from the original file."""
    files = sorted(SWEEP.glob("*.run.json"))
    assert files
    first = files[0]
    configuration = migrate_doc(json.loads(first.read_text()))["run"]["configuration"]
    assert configuration["workflow"]["id"] == "plan_implement_review"
    assert configuration["ext"][INFERRED_KEY]["confidence"] >= 0.9


def test_default_inference_is_the_workflow_inference():
    out = migrate_doc(load(MIGRATE_GOLDEN / "minimal-v01.ocp.json"))
    configuration = out["run"]["configuration"]
    assert configuration["workflow"]["id"] == "implement_review" and configuration["source"] == "habit"
    assert configuration["ext"][INFERRED_KEY]["tier"] == "heuristic"
    assert [f for f in conformance.validate_doc(out) if f.level == "error"] == []


@pytest.mark.parametrize("name, edit", [
    ("labeled-v02.ocp.json", {}),                 # no attempt records a model: InferError
    ("minimal-v01.ocp.json", {"attempts": 1.5}),  # malformed: inference raises TypeError
], ids=["unreadable", "malformed"])
def test_default_inference_degrades_to_habit_when_it_cannot_read_the_document(name, edit):
    doc = {**load(MIGRATE_GOLDEN / name), **edit}
    assert migrate_doc(doc)["run"]["configuration"] == {"source": "habit"}


@pytest.mark.parametrize("doc", [{"ocp": "0.4"}, {"ocp": "1.0"}, {"ocp": 0.2}, {"nodes": []}, []],
                         ids=["future", "major", "number", "no-version", "not-object"])
def test_unknown_inputs_are_refused(doc):
    with pytest.raises(MigrationError):
        migrate_doc(doc)


def test_migrated_goldens_are_stable_json():
    for path in INPUTS:
        text = expected_path(path).read_text()
        assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"


@pytest.mark.parametrize("path", INPUTS, ids=lambda p: p.name)
def test_migration_to_v04_is_the_v03_golden_raised(path):
    out = migrate_doc(load(path), infer=None, target="0.4")
    assert out == dict(load(expected_path(path)), ocp="0.4")
    assert migrate_doc(out, infer=None, target="0.4") == out


def test_v03_documents_raise_to_v04_with_the_migration_record():
    doc = example("full-fields")
    expected = dict(copy.deepcopy(doc), ocp="0.4")
    expected["run"].setdefault("ext", {})[MIGRATION_KEY] = {"from": "0.3"}
    assert migrate_doc(doc, target="0.4") == expected


def test_v04_targets_and_inputs_are_refused_by_name():
    with pytest.raises(MigrationError, match="targets: 0.3, 0.4"):
        migrate_doc(example("solo"), target="0.5")
    with pytest.raises(MigrationError, match="down to 0.3; use --to 0.4"):
        migrate_doc(dict(example("solo"), ocp="0.4"))
    raised = dict(example("solo"), ocp="0.4")
    assert migrate_doc(raised, target="0.4") == raised


def _cli(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


@pytest.mark.parametrize("path", [EXAMPLES / "swarm-v02.ocp.json", EXAMPLES / "v0.3" / "full-fields.ocp.json"],
                         ids=lambda p: p.name)
def test_cli_migrate_to_v04_is_valid_and_idempotent(capsys, tmp_path, path):
    code, out, err = _cli(capsys, "ocp", "migrate", "--to", "0.4", str(path))
    assert code == 0, err
    migrated = json.loads(out)
    assert migrated["ocp"] == "0.4"
    assert migrated["run"]["ext"][MIGRATION_KEY] == {"from": load(path)["ocp"]}
    assert [f for f in conformance.validate_doc(migrated) if f.level == "error"] == []
    again = tmp_path / "again.ocp.json"
    again.write_text(out)
    code, out, err = _cli(capsys, "ocp", "migrate", "--to", "0.4", str(again))
    assert code == 0, err
    assert json.loads(out) == migrated


def test_cli_migrate_default_stays_v03_and_refuses_a_v04_input(capsys, tmp_path):
    code, out, _ = _cli(capsys, "ocp", "migrate", str(EXAMPLES / "swarm-v02.ocp.json"))
    assert code == 0 and json.loads(out)["ocp"] == "0.3"
    newer = tmp_path / "newer.ocp.json"
    newer.write_text(json.dumps(dict(example("solo"), ocp="0.4")))
    code, out, err = _cli(capsys, "ocp", "migrate", str(newer))
    assert code == 1 and out == ""
    assert "--to 0.4" in err


def test_cli_migrate_json_keeps_its_shape_for_v04(capsys):
    path = str(EXAMPLES / "swarm-v02.ocp.json")
    _, out_03, _ = _cli(capsys, "ocp", "migrate", "--json", path)
    code, out_04, _ = _cli(capsys, "ocp", "migrate", "--json", "--to", "0.4", path)
    at_03, at_04 = json.loads(out_03), json.loads(out_04)
    assert code == 0 and at_04["schema"] == at_03["schema"] == "loopmath.ocp.migrate/1"
    assert set(at_04) == set(at_03) and at_04["ok"] is True
    assert [set(row) for row in at_04["files"]] == [set(row) for row in at_03["files"]]
    assert at_04["files"][0]["doc"]["ocp"] == "0.4" and at_04["files"][0]["from"] == "0.2"
