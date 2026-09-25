"""Rules E190 to E195, W196 and W182: goldens, edge cases and version gating (spec 01 section 3)."""

from __future__ import annotations

import copy
import json

import pytest

from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id
from loopmath.ocp.emit import ARTIFACT_KINDS, workflow_to_ocp
from loopmath.workflows.format import catalog
from loopmath.workflows.ocp import RECOMMENDED_KINDS

from ._common import EXAMPLES, SHAPES, V03_GOLDEN, codes, example, load

RULES = ("E190", "E191", "E192", "E193", "E194", "E195", "W196", "W182")
EXPECTED = json.loads((V03_GOLDEN / "expected.json").read_text())


def check(doc, resolve=None):
    return codes(conformance.validate_doc(doc, resolve=resolve))


def rehash(doc):
    c = doc["run"]["configuration"]
    c["id"] = config_id(c["workflow"], c["settings"])
    return doc


def test_every_rule_has_a_pass_and_a_fail_golden():
    names = {p.name for p in V03_GOLDEN.glob("*.ocp.json")}
    assert names == {f"{rule}-{kind}.ocp.json" for rule in RULES for kind in ("pass", "fail")}
    assert set(EXPECTED) == names


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_golden_gives_exactly_its_expected_codes(name):
    findings = conformance.validate_file(V03_GOLDEN / name)
    assert codes(findings) == EXPECTED[name]
    rule, kind = name.removesuffix(".ocp.json").split("-")
    assert EXPECTED[name] == ([] if kind == "pass" else [rule])
    exit_code = conformance.main([str(V03_GOLDEN / name)])
    assert exit_code == (1 if kind == "fail" and rule.startswith("E") else 0)


# E190 ---------------------------------------------------------------------------------------------

def test_e190_workflow_ref_is_checked_through_the_resolver_and_skipped_without_one():
    doc = example("implement_review")
    workflow = doc["run"]["configuration"]["workflow"]
    doc["run"]["configuration"]["workflow"] = {"ref": "implement_review", "version": 1}
    assert check(doc) == []  # no catalog: E190 and E191 cannot be judged, so they are skipped
    catalog = {("implement_review", 1): workflow}
    resolve = lambda ref, version=None: copy.deepcopy(catalog.get((ref, version)))  # noqa: E731
    assert check(doc, resolve) == []
    doc["run"]["configuration"]["id"] = "cfg_000000000000"
    assert check(doc, resolve) == ["E190"]
    del doc["run"]["configuration"]["settings"]["review"]
    assert "E191" in check(doc, resolve)


@pytest.mark.parametrize("shape", SHAPES)
def test_e190_and_e191_resolve_a_workflow_ref_through_the_catalog_by_default(shape):
    doc = example(shape)
    doc["run"]["configuration"]["workflow"] = {"ref": shape, "version": 1}
    assert codes(conformance.validate_doc(doc)) == []  # same id as the inline catalog shape
    doc["run"]["configuration"]["id"] = "cfg_000000000000"
    assert codes(conformance.validate_doc(doc)) == ["E190"]
    doc["run"]["configuration"]["workflow"]["version"] = 99
    assert codes(conformance.validate_doc(doc)) == []  # the catalog has no version 99: skipped
    doc["run"]["configuration"]["workflow"]["version"] = 1
    doc["run"]["configuration"]["settings"].popitem()
    assert "E191" in codes(conformance.validate_doc(doc))


# E191 ---------------------------------------------------------------------------------------------

def test_e191_nested_workflow_piece_needs_no_setting_of_its_own():
    doc = example("solo")
    workflow = doc["run"]["configuration"]["workflow"]
    inner = {"pieces": [{"id": "fix", "role": "implementer"}], "artifacts": [{"id": "fix_patch", "kind": "diff"}],
             "edges": [["fix", "fix_patch"]], "control": {"gates": [], "repair": {}, "budget": 1}}
    workflow["pieces"].append({"id": "inner", "workflow": inner, "width": 1})
    workflow["artifacts"].append({"id": "report", "kind": "report"})
    workflow["edges"] += [["diff", "inner"], ["inner", "report"]]
    assert check(rehash(doc)) == []


def test_e191_setting_must_be_an_object():
    doc = example("solo")
    doc["run"]["configuration"]["settings"]["implement"] = "claude-opus-5-5"
    assert "E191" in check(doc)


# E192 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    lambda w: w["edges"].append(["implement", "review"]),          # piece to piece
    lambda w: w["edges"].append(["issue", "diff"]),                # artifact to artifact
    lambda w: w["edges"].append(["implement", "nowhere"]),         # unknown vertex
    lambda w: w["pieces"].append({"id": "diff", "role": "worker"}),  # id used twice
    lambda w: w["edges"].append(["verdict", "implement"]),         # the old repair back edge
], ids=["piece-piece", "artifact-artifact", "unknown", "duplicate", "back-edge"])
def test_e192_cases(edit):
    doc = example("implement_review")
    edit(doc["run"]["configuration"]["workflow"])
    settings = doc["run"]["configuration"]["settings"]
    settings.setdefault("diff", copy.deepcopy(settings["implement"]))
    assert "E192" in check(rehash(doc))


def test_e192_nested_inline_workflow_is_checked_too():
    doc = example("solo")
    workflow = doc["run"]["configuration"]["workflow"]
    inner = {"id": "inner_flow", "pieces": [{"id": "a", "role": "worker"}], "artifacts": [{"id": "x", "kind": "other"}],
             "edges": [["a", "x"], ["x", "a"]], "control": {"gates": [], "repair": {}, "budget": 1}}
    workflow["pieces"].append({"id": "inner", "workflow": inner})
    assert "E192" in check(rehash(doc))


# E193 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("control", [
    {"gates": ["verdict"], "repair": {}},                 # a gate that is an artifact
    {"gates": ["ghost"], "repair": {}},                   # a gate that does not exist
    {"gates": ["review"], "repair": {"implement": "implement"}},  # repair key is not a gate
    {"gates": ["review"], "repair": {"review": "diff"}},  # repair target is an artifact
], ids=["artifact-gate", "unknown-gate", "repair-key", "repair-target"])
def test_e193_cases(control):
    doc = example("implement_review")
    doc["run"]["configuration"]["workflow"]["control"].update(control)
    assert "E193" in check(rehash(doc))


# E194 ---------------------------------------------------------------------------------------------

def test_e194_single_file_cases():
    base = example("full-fields")
    cases = []
    doc = copy.deepcopy(base)
    doc["run"]["slate"]["members"] = ["run_other_a", "run_other_b"]
    cases.append(doc)
    doc = copy.deepcopy(base)
    doc["run"]["slate"]["base_commit"] = "fffffff"
    cases.append(doc)
    doc = copy.deepcopy(base)
    doc["run"]["preferences"][0]["winner"] = "run_not_in_slate"
    cases.append(doc)
    for doc in cases:
        assert check(doc) == ["E194"]
    doc = copy.deepcopy(base)
    doc["run"]["preferences"][0]["winner"] = "tie"
    assert check(doc) == []


def _pair(base_commit_b="a1b2c3d", task_b="tsk_ex_full"):
    a = example("full-fields")
    b = example("implement_review")
    b["run"]["id"] = "run_ex_full_b"
    b["run"]["task"]["id"] = task_b
    b["run"]["task"]["base_commit"] = base_commit_b
    b["run"]["slate"] = copy.deepcopy(a["run"]["slate"])
    b["run"]["slate"]["base_commit"] = base_commit_b
    return a, b


def test_e194_pair_members_share_task_and_commit_across_files():
    a, b = _pair()
    assert [codes(f) for f in conformance.validate_many([a, b])] == [[], []]
    a, b = _pair(base_commit_b="0000000")
    assert [codes(f) for f in conformance.validate_many([a, b])] == [[], ["E194"]]
    a, b = _pair(task_b="tsk_other")
    assert [codes(f) for f in conformance.validate_many([a, b])] == [[], ["E194"]]


def test_validate_files_runs_the_cross_file_check(tmp_path):
    a, b = _pair(base_commit_b="0000000")
    paths = []
    for doc in (a, b):
        path = tmp_path / f"{doc['run']['id']}.ocp.json"
        path.write_text(json.dumps(doc))
        paths.append(str(path))
    assert [codes(f) for f in conformance.validate_files(paths)] == [[], ["E194"]]
    assert conformance.main(paths) == 1


# E195 ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [
    {"kind": "verdict", "name": "tests", "value": None},
    {"kind": "verdict", "name": "tests", "value": "ok"},
    {"kind": "score", "name": "quality", "value": "high"},
    {"kind": "score", "name": "quality", "value": True},
    {"kind": "score", "name": "pass_rate", "value": 1.5, "scale": "fraction"},
    {"kind": "event", "name": "incident", "value": None},
    {"kind": "event", "name": "incident", "value": 3},
], ids=["verdict-null", "verdict-word", "score-string", "score-bool", "fraction-range", "event-null",
        "event-number"])
def test_e195_signal_values(signal):
    doc = example("solo")
    doc["run"]["signals"].append({"id": "sig_x", "observed_at": "2026-09-23T11:00:00-07:00", "tier": "reported",
                                  **signal})
    assert "E195" in check(doc)


def test_e195_declared_unmeasured_score_is_fine():
    doc = example("solo")
    doc["run"]["signals"].append({"id": "sig_x", "kind": "score", "name": "quality", "value": None,
                                  "observed_at": "2026-09-23T11:00:00-07:00", "tier": "asserted"})
    assert check(doc) == []


def test_e195_rule_names_refer_to_the_right_kinds():
    doc = example("solo")
    doc["run"]["acceptance_rule"]["requires"] = ["runtime_s"]  # runtime_s is a score here
    assert check(doc) == ["E195"]
    doc = example("solo")
    doc["run"]["acceptance_rule"]["score"] = {"name": "tests", "target": 1, "better": "higher"}
    assert check(doc) == ["E195"]
    doc = example("solo")
    doc["run"]["acceptance_rule"]["score"] = {"name": "pass_rate", "target": 80, "better": "higher",
                                              "scale": "fraction"}
    assert check(doc) == ["E195"]
    doc = example("solo")
    doc["run"]["acceptance_rule"]["score"] = {"name": "heldout_perf", "target": 2400, "better": "higher"}
    assert check(doc) == []  # a score the run has not reported yet


# W196, W182 and version gating ------------------------------------------------------------------

def test_w196_pre_rename_producer_is_not_loopmath():
    doc = example("full-fields")
    doc["producer"]["name"] = "dagr"
    assert check(doc) == ["W196"]
    doc["producer"]["name"] = "loopmath-graph"
    assert check(doc) == []


def test_w182_only_in_v03_documents():
    v02 = load(EXAMPLES / "swarm-v02.ocp.json")
    assert any("dev.dagr." in key for key in json.dumps(v02).split('"'))
    assert "W182" not in check(v02)
    v03 = example("solo")
    v03["nodes"][0]["ext"] = {"dev.dagr.policy": {}}
    assert check(v03) == ["W182"]


def test_retired_dagr_prefix_names_the_new_namespace_in_v03():
    doc = example("solo")
    doc["run"]["ext"] = {"dagr.note": 1}
    findings = [f for f in conformance.validate_doc(doc) if "dagr.note" in f.path]
    assert findings and all("dev.loopmath." in f.message for f in findings)


def test_v03_rules_do_not_fire_on_v02_documents():
    v02 = load(EXAMPLES / "golden" / "minimal-run.ocp.json")
    run = v02["run"]
    run["receipt"] = {}
    assert "W196" not in check(v02)


def test_v03_commit_and_merge_artifact_kinds():
    """Lane 07's request: `run artifact --kind commit` draws no W200 in v0.3; v0.2 is unchanged."""
    doc = example("full-fields")
    doc["artifacts"][0]["kind"] = {"value": "commit", "tier": "reported"}
    doc["artifacts"][1]["kind"] = {"value": "merge", "tier": "reported"}
    assert check(doc) == []
    doc["artifacts"][1]["kind"]["value"] = "tarball"
    assert check(doc) == ["W200"]
    v02 = load(EXAMPLES / "swarm-v02.ocp.json")
    v02["artifacts"][0]["kind"] = {"value": "commit", "tier": "reported"}
    assert "W200" in check(v02)


CATALOG_KINDS = {a["kind"] for w in catalog().values() for a in workflow_to_ocp(w)["artifacts"]}


@pytest.mark.parametrize("kind", sorted(CATALOG_KINDS | set(RECOMMENDED_KINDS) | set(ARTIFACT_KINDS)))
def test_v03_workflow_artifact_kinds_draw_no_w200(kind):
    """An artifact recorded with the kind its workflow names draws no W200 in v0.3; v0.2 is unchanged."""
    assert {"diff", "issue", "repo", "verdict"} <= CATALOG_KINDS
    doc = example("full-fields")
    doc["artifacts"][0]["kind"] = {"value": kind, "tier": "reported"}
    assert check(doc) == []
    v02 = load(EXAMPLES / "swarm-v02.ocp.json")
    v02["artifacts"][0]["kind"] = {"value": kind, "tier": "reported"}
    was_recommended = kind in conformance.RECOMMENDED_VOCABULARY["artifact.kind.value"]
    assert ("W200" in check(v02)) is not was_recommended


def test_v03_event_and_task_vocabulary():
    doc = example("solo")
    doc["run"]["task"]["type"] = "chore"
    doc["events"].append({"at": "2026-09-23T11:00:00-07:00", "type": "custom_thing"})
    assert check(doc) == ["W200"]
    doc = example("solo")
    for event_type in ("signal_observed", "receipt_written", "run_finished"):
        doc["events"].append({"at": "2026-09-23T11:00:00-07:00", "type": event_type})
    assert check(doc) == []
