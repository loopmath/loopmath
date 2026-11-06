"""The frozen flow and plan records: field lists, round trips as written, and no move in any coding path."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from loopmath.flows.records import (
    PLAN_SCHEMA,
    Allow,
    AllowChecks,
    Arm,
    CheckResult,
    Edit,
    Excluded,
    Experiment,
    Flow,
    FlowInputs,
    Judge,
    Merge,
    Objective,
    PieceAllow,
    Plan,
    RunSpec,
    flow_document,
    judge_rate,
    judge_scale,
    judge_score,
    reference,
)
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.types import Check, Configuration, Control, FlowRef
from loopmath.workflows.format import catalog
from loopmath.workflows.ocp import configuration_from_ocp, configuration_to_ocp

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "spec" / "examples"
SPEC_FLOW = EXAMPLES / "v0.4" / "lit-brief.flow.ocp.json"
FIX = ROOT / "tests" / "fixtures"
BASE = json.loads((FIX / "flows" / "to_dict.base.json").read_text(encoding="utf-8"))
PLAN_SAMPLE = FIX / "flows" / "plan.sample.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _flow_goldens(kind: str) -> list[Path]:
    return [p for p in sorted((EXAMPLES / "golden" / "v0.4").glob(f"*-{kind}.ocp.json"))
            if _read(p).get("kind") == "flow"]


# 1. Field lists

FIELDS = {
    Check: ["id", "after", "rule", "on_fail", "extra", "given"],
    Control: ["gates", "budget_rounds", "rescue", "extra", "checks"],
    FlowRef: ["id", "version", "digest", "arm", "plan", "split", "extra", "given"],
    FlowInputs: ["ref", "type", "ids", "holdout", "n", "extra", "given"],
    Objective: ["kind", "score", "margin", "reference", "cost_cap_usd", "extra", "given"],
    Judge: ["id", "kind", "model", "rubric", "score", "scale", "range", "rate", "blinded", "extra", "given"],
    PieceAllow: ["models", "efforts", "options", "widths", "drop", "select", "extra", "given"],
    Merge: ["id", "replace", "into", "setting", "extra", "given"],
    AllowChecks: ["add", "remove", "extra", "given"],
    Allow: ["pieces", "budgets", "checks", "merges", "extra", "given"],
    Flow: ["id", "version", "configuration", "inputs", "acceptance_rule", "objective", "title", "digest",
           "hypothesis", "judges", "allow", "ext", "extra", "given"],
    Edit: ["kind", "piece", "key", "value", "extra", "given"],
    Arm: ["id", "configuration", "edits", "extra", "given"],
    Experiment: ["id", "kind", "arms", "reference", "split", "design", "pair_cap", "state", "verdict", "extra",
                 "given"],
    Excluded: ["edit", "reason", "extra", "given"],
    Plan: ["schema", "id", "flow", "seed", "margin", "arms", "experiments", "queued", "excluded", "budget_usd",
           "extra", "given"],
    RunSpec: ["run_id", "input", "arm", "configuration_id", "settings", "experiment", "split", "slate", "judges",
              "person", "flow", "record", "extra", "given"],
    CheckResult: ["arm", "reference", "pairs", "p_quality", "p_cost", "threshold", "margin", "pair_cap", "verdict",
                  "more_pairs", "extra", "given"],
}


@pytest.mark.parametrize("record", list(FIELDS), ids=lambda r: r.__name__)
def test_field_list_is_pinned(record):
    assert [f.name for f in dataclasses.fields(record)] == FIELDS[record]
    given = {f.name: f for f in dataclasses.fields(record)}.get("given")
    if given is not None:
        assert given.default == frozenset() and given.compare is False and given.repr is False


def test_edit_writes_kind_piece_key_value_then_extras():
    edit = Edit(kind="option", piece="synthesize", key="prompt", value="v4", extra={"note": "kept"})
    assert list(edit.to_dict()) == ["kind", "piece", "key", "value", "note"]
    assert list(Edit.from_dict({"value": 3, "kind": "budget"}).to_dict()) == ["kind", "value"]
    drop = {"kind": "drop", "piece": "critic"}
    assert Edit.from_dict(drop).to_dict() == drop
    assert Edit(kind="drop", piece="critic").to_dict() == drop


# 2. No byte moves against the stored step 4 outputs


def _configs(value):
    """The `config` members holding a workflow and settings, in walk order (as the stored file was made)."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "config" and isinstance(child, dict) and {"workflow", "settings"} <= child.keys():
                yield child
            else:
                yield from _configs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _configs(child)


def _sorted(value) -> str:
    return json.dumps(value, sort_keys=True)


def test_stored_base_covers_the_same_members():
    keys = {f"catalog/{wid}" for wid in catalog()}
    for path in sorted((FIX / "v0_1").glob("*.json")):
        keys |= {f"fixtures/{path.name}/{n}" for n, _ in enumerate(_configs(_read(path)))}
    ocp = set()
    for path in sorted(EXAMPLES.rglob("*.ocp.json")):
        run = _read(path).get("run")
        if isinstance(run, dict) and isinstance(run.get("configuration"), dict):
            ocp.add(f"ocp/{path.relative_to(EXAMPLES)}")
    assert {k for k in BASE if not k.startswith("cli/")} - {"skipped"} == keys | (ocp - set(BASE["skipped"]))
    assert set(BASE["skipped"]) <= ocp


@pytest.mark.parametrize("key", sorted(k for k in BASE if k.startswith("catalog/")))
def test_catalog_workflows_write_what_they_wrote(key):
    assert catalog()[key.split("/", 1)[1]].to_dict() == BASE[key]


@pytest.mark.parametrize("name", sorted(p.name for p in (FIX / "v0_1").glob("*.json")))
def test_fixture_configurations_write_what_they_wrote(name):
    for n, c in enumerate(_configs(_read(FIX / "v0_1" / name))):
        assert Configuration.from_dict(c).to_dict() == BASE[f"fixtures/{name}/{n}"], n


@pytest.mark.parametrize("key", sorted(k for k in BASE if k.startswith("ocp/")))
def test_ocp_configurations_read_and_emit_what_they_did(key):
    stored = BASE[key]
    c = _read(EXAMPLES / key.split("/", 1)[1])["run"]["configuration"]
    record = configuration_from_ocp(c)
    assert record.to_dict() == stored["to_dict"]
    emitted = configuration_to_ocp(record)
    assert _sorted(emitted) == _sorted(stored["ocp"])
    assert config_id(emitted["workflow"], emitted["settings"]) == stored["config_id"]
    # The stored record form read back: checks now land in Control.checks, and the emitted bytes hold
    again = Configuration.from_dict(stored["to_dict"])
    assert again.to_dict() == stored["to_dict"]
    emitted = configuration_to_ocp(again)
    assert _sorted(emitted) == _sorted(stored["ocp"])
    assert config_id(emitted["workflow"], emitted["settings"]) == stored["config_id"]


@pytest.mark.parametrize("key", sorted(BASE["skipped"]))
def test_skipped_files_still_raise_the_same_error(key):
    c = _read(EXAMPLES / key.split("/", 1)[1])["run"]["configuration"]
    with pytest.raises(Exception) as exc:
        configuration_from_ocp(c)
    assert f"{type(exc.value).__name__}: {exc.value}" == BASE["skipped"][key]


def test_control_without_checks_writes_three_members():
    assert list(Control().to_dict()) == ["gates", "budget_rounds", "rescue"]


# 3. Control.checks and Check

CHECKS = [
    {"id": "review-approves", "after": "review", "rule": "review_approve", "on_fail": "implement", "note": "kept"},
    {"id": "lint-passes", "after": "review", "rule": "command:lint"},
    {"id": "review-approves-again", "after": "review", "rule": "review_approve", "on_fail": None},
]


def test_control_checks_round_trip_with_ids_and_extras():
    data = {"gates": [], "budget_rounds": 3, "rescue": "redo_usual", "checks": copy.deepcopy(CHECKS), "x-u": 1}
    control = Control.from_dict(data)
    assert all(isinstance(c, Check) for c in control.checks)
    assert [c.id for c in control.checks] == ["review-approves", "lint-passes", "review-approves-again"]
    assert control.extra == {"x-u": 1}
    assert control.to_dict() == data and "checks" in control.to_dict()
    assert Control.from_dict(control.to_dict()) == control


def test_check_keeps_unknown_members_and_on_fail_as_read():
    kept, absent, null = (Check.from_dict(c) for c in CHECKS)
    assert kept.extra == {"note": "kept"} and kept.to_dict() == CHECKS[0]
    assert absent.on_fail is None and "on_fail" not in absent.to_dict()
    assert null.on_fail is None and null.to_dict() == CHECKS[2]
    assert Check(id="lint-passes", after="review", rule="command:lint").to_dict() == CHECKS[1]
    named = {"id": "c", "after": "review", "rule": "review_approve", "given": ["x"], "extra": 2}
    assert Check.from_dict(named).extra == {"given": ["x"], "extra": 2}
    assert Check.from_dict(named).to_dict() == named


@pytest.mark.parametrize("member", ["id", "after", "rule"])
def test_check_without_a_required_member_is_refused_by_name(member):
    data = {k: v for k, v in CHECKS[1].items() if k != member}
    with pytest.raises(ValueError, match=f"Check: the member '{member}' is missing"):
        Check.from_dict(data)
    with pytest.raises(ValueError, match=f"Check: the member '{member}' is missing"):
        Control.from_dict({"checks": [data]})


# 4. FlowRef


def test_flow_ref_built_in_code_writes_only_what_it_has():
    assert FlowRef(id="lit-brief", version=1).to_dict() == {"id": "lit-brief", "version": 1}


def test_flow_ref_round_trips_with_ext_and_unknown_members():
    data = {"id": "lit-brief", "version": 1, "digest": "flw_774ab7a14c50", "arm": "extract-sonnet",
            "plan": "pln_lit-brief_0001", "split": "holdout", "ext": {"dev.example.note": 1}, "x-u": [1]}
    ref = FlowRef.from_dict(data)
    assert ref.extra == {"ext": {"dev.example.note": 1}, "x-u": [1]}
    assert ref.to_dict() == data
    nulls = {"id": "lit-brief", "version": 1, "digest": None, "arm": None}
    assert FlowRef.from_dict(nulls).to_dict() == nulls


@pytest.mark.parametrize("member", ["id", "version"])
def test_flow_ref_without_a_required_member_is_refused_by_name(member):
    data = {k: v for k, v in {"id": "lit-brief", "version": 1, "arm": "base"}.items() if k != member}
    with pytest.raises(ValueError, match=f"FlowRef: the member '{member}' is missing"):
        FlowRef.from_dict(data)


# 5. Flow round trip


@pytest.mark.parametrize("path", [SPEC_FLOW] + _flow_goldens("pass"), ids=lambda p: p.name)
def test_flow_round_trips_as_written(path):
    flow = _read(path)["flow"]
    record = Flow.from_dict(flow)
    assert record.to_dict() == flow
    assert flow_digest(record.to_dict()) == flow_digest(flow)


@pytest.mark.parametrize("path", _flow_goldens("fail"), ids=lambda p: p.name)
def test_fail_flow_golden_round_trips_or_is_refused(path):
    flow = _read(path)["flow"]
    try:
        record = Flow.from_dict(flow)
    except ValueError:
        return
    assert record.to_dict() == flow


def _loaded_up() -> dict:
    """The spec example with an unknown member at every level, explicit nulls and empty lists."""
    flow = _read(SPEC_FLOW)["flow"]
    flow["x-u"] = {"a": 1}
    flow["given"] = ["not", "the", "field"]
    flow["extra"] = "an unknown member"
    flow["title"] = None
    flow["inputs"]["x-u"] = 1
    flow["inputs"]["ids"] = []
    flow["inputs"]["n"] = None
    flow["objective"]["x-u"] = 2
    flow["objective"]["cost_cap_usd"] = None
    for n, judge in enumerate(flow["judges"]):
        judge["x-u"] = n
    flow["judges"][0]["scale"] = None
    allow = flow["allow"]
    allow["x-u"] = 3
    allow["budgets"] = []
    allow["pieces"]["extract"]["x-u"] = 4
    allow["pieces"]["extract"]["efforts"] = []
    allow["pieces"]["synthesize"]["options"] = {"prompt": ["v3", "v4"], "tools": []}
    allow["merges"][0]["x-u"] = 5
    allow["checks"] = {"add": [{"id": "claims-again", "after": "synthesize", "rule": "command:claims-check"}],
                       "remove": [], "x-u": 6}
    return flow


def test_flow_keeps_unknown_members_nulls_and_empty_lists():
    flow = _loaded_up()
    record = Flow.from_dict(flow)
    assert record.to_dict() == flow
    assert flow_digest(record.to_dict()) == flow_digest(flow)
    assert record.extra == {"x-u": {"a": 1}, "given": ["not", "the", "field"], "extra": "an unknown member"}
    assert record.inputs.ids == () and record.inputs.holdout is not None and record.title is None


# 6. Types on read and the reading helpers


def test_members_read_into_records_and_tuples():
    record = Flow.from_dict(_loaded_up())
    assert all(isinstance(j, Judge) for j in record.judges) and isinstance(record.judges, tuple)
    assert isinstance(record.inputs, FlowInputs) and isinstance(record.objective, Objective)
    assert isinstance(record.allow, Allow) and isinstance(record.allow.checks, AllowChecks)
    assert all(isinstance(p, PieceAllow) for p in record.allow.pieces.values())
    assert all(isinstance(m, Merge) for m in record.allow.merges)
    assert record.allow.merges[0].replace == ("extract", "synthesize")
    assert record.allow.pieces["extract"].models == ({"id": "claude-sonnet-5-5"}, {"id": "claude-haiku-5-5"})
    assert record.allow.pieces["synthesize"].options == {"prompt": ("v3", "v4"), "tools": ()}
    assert record.allow.pieces["scout"].widths == (1, 2) and record.allow.budgets == ()
    assert isinstance(record.inputs.holdout, tuple) and isinstance(record.configuration, dict)


def test_reading_helpers_apply_the_defaults():
    record = Flow.from_dict(_read(SPEC_FLOW)["flow"])
    claude, _, person = record.judges
    assert reference(Objective(kind="cheapest_within_margin")) == "base"
    assert reference(record.objective) == "base"
    assert judge_score(claude, record.objective) == "quality"
    assert judge_score(Judge(id="j", score="faithful"), record.objective) == "faithful"
    assert judge_scale(claude) == "fraction" and judge_scale(Judge(id="j", scale="linear")) == "linear"
    assert judge_rate(claude) == 1 and judge_rate(person) == 0.2


def test_a_list_member_that_is_not_a_list_is_refused():
    flow = _read(SPEC_FLOW)["flow"]
    flow["judges"] = "j-claude"
    with pytest.raises(ValueError, match="Flow.judges"):
        Flow.from_dict(flow)
    flow = _read(SPEC_FLOW)["flow"]
    del flow["objective"]
    with pytest.raises(ValueError, match="objective"):
        Flow.from_dict(flow)


# 7. The digest and the document


def test_with_digest_and_flow_document_rebuild_the_example():
    doc = _read(SPEC_FLOW)
    record = Flow.from_document(doc)
    bare = Flow.from_dict({k: v for k, v in doc["flow"].items() if k != "digest"})
    assert bare.digest is None and "digest" not in bare.to_dict()
    assert bare.with_digest().digest == doc["flow"]["digest"] == "flw_774ab7a14c50"
    rebuilt = flow_document(record, producer=doc["producer"], privacy=doc["privacy"])
    assert rebuilt == doc
    assert conformance.validate_doc(rebuilt) == []
    assert flow_document(record, producer=doc["producer"])["privacy"] == {"profile": "metadata_only"}


def test_from_document_refuses_a_run_document():
    with pytest.raises(ValueError, match="not a flow document"):
        Flow.from_document(_read(EXAMPLES / "v0.4" / "two-checks.ocp.json"))


# 8. plan.json


def test_sample_plan_round_trips_and_its_arm_ids_recompute():
    data = _read(PLAN_SAMPLE)
    plan = Plan.from_dict(data)
    assert plan.to_dict() == data and plan.schema == PLAN_SCHEMA
    assert [a.id for a in plan.arms] == ["base", "extract-sonnet"]
    assert isinstance(plan.flow, FlowRef) and all(isinstance(e, Edit) for e in plan.queued)
    assert isinstance(plan.experiments[0], Experiment) and isinstance(plan.excluded[0], Excluded)
    assert plan.arms[0].edits == () and plan.arms[0].to_dict()["edits"] == []
    for arm in plan.arms:
        c = arm.configuration
        assert c["id"] == config_id(c["workflow"], c["settings"]), arm.id
    assert plan.arms[0].configuration["id"] == "cfg_9bc4f6e876f9"


@pytest.mark.parametrize("change, named", [
    (lambda d: d.update(schema="loopmath.flow-plan/2"), "loopmath.flow-plan/2"),
    (lambda d: d["arms"][1].update(id="unplanned"), "'unplanned'"),
    (lambda d: d["arms"][1].update(id="Bad Id"), "'Bad Id'"),
    (lambda d: d["arms"][1].update(id="extract-sonnet\n"), "'extract-sonnet\\n'"),
    (lambda d: d["flow"].pop("id"), "FlowRef: the member 'id' is missing"),
])
def test_plan_refusals_name_what_is_wrong(change, named):
    data = _read(PLAN_SAMPLE)
    change(data)
    with pytest.raises(ValueError) as exc:
        Plan.from_dict(data)
    assert named in str(exc.value)


# 9. RunSpec and CheckResult

RUN_SPEC = {
    "run_id": "run_lit-brief_0001_q13_extract-sonnet", "input": "q13", "arm": "extract-sonnet",
    "configuration_id": "cfg_b8de56a265ea",
    "settings": {"extract": {"harness": "api", "model": {"id": "claude-sonnet-5-5"}, "effort": "high"}},
    "experiment": "single-extract-sonnet", "split": "holdout", "slate": "slt_lit-brief_0001_q13",
    "judges": ["j-claude", "j-astra"], "person": False,
    "flow": {"id": "lit-brief", "version": 1, "digest": "flw_774ab7a14c50", "arm": "extract-sonnet",
             "plan": "pln_lit-brief_0001", "split": "holdout"},
    "record": "loopmath flow record FLOW --ocp RUN.ocp.json", "x-u": {"kept": True},
}


def test_run_spec_round_trips_with_unknown_members():
    spec = RunSpec.from_dict(RUN_SPEC)
    assert isinstance(spec.flow, FlowRef) and spec.judges == ("j-claude", "j-astra")
    assert spec.to_dict() == RUN_SPEC
    single = {k: v for k, v in RUN_SPEC.items() if k not in ("slate", "record")}
    assert RunSpec.from_dict(single).slate is None and RunSpec.from_dict(single).to_dict() == single
    built = dataclasses.replace(spec, slate=None, record=None, extra={}, given=frozenset())  # as built in code
    assert "slate" not in built.to_dict() and RunSpec.from_dict(built.to_dict()) == built


def test_check_result_round_trips_with_unknown_members():
    data = {"arm": "extract-sonnet", "reference": "base", "pairs": 12, "p_quality": None, "p_cost": 0.97,
            "threshold": 0.95, "margin": 0.1, "pair_cap": 24, "verdict": "more", "more_pairs": 12,
            "x-u": "kept"}
    result = CheckResult.from_dict(data)
    assert result.p_quality is None and result.to_dict() == data
    short = {k: v for k, v in data.items() if k not in ("p_quality", "more_pairs")}
    assert CheckResult.from_dict(short).to_dict() == short
