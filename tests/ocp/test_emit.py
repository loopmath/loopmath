"""Emit helpers: types records to OCP v0.3 (spec 03 section 4)."""

from __future__ import annotations

import dataclasses

import pytest

from loopmath.ocp import emit
from loopmath.ocp.canonical import GATE_RULES_KEY, config_id
from loopmath.types import AcceptanceRule, Configuration, Control, Gate, Piece, Setting, Task, Workflow
from loopmath.workflows.ocp import configuration_from_ocp

from ._common import build_fixtures_module, example

OPUS = Setting("claude-code", "claude-opus-5-5", "high")
ASTRA = Setting("codex", "gpt-6-astra", "xhigh")
TASK = Task(id="tsk_emit01", type="feature", repo="example/widget", title="t", base_commit="a1b2c3d",
            labeled_by="model:claude-fable-5-1")
RULE = AcceptanceRule(name="tests+review", definition="tests pass and the reviewer approves",
                      requires=("tests", "review"))


def implement_review(rule="review_approve", control_extra=None):
    """Lane 04's D41 repro shape."""
    return Workflow(
        id="implement_review", version=1, title="Implement, then review",
        pieces=(Piece("implement", "implementer"), Piece("review", "reviewer")),
        artifacts=("patch", "review_notes"),
        edges=(("implement", "patch"), ("patch", "review"), ("review", "review_notes")),
        control=Control(gates=(Gate("g_review", "review", rule, on_fail="implement"),), budget_rounds=3,
                        extra=control_extra or {}),
    )


def ocp_config(workflow):
    return emit.configuration_to_ocp(Configuration("", workflow, {"implement": OPUS, "review": ASTRA}))


def test_d41_control_ext_is_merged_with_the_gate_rules():
    """Lane 04's repro: `command:lint` with Control.extra = {"ext": {"x.y": 1}} keeps its gate rule."""
    plain = ocp_config(implement_review("command:lint"))
    with_ext = ocp_config(implement_review("command:lint", {"ext": {"x.y": 1}}))
    default = ocp_config(implement_review("review_approve", {"ext": {"x.y": 1}}))
    assert with_ext["id"] == plain["id"] != default["id"]
    assert with_ext["workflow"]["control"]["ext"] == {"x.y": 1, GATE_RULES_KEY: {"review": "command:lint"}}
    assert default["workflow"]["control"]["ext"] == {"x.y": 1}


def test_d41_the_gates_win_over_a_stale_gate_rules_key():
    stale = {"ext": {GATE_RULES_KEY: {"review": "command:old"}}}
    assert ocp_config(implement_review("command:lint", stale))["workflow"]["control"]["ext"] == {
        GATE_RULES_KEY: {"review": "command:lint"}}
    assert "ext" not in ocp_config(implement_review("review_approve", stale))["workflow"]["control"]


def test_workflow_mapping():
    w = emit.workflow_to_ocp(implement_review())
    assert w["pieces"] == [{"id": "implement", "role": "implementer", "width": 1},
                           {"id": "review", "role": "reviewer", "width": 1}]  # Roles verbatim
    assert w["artifacts"] == [{"id": "patch", "kind": "other"}, {"id": "review_notes", "kind": "other"}]
    assert w["control"] == {"gates": ["review"], "repair": {"review": "implement"}, "budget": 3,
                            "rescue": {"kind": "configuration", "ref": "usual"}}
    kinds = dataclasses.replace(implement_review(), extra={"artifact_kinds": {"patch": "diff"}})
    assert emit.workflow_to_ocp(kinds)["artifacts"][0] == {"id": "patch", "kind": "diff"}
    assert emit.workflow_to_ocp(w) == w  # an OCP-form dict passes through


def test_a_cyclic_workflow_raises():
    """Repair loops live in control, never as graph edges."""
    cyclic = dataclasses.replace(implement_review(), edges=implement_review().edges + (("review_notes", "implement"),))
    with pytest.raises(emit.WorkflowCycleError):
        emit.workflow_to_ocp(cyclic)
    with pytest.raises(emit.WorkflowCycleError):
        emit.workflow_to_ocp({"pieces": [{"id": "a"}], "artifacts": [{"id": "x"}], "edges": [["a", "x"], ["x", "a"]],
                              "control": {}})


def test_setting_mapping_writes_context_policy():
    """Fresh when absent, always written."""
    assert emit.setting_to_ocp(ASTRA) == {"harness": "codex", "model": {"raw": "gpt-6-astra", "id": "gpt-6-astra"},
                                          "effort": "xhigh", "context_policy": "fresh", "options": {}}
    assert emit.setting_to_ocp({"harness": "codex", "model": "m"})["context_policy"] == "fresh"


MODEL_REF = {"raw": "opus", "id": "claude-opus-5-5", "provider": "anthropic"}


def test_kept_model_object_folds_back_into_model():
    """Extra["model_ref"] (kept by lane 04's setting_from_ocp) becomes `model`; no top-level model_ref."""
    kept = Setting("claude-code", "claude-opus-5-5", "high", extra={"model_ref": MODEL_REF})
    out = emit.setting_to_ocp(kept)
    assert out["model"] == MODEL_REF and "model_ref" not in out
    assert emit.setting_to_ocp({"harness": "codex", "model": "m", "model_ref": {"provider": "p"}})["model"] == {
        "raw": "m", "provider": "p", "id": "m"}
    shape = implement_review()
    c = emit.configuration_to_ocp(Configuration("x", shape, {"implement": kept, "review": ASTRA}))
    assert c["settings"]["implement"]["model"] == MODEL_REF and "model_ref" not in c["settings"]["implement"]
    assert c["id"] == emit.configuration_to_ocp(Configuration("x", shape, {"implement": OPUS, "review": ASTRA}))["id"]


def test_lane04_model_ref_repro():
    """D52, lane 04's repro: configuration_to_ocp(configuration_from_ocp(doc)) keeps the model object."""
    configuration = example("implement_review")["run"]["configuration"]
    configuration["settings"]["implement"]["model"] = dict(MODEL_REF)
    out = emit.configuration_to_ocp(configuration_from_ocp(configuration))
    assert out["settings"]["implement"]["model"] == MODEL_REF and "model_ref" not in out["settings"]["implement"]


def test_configuration_id_is_recomputed():
    c = emit.configuration_to_ocp(Configuration("cfg_stale0000000", implement_review(), {"implement": OPUS,
                                                                                        "review": ASTRA}),
                                  source="usual", rec="rec_1")
    assert c["id"] == config_id(c["workflow"], c["settings"]) != "cfg_stale0000000"
    assert (c["source"], c["rec"]) == ("usual", "rec_1")


def test_task_rule_and_signal_mapping():
    task = emit.task_to_ocp(TASK)
    assert task["labeled_by"] == {"how": "labeler", "model": "claude-fable-5-1"}
    assert task["source"] == {"kind": "live"}
    rule = emit.rule_to_ocp(RULE)
    assert rule["requires"] == ["tests", "review"] and rule["score"] is None
    assert rule["excludes_events"] == ["revert", "incident"] and rule["window_days"] == 14
    signal = emit.signal_to_ocp({"id": "s1", "run": "r", "kind": "verdict", "name": "tests", "value": "pass",
                                 "unit": "x", "better": "higher", "observed_at": "2026-09-23T10:00:00-07:00"})
    assert signal == {"id": "s1", "kind": "verdict", "name": "tests", "value": "pass",
                      "observed_at": "2026-09-23T10:00:00-07:00", "source": {"kind": "orchestrator"},
                      "tier": "reported"}


def new_doc(config=None, **kwargs):
    config = config or Configuration("", implement_review(), {"implement": OPUS, "review": ASTRA})
    return emit.new_run_doc(run_id="run_emit01", task=TASK, configuration=config, acceptance_rule=RULE,
                            provenance="designed", **kwargs)


def test_new_run_doc_is_strictly_valid():
    doc = new_doc(source="usual", slate="slt_1")
    assert emit.validate_strict(doc) == []
    assert doc["producer"]["name"] == "loopmath" and doc["ocp"] == "0.3"
    assert [n["id"] for n in doc["nodes"]] == ["implement", "review"]
    assert [n["vertex"] for n in doc["nodes"]] == ["implement", "review"]
    assert doc["edges"] == [{"from": "implement", "to": "review", "kind": "dep", "tier": "reported"}]
    assert doc["run"]["slate"] == {"id": "slt_1", "members": ["run_emit01"], "base_commit": "a1b2c3d"}
    offset = doc["run"]["started_at"][-6:]
    assert offset[0] in "+-" and offset[3] == ":"  # local offset, never a bare Z


def test_records_added_during_a_run_stay_valid():
    doc = new_doc()
    doc["attempts"].append(emit.attempt_record(attempt_id="implement.a1", node="implement", harness="claude-code",
                                               model="claude-opus-5-5", effort="high", round=1, n=1))
    emit.add_signal(doc, {"id": "s1", "kind": "verdict", "name": "review", "value": "accept",
                          "at_attempt": "implement.a1"})
    emit.add_event(doc, "run_finished", detail="accepted")
    assert [e["type"] for e in doc["events"]] == ["note", "signal_observed", "run_finished"]
    assert emit.validate_strict(doc) == []
    doc["run"]["preferences"] = [emit.preference_record(pref_id="p1", slate="slt_1", winner="tie",
                                                        members=["run_emit01"], blinded=True)]
    doc["run"]["slate"] = {"id": "slt_1", "members": ["run_emit01"]}
    assert emit.validate_strict(doc) == []


def test_validate_strict_takes_only_v03():
    doc = new_doc()
    doc["ocp"] = "0.2"
    findings = emit.validate_strict(doc)
    assert findings[0]["code"] == "E005" and "ocp migrate" in findings[0]["message"]


@pytest.mark.parametrize("name", ["USUAL", "GOAL", "EXPLORE", "CHEAP", "ALT2", "ALT3"])
def test_every_scaffold_fixture_configuration_emits_a_valid_run(name):
    """The scaffold's view fixtures (after D3 dropped the back edges) as v0.3 runs."""
    fixtures = build_fixtures_module()
    doc = emit.new_run_doc(run_id=f"run_{name.lower()}", task=fixtures.TASK, configuration=getattr(fixtures, name),
                           acceptance_rule=fixtures.BINARY, provenance="designed")
    assert emit.validate_strict(doc) == []
    doc["run"]["acceptance_rule"] = emit.rule_to_ocp(fixtures.SCORE)
    assert emit.validate_strict(doc) == []
