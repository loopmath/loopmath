"""Configuration ids and the types to OCP mapping (lane 04)."""

from __future__ import annotations

import dataclasses

import pytest

from loopmath.types import Configuration, Setting
from loopmath.workflows import ocp as wocp
from loopmath.workflows.format import catalog
from loopmath.workflows.ids import canonical, config_id, config_id_ocp, make_config, structure_key
from loopmath.workflows.ocp import (CycleError, GATE_RULES_KEY, configuration_from_any, configuration_from_ocp,
                                    configuration_to_ocp, workflow_from_ocp, workflow_to_ocp)

OPUS = Setting("claude-code", "claude-opus-5-5", "high")
ASTRA = Setting("codex", "gpt-6-astra", "high", options={"x": "1"})


def _review_rule_rescue():
    wf = catalog()["implement_review"]
    gate = dataclasses.replace(wf.control.gates[0], rule="command:lint")
    rescue = {"kind": "configuration", "ref": "cfg_000000000000", "cost_usd": 3.0}
    return dataclasses.replace(wf, control=dataclasses.replace(wf.control, gates=(gate,), rescue=rescue))


def test_ids_are_pinned_and_match_lane_01_canonical():
    # Computed with lane 01's loopmath.ocp.canonical and emit at 4d91137.
    assert config_id(catalog()["solo"], {"implement": OPUS}) == "cfg_261aaa36000b"
    assert config_id(catalog()["implement_review"], {"implement": OPUS, "review": ASTRA}) == "cfg_a704cc721a8c"
    assert config_id(_review_rule_rescue(), {"implement": OPUS, "review": ASTRA}) == "cfg_c90540c9298b"


def test_id_ignores_workflow_id_title_version_order_and_extras():
    wf = catalog()["implement_review"]
    s = {"implement": OPUS, "review": ASTRA}
    base = config_id(wf, s)
    renamed = dataclasses.replace(wf, id="mine", title="Other", version=7, extra={**wf.extra, "owner": "x"})
    assert config_id(renamed, s) == base
    shuffled = dataclasses.replace(wf, pieces=wf.pieces[::-1], artifacts=wf.artifacts[::-1], edges=wf.edges[::-1])
    assert config_id(shuffled, dict(reversed(list(s.items())))) == base
    gate = dataclasses.replace(wf.control.gates[0], id="other_gate_id", extra={"note": 1})
    assert config_id(dataclasses.replace(wf, control=dataclasses.replace(wf.control, gates=(gate,))), s) == base


@pytest.mark.parametrize("change", [
    lambda wf, s: (dataclasses.replace(wf, control=dataclasses.replace(wf.control, budget_rounds=4)), s),
    lambda wf, s: (dataclasses.replace(wf, control=dataclasses.replace(wf.control, rescue="person")), s),
    lambda wf, s: (_review_rule_rescue(), s),
    lambda wf, s: (wf, {**s, "review": dataclasses.replace(s["review"], effort="xhigh")}),
    lambda wf, s: (wf, {**s, "review": dataclasses.replace(s["review"], context_policy="inherit")}),
    lambda wf, s: (wf, {**s, "review": dataclasses.replace(s["review"], options={})}),
    lambda wf, s: (dataclasses.replace(wf, pieces=(dataclasses.replace(wf.pieces[0], width=2),) + wf.pieces[1:]), s),
    lambda wf, s: (dataclasses.replace(wf, extra={"artifact_kinds": {**wf.extra["artifact_kinds"], "diff": "other"}}), s),
])
def test_id_changes_with_shape_and_settings(change):
    wf = catalog()["implement_review"]
    s = {"implement": OPUS, "review": ASTRA}
    assert config_id(*change(wf, s)) != config_id(wf, s)


def test_default_gate_rule_is_not_written_and_other_rules_are():
    wf = catalog()["implement_review"]
    assert "ext" not in workflow_to_ocp(wf)["control"]
    assert workflow_to_ocp(_review_rule_rescue())["control"]["ext"] == {GATE_RULES_KEY: {"review": "command:lint"}}


def test_budget_zero_or_missing_reads_as_one_and_model_id_wins_over_raw():
    w = workflow_to_ocp(catalog()["solo"])
    s = {"implement": {"harness": "claude-code", "model": {"raw": "opus", "id": "claude-opus-5-5"}, "effort": "high"}}
    one = config_id_ocp(w, s)
    for budget in (0, None, 1):
        w2 = {**w, "control": {**w["control"], "budget": budget}}
        assert config_id_ocp(w2, s) == one
    plain = {"implement": {"harness": "claude-code", "model": "claude-opus-5-5", "effort": "high",
                           "context_policy": "fresh", "options": {}}}
    assert config_id_ocp(w, plain) == one


def test_ocp_mapping_follows_d29_to_d31():
    cfg = make_config(catalog()["best_of_n"], {"implement": OPUS, "select": ASTRA})
    doc = configuration_to_ocp(cfg, source="usual")
    w = doc["workflow"]
    assert doc["id"] == cfg.id and doc["source"] == "usual"
    assert {"id": "candidates", "kind": "diff"} in w["artifacts"]
    assert w["pieces"][0] == {"id": "implement", "role": "implementer", "width": 3}
    assert w["control"] == {"gates": ["select"], "repair": {}, "budget": 1,
                            "rescue": {"kind": "configuration", "ref": "usual"}}
    assert doc["settings"]["implement"] == {"harness": "claude-code", "model": {"raw": "claude-opus-5-5", "id": "claude-opus-5-5"},
                                           "effort": "high", "context_policy": "fresh", "options": {}}
    with pytest.raises(ValueError):
        configuration_to_ocp(cfg, source="nope")


@pytest.mark.parametrize("name", list(catalog()))
def test_ocp_inverse_round_trips_every_catalog_shape(name):
    wf = catalog()[name]
    settings = {p.id: (ASTRA if p.role in ("reviewer", "referee") else OPUS) for p in wf.pieces}
    cfg = make_config(wf, settings)
    back = configuration_from_ocp(configuration_to_ocp(cfg))
    assert back.id == cfg.id
    assert back.workflow.control == wf.control and back.workflow.pieces == wf.pieces
    assert structure_key(back.workflow) == structure_key(wf)
    assert "declared_id" not in back.extra


def test_inverse_keeps_rescue_objects_ext_and_model_refs():
    wf = _review_rule_rescue()
    wf = dataclasses.replace(wf, control=dataclasses.replace(wf.control, extra={"ext": {"x.y": 1}}))
    s = {"implement": OPUS, "review": ASTRA}
    cfg = make_config(wf, s)
    doc = configuration_to_ocp(cfg)
    assert doc["workflow"]["control"]["ext"] == {"x.y": 1, GATE_RULES_KEY: {"review": "command:lint"}}
    doc["settings"]["implement"]["model"] = {"raw": "opus", "id": "claude-opus-5-5", "provider": "anthropic"}
    back = configuration_from_ocp(doc)
    assert back.id == cfg.id
    assert back.workflow.control == wf.control
    assert back.settings["implement"].extra["model_ref"]["raw"] == "opus"
    assert configuration_to_ocp(back)["settings"]["implement"]["model"]["provider"] == "anthropic"


def test_a_declared_id_that_does_not_match_is_kept_aside():
    cfg = make_config(catalog()["solo"], {"implement": OPUS})
    doc = configuration_to_ocp(cfg)
    doc["id"] = "cfg_000000000000"
    back = configuration_from_ocp(doc)
    assert back.id == cfg.id and back.extra["declared_id"] == "cfg_000000000000"


def test_workflow_refs_resolve_through_the_catalog():
    assert workflow_from_ocp({"ref": "swarm", "version": 1}) == catalog()["swarm"]
    with pytest.raises(LookupError):
        workflow_from_ocp({"ref": "swarm", "version": 9})


def test_configuration_from_any_reads_types_dicts_and_ocp():
    cfg = make_config(catalog()["plan_implement"], {"plan": OPUS, "implement": OPUS})
    assert configuration_from_any(cfg.to_dict()).id == cfg.id
    assert configuration_from_any(configuration_to_ocp(cfg)).id == cfg.id
    assert configuration_from_any(cfg) is cfg


def test_cyclic_workflow_raises():
    wf = catalog()["implement_review"]
    looped = dataclasses.replace(wf, edges=wf.edges + (("verdict", "implement"),))
    with pytest.raises(CycleError):
        workflow_to_ocp(looped)
    with pytest.raises(CycleError):
        config_id(looped, {"implement": OPUS, "review": ASTRA})


def test_canonical_is_compact_sorted_json():
    text = canonical(catalog()["solo"], {"implement": OPUS})
    assert text.startswith('{"settings":{"implement":{"context_policy":"fresh"')
    assert " " not in text


def test_make_config_orders_settings_by_piece():
    cfg = make_config(catalog()["implement_review"], {"review": ASTRA, "implement": OPUS})
    assert list(cfg.settings) == ["implement", "review"]
    assert isinstance(cfg, Configuration) and cfg.id == config_id(cfg.workflow, cfg.settings)
