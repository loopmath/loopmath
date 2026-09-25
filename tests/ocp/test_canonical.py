"""The one canonical configuration id: pinned, order independent, D29 to D31 applied."""

from __future__ import annotations

import copy
import random

from loopmath.ocp.canonical import GATE_RULES_KEY, canonical_json, config_id
from loopmath.ocp.emit import configuration_to_ocp

from ._common import SHAPES, build_fixtures_module, example

WORKFLOW = {
    "id": "implement_review", "version": 1, "title": "Implement, then review",
    "pieces": [{"id": "implement", "role": "implementer"}, {"id": "review", "role": "reviewer"}],
    "artifacts": [{"id": "patch", "kind": "diff"}, {"id": "review_notes", "kind": "review"}],
    "edges": [["implement", "patch"], ["patch", "review"], ["review", "review_notes"]],
    "control": {"gates": ["review"], "repair": {"review": "implement"}, "budget": 3},
}
SETTINGS = {
    "implement": {"harness": "claude-code", "model": {"raw": "claude-opus-5-5", "id": "claude-opus-5-5"},
                  "effort": "high"},
    "review": {"harness": "codex", "model": "gpt-6-astra", "effort": "xhigh", "context_policy": "fresh",
               "options": {}},
}
PINNED_JSON = (
    '{"settings":{"implement":{"context_policy":"fresh","effort":"high","harness":"claude-code",'
    '"model":"claude-opus-5-5","options":{}},"review":{"context_policy":"fresh","effort":"xhigh",'
    '"harness":"codex","model":"gpt-6-astra","options":{}}},"workflow":{"artifacts":[{"id":"patch",'
    '"kind":"diff"},{"id":"review_notes","kind":"review"}],"control":{"budget":3,"gates":["review"],'
    '"repair":{"review":"implement"}},"edges":[["implement","patch"],["patch","review"],'
    '["review","review_notes"]],"pieces":[{"id":"implement","role":"implementer","width":1},'
    '{"id":"review","role":"reviewer","width":1}]}}'
)
PINNED_ID = "cfg_f03e792941f8"


def _shuffled(value, rng):
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {k: _shuffled(v, rng) for k, v in items}
    if isinstance(value, list):
        return [_shuffled(v, rng) for v in value]
    return value


def test_pinned_literal_is_stable_across_python_versions():
    assert canonical_json(WORKFLOW, SETTINGS) == PINNED_JSON
    assert config_id(WORKFLOW, SETTINGS) == PINNED_ID


def test_key_order_and_record_order_do_not_change_the_id():
    rng = random.Random(1)
    for _ in range(20):
        workflow = _shuffled(WORKFLOW, rng)
        for key in ("pieces", "artifacts", "edges"):
            rng.shuffle(workflow[key])
        rng.shuffle(workflow["control"]["gates"])
        assert config_id(workflow, _shuffled(SETTINGS, rng)) == PINNED_ID


def test_labels_and_extensions_are_outside_the_id():
    workflow = copy.deepcopy(WORKFLOW)
    workflow.update(id="my_flow", version=7, title="renamed", ext={"dev.example.note": 1})
    assert config_id(workflow, SETTINGS) == PINNED_ID


def test_defaults_read_the_same_as_their_absence():
    workflow, settings = copy.deepcopy(WORKFLOW), copy.deepcopy(SETTINGS)
    for piece in workflow["pieces"]:
        piece["width"] = 1
    settings["implement"].update(context_policy="fresh", options={})  # Fresh when absent
    settings["implement"]["model"] = "claude-opus-5-5"  # modelRef id or a bare string
    assert config_id(workflow, settings) == PINNED_ID


def test_budget_counts_round_one():
    """A missing or zero budget reads as 1."""
    ids = set()
    for budget in (None, 0, 1):
        workflow = copy.deepcopy(WORKFLOW)
        if budget is None:
            del workflow["control"]["budget"]
        else:
            workflow["control"]["budget"] = budget
        ids.add(config_id(workflow, SETTINGS))
    assert len(ids) == 1
    assert ids != {PINNED_ID}


def test_what_ran_changes_the_id():
    changed = []
    settings = copy.deepcopy(SETTINGS)
    settings["review"]["effort"] = "high"
    changed.append(config_id(WORKFLOW, settings))
    settings = copy.deepcopy(SETTINGS)
    settings["review"]["context_policy"] = "inherit"
    changed.append(config_id(WORKFLOW, settings))
    workflow = copy.deepcopy(WORKFLOW)
    workflow["pieces"][0]["width"] = 3
    changed.append(config_id(workflow, SETTINGS))
    workflow = copy.deepcopy(WORKFLOW)
    workflow["control"]["ext"] = {GATE_RULES_KEY: {"review": "command:lint"}}
    changed.append(config_id(workflow, SETTINGS))
    workflow = copy.deepcopy(WORKFLOW)
    workflow["control"]["rescue"] = {"kind": "person"}
    changed.append(config_id(workflow, SETTINGS))
    assert PINNED_ID not in changed
    assert len(set(changed)) == len(changed)


def test_other_control_extensions_stay_outside_the_id():
    workflow = copy.deepcopy(WORKFLOW)
    workflow["control"]["ext"] = {"dev.example.hint": "x"}
    assert config_id(workflow, SETTINGS) == PINNED_ID


def test_types_configurations_and_their_ocp_form_share_the_id():
    fixtures = build_fixtures_module()
    for config in fixtures.ALL:
        ocp = configuration_to_ocp(config)
        assert ocp["id"] == config_id(ocp["workflow"], ocp["settings"])


def test_every_example_carries_its_canonical_id():
    for name in SHAPES:
        configuration = example(name)["run"]["configuration"]
        assert configuration["id"] == config_id(configuration["workflow"], configuration["settings"])
