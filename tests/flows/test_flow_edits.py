"""One-step edits and candidate arms on the example flow, and on test-local copies of it."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from loopmath.flows import edits
from loopmath.flows.folder import load
from loopmath.flows.records import ARM_ID, Edit, Flow, Plan, hypothesis_edits
from loopmath.ocp.canonical import config_id
from loopmath.ocp.conformance import catalog_resolver

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
PLAN_SAMPLE = ROOT / "tests" / "fixtures" / "flows" / "plan.sample.json"

SONNET, HAIKU = {"id": "claude-sonnet-5-5"}, {"id": "claude-haiku-5-5"}
MERGE = "extract+synthesize"
EXAMPLE_ARMS = ["extract-claude-sonnet-5-5", "extract-claude-haiku-5-5", "synthesize-effort-medium", "drop-critic",
                "scout-width-2", "budget-1", "budget-3", "merge-extract-synthesize"]
BASE_EDGES = [["question", "scout"], ["scout", "sources"], ["sources", "extract"], ["extract", "claims"],
              ["claims", "synthesize"], ["question", "synthesize"], ["synthesize", "brief"], ["brief", "critic"],
              ["claims", "critic"], ["critic", "verdict"]]
RESCUE = {"kind": "configuration", "ref": "usual"}
RULES = "dev.loopmath.gate_rules"


def _example() -> Flow:
    return load(EXAMPLE).flow


def _flow(change) -> Flow:
    """A test-local copy of the example flow, changed by `change(flow_dict)`, with the id of an inline
    configuration recomputed. The example itself is never written."""
    data = json.loads((EXAMPLE / "flow.ocp.json").read_text(encoding="utf-8"))["flow"]
    change(data)
    c = data["configuration"]
    if "pieces" in c["workflow"]:
        c["id"] = config_id(c["workflow"], c["settings"], resolve=catalog_resolver)
    return Flow.from_dict(data)


def _ok(c: dict) -> dict:
    assert edits.validate(c) == []
    assert c["id"] == config_id(c["workflow"], c["settings"])
    assert c["source"] == "designed"
    return c


def _without(c: dict, *paths: tuple) -> dict:
    """`c` without `id`, `source` and the members at `paths`."""
    out = copy.deepcopy(c)
    out.pop("id")
    out.pop("source")
    for path in paths:
        at = out
        for key in path[:-1]:
            at = at[key]
        at.pop(path[-1])
    return out


def _ids(c: dict) -> list[str]:
    return [p["id"] for p in c["workflow"]["pieces"]]


def _artifacts(c: dict) -> list[str]:
    return [a["id"] for a in c["workflow"]["artifacts"]]


def _gate_after_scout(flow: dict) -> None:
    control = flow["configuration"]["workflow"]["control"]
    control["gates"] = ["scout", "extract", "critic"]
    control["repair"] = {"scout": "scout", "extract": "extract", "critic": "synthesize"}


def _checks_form(flow: dict) -> None:
    flow["configuration"]["workflow"]["control"] = {
        "checks": [
            {"id": "claims", "after": "extract", "rule": "command:claims-check", "on_fail": "extract",
             "ext": {"dev.example.note": "kept"}},
            {"id": "brief-format", "after": "synthesize", "rule": "tests_pass", "on_fail": None},
            {"id": "critic-review", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"},
        ],
        "budget": 2,
        "rescue": RESCUE,
    }
    flow["acceptance_rule"]["requires"] = ["claims"]


def _arm(arms: list, arm_id: str):
    found = [a for a in arms if a.id == arm_id]
    assert len(found) == 1, [a.id for a in arms]
    return found[0]


# 1 to 3: setting edits


def test_model_changes_only_the_setting():
    flow = _example()
    base = flow.configuration
    c = _ok(edits.apply(base, Edit(kind="model", piece="extract", value=SONNET), flow))
    assert c["workflow"] == base["workflow"]
    assert c["settings"]["extract"]["model"] == SONNET
    assert _without(c, ("settings", "extract", "model")) == _without(base, ("settings", "extract", "model"))
    assert c["id"] != base["id"]


def test_effort_changes_only_the_setting():
    flow = _example()
    base = flow.configuration
    c = _ok(edits.apply(base, Edit(kind="effort", piece="synthesize", value="medium"), flow))
    assert c["settings"]["synthesize"]["effort"] == "medium"
    assert _without(c, ("settings", "synthesize", "effort")) == _without(base, ("settings", "synthesize", "effort"))


def test_option_changes_only_the_setting():
    flow = _flow(lambda f: f["allow"]["pieces"]["synthesize"].update(options={"prompt": ["v3", "v4"]}))
    base = flow.configuration
    c = _ok(edits.apply(base, Edit(kind="option", piece="synthesize", key="prompt", value="v3"), flow))
    assert c["settings"]["synthesize"]["options"] == {"prompt": "v3"}
    path = ("settings", "synthesize", "options")
    assert _without(c, path) == _without(base, path)
    arms, excluded = edits.candidates(flow)
    assert excluded == []
    assert [a.id for a in arms if "prompt" in a.id] == ["synthesize-prompt-v3", "synthesize-prompt-v4"]


# 4: width


def test_width_two_adds_a_referee_and_width_one_removes_it():
    flow = _example()
    base = flow.configuration
    c = _ok(edits.apply(base, Edit(kind="width", piece="scout", value=2), flow))
    w = c["workflow"]
    assert w["pieces"][:2] == [{"id": "scout", "role": "worker", "width": 2},
                               {"id": "select_scout", "role": "referee", "width": 1}]
    assert _ids(c) == ["scout", "select_scout", "extract", "synthesize", "critic"]
    assert _artifacts(c) == ["question", "sources", "sources.cands", "claims", "brief", "verdict"]
    assert w["artifacts"][2] == {"id": "sources.cands", "kind": "other"}
    assert w["edges"] == [["question", "scout"], ["scout", "sources.cands"], ["sources.cands", "select_scout"],
                          ["select_scout", "sources"]] + BASE_EDGES[2:]
    assert ["sources", "extract"] in w["edges"]
    assert list(c["settings"]) == ["scout", "select_scout", "extract", "synthesize", "critic"]
    assert c["settings"]["select_scout"] == {"harness": "api", "model": HAIKU}
    assert w["control"] == base["workflow"]["control"]
    back = edits.apply(c, Edit(kind="width", piece="scout", value=1), flow)
    assert back == {**base, "source": "designed"}


def test_width_moves_a_gate_to_the_referee_and_back():
    flow = _flow(_gate_after_scout)
    base = flow.configuration
    c = _ok(edits.apply(base, Edit(kind="width", piece="scout", value=2), flow))
    control = c["workflow"]["control"]
    assert control["gates"] == ["select_scout", "extract", "critic"]
    assert control["repair"] == {"select_scout": "scout", "extract": "extract", "critic": "synthesize"}
    assert control["ext"] == {RULES: {"select_scout": "review_approve", "extract": "command:claims-check"}}
    back = _ok(edits.apply(c, Edit(kind="width", piece="scout", value=1), flow))
    assert back["workflow"]["control"]["ext"] == {RULES: {"extract": "command:claims-check"}}
    assert back == {**base, "source": "designed"}


def test_width_on_a_present_referee_sets_only_the_width():
    flow = _example()
    two = edits.apply(flow.configuration, Edit(kind="width", piece="scout", value=2), flow)
    three = _ok(edits.apply(two, Edit(kind="width", piece="scout", value=3), flow))
    assert three["workflow"]["pieces"][0]["width"] == 3
    assert _without(three, ("workflow", "pieces")) == _without(two, ("workflow", "pieces"))


def _user_referee(flow: dict) -> None:
    """A referee of the flow's own named `select_scout`, reading `brief` and writing `vote`."""
    c = flow["configuration"]
    w = c["workflow"]
    w["pieces"].append({"id": "select_scout", "role": "referee", "width": 1})
    w["artifacts"].append({"id": "vote", "kind": "other"})
    w["edges"] += [["brief", "select_scout"], ["select_scout", "vote"]]
    c["settings"]["select_scout"] = {"harness": "api", "model": HAIKU}


def test_width_leaves_another_piece_of_the_referee_name_alone():
    flow = _flow(_user_referee)
    base = flow.configuration
    assert edits.validate(base) == []
    assert edits.apply(base, Edit(kind="width", piece="scout", value=1), flow) == {**base, "source": "designed"}
    with pytest.raises(ValueError, match="'select_scout' is already in the workflow"):
        edits.apply(base, Edit(kind="width", piece="scout", value=2), flow)
    arms, excluded = edits.candidates(flow)
    assert [a.id for a in arms if a.edits[0].kind == "width"] == []
    assert [(x.edit.kind, x.edit.piece, x.edit.value) for x in excluded] == [("width", "scout", 2)]
    assert "'select_scout' is already in the workflow" in excluded[0].reason


def test_width_reverses_only_the_wiring_it_made():
    flow = _example()
    two = edits.apply(flow.configuration, Edit(kind="width", piece="scout", value=2), flow)
    two["workflow"]["edges"].append(["sources.cands", "critic"])  # a second reader: no longer the made wiring
    one = edits.apply(two, Edit(kind="width", piece="scout", value=1), flow)
    assert one["workflow"]["pieces"][:2] == [{"id": "scout", "role": "worker", "width": 1},
                                             {"id": "select_scout", "role": "referee", "width": 1}]
    assert _without(one, ("workflow", "pieces")) == _without(two, ("workflow", "pieces"))
    with pytest.raises(ValueError, match="'select_scout' is already in the workflow"):
        edits.apply(two, Edit(kind="width", piece="scout", value=3), flow)


def _selects(form):
    def change(f):
        if form is not None:
            form(f)
        for p in ("scout", "extract", "synthesize", "critic"):
            f["allow"]["pieces"].setdefault(p, {})["select"] = {"harness": "api", "model": HAIKU}

    return change


@pytest.mark.parametrize("form", [None, _gate_after_scout, _checks_form], ids=["gates", "gate-after-scout", "checks"])
@pytest.mark.parametrize("piece", ["scout", "extract", "synthesize", "critic"])
def test_width_round_trips_at_every_piece(form, piece):
    flow = _flow(_selects(form))
    base = flow.configuration
    two = _ok(edits.apply(base, Edit(kind="width", piece=piece, value=2), flow))
    assert f"select_{piece}" in _ids(two)
    three = _ok(edits.apply(two, Edit(kind="width", piece=piece, value=3), flow))
    assert _without(three, ("workflow", "pieces")) == _without(two, ("workflow", "pieces"))
    for made in (two, three):
        assert edits.apply(made, Edit(kind="width", piece=piece, value=1), flow) == {**base, "source": "designed"}


def test_width_round_trip_keeps_the_id_when_a_required_name_moved():
    def change(f):
        _gate_after_scout(f)
        f["acceptance_rule"]["requires"] = ["claims-check", "scout"]

    flow = _flow(change)
    base = flow.configuration
    two = edits.apply(base, Edit(kind="width", piece="scout", value=2), flow)
    back = _ok(edits.apply(two, Edit(kind="width", piece="scout", value=1), flow))
    assert back["id"] == base["id"]
    assert back["workflow"]["control"]["checks"][0] == {"id": "scout", "after": "scout", "rule": "review_approve",
                                                         "on_fail": "scout"}
    assert _without(back, ("workflow", "control")) == _without(base, ("workflow", "control"))


def test_width_round_trip_drops_a_written_default_rule_of_a_moved_gate():
    flow = _flow(_selects(lambda f: f["configuration"]["workflow"]["control"]["ext"][RULES].update(
        critic="review_approve")))
    base = flow.configuration
    two = edits.apply(base, Edit(kind="width", piece="critic", value=2), flow)
    back = _ok(edits.apply(two, Edit(kind="width", piece="critic", value=1), flow))
    assert back["workflow"]["control"]["ext"] == {RULES: {"extract": "command:claims-check"}}
    assert _without(back, ("workflow", "control", "ext")) == _without(base, ("workflow", "control", "ext"))


def test_width_refuses_a_piece_with_no_output():
    def change(f):
        w = f["configuration"]["workflow"]
        w["artifacts"] = [a for a in w["artifacts"] if a["id"] != "verdict"]
        w["edges"].remove(["critic", "verdict"])
        f["allow"]["pieces"]["critic"].update(widths=[1, 2], select={"harness": "api", "model": HAIKU})

    flow = _flow(change)
    base = flow.configuration
    assert edits.validate(base) == []
    with pytest.raises(ValueError, match="'critic' produces no artifact"):
        edits.apply(base, Edit(kind="width", piece="critic", value=2), flow)
    assert edits.apply(base, Edit(kind="width", piece="critic", value=1), flow) == {**base, "source": "designed"}
    arms, excluded = edits.candidates(flow)
    assert [a.id for a in arms if a.edits[0].kind == "width"] == ["scout-width-2"]
    assert [(x.edit.kind, x.edit.piece, x.edit.value) for x in excluded] == [("width", "critic", 2)]
    assert "'critic' produces no artifact" in excluded[0].reason


# 5: budget


@pytest.mark.parametrize("budget", [1, 3])
def test_budget_sets_only_the_budget(budget):
    flow = _example()
    base = flow.configuration
    c = _ok(edits.apply(base, Edit(kind="budget", value=budget), flow))
    assert c["workflow"]["control"]["budget"] == budget
    path = ("workflow", "control", "budget")
    assert _without(c, path) == _without(base, path)


# 6: drop


def test_drop_critic():
    flow = _example()
    base = flow.configuration
    c = _ok(edits.apply(base, Edit(kind="drop", piece="critic"), flow))
    w = c["workflow"]
    assert _ids(c) == ["scout", "extract", "synthesize"]
    assert _artifacts(c) == ["question", "sources", "claims", "brief"]
    assert w["edges"] == BASE_EDGES[:7]
    assert w["control"] == {"gates": ["extract"], "repair": {"extract": "extract"}, "budget": 2, "rescue": RESCUE,
                            "ext": {RULES: {"extract": "command:claims-check"}}}
    assert list(c["settings"]) == ["scout", "extract", "synthesize"]


def test_drop_rewires_consumers_and_repair():
    flow = _example()
    c = _ok(edits.apply(flow.configuration, Edit(kind="drop", piece="synthesize"), flow))
    w = c["workflow"]
    assert _artifacts(c) == ["question", "sources", "claims", "verdict"]
    assert w["edges"] == BASE_EDGES[:4] + [["question", "critic"], ["claims", "critic"], ["critic", "verdict"]]
    assert [e for e in w["edges"] if e[1] == "critic"] == [["question", "critic"], ["claims", "critic"]]
    assert w["control"]["gates"] == ["extract", "critic"]
    assert w["control"]["repair"] == {"extract": "extract", "critic": "extract"}


def test_drop_with_no_producer_left_removes_the_repair_entry():
    flow = _flow(lambda f: f["configuration"]["workflow"]["control"]["repair"].update(critic="scout"))
    c = _ok(edits.apply(flow.configuration, Edit(kind="drop", piece="scout"), flow))
    w = c["workflow"]
    assert _artifacts(c) == ["question", "claims", "brief", "verdict"]
    assert w["edges"][0] == ["question", "extract"]
    assert ["sources", "extract"] not in w["edges"]
    assert w["control"]["gates"] == ["extract", "critic"]
    assert w["control"]["repair"] == {"extract": "extract"}


# 7: check_add


def test_check_add_writes_the_checks_form():
    added = [{"id": "sources-ok", "after": "scout", "rule": "review_approve"},
             {"id": "claims-tests", "after": "extract", "rule": "tests_pass"}]
    flow = _flow(lambda f: f["allow"].update(checks={"add": added}))
    base = flow.configuration
    first = _ok(edits.apply(base, Edit(kind="check_add", value=added[0]), flow))
    assert first["workflow"]["control"] == {
        "checks": [{"id": "extract", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
                   {"id": "critic", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"},
                   added[0]],
        "budget": 2,
        "rescue": RESCUE,
    }
    assert _without(first, ("workflow", "control")) == _without(base, ("workflow", "control"))
    second = _ok(edits.apply(base, Edit(kind="check_add", value=added[1]), flow))
    assert [c["after"] for c in second["workflow"]["control"]["checks"]] == ["extract", "critic", "extract"]
    assert len({base["id"], first["id"], second["id"]}) == 3
    arms, excluded = edits.candidates(flow)
    assert excluded == []
    assert [a.id for a in arms if a.id.startswith("check-add")] == ["check-add-sources-ok", "check-add-claims-tests"]


# 8: check_remove


def test_check_remove_drops_a_gate_and_refuses_the_required_one():
    flow = _flow(lambda f: f["allow"].update(checks={"remove": ["critic", "extract"]}))
    c = _ok(edits.apply(flow.configuration, Edit(kind="check_remove", value="critic"), flow))
    assert c["workflow"]["control"] == {"gates": ["extract"], "repair": {"extract": "extract"}, "budget": 2,
                                        "rescue": RESCUE, "ext": {RULES: {"extract": "command:claims-check"}}}
    assert c["workflow"]["pieces"] == flow.configuration["workflow"]["pieces"]
    arms, excluded = edits.candidates(flow)
    assert "check-remove-critic" in [a.id for a in arms]
    assert [(x.edit.kind, x.edit.value) for x in excluded] == [("check_remove", "extract")]
    assert "claims-check" in excluded[0].reason


def test_check_remove_in_the_checks_form():
    flow = _flow(_checks_form)
    c = _ok(edits.apply(flow.configuration, Edit(kind="check_remove", value="brief-format"), flow))
    assert [x["id"] for x in c["workflow"]["control"]["checks"]] == ["claims", "critic-review"]


# 9: merge


def test_merge_extract_synthesize():
    flow = _example()
    c = _ok(edits.apply(flow.configuration, Edit(kind="merge", value=MERGE), flow))
    w = c["workflow"]
    edges = w["edges"]
    assert [e[0] for e in edges if e[1] == "extract_synthesize"] == ["sources", "question"]
    assert [e[1] for e in edges if e[0] == "extract_synthesize"] == ["claims", "brief"]
    assert w["control"]["gates"] == ["extract_synthesize", "critic"]
    assert w["control"]["repair"] == {"extract_synthesize": "extract_synthesize", "critic": "extract_synthesize"}
    assert w["control"]["ext"] == {RULES: {"extract_synthesize": "command:claims-check"}}
    assert w["pieces"] == [{"id": "scout", "role": "worker", "width": 1},
                           {"id": "extract_synthesize", "role": "worker", "width": 1},
                           {"id": "critic", "role": "reviewer", "width": 1}]
    assert _artifacts(c) == ["question", "sources", "claims", "brief", "verdict"]
    assert edges == [["question", "scout"], ["scout", "sources"], ["sources", "extract_synthesize"],
                     ["question", "extract_synthesize"], ["extract_synthesize", "claims"],
                     ["extract_synthesize", "brief"], ["brief", "critic"], ["claims", "critic"], ["critic", "verdict"]]
    assert list(c["settings"]) == ["scout", "extract_synthesize", "critic"]
    assert c["settings"]["extract_synthesize"] == flow.allow.merges[0].setting
    assert edits.removed_checks(flow.configuration, Edit(kind="merge", value=MERGE), flow) == []


# 10: candidates on the example


def test_candidates_on_the_example():
    flow = _example()
    base_id = flow.configuration["id"]
    arms, excluded = edits.candidates(flow)
    assert excluded == []
    assert [a.id for a in arms] == EXAMPLE_ARMS
    ids = [a.configuration["id"] for a in arms]
    assert len(set(ids)) == len(ids) == 8 and base_id not in ids
    for arm in arms:
        _ok(arm.configuration)
        assert len(arm.edits) == 1
        assert arm.configuration == edits.apply(flow.configuration, arm.edits[0], flow)
    assert [a.edits[0] for a in arms] == [
        Edit(kind="model", piece="extract", value=SONNET), Edit(kind="model", piece="extract", value=HAIKU),
        Edit(kind="effort", piece="synthesize", value="medium"), Edit(kind="drop", piece="critic"),
        Edit(kind="width", piece="scout", value=2), Edit(kind="budget", value=1), Edit(kind="budget", value=3),
        Edit(kind="merge", value=MERGE)]
    assert all(any(a.edits == (e,) for a in arms) for e in hypothesis_edits(flow))


# 11: exclusions


def test_drop_of_the_required_check_is_excluded():
    flow = _flow(lambda f: f["allow"]["pieces"]["extract"].update(drop=True))
    arms, excluded = edits.candidates(flow)
    assert [(x.edit.kind, x.edit.piece) for x in excluded] == [("drop", "extract")]
    assert "claims-check" in excluded[0].reason
    assert "drop-extract" not in [a.id for a in arms]


def test_merge_that_drops_the_required_check_is_excluded():
    flow = _flow(lambda f: f["configuration"]["workflow"]["edges"].remove(["claims", "critic"]))
    edit = Edit(kind="merge", value=MERGE)
    assert edits.removed_checks(flow.configuration, edit, flow) == [("extract", "command:claims-check")]
    arms, excluded = edits.candidates(flow)
    assert [x.edit for x in excluded] == [edit]
    assert "claims-check" in excluded[0].reason
    assert "merge-extract-synthesize" not in [a.id for a in arms]


def test_a_repeated_configuration_is_excluded():
    flow = _flow(lambda f: f["allow"]["pieces"]["extract"]["models"].append(SONNET))
    arms, excluded = edits.candidates(flow)
    assert [a.id for a in arms] == EXAMPLE_ARMS
    assert [(x.edit, x.reason) for x in excluded] == [
        (Edit(kind="model", piece="extract", value=SONNET), "same configuration as arm extract-claude-sonnet-5-5")]


def test_removed_checks_of_drop_critic():
    flow = _example()
    assert edits.removed_checks(flow.configuration, Edit(kind="drop", piece="critic"), flow) == [
        ("critic", "review_approve")]


# 12: refusals


def _nested(flow: dict) -> None:
    flow["configuration"]["workflow"]["pieces"].append({"id": "sub", "workflow": {"ref": "solo", "version": 1}})


def _no_setting(flow: dict) -> None:
    del flow["configuration"]["settings"]["extract"]


def _artifact(vid: str):
    return lambda f: f["configuration"]["workflow"]["artifacts"].append({"id": vid, "kind": "other"})


def _ref(flow: dict) -> None:
    flow["configuration"]["workflow"] = {"ref": "no-such-workflow", "version": 1}


REFUSALS = [
    (None, Edit(kind="model", piece="nope", value=SONNET), "'nope' is no piece"),
    (_nested, Edit(kind="effort", piece="sub", value="low"), "no setting"),
    (_no_setting, Edit(kind="model", piece="extract", value=SONNET), "no setting"),
    (None, Edit(kind="model", piece="extract", value="claude-sonnet-5-5"), "not a model reference"),
    (None, Edit(kind="option", piece="extract", value="v3"), "names its key"),
    (None, Edit(kind="merge", value="x+y"), r"no merge 'x\+y'"),
    (None, Edit(kind="check_remove", value="nope"), "no check 'nope'"),
    (None, Edit(kind="width", piece="extract", value=2), "select"),
    (None, Edit(kind="width", piece="scout", value=0), "at least 1"),
    (None, Edit(kind="width", piece="scout", value="2"), "at least 1"),
    (None, Edit(kind="budget", value=0), "at least 1"),
    (None, Edit(kind="budget", value=True), "at least 1"),
    (_artifact("select_scout"), Edit(kind="width", piece="scout", value=2), "'select_scout' is already"),
    (_artifact("sources.cands"), Edit(kind="width", piece="scout", value=2), "'sources.cands' is already"),
    (_artifact("extract_synthesize"), Edit(kind="merge", value=MERGE), "'extract_synthesize' is already"),
    (None, Edit(kind="check_add", value={"id": "x", "after": "nope", "rule": "tests_pass"}), "'nope'"),
    (None, Edit(kind="check_add", value="x"), "not a check object"),
    (None, Edit(kind="rename", piece="extract"), "unknown edit kind 'rename'"),
    (_ref, Edit(kind="budget", value=1), "'no-such-workflow' does not resolve"),
]


@pytest.mark.parametrize("change, edit, message", REFUSALS, ids=[f"{e.kind}-{i}" for i, (_, e, _m) in
                                                                   enumerate(REFUSALS)])
def test_apply_refuses_what_it_cannot_do(change, edit, message):
    flow = _example() if change is None else _flow(change)
    with pytest.raises(ValueError, match=message):
        edits.apply(flow.configuration, edit, flow)


def test_merge_refuses_a_chain_that_is_gone():
    flow = _example()
    dropped = edits.apply(flow.configuration, Edit(kind="drop", piece="synthesize"), flow)
    with pytest.raises(ValueError, match="chain piece 'synthesize'"):
        edits.apply(dropped, Edit(kind="merge", value=MERGE), flow)


def test_a_nested_catalog_workflow_is_hashed_through_the_catalog():
    flow = _flow(_nested)
    c = edits.apply(flow.configuration, Edit(kind="model", piece="extract", value=SONNET), flow)
    assert c["workflow"]["pieces"][-1] == {"id": "sub", "workflow": {"ref": "solo", "version": 1}}
    assert c["id"] == config_id(c["workflow"], c["settings"], resolve=catalog_resolver) != flow.configuration["id"]


def test_apply_refuses_a_configuration_that_is_not_an_object():
    flow = _example()
    with pytest.raises(ValueError, match="not an object"):
        edits.apply("cfg_9bc4f6e876f9", Edit(kind="budget", value=1), flow)
    with pytest.raises(ValueError, match="not an Edit"):
        edits.apply(flow.configuration, {"kind": "budget", "value": 1}, flow)


# 13: no mutation


def test_apply_and_candidates_do_not_change_their_inputs():
    flow = _flow(lambda f: f["allow"].update(checks={"add": [{"id": "x", "after": "scout", "rule": "tests_pass"}],
                                                     "remove": ["critic"]}))
    before_flow, before_cfg = copy.deepcopy(flow.to_dict()), copy.deepcopy(flow.configuration)
    arms, _ = edits.candidates(flow)
    for arm in arms:
        edits.apply(flow.configuration, arm.edits[0], flow)
        edits.removed_checks(flow.configuration, arm.edits[0], flow)
    assert flow.to_dict() == before_flow and flow.configuration == before_cfg
    two = edits.apply(flow.configuration, Edit(kind="width", piece="scout", value=2), flow)
    before_two = copy.deepcopy(two)
    edits.apply(two, Edit(kind="width", piece="scout", value=1), flow)
    edits.apply(two, Edit(kind="drop", piece="scout"), flow)
    assert two == before_two


# 14: a workflow given by reference


def test_a_reference_workflow_is_resolved_inline():
    inline = catalog_resolver("implement_review", 1)
    settings = {p["id"]: {"harness": "api", "model": SONNET} for p in inline["pieces"]}
    ref = {"ref": "implement_review", "version": 1}
    configuration = {"id": config_id(ref, settings, resolve=catalog_resolver), "workflow": ref,
                     "settings": settings, "source": "usual"}
    c = _ok(edits.apply(configuration, Edit(kind="budget", value=3), _example()))
    assert c["workflow"] == {**inline, "control": {**inline["control"], "budget": 3}}
    assert c["settings"] == settings
    assert configuration["workflow"] == ref


# 15: required check names survive moves


def test_width_keeps_a_required_gate_name():
    def change(f):
        _gate_after_scout(f)
        f["acceptance_rule"]["requires"] = ["claims-check", "scout"]

    flow = _flow(change)
    c = _ok(edits.apply(flow.configuration, Edit(kind="width", piece="scout", value=2), flow))
    assert c["workflow"]["control"] == {
        "checks": [{"id": "scout", "after": "select_scout", "rule": "review_approve", "on_fail": "scout"},
                   {"id": "extract", "after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
                   {"id": "critic", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}],
        "budget": 2,
        "rescue": RESCUE,
    }
    arms, excluded = edits.candidates(flow)
    assert "scout-width-2" in [a.id for a in arms] and excluded == []


def test_merge_keeps_a_required_gate_name():
    flow = _flow(lambda f: f["acceptance_rule"].update(requires=["claims-check", "extract"]))
    c = _ok(edits.apply(flow.configuration, Edit(kind="merge", value=MERGE), flow))
    assert c["workflow"]["control"]["checks"] == [
        {"id": "extract", "after": "extract_synthesize", "rule": "command:claims-check",
         "on_fail": "extract_synthesize"},
        {"id": "critic", "after": "critic", "rule": "review_approve", "on_fail": "extract_synthesize"}]
    arms, excluded = edits.candidates(flow)
    assert "merge-extract-synthesize" in [a.id for a in arms] and excluded == []


def test_merge_in_the_checks_form_keeps_ids_ext_and_null_on_fail():
    flow = _flow(_checks_form)
    c = _ok(edits.apply(flow.configuration, Edit(kind="merge", value=MERGE), flow))
    assert c["workflow"]["control"] == {
        "checks": [
            {"id": "claims", "after": "extract_synthesize", "rule": "command:claims-check",
             "on_fail": "extract_synthesize", "ext": {"dev.example.note": "kept"}},
            {"id": "brief-format", "after": "extract_synthesize", "rule": "tests_pass", "on_fail": None},
            {"id": "critic-review", "after": "critic", "rule": "review_approve", "on_fail": "extract_synthesize"},
        ],
        "budget": 2,
        "rescue": RESCUE,
    }
    arms, excluded = edits.candidates(flow)
    assert "merge-extract-synthesize" in [a.id for a in arms] and excluded == []


# 16: no silent collapse of checks


def test_merge_does_not_collapse_two_checks():
    def change(f):
        f["configuration"]["workflow"]["control"] = {
            "checks": [{"id": "a", "after": "extract", "rule": "review_approve", "on_fail": "extract"},
                       {"id": "b", "after": "synthesize", "rule": "review_approve", "on_fail": "extract"},
                       {"id": "critic", "after": "critic", "rule": "review_approve", "on_fail": "synthesize"}],
            "budget": 2, "rescue": RESCUE}
        f["acceptance_rule"]["requires"] = ["a"]

    flow = _flow(change)
    edit = Edit(kind="merge", value=MERGE)
    c = edits.apply(flow.configuration, edit, flow)
    assert [x["id"] for x in c["workflow"]["control"]["checks"]] == ["a", "b", "critic"]
    assert "E217" in [f.code for f in edits.validate(c)]
    arms, excluded = edits.candidates(flow)
    assert [x.edit for x in excluded] == [edit]
    assert "E217" in excluded[0].reason
    assert "merge-extract-synthesize" not in [a.id for a in arms]


# 17: arm id bounds


def _plan_with(arms: list) -> Plan:
    data = json.loads(PLAN_SAMPLE.read_text(encoding="utf-8"))
    data["arms"] = [data["arms"][0]] + [a.to_dict() for a in arms]
    return Plan.from_dict(data)


def test_colliding_slugs_get_a_suffix():
    flow = _flow(lambda f: f["allow"]["pieces"]["synthesize"].update(options={"prompt": ["v 4", "v-4"]}))
    arms, excluded = edits.candidates(flow)
    assert excluded == []
    assert [a.id for a in arms if "prompt" in a.id] == ["synthesize-prompt-v-4", "synthesize-prompt-v-4-2"]
    assert len(_plan_with(arms).arms) == len(arms) + 1


def test_long_ids_stay_within_the_limit():
    key = "k" * 60
    flow = _flow(lambda f: f["allow"]["pieces"]["synthesize"].update(options={key: ["v 4", "v-4"]}))
    arms, excluded = edits.candidates(flow)
    assert excluded == []
    first, second = [a.id for a in arms if "kkk" in a.id]
    assert len(first) == 64 and first.startswith("synthesize-kkk") and first[54] != "-" and first[55] == "-"
    assert second == first[:62].rstrip("._-") + "-2" and len(second) <= 64
    assert all(ARM_ID.fullmatch(a.id) and a.id not in ("base", "unplanned") for a in arms)
    assert len(_plan_with(arms).arms) == len(arms) + 1


def test_example_arm_ids_pass_the_plan_check():
    arms, _ = edits.candidates(_example())
    assert [a.id for a in _plan_with(arms).arms[1:]] == EXAMPLE_ARMS
