"""Human diff lines (lane 04)."""

from __future__ import annotations

import dataclasses
import json

from loopmath.cli import main
from loopmath.types import Setting
from loopmath.workflows.candidates import one_step_edits
from loopmath.workflows.diff import diff, diff_workflows
from loopmath.workflows.format import catalog
from loopmath.workflows.ids import make_config
from loopmath.workflows.ocp import configuration_to_ocp

OPUS = Setting("claude-code", "claude-opus-5-5", "high")
ASTRA = Setting("codex", "gpt-6-astra", "xhigh")
FABLE = Setting("claude-code", "claude-fable-5-1", "medium")


def _usual():
    return make_config(catalog()["implement_review"], {"implement": OPUS, "review": ASTRA})


def test_same_configuration_has_no_lines():
    assert diff(_usual(), _usual()) == ()


def test_lines_match_the_fixture_style():
    usual = _usual()
    cheap = make_config(catalog()["solo"], {"implement": FABLE})
    assert diff(usual, cheap) == ("shape: implement_review to solo",
                                  "implementer: claude-opus-5-5/high to claude-fable-5-1/medium",
                                  "reviewer removed")
    xhigh = make_config(usual.workflow, {"implement": dataclasses.replace(OPUS, effort="xhigh"), "review": ASTRA})
    assert diff(usual, xhigh) == ("implementer effort: high to xhigh",)
    swap = make_config(usual.workflow, {"implement": Setting("codex", "gpt-6-sol", "high"),
                                        "review": dataclasses.replace(OPUS)})
    assert diff(usual, swap) == ("implementer: claude-opus-5-5/high to gpt-6-sol/high",
                                 "reviewer: gpt-6-astra/xhigh to claude-opus-5-5/high")
    planned = make_config(catalog()["plan_implement_review"],
                          {"plan": dataclasses.replace(OPUS, effort="xhigh"), "implement": OPUS, "review": ASTRA})
    assert diff(usual, planned) == ("shape: implement_review to plan_implement_review",
                                    "planner added: claude-opus-5-5/xhigh")


def test_width_rounds_gates_rescue_and_context_lines():
    wf = catalog()["swarm"]
    s = {"plan": OPUS, "work": OPUS, "review": ASTRA}
    a = make_config(wf, s)
    wider = dataclasses.replace(wf, pieces=(wf.pieces[0], dataclasses.replace(wf.pieces[1], width=4), wf.pieces[2]))
    ctl = dataclasses.replace(wf.control, budget_rounds=4, rescue="person",
                              gates=(dataclasses.replace(wf.control.gates[0], rule="command:lint"),))
    b = make_config(dataclasses.replace(wider, control=ctl), {**s, "work": dataclasses.replace(OPUS, context_policy="inherit")})
    assert diff(a, b) == ("worker context: fresh to inherit", "worker width: 3 to 4", "round limit: 3 to 4",
                          "reviewer gate: review_approve to command:lint", "rescue: redo_usual to person")
    stop = dataclasses.replace(wf.control, budget_rounds=1, gates=(dataclasses.replace(wf.control.gates[0], on_fail=None),))
    c = make_config(dataclasses.replace(wf, control=stop), s)
    assert diff(a, c) == ("reviewer gate on fail: work to stop",)  # rounds without repair do not matter
    assert "round limit: 3 to 1" in diff(a, c, detail=True)


def test_added_parallel_pieces_show_their_width_and_same_role_pieces_their_ids():
    solo = make_config(catalog()["solo"], {"implement": OPUS})
    swarm = make_config(catalog()["swarm"], {"plan": OPUS, "work": OPUS, "review": ASTRA})
    assert diff(solo, swarm) == ("shape: solo to swarm", "planner added: claude-opus-5-5/high",
                                 "worker added: claude-opus-5-5/high x3", "reviewer added: gpt-6-astra/xhigh",
                                 "implementer removed")
    wf = catalog()["implement_review"]
    two = dataclasses.replace(wf, pieces=wf.pieces + (dataclasses.replace(wf.pieces[1], id="review2"),),
                              edges=wf.edges + (("diff", "review2"),))
    assert diff_workflows(wf, two)[0] == "review2 (reviewer) added"


def test_detail_adds_artifacts_and_edges():
    a, b = catalog()["implement_review"], catalog()["plan_implement_review"]
    lines = diff_workflows(a, b, detail=True)
    assert "artifact added: plan_doc (plan)" in lines
    assert "edge added: plan_doc -> implement" in lines
    assert "edge removed" not in " ".join(lines)
    kinds = dataclasses.replace(a, extra={"artifact_kinds": {**a.extra["artifact_kinds"], "verdict": "review"}})
    s = {"implement": OPUS, "review": ASTRA}
    assert diff(make_config(a, s), make_config(kinds, s)) == ("artifact verdict kind: verdict to review",)


def test_every_one_step_edit_has_a_line():
    usual = _usual()
    for edit in one_step_edits(usual, {"models": ["claude-opus-5-5", "gpt-6-astra", "claude-fable-5-1"]}):
        lines = diff(usual, edit)
        assert lines and not any("changed (same" in x for x in lines), edit.workflow.id


def test_detail_names_settings_that_only_one_side_has(tmp_path, capsys):
    # a bare shape against a configuration of it is not "no differences"
    wf = catalog()["implement_review"]
    s = {"implement": OPUS, "review": ASTRA}
    assert diff_workflows(wf, wf, settings_b=s) == ()  # recommend's lines stay as they were
    assert diff_workflows(wf, wf, detail=True, settings_b=s) == (
        "implementer: unset to claude-opus-5-5/high", "reviewer: unset to gpt-6-astra/xhigh")
    assert diff_workflows(wf, wf, detail=True, settings_a={"implement": OPUS}) == ("implementer: claude-opus-5-5/high to unset",)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(configuration_to_ocp(make_config(wf, s))), encoding="utf-8")
    assert main(["workflows", "diff", "implement_review", str(path), "--home", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no differences" not in out and "implementer: unset to claude-opus-5-5/high" in out
