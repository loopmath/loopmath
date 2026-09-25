"""Recorded configurations as candidates and the efforts each model offers (spec 05 section 1, 0.1.1 lane 1A).

The RQ1 test (F3, F5): designed runs at codex `max`, a planner with 3 workers and 3 workers without a selector
were never candidates, because candidates came from the usual, its edits, workflow files and the catalog only.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

from loopmath.types import Configuration, Control, Piece, Setting, Task, Workflow
from loopmath.workflows.candidates import (allowed_from, candidates, catalog_configurations, one_step_edits,
                                          twin_key)
from loopmath.workflows.ids import make_config
from loopmath.workflows.models import efforts_for, sort_efforts

TASK = Task("tsk_1", "feature", "ale-bench")
SOL_MAX = Setting("codex", "gpt-5.6-sol", "max")
LUNA_XHIGH = Setting("codex", "gpt-5.6-luna", "xhigh")
LUNA_LOW = Setting("codex", "gpt-5.6-luna", "low")

# 3 workers and no selector, and 1 planner plus 3 workers: the RQ1 shapes the catalog does not have
BEST3 = Workflow(id="best_of_n", version=1, title="Three implementers",
                 pieces=(Piece("implement", "implementer", width=3),), artifacts=("patch",),
                 edges=(("implement", "patch"),), control=Control())
PLAN3 = Workflow(id="plan_implement", version=1, title="Plan, then 3 workers",
                 pieces=(Piece("plan", "planner"), Piece("work", "worker", width=3)),
                 artifacts=("plan_doc", "patch"),
                 edges=(("plan", "plan_doc"), ("plan_doc", "work"), ("work", "patch")), control=Control())


def recorded() -> list[Configuration]:
    return [make_config(BEST3, {"implement": SOL_MAX}),
            make_config(PLAN3, {"plan": SOL_MAX, "work": LUNA_XHIGH})]


def test_codex_offers_max_for_the_gpt_5_6_and_gpt_6_models_but_not_older_ones():
    assert efforts_for("codex") == ("low", "medium", "high", "xhigh", "max")
    for model in ("gpt-5.6-sol", "gpt-5.6-luna", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"):
        assert efforts_for("codex", model=model)[-1] == "max"
    for model in ("gpt-5.5", "gpt-5.4", "gpt-5.3-codex"):
        assert efforts_for("codex", model=model) == ("low", "medium", "high", "xhigh")
    assert "ultra" not in efforts_for("codex")  # delegates to more agents, so it changes the shape
    assert efforts_for("codex", {"codex": ("low", "high")}, model="gpt-6-sol") == ("low", "high")
    assert sort_efforts(["max", "low", "odd", "high", "low"]) == ("low", "high", "max", "odd")


def test_every_recorded_effort_is_offered_for_its_model_even_under_an_override():
    al = allowed_from({"efforts": {"codex": ["low", "high"]}}, seen=[SOL_MAX, LUNA_LOW])
    assert al.efforts_of("gpt-5.6-sol") == ("low", "high", "max")
    assert al.efforts_of("gpt-5.6-luna") == ("low", "high")
    old = Setting("codex", "gpt-5.5", "max")  # recorded at an effort the default list leaves out
    assert allowed_from({}, seen=[old]).efforts_of("gpt-5.5")[-1] == "max"
    efforts = Counter(c.settings["implement"].effort for c in catalog_configurations(al)
                      if c.workflow.id == "solo" and c.settings["implement"].model == "gpt-5.6-sol")
    assert set(efforts) == {"low", "high", "max"}


def test_recorded_configurations_are_candidates_with_their_own_shape_and_their_edits():
    rec = recorded()
    out = candidates(TASK, usual=None, allowed={}, recorded=rec)
    origins = dict((c.id, o) for c, o in out)
    assert [origins[c.id] for c in rec] == ["recorded", "recorded"]
    kept = {c.id: c for c, _ in out}
    assert kept[rec[0].id].workflow.pieces[0].width == 3 and len(kept[rec[0].id].workflow.pieces) == 1
    assert [p.width for p in kept[rec[1].id].workflow.pieces] == [1, 3]
    # the usual's edit set applies to each recorded configuration: effort, model and width steps
    edits = {c.id for c in one_step_edits(rec[0], allowed_from({}, seen=[SOL_MAX, LUNA_XHIGH]))}
    assert edits and edits <= {c.id for c, o in out if o == "edit"}
    widths = {c.workflow.pieces[0].width for c, o in out if o == "edit" and c.workflow.id == "best_of_n"
              and len(c.workflow.pieces) == 1}
    assert {2, 4} <= widths
    # with no habit, the recorded models and efforts set the catalog's settings
    models = {s.model for c, o in out if o == "catalog" for s in c.settings.values()}
    assert models == {"gpt-5.6-sol", "gpt-5.6-luna"}
    assert any(s.effort == "max" for c, o in out if o == "catalog" for s in c.settings.values())


def test_recorded_configurations_are_never_cut():
    rec = recorded()
    out = candidates(TASK, usual=None, allowed={}, recorded=rec, limit=3)
    assert len(out) == 3 and [o for _, o in out][:2] == ["recorded", "recorded"]
    out = candidates(TASK, usual=None, allowed={}, recorded=rec, limit=1)
    assert [o for _, o in out] == ["recorded", "recorded"]


def test_a_recorded_configuration_that_is_also_the_usual_keeps_origin_usual():
    rec = recorded()
    out = candidates(TASK, usual=rec[0], allowed={}, recorded=rec)
    assert out[0] == (rec[0], "usual") and Counter(o for _, o in out)["recorded"] == 1


def test_a_catalog_or_edit_twin_of_a_recorded_configuration_is_dropped():
    # an import wrote solo without the repo input and with rescue "none": another id, the same workflow to a user
    imported = Workflow(id="solo", version=1, title="A07: solo (luna xhigh)",
                        pieces=(Piece("implement", "implementer"),), artifacts=("diff",),
                        edges=(("implement", "diff"),), control=Control(rescue="none"))
    rec = make_config(imported, {"implement": LUNA_XHIGH})
    out = candidates(TASK, usual=None, allowed={}, recorded=[rec])

    def solos(effort):
        return [(c.id, o) for c, o in out if c.workflow.id == "solo" and c.settings["implement"].model == "gpt-5.6-luna"
                and c.settings["implement"].effort == effort]

    assert solos("xhigh") == [(rec.id, "recorded")]
    assert [o for _, o in solos("medium")] == ["edit"]  # the edit keeps the recorded graph; its catalog twin goes
    assert any(o == "catalog" and c.workflow.id != "solo" for c, o in out)
    # a different round cap is a different workflow: both stay
    capped = dataclasses.replace(imported, control=Control(rescue="none", budget_rounds=5))
    assert twin_key(make_config(capped, {"implement": LUNA_XHIGH})) != twin_key(rec)


def test_a_different_gate_failure_route_is_not_a_twin():
    # review finding (c676ad4): stop on failure and repair on failure are different runs, both stay
    from loopmath.workflows.format import catalog

    wf = catalog()["implement_review"]
    high = Setting("codex", "gpt-5.6-sol", "high")
    repair = make_config(wf, {p.id: high for p in wf.pieces})
    assert any(g.on_fail for g in wf.control.gates)
    stop_wf = dataclasses.replace(wf, control=dataclasses.replace(
        wf.control, gates=tuple(dataclasses.replace(g, on_fail=None) for g in wf.control.gates)))
    stop = make_config(stop_wf, {p.id: high for p in wf.pieces})
    assert twin_key(stop) != twin_key(repair)
    allowed = {"models": ["gpt-5.6-sol"], "efforts": {"codex": ["high"]}}
    assert repair.id in {c.id for c in catalog_configurations(allowed_from(allowed, seen=[high]))}
    out = dict((c.id, o) for c, o in candidates(TASK, usual=None, allowed=allowed, recorded=[stop]))
    assert out[stop.id] == "recorded" and out[repair.id] == "catalog"
    # the cosmetic twin still goes: same gates and route, only the title and the workflow's rescue field differ
    renamed = dataclasses.replace(wf, title="imported", control=dataclasses.replace(wf.control, rescue="none"))
    rec = make_config(renamed, {p.id: high for p in wf.pieces})
    assert rec.id != repair.id and twin_key(rec) == twin_key(repair)
    out = dict((c.id, o) for c, o in candidates(TASK, usual=None, allowed=allowed, recorded=[rec]))
    assert out[rec.id] == "recorded" and repair.id not in out
