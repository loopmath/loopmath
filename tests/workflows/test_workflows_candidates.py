"""Candidates and one-step edits (spec 05 section 1, lane 04)."""

from __future__ import annotations

import dataclasses
from collections import Counter

from loopmath.types import Setting, Task
from loopmath.workflows.candidates import (Allowed, allowed_from, candidates, catalog_configurations, one_step_edits,
                                           settings_for_shape, settings_like, working_setting)
from loopmath.workflows.format import catalog, dump_workflow
from loopmath.workflows.ids import make_config
from loopmath.workflows.models import family_of
from loopmath.workflows.shapes import shape_params

TASK = Task("tsk_1", "bugfix", "me/repo")
OPUS = Setting("claude-code", "claude-opus-5-5", "high")
ASTRA = Setting("codex", "gpt-6-astra", "high")
SIX = ["claude-opus-5-5", "claude-sonnet-5", "claude-fable-5-1", "gpt-6-astra", "gpt-6-sol", "gpt-5.6-luna"]


def _usual():
    wf = catalog()["implement_review"]
    return make_config(wf, {"implement": OPUS, "review": ASTRA})


def test_allowed_reads_lane_06_dict_list_or_map_of_harnesses_and_efforts():
    al = allowed_from({"models": ["claude-opus-5-5", "gpt-6-astra", "gemini-3"], "harnesses": ["codex"],
                       "efforts": {"codex": ["low", "high"]}})
    assert al.models == ("gpt-6-astra",)
    assert [s.effort for s in al.settings()] == ["low", "high"]
    al = allowed_from({"models": ["gemini-3"], "harnesses": {"gemini-3": "gemini-cli"}})
    assert al.harness("gemini-3") == "gemini-cli" and al.efforts_of("gemini-3") == ("default",)
    al = allowed_from({}, seen=[OPUS, ASTRA])
    assert al.models == ("claude-opus-5-5", "gpt-6-astra")


def test_checkers_prefer_another_provider_and_never_the_same_family():
    al = allowed_from({"models": SIX})
    picks = [s.model for s in al.checkers(OPUS)]
    assert picks == ["gpt-6-astra", "gpt-6-sol", "gpt-5.6-luna"]
    picks = [s.model for s in al.checkers(Setting("codex", "gpt-6-astra", "xhigh"))]
    assert picks[0] == "claude-opus-5-5" and "gpt-6-astra" not in picks
    assert al.checkers(Setting("claude-code", "claude-opus-5-5", "max"))[0].effort == "xhigh"  # codex has no max
    assert allowed_from({"models": ["claude-opus-5-5"]}).checkers(OPUS) == [OPUS]


def test_catalog_configurations_are_linear_and_reviewers_differ_in_family():
    configs = catalog_configurations({"models": SIX})
    settings = 3 * 5 + 3 * 4
    assert len(configs) == settings * (1 + 3 + 1 + 3 + 3 + 3)
    assert len({c.id for c in configs}) == len(configs)
    for c in configs:
        work = working_setting(c)
        for p in c.workflow.pieces:
            if p.role in ("reviewer", "referee"):
                assert family_of(c.settings[p.id].model) != family_of(work.model)
            else:
                assert c.settings[p.id] == work


def test_candidates_dedupe_keep_usual_and_user_and_stay_bounded():
    usual = _usual()
    user_wf = dataclasses.replace(catalog()["plan_implement_review"], id="mine")
    out = candidates(TASK, usual=usual, allowed={"models": SIX}, user=[user_wf])
    ids = [c.id for c, _ in out]
    assert len(ids) == len(set(ids))
    assert out[0] == (usual, "usual")
    origins = Counter(o for _, o in out)
    assert origins["usual"] == 1 and origins["user"] == 1 and origins["edit"] > 10 and origins["catalog"] > 300
    user_cfg = next(c for c, o in out if o == "user")
    assert user_cfg.settings == {"plan": OPUS, "implement": OPUS, "review": ASTRA}
    assert len(out) < 2000
    cut = candidates(TASK, usual=usual, allowed={"models": SIX}, user=[user_wf], limit=40)
    assert len(cut) == 40 and cut[0][1] == "usual" and any(o == "user" for _, o in cut)
    assert Counter(o for _, o in cut)["edit"] == origins["edit"]


def test_candidates_without_usual_or_models():
    assert candidates(TASK, usual=None, allowed={}) == []
    out = candidates(TASK, usual=None, allowed={"models": ["claude-opus-5-5"]})
    assert {o for _, o in out} == {"catalog"} and len(out) == 5 * 6


def test_usual_found_in_the_catalog_keeps_origin_usual():
    usual = _usual()
    out = candidates(TASK, usual=usual, allowed={"models": ["claude-opus-5-5", "gpt-6-astra"]})
    assert [o for c, o in out if c.id == usual.id] == ["usual"]


def test_one_step_edits_cover_each_kind_of_step():
    usual = _usual()
    edits = one_step_edits(usual, {"models": ["claude-opus-5-5", "claude-sonnet-5", "gpt-6-astra"]})
    assert usual.id not in {c.id for c in edits}
    shapes = Counter(c.workflow.id for c in edits)
    assert shapes["solo"] == 1 and shapes["plan_implement_review"] == 1
    solo = next(c for c in edits if c.workflow.id == "solo")
    assert solo.id == make_config(catalog()["solo"], {"implement": OPUS}).id
    planned = next(c for c in edits if c.workflow.id == "plan_implement_review")
    assert planned.settings["plan"] == OPUS and planned.settings["review"] == ASTRA
    rounds = sorted(c.workflow.control.budget_rounds for c in edits if c.workflow.id == "implement_review"
                    and c.settings == usual.settings)
    assert rounds == [2, 4]
    models = {(p, c.settings[p].model) for c in edits for p in c.settings if c.settings[p] != usual.settings.get(p)}
    assert ("implement", "claude-sonnet-5") in models and ("review", "claude-opus-5-5") in models
    efforts = {c.settings["implement"].effort for c in edits if c.workflow == usual.workflow}
    assert efforts == {"low", "medium", "high", "xhigh", "max"}


def test_edits_add_a_reviewer_of_another_family_and_change_width():
    wf = catalog()["best_of_n"]
    cfg = make_config(wf, {"implement": OPUS, "select": ASTRA})
    edits = one_step_edits(cfg, {"models": ["claude-opus-5-5", "gpt-6-astra"]})
    added = next(c for c in edits if c.workflow.id == "best_of_n_review")
    assert added.settings["review"].model == "gpt-6-astra" and added.workflow.control.budget_rounds == 3
    widths = {c.workflow.pieces[0].width for c in edits if c.workflow.id == "best_of_n" and c.settings == cfg.settings}
    assert widths == {2, 4}


def test_edits_of_a_user_workflow_skip_structural_steps():
    wf = dataclasses.replace(catalog()["implement_review"], pieces=catalog()["implement_review"].pieces
                             + (catalog()["swarm"].pieces[0],))
    wf = dataclasses.replace(wf, edges=wf.edges + (("issue", "plan"),))
    cfg = make_config(wf, {"implement": OPUS, "review": ASTRA, "plan": OPUS})
    assert shape_params(wf) is None
    edits = one_step_edits(cfg, Allowed(("claude-opus-5-5",), {"claude-opus-5-5": "claude-code"}))
    assert all(c.workflow.pieces == wf.pieces for c in edits)
    assert sorted(c.workflow.control.budget_rounds for c in edits if c.settings == cfg.settings) == [2, 4]


def test_settings_like_and_store_workflows(tmp_path):
    usual = _usual()
    wf = catalog()["plan_implement_review"]
    assert settings_like(wf, usual) == {"plan": OPUS, "implement": OPUS, "review": ASTRA}
    assert settings_like(wf, None) is None
    assert settings_for_shape(wf, OPUS, ASTRA)["review"] == ASTRA
    folder = tmp_path / "workflows"
    folder.mkdir()
    (folder / "mine.toml").write_text(dump_workflow(dataclasses.replace(wf, id="mine"), {"plan": ASTRA}))
    (folder / "broken.toml").write_text("id = ")
    out = candidates(TASK, usual=usual, allowed={}, home=tmp_path)
    users = [c for c, o in out if o == "user"]
    assert len(users) == 1 and users[0].settings["plan"] == ASTRA and users[0].settings["implement"] == OPUS
