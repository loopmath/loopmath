"""The search space's settings come from the candidates (spec 05 section 1a): every allowed (model, effort) must
appear in at least one catalog or edit candidate, so the search sees every allowed setting and `--models` limits
it."""

from __future__ import annotations

import pytest

from loopmath.recommend.search import setting_key, space_settings
from loopmath.types import Setting, Task
from loopmath.workflows.candidates import allowed_from, candidates
from loopmath.workflows.format import catalog
from loopmath.workflows.ids import make_config

TASK = Task("tsk_1", "bugfix", "me/repo")
OPUS = Setting("claude-code", "claude-opus-5-5", "high")
ASTRA = Setting("codex", "gpt-6-astra", "high")
SIX = ["claude-opus-5-5", "claude-sonnet-5", "claude-fable-5-1", "gpt-6-astra", "gpt-6-sol", "gpt-5.6-luna"]
USUAL = make_config(catalog()["implement_review"], {"implement": OPUS, "review": ASTRA})


@pytest.mark.parametrize("models", [SIX[:1], SIX[:2], SIX[3:5], SIX])
def test_every_allowed_setting_is_in_some_generated_candidate(models):
    al = allowed_from({"models": models}, [OPUS, ASTRA])
    want = {setting_key(s) for s in al.settings()}
    out = candidates(TASK, usual=USUAL, allowed={"models": models})
    generated = {setting_key(s) for c, o in out if o in ("catalog", "edit") for s in c.settings.values()}
    assert want <= generated
    space = {setting_key(s) for s in space_settings(out)}
    assert want <= space
    usual_models = {setting_key(s)[1] for s in USUAL.settings.values()}
    assert {k[1] for k in space} <= set(models) | usual_models  # the usual's edits keep its other pieces' models
