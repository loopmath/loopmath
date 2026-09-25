"""The strategy sentence (spec 05 section 4): when the goal pick is less likely to succeed than the reference and
misses are priced, the paragraph says to try the pick first and rescue its misses, with the numbers of
`goal.strategy`."""

from __future__ import annotations

import pytest

from loopmath.recommend import message as M
from loopmath.recommend.engine import Settings, recommend
from loopmath.types import DEFAULT_RULE

from recommend_fakes import IR, TASK, FakeBelief, Num, cfg, solo

USUAL = cfg(IR, implement="opus", review="astra")
CHEAP = solo("luna")  # less likely, much cheaper: the lowest cost per accepted result
MID = solo("sol")


def nums(**over) -> dict:
    out = {USUAL.id: Num(0.80, 2.00), CHEAP.id: Num(0.60, 0.40), MID.id: Num(0.70, 1.50)}
    out.update(over)
    return out


def run(settings: Settings | None = None, usual_from: str = "history", **over):
    return recommend(FakeBelief(nums(**over)), TASK, DEFAULT_RULE, usual=USUAL, usual_from=usual_from,
                     configs=[(c, "catalog") for c in (CHEAP, MID)], settings=settings or Settings())


def pred(config, g: float, usd: float, rescue: float):
    return FakeBelief({config.id: Num(g, usd)}).predict(TASK, config, None, rescue)


@pytest.mark.parametrize("g, want", [(0.70, (3, 10, "about 3 in 10 tasks")), (0.96, (1, 25, "about 1 in 25 tasks")),
                                     (0.97, (1, 33, "about 1 in 33 tasks")),
                                     (0.02, (98, 100, "about 98 in 100 tasks"))])
def test_misses_are_read_as_n_in_10_or_1_in_m_or_n_in_100(g, want):
    assert M.misses(g) == want


def test_the_sentence_says_try_first_then_rescue_with_the_numbers():
    rec = run()
    assert rec.goal.config.id == CHEAP.id and rec.rescue.kind == "redo_usual"
    s = rec.strategy
    label = rec.label(CHEAP)
    assert s["text"] == (f"Try {label} first; if it misses (about 4 in 10 tasks), your usual workflow rescues it; "
                         f"expected $1.40 per accepted result, against $2.50 with your usual workflow alone.")
    assert s["kind"] == "try_then_rescue" and s["misses"] == {"n": 4, "of": 10, "share": 0.4}
    assert s["pick"] == {"config": CHEAP.id, "label": label}
    assert s["reference"] == {"config": USUAL.id, "kind": "usual", "name": "your usual workflow"}
    assert s["chance"] == {"pick": 0.6, "reference": 0.8}
    assert s["rescue"] == {"kind": "redo_usual", "usd": pytest.approx(2.5)}


def test_the_text_is_in_the_paragraph_and_the_numbers_are_the_goals():
    rec = run()
    out = rec.payload()
    s = out["goal"]["strategy"]
    assert s == rec.strategy and out["choices"][0]["strategy"] == s
    assert f"Your goal is the lowest expected cost of an accepted result. {s['text']}" in rec.message
    goal = next(c for c in rec.candidates if c.config.id == s["pick"]["config"])
    n = rec.numbers(goal.prediction)
    assert s["run_cost_usd"] == n["run_cost_usd"]["mean"]
    assert s["expected_rescue_usd"] == n["expected_rescue_usd"]
    assert s["cost_per_accepted_usd"] == n["cost_per_accepted_usd"]["mean"]
    assert s["reference_cost_per_accepted_usd"] == out["reference"]["numbers"]["cost_per_accepted_usd"]["mean"]
    assert s["run_cost_usd"] + s["expected_rescue_usd"] == pytest.approx(s["cost_per_accepted_usd"])


def test_a_person_rescue_names_the_person_and_the_price():
    rec = run(Settings(rescue_kind="person", person_usd_per_hour=60.0, rescue_hours=0.05))
    s = rec.strategy
    assert s is not None and s["rescue"]["kind"] == "person"
    assert (f"if it misses (about 4 in 10 tasks), a person finishes it for about $3.00; expected $1.60 per accepted "
            f"result, against $2.60 with your usual workflow.") in s["text"]
    assert s["text"] in rec.message


def test_a_reference_that_is_not_the_usual_is_named_as_such():
    rec = run(usual_from="recorded")
    assert rec.reference_kind == "best_recorded"
    s = rec.strategy
    assert s["reference"]["name"] == "your best recorded workflow"
    assert "your best recorded workflow rescues it" in s["text"] and s["text"].endswith(
        "with your best recorded workflow alone.")


def test_a_level_goal_is_followed_by_the_sentence():
    rec = run(Settings(goal="p70"))
    assert rec.goal_level == 70 and rec.goal.config.id == MID.id
    s = rec.strategy
    assert s is not None and "if it misses (about 3 in 10 tasks)" in s["text"]
    assert f"Your goal is the 70% row: {rec.label(MID)} at about $1.50. {s['text']}" in rec.message


@pytest.mark.parametrize("case", ["pick_is_reference", "no_rescue", "same_shown_chance"])
def test_no_sentence_when_it_does_not_apply(case):
    if case == "pick_is_reference":
        rec = run(**{CHEAP.id: Num(0.60, 2.40), MID.id: Num(0.70, 2.40)})
        assert rec.goal.config.id == USUAL.id
    elif case == "no_rescue":
        rec = run(Settings(rescue_kind="none"))
    else:
        rec = run(**{CHEAP.id: Num(0.796, 0.40), USUAL.id: Num(0.804, 2.00)})
        assert rec.goal.config.id == CHEAP.id
    assert rec.strategy is None and rec.payload()["goal"]["strategy"] is None
    assert " first; if it misses " not in rec.message


def test_strategy_sentence_and_strategy_agree_without_the_engine():
    p, r = pred(CHEAP, 0.97, 0.40, 2.0), pred(USUAL, 0.99, 2.00, 2.0)
    s = M.strategy(pick=CHEAP, pick_pred=p, pick_label="Luna solo", ref=USUAL, ref_pred=r, reference="default",
                   rescue_kind="redo_usual", rescue_usd=2.0)
    assert s["misses"] == {"n": 1, "of": 33, "share": 0.03}
    assert s["text"] == M.strategy_sentence("Luna solo", p, r, reference="default", rescue_kind="redo_usual",
                                            rescue_usd=2.0)
    assert s["text"] == ("Try Luna solo first; if it misses (about 1 in 33 tasks), the default workflow rescues it; "
                         "expected $0.46 per accepted result, against $2.02 with the default workflow alone.")
