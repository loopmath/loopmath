"""Exploration picks, payback, budget pause, pair and message (spec 05 sections 4, 5 and 8)."""

from __future__ import annotations

import pytest

from loopmath.recommend import gain as Gm
from loopmath.recommend.engine import Settings, recommend
from loopmath.types import DEFAULT_RULE

from recommend_fakes import IR, PIR, TASK, FakeBelief, Num, cfg, score_rule, solo

USUAL = cfg(IR, implement="opus", review="astra")
GOAL = cfg(PIR, plan="opusx", implement="opus", review="astra")
CHEAP = solo("luna")  # cheap, modest gain
COSTLY = solo("opusx")  # costly, large gain
OTHER = solo("fable")  # no gain
MID = solo("sol")  # between


def nums() -> dict:
    return {USUAL.id: Num(0.80, 2.00), GOAL.id: Num(0.90, 3.00), CHEAP.id: Num(0.55, 0.40),
            COSTLY.id: Num(0.85, 5.00), OTHER.id: Num(0.60, 0.80), MID.id: Num(0.70, 1.50)}


def gains() -> dict:
    return {CHEAP.id: {"usd": 0.30, "success_pp": 0.0, "cost_pct": -30.0, "score": None},
            COSTLY.id: {"usd": 1.00, "success_pp": 6.0, "cost_pct": 4.0, "score": None},
            MID.id: {"usd": 0.40, "success_pp": 2.0, "cost_pct": -5.0, "score": None},
            OTHER.id: {"usd": 0.0, "success_pp": 0.0, "cost_pct": 0.0, "score": None}}


BEATS = {CHEAP.id: 0.3, COSTLY.id: 0.45, MID.id: 0.35, OTHER.id: 0.1}
CONFIGS = [(c, "catalog") for c in (GOAL, CHEAP, COSTLY, OTHER, MID)]


def run(settings: Settings | None = None, **kw):
    b = FakeBelief(nums(), gains(), dict(BEATS), **kw)
    st = settings or Settings(goal="p90")
    return recommend(b, TASK, DEFAULT_RULE, usual=USUAL, usual_from="history", configs=CONFIGS, settings=st)


def test_best_value_prefers_cheap_low_gain_and_max_gain_the_costly_high_gain():
    rec = run()
    assert rec.goal.config.id == GOAL.id
    ex = rec.exploration
    assert ex.method == "lookahead" and ex.note is None
    bv, mg = ex.best_value, ex.max_gain
    assert bv.state == "pick" and bv.pick.kind == "best_value" and bv.pick.candidate.config.id == CHEAP.id
    assert mg.state == "pick" and mg.pick.kind == "max_gain" and mg.pick.candidate.config.id == COSTLY.id
    assert bv.pick.payback_runs == pytest.approx(0.40 / 0.30)
    assert mg.pick.payback_runs == pytest.approx(5.00 / 1.00)
    assert bv.pick.price.usd.mean == pytest.approx(0.40)
    assert bv.pick.price.tokens.mean == pytest.approx(0.40 * 180_000)
    assert bv.pick.p_beats_goal == pytest.approx(0.3)
    # runner-ups by each pick's own criterion; OTHER has no gain and does not qualify
    assert [c.config.id for c in bv.pick.runner_ups] == [MID.id, COSTLY.id]
    assert [c.config.id for c in mg.pick.runner_ups] == [MID.id, CHEAP.id]
    out = ex.to_dict()
    assert out["best_value"]["kind"] == "best_value" and out["max_gain"]["kind"] == "max_gain"
    assert out["method"] == "lookahead"


def test_one_candidate_winning_both_makes_max_gain_same_as():
    g = gains()
    g[COSTLY.id] = {"usd": 0.10, "success_pp": 1.0, "cost_pct": 0.0, "score": None}
    g[MID.id] = {"usd": 0.05, "success_pp": 1.0, "cost_pct": 0.0, "score": None}
    g[CHEAP.id] = {"usd": 0.50, "success_pp": 0.0, "cost_pct": -40.0, "score": None}
    b = FakeBelief(nums(), g, dict(BEATS))
    rec = recommend(b, TASK, DEFAULT_RULE, usual=USUAL, usual_from="history", configs=CONFIGS,
                    settings=Settings(goal="p90"))
    assert rec.exploration.best_value.pick.candidate.config.id == CHEAP.id
    assert rec.exploration.max_gain.state == "same_as"
    assert rec.exploration.to_dict()["max_gain"] == {"same_as": "best_value"}
    assert "biggest gain" not in rec.message


def test_payback_is_price_over_gain_and_zero_gain_does_not_qualify():
    rec = run()
    for s in rec.exploration.screened:
        if s.G > 0:
            assert s.payback == pytest.approx(s.price.usd.mean / s.G)
        else:
            assert s.payback is None
    ids = {s.config_id for s in rec.exploration.screened}
    assert GOAL.id not in ids  # the goal is never its own exploration


def test_gain_must_exceed_one_percent_of_the_goal_ell():
    g = {k: {"usd": 0.001, "success_pp": 0.0, "cost_pct": -1.0, "score": None} for k in gains()}
    b = FakeBelief(nums(), g, dict(BEATS))
    rec = recommend(b, TASK, DEFAULT_RULE, usual=USUAL, usual_from="history", configs=CONFIGS,
                    settings=Settings(goal="p90"))
    assert rec.exploration.best_value.state == "none"
    assert rec.exploration.to_dict()["best_value"] == {"none": "no candidate has positive gain"}
    assert rec.pair()["members"] == [GOAL.id]
    assert "No workflow is worth trying" in rec.message


def test_budget_cap_pauses_each_pick_on_its_own_price():
    # 9.0 spent of 10: only CHEAP (0.40) fits; COSTLY (5.00) and MID (1.50) would pass the cap,
    # so the biggest gain within the cap is CHEAP too, and max_gain is same_as.
    rec = run(Settings(goal="p90", spend=9.0, cap=10.0))
    assert rec.exploration.best_value.state == "pick"
    assert rec.exploration.max_gain.state == "same_as"
    # 9.8 spent: nothing fits; both picks pause and keep what they would have been
    rec = run(Settings(goal="p90", spend=9.8, cap=10.0))
    bv, mg = rec.exploration.to_dict()["best_value"], rec.exploration.to_dict()["max_gain"]
    assert bv["paused"] == "budget cap reached" and bv["would_have_been"]["candidate"]["config"]["id"] == CHEAP.id
    assert mg["paused"] == "budget cap reached" and mg["would_have_been"]["candidate"]["config"]["id"] == COSTLY.id
    assert rec.pair()["members"] == [GOAL.id] and "paused" in rec.pair()["note"]
    assert "passes your budget cap" in rec.message
    # 8.0 spent: CHEAP and MID fit, COSTLY does not; biggest gain is MID, best value stays CHEAP
    rec = run(Settings(goal="p90", spend=8.0, cap=10.0))
    assert rec.exploration.best_value.pick.candidate.config.id == CHEAP.id
    assert rec.exploration.max_gain.pick.candidate.config.id == MID.id
    assert rec.exploration.to_dict()["budget"] == {"cap_usd": 10.0, "spent_usd": 8.0, "remaining_usd": 2.0}


def test_best_value_paused_while_a_cheaper_pick_fits():
    g = gains()
    g[CHEAP.id] = {"usd": 0.05, "success_pp": 0.0, "cost_pct": -5.0, "score": None}  # payback 8
    g[MID.id] = {"usd": 0.60, "success_pp": 3.0, "cost_pct": 0.0, "score": None}  # payback 2.5
    b = FakeBelief(nums(), g, dict(BEATS))
    rec = recommend(b, TASK, DEFAULT_RULE, usual=USUAL, usual_from="history", configs=CONFIGS,
                    settings=Settings(goal="p90", spend=9.0, cap=10.0))
    assert rec.exploration.best_value.state == "paused"
    assert rec.exploration.best_value.pick.candidate.config.id == MID.id
    assert rec.exploration.max_gain.state == "pick"
    assert rec.exploration.max_gain.pick.candidate.config.id == CHEAP.id


def test_auto_ok_follows_the_standing_payback_rule():
    rec = run(Settings(goal="p90", auto_payback_runs=2))
    assert rec.exploration.best_value.pick.auto_ok is True  # 1.33 < 2
    assert rec.exploration.max_gain.pick.auto_ok is False  # 5 >= 2
    rec = run()
    assert rec.exploration.best_value.pick.auto_ok is False


def test_pair_uses_the_default_pick_setting():
    rec = run()
    pair = rec.pair()
    assert pair["members"] == [GOAL.id, CHEAP.id] and pair["explore_pick"] == "best_value"
    assert "separate worktrees" in pair["instructions"]
    rec = run(Settings(goal="p90", default_pick="max_gain"))
    assert rec.pair()["members"] == [GOAL.id, COSTLY.id] and rec.pair()["explore_pick"] == "max_gain"


def test_message_follows_the_template():
    rec = run()
    m = rec.message
    assert m.startswith("Your usual workflow (implement_review: claude-opus-5-5/high, gpt-6-astra/xhigh) "
                        "has an 80% chance of an accepted result at about $2.00 (360,000 tokens). "
                        "Your goal is the 90% row: plan_implement_review")
    assert ("Trying solo: gpt-6-luna/low alongside it costs $0.40 (72,000 tokens) now. There is a 30% chance "
            "it beats your goal. Trying it once is expected to save about $0.30 on each future similar run, so it "
            "pays for itself after about 1 similar run.") in m
    assert m.endswith(" The option with the biggest gain is solo: claude-opus-5-5/xhigh: it costs $5.00 "
                      "(900,000 tokens) now, has a 45% chance to beat your goal, is expected to save about $1.00 "
                      "per future similar run, and pays for itself after about 5 runs.")
    assert chr(0x2014) not in m


def test_the_gain_reads_as_an_expected_saving_not_a_conditional_one():
    """Dogfood (D96): "has a 1% chance to beat your goal, gains about 28 percent lower cost ($1.06) per future
    similar run if it does" read lane 05's expected gain as one on the condition that the pick wins, though payback
    divides the price by it as it stands; and the 28 percent was the best workflow's after learning, not the pick's."""
    rec = run()
    m = rec.message
    assert "if it does" not in m and "percent lower cost" not in m and "points of success" not in m
    for slot in (rec.exploration.best_value, rec.exploration.max_gain):
        pick = slot.pick
        assert f"expected to save about ${pick.gain_per_run['usd']:.2f}" in m
        assert pick.payback_runs == pytest.approx(pick.price.usd.mean / pick.gain_per_run["usd"])
        assert set(pick.gain_per_run) >= {"usd", "success_pp", "cost_pct"}  # the parts stay in the JSON


def test_the_article_before_a_percentage_is_the_one_read_aloud():
    """Dogfood: the message said "has a 88% chance"."""
    from loopmath.recommend.message import a_pct

    assert [a_pct(x / 100) for x in (8, 11, 18, 80, 88, 89)] == ["an 8%", "an 11%", "an 18%", "an 80%", "an 88%",
                                                              "an 89%"]
    assert [a_pct(x / 100) for x in (0, 1, 7, 9, 12, 50, 79, 90, 100)] == ["a 0%", "a 1%", "a 7%", "a 9%", "a 12%",
                                                                       "a 50%", "a 79%", "a 90%", "a 100%"]


def test_score_rule_message_names_the_target_and_only_the_dollar_saving():
    rule = score_rule("runtime_s", "<=", 200.0)
    n = nums()
    for k, v in n.items():
        n[k] = Num(v.g, v.usd, score=190.0, p_reach=v.g)
    g = gains()
    g[CHEAP.id] = dict(g[CHEAP.id], score=-12.0)
    b = FakeBelief(n, g, dict(BEATS), score_name="runtime_s", score_unit="s")
    rec = recommend(b, TASK, rule, usual=USUAL, usual_from="history", configs=CONFIGS, settings=Settings(goal="p90"))
    assert "has an 80% chance of reaching runtime_s <= 200 at about $2.00" in rec.message
    assert "expected to save about $0.30 on each future similar run" in rec.message
    assert "-12 s" not in rec.message  # the best workflow's score after learning, not the pick's (D96)
    assert rec.exploration.best_value.pick.gain_per_run["score"] == -12.0


def test_a_score_target_goal_is_priced_per_accepted_result_as_the_default_pick_is():
    """D109: with a score target the goal sentence said "about $X per success" while the Default pick line said
    "per accepted result" for the same number."""
    rule = score_rule("runtime_s", "<=", 200.0)
    b = FakeBelief({k: Num(v.g, v.usd, score=190.0, p_reach=v.g) for k, v in nums().items()}, gains(),
                   dict(BEATS), score_name="runtime_s", score_unit="s")
    rec = recommend(b, TASK, rule, usual=USUAL, usual_from="history", configs=CONFIGS, settings=Settings())
    assert rec.score_backed and rec.goal_level is None and rec.goal.config.id != USUAL.id
    ell = rec.goal.prediction.ell.usd.mean
    assert "Your goal is the lowest expected cost to reaching runtime_s <= 200: " in rec.message
    assert f", about ${ell:,.2f} per accepted result." in rec.message and "per success" not in rec.message


def test_a_score_rule_the_fit_cannot_predict_says_its_chances_are_of_an_accepted_result():
    """Dogfood: with no runtime_s scores the belief gives the success head, which the output called the chance
    of reaching runtime_s <= 200."""
    rule = score_rule("runtime_s", "<=", 200.0)
    b = FakeBelief(nums(), gains(), dict(BEATS), score_name="runtime_s", score_unit="s")  # no scores: success head
    rec = recommend(b, TASK, rule, usual=USUAL, usual_from="history", configs=CONFIGS)
    assert not rec.score_backed and rec.usual.prediction.success_from == "success_head"
    assert "has an 80% chance of an accepted result at about $2.00" in rec.message and "runtime_s" not in rec.message
    assert rec.notes[0] == ("the fit has too few runtime_s scores for this kind of task to predict runtime_s <= 200, "
                            "so each chance here is of an accepted result")
    backed = FakeBelief({k: Num(v.g, v.usd, score=190.0, p_reach=v.g) for k, v in nums().items()}, gains(),
                        dict(BEATS), score_name="runtime_s", score_unit="s")
    rec = recommend(backed, TASK, rule, usual=USUAL, usual_from="history", configs=CONFIGS)
    assert rec.score_backed and "chance of reaching runtime_s <= 200" in rec.message and not rec.notes


def test_screen_keeps_top_50_by_tenth_percentile_of_ell():
    rec = run()
    cands = rec.candidates
    top = Gm.screen(cands, GOAL.id, n=2)
    order = sorted([c for c in cands if c.config.id != GOAL.id],
                   key=lambda c: (c.prediction.ell.usd.lo, c.prediction.ell.usd.mean))
    assert [c.config.id for c in top] == [c.config.id for c in order[:2]]


def test_the_plain_diff_names_the_round_limit_as_lane_04_does():
    """The plan's diff lines (`simple_diff`) say `round limit`, which counts the first round (D30, D81)."""
    import dataclasses

    from loopmath.recommend.engine import simple_diff

    a = cfg(IR, implement="opus", review="astra")
    n = a.workflow.control.budget_rounds
    b = cfg(dataclasses.replace(IR, control=dataclasses.replace(IR.control, budget_rounds=n + 1)),
            implement="opus", review="astra")
    assert simple_diff(a, b) == (f"round limit: {n} to {n + 1}",)
