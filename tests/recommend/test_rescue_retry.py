"""The `retry` rescue (0.2.1, F7 and T2), its rescue workflow, `p_accepted_within`, `bands`, the `most_likely`
choice (I19), `option`, the score-target reference and the `rescue.*` config keys (spec 05 section 2)."""

from __future__ import annotations

import argparse

import pytest

from loopmath.recommend import curve as C
from loopmath.recommend import engine as E
from loopmath.recommend.commands import UserError, settings_from
from loopmath.recommend.engine import Settings, recommend
from loopmath.recommend.storeread import Conf
from loopmath.store.config import ConfigError, coerce, parse_rule
from loopmath.types import DEFAULT_RULE

from recommend_fakes import IR, TASK, FakeBelief, Num, cfg, solo

USUAL = cfg(IR, implement="opus", review="astra")  # 80%, $2.00
CHEAP = solo("luna")  # 60%, $0.40: the lowest cost per accepted result
MID = solo("sol")  # 70%, $1.50: the cheapest with at least a 70% chance, so the rescue workflow
SURE = cfg(IR, implement="opusx", review="astra")  # 95%, $6.00
TARGET = parse_rule("heldout_perf>=2400")


def nums(score: bool = False) -> dict:
    def n(g: float, usd: float) -> Num:
        return Num(g, usd, score=2500.0, p_reach=g) if score else Num(g, usd)

    return {USUAL.id: n(0.80, 2.00), CHEAP.id: n(0.60, 0.40), MID.id: n(0.70, 1.50), SURE.id: n(0.95, 6.00)}


def run(settings: Settings | None = None, rule=DEFAULT_RULE, usual_from: str = "history", sure: bool = False):
    configs = [(c, "catalog") for c in (CHEAP, MID, *((SURE,) if sure else ()))]
    return recommend(FakeBelief(nums(score=rule.score is not None)), TASK, rule, usual=USUAL,
                     usual_from=usual_from, configs=configs, settings=settings or Settings())


# ---------------------------------------------------------------- the formula
def test_hand_computed_values():
    # q 0.9: retries at 0.45 and 0.225; spend 1 + 0.55 = 1.55; fixed 1 - 0.55 x 0.775 = 0.57375
    usd, tokens, fixes = C.retry_rescue([0.9], [1.0], [10.0])
    assert usd == pytest.approx(1.55 / 0.57375) and tokens == pytest.approx(15.5 / 0.57375)
    assert fixes == pytest.approx(0.57375)
    # q 0.32, c 1.42 (A07 in 0.2.0): retries at 0.16 and 0.08; 1.42 x 1.84 / (1 - 0.84 x 0.92) = 11.5
    usd, _, fixes = C.retry_rescue([0.32], [1.42])
    assert usd == pytest.approx(11.5) and fixes == pytest.approx(0.2272)


def test_a_ratio_of_means_stays_finite_with_draws_at_zero():
    usd, _, fixes = C.retry_rescue([0.0, 0.9], [1.0, 1.0])
    assert usd == pytest.approx((2.0 + 1.55) / 2 / (0.57375 / 2)) and fixes == pytest.approx(0.57375 / 2)
    with pytest.raises(C.NoFiniteRescue, match="set rescue.kind to person or none"):
        C.retry_rescue([0.0, 0.0], [1.0, 1.0])


def test_no_decay_and_many_attempts_is_the_old_cost_over_chance():
    usd, _, fixes = C.retry_rescue([0.32], [1.42], decay=1.0, max_attempts=200)
    assert usd == pytest.approx(1.42 / 0.32, rel=1e-6) and fixes == pytest.approx(1.0)


def test_one_attempt_means_no_retries():
    assert C.retry_rescue([0.5], [1.0], max_attempts=1) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("kw", [{"decay": 0.0}, {"decay": 1.5}, {"max_attempts": 0}])
def test_bad_retry_settings_are_errors(kw):
    with pytest.raises(ValueError):
        C.retry_rescue([0.5], [1.0], **kw)


# ---------------------------------------------------------------- the recommender
def test_retry_is_the_default_and_rescues_with_the_cheapest_likely_workflow():
    rec = run()
    r = rec.rescue
    fixed = 1 - 0.65 * 0.825  # MID: retries at 0.35 and 0.175
    assert r.kind == "retry" and r.config == MID.id
    assert r.usd == pytest.approx(1.5 * 1.65 / fixed) and r.p_accepted == pytest.approx(fixed, abs=1e-6)
    out = rec.payload()["rescue"]
    assert out["of"] == "solo: gpt-6-sol/high" and out["config"] == MID.id
    assert out["chance"] == {"mean": 0.7, "lo": 0.65, "hi": 0.75} and out["run_cost_usd"] == 1.5
    assert (out["decay"], out["max_attempts"], out["min_chance"]) == (0.5, 3, 0.7)
    assert out["basis"] == ("the cheapest workflow with at least a 70% chance, retried: each extra attempt has "
                            "half the chance of the one before, up to 3 attempts in all")
    assert out["text"] == ("Run cost is what the agents cost for one run; cost per accepted result adds the "
                           "expected cost of fixing a miss with solo: gpt-6-sol/high, where each extra attempt has "
                           "half the chance of the one before, up to 3 attempts in all.")
    # ell keeps its form: run cost plus the chance of a miss times the rescue
    goal = rec.goal.prediction
    assert rec.goal.config.id == CHEAP.id
    assert goal.ell.usd.mean == pytest.approx(0.4 + 0.4 * r.usd)


def test_with_no_workflow_likely_enough_the_most_likely_one_rescues_and_basis_says_so():
    rec = run(Settings(rescue_min_chance=0.9))
    assert rec.rescue.config == USUAL.id
    assert rec.rescue.basis.startswith("no workflow has a 90% chance, so the most likely one, retried")


def test_p_accepted_within_is_the_chance_within_the_attempts():
    rec = run()
    fixed = rec.rescue.p_accepted
    within = rec.numbers(rec.goal.prediction)["p_accepted_within"]
    assert within["attempts"] == 3
    assert within["mean"] == pytest.approx(0.6 + 0.4 * fixed, abs=1e-6)
    assert within["lo"] == pytest.approx(0.55 + 0.45 * fixed, abs=1e-6)
    assert within["hi"] == pytest.approx(0.65 + 0.35 * fixed, abs=1e-6)
    one = run(Settings(rescue_max_attempts=1))
    assert one.rescue.usd == 0.0 and one.goal.prediction.ell.usd.mean == pytest.approx(0.4)
    w = one.numbers(one.goal.prediction)["p_accepted_within"]
    assert w == {"mean": 0.6, "lo": 0.55, "hi": 0.65, "attempts": 1}
    assert "a miss is not retried" in one.payload()["rescue"]["text"]


def test_other_kinds_keep_working_and_their_chance_within_is_one_run():
    rec = run(Settings(rescue_kind="redo_usual"))
    assert rec.rescue.kind == "redo_usual" and rec.rescue.usd == pytest.approx(2.0 / 0.8)
    out = rec.payload()["rescue"]
    assert out["of"] == "implement_review: claude-opus-5-5/high, gpt-6-astra/xhigh" and "config" not in out
    assert rec.numbers(rec.goal.prediction)["p_accepted_within"]["attempts"] == 1
    assert run(Settings(rescue_kind="none")).payload()["rescue"]["text"].endswith("is the run cost.")


def test_the_message_and_the_strategy_say_the_rescue_plainly():
    rec = run()
    assert rec.strategy["text"] == (
        "Try solo: gpt-6-luna/low first; if it misses (about 4 in 10 tasks), retry with solo: gpt-6-sol/high, "
        "up to 2 more attempts, each with half the chance of the one before; expected $2.53 per accepted result, "
        "against $3.07 with your usual workflow alone.")
    assert rec.strategy["rescue"]["of"] == "solo: gpt-6-sol/high"
    assert ("fixing a miss with solo: gpt-6-sol/high, where each extra attempt has half the chance of the one "
            "before, up to 3 attempts in all. The recommended workflow has a 60% chance of an accepted result in "
            "one run, and a 79% chance within 3 attempts.") in rec.message


# ---------------------------------------------------------------- choices, bands, option
def test_choices_are_numbered_and_carry_the_new_fields():
    rec = run()
    choices = rec.choices()
    assert [c["option"] for c in choices] == list(range(1, len(choices) + 1))
    for c in choices:
        assert set(c["p_accepted_within"]) == {"mean", "lo", "hi", "attempts"}
        # a belief without draws gives the 80% band from its interval; the other levels are absent
        assert set(c["bands"]) == {"chance", "run_cost_usd", "cost_per_accepted_usd"}
        assert all(list(b) == ["80"] for b in c["bands"].values())
    goal = choices[0]
    assert goal["key"] == "goal" and goal["bands"]["chance"]["80"] == [0.55, 0.65]
    assert goal["bands"]["run_cost_usd"]["80"] == [pytest.approx(0.28), pytest.approx(0.52)]
    assert "most_likely" not in {c["key"] for c in choices}  # no score target


def test_a_score_target_adds_the_most_likely_choice():
    rec = run(rule=TARGET, sure=True)
    choices = rec.choices()
    keys = [c["key"] for c in choices]
    assert keys[-1] == "most_likely" and len(choices) <= 5
    likely = choices[-1]
    assert likely["config"] == SURE.id and likely["title"] == "Run the workflow most likely to reach the target"
    assert likely["chance"]["mean"] == 0.95 and likely["option"] == len(choices)


def test_the_most_likely_choice_is_skipped_when_it_is_already_a_choice():
    rec = run(rule=TARGET)  # the usual is the most likely and is the reference choice
    keys = [c["key"] for c in rec.choices()]
    assert "reference" in keys and "most_likely" not in keys


def test_band_set_reads_central_intervals_from_draws():
    b = E.band_set([i / 100 for i in range(101)], [1.0] * 101, list(range(101)))
    assert b["chance"] == {"50": [0.25, 0.75], "80": [0.1, 0.9], "90": [0.05, 0.95], "95": [0.025, 0.975]}
    assert b["run_cost_usd"]["95"] == [1.0, 1.0] and b["cost_per_accepted_usd"]["80"] == [10.0, 90.0]


# ---------------------------------------------------------------- the reference
def test_with_a_score_target_the_reference_is_the_most_likely_recorded_workflow():
    b = FakeBelief(nums(score=True))
    assert E.best_recorded(b, TASK, TARGET, [CHEAP, MID, USUAL]).id == USUAL.id
    plain = FakeBelief(nums())
    assert E.best_recorded(plain, TASK, DEFAULT_RULE, [CHEAP, MID, USUAL]).id == CHEAP.id  # lowest E[C] / g


# ---------------------------------------------------------------- config
@pytest.mark.parametrize("key,value", [("rescue.decay", "0"), ("rescue.decay", "1.5"), ("rescue.max_attempts", "0"),
                                       ("rescue.max_attempts", "2.5"), ("rescue.min_chance", "1.2"),
                                       ("rescue.decay", "half")])
def test_bad_rescue_values_are_refused(key, value):
    with pytest.raises(ConfigError, match=key):
        coerce(key, value)


def test_good_rescue_values_are_kept():
    assert coerce("rescue.kind", "retry") == "retry"
    assert coerce("rescue.decay", "1") == 1 and coerce("rescue.max_attempts", "5") == 5
    assert coerce("rescue.min_chance", "0.8") == 0.8


def test_settings_read_and_check_the_rescue_keys():
    args = argparse.Namespace(goal=None)
    st = settings_from(Conf({}), args, 0.0, None)
    assert (st.rescue_kind, st.rescue_decay, st.rescue_max_attempts, st.rescue_min_chance) == ("retry", 0.5, 3, 0.7)
    st = settings_from(Conf({"rescue": {"kind": "retry", "decay": 0.25, "max_attempts": 4, "min_chance": 0.8}}),
                       args, 0.0, None)
    assert (st.rescue_decay, st.rescue_max_attempts, st.rescue_min_chance) == (0.25, 4, 0.8)
    with pytest.raises(UserError, match="rescue.max_attempts"):
        settings_from(Conf({"rescue": {"max_attempts": 0}}), args, 0.0, None)
    with pytest.raises(UserError, match="rescue.kind"):
        settings_from(Conf({"rescue": {"kind": "forever"}}), args, 0.0, None)
