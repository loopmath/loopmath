"""Curve, rescue, default pick and score rules (spec 05 sections 2 and 8)."""

from __future__ import annotations

import pytest

from loopmath.recommend import curve as C
from loopmath.recommend.engine import Settings, recommend, shown_labels
from loopmath.types import DEFAULT_RULE, Interval

from recommend_fakes import IR, PIR, TASK, FakeBelief, Num, cfg, score_rule, solo

LUNA, FABLE, SOL, OPUSX = solo("luna"), solo("fable"), solo("sol"), solo("opusx")
IRC = cfg(IR, implement="opus", review="astra")
PIRC = cfg(PIR, plan="opusx", implement="opus", review="astra")
ALL = [LUNA, FABLE, SOL, OPUSX, IRC, PIRC]


def belief(**over) -> FakeBelief:
    nums = {LUNA.id: Num(0.55, 0.30), FABLE.id: Num(0.72, 0.60), SOL.id: Num(0.82, 1.00),
            OPUSX.id: Num(0.74, 1.50), IRC.id: Num(0.91, 2.20), PIRC.id: Num(0.96, 4.00)}
    nums.update(over)
    return FakeBelief(nums)


def preds(b: FakeBelief, rescue_usd: float = 1.0):
    return b.predict_many(TASK, ALL, rescue_usd=rescue_usd)


def test_rows_pick_the_known_cheapest_configuration_per_level():
    rows = C.curve(preds(belief()))
    got = {r.levels: r.config for r in rows}
    assert got == {(50,): LUNA.id, (70,): FABLE.id, (80,): SOL.id, (90,): IRC.id, (95,): PIRC.id, (99,): None}
    last = rows[-1]
    assert last.reached is False and last.uncertain is True and last.prediction is None
    assert all(r.reached for r in rows[:-1])


def test_rows_with_the_same_configuration_merge():
    b = belief()
    ps = [p for p in preds(b) if p.config != LUNA.id]
    rows = C.curve(ps)
    assert rows[0].levels == (50, 70) and rows[0].config == FABLE.id
    assert [r.levels for r in rows] == [(50, 70), (80,), (90,), (95,), (99,)]


def test_uncertain_flags_match_the_draw_share():
    draws = {SOL.id: [0.79] * 25 + [0.85] * 75,  # 75 percent reach 0.80: uncertain
             IRC.id: [0.89] * 15 + [0.93] * 85,  # 85 percent reach 0.90: certain
             LUNA.id: [0.6] * 100, FABLE.id: [0.75] * 100, PIRC.id: [0.94] * 30 + [0.97] * 70,
             OPUSX.id: [0.74] * 100}

    def share(p, x):
        d = draws[p.config]
        return sum(v >= x for v in d) / len(d)

    rows = {r.levels: r for r in C.curve(preds(belief()), share=share)}
    assert rows[(80,)].uncertain is True
    assert rows[(90,)].uncertain is False
    assert rows[(95,)].uncertain is True  # 70 percent of draws reach 0.95
    assert rows[(50,)].uncertain is False


def test_interval_share_flags_a_wide_chance_as_uncertain():
    b = belief(**{SOL.id: Num(0.82, 1.00, g_half=0.15), IRC.id: Num(0.91, 2.20, g_half=0.01)})
    rows = {r.levels: r for r in C.curve(preds(b))}
    assert rows[(80,)].uncertain is True
    assert rows[(90,)].uncertain is False


def test_redo_usual_keeps_the_formula_below_one_percent_and_refuses_zero():
    b = belief(**{IRC.id: Num(0.001, 1.00, g_half=0.0005)})
    r = C.rescue_cost("redo_usual", b.predict(TASK, IRC))
    assert r.usd == pytest.approx(1000.0)  # $1 / 0.001, not $1 / 0.01
    b = belief(**{IRC.id: Num(0.0, 1.00, g_half=0.0)})
    with pytest.raises(C.NoFiniteRescue, match="set rescue.kind to person or none"):
        C.rescue_cost("redo_usual", b.predict(TASK, IRC))
    assert issubclass(C.NoFiniteRescue, ValueError)


def test_redo_usual_rescue_is_expected_cost_over_chance():
    b = belief()
    usual = b.predict(TASK, IRC)
    r = C.rescue_cost("redo_usual", usual)
    assert r.usd == pytest.approx(2.20 / 0.91)
    assert r.tokens == pytest.approx(2.20 * 180_000 / 0.91)
    assert C.rescue_cost("person", usual, person_usd_per_hour=150.0).usd == 150.0
    assert C.rescue_cost("person", usual, person_usd_per_hour=150.0, hours=2).usd == 300.0
    assert C.rescue_cost("none", usual).usd == 0.0
    with pytest.raises(ValueError):
        C.rescue_cost("person", usual)
    with pytest.raises(ValueError):
        C.rescue_cost("bogus", usual)


def test_changing_the_rescue_kind_changes_the_default_pick():
    cheap, sure = solo("luna"), cfg(IR, implement="opus", review="astra")
    b = FakeBelief({cheap.id: Num(0.50, 0.50), sure.id: Num(0.95, 2.00)})
    configs = [(cheap, "catalog"), (sure, "catalog")]
    redo = recommend(b, TASK, DEFAULT_RULE, usual=sure, usual_from="history", configs=configs)
    assert redo.rescue.usd == pytest.approx(2.0 / 0.95)
    assert redo.default.config.id == cheap.id  # 0.5 + 0.5 * 2.105 < 2.0 + 0.05 * 2.105
    person = recommend(b, TASK, DEFAULT_RULE, usual=sure, usual_from="history", configs=configs,
                       settings=Settings(rescue_kind="person", person_usd_per_hour=200.0))
    assert person.default.config.id == sure.id  # 0.5 + 0.5 * 200 > 2.0 + 0.05 * 200
    none = recommend(b, TASK, DEFAULT_RULE, usual=sure, usual_from="history", configs=configs,
                     settings=Settings(rescue_kind="none"))
    assert none.default.config.id == cheap.id
    assert none.default.prediction.ell.usd.mean == pytest.approx(0.5)


def test_rescue_usd_is_passed_to_the_belief():
    b = belief()
    rec = recommend(b, TASK, DEFAULT_RULE, usual=IRC, usual_from="history", configs=[(c, "catalog") for c in ALL])
    kinds = [c for c in b.calls if c[0] == "predict_many"]
    assert kinds and kinds[0][2] == pytest.approx(rec.rescue.usd)
    looks = [c for c in b.calls if c[0] == "lookahead"]
    assert looks and all(x[2] == pytest.approx(rec.rescue.usd) for x in looks)


def test_score_rule_ranks_by_p_reach():
    rule = score_rule("runtime_s", "<=", 200.0)
    fast, slow, mid = solo("sol"), solo("opusx"), solo("fable")
    # The success head says the slow one is the surest; the score rule must ignore that.
    b = FakeBelief({fast.id: Num(0.30, 1.0, score=150.0, p_reach=0.93),
                    slow.id: Num(0.95, 1.0, score=260.0, p_reach=0.20),
                    mid.id: Num(0.95, 1.0, score=195.0, p_reach=0.55)},
                   score_name="runtime_s", score_unit="s")
    rec = recommend(b, TASK, rule, usual=slow, usual_from="history", configs=[(fast, "catalog"), (mid, "catalog")])
    assert rec.usual.prediction.p_success.mean == pytest.approx(0.20)
    assert [c.config.id for c in rec.candidates] == [fast.id, mid.id, slow.id]
    assert rec.default.config.id == fast.id
    assert rec.curve[0].levels == (50, 70, 80, 90) and rec.curve[0].config == fast.id
    assert "chance of reaching runtime_s <= 200" in rec.message
    alt = rec.payload()["alternatives"]
    # every configuration but the goal and the usual (spec 05 section 3, D89)
    assert [a["config"]["id"] for a in alt] == [mid.id]
    assert alt[0]["deltas"]["score"] == pytest.approx(-65.0)
    assert alt[0]["prediction"]["scores"]["runtime_s"]["value"]["mean"] == 195.0


def test_equal_labels_get_what_tells_them_apart():
    """D89: `label()` leaves out harnesses and settings such as the round limit."""
    import dataclasses

    from loopmath.types import Configuration, Setting
    from loopmath.workflows.ids import config_id

    from recommend_fakes import MODELS

    a = cfg(IR, implement="opus", review="astra")
    s = {"implement": Setting("codex", "claude-opus-5-5", "high"), "review": MODELS["astra"]}
    b = Configuration(config_id(IR, s), IR, s)  # the same models by another harness
    ir4 = dataclasses.replace(IR, control=dataclasses.replace(IR.control, budget_rounds=4))
    c = cfg(ir4, implement="opus", review="astra")  # the same harnesses, another round limit
    luna = solo("luna")
    assert a.label() == b.label() == c.label() and len({a.id, b.id, c.id}) == 3
    labels = shown_labels([a, b, c, luna, a])
    assert labels[b.id] == f"{a.label()} (harnesses codex, codex)"
    assert (labels[a.id], labels[c.id]) == (f"{a.label()} [{a.id}]", f"{a.label()} [{c.id}]")
    assert labels[luna.id] == luna.label()
    assert shown_labels([a, b])[a.id] == f"{a.label()} (harnesses claude-code, codex)"


def test_goal_rows_and_fallback():
    b = belief()
    configs = [(c, "catalog") for c in ALL]
    rec = recommend(b, TASK, DEFAULT_RULE, usual=IRC, usual_from="history", configs=configs,
                    settings=Settings(goal="p80"))
    assert rec.goal.config.id == SOL.id and rec.goal_level == 80 and rec.goal_note is None
    rec = recommend(b, TASK, DEFAULT_RULE, usual=IRC, usual_from="history", configs=configs,
                    settings=Settings(goal="p99"))
    assert rec.goal.config.id == PIRC.id and rec.goal_level == 95
    assert "no workflow reaches 99%" in rec.goal_note
    rec = recommend(b, TASK, DEFAULT_RULE, usual=IRC, usual_from="history", configs=configs)
    assert rec.goal.config.id == rec.default.config.id and rec.goal_level is None
    with pytest.raises(ValueError):
        C.parse_goal("p85")


def test_p_at_least_is_monotone_and_bounded():
    iv = Interval(0.8, 0.7, 0.88)
    vals = [C.p_at_least(iv, x) for x in (0.5, 0.7, 0.8, 0.88, 0.95)]
    assert vals == sorted(vals, reverse=True)
    assert vals[1] == pytest.approx(0.9, abs=0.01) and vals[3] == pytest.approx(0.1, abs=0.01)
    assert C.p_at_least(iv, 0.0) == 1.0 and C.p_at_least(iv, 1.0) == 0.0
