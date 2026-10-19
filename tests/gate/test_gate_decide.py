"""loopmath.gate.decide: the loop policy around loopmath.compare.paired.

The comparator is injected, so these tests pin the gate's own rules: when it may adopt, when
it rejects, how many more pairs it asks for, and what it reports. The statistics belong to
loopmath.compare.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _iv(mean, p80, p95):
    """An interval as compare reports it: lo and hi are the 80% range."""
    return {
        "mean": mean,
        "median": mean,
        "lo": p80[0],
        "hi": p80[1],
        "bands": {"80": list(p80), "95": list(p95)},
    }


def _cmp(
    *,
    n_pairs,
    mean=0.0,
    p80=(-1.0, 1.0),
    p95=(-2.0, 2.0),
    p_better=0.5,
    verdict="more",
    cost_base=1.0,
    cost_cand=1.0,
    next_pair_value_usd=10.0,
    floor=0.0,
    **extra,
):
    """A Comparison with the contract fields of loopmath.compare.paired (score rule)."""
    return SimpleNamespace(
        n_pairs=n_pairs,
        gain=_iv(mean, p80, p95),
        p_better=p_better,
        cost_base=_iv(cost_base, (cost_base,) * 2, (cost_base,) * 2),
        cost_cand=_iv(cost_cand, (cost_cand,) * 2, (cost_cand,) * 2),
        cpar_base=None,
        cpar_cand=None,
        verdict=verdict,
        next_pair_value_usd=next_pair_value_usd,
        reasons=["fake"],
        floor=floor,
        **extra,
    )


SCHEMA = "loopmath.harness-runs/0.1"


def _row(arm, score, usd=0.5):
    outcome = None if score is ... else {"name": "judge", "score": score, "judge": "fake"}
    return {"schema": SCHEMA, "task_id": "t1", "arm": arm, "outcome": outcome, "cost": {"usd": usd}}


def _runs(arm, scores, usd=0.5):
    return [_row(arm, s, usd) for s in scores]


class Recorder:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, base, cand, *, rule, budget_usd=None, **kw):
        self.calls.append({"base": base, "cand": cand, "rule": rule, "budget_usd": budget_usd, **kw})
        return self.result


def _decide(cmp, base_scores=(50, 52), cand_scores=(60, 61), **policy):
    from loopmath.gate import GatePolicy, decide

    rec = Recorder(cmp)
    d = decide(
        _runs("base", base_scores), _runs("cand", cand_scores), GatePolicy(**policy), compare=rec
    )
    return d, rec


def test_adopt_needs_every_condition():
    good = dict(n_pairs=4, mean=10, p80=(6, 14), p95=(4, 16), p_better=0.999, verdict="adopt")
    d, rec = _decide(_cmp(**good), base_scores=(50,) * 4, cand_scores=(60,) * 4)
    assert d.action == "adopt" and d.n_more == 0
    assert any("95%" in r for r in d.reasons)


def test_what_the_gate_asks_compare_for():
    _d, rec = _decide(_cmp(n_pairs=2))
    call = rec.calls[0]
    assert call["rule"] == "score:judge"
    # alpha 0.025 one-sided over all the gate's looks, and the cap fixed before the first look:
    # compare's threshold is calibrated for looks up to max_pairs.
    assert call["confidence"] == pytest.approx(0.975)
    assert call["max_pairs"] == 8
    _d, rec = _decide(_cmp(n_pairs=2), r_max=16)
    assert rec.calls[0]["max_pairs"] == 16
    # No floor given: compare's default applies (0 under a score rule), read back from c.floor.
    assert "floor" not in call
    _d, rec = _decide(_cmp(n_pairs=2, floor=2.0), floor=2.0)
    assert rec.calls[0]["floor"] == 2.0


@pytest.mark.parametrize(
    "change, why",
    [
        ({"n_pairs": 1}, "--r-min"),
        ({"verdict": "more"}, "compare"),
        ({"p95": (-0.5, 16)}, "floor"),
        ({"p_better": 0.99}, "P(better)"),
    ],
)
def test_adopt_is_withheld_when_one_condition_fails(change, why):
    good = dict(n_pairs=4, mean=10, p80=(6, 14), p95=(4, 16), p_better=0.999, verdict="adopt")
    good.update(change)
    n = good["n_pairs"]
    d, _ = _decide(_cmp(**good), base_scores=(50,) * n, cand_scores=(60,) * n)
    assert d.action != "adopt"
    assert any(why in r for r in d.reasons), d.reasons


def test_p_adopt_spends_five_percent_over_the_planned_looks():
    from loopmath.gate import GatePolicy, p_adopt

    # Looks at 1, 2, 4 and 8 pairs; adoption from r_min = 2 pairs, so it can adopt at 2, 4 and 8, and at
    # one final look off the plan: L = 4; 2.5% one-sided, split over the 4.
    assert GatePolicy(r_max=8).planned_looks() == 4
    assert GatePolicy(r_max=8).adopting_looks() == 4
    assert p_adopt(GatePolicy(r_max=8)) == pytest.approx(1 - 0.05 / 8)
    assert GatePolicy(r_max=16).planned_looks() == 5
    # With r_min = 1 the look at 1 pair can adopt too: L = 5.
    assert GatePolicy(r_min=1, r_max=8).adopting_looks() == 5


def test_reject_on_compare_reject_or_futility():
    d, _ = _decide(_cmp(n_pairs=2, verdict="reject", p_better=0.001))
    assert d.action == "reject"
    # Futility: the 80% range sits wholly below the floor.
    d, _ = _decide(_cmp(n_pairs=2, mean=-3, p80=(-5, -0.5), p95=(-7, 1)))
    assert d.action == "reject"
    assert any("80%" in r for r in d.reasons)


@pytest.mark.parametrize("n, more", [(1, 1), (2, 2), (3, 1), (4, 4), (6, 2)])
def test_replicate_up_to_the_next_look(n, more):
    d, _ = _decide(_cmp(n_pairs=n), base_scores=(50,) * n, cand_scores=(51,) * n, r_max=8)
    assert d.action == "replicate"
    assert d.n_more == more


def test_unresolved_at_the_cap_keeps_the_base():
    d, _ = _decide(_cmp(n_pairs=8), base_scores=(50,) * 8, cand_scores=(51,) * 8, r_max=8)
    assert d.action == "reject"
    assert any("--r-max 8" in r and "keeping the base" in r for r in d.reasons)
    d, _ = _decide(
        _cmp(n_pairs=8), base_scores=(50,) * 8, cand_scores=(51,) * 8, r_max=8, on_unresolved="adopt"
    )
    assert d.action == "adopt"


def test_budget_caps_the_replicates_and_is_passed_on():
    # compare prices a run at $0.50 an arm, so a pair at $1.00; $2.50 left buys 2 pairs, not
    # the 4 the next look wants.
    d, rec = _decide(
        _cmp(n_pairs=4, cost_base=0.5, cost_cand=0.5),
        base_scores=(50,) * 4,
        cand_scores=(51,) * 4,
        r_max=16,
        budget_usd=6.5,
    )
    assert d.action == "replicate" and d.n_more == 2
    # compare takes the whole cap and what is spent, and subtracts itself.
    assert rec.calls[0]["budget_usd"] == pytest.approx(6.5)
    assert rec.calls[0]["spent_usd"] == pytest.approx(4.0)
    assert d.cost["spent_usd"] == pytest.approx(4.0)
    assert d.cost["next_usd"] == pytest.approx(2.0)


@pytest.mark.parametrize("budget, sunk, base, cand, usd, cmp", [
    (1.0, 1.0, (), (), 0.5, None),  # the search rollouts spent the budget, no pair yet
    (0.5, 1.0, (), (), 0.5, None),  # the search overspent it
    (1.5, 1.0, (50,), (), 0.5, None),  # one priced base run and no candidate run: no comparison, no pair price
    (1.0, 2.0, (50, 52), (60, 61), 0.0, _cmp(n_pairs=2, cost_base=0.0, cost_cand=0.0)),  # free pairs
], ids=["spent", "overspent", "one-arm", "free-pairs"])
def test_a_spent_budget_buys_no_pair_whatever_a_pair_costs(budget, sunk, base, cand, usd, cmp):
    """D4 review round 4 (R4-B2): compare buys no new run once the budget left is zero or less, priced or not
    (D2 R3-B1, in `loopmath.compare.paired`). The gate keeps the same rule where compare is not asked
    (before the first confirmation, with the search cost sunk) or reports no stop of its own."""
    from loopmath.gate import GatePolicy, decide

    rec = Recorder(cmp or _cmp(n_pairs=0))
    d = decide(_runs("base", base, usd), _runs("cand", cand, usd), GatePolicy(budget_usd=budget), compare=rec,
               sunk_usd=sunk)
    assert len(rec.calls) == (cmp is not None)
    assert (d.action, d.n_more, d.stop) == ("reject", 0, "budget")
    assert d.cost["left_usd"] <= 0 and d.cost["next_usd"] == 0.0
    assert any(r.startswith("budget: the budget is spent") for r in d.reasons)


def test_a_budget_left_buys_the_first_pair_before_its_price_is_known():
    from loopmath.gate import GatePolicy, decide

    d = decide([], [], GatePolicy(budget_usd=2.0), compare=Recorder(_cmp(n_pairs=0)), sunk_usd=1.0)
    assert (d.action, d.n_more) == ("replicate", 1)


def test_value_of_the_next_pair_below_its_cost_stops():
    d, _ = _decide(_cmp(n_pairs=2, next_pair_value_usd=0.2))
    assert d.action == "reject"
    assert any("not worth" in r for r in d.reasons)


def test_no_value_for_the_next_pair_is_not_a_stop():
    # Under a score rule with no dollar value per point, compare reports no value.
    d, _ = _decide(_cmp(n_pairs=2, next_pair_value_usd=None))
    assert d.action == "replicate" and d.n_more == 2


def test_a_reject_for_cost_alone_is_unresolved():
    # compare keeps the base when the budget or the next pair's value runs out with
    # P(better) in between; the gate reports that as unresolved, not as evidence against.
    d, _ = _decide(_cmp(n_pairs=2, verdict="reject", p_better=0.6))
    assert d.action == "reject"
    assert any("keeping the base" in r for r in d.reasons)
    d, _ = _decide(_cmp(n_pairs=2, verdict="reject", p_better=0.6), on_unresolved="adopt")
    assert d.action == "adopt"
    d, _ = _decide(_cmp(n_pairs=2, verdict="reject", p_better=0.001), on_unresolved="adopt")
    assert d.action == "reject"


def test_floor_checks_read_the_objective_improvement_when_given():
    # Under an acceptance rule compare's objective is cost per accepted result, and its
    # improvement interval, not the gain in acceptance, is what the floor applies to.
    good = dict(n_pairs=4, mean=0.2, p80=(0.1, 0.3), p95=(0.05, 0.35), p_better=0.999, verdict="adopt")
    worse = _iv(0.01, (-0.02, 0.04), (-0.04, 0.06))
    d, _ = _decide(_cmp(improvement=worse, **good), base_scores=(50,) * 4, cand_scores=(60,) * 4)
    assert d.action != "adopt"
    assert any("floor" in r for r in d.reasons)


def test_cost_clause():
    good = dict(n_pairs=4, mean=10, p80=(6, 14), p95=(4, 16), p_better=0.999, verdict="adopt")
    d, _ = _decide(
        _cmp(cost_base=1.0, cost_cand=3.0, **good),
        base_scores=(50,) * 4,
        cand_scores=(60,) * 4,
        max_cost_ratio=2.0,
    )
    assert d.action == "reject"
    assert any("cost" in r for r in d.reasons)


def test_unscored_runs_reach_compare_and_are_counted_not_zero():
    from loopmath.gate import GatePolicy, decide

    rec = Recorder(_cmp(n_pairs=2))
    # A judge that ran and gave no score, and a rollout nobody judged.
    base = _runs("base", (50, 52)) + [_row("base", None)]
    cand = _runs("cand", (60, 61)) + [_row("cand", ...)]
    d = decide(base, cand, GatePolicy(), compare=rec)
    # compare pairs first and leaves out the unscored after, so it gets every run (review round 2):
    # a filter here would shift which runs pair when there is no pair key.
    assert rec.calls[0]["base"] == base and rec.calls[0]["cand"] == cand
    assert d.inputs["base"] == {"n": 3, "scored": 2, "unscored": 1}
    assert d.inputs["cand"] == {"n": 3, "scored": 2, "unscored": 1}


def test_an_arm_with_nothing_scored_asks_for_more_without_calling_compare():
    from loopmath.gate import GatePolicy, decide

    rec = Recorder(_cmp(n_pairs=0))
    d = decide(_runs("base", (50,)), [_row("cand", ...)], GatePolicy(), compare=rec)
    assert rec.calls == []
    assert d.action == "replicate" and d.n_more == 1


def test_decision_json_shape():
    d, _ = _decide(_cmp(n_pairs=2))
    j = d.to_json()
    assert j["schema"] == "loopmath.gate/1"
    for key in ("action", "n_more", "reasons", "comparison", "cost", "inputs", "policy"):
        assert key in j
    for key in ("n_pairs", "gain", "p_better", "verdict", "next_pair_value_usd"):
        assert key in j["comparison"]
    assert set(j["cost"]) >= {"base_usd_per_run", "cand_usd_per_run", "spent_usd", "next_usd", "unrecorded"}


def test_unrecorded_costs_refuse_budget_mode():
    from loopmath.gate import GatePolicy, decide

    base = [_row("base", 50, usd=None)] * 2
    cand = _runs("cand", (60, 61))
    with pytest.raises(ValueError) as exc:
        decide(base, cand, GatePolicy(budget_usd=5.0), compare=Recorder(_cmp(n_pairs=2)))
    assert "budget_usd needs every run's dollar cost, and 2 of 4 runs have no dollar cost" in str(exc.value)
    # Without a budget the gate runs and reports the count.
    d = decide(base, cand, GatePolicy(), compare=Recorder(_cmp(n_pairs=2)))
    assert d.cost["unrecorded"] == 2


def test_adoption_only_at_an_adopting_look():
    good = dict(mean=10, p80=(6, 14), p95=(4, 16), p_better=0.999, verdict="adopt")
    # 3 pairs is not a planned look (1, 2, 4, 8) and more pairs are affordable: go on to 4.
    d, _ = _decide(_cmp(n_pairs=3, **good), base_scores=(50,) * 3, cand_scores=(60,) * 3)
    assert d.action == "replicate" and d.n_more == 1
    assert any("look" in r for r in d.reasons)
    # When the budget allows no further pair, this is the final look, and it may adopt.
    d, _ = _decide(
        _cmp(n_pairs=3, cost_base=0.5, cost_cand=0.5, **good),
        base_scores=(50,) * 3,
        cand_scores=(60,) * 3,
        budget_usd=3.5,
    )
    assert d.action == "adopt"


def test_r_max_counts_executed_runs_not_scored_pairs():
    from loopmath.gate import GatePolicy, decide

    # Eight runs a side with the candidate's all unscored: the cap is reached; keep the base.
    rec = Recorder(_cmp(n_pairs=0))
    d = decide(_runs("base", (50,) * 8), [_row("cand", None)] * 8, GatePolicy(r_max=8), compare=rec)
    assert rec.calls == []
    assert d.action == "reject"
    assert any("--r-max 8" in r and "keeping the base" in r for r in d.reasons)
    # Four runs a side, two scored pairs, r_max = 5: the look at 4 wants two more pairs, but
    # only one more run fits under the cap.
    base = _runs("base", (50, 51)) + [_row("base", None)] * 2
    cand = _runs("cand", (52, 53)) + [_row("cand", None)] * 2
    d = decide(base, cand, GatePolicy(r_max=5), compare=Recorder(_cmp(n_pairs=2)))
    assert d.action == "replicate" and d.n_more == 1


def test_sunk_cost_is_spent_and_passed_on():
    from loopmath.gate import GatePolicy, decide

    rec = Recorder(_cmp(n_pairs=2, cost_base=0.5, cost_cand=0.5))
    base, cand = _runs("base", (50, 51)), _runs("cand", (52, 53))
    d = decide(base, cand, GatePolicy(budget_usd=10.0), compare=rec, sunk_usd=2.0)
    assert rec.calls[0]["spent_usd"] == pytest.approx(4.0)
    assert d.cost["sunk_usd"] == 2.0 and d.cost["spent_usd"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        decide(base, cand, GatePolicy(budget_usd=10.0), compare=rec, sunk_usd=None)


def test_runs_reach_compare_unchanged():
    # The pair object (pair key and experiment) is compare's to read; the gate passes runs as given.
    from loopmath.gate import GatePolicy, decide

    base, cand = _runs("base", (50, 51)), _runs("cand", (52, 53))
    for i, r in enumerate(base + cand):
        r["pair"], r["experiment"] = f"t1#{i % 2 + 1}", "exp-1"
    rec = Recorder(_cmp(n_pairs=2))
    decide(base, cand, GatePolicy(), compare=rec)
    assert all(a is b for a, b in zip(rec.calls[0]["base"], base))
    assert all(a is b for a, b in zip(rec.calls[0]["cand"], cand))


def _scaled(arm, scores, scale):
    rows = _runs(arm, scores)
    for r in rows:
        r["outcome"]["scale"] = scale
    return rows


def test_the_unit_is_the_runs_scale():
    from loopmath.gate import GatePolicy, decide

    d = decide(_scaled("b", [0.5, 0.52], "fraction"), _scaled("c", [0.55, 0.57], "fraction"), GatePolicy(),
               compare=Recorder(_cmp(n_pairs=2)))
    assert d.unit == "fraction"
    assert d.to_json()["unit"] == "fraction"
    d = decide(_runs("b", [50, 52]), _runs("c", [55, 57]), GatePolicy(), compare=Recorder(_cmp(n_pairs=2)))
    assert d.unit == "linear"
    d = decide(_runs("b", [50]), _runs("c", [55]), GatePolicy(rule="accept"), compare=Recorder(_cmp(n_pairs=1)))
    assert d.unit == "fraction" and d.measure == "cut in cost per accepted result"


def test_mixed_scales_refuse_by_name():
    from loopmath.gate import GatePolicy, decide

    rec = Recorder(_cmp(n_pairs=2))
    with pytest.raises(ValueError) as exc:
        decide(_scaled("b", [0.5, 0.52], "fraction"), _runs("c", [55, 57]), GatePolicy(), compare=rec)
    msg = str(exc.value)
    assert "fraction: 2 base and 0 candidate runs" in msg and "linear: 0 base and 2 candidate runs" in msg
    assert rec.calls == []


@pytest.mark.parametrize("floor, hint", [(2.0, "0.02"), (1.0, "0.01")], ids=["two", "one"])
def test_a_floor_of_one_or_more_on_fractions_refuses_with_a_hint(floor, hint):
    """A gain in a fraction from 0 to 1 is under 1, as compare's floor rule says; the gate refuses before
    compare is asked, with the likely meant floor."""
    from loopmath.gate import GatePolicy, decide

    rec = Recorder(_cmp(n_pairs=2))
    with pytest.raises(ValueError) as exc:
        decide(_scaled("b", [0.5, 0.52], "fraction"), _scaled("c", [0.55, 0.57], "fraction"),
               GatePolicy(floor=floor), compare=rec)
    assert "is not under 1" in str(exc.value) and f"did you mean {hint}?" in str(exc.value)
    assert rec.calls == []
    # On points, the same floor is in points.
    d = decide(_runs("b", [50, 52]), _runs("c", [55, 57]), GatePolicy(floor=floor), compare=Recorder(_cmp(n_pairs=2)))
    assert d.unit == "linear"


@pytest.mark.parametrize("floor", [-0.05, -1, float("nan"), float("inf"), True, "0.1"],
                         ids=["negative", "minus-one", "nan", "inf", "bool", "text"])
def test_a_floor_under_zero_or_not_a_number_is_refused_with_the_policy(floor):
    """compare refuses such a floor only once it is asked, when a loop has bought the runs; the policy
    refuses it when it is made, before any run."""
    from loopmath.gate import GatePolicy

    with pytest.raises(ValueError, match=r"floor must be a number, 0 or more"):
        GatePolicy(floor=floor)
    assert GatePolicy(floor=0).floor == 0 and GatePolicy(floor=0.02).floor == 0.02


def test_the_decision_names_where_it_stopped():
    from loopmath.gate import GatePolicy, decide

    two_b, two_c = _runs("b", [50, 51]), _runs("c", [56, 57])
    adopt = _cmp(n_pairs=2, mean=6.0, p80=(4.0, 8.0), p95=(2.0, 10.0), p_better=0.999, verdict="adopt")
    assert decide(two_b, two_c, GatePolicy(), compare=Recorder(adopt)).stop == "adopt"
    futile = _cmp(n_pairs=2, mean=-3.0, p80=(-5.0, -1.0), p95=(-6.0, 0.5), p_better=0.1)
    assert decide(two_b, two_c, GatePolicy(), compare=Recorder(futile)).stop == "futility"
    more = _cmp(n_pairs=2, mean=1.0, p80=(-1.0, 3.0), p95=(-2.0, 4.0), p_better=0.7)
    d = decide(two_b, two_c, GatePolicy(), compare=Recorder(more))
    assert (d.action, d.stop) == ("replicate", "replicate")
    eight_b, eight_c = _runs("b", [50] * 8), _runs("c", [51] * 8)
    capped = _cmp(n_pairs=8, mean=1.0, p80=(-1.0, 3.0), p95=(-2.0, 4.0), p_better=0.7)
    d = decide(eight_b, eight_c, GatePolicy(), compare=Recorder(capped))
    assert (d.action, d.stop) == ("reject", "cap")
    assert d.to_json()["stop"] == "cap"


@pytest.mark.parametrize("said, budget, stop, reason", [
    ("budget", 2.5, "budget", "budget: compare stopped buying pairs at P(better) 0.700, as $0.50 left is under "
                              "one pair's price"),
    ("budget", 2.0, "budget", "budget: compare stopped buying pairs at P(better) 0.700, as the budget is spent "
                              "($2.00 of $2.00)"),
    ("value", None, "value", "compare stopped buying pairs (the next pair's value is under its price) at "
                             "P(better) 0.700"),
    (None, None, "value", "compare stopped buying pairs (budget or the next pair's value) at P(better) 0.700"),
    ("cap", None, "cap", "not resolved within --r-max 8 pairs, where compare stops"),
], ids=["budget-left", "budget-spent", "value", "no-stop-field", "cap"])
def test_compare_s_stop_is_the_gate_s_stop(said, budget, stop, reason):
    """compare's `stop` says why it rejects short of a decision. Its budget stop is the gate's budget stop;
    a comparison without the field keeps the gate's older reading."""
    from loopmath.gate import GatePolicy, decide

    two_b, two_c = _runs("b", [50, 51]), _runs("c", [56, 57])  # four runs at $0.50: $2 spent
    c = _cmp(n_pairs=2, mean=1.0, p80=(-1.0, 3.0), p95=(-2.0, 4.0), p_better=0.7, verdict="reject",
             **({} if said is None else {"stop": said}))
    d = decide(two_b, two_c, GatePolicy(budget_usd=budget), compare=Recorder(c))
    assert (d.action, d.n_more, d.stop) == ("reject", 0, stop)
    assert reason + "; keeping the base" in d.reasons


GOOD = dict(n_pairs=4, mean=10, p80=(6, 14), p95=(4, 16), verdict="adopt")
BOUND = {"alpha": 0.025, "cap": 8, "looks": 8, "families": ["normal"], "held_out": {"rate": 0.02}}


def test_the_threshold_is_compares_where_it_states_a_bound():
    four_b, four_c = _runs("b", [50] * 4), _runs("c", [60] * 4)
    from loopmath.gate import GatePolicy, decide

    stated = _cmp(p_better=0.99, adopt_threshold=0.985, bound=BOUND, **GOOD)
    d = decide(four_b, four_c, GatePolicy(), compare=Recorder(stated))
    assert d.action == "adopt"
    assert d.reasons[0].startswith("adopt:") and "at least 0.98500" in d.reasons[0]
    assert "calibrated for looks up to 8 pairs" in d.reasons[-1]
    # No bound stated: the gate also splits alpha over its 4 looks, 1 - 0.025 / 4 = 0.99375.
    unstated = _cmp(p_better=0.99, adopt_threshold=0.985, bound=None, **GOOD)
    d = decide(four_b, four_c, GatePolicy(), compare=Recorder(unstated))
    assert d.action != "adopt"
    assert any("below 0.99375" in r for r in d.reasons)
    assert "no false adopt bound" in d.reasons[-1] and "the 4 looks that can adopt" in d.reasons[-1]
    # A compare without the round 2 fields is treated as stating no bound.
    d = decide(four_b, four_c, GatePolicy(), compare=Recorder(_cmp(p_better=0.99, **GOOD)))
    assert d.action != "adopt"


def test_the_fallback_counts_every_evaluation_under_an_acceptance_rule():
    """Under an acceptance rule a lone run moves P(better), so a repeat at the same pair count is a new
    chance to adopt: with no bound stated, the gate splits alpha over r_max evaluations, not its 4 looks."""
    from loopmath.gate import GatePolicy, decide, p_adopt

    accept = GatePolicy(rule="accept")
    assert GatePolicy().fallback_looks() == 4 and accept.fallback_looks() == 8
    assert p_adopt(accept) == pytest.approx(1 - 0.025 / 8)
    assert GatePolicy(rule="accept", r_min=1, r_max=1).fallback_looks() == 2  # never under the adopting looks
    assert accept.to_json()["fallback_looks"] == 8

    def rows(arm, n):
        return [{"schema": SCHEMA, "task_id": "t1", "arm": arm, "outcome": {"accepted": True},
                 "cost": {"usd": 0.5}} for _ in range(n)]

    unstated = _cmp(p_better=0.996, bound=None, **GOOD)
    d = decide(rows("b", 4), rows("c", 4), accept, compare=Recorder(unstated))
    assert d.action != "adopt"
    assert any("below 0.99687" in r for r in d.reasons), d.reasons  # 1 - 0.025 / 8 = 0.996875
    assert "up to 8 evaluations" in d.reasons[-1]
    d = decide(_runs("b", [50] * 4), _runs("c", [60] * 4), GatePolicy(), compare=Recorder(unstated))
    assert d.action == "adopt"  # a score rule keeps the 4 looks: 0.996 is above 0.99375


def test_compares_reject_at_the_cap_is_the_cap():
    from loopmath.gate import GatePolicy, decide

    at_cap = _cmp(n_pairs=8, mean=1.0, p80=(-1.0, 3.0), p95=(-2.0, 4.0), p_better=0.7, verdict="reject")
    d = decide(_runs("b", [50] * 8), _runs("c", [51] * 8), GatePolicy(), compare=Recorder(at_cap))
    assert (d.action, d.stop) == ("reject", "cap")
    assert any("--r-max 8" in r and "keeping the base" in r for r in d.reasons)


def test_the_policy_json_names_what_compare_was_asked():
    from loopmath.gate import GatePolicy

    j = GatePolicy(r_max=16).to_json()
    assert j["confidence"] == pytest.approx(0.975) and j["max_pairs"] == 16
    assert j["p_adopt"] == pytest.approx(1 - 0.025 / 5)
