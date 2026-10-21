"""One value of the next pairs, `loopmath.compare.value.pairs_value`, for check-change and the gate.

The value of going on with the check as the procedure that buys the pairs runs it (its `Continuation`: a look
after every pair for check-change; the gate's looks and guards for the gate), against stopping now, in dollars
over the horizon. Each simulated future draws its world once, and its action is scored against that world's
true saving, so a wrong adoption counts with its true loss (`loss`). `worth` is true, false or null
(unresolved): the value stop fires only on a certified false. `pair_price` prices a pair for that comparison
only (0: no marginal dollars); the budget still counts the runs' recorded dollars.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pytest

from loopmath.compare import Observation, paired
from loopmath.compare import model as M

ROWS = Path(__file__).resolve().parents[2] / "spec" / "examples" / "harness-runs" / "plugin-evals.jsonl"
THRESHOLD = 0.992501  # check-change's calibrated adopt threshold at alpha 0.05, cap 24 (cpar)
Z = 2.0


def V():
    from loopmath.compare import value

    return value


def shipped():
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["arm"] == "without"], [r for r in rows if r["arm"] == "with-lint-plugin"]


def _obs(n, *, base_ok, cand_ok, base_usd=0.32, cand_usd=0.28):
    """n pairs with the given outcomes (a bool, or a function of the pair index) and costs."""
    pick = (lambda f, i: f(i) if callable(f) else f)
    base = [Observation(f"t{i}", "base", pick(base_ok, i), base_usd * (1 + 0.01 * (i % 5)), run=f"b{i}")
            for i in range(n)]
    cand = [Observation(f"t{i}", "cand", pick(cand_ok, i), cand_usd * (1 + 0.01 * (i % 3)), run=f"c{i}")
            for i in range(n)]
    return base, cand


def _scores(n: int):
    base = [Observation(f"t{i}", "base", 0.5 + 0.02 * (i % 3), 0.5, run=f"b{i}") for i in range(n)]
    cand = [Observation(f"t{i}", "cand", 0.6 + 0.03 * (i % 2), 0.5, run=f"c{i}") for i in range(n)]
    return base, cand


def two_state(n=1000, bad=100, p_bad=0.001):
    """A two-state world, as posterior draws: costs 1 a run, the base's chance 0.5, the candidate's 0.6, or
    p_bad in `bad` of the n draws. Cells from independent arms."""
    pb = np.full(n, 0.5)
    pc = np.full(n, 0.6)
    pc[:bad] = p_bad
    theta = np.stack([pb * pc, (1 - pb) * pc, pb * (1 - pc), (1 - pb) * (1 - pc)], axis=1)
    var = 0.05 ** 2
    cmu = np.full((n, 2), -var / 2)  # exp(mu + var / 2) = 1
    ccov = np.tile(np.array([[var, 0.0], [0.0, var]]), (n, 1, 1))
    return M.Posterior(n=n, theta=theta, cmu=cmu, ccov=ccov)


def clear_data(pairs=2000, seed=0):
    """Data whose P(better) is far above any adopt threshold: the base accepted half the time, the candidate
    60%, costs about $1 a run."""
    rng = np.random.default_rng(seed)
    both, cand_only, base_only = int(0.3 * pairs), int(0.3 * pairs), int(0.2 * pairs)
    cells = np.array([both, cand_only, base_only, pairs - both - cand_only - base_only], dtype=float)
    costs = [tuple(x) for x in rng.normal(0.0, 0.1, size=(pairs, 2))]
    return M.Data(kind="binary", cells=cells, cost_pairs=costs)


def cpar_plan(**kw):
    base = dict(objective="cpar", floor=0.05, confidence=0.95, threshold=THRESHOLD, horizon_runs=50,
                pairs_left=4, pairs_scored=2000)
    base.update(kw)
    return V().Plan(**base)


def certified(v: dict) -> None:
    """worth true: the best plan's net is above 0 by 2 standard errors; false: every plan's net below 0 by 2
    (or every gross exactly 0); null otherwise."""
    steps = v["steps"]
    if v["worth"] is True:
        best = max(steps, key=lambda s: s["net"])
        assert best["net"] - Z * best["se"] > 0
    elif v["worth"] is False:
        assert all(s["gross"] == 0 for s in steps) or all(
            s["se"] is not None and s["net"] + Z * s["se"] < 0 for s in steps)
    else:
        assert v["why"]


# ---------------------------------------------------------------- 1. the published seam and the utility
def test_the_published_seam():
    v = V()
    params = list(inspect.signature(v.pairs_value).parameters)
    assert params[:4] == ["data", "post", "plan", "rng"] and "sims" in params
    plan = {f.name for f in dataclasses.fields(v.Plan)}
    assert {"objective", "floor", "confidence", "threshold", "horizon_runs", "value_usd", "pairs_left", "price",
            "zero_cost", "pairs_scored", "bound_pairs", "continuation", "budget_left"} <= plan
    cont = {f.name for f in dataclasses.fields(v.Continuation)}
    assert {"name", "looks", "adopt_from", "band", "futility", "max_cost_ratio", "fallback", "at_end"} <= cont
    assert all(callable(getattr(v.Continuation, m)) for m in ("step", "batch", "points", "describe"))
    got = {f.name for f in dataclasses.fields(v.PairValue)}
    assert {"per_pair", "price", "price_from", "pairs", "expected_pairs", "gross", "cost", "net", "se", "sims",
            "loss", "adoptions", "worth", "steps", "horizon_runs", "threshold", "pairs_scored", "pairs_left",
            "budget_left", "continuation", "stop_now", "why"} <= got
    for cls in (v.Plan, v.Continuation, v.PairValue):
        assert cls.__dataclass_params__.frozen
    assert v.Plan(objective="cpar", floor=0.05, confidence=0.95, threshold=0.99).continuation == v.Continuation()


def test_the_old_value_functions_and_the_plug_in_are_gone():
    for name in ("next_pair_value", "decision_value", "plug_in_saving"):
        assert not hasattr(M, name), name


def test_the_saving_is_the_expected_cost_per_accepted_result_not_a_ratio_of_means():
    # P(better by 5%) is 0.999, the ratio of posterior means says +0.33 a task, the expectation -999.67.
    post = two_state(n=1000, bad=1, p_bad=1e-6)
    u = V().saving(post, cpar_plan())
    assert u.shape == (1000,)
    assert float(np.mean(u)) == pytest.approx(2.0 - (0.999 / 0.6 + 0.001 / 1e-6), rel=1e-6)  # -999.665
    plug_in = 1 / 0.5 - 1 / float(np.mean(post.chance("cand")))
    assert plug_in == pytest.approx(0.331665, abs=1e-5)
    imp = 1 - (1 / post.chance("cand")) / (1 / post.chance("base"))
    assert float(np.mean(imp > 0.05)) == pytest.approx(0.999)


def test_adoptions_at_the_calibrated_threshold_are_scored_against_their_worlds():
    # Data far past the threshold: every future adopts at its first look. 10% of the worlds have a candidate that
    # is almost never accepted, so adopting loses about 998 dollars a task there: E[U] = H x (-99.5).
    post = two_state(n=1000, bad=100, p_bad=0.001)
    plan = cpar_plan()
    pv = V().pairs_value(clear_data(), post, plan, V().value_rng(0))
    expect = 50 * float(np.mean(V().saving(post, plan)))
    assert expect == pytest.approx(50 * (0.9 * (2 - 1 / 0.6) + 0.1 * (2 - 1000)), rel=1e-6)
    first = pv.steps[0]
    assert first["p_adopt"] == 1.0 and first["adoptions"] == 256
    assert first["gross"] < 0 and abs(first["gross"] - expect) < 4 * first["se"]
    assert first["loss"] < first["gross"] < 0  # the losing worlds' part, less the good worlds' gains


def test_a_near_null_check_counts_its_wrong_adoptions():
    # Six pairs with the candidate a little ahead: the posterior leaves room for a worse candidate, and a few
    # futures adopt in such worlds at the calibrated threshold (rarely: that is what the calibration buys). Their
    # dollars are in the gross, as `loss`: never above 0, and below 0 on some of the value seeds.
    base, cand = _obs(6, base_ok=lambda i: i % 3 != 0, cand_ok=lambda i: i % 6 != 0)
    losses = []
    for seed in range(5):
        c = paired(base, cand, rule="tests", seed=seed)
        assert c.verdict == "more" and c.value["per_pair"] is not None
        top = c.value["steps"][-1]
        assert top["adoptions"] > 0 and top["loss"] <= 0 and top["gross"] > top["loss"]
        losses.append(top["loss"])
    assert min(losses) < 0, losses


# ---------------------------------------------------------------- 2. when the value exists
def test_no_accepted_base_run_leaves_the_value_unstated():
    base, cand = _obs(2, base_ok=False, cand_ok=lambda i: i == 0)
    c = paired(base, cand, rule="tests", value_stop=True, min_pairs=1)
    v = c.value
    assert v["per_pair"] is None and c.next_pair_value_usd is None and "base" in v["why"]
    assert c.stop != "value"


def test_no_accepted_candidate_run_leaves_the_value_unstated():
    base, cand = _obs(2, base_ok=lambda i: i == 0, cand_ok=False)
    c = paired(base, cand, rule="tests", value_stop=True, min_pairs=1)
    v = c.value
    assert v is not None and v["per_pair"] is None and "candidate" in v["why"]
    assert c.stop != "value"


def test_one_accepted_run_gives_a_mean_but_no_stated_error():
    base, cand = _obs(5, base_ok=lambda i: i == 0, cand_ok=lambda i: i < 3)
    c = paired(base, cand, rule="tests", value_stop=True, min_pairs=1)
    v = c.value
    assert v["per_pair"] is not None and v["se"] is None and v["worth"] is None and v["why"]
    assert c.stop != "value"


@pytest.mark.parametrize("n, mean, se", [(1, False, False), (2, True, False), (3, True, True)])
def test_a_score_needs_two_pairs_for_a_mean_and_three_for_its_error(n, mean, se):
    c = paired(*_scores(n), rule="score:judge", max_pairs=24, seed=1, value_usd=10.0)
    v = c.value
    assert v is not None
    assert (v["per_pair"] is not None) is mean and (v["se"] is not None) is se
    if not se:
        assert v["worth"] is None and v["why"]


def test_a_score_without_a_dollar_value_has_no_value():
    c = paired(*_scores(3), rule="score:judge", max_pairs=24, seed=1)
    assert c.next_pair_value_usd is None and c.value["per_pair"] is None and "value_usd" in c.value["why"]


def dear(n, usd=1e308):
    """n pairs at usd dollars a run under an acceptance rule, the base accepted on every other pair, the candidate
    on all: at $1e308 two runs' dollars overflow to inf."""
    return ([Observation(f"t{i}", "base", i % 2 == 0, usd, run=f"b{i}") for i in range(n)],
            [Observation(f"t{i}", "cand", True, usd, run=f"c{i}") for i in range(n)])


def test_a_cost_that_is_not_finite_is_named_unknown():
    assert "a pair's expected cost is not finite (inf USD" in V().unknown_cost(float("inf"), np.zeros(3))
    assert "the saving of an adoption is not finite" in V().unknown_cost(1.0, np.array([0.1, np.nan]))
    assert V().unknown_cost(1.0, np.zeros(3)) is None and V().unknown_cost(None, np.zeros(3)) is None


@pytest.mark.parametrize("price", [None, 0.0])
def test_a_cost_past_the_float_limit_is_unknown_and_never_stops_the_check(price):
    # Each run's $1e308 is a valid amount, but a pair's dollars overflow: an unknown cost to the value, which
    # stops nothing, with the stop on from the first pair and at a given price of 0 too.
    with np.errstate(over="ignore", invalid="ignore"):
        c = paired(*dear(4), rule="tests", max_pairs=24, value_stop=True, min_pairs=1, pair_price=price, seed=1)
    v = c.value
    assert c.verdict == "more" and c.stop != "value"
    assert v["per_pair"] is None and v["worth"] is None and "not finite" in v["why"]
    assert "not finite" in " ".join(c.reasons) and c.next_pair_value_usd is None


# ---------------------------------------------------------------- 3. one number
def test_the_comparison_carries_the_value_and_its_plan():
    c = paired(*shipped(), rule="accept")
    v = c.value
    assert v["schema"] == "loopmath.pair-value/1"
    assert c.next_pair_value_usd == v["per_pair"]
    assert v["threshold"] == c.adopt_threshold and v["pairs_left"] == c.max_pairs - c.counts["pairs"]
    assert v["continuation"]["name"] == "check-change" and v["continuation"]["looks"] is None
    assert v["stop_now"] == "keep the base"
    assert v["price_from"] == "cost" and v["price"] == pytest.approx(c.pair_price_usd, abs=1e-6)
    assert c.to_dict()["value"] == v


def test_same_data_seed_and_plan_give_the_same_number():
    a = paired(*shipped(), rule="accept", confidence=0.975, max_pairs=8)
    b = paired(*shipped(), rule="accept", confidence=0.975, max_pairs=8)
    assert a.value == b.value and a.next_pair_value_usd == b.next_pair_value_usd


def test_the_value_does_not_move_with_the_other_draws():
    a = paired(*shipped(), rule="accept")
    b = paired(*shipped(), rule="accept", spent_usd=0.0)
    assert a.value == b.value


def test_pairs_value_called_directly_gives_the_comparisons_number(monkeypatch):
    v = V()
    got = {}
    real = v.pairs_value

    def spy(data, post, plan, rng, **kw):
        got.update(data=data, post=post, plan=plan)
        return real(data, post, plan, rng, **kw)

    monkeypatch.setattr(v, "pairs_value", spy)
    c = paired(*shipped(), rule="accept")
    monkeypatch.setattr(v, "pairs_value", real)
    assert v.pairs_value(got["data"], got["post"], got["plan"], v.value_rng(0)).to_dict() == c.value


# ---------------------------------------------------------------- 4. worth: true, false or null
@pytest.mark.parametrize("kw", [{}, {"confidence": 0.975, "max_pairs": 8}, {"pair_price": 0},
                                {"horizon_runs": 500}, {"seed": 3}])
def test_worth_is_certified_by_two_standard_errors(kw):
    c = paired(*shipped(), rule="accept", **kw)
    certified(c.value)
    best = max(c.value["steps"], key=lambda s: s["net"])
    assert c.value["pairs"] == best["pairs"] and c.value["net"] == best["net"]
    assert c.value["per_pair"] == pytest.approx(best["gross"] / best["expected_pairs"], rel=1e-5, abs=1e-6)
    assert c.value["sims"] == 256 and c.value["se"] is not None and c.value["cost"] == best["cost"]
    for s in c.value["steps"]:
        # The net is the gross less the dollars the plan pays: a given price per pair, or the drawn pairs' own
        # recorded dollars, which average about a pair's expected cost.
        assert s["net"] == pytest.approx(s["gross"] - s["cost"], abs=1e-5)
        if kw.get("pair_price") is not None:
            assert s["cost"] == pytest.approx(kw["pair_price"] * s["expected_pairs"], abs=1e-5)
        else:
            assert s["cost"] / s["expected_pairs"] == pytest.approx(c.value["price"], rel=0.25)
        assert 1 <= s["expected_pairs"] <= s["pairs"] and s["loss"] <= 0


def test_the_value_states_its_futures_beside_its_error():
    # The JSON carries the sims count beside se and worth; 0 where no future was run.
    c = paired(*shipped(), rule="accept", seed=1)
    assert c.value["sims"] == 256 and {"se", "worth", "sims"} <= set(c.value)
    none = paired(*_obs(2, base_ok=False, cand_ok=True), rule="accept")
    assert none.value["per_pair"] is None and none.value["sims"] == 0 and none.value["worth"] is None


def small_data(pairs=20, seed=0):
    """20 pairs with more than two accepted runs on each arm, so every moment the value needs exists."""
    rng = np.random.default_rng(seed)
    costs = [tuple(x) for x in rng.normal(0.0, 0.1, size=(pairs, 2))]
    return M.Data(kind="binary", cells=np.array([6, 6, 4, 4], dtype=float), cost_pairs=costs)


@pytest.mark.parametrize("sims", [1, 0, -3, 2.5, True, None])
def test_the_futures_must_number_two_or_more(sims):
    # One future gives no standard error, so worth could not be certified: a count under 2 is refused by name.
    v, data = V(), small_data()
    post = M.posterior(data, np.random.default_rng(0))
    plan = cpar_plan(pairs_left=1, pairs_scored=20)
    with pytest.raises(ValueError, match="sims must be a whole number, 2 or more"):
        v.pairs_value(data, post, plan, v.value_rng(0), sims=sims)
    two = v.pairs_value(data, post, plan, v.value_rng(0), sims=2)
    assert two.sims == 2 and two.se is not None and two.worth in (True, False, None)


def test_the_horizon_is_applied_once():
    # U carries H once. The futures' actions do not depend on H, so on the same seed and
    # plan, doubling it doubles every gross and leaves the pairs and their dollars as they were.
    v = V()
    a = v.pairs_value(clear_data(), two_state(), cpar_plan(horizon_runs=50), v.value_rng(0))
    b = v.pairs_value(clear_data(), two_state(), cpar_plan(horizon_runs=100), v.value_rng(0))
    assert a.gross < 0
    for x, y in zip(a.steps, b.steps):
        assert y["gross"] == pytest.approx(2 * x["gross"], rel=1e-5)
        assert (y["cost"], y["expected_pairs"], y["adoptions"]) == (x["cost"], x["expected_pairs"], x["adoptions"])


def test_check_change_plans_stop_at_the_pairs_left():
    c = paired(*shipped(), rule="accept")
    assert [s["pairs"] for s in c.value["steps"]] == [1, 2, 4, 8, 16]
    c = paired(*shipped(), rule="accept", max_pairs=8)
    assert c.value["pairs_left"] == 5 and [s["pairs"] for s in c.value["steps"]] == [1, 2, 4, 5]
    c = paired(*shipped(), rule="accept", max_pairs=4)
    assert c.value["pairs_left"] == 1 and [s["pairs"] for s in c.value["steps"]] == [1]


def test_a_positive_net_needs_more_than_one_good_future():
    # Most futures adopt a good candidate; the 10% that adopt a bad one outweigh them. Free pairs are still not
    # worth buying: going on adopts, and adopting loses on average.
    pv = V().pairs_value(clear_data(), two_state(), cpar_plan(price=0.0), V().value_rng(0))
    assert pv.price == 0 and pv.worth is False
    assert all(s["net"] + Z * s["se"] < 0 for s in pv.steps)


# ---------------------------------------------------------------- 5. a $0 pair price and rare futures
def _rare(cap, seed):
    """The shipped rows under the gate's continuation with a cap of `cap`, at a $0 pair price with the value stop
    on from the first pair. At cap 5 two pairs are left (looks at 4 and 5) and an adoption is rare; at cap 4 one
    pair is left."""
    from loopmath.gate import GatePolicy

    cont = GatePolicy(rule="accept", r_max=cap).continuation()
    return paired(*shipped(), rule="accept", confidence=0.975, max_pairs=cap, continuation=cont, pair_price=0,
                  value_stop=True, min_pairs=1, seed=seed)


@pytest.mark.parametrize("cap", [4, 5])
def test_at_a_zero_price_an_empty_sample_never_stops_the_check(cap):
    # Sampled futures that miss a rare adoption are unresolved, never "not worth it".
    seen = {"empty": 0, "some": 0}
    for seed in range(16):
        c = _rare(cap, seed)
        assert c.stop != "value" and c.verdict == "more", (seed, c.value)
        v = c.value
        if all(s["adoptions"] == 0 for s in v["steps"]):
            seen["empty"] += 1
            assert v["worth"] is None and "no sampled future adopts" in v["why"] and v["gross"] == 0
        else:
            seen["some"] += 1
    assert seen["empty"] > 0, "the fixture no longer exercises an empty sample"
    if cap == 5:
        assert seen["some"] > 0, "at cap 5 the fixture should also sample a rare adoption"


def test_no_look_that_can_adopt_is_a_structural_zero():
    v = V()
    cont = v.Continuation(name="gate", looks=(1, 2, 4, 8), adopt_from=4, band=True, futility=True)
    data = clear_data(pairs=1)
    post = M.posterior(data, np.random.default_rng(0))
    plan = cpar_plan(pairs_scored=1, pairs_left=2, price=0.0, continuation=cont)  # 1 + 2 < r_min 4
    pv = v.pairs_value(data, post, plan, v.value_rng(0))
    assert pv.gross == 0 and pv.net == 0 and pv.worth is False and "adopt" in pv.why


def test_a_zero_pair_price_compares_the_value_with_zero():
    c = paired(*shipped(), rule="accept", pair_price=0)
    v = c.value
    assert v["price"] == 0 and v["price_from"] == "given"
    assert c.pair_price_usd == paired(*shipped(), rule="accept").pair_price_usd > 0  # the pair's cost, for the budget
    assert v["net"] == v["gross"] and c.to_dict()["pair_price"] == 0


def test_a_zero_price_keeps_the_budget_in_recorded_dollars():
    c = paired(*shipped(), rule="accept", pair_price=0, budget_usd=2.0)
    assert c.spent_usd == pytest.approx(1.8) and (c.verdict, c.stop) == ("reject", "budget")


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), True])
def test_a_bad_pair_price_is_refused(bad):
    with pytest.raises(ValueError, match="pair_price"):
        paired(*shipped(), rule="accept", pair_price=bad)


# ---------------------------------------------------------------- 5a. runs recorded at $0
def tied(n, usd):
    """n tied pairs under an acceptance rule, both arms accepted on the even pairs, at usd dollars a run, or
    usd(arm, i)."""
    pick = usd if callable(usd) else (lambda arm, i: usd)
    return ([Observation(f"t{i}", "base", i % 2 == 0, pick("base", i), run=f"b{i}") for i in range(n)],
            [Observation(f"t{i}", "cand", i % 2 == 0, pick("cand", i), run=f"c{i}") for i in range(n)])


def _but_source(v):
    return {k: x for k, x in v.items() if k != "price_from"}


def test_a_plan_whose_runs_all_cost_0_prices_a_pair_at_0_unless_a_price_is_given():
    v = V()
    free = v.pairs_value(clear_data(), two_state(), cpar_plan(zero_cost=True), v.value_rng(0))
    given = v.pairs_value(clear_data(), two_state(), cpar_plan(price=0.0), v.value_rng(0))
    assert (free.price, free.price_from, given.price_from) == (0.0, "cost", "given")
    assert _but_source(free.to_dict()) == _but_source(given.to_dict())
    priced = v.pairs_value(clear_data(), two_state(), cpar_plan(zero_cost=True, price=0.5), v.value_rng(0))
    assert (priced.price, priced.price_from) == (0.5, "given")


@pytest.mark.parametrize("n, min_pairs", [(4, None), (4, 1), (6, None)])
def test_runs_recorded_at_0_price_a_pair_at_0_as_a_given_zero_price_does(n, min_pairs):
    # Every recorded cost is $0: a pair's price is 0, not the two millionths of a dollar the cost model gives two
    # $0 runs, so a sample with no adoption never stops the check, as at pair_price 0.
    kw = dict(rule="accept", value_stop=True, min_pairs=min_pairs, seed=1)
    free, given = paired(*tied(n, 0.0), **kw), paired(*tied(n, 0.0), pair_price=0, **kw)
    assert free.stop != "value" and (free.verdict, free.stop) == (given.verdict, given.stop)
    assert free.value["price"] == 0 and free.value["price_from"] == "cost"
    assert _but_source(free.value) == _but_source(given.value)
    assert free.pair_price_usd == given.pair_price_usd > 0  # the budget's price is still the cost model's
    assert "every recorded cost is $0, so the next pairs' value prices a pair at $0" in " ".join(free.reasons)
    priced = paired(*tied(n, 0.0), pair_price=2.0, **kw)  # a given price is the value's price
    assert priced.value["price"] == 2.0 and priced.value["price_from"] == "given"
    assert "prices a pair at $0" not in " ".join(priced.reasons)


def test_priced_tied_runs_keep_their_price_and_their_stop():
    c = paired(*tied(4, 0.5), rule="accept", value_stop=True, seed=1)
    v = c.value
    assert v["price_from"] == "cost" and v["price"] == pytest.approx(1.054837, abs=1e-6)
    assert (c.verdict, c.stop, v["worth"]) == ("reject", "value", False)
    assert not any("prices a pair at $0" in r for r in c.reasons)


def one_at(z, arm="base", i=0, rest=0.5):
    """usd(arm, i) for tied(): run i of the arm recorded at z dollars, every other run at rest."""
    return lambda a, j: z if (a, j) == (arm, i) else rest


def _same_answer(c, u):
    return (c.verdict, c.stop, c.p_better, c.objective) == (u.verdict, u.stop, u.p_better, u.objective)


@pytest.mark.parametrize("arm", ["base", "cand"])
@pytest.mark.parametrize("n", [4, 8, 12, 16, 24])
def test_an_arm_whose_costs_mix_0_and_priced_runs_has_no_cost_range(arm, n):
    # One $0 run among runs at $0.50 on tied pairs: read as a millionth of a dollar, it spread the arm's log costs
    # so wide that the arm's cost per run grew with the pairs ($12.41 at 16 against $0.47 recorded), and the dearer
    # arm was adopted or the cheaper one rejected on cost. The arm now has no cost range: the answer is on
    # acceptance alone, as with that run's cost unrecorded.
    kw = dict(rule="accept", value_stop=True, seed=1)
    c = paired(*tied(n, one_at(0.0, arm)), **kw)
    u = paired(*tied(n, one_at(None, arm)), **kw)
    assert c.cost_base is None and c.cost_cand is None and c.cpar_base is None and c.cpar_cand is None
    assert c.outcomes_only and c.objective == "gain" and c.value is None and c.next_pair_value_usd is None
    assert _same_answer(c, u) and c.verdict != "adopt"
    word = "base" if arm == "base" else "candidate"
    said = " ".join(c.reasons)
    assert f"the {word}'s recorded costs mix $0 and priced runs (1 of {n} at $0)" in said
    assert "the cost model states no cost range for the base and candidate" in said
    with pytest.raises(ValueError, match="objective cpar"):
        paired(*tied(n, one_at(0.0, arm)), objective="cpar", **kw)


def test_a_cost_range_is_kept_for_a_tiny_priced_run_an_arm_all_at_0_and_the_switched_model(monkeypatch):
    kw = dict(rule="accept", value_stop=True, seed=1)
    # A run recorded at a millionth of a dollar is a priced run: the cost model reads it, as before.
    tiny = paired(*tied(4, one_at(1e-6)), **kw)
    assert tiny.cost_base is not None and not tiny.outcomes_only and tiny.value["price"] > 1.0
    # One arm all at $0 and the other priced: no arm mixes, and the costs decide (reject on tied outcomes).
    apart = paired(*tied(4, lambda a, i: 0.0 if a == "base" else 0.5), **kw)
    assert apart.cost_base is not None and apart.cost_cand is not None
    assert (apart.verdict, apart.stop) == ("reject", "decided")
    # The switched cost model, an opt-in setting, reads the mix as it did.
    monkeypatch.setattr(M, "COST_MODEL", "sw")
    sw = paired(*tied(4, one_at(0.0)), **kw)
    assert sw.cost_base is not None and not sw.outcomes_only
    assert not any("mix $0 and priced runs" in r for r in sw.reasons)


def test_varied_costs_with_a_failed_run_at_0_are_not_adopted_on_cost():
    # Every run at $0.50 times a lognormal factor; the base's pair-3 run failed before any cost and is recorded at
    # $0. The base's recorded costs are the lower ones and acceptance is tied: the candidate is no better.
    from loopmath.gate import GatePolicy, decide

    rng = np.random.default_rng(7)
    f = 0.5 * np.exp(rng.normal(0.0, 0.3, size=(2, 48)))
    for n in (12, 16, 24):
        base = [Observation(f"t{i}", "base", i % 2 == 0, 0.0 if i == 3 else float(f[0, i]), run=f"b{i}", failed=i == 3)
                for i in range(n)]
        cand = [Observation(f"t{i}", "cand", i % 2 == 0, float(f[1, i]), run=f"c{i}") for i in range(n)]
        c = paired(base, cand, rule="accept", seed=1)
        assert c.verdict != "adopt" and c.outcomes_only and c.cost_base is None, n
        assert decide(base, cand, GatePolicy(rule="accept", r_max=24)).action != "adopt", n


# ---------------------------------------------------------------- 5b. the budget in the futures
CELL = {0: (True, True), 1: (False, True), 2: (True, False), 3: (False, False)}  # cell: (base, candidate)


def pairs_obs(cells, log_costs, start=0):
    """Observations for pairs given as model cells and log costs (base, candidate), from task `start` on."""
    base, cand = [], []
    for j, (cell, (lb, lc)) in enumerate(zip(cells, log_costs)):
        i, (b, c) = start + j, CELL[int(cell)]
        base.append(Observation(f"t{i}", "base", b, round(float(np.exp(lb)), 9), run=f"b{i}"))
        cand.append(Observation(f"t{i}", "cand", c, round(float(np.exp(lc)), 9), run=f"c{i}"))
    return base, cand


def undecided(n, seed=5):
    """n pairs, each won by one arm in turn (P(better) near a half), at about $0.50 a run."""
    rng = np.random.default_rng(seed)
    costs = np.log(0.5 * (1 + 0.01 * rng.standard_normal((n, 2))))
    return pairs_obs([1, 2] * (n // 2), costs)


def captured(monkeypatch, call):
    """What the value is computed on in `call()`: the comparison's data, posterior and plan."""
    v, got = V(), {}
    real = v.pairs_value

    def spy(data, post, plan, rng, **kw):
        got.update(data=data, post=post, plan=plan)
        return real(data, post, plan, rng, **kw)

    monkeypatch.setattr(v, "pairs_value", spy)
    out = call()
    monkeypatch.setattr(v, "pairs_value", real)
    return out, got


def future_path(got, cells, per_run, first, top):
    """One future, drawn by hand (`cells`, both runs at `per_run` dollars), run as the value's futures run it."""
    v, plan = V(), got["plan"]
    fut = {"cells": np.asarray(cells), "cost_pairs": np.log(np.full((len(cells), 2), per_run))}
    st = M.Settings(objective=plan.objective, floor=plan.floor, confidence=plan.confidence, threshold=plan.threshold,
                    horizon_runs=plan.horizon_runs, value_usd=plan.value_usd)
    return v._future(got["data"], plan, st, fut, first, top, v.value_rng(0)), fut


def test_the_plan_carries_the_budget_left_and_the_caps_pairs_left(monkeypatch):
    b0, c0 = undecided(6)
    budget = sum(o.usd for o in b0 + c0) + 6.5
    c, got = captured(monkeypatch, lambda: paired(b0, c0, rule="accept", budget_usd=budget))
    assert got["plan"].budget_left == pytest.approx(budget - c.spent_usd, abs=1e-9)
    assert got["plan"].pairs_left == 24 - 6  # the cap alone: the budget is spent in the futures, not counted here
    assert c.value["budget_left"] == pytest.approx(6.5, abs=1e-6) and c.value["pairs_left"] == 18


@pytest.mark.parametrize("per_run, left0, stops", [(2.0, 6.5, "before"), (0.2, 3.3, "after")])
def test_a_futures_budget_stop_is_compares_on_the_same_runs(monkeypatch, per_run, left0, stops):
    # A future spends the budget on its drawn pairs and, at every look, stops where compare would
    # on the same runs (the budget spent, or the dollars left under a pair's expected cost at that look). Pairs
    # that cost more than expected run out before the count the budget bought at the start; cheaper ones run on.
    b0, c0 = undecided(6)
    budget = sum(o.usd for o in b0 + c0) + left0
    now, got = captured(monkeypatch, lambda: paired(b0, c0, rule="accept", budget_usd=budget))
    static = int((budget - now.spent_usd) // now.pair_price_usd)
    path, fut = future_path(got, [1, 2] * 4, per_run, 1, 8)
    for look in path:
        bk, ck = pairs_obs(fut["cells"][:look["k"]], fut["cost_pairs"][:look["k"]], start=6)
        c = paired(b0 + bk, c0 + ck, rule="accept", budget_usd=budget)
        assert look["left"] == pytest.approx(budget - c.spent_usd, abs=1e-6)
        assert look["price"] == pytest.approx(c.pair_price_usd, rel=0.05)  # the same posterior, its own draws
        assert (look["act"], c.verdict, c.stop) in ((None, "more", None), ("end", "reject", "budget")), look
    end = path[-1]
    assert end["act"] == "end"
    assert end["k"] < static if stops == "before" else end["k"] > static


def test_the_value_spends_the_budget_in_its_futures(monkeypatch):
    # With the budget nearly spent, the plans buy fewer pairs on average than with none, and no plan buys more
    # than the cap's pairs left.
    b0, c0 = undecided(6)
    spent = sum(o.usd for o in b0 + c0)
    free = paired(b0, c0, rule="accept", pair_price=0)
    tight = paired(b0, c0, rule="accept", pair_price=0, budget_usd=spent + 2.5)
    assert tight.value["budget_left"] == pytest.approx(2.5, abs=1e-6) and free.value["budget_left"] is None
    for f, t in zip(free.value["steps"], tight.value["steps"]):
        assert f["pairs"] == t["pairs"] and t["expected_pairs"] <= f["expected_pairs"]
    assert tight.value["steps"][-1]["expected_pairs"] < 4 < free.value["steps"][-1]["expected_pairs"]
    assert "$2.50 of budget left" in V().plan_text(tight.value) and "budget" not in V().plan_text(free.value)


# ---------------------------------------------------------------- 6. the stop alternative
def test_a_stop_that_adopts_is_the_baseline():
    # Every future adopts at its first look, and stopping now adopts too: going on changes nothing.
    v = V()
    cont = v.Continuation(name="gate", at_end="adopt")
    pv = v.pairs_value(clear_data(), two_state(), cpar_plan(price=0.5, continuation=cont), v.value_rng(0))
    assert pv.stop_now == "adopt"
    assert all(s["gross"] == 0 for s in pv.steps) and pv.worth is False


# ---------------------------------------------------------------- 7. small n
@pytest.mark.parametrize("n", [1, 2, 3])
def test_small_n_under_an_acceptance_rule(n):
    c = paired(*_obs(n, base_ok=True, cand_ok=True), rule="tests", max_pairs=24, seed=1)
    v = c.value
    assert v is not None and v["per_pair"] is not None and c.next_pair_value_usd == v["per_pair"]
    assert v["pairs_left"] == 24 - n
    certified(v)


def test_no_scored_pair_has_no_value():
    base = [Observation("t0", "base", None, 0.5, run="b0")]
    cand = [Observation("t0", "cand", None, 0.5, run="c0")]
    c = paired(base, cand, rule="tests")
    assert c.next_pair_value_usd is None and c.value is None


def test_a_decided_comparison_has_no_value():
    base = [Observation(f"t{i}", "base", False, 1.0, run=f"b{i}") for i in range(20)]
    cand = [Observation(f"t{i}", "cand", True, 0.5, run=f"c{i}") for i in range(20)]
    c = paired(base, cand, rule="tests")
    assert c.verdict == "adopt" and c.value is None and c.next_pair_value_usd is None


# ---------------------------------------------------------------- 8. speed
@pytest.mark.parametrize("gate", [False, True])
def test_one_comparison_stays_fast(gate):
    from loopmath.gate import GatePolicy

    kw = {"continuation": GatePolicy(rule="accept").continuation(), "confidence": 0.975, "max_pairs": 8} if gate else {}
    t = time.perf_counter()
    paired(*shipped(), rule="accept", **kw)
    assert time.perf_counter() - t < 3.0
