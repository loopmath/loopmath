"""`loopmath.gate.winner`: how much of a search loop's winning lead is real.

A loop scores N candidates and keeps the best; the best of N noisy scores is high partly by luck. `assess`
shrinks the winner's lead with a hierarchical prior learnt from all N scores (censored ones too), reports
its range and P(beats parent), and asks `gate.decide` what to buy before adopting.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from loopmath.gate import GatePolicy, decide


def gw():
    from loopmath.gate import winner

    return winner


def _search(scores, parent=0.0, se=0.02, censored=(), better="higher", scale="linear", **kw):
    w = gw()
    return w.Search(
        parent=w.Score("seed", float(parent), se=se),
        candidates=[w.Score(f"c{i}", float(s), se=se) for i, s in enumerate(scores)],
        censored=[w.Censored(count=k, at_most=c, se=se) for k, c in censored],
        better=better,
        scale=scale,
        **kw,
    )


def _null_search(se=0.02):
    rng = np.random.default_rng(7)
    scores = rng.normal(0.0, 0.02, 64)
    return _search(scores, parent=rng.normal(0.0, 0.02), se=se), scores


def _row(arm, score, task):
    return {"schema": "loopmath.harness-runs/0.1", "task_id": task, "arm": arm,
            "outcome": {"name": "judge", "score": score, "judge": "fake-judge", "scale": "fraction"}}


def test_a_null_search_shrinks_its_lead_toward_zero():
    search, scores = _null_search()
    v = gw().assess(search, draws=0)
    assert v.winner == f"c{int(np.argmax(scores))}"
    assert v.observed_lead > 0.04  # the best of 64 null scores, 0.0516 with this seed
    assert v.gain["mean"] < 0.5 * v.observed_lead
    lo, hi = v.gain["bands"]["80"]
    assert lo < 0 < hi
    assert v.p_better < 0.9
    assert 0.5 < v.shrinkage <= 1.0


def test_the_default_view_is_sparse_and_the_others_can_be_asked_for():
    search, _ = _null_search()
    w = gw()
    v = w.assess(search, draws=0)
    assert v.prior["view"] == "sparse" and v.prior["views"] == {"ridge": 0.0, "sparse": pytest.approx(1.0)}
    for view in ("ridge", "auto"):
        assert w.assess(search, prior=view, draws=0).prior["view"] == view
    assert w.assess(search, prior="auto", draws=0).prior["views"]["ridge"] > 0


def test_regimes_where_the_ranges_ran_narrow_are_named_and_move_no_number(monkeypatch):
    w = gw()
    rng = np.random.default_rng(11)
    normal = rng.normal(0.0, 0.02, 128) + rng.normal(0.0, 0.02, 128)
    heavy = 0.02 * rng.standard_t(3, 128) + rng.normal(0.0, 0.02, 128)
    plain = w.assess(_search(normal), draws=0)
    assert not any("the parent's se" in r or "heavier tails" in r for r in plain.reasons), plain.reasons
    tails = w.assess(_search(heavy), draws=0)
    assert any(r.startswith("the candidates' scores have heavier tails") for r in tails.reasons), tails.reasons
    precise = w.Search(parent=w.Score("seed", 0.0, se=0.005),
                       candidates=[w.Score(f"c{i}", float(s), se=0.02) for i, s in enumerate(normal)])
    v = w.assess(precise, draws=0)
    assert any(r.startswith("the parent's se is 0.25 of a typical candidate's") for r in v.reasons), v.reasons
    for search in (_search(heavy), precise):  # the reasons are text: without them every number is the same
        named = w.assess(search, draws=0).to_json()
        with monkeypatch.context() as m:
            m.setattr(w, "_shape_reasons", lambda d: [])
            bare = w.assess(search, draws=0).to_json()
        assert len(named["reasons"]) == len(bare["reasons"]) + 1
        assert {k: x for k, x in named.items() if k != "reasons"} == {k: x for k, x in bare.items() if k != "reasons"}


def test_the_value_stop_is_stated_not_left_to_the_gate_default(monkeypatch):
    from dataclasses import dataclass

    w = gw()

    @dataclass(frozen=True)
    class WithStop:
        rule: str = "score:judge"
        value_stop: bool = False

    @dataclass(frozen=True)
    class WithoutStop:
        rule: str = "score:judge"

    monkeypatch.setattr(w, "GatePolicy", WithStop)
    assert w.default_policy().value_stop is True  # on, as `loopmath gate` has it, whatever the default
    assert w.default_policy(value_stop=False).value_stop is False
    monkeypatch.setattr(w, "GatePolicy", WithoutStop)  # a gate with no such setting always stops on value
    assert w.default_policy(rule="score:x") == WithoutStop(rule="score:x")


def test_a_clear_winner_keeps_its_lead():
    rng = np.random.default_rng(5)
    scores = list(rng.normal(0.0, 0.02, 15)) + [0.30]
    v = gw().assess(_search(scores), draws=0)
    assert v.winner == "c15"
    assert abs(v.gain["mean"] - v.observed_lead) < 0.03
    assert v.gain["bands"]["95"][0] > 0
    assert v.p_better > 0.999


def test_unseen_candidates_shrink_the_winner_harder():
    """'Best of 64, only the best printed': the 63 others are known only to score at most the best."""
    means, ps = [], []
    for k in (0, 15, 63):
        v = gw().assess(_search([0.06], censored=[(k, 0.06)] if k else ()), draws=0)
        assert v.n_candidates == {"scored": 1, "censored": k, "total": 1 + k}
        means.append(v.gain["mean"])
        ps.append(v.p_better)
    assert means[0] > means[1] > means[2]
    assert ps[0] > ps[1] > ps[2]


def test_a_larger_spread_shrinks_harder():
    rng = np.random.default_rng(9)
    scores = list(rng.normal(0.0, 0.02, 32)) + [0.08]
    search = _search(scores, se=None)
    tight = gw().assess(search, spread=0.01, draws=0)
    loose = gw().assess(search, spread=0.03, draws=0)
    assert tight.observed_lead == loose.observed_lead
    assert loose.gain["mean"] < tight.gain["mean"]
    width = lambda v: v.gain["bands"]["80"][1] - v.gain["bands"]["80"][0]
    assert width(loose) > width(tight)
    assert tight.spread["source"] == "given" and tight.spread["value"] == 0.01


def test_a_spread_range_reports_each_spread():
    """A table with no replicate spread: the answer over a stated range, and one answer integrated over it."""
    rng = np.random.default_rng(9)
    scores = list(rng.normal(0.0, 0.02, 32)) + [0.08]
    v = gw().assess(_search(scores, se=None), spread=(0.01, 0.04), draws=0)
    assert v.spread["source"] == "range"
    rows = v.by_spread
    assert len(rows) == 5
    spreads = [r["spread"] for r in rows]
    assert spreads == sorted(spreads) and spreads[0] == pytest.approx(0.01) and spreads[-1] == pytest.approx(0.04)
    row_means = [r["gain"]["mean"] for r in rows]
    assert min(row_means) - 1e-9 <= v.gain["mean"] <= max(row_means) + 1e-9


def test_a_named_winner_is_used():
    rng = np.random.default_rng(9)
    scores = list(rng.normal(0.0, 0.02, 8))
    v = gw().assess(_search(scores), winner="c3", draws=0)
    assert v.winner == "c3"
    assert v.observed_lead == pytest.approx(scores[3])


def test_lower_is_better_reports_the_improvement():
    v = gw().assess(_search([101.0, 99.0, 90.0, 102.0], parent=100.0, se=1.0, better="lower"), draws=0)
    assert v.winner == "c2"
    assert v.observed_lead == pytest.approx(10.0)  # the improvement, in the score's own direction
    assert v.gain["mean"] > 0 and v.p_better > 0.5
    assert v.to_json()["better"] == "lower"


def test_the_same_seed_gives_the_same_answer():
    search, _ = _null_search()
    a = gw().assess(search, draws=40, seed=3).to_json()
    b = gw().assess(search, draws=40, seed=3).to_json()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["schema"] == "loopmath.gate-winner/1"


def test_doubling_the_grid_moves_the_answer_little():
    """The posterior is a sum over a grid, not a sampler: its convergence check is a finer grid."""
    search, _ = _null_search()
    w = gw()
    base = w.assess(search, draws=0)
    fine = w.assess(search, draws=0, points=tuple(2 * p for p in w.DEFAULT_POINTS))
    assert abs(base.gain["mean"] - fine.gain["mean"]) < 0.002
    for level in ("80", "95"):
        for a, b in zip(base.gain["bands"][level], fine.gain["bands"][level]):
            assert abs(a - b) < 0.003
    assert abs(base.p_better - fine.p_better) < 0.01


def test_the_prior_mass_and_the_evidence_do_not_move_with_the_grid():
    """A spread range weighs each spread by its evidence, which needs the tau prior's mass on the grid to be
    its density times the step in log tau: the mass sums to 1 and the evidence is the same on any grid."""
    from loopmath.gate import winner_model as wm

    prior = wm.default_prior("linear", np.ones(10), np.linspace(0.0, 1.0, 10))
    for n in (24, 48, 96):
        lp = wm._hyper_grid(prior, "ridge", np.geomspace(prior.tau_lo, prior.tau_hi, n), wm._log_tau_norm(prior))[3]
        assert np.exp(lp).sum() == pytest.approx(1.0, abs=0.05), n
    rng = np.random.default_rng(5)
    y = rng.normal(0.0, 1.4, 40)
    d = wm.Data(y_p=0.0, s_p=1.0, y=y, s=np.ones(40), w=int(np.argmax(y)), bound=np.zeros(0),
                count=np.zeros(0, int), s_c=np.zeros(0))
    ev = [wm.fit(d, scale="linear", view="sparse", points=(n, 61)).log_evidence for n in (16, 24, 48, 96)]
    assert max(ev) - min(ev) < 0.01, ev



def test_a_spread_range_weighs_its_spreads_the_same_on_any_grid():
    """Each spread's fit zooms its own tau grid; the weights must come from the evidence alone, so the default
    grid and a fine grid that never zooms give the same weights."""
    w = gw()
    rng = np.random.default_rng(3)
    scores = list(rng.normal(0.0, 0.02, 20)) + [0.09]
    search = _search(scores, se=None)
    base = w.assess(search, spread=(0.005, 0.04), draws=0).spread["weights"]
    fine = w.assess(search, spread=(0.005, 0.04), draws=0, points=(400, 61)).spread["weights"]
    assert max(abs(a - b) for a, b in zip(base, fine)) < 0.005, (base, fine)

def test_quick_recovery_holds_the_80_percent_range():
    """300 simulated searches of 16 candidates with true deviations as large as the noise: the winner's true
    gain falls in its 80% range about 80% of the time (the full recovery study is far larger)."""
    rng = np.random.default_rng(20261020)
    s, hits, n = 0.02, 0, 300
    w = gw()
    for _ in range(n):
        theta = rng.normal(0.0, s, 16)
        y = theta + rng.normal(0.0, s, 16)
        y_p = rng.normal(0.0, s)
        v = w.assess(_search(y, parent=y_p, se=s), draws=0)
        true_gain = theta[int(np.argmax(y))]
        lo, hi = v.gain["bands"]["80"]
        hits += lo <= true_gain <= hi
    assert 0.72 <= hits / n <= 0.88


def test_the_gate_block_is_gate_decide_on_the_confirmatory_runs():
    search, _ = _null_search()
    base = [_row("base", 0.70, "t1"), _row("base", 0.71, "t2")]
    cand = [_row("candidate", 0.72, "t1"), _row("candidate", 0.74, "t2")]
    policy = GatePolicy(rule="score:judge")
    v = gw().assess(search, confirm=(base, cand), policy=policy, draws=0)
    assert v.gate == decide(base, cand, policy).to_json()
    assert v.buy["now"] == v.gate["n_more"]


def test_with_no_confirmatory_runs_the_gate_asks_for_its_first_look():
    search, _ = _null_search()
    v = gw().assess(search, draws=0)
    assert v.gate["action"] == "replicate" and v.gate["n_more"] == 1
    assert v.buy == {"now": 1, "at_least": 2, "expected": None, "at_most": 8}


def test_the_projection_tells_a_clear_winner_from_a_null_one():
    rng = np.random.default_rng(5)
    clear = gw().assess(_search(list(rng.normal(0.0, 0.02, 15)) + [0.30]), draws=60, seed=1)
    null = gw().assess(_null_search()[0], draws=60, seed=1)
    assert clear.projection["p_adopt"] > 0.9
    assert null.projection["p_adopt"] < 0.3
    for v in (clear, null):
        p = v.projection
        assert p["draws"] == 60 and 1 <= p["mean_more"] <= 8
        assert v.buy["expected"] == pytest.approx(p["mean_more"])


@pytest.mark.parametrize(
    "build, needle",
    [
        (lambda w: w.Search(parent=None, candidates=[w.Score("a", 0.5, se=0.1)]), "parent"),
        (lambda w: w.Search(parent=w.Score("p", 0.5, se=0.1), candidates=[]), "candidate"),
        (lambda w: w.Search(parent=w.Score("p", 0.5, se=0.1), candidates=[w.Score("a", float("nan"), se=0.1)]),
         "finite"),
        (lambda w: w.Search(parent=w.Score("p", 0.5, se=0.1), candidates=[w.Score("a", 0.6, se=0.1)],
                            censored=[w.Censored(count=3, at_most=0.9, se=0.1)]), "at_most"),
        (lambda w: w.Search(parent=w.Score("p", 0.5), candidates=[w.Score("a", 0.6)],
                            censored=[w.Censored(count=3, at_most=0.55, reps=0)], spread=0.1), "reps"),
    ],
)
def test_bad_searches_are_refused_by_name(build, needle):
    w = gw()
    with pytest.raises(ValueError, match=needle):
        w.assess(build(w), draws=0)


def test_a_zero_floor_gives_p_above_floor():
    """A floor of 0 is a floor: P(D > 0) is reported as P(gain above the floor), not left out."""
    search, _ = _null_search()
    v = gw().assess(search, policy=GatePolicy(floor=0.0), draws=0)
    assert v.floor == 0.0 and v.p_above_floor == pytest.approx(v.p_better)
    assert gw().assess(search, draws=0).p_above_floor is None


@pytest.mark.parametrize("spread", [None, 0.0, -0.1, (0.03, 0.01)])
def test_a_missing_or_bad_spread_is_refused(spread):
    w = gw()
    search = w.Search(parent=w.Score("p", 0.5), candidates=[w.Score("a", 0.6), w.Score("b", 0.55)])
    with pytest.raises(ValueError, match="spread"):
        w.assess(search, spread=spread, draws=0)


# The projection: bounded draws keep the drawn gain, and it starts from the gate's actual state.


def _fraction_search(se=0.02):
    rng = np.random.default_rng(13)
    scores = list(0.70 + rng.normal(0.0, 0.02, 32)) + [0.76]
    return _search(scores, parent=0.70, se=se, scale="fraction")


def _priced(row, usd):
    return {**row, "cost": {"usd": usd}}


def test_beta_draws_match_the_mean_and_spread_at_the_boundary():
    """Means 0.90 and 0.95 with a spread of 0.10: clipping normals would project a gain of 0.0386, not 0.05."""
    w = gw()
    for m in (0.90, 0.95):
        a, b = w.beta_params(m, 0.10)
        assert a / (a + b) == pytest.approx(m, abs=1e-12)
        assert a * b / ((a + b) ** 2 * (a + b + 1)) == pytest.approx(0.01, abs=1e-12)
    rng = np.random.default_rng(1)
    k = 200_000
    base = w.draw_evaluations(0.90, 0.10, "fraction", rng, k)
    cand = w.draw_evaluations(0.95, 0.10, "fraction", rng, k)
    assert base.min() >= 0 and cand.max() <= 1
    mc_se = float(np.sqrt((base.var() + cand.var()) / k))
    assert abs((cand.mean() - base.mean()) - 0.05) < 4 * mc_se


@pytest.mark.parametrize("m, sd", [(0.99, 0.2), (1.0, 0.01), (0.0, 0.01), (1.2, 0.01)])
def test_an_infeasible_bounded_draw_is_refused(m, sd):
    with pytest.raises(ValueError, match="feasible"):
        gw().beta_params(m, sd)


def test_a_projection_past_the_bound_is_withheld_and_the_gate_still_answers():
    rng = np.random.default_rng(3)
    scores = list(0.985 + rng.normal(0.0, 0.003, 16)) + [0.995]
    v = gw().assess(_search(scores, parent=0.985, se=0.15, scale="fraction"), draws=40, seed=1)
    assert v.projection["withheld"] is True
    assert v.projection["infeasible_share"] > 0.02
    assert "bound" in v.projection["reason"]
    assert v.gate["action"] == "replicate" and v.buy["now"] == 1 and v.buy["expected"] is None


def test_a_partly_confirmed_winner_projects_from_the_held_pair():
    base, cand = [_row("base", 0.70, "t1")], [_row("candidate", 0.72, "t1")]
    w = gw()
    v = w.assess(_fraction_search(), confirm=(base, cand), draws=40, seed=2)
    assert v.gate["action"] == "replicate" and v.gate["n_more"] == 1
    p = v.projection
    assert p["withheld"] is False and p["held_pairs"] == 1
    assert 0 <= p["mean_more"] <= 7
    assert v.buy == {"now": 1, "at_least": 1, "expected": pytest.approx(p["mean_more"]), "at_most": 7}
    assert v.gain_confirmed is not None and v.gain_confirmed["mean"] != v.gain["mean"]
    assert w.assess(_fraction_search(), draws=0).gain_confirmed is None


def test_a_gate_that_has_stopped_buys_nothing():
    base = [_row("base", 0.50 + 0.01 * i, f"t{i}") for i in range(4)]
    cand = [_row("candidate", 0.90 + 0.005 * i, f"t{i}") for i in range(4)]
    v = gw().assess(_fraction_search(), confirm=(base, cand), draws=40, seed=2)
    assert v.gate["action"] == "adopt"
    assert v.buy == {"now": 0, "at_least": 0, "expected": 0, "at_most": 0}
    assert v.projection["p_adopt"] == 1.0 and v.projection["mean_more"] == 0 and v.projection["draws"] == 0


def test_a_budget_needs_priced_runs_and_projects_dollars_when_they_are():
    base = [_row("base", 0.70, "t1"), _row("base", 0.70, "t2")]
    cand = [_row("candidate", 0.70, "t1"), _row("candidate", 0.701, "t2")]
    policy = GatePolicy(rule="score:judge", budget_usd=10.0)
    with pytest.raises(ValueError, match="budget_usd"):
        gw().assess(_fraction_search(), confirm=(base, cand), policy=policy, draws=20)
    base = [_priced(r, 0.5) for r in base]
    cand = [_priced(r, 0.6) for r in cand]
    v = gw().assess(_fraction_search(), confirm=(base, cand), policy=policy, draws=20, seed=4)
    assert v.projection["withheld"] is False
    assert v.projection["usd_more"] == pytest.approx(1.1 * v.projection["mean_more"])


def test_priced_held_runs_price_the_projected_runs_without_a_budget(monkeypatch):
    """The gate prices a pair (and stops on value) from the runs' costs, budget or not: a projected run costs
    what its arm's held runs did, so the projected gate sees the same prices as the real one."""
    w = gw()
    seen = []

    def spy(base, cand, policy):
        d = decide(base, cand, policy)
        seen.append((list(base), list(cand), d))
        return d

    monkeypatch.setattr(w, "decide", spy)
    base, cand = [_priced(_row("base", 0.70, "t1"), 0.5)], [_priced(_row("candidate", 0.72, "t1"), 0.6)]
    v = w.assess(_fraction_search(), confirm=(base, cand), policy=GatePolicy(rule="score:judge"), draws=20, seed=4)
    projected = [(r, usd) for b, c, _ in seen for rows, usd in ((b, 0.5), (c, 0.6)) for r in rows
                 if r["task_id"].startswith("projected-")]
    assert projected, "the projection drew no further pairs"
    assert all(r.get("cost") == {"usd": usd} for r, usd in projected)
    # and the projected gate prices its pairs, as the real one does on priced runs: no run unrecorded
    later = [d for b, _, d in seen if any(r["task_id"].startswith("projected-") for r in b)]
    assert later and all(d.cost["unrecorded"] == 0 and d.cost["pair_usd"] is not None for d in later)
    assert v.projection["usd_more"] == pytest.approx(1.1 * v.projection["mean_more"])


# The runs format: a candidate's score is its mean of task means, with the suite-mean standard error.

RUNS = {  # arm -> task -> replicate scores (unequal replicate counts on purpose)
    "base": {"t1": [0.60, 0.64], "t2": [0.50, 0.46], "t3": [0.80]},
    "c1": {"t1": [0.70, 0.66, 0.71], "t2": [0.52], "t3": [0.81, 0.85]},
    "c2": {"t1": [0.62], "t2": [0.55, 0.49], "t3": [0.90, 0.88]},
}


def _runs_rows(runs):
    return [_row(arm, x, task) for arm, tasks in runs.items() for task, xs in tasks.items() for x in xs]


def test_runs_give_the_suite_mean_and_its_standard_error_exactly():
    tasks = ["t1", "t2", "t3"]
    sigma2 = {}
    for t in tasks:  # pooled over the arms with two or more runs on the task
        ss = df = 0.0
        for arm in RUNS:
            xs = RUNS[arm][t]
            if len(xs) >= 2:
                mean = sum(xs) / len(xs)
                ss += sum((x - mean) ** 2 for x in xs)
                df += len(xs) - 1
        sigma2[t] = ss / df
    T = len(tasks)
    search = gw().search_from_runs(_runs_rows(RUNS), parent="arm:base", rule="score:judge")
    got = {c.id: c for c in [search.parent, *search.candidates]}
    for arm in RUNS:
        y = sum(sum(RUNS[arm][t]) / len(RUNS[arm][t]) for t in tasks) / T
        se = (sum(sigma2[t] / len(RUNS[arm][t]) for t in tasks) / T ** 2) ** 0.5
        assert got[arm].score == pytest.approx(y, abs=1e-12)
        assert got[arm].se == pytest.approx(se, abs=1e-12)
        assert got[arm].sd == pytest.approx((sum(sigma2.values())) ** 0.5 / T, abs=1e-12)
    assert search.spread == pytest.approx((sum(sigma2.values())) ** 0.5 / T, abs=1e-12)
    assert search.spread_source == "runs"


def test_runs_with_a_stated_spread_use_it_for_every_task():
    search = gw().search_from_runs(_runs_rows(RUNS), parent="arm:base", rule="score:judge", spread=0.05)
    c1 = next(c for c in search.candidates if c.id == "c1")
    assert c1.se == pytest.approx(0.05 * (1 / 3 + 1 + 1 / 2) ** 0.5 / 3, abs=1e-12)
    assert search.spread_source == "given"


def test_runs_on_different_task_sets_are_refused_by_arm_and_task():
    runs = {arm: dict(tasks) for arm, tasks in RUNS.items()}
    del runs["c2"]["t3"]
    with pytest.raises(ValueError, match=r"c2.*t3"):
        gw().search_from_runs(_runs_rows(runs), parent="arm:base", rule="score:judge")
