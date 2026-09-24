"""The fitted state: tree law, misclassification, score rule switch, summaries, D8, D9, D10 (spec 04 sections 2, 6, 8)."""

from __future__ import annotations

import json
import math
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief.design import run_row, structure, task_from_doc
from loopmath.belief.state import load, load_latest
from loopmath.types import AcceptanceRule, Setting, ScoreTarget, Task

PERF_RULE = AcceptanceRule("perf>=2100", "perf reaches 2100", requires=(), score=ScoreTarget("perf", 2100.0, "higher"))


@pytest.fixture(scope="module")
def s(sim_fit):
    return sim_fit["state"]


@pytest.fixture(scope="module")
def task(sim_fit):
    return task_from_doc(sim_fit["docs"][0])


def _cfg(model, harness="claude-code", effort="high", wf=simdata.SOLO):
    return simdata.config(wf, Setting(harness, model, effort))


def _width(iv):
    return iv.hi - iv.lo


def test_load_and_predict_shapes(s, task, sim_fit):
    assert s.fit_id == sim_fit["path"].name and s.created_at
    p = s.predict(task, simdata.config(simdata.SWEEP, simdata.SETTINGS[0]))
    assert 0 < p.p_success.lo <= p.p_success.mean <= p.p_success.hi < 1
    assert 0 < p.cost.usd.lo < p.cost.usd.mean < p.cost.usd.hi
    assert p.cost.tokens.mean > 1000 and 1 <= p.rounds.mean <= 3
    assert set(p.per_piece) == {"plan", "implement", "review"}
    assert p.per_piece["plan"].gate_pass is None and p.per_piece["implement"].gate_pass is not None
    # D60: piece costs are full-run contributions that add up to the run; cost_per_round is one execution
    assert math.isclose(sum(pp.cost.usd.mean for pp in p.per_piece.values()), p.cost.usd.mean, rel_tol=1e-9)
    plan, impl = p.per_piece["plan"], p.per_piece["implement"]
    assert plan.cost_per_round == plan.cost  # runs once
    assert math.isclose(impl.cost_per_round.usd.mean, impl.cost.usd.mean / impl.rounds.mean, rel_tol=1e-9)  # round effects
    assert impl.cost_per_round.usd.lo < impl.cost_per_round.usd.mean < impl.cost_per_round.usd.hi
    assert p.success_from == "success_head" and "perf" in p.scores
    again = s.predict(task, simdata.config(simdata.SWEEP, simdata.SETTINGS[0]))
    assert again == p  # draws and simulation seeds are fixed by the fit


def _row_eval(s, head, task, cfg, drop=()):
    terms = [t for t in run_row(task, "user", structure(cfg)) if t[0] not in drop]
    mu, D = s.heads[head].eval_rows([terms])
    return mu[0], D[0]


def _widening(s, head, nodes):
    """The draws an unseen node set adds: each node's fitted scale times its own standard normals."""
    h = s.heads[head]
    return sum(h.node_scale(n) * h.virtual_unit(n) for n in nodes)


def test_tree_law_a_new_version_predicts_its_family_with_a_wider_interval(s, task):
    cfg = _cfg("claude-opus-9-7")
    for head in ("cost", "success"):
        mu_new, d_new = _row_eval(s, head, task, cfg)
        mu_fam, d_fam = _row_eval(s, head, task, cfg, drop=("model:claude-opus-9-7",))
        assert mu_new == pytest.approx(mu_fam)  # the family's mean
        widen = _widening(s, head, ["model:claude-opus-9-7"])
        np.testing.assert_allclose(d_new - d_fam, widen, atol=1e-12)  # wider by the version scale
        assert s.heads[head].node_scale("model:claude-opus-9-7") == s.heads[head].phi["model"] > 0
    new = s.predict(task, cfg)
    assert new.support == 0 and s.predict(task, _cfg("claude-opus-9")).support > 0
    # every configuration that uses the unseen version shares its draws
    other = simdata.config(simdata.IR, Setting("claude-code", "claude-opus-9-7", "high"))
    _, d_other = _row_eval(s, "success", task, other)
    _, d_other_fam = _row_eval(s, "success", task, other, drop=("model:claude-opus-9-7",))
    np.testing.assert_allclose(d_other - d_other_fam, _widening(s, "success", ["model:claude-opus-9-7"]), atol=1e-12)


def test_tree_law_a_new_family_starts_from_its_provider(s, task):
    cfg = _cfg("gpt-9-nova", "codex")
    unseen = ["family:nova", "model:gpt-9-nova", "family_effort:nova|high", "role_family:implementer|nova"]
    for head in ("cost", "success"):
        mu_new, d_new = _row_eval(s, head, task, cfg)
        mu_prov, d_prov = _row_eval(s, head, task, cfg, drop=unseen)
        assert mu_new == pytest.approx(mu_prov)  # the provider's mean
        np.testing.assert_allclose(d_new - d_prov, _widening(s, head, unseen), atol=1e-12)
    assert "provider:openai" in s.heads["cost"].index


def test_misclassification_widens_the_success_head(tmp_path):
    docs, _ = simdata.simulate(300, seed=31, source="live")
    noisy = json.loads(json.dumps(docs))
    for doc in noisy:
        for sig in doc["run"]["signals"]:
            sig["tier"] = "asserted"  # q 0.7 on the same z
    now = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
    sure = load(F.fit(tmp_path / "sure", docs=docs, no_prior=True, now=now))
    unsure = load(F.fit(tmp_path / "unsure", docs=noisy, no_prior=True, now=now))
    task = task_from_doc(docs[0])
    for cfg in simdata.all_configs()[:6]:
        assert _width(unsure.predict(task, cfg).p_success) > _width(sure.predict(task, cfg).p_success)
    assert np.std(unsure.heads["success"].draws, axis=1).mean() > np.std(sure.heads["success"].draws, axis=1).mean()


def test_score_rule_switch(s, task):
    cfg = simdata.config(simdata.IR, simdata.SETTINGS[2])
    binary = s.predict(task, cfg)
    scored = s.predict(task, cfg, PERF_RULE)
    assert binary.success_from == "success_head" and scored.success_from == "score_head"
    assert scored.p_success.mean == pytest.approx(scored.scores["perf"].p_reach, abs=1e-12)
    assert binary.scores["perf"].p_reach is None
    # a task type without score runs keeps the success head
    docs_task = Task("tsk_new", "docs", "acme/new")
    assert s.predict(docs_task, cfg, PERF_RULE).success_from == "success_head"


def test_rescue_enters_ell(s, task):
    cfg = simdata.config(simdata.PIR, simdata.SETTINGS[4])
    base = s.predict(task, cfg)
    rescued = s.predict(task, cfg, rescue_usd=10.0)
    assert base.ell.usd.mean == pytest.approx(base.cost.usd.mean)
    assert rescued.ell.usd.mean == pytest.approx(base.cost.usd.mean + (1 - base.p_success.mean) * 10.0)
    assert rescued.ell.tokens.mean > base.ell.tokens.mean


def test_success_and_ell_draws_are_joint_and_aligned(s, task):
    cfgs = simdata.all_configs()[:3]
    g = s.success_draws(task, cfgs)
    ell = s.ell_draws(task, cfgs, rescue_usd=5.0)
    assert g.shape == ell.shape == (3, 400)
    preds = s.predict_many(task, cfgs, rescue_usd=5.0)
    for i, p in enumerate(preds):
        assert g[i].mean() == pytest.approx(p.p_success.mean)
        assert ell[i].mean() == pytest.approx(p.ell.usd.mean)


def test_predict_many_equals_predict(s, task):
    cfgs = simdata.all_configs()[:8]
    many = s.predict_many(task, cfgs)
    assert [p.config for p in many] == [c.id for c in cfgs]
    assert many[3] == s.predict(task, cfgs[3])


def test_support_and_node_summary(s, task, sim_fit):
    sup = s.support(task)
    assert sup["all"] == sup["user"] == 800
    assert sup["type"] >= sup["repo"] >= sup["task"] > 0
    rows = s.node_summary("model", "cost")
    by_key = {r.key: r for r in rows}
    assert {"anthropic", "opus", "claude-opus-9", "gpt-9-astra"} <= set(by_key)
    r = by_key["claude-opus-9"]
    assert r.level == "model" and r.parent == "family:opus" and r.support > 0 and r.effect.mean > 0
    assert r.source_mix == {"user": r.support}
    truth = sim_fit["truth"]
    implied = math.exp(sum(truth.value("cost", n) for n in ("model:claude-opus-9", "family:opus",
                                                            "provider:anthropic")))
    assert r.effect.lo / 1.5 < implied < r.effect.hi * 1.5
    # success effects show in pp at the baseline; percentiles commute with that monotone map
    succ = {r.key: r for r in s.node_summary("model", "success")}["claude-opus-9"]
    base = s.heads["success"].baseline

    def pp(x):
        return 100 * (1 / (1 + math.exp(-(base + x))) - 1 / (1 + math.exp(-base)))

    assert succ.display.lo == pytest.approx(pp(succ.effect.lo), abs=0.5)
    assert succ.display.hi == pytest.approx(pp(succ.effect.hi), abs=0.5)
    assert all(r.level.startswith("feature:") for r in s.node_summary("feature"))


def test_conditioned_shrinks_variance_and_keeps_means(s, task):
    cfg = simdata.config(simdata.IR, simdata.SETTINGS[5])
    after = s.conditioned(task, cfg)
    g0, g1 = s.success_draws(task, [cfg])[0], after.success_draws(task, [cfg])[0]
    assert np.std(g1) < np.std(g0)
    before_eta = s.heads["success"].eval_rows([run_row(task, "user", structure(cfg))])[0][0]
    after_eta = after.heads["success"].eval_rows([run_row(task, "user", structure(cfg))])[0][0]
    assert after_eta == pytest.approx(before_eta)
    other = simdata.config(simdata.SOLO, simdata.SETTINGS[0])
    assert np.std(after.success_draws(task, [other])[0]) <= np.std(s.success_draws(task, [other])[0]) + 1e-12
    # conditioning on an unseen model promotes its virtual node into the head
    new = _cfg("claude-opus-9-7")
    grown = s.conditioned(task, new)
    assert "model:claude-opus-9-7" in grown.heads["cost"].index
    assert np.std(grown.success_draws(task, [new])[0]) < np.std(s.success_draws(task, [new])[0])


def test_load_latest_without_a_fit(tmp_path):
    assert load_latest(tmp_path) is None


def _new_task(task, i, repo="someone/new-repo"):
    return Task.from_dict({**task.to_dict(), "id": f"tsk_unseen_{i:03d}", "repo": repo})


def _means(p):
    out = [p.p_success.mean, p.cost.usd.mean, p.cost.tokens.mean, p.ell.usd.mean, p.ell.tokens.mean, p.rounds.mean]
    for pp in p.per_piece.values():
        out += [pp.cost.usd.mean, pp.rounds.mean, pp.gate_pass.mean if pp.gate_pass else 0.0,
                pp.cost_per_round.usd.mean if pp.cost_per_round else 0.0]
    out += [sc.value.mean for sc in p.scores.values()] + [sc.p_reach or 0.0 for sc in p.scores.values()]
    return np.array(out)


def test_new_task_means_do_not_depend_on_the_task_id(s, task):
    """D98: an unseen task's effect is integrated out of the means, so two new tasks that differ
    only in id get the same means and the same look-ahead gain; their draws still differ."""
    sweep = simdata.config(simdata.SWEEP, simdata.SETTINGS[0])
    cfgs = [sweep, simdata.config(simdata.IR, simdata.SETTINGS[2]), simdata.config(simdata.SOLO, simdata.SETTINGS[1])]
    a, b = _new_task(task, 1), _new_task(task, 2)
    for rule in (None, PERF_RULE):
        pa = s.predict_many(a, cfgs, rule, rescue_usd=4.0)
        pb = s.predict_many(b, cfgs, rule, rescue_usd=4.0)
        for x, y in zip(pa, pb):
            np.testing.assert_allclose(_means(x), _means(y), rtol=1e-12)
    assert not np.allclose(s.success_draws(a, [sweep]), s.success_draws(b, [sweep]))
    la = s.lookahead(a, cfgs[1], sweep, cfgs, rescue_usd=4.0).gain_per_run
    lb = s.lookahead(b, cfgs[1], sweep, cfgs, rescue_usd=4.0).gain_per_run
    assert la == pytest.approx(lb, rel=1e-12)


def test_new_task_means_are_what_the_draws_estimate(s, task):
    """D98: averaged over many new tasks in new repos (each with its own seeded draws of the unseen
    effects), the draws' means converge to the integrated means."""
    cfg = simdata.config(simdata.SWEEP, simdata.SETTINGS[0])
    new = [_new_task(task, i, repo=f"someone/repo-{i}") for i in range(200)]
    g = np.mean([s.success_draws(t, [cfg]).mean() for t in new])
    cost = np.mean([s.ell_draws(t, [cfg]).mean() for t in new])
    ell = np.mean([s.ell_draws(t, [cfg], rescue_usd=4.0).mean() for t in new])
    reach = np.mean([s.success_draws(t, [cfg], PERF_RULE).mean() for t in new])
    p = s.predict(_new_task(task, 999, repo="someone/repo-999"), cfg, rescue_usd=4.0)
    scored = s.predict(_new_task(task, 999, repo="someone/repo-999"), cfg, PERF_RULE)
    assert scored.success_from == "score_head"
    assert g == pytest.approx(p.p_success.mean, rel=0.01)
    assert reach == pytest.approx(scored.p_success.mean, rel=0.01)
    assert cost == pytest.approx(p.cost.usd.mean, rel=0.02)
    assert ell == pytest.approx(p.ell.usd.mean, rel=0.02)


def _reference_state(cfg, level: str, new: Task):
    """A fit that knows every node of the configuration's rows at effect 0, except those of one
    level, which are unseen with scale 1 on the cost head and 0 elsewhere (D106)."""
    from pathlib import Path

    from loopmath.belief.design import task_terms
    from loopmath.belief.state import FitState, HeadState

    plan = FitState(Path("."), meta={"fit": "reference"}, design={})._plan(cfg)
    rows = [*plan["cost"].values(), *plan["gate"].values(), plan["run"], tuple(task_terms(new, "user"))]
    nodes = sorted({t[0] for r in rows for t in r if not t[0].startswith(level + ":")})
    p = len(nodes)
    heads = {name: HeadState(name, {"sigma": 0.0, "phi": {level: 1.0 if name == "cost" else 0.0}}, nodes, None,
                             "reference", mean=np.zeros(p), U=np.zeros((p, p)), draws=np.zeros((p, 400)))
             for name in ("cost", "tokens", "gate", "success")}
    return FitState(Path("."), meta={"fit": "reference"}, heads=heads, design={})


def _mean_inverse(a: list[float]) -> float:
    """E[1 / sum_i a_i exp(u_i)] for independent u_i ~ N(0, 1), by a dense Gauss-Hermite grid."""
    from numpy.polynomial.hermite_e import hermegauss

    x, w = hermegauss(40)
    w = w / w.sum()
    grid = np.meshgrid(*[x] * len(a), indexing="ij")
    weight = np.prod(np.meshgrid(*[w] * len(a), indexing="ij"), axis=0)
    return float(np.sum(weight / sum(ai * np.exp(g) for ai, g in zip(a, grid))))


@pytest.mark.parametrize("level", ["task", "model"])
def test_the_ell_tokens_mean_is_the_mean_of_the_quantity_its_draws_hold(level):
    """D106, the reviewer's reference case: one piece, cost C = exp(e) with e ~ N(0, 1) the unseen
    task (or model) effect, tokens T = 1, g = 0.5 and a $2 rescue. The draws hold
    1 + (1 - g) 2 T / C = 1 + exp(-e), whose mean is 1 + exp(1 / 2), not the ratio of expectations
    1 + exp(-1 / 2)."""
    new = Task.from_dict({"id": "tsk_unseen", "type": "feature", "repo": "synthetic/repo"})
    cfg = simdata.config(simdata.SOLO, simdata.SETTINGS[0])
    pred = _reference_state(cfg, level, new).predict(new, cfg, rescue_usd=2.0)
    assert pred.cost.usd.mean == pytest.approx(math.exp(0.5), rel=1e-9)
    assert pred.cost.tokens.mean == pytest.approx(1.0, rel=1e-9)
    assert pred.p_success.mean == pytest.approx(0.5, rel=1e-9)
    assert pred.ell.tokens.mean == pytest.approx(2.64872127, rel=1e-8)
    assert pred.ell.tokens.lo <= pred.ell.tokens.mean <= pred.ell.tokens.hi


@pytest.mark.parametrize("shape,settings,weights", [
    ("bon", (0, 0), [4.0]),  # one model in both pieces: its effect is common
    ("bon", (0, 4), [3.0, 1.0]),  # three implementers on one unseen model, the referee on another
    ("pir", (0, 4, 5), [1.0, 1.75, 1.75]),  # three unseen models, implement and review in a loop
])
def test_the_ell_tokens_mean_integrates_unseen_models_across_pieces(shape, settings, weights):
    """D106 with several pieces: each piece's cost is exp(u) of its model's unseen effect times its
    expected executions (width, and 1 + 1/2 + 1/4 in the loop, as the gate passes half the time),
    tokens 1 per execution, g = 0.5, a $2 rescue. The ell.tokens mean is T + (1 - g) 2 T E[1 / C]."""
    new = Task.from_dict({"id": "tsk_unseen", "type": "feature", "repo": "synthetic/repo"})
    wf = {"bon": simdata.BON, "pir": simdata.PIR}[shape]
    cfg = simdata.config(wf, *(simdata.SETTINGS[i] for i in settings))
    pred = _reference_state(cfg, "model", new).predict(new, cfg, rescue_usd=2.0)
    T = sum(weights)
    assert [pp.rounds.mean for pp in pred.per_piece.values()] == pytest.approx(
        [1.0, 1.75, 1.75] if shape == "pir" else [1.0, 1.0])
    assert pred.cost.tokens.mean == pytest.approx(T, rel=1e-9)
    assert pred.cost.usd.mean == pytest.approx(T * math.exp(0.5), rel=1e-9)
    assert pred.ell.tokens.mean == pytest.approx(T + 0.5 * 2.0 * T * _mean_inverse(weights), rel=5e-5)
    assert pred.ell.tokens.lo <= pred.ell.tokens.mean <= pred.ell.tokens.hi


def _chain(n: int):
    """A valid custom workflow of `n` sequential pieces, each with its own position node."""
    from loopmath.types import Control, Piece, Workflow
    from loopmath.workflows.format import validate_configuration

    pieces = tuple(Piece(f"step{i}", "implementer") for i in range(n))
    arts = tuple(f"artifact{i}" for i in range(n))
    edges = [(p.id, a) for p, a in zip(pieces, arts)] + [(arts[i], pieces[i + 1].id) for i in range(n - 1)]
    cfg = simdata.config(Workflow("custom_chain", 1, "Custom sequential workflow", pieces, arts, tuple(edges),
                                  Control()), simdata.SETTINGS[0])
    assert not validate_configuration(cfg)
    return cfg


def _mean_inverse_iid(n: int) -> float:
    """E[1 / sum of n exp(u_i)] for independent u_i ~ N(0, 1), through the Laplace transform:
    the integral over t > 0 of E[exp(-t exp(u))] ** n, inner Gauss-Hermite, outer on log t."""
    from numpy.polynomial.hermite_e import hermegauss

    x, w = hermegauss(80)
    s = np.linspace(-30.0, 15.0, 20001)
    inner = np.exp(-np.outer(np.exp(s), np.exp(x))) @ (w / w.sum())
    return float(np.trapezoid(inner ** n * np.exp(s), s))


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5, 7, 19, 60, 400])
def test_the_inverse_cost_rule_is_bounded(k):
    """D108: whatever the number of axes, the rule has at most MAX_POINTS points: a Gauss-Hermite
    grid up to four axes, a fixed Sobol set beyond."""
    from loopmath.belief import state as S

    pts, wts = S._rule(k)
    assert len(wts) <= S.MAX_POINTS and pts.shape == (len(wts), k)
    assert wts.sum() == pytest.approx(1.0)
    assert np.allclose(wts, wts[0]) == (k == 0 or k > 4)
    assert np.all(np.isfinite(pts))
    if k:
        assert np.abs(wts @ pts).max() < 0.05 and np.abs(wts @ pts ** 2 - 1.0).max() < 0.2
    assert np.array_equal(S._rule.__wrapped__(k)[0], pts)  # the same points every time


def test_a_long_chain_integrates_within_the_point_budget(monkeypatch):
    """D108, the reviewer's case: 20 sequential pieces, each with its own unseen position (scale 1
    on cost), give 20 groups and 19 axes. The product grid would have 2 ** 19 points; the rule
    uses MAX_POINTS, and the ell.tokens mean stays within 1% of the exact value
    T + (1 - g) 2 T E[1 / C], T = 20, g = 0.5."""
    from loopmath.belief import state as S

    new = Task.from_dict({"id": "tsk_unseen", "type": "feature", "repo": "synthetic/repo"})
    cfg = _chain(20)
    seen = []
    rule = S._rule
    monkeypatch.setattr(S, "_rule", lambda k: seen.append(k) or rule(k))
    pred = _reference_state(cfg, "position", new).predict(new, cfg, rescue_usd=2.0)
    assert seen == [19] and len(rule(19)[1]) <= S.MAX_POINTS
    assert pred.cost.tokens.mean == pytest.approx(20.0, rel=1e-9)
    assert pred.ell.tokens.mean == pytest.approx(20.0 + 20.0 * _mean_inverse_iid(20), rel=0.01)


@pytest.mark.parametrize("rescue", [None, 0.0])
def test_without_a_rescue_the_inverse_cost_is_not_computed(monkeypatch, rescue):
    """D108: the rescue's tokens per dollar only matter when there is a rescue."""
    from loopmath.belief import state as S

    def fail(*_a, **_k):
        raise AssertionError("_inv_cost called without a rescue")

    new = Task.from_dict({"id": "tsk_unseen", "type": "feature", "repo": "synthetic/repo"})
    cfg = _chain(20)
    monkeypatch.setattr(S.FitState, "_inv_cost", fail)
    pred = _reference_state(cfg, "position", new).predict(new, cfg, rescue_usd=rescue)
    assert pred.ell.tokens.mean == pytest.approx(pred.cost.tokens.mean, rel=1e-12)
