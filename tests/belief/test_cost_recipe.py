"""The source-only setting cost law shares FrontierCost's Gaussian predictions."""

import math

import numpy as np
import pytest
from loopmath.belief.cost_recipe import (GROUPS, anchor, describe, fit_recipe, noise_variance,
                                        row_terms, source_prior)
from loopmath.belief.fit import HeadRows
from loopmath.belief.forest import Forest
from loopmath.belief.lookahead import _Rows, _condition_head
from loopmath.belief.state import HeadState
from loopmath.frontier.cost import FrontierCost


def terms(model, effort, position="solo#0"):
    return [("fixed:intercept", None, 1.), ("task:one", None, 1.),
            (f"model:{model}", None, 1.), (f"effort:{effort}", None, 1.),
            (f"position:{position}", None, 1.), ("fsrc:family|user", None, 1.)]


ROWS = [terms("claude-opus-5", "low"), terms("claude-opus-5", "max"),
        terms("unknown-model", "high")]


def direct(recipe, rows):
    descs = [describe(row) for row in rows]
    values = [anchor(recipe, model, effort) for _, model, effort in descs]
    model = np.array([d[1] for d in descs])
    sd = np.array([v[1] for v in values])
    E = (np.array([d[2] for d in descs])[:, None] >= np.arange(1, 5)).astype(float)
    cm = FrontierCost(np.array([v[0] for v in values]), components={
        "level": np.full((len(rows), len(rows)), .25),
        "transfer": (model[:, None] == model[None, :]) * np.outer(sd, sd),
        "setting": np.eye(len(rows)) * .09, "steps": E @ E.T * .04})
    cm._set_widths(np.array([recipe["widths"][g] for g in GROUPS]))
    cm.fit()
    return cm


def build(observations=(), *, eb=False, foreign=(), weights=None):
    hr = HeadRows("cost", "gaussian", "cost")
    for i, (row, value) in enumerate(observations):
        hr.add(row, math.log(value), 1. if weights is None else weights[i], f"run_{i}", "user", "test")
    for i, (row, value) in enumerate(foreign):
        hr.add(row, math.log(value), 1., f"foreign_{i}", "sweep", "test")
    fitted = fit_recipe(hr, Forest(), eb=eb)
    result = fitted["fit"]
    head = HeadState("cost", {"recipe": fitted["recipe"], "sigma": result.sigma}, fitted["node_ids"],
                     None, "test", mean=result.mean, U=result.cov_factor)
    return head


def covariance(rows):
    return rows.B @ rows.B.T + (rows.Xv.multiply(rows.vphi2) @ rows.Xv.T).toarray()


def test_source_prior_is_mean_preserving_and_matches_four_kernels():
    head = build()
    rows = _Rows.build(head, ROWS)
    cm = direct(head.recipe, ROWS)
    np.testing.assert_allclose(rows.mu, cm.mc, atol=1e-12)
    np.testing.assert_allclose(covariance(rows), cm.Sc, atol=1e-12)
    np.testing.assert_allclose(np.exp(rows.mu + (rows.var + rows.noise) / 2), cm.anchor, rtol=1e-12)
    assert head.row_terms([("task:unseen", None, 1.)]) == []
    assert head.node_scale("fsrc:unseen|user") == 0
    assert head.node_scale("task:unseen") == 0


def test_source_anchor_is_independent_of_candidate_order_and_task_identity():
    recipe = source_prior()
    assert "gpt-5.6-sol" in recipe["tb"] and "gemini-3.8-flash" in recipe["tb"]
    assert "gpt-5-6-sol" not in recipe["tb"]
    assert anchor(recipe, "gpt-5.6-sol", 4)[0] == pytest.approx(7.670741187878788)
    a = row_terms(recipe, ROWS[0])
    row_terms(recipe, ROWS[-1])
    b = row_terms(recipe, [(n.replace("task:one", "task:other"), p, v) for n, p, v in ROWS[0]])
    assert a == b
    assert row_terms(recipe, a) == a


@pytest.mark.parametrize("eb", [False, True])
def test_conditioning_and_joint_width_refit_match_frontier(eb):
    observed = [(ROWS[0], v) for v in (1., 2., 3., 1.5)] + [(ROWS[1], v) for v in (8., 9., 11.)]
    head = build(observed, eb=eb)
    # Foreign costs cannot change a source-only anchor or the user's likelihood.
    foreign = build(observed, eb=eb, foreign=[(ROWS[0], 1e8), (ROWS[2], .0001)])
    assert head.recipe == foreign.recipe
    np.testing.assert_array_equal(head.mean, foreign.mean)
    cm = direct(head.recipe, ROWS)
    for row, value in observed:
        cm.add(0 if row == ROWS[0] else 1, value)
    cm.fit()
    rows = _Rows.build(head, ROWS)
    np.testing.assert_allclose(rows.mu, cm.mc, atol=1e-10)
    np.testing.assert_allclose(covariance(rows), cm.Sc, atol=1e-10)
    np.testing.assert_allclose(rows.noise, cm.sig2, atol=1e-12)
    if eb:
        # Independent FrontierCost refit, with the same observed setting order.
        own = sorted(ROWS[:2], key=lambda row: describe(row)[0])
        other = direct(source_prior(), own)
        for row, value in observed:
            other.add(own.index(row), value)
        other.refit()
        np.testing.assert_allclose(list(head.recipe["widths"].values()), other.widths, rtol=1e-8)


def test_lookahead_and_conditioned_share_observation_covariance_and_noise():
    head = build([(ROWS[0], v) for v in (.7, 1., 4.)], eb=True)
    rows = _Rows.build(head, ROWS)
    mu, var, kappa = rows.observation(ROWS[1])
    tau2 = noise_variance(head.recipe, ROWS[1])
    conditioned = _condition_head(head, ROWS[1], tau2)
    after = _Rows.build(conditioned, ROWS)
    np.testing.assert_allclose(after.mu, rows.mu, atol=1e-12)
    np.testing.assert_allclose(covariance(after), covariance(rows) - np.outer(kappa, kappa) / (var + tau2), atol=1e-10)
    assert conditioned.recipe == head.recipe


def test_composite_cost_observation_maps_each_piece_before_averaging():
    head = build()
    a, b = [row_terms(head.recipe, row) for row in ROWS[:2]]
    combo = [(n, p, v / 2) for row in (a, b) for n, p, v in row]
    rows = _Rows.build(head, ROWS)
    mu, var, cross = rows.observation(combo)
    C = covariance(rows)
    assert mu == pytest.approx(np.mean(rows.mu[:2]))
    assert var == pytest.approx(C[:2, :2].sum() / 4)
    np.testing.assert_allclose(cross, C[:, :2].mean(axis=1))


@pytest.mark.parametrize("eb", [False, True])
@pytest.mark.parametrize("usd", [.1, 10.])
def test_one_discounted_observation_has_zero_centered_dispersion(eb, usd, monkeypatch):
    from loopmath.frontier import cost

    evidence_means = []
    original = cost._gauss_evidence

    def evidence(m0, K, xbar, nc, sig2):
        evidence_means.append(xbar[nc > 0].copy())
        return original(m0, K, xbar, nc, sig2)

    monkeypatch.setattr(cost, "_gauss_evidence", evidence)
    head = build([(ROWS[0], usd)], weights=[.7], eb=eb)
    key = describe(ROWS[0])[0]
    assert head.recipe["run_variance"][key] == pytest.approx(.36, abs=1e-14)
    cm = direct(head.recipe, ROWS)
    y = math.log(usd)
    cm.nc[0], cm.sx[0], cm.sxx[0] = .7, .7 * y, .7 * y * y
    xbar, ss = cm._statistics()
    assert xbar[0] == pytest.approx(y, abs=1e-14)
    assert ss[0] == pytest.approx(0., abs=1e-14)
    cm.fit()
    row = _Rows.build(head, ROWS)
    np.testing.assert_allclose(row.mu, cm.mc, atol=1e-10)
    np.testing.assert_allclose(covariance(row), cm.Sc, atol=1e-10)
    # An independent one-observation Gaussian update pins the actual log value,
    # not its reliability times that value, in both the fit and width evidence.
    expected = cm.mc0 + cm.Sc0[:, 0] * (y - cm.mc0[0]) / (cm.Sc0[0, 0] + .36 / .7)
    np.testing.assert_allclose(row.mu, expected, atol=1e-10)
    if eb:
        assert evidence_means
        for mean in evidence_means:
            np.testing.assert_allclose(mean, [y], atol=1e-14)
    else:
        assert not evidence_means


@pytest.mark.parametrize("eb", [False, True])
def test_weighted_posterior_matches_centered_weighted_mean_likelihood(eb):
    observed = [(ROWS[0], v) for v in (.1, 2., 7.)] + [(ROWS[1], v) for v in (10., 12.)]
    reliability = [.7, .2, .7, .7, 1.]
    head = build(observed, weights=reliability, eb=eb)
    cm = direct(head.recipe, ROWS)
    counts, means, sig2 = np.zeros(3), np.zeros(3), np.full(3, .36)
    for i, row in enumerate(ROWS[:2]):
        cells = [(math.log(value), weight) for (r, value), weight in zip(observed, reliability) if r == row]
        values, weights = map(np.array, zip(*cells))
        counts[i] = weights.sum()
        means[i] = np.average(values, weights=weights)
        centered = np.sum(weights * (values - means[i]) ** 2)
        sig2[i] = (5 * .36 + centered) / (5 + max(counts[i] - 1, 0))
    O = np.flatnonzero(counts > 0)
    cross = cm.Sc0[:, O]
    C = cm.Sc0[np.ix_(O, O)] + np.diag(sig2[O] / counts[O])
    expected = cm.mc0 + cross @ np.linalg.solve(C, means[O] - cm.mc0[O])
    expected_covariance = cm.Sc0 - cross @ np.linalg.solve(C, cross.T)
    rows = _Rows.build(head, ROWS)
    np.testing.assert_allclose(rows.mu, expected, atol=1e-10)
    np.testing.assert_allclose(covariance(rows), expected_covariance, atol=1e-10)
    np.testing.assert_allclose(rows.noise, sig2, atol=1e-12)
    if eb:
        own = sorted(ROWS[:2], key=lambda row: describe(row)[0])
        direct_fit = direct(source_prior(), own)
        for (row, value), weight in zip(observed, reliability):
            i, y = own.index(row), math.log(value)
            direct_fit.nc[i] += weight
            direct_fit.sx[i] += weight * y
            direct_fit.sxx[i] += weight * y * y
        direct_fit.refit()
        np.testing.assert_allclose(list(head.recipe["widths"].values()), direct_fit.widths, rtol=1e-8)


def test_statistics_preserve_empty_and_integer_count_arithmetic():
    cm = direct(source_prior(), ROWS)
    before_mean, before_covariance = cm.mc.copy(), cm.Sc.copy()
    assert cm.refit() is None
    np.testing.assert_array_equal(cm.mc, before_mean)
    np.testing.assert_array_equal(cm.Sc, before_covariance)
    for i, values in [(1, [10.]), (2, [.1, 2., 7.])]:
        for value in values:
            cm.add(i, value)
    legacy_mean = cm.sx / np.maximum(cm.nc, 1)
    legacy_ss = np.maximum(cm.sxx - cm.nc * legacy_mean ** 2, 0)
    mean, ss = cm._statistics()
    np.testing.assert_array_equal(mean, legacy_mean)
    np.testing.assert_array_equal(ss, legacy_ss)
    cm.fit()
    np.testing.assert_array_equal(cm.sig2, (5 * .36 + legacy_ss) / (5 + np.maximum(cm.nc - 1, 0)))


def test_saved_cost_law_roundtrips_and_legacy_fits_keep_their_reader(tmp_path, monkeypatch):
    import simdata

    from loopmath.belief import design
    from loopmath.belief.fit import fit
    from loopmath.belief.state import load

    docs, _ = simdata.simulate(12, seed=123, source="live", n_tasks=4)
    path = fit(tmp_path / "current", docs=[("user", d) for d in docs], eb=False)
    state = load(path)
    assert state.heads["cost"].recipe["run_variance"]
    assert state.meta["model_options"]["belief.cost_recipe"] == "frontier-setting-v1"
    assert design.other_design(state.meta) is None
    from loopmath.belief.design import parse_run

    run = parse_run(docs[0])
    prediction = state.predict(run.task, run.config)
    shift = state.lookahead(run.task, run.config, run.config, [run.config]).posterior_shift["cost_usd"]
    assert prediction.cost.usd.mean > 0 and shift.mean > 0
    # On a learned setting, predict integrates the known Gaussian using finite
    # draws while lookahead uses its analytic moment; allow Monte Carlo error.
    assert shift.mean == pytest.approx(prediction.cost.usd.mean, rel=.1)
    legacy = load(fit(tmp_path / "legacy", docs=docs, no_prior=True, eb=False))
    assert legacy.heads["cost"].recipe is None
    assert design.other_design(legacy.meta) is None
    monkeypatch.delitem(design.MODEL_OPTIONS, "belief.cost_recipe")
    assert "belief.cost_recipe" in design.other_design(state.meta)
