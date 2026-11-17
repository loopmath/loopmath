"""The Laplace engine (`engine_laplace`): kernels, the block factorisation, the marginal likelihood, the
gradient, the fit and the fit file it writes.

Every check here compares with a value computed another way: a dense multivariate normal built from the
study rows by the model's definition, numerical integration, dense linear algebra, finite differences,
or the truth of a simulated study. All data is synthetic.
"""

from __future__ import annotations

import functools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from loopmath import cli
from loopmath.eval import contracts as C
from loopmath.eval import engine_laplace as EL
from loopmath.eval import engine_pymc
from loopmath.eval import fit as FT
from loopmath.eval import fitfile as F
from loopmath.eval import model as M
from loopmath.eval import simulate as S
from loopmath.eval import specs

slow = pytest.mark.slow
STAMP = "2026-01-15T09:00:00-08:00"


def catalog_design(name: str, inputs: int, seed: int = 0, **build):
    sim = S.generate(specs.get(name, inputs=inputs), seed)
    study = C.Study(Path("."), sim.runs, sim.scores, sim.pairs, sim.items)
    return M.build(study, sim.rubric, sim.judges, allow_simulated=True, **build), sim


def set_theta(prob: EL.Problem, values: dict) -> np.ndarray:
    """theta with named hyperparameters set on their natural scale (an sd, a share, a correlation)."""
    P = prob.params
    theta = P.init.copy()
    for label, value in values.items():
        k = P.labels.index(label)
        code = P.code[k]
        theta[k] = {EL.HN: np.log, EL.GAMMA: np.log, EL.BETA12: special.logit, EL.CPC: np.arctanh,
                    EL.REL: lambda x: x}[code](value)
    return theta


# ---------------------------------------------------------------------------------------------------
# Kernels

def test_log_interval_against_numerical_integration_in_the_middle_and_in_both_tails():
    cases = [(-0.3, 0.4), (-1e-7, 1e-7), (1.0, 1.5), (-2.0, -1.9), (6.0, 6.5), (25.0, 26.0), (-38.0, -37.5),
             (30.0, 1e6), (-1e6, -30.0), (-1e6, 1e6), (0.0, 1e6)]
    for a, b in cases:
        got = float(EL.log_interval(np.array([a]), np.array([b]))[0])
        # the integral of the normal density from a to b, scaled by the density at the end nearer 0
        near = a if a >= 0 else (b if b <= 0 else 0.0)
        top = min(b, near + 60.0) if a >= 0 else b
        bottom = max(a, near - 60.0) if b <= 0 else a
        if a < 0 < b:
            bottom, top = max(a, -60.0), min(b, 60.0)
        value, _ = integrate.quad(lambda x: np.exp(-0.5 * (x * x - near * near)), bottom, top, epsabs=0,
                                  epsrel=1e-13, limit=200)
        want = np.log(value) - 0.5 * near * near - 0.5 * np.log(2 * np.pi)
        assert abs(got - want) < 1e-9 * max(1.0, abs(want)), (a, b, got, want)


def test_interval_kernel_derivatives_against_finite_differences():
    rng = np.random.default_rng(3)
    lo = np.array([-1e6, -0.4, 0.0, 1.2, -2.0, -1e6, 3.0])
    hi = np.array([0.0, 0.7, 1e6, 1.5, -1.0, 1e6, 1e6])
    eta = rng.normal(size=lo.size) * 1.5
    eta[-1] = -4.0   # far into the tail of a pass: the Mills ratio regime
    k = EL.interval_kernel(eta, lo, hi)
    h = 1e-5

    def at(e=eta, l=lo, u=hi):
        return EL.interval_kernel(e, l, u)

    for name, lower in (("logp", "d1"), ("d1", "d2"), ("d2", "d3")):
        fd = (at(e=eta + h)[name] - at(e=eta - h)[name]) / (2 * h)
        assert np.allclose(k[lower], fd, rtol=1e-5, atol=1e-6), (lower, k[lower], fd)
    finite_lo, finite_hi = np.abs(lo) < 1e5, np.abs(hi) < 1e5
    for name, prefix in (("logp", "d"), ("d1", "d1_d"), ("d2", "d2_d")):
        fd = (at(l=lo + h)[name] - at(l=lo - h)[name]) / (2 * h)
        assert np.allclose(k[prefix + "lo"][finite_lo], fd[finite_lo], rtol=1e-5, atol=1e-6), (prefix, "lo")
        fd = (at(u=hi + h)[name] - at(u=hi - h)[name]) / (2 * h)
        assert np.allclose(k[prefix + "hi"][finite_hi], fd[finite_hi], rtol=1e-5, atol=1e-6), (prefix, "hi")
    # an open bound contributes nothing, and no value is NaN
    assert np.all(k["dlo"][~finite_lo] == 0.0) and np.all(k["dhi"][~finite_hi] == 0.0)
    assert all(np.isfinite(value).all() for value in k.values())


def test_interval_kernel_far_in_the_tails_follows_the_mills_ratio_series():
    """A pass read when the latent sits x noise sds below 0 (and the mirror case). For large x the inverse
    Mills ratio has the series lam = x + 1/x - 2/x^3 + 10/x^5 - 74/x^7 (from Phi(-x)/phi(x) = 1/x - 1/x^3
    + 3/x^5 - 15/x^7), so for the fail side d1 = -lam, d2 = -lam (lam - x) = -1 + 1/x^2 - 6/x^4 + 50/x^6
    and d3 = -2/x^3 + 24/x^5 - 300/x^7. Before the tail formulas, x near 1e7 gave d2 near -7e11 and a
    Hessian that was not positive definite."""
    x = np.array([30.0, 1e3, 1e5, 1e7])
    k = EL.interval_kernel(x, np.full(4, -1e13), np.zeros(4))    # a fail read far above the cut
    lam = x + 1 / x - 2 / x ** 3 + 10 / x ** 5 - 74 / x ** 7
    d3 = -2 / x ** 3 + 24 / x ** 5 - 300 / x ** 7
    assert np.allclose(k["d1"], -lam, rtol=1e-11)
    assert np.allclose(k["d2"], -1 + 1 / x ** 2 - 6 / x ** 4 + 50 / x ** 6, rtol=1e-9, atol=0)
    assert np.allclose(k["d3"], d3, rtol=1e-5, atol=0)
    m = EL.interval_kernel(-x, np.zeros(4), np.full(4, 1e13))    # the mirror: a pass read far below the cut
    assert np.allclose(m["d1"], lam, rtol=1e-11) and np.allclose(m["d3"], -d3, rtol=1e-5, atol=0)
    # log P against the series of log Phi(-x)
    want = -0.5 * x ** 2 - np.log(x) - 0.5 * np.log(2 * np.pi) + np.log1p(-1 / x ** 2 + 3 / x ** 4 - 15 / x ** 6)
    assert np.allclose(k["logp"], want, rtol=1e-12, atol=0)
    # every interval, however far out: finite, and d2 within [-1, 0]
    rng = np.random.default_rng(5)
    eta = rng.choice([-1, 1], 400) * 10 ** rng.uniform(-3, 7, 400)
    lo = rng.choice([-1e13, -1.0, 0.0, 0.5], 400)
    hi = lo + rng.choice([1e-6, 0.3, 2.0, 1e13], 400)
    keep = hi > lo   # lo + 1e-6 rounds to lo at -1e13: an empty interval, log P = -inf by definition
    eta, lo, hi = eta[keep], lo[keep], hi[keep]
    k = EL.interval_kernel(eta, lo, hi)
    assert all(np.isfinite(value).all() for value in k.values())
    assert np.all((k["d2"] >= -1.0) & (k["d2"] <= 0.0))


def test_gaussian_kernel_against_scipy_and_finite_differences():
    eta, y, var = np.array([0.3, -1.0]), np.array([0.5, 0.2]), np.array([0.04, 2.0])
    k = EL.gaussian_kernel(eta, y, var)
    assert np.allclose(k["logp"], stats.norm.logpdf(y, eta, np.sqrt(var)))
    h = 1e-6
    fd = (EL.gaussian_kernel(eta, y, var + h)["logp"] - EL.gaussian_kernel(eta, y, var - h)["logp"]) / (2 * h)
    assert np.allclose(k["dvar"], fd, rtol=1e-6)


# ---------------------------------------------------------------------------------------------------
# The block factorisation

def test_block_factor_matches_dense_linear_algebra():
    rng = np.random.default_rng(7)
    G, I, B = 5, 4, 3
    n = G + I * B
    X = rng.normal(size=(40, n))
    # an arrow: every row touches the global unknowns and the unknowns of one input
    for row in range(40):
        i = row % I
        keep = np.zeros(n, dtype=bool)
        keep[:G] = True
        keep[G + i * B:G + (i + 1) * B] = True
        X[row, ~keep] = 0.0
    W = rng.uniform(0.1, 3.0, size=40)
    H = np.eye(n) + X.T @ (W[:, None] * X)
    gg, bg = H[:G, :G], np.stack([H[G + i * B:G + (i + 1) * B, :G] for i in range(I)])
    bb = np.stack([H[G + i * B:G + (i + 1) * B, G + i * B:G + (i + 1) * B] for i in range(I)])
    fac = EL.Factor(gg, bg, bb)
    assert np.isclose(fac.logdet, np.linalg.slogdet(H)[1], rtol=1e-12)
    r = rng.normal(size=n)
    xg, xb = fac.solve(r[:G], r[G:].reshape(I, B))
    assert np.allclose(np.concatenate([xg, xb.ravel()]), np.linalg.solve(H, r))
    inv = np.linalg.inv(H)
    sgg, sbg, sbb = fac.selected()
    assert np.allclose(sgg, inv[:G, :G])
    for i in range(I):
        part = slice(G + i * B, G + (i + 1) * B)
        assert np.allclose(sbg[i], inv[part, :G]) and np.allclose(sbb[i], inv[part, part])
    dg, db = fac.draw(np.random.default_rng(0), 200_000)
    draws = np.concatenate([dg, db.reshape(len(dg), -1)], axis=1)
    assert np.abs(np.cov(draws.T) - inv).max() < 0.02 * np.abs(inv).max()


# ---------------------------------------------------------------------------------------------------
# The marginal likelihood against independent values

def _linear_study(seed: int = 5):
    """2 linear components, 2 arms on 3 inputs, a person anchor who reads everything once and two model
    judges who read everything twice. Values are drawn at random: only the likelihood is checked."""
    rng = np.random.default_rng(seed)
    comps = [{"id": c, "title": c, "scale": {"kind": "linear", "range": [0, 10]}} for c in ("depth", "clarity")]
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "t", "version": "1", "components": comps})
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"], "min_anchor_targets": 3,
                                "judges": [{"id": "reviewer", "kind": "person"},
                                           {"id": "strong", "kind": "model", "model": {"id": "m1", "family": "alpha"}},
                                           {"id": "cheap", "kind": "model", "model": {"id": "m2", "family": "beta"}}]})
    runs, scores = [], []
    for i in range(3):
        for arm in ("base", "new"):
            run = f"{arm}-{i}"
            runs.append(C.validate_run({"schema": C.SCHEMA_RUNS, "run": run, "input": f"in-{i}", "arm": arm,
                                        "steps": [{"step": "final", "model": {"id": "x", "family": "alpha"}}],
                                        "simulated": True}))
            for comp in ("depth", "clarity"):
                for judge, calls in (("reviewer", 1), ("strong", 2), ("cheap", 2)):
                    for _ in range(calls):
                        scores.append(C.validate_score({
                            "schema": C.SCHEMA_SCORES, "id": f"s{len(scores)}", "run": run, "component": comp,
                            "judge": judge, "call": f"k{len(scores)}", "rubric": "t@1", "observed_at": STAMP,
                            "value": float(np.round(rng.uniform(2, 8), 2))}))
    study = C.Study(Path("."), runs, scores, [], [])
    return study, rubric, judges


def _dense_gaussian_loglik(study, nu: dict) -> float:
    """log p(readings) of the model, from the rows: a multivariate normal whose covariance is summed from
    every term of the model's definition (engine_pymc's module text). Independent of the engine's layout."""
    judges = ["reviewer", "strong", "cheap"]
    dims = ["depth", "clarity"]
    arms = ["base", "new"]
    run_of = {row["run"]: row for row in study.runs}
    rows = study.scores
    n = len(rows)
    y = np.array([row["value"] / 10.0 for row in rows])
    s = 0.25
    w = np.array([1.0, 0.0, 0.0])            # the anchor carries the bias of every dimension
    cov_in = np.outer(nu["sd_in"], nu["sd_in"]) * nu["corr_in"]
    cov_run = np.outer(nu["sd_run"], nu["sd_run"]) * nu["corr_run"]
    V = np.zeros((n, n))
    for a, ra in enumerate(rows):
        run_a = run_of[ra["run"]]
        ja, da = judges.index(ra["judge"]), dims.index(ra["component"])
        for b, rb in enumerate(rows):
            run_b = run_of[rb["run"]]
            jb, db = judges.index(rb["judge"]), dims.index(rb["component"])
            v = 0.0
            if run_a["arm"] == run_b["arm"] and da == db:
                v += (2 * s) ** 2                                   # the arm mean, N(0.5, (2 s)^2)
            if run_a["input"] == run_b["input"]:
                v += cov_in[da, db]
            if ra["run"] == rb["run"]:
                v += cov_run[da, db]
                if ja == jb:   # the judge's view of the output: shared by dimensions, and the own part
                    v += nu["tau"][ja, da] * nu["tau"][jb, db] * nu["h"][ja]
                    if da == db:
                        v += nu["tau"][ja, da] ** 2 * (1 - nu["h"][ja])
            if da == db:   # bias = s (z_j - sum_k w_k z_k)
                v += s * s * ((ja == jb) - w[ja] - w[jb] + (w * w).sum())
            if a == b:
                v += nu["sigma"][ja, da] ** 2
            V[a, b] = v
    return float(stats.multivariate_normal(np.full(n, 0.5), V).logpdf(y))


def test_gaussian_marginal_likelihood_equals_the_dense_multivariate_normal():
    study, rubric, judges = _linear_study()
    design = M.build(study, rubric, judges, allow_simulated=True, terms=["family=off", "judge_input=off"])
    assert np.allclose(design.data["bias_w"], [[1, 1], [0, 0], [0, 0]])
    prob = EL.Problem(design.data)
    nu = {"sd_in": np.array([0.07, 0.05]), "sd_run": np.array([0.04, 0.06]), "corr": 0.35, "corr_r": -0.2,
          "tau": np.array([[0.03, 0.02], [0.05, 0.04], [0.08, 0.09]]),
          "sigma": np.array([[0.04, 0.03], [0.05, 0.04], [0.07, 0.06]]), "h": np.array([0.2, 0.5, 0.7])}
    values = {"sd_input[0]": 0.07, "sd_input[1]": 0.05, "sd_run[0]": 0.04, "sd_run[1]": 0.06,
              "cpc_input[0]": 0.35, "cpc_run[0]": -0.2}
    for j in range(3):
        values[f"halo_target[{j}]"] = nu["h"][j]
        for d in range(2):
            values[f"tau[{j}, {d}]"] = nu["tau"][j, d]
            values[f"sigma[{j}, {d}]"] = nu["sigma"][j, d]
    theta = set_theta(prob, values)
    got = prob.evaluate(theta, grad=False)["laplace"]
    corr = lambda r: np.array([[1.0, r], [r, 1.0]])
    want = _dense_gaussian_loglik(study, {**nu, "corr_in": corr(0.35), "corr_run": corr(-0.2)})
    assert abs(got - want) < 1e-8 * abs(want), (got, want)


def _probit_design(passes: list[int], tau: float):
    comps = [{"id": "ok", "title": "ok", "scale": {"kind": "binary"}}]
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "t", "version": "1", "components": comps})
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "judges": [
        {"id": "a", "kind": "model", "model": {"id": "m1", "family": "alpha"}}]})
    runs = [C.validate_run({"schema": C.SCHEMA_RUNS, "run": "r", "input": "i", "arm": "x",
                            "steps": [{"step": "final", "model": {"id": "x", "family": "alpha"}}], "simulated": True})]
    scores = [C.validate_score({"schema": C.SCHEMA_SCORES, "id": f"s{k}", "run": "r", "component": "ok", "judge": "a",
                                "call": f"k{k}", "rubric": "t@1", "observed_at": STAMP, "value": int(v)})
              for k, v in enumerate(passes)]
    design = M.build(C.Study(Path("."), runs, scores, [], []), rubric, judges, allow_simulated=True)
    prob = EL.Problem(design.data)
    return prob, set_theta(prob, {"sd_input[0]": 0.6, "sd_run[0]": 0.5, "tau[0, 0]": tau})


@pytest.mark.parametrize("passes,tau,gap", [([1, 0], 0.8, 0.02), ([1, 1, 1], 0.5, 0.10), ([1, 1], 2.0, 0.13),
                                            ([0], 1.0, 0.03)])
def test_probit_laplace_is_the_one_dimensional_laplace_and_near_the_exact_integral(passes, tau, gap):
    """One run read by one judge (probit units, s = 1; one judge, so its relative call noise is 1). With
    repeated calls the readings depend on x = mu + v + u + tau t alone, a normal of variance
    4 + 0.6^2 + 0.5^2 + tau^2, and given x the calls are independent with unit noise. With one call the
    judge's view of the output joins the call noise: the reading is a probit in q = mu + v + u (variance
    4 + 0.6^2 + 0.5^2) with noise sd sqrt(1 + tau^2). The engine's value must equal the Laplace
    approximation in that one latent, computed here by plain numerical means. Its distance to the exact
    integral by quadrature is the Laplace error itself (measured: 0.012 for a mixed cell, 0.023 for one
    call, 0.078 and 0.107 for cells of 3 and 2 passes, where the posterior is skewed). `gap` bounds that
    measured error. The EP inner step closes it: here every reading of the run shares one predictor, so
    they form one site, and EP with one site is the exact integral (to the cell rule's error)."""
    prob, theta = _probit_design(passes, tau)
    got = prob.evaluate(theta, grad=False, inner="laplace")["laplace"]
    ep = prob.evaluate(theta, grad=False, inner="ep")
    k, n = sum(passes), len(passes)
    var = 4.0 + 0.36 + 0.25 + (tau ** 2 if n > 1 else 0.0)
    noise = 1.0 if n > 1 else np.sqrt(1.0 + tau ** 2)

    def log_f(x):   # log likelihood plus log prior of the latent
        return (k * special.log_ndtr(x / noise) + (n - k) * special.log_ndtr(-x / noise)
                + stats.norm.logpdf(x, 0, np.sqrt(var)))

    mode = optimize.minimize_scalar(lambda x: -log_f(x), bounds=(-20, 20), method="bounded",
                                    options={"xatol": 1e-12}).x
    h = 1e-4
    curv = -(log_f(mode + h) - 2 * log_f(mode) + log_f(mode - h)) / h ** 2
    laplace_1d = log_f(mode) + 0.5 * np.log(2 * np.pi / curv)
    assert abs(got - laplace_1d) < 1e-5, (got, laplace_1d)
    exact = np.log(integrate.quad(lambda x: np.exp(log_f(x)), -40, 40, epsabs=0, epsrel=1e-12, limit=200)[0])
    assert abs(got - exact) < gap, (got, exact)
    assert ep["mode"]["ok"] and prob.n_groups == 1
    assert abs(ep["laplace"] - exact) < 1e-8, (ep["laplace"], exact)


# ---------------------------------------------------------------------------------------------------
# The cell rule and the EP inner step

def test_cell_integral_matches_lane_s2s_rule_on_levels_cells():
    """`cell_integral` against an independent rule: lane S2's `laplace_kernels.cell_integral` (adaptive
    Gauss-Hermite on smooth cells, Gauss-Kronrod panels on sharp ones, each with its own error estimate),
    on cells of two levels readings, sharp to smooth. S2 works in the reading's units with a noise sd;
    the engine's rule has unit noise, so the cavity and the cuts are divided by the sd. With 6 panels per
    half gap the rules agree to 1e-8; the engine's default (EL.PANELS = 4, chosen for speed) to 3e-7."""
    K = pytest.importorskip("loopmath.eval.laplace_kernels")
    rng = np.random.default_rng(3)
    for sd, tau in ((0.01, 0.1), (0.3, 0.1), (0.003, 0.03), (1.0, 2.0)):
        n = 300
        m = rng.uniform(0.1, 0.9, n)
        cuts = np.array([-np.inf, 0.25, 0.75, np.inf])
        lev = rng.integers(0, 3, (n, 2))
        lev[:, 1] = np.where(rng.random(n) < 0.7, lev[:, 0], lev[:, 1])
        lo, hi = cuts[lev], cuts[lev + 1]
        ref = K.cell_integral("levels", np.repeat(m, 2), {"lo": lo.ravel(), "hi": hi.ravel()}, {"sd": sd},
                              np.repeat(np.arange(n), 2), np.full(n, tau))
        args = (m / sd, np.full(n, (tau / sd) ** 2), np.where(np.isfinite(lo), lo / sd, -EL.OPEN),
                np.where(np.isfinite(hi), hi / sd, EL.OPEN), np.ones((n, 2)))
        got = EL.cell_integral(*args, panels=6)
        assert np.max(np.abs(got["logZ"] - np.asarray(ref.logl))) < 1e-8, (sd, tau)
        assert np.max(got["tail"]) < 1e-12
        got = EL.cell_integral(*args)
        assert np.max(np.abs(got["logZ"] - np.asarray(ref.logl))) < 3e-7, (sd, tau)


def test_cell_integral_moments_and_bound_derivatives_against_quad():
    """Weighted cells (passed and failed items of one call, repeated binary calls, a levels reading)
    against scipy quad: log Z, the mean and the variance of the predictor; the bound derivatives against
    central differences of log Z."""
    rng = np.random.default_rng(8)
    n = 40
    m = rng.normal(0, 3, n)
    s2 = 10 ** rng.uniform(-1, 3, n)
    lo = np.stack([np.zeros(n), np.full(n, -EL.OPEN), rng.uniform(-3, 1, n)], axis=1)
    hi = np.stack([np.full(n, EL.OPEN), np.zeros(n), lo[:, 2] + rng.uniform(0.2, 4, n)], axis=1)
    w = np.stack([rng.integers(1, 6, n), rng.integers(0, 4, n), rng.integers(0, 2, n)], axis=1).astype(float)
    got = EL.cell_integral(m, s2, lo, hi, w, bounds=True)
    for g in range(n):
        def f(u, power=0):
            val = stats.norm.logpdf(u, m[g], np.sqrt(s2[g]))
            for r in range(3):
                if w[g, r]:   # the reference underflows to log 0 far out, where the integrand is 0
                    with np.errstate(divide="ignore"):
                        val += w[g, r] * np.log(special.ndtr(hi[g, r] - u) - special.ndtr(lo[g, r] - u))
            return u ** power * np.exp(val)
        pts = sorted({0.0, lo[g, 2], hi[g, 2]})
        Z, M1, M2 = (sum(integrate.quad(f, a, b, args=(k,), limit=400, epsabs=0, epsrel=1e-12)[0]
                         for a, b in zip([-np.inf] + pts, pts + [np.inf])) for k in (0, 1, 2))
        assert abs(got["logZ"][g] - np.log(Z)) < 1e-8
        assert abs(got["mean"][g] - M1 / Z) < 1e-7 * (1 + np.sqrt(s2[g]))
        assert abs(got["var"][g] - (M2 / Z - (M1 / Z) ** 2)) < 1e-6 * (1 + s2[g])
    eps = 1e-5
    for r, name, edge in ((0, "dlo", lo), (1, "dhi", hi), (2, "dlo", lo), (2, "dhi", hi)):
        up, down = edge.copy(), edge.copy()
        up[:, r] += eps
        down[:, r] -= eps
        a_up = EL.cell_integral(m, s2, up if edge is lo else lo, up if edge is hi else hi, w)["logZ"]
        a_dn = EL.cell_integral(m, s2, down if edge is lo else lo, down if edge is hi else hi, w)["logZ"]
        fd = (a_up - a_dn) / (2 * eps)
        live = w[:, r] > 0
        assert np.allclose(got[name][live, r], fd[live], rtol=1e-5, atol=1e-7), (r, name)


def test_cell_integral_finds_the_mode_however_far_from_its_guess():
    """A guess from other arguments can be hundreds of curvature scales off (the boundary scan moves a
    cavity mean from about 60 to 571). The first two cells are the S3 pilot's (panel-vs-per-component,
    seed 1), where the old search stopped short of the mode and exp overflowed. Every cell must match
    scipy quad, be ok, and agree with the call without a guess."""
    m = np.array([571.14, -751.812, 600.0, 2000.0, -40.0])
    s2 = np.array([1.04995, 0.906432, 1.0, 0.01, 4.0])
    far, wide = 685348.28281101, EL.OPEN
    lo = np.array([[0.0, 0.0, -wide], [-far, -far, -wide], [0.0, 598.7, -wide], [0.0, 1999.2, -wide],
                   [-wide, -41.5, -wide]])
    hi = np.array([[far, far, wide], [0.0, 0.0, wide], [wide, 600.4, 601.0], [wide, 1999.9, wide],
                   [0.0, -40.2, -38.0]])
    w = np.array([[1, 1, 0], [1, 1, 0], [2, 1, 1], [1, 1, 0], [3, 1, 1]], dtype=float)
    g_mode = np.array([61.0646, -136.9917, 60.0, 30.0, 500.0])
    guess = {"mode": g_mode, "A": g_mode - 10.0, "B": g_mode + 10.0}
    with np.errstate(over="raise", invalid="raise"):
        got = EL.cell_integral(m, s2, lo, hi, w, guess=guess)
    alone = EL.cell_integral(m, s2, lo, hi, w)
    assert got["ok"].all() and alone["ok"].all()
    for name in ("logZ", "mean", "var", "mode"):
        assert np.allclose(got[name], alone[name], rtol=0, atol=1e-9 * (1 + np.abs(alone[name]))), name
    for g in range(len(m)):
        sd = np.sqrt(s2[g])

        def f(u, power=0):
            val = stats.norm.logpdf(u, m[g], sd)
            for r in range(lo.shape[1]):
                if w[g, r]:
                    with np.errstate(divide="ignore"):
                        val += w[g, r] * np.log(special.ndtr(hi[g, r] - u) - special.ndtr(lo[g, r] - u))
            return (u - m[g]) ** power * np.exp(val)

        cuts = [x for x in np.concatenate([lo[g][w[g] > 0], hi[g][w[g] > 0]]) if abs(x - m[g]) < 40 * sd]
        pts = sorted(set(cuts) | {m[g] + k * sd for k in (-40, -10, 0, 10, 40)})
        Z, M1, M2 = (sum(integrate.quad(f, a, b, args=(k,), limit=400, epsabs=0, epsrel=1e-12)[0]
                         for a, b in zip(pts[:-1], pts[1:])) for k in (0, 1, 2))
        assert abs(got["logZ"][g] - np.log(Z)) < 1e-8, g
        assert abs(got["mean"][g] - (m[g] + M1 / Z)) < 1e-7 * (1 + sd), g
        assert abs(got["var"][g] - (M2 / Z - (M1 / Z) ** 2)) < 1e-6 * (1 + s2[g]), g


def test_cell_integral_from_a_stale_guess_matches_the_gaussian_limit():
    """Cavity N(400, 1), two passes on [0, 1e6], an earlier mode of 0 (the review's case): the passes are
    certain near 400, so log Z = 0, the mean 400 and the variance 1 to rounding. A stale guess only starts
    the search: the old search stopped at 320 with log Z -2775.6."""
    a = (np.array([400.0]), np.array([1.0]), np.array([[0.0, 0.0]]), np.array([[1e6, 1e6]]), np.ones((1, 2)))
    with np.errstate(over="raise", invalid="raise"):
        warm = EL.cell_integral(*a, guess={"mode": np.array([0.0]), "A": np.array([-9.0]), "B": np.array([9.0])})
    assert warm["ok"][0] and warm["mode"][0] == pytest.approx(400.0, abs=1e-6)
    assert abs(warm["logZ"][0]) < 1e-9 and abs(warm["mean"][0] - 400.0) < 1e-7 and abs(warm["var"][0] - 1.0) < 1e-7


def test_a_smooth_cell_of_many_passes_is_caught_by_the_doubled_rule():
    """500 passes and 1 failure on one cut, cavity N(0, 100) (the review's case): the curvature scale is
    0.31, so the cell counts as smooth, yet the product is skewed and GH_ORDER nodes are off by about
    1.2e-4. The smooth rule is a heuristic; the cell rule check, with twice the nodes, must see it."""
    def lh(x):
        return 500 * special.log_ndtr(x) + special.log_ndtr(-x) - x * x / 200

    peak = lh(3.0)
    exact = np.log(integrate.quad(lambda x: np.exp(lh(x) - peak), -20, 40, epsabs=1e-13, epsrel=1e-13,
                                  limit=400)[0]) + peak - 0.5 * np.log(200 * np.pi)
    a = (np.array([0.0]), np.array([100.0]), np.array([[0.0, -EL.OPEN]]), np.array([[EL.OPEN, 0.0]]),
         np.array([[500.0, 1.0]]))
    base, fine = EL.cell_integral(*a), EL.cell_integral(*a, panels=2 * EL.PANELS)
    assert base["smooth"][0] and base["ok"][0]
    assert abs(fine["logZ"][0] - exact) < 1e-5
    assert abs(base["logZ"][0] - fine["logZ"][0]) > EL.CELL_RULE_MAX

def _oracle():
    try:
        import laplace_oracle
    except ImportError:
        pytest.skip("tests/eval/laplace_oracle.py (lane S4's exact oracle) is not on this branch")
    return laplace_oracle


def _engine_units(prob: EL.Problem, th: dict) -> dict:
    """An oracle theta in the engine's units. On a dimension whose call noise is relative (binary, pairs
    only) the engine fixes the judges' weighted geometric mean noise at 1, so every latent-scale value is
    divided by that mean. The likelihood of the readings does not change; both sides then evaluate the
    same model at the same theta."""
    P = prob.params
    th = {k: (np.array(v, dtype=float) if np.ndim(v) else v) for k, v in th.items()}
    for key, wkey, flag in (("sigma", "sigma_w", "fixed"), ("sigma_pair", "pair_w", "only")):
        if not np.asarray(getattr(P, flag)).any():
            continue
        wts = np.asarray(getattr(P, wkey), dtype=float)
        g = float(np.exp((wts * np.log(np.asarray(th[key], dtype=float))).sum(axis=0))[0])
        for name in ("sd_input", "sd_run", "tau", "sigma", "sigma_pair", "tie_width", "position", "mu", "bias"):
            if name in th:
                th[name] = np.asarray(th[name], dtype=float) / g
    return th


def _theta_vector(prob: EL.Problem, th: dict) -> np.ndarray:
    P = prob.params
    theta = P.init.copy()
    for k, label in enumerate(P.labels):
        name, idx = label.split("[")[0], tuple(int(x) for x in label.split("[")[1].rstrip("]").split(","))
        value = float(np.asarray(th[name], dtype=float)[idx])
        theta[k] = special.logit(value) if P.code[k] == EL.BETA12 else np.log(value)
    return theta


_SHARP = {"binary": dict(tau=0.4, sigma=1.0), "levels": dict(tau=0.06, sigma=0.15), "items": dict(tau=0.4, sigma=0.5),
          "pairs": dict(tau=0.4, sigma_pair=1.0)}


def _against_oracle(kind: str, opts: dict, f: float, seed: int):
    O = _oracle()
    b = O.tiny(kind, seed=seed, theta={k: v * f for k, v in _SHARP[kind].items()}, **opts)
    prob = EL.Problem(b.data)
    th = _engine_units(prob, O.full_theta(b.data, b.theta))
    exact = O.exact(b.data, th, random=("mu", "bias", "position"))
    const = O.pymc_constant(b.data) if kind == "items" else 0.0   # log C(n, k): the engine drops it
    theta = _theta_vector(prob, th)
    lap = prob.evaluate(theta, grad=False, inner="laplace")["laplace"] + const
    ep = prob.evaluate(theta, grad=False, inner="ep")
    assert ep["mode"]["ok"]
    return prob, exact, lap, ep["laplace"] + const


@pytest.mark.parametrize("kind,opts", [("binary", dict(runs=1, calls=1)), ("binary", dict(runs=1, calls=2)),
                                       ("binary", dict(runs=1, calls=3)), ("levels", dict(runs=1, calls=1)),
                                       ("levels", dict(runs=1, calls=3)), ("items", dict(runs=1, calls=1)),
                                       ("pairs", dict(runs=2, pair_calls=1))])
@pytest.mark.parametrize("f", [1.0, 0.1])
def test_ep_is_exact_where_one_site_holds_every_reading(kind, opts, f):
    """Lane S4's exact marginal likelihood (`O.exact`, random = mu, bias and position) on tiny designs,
    at default sharpness (f = 1) and with tau and the call noise times 0.1 (sharp judges, where S4
    measured Laplace 23% low on binary). Here every reading shares one predictor, so EP has one site and
    must equal the exact value. The engine's Laplace value must equal the oracle's Laplace value: that
    checks both sides read the same model."""
    for seed in (0, 1):
        prob, exact, lap, ep = _against_oracle(kind, opts, f, seed)
        assert prob.n_groups == 1
        assert abs(lap - exact.laplace) < 1e-6, (lap, exact.laplace)
        assert abs(ep - exact.loglik) < 1e-6, (ep, exact.loglik, lap)


@pytest.mark.parametrize("kind,opts,bound", [("items", dict(runs=1, calls=2), 0.01),
                                             ("pairs", dict(runs=2, pair_calls=3), 0.12),
                                             ("binary", dict(runs=2, judges=2, calls=1), 0.03)])
def test_ep_beats_laplace_where_several_sites_share_unknowns(kind, opts, bound):
    """Several sites on shared unknowns (two calls of items with their own call noise; pairs shown in
    turn a, b and unknown, whose position term differs; two runs read by two judges): EP is an
    approximation. Measured |log exact - EP| at most 0.0024, 0.097 and 0.0063, against Laplace 0.016,
    0.40 and 0.087. `bound` holds the EP error; EP must also be at least three times closer than Laplace."""
    checked = 0
    for f in ((1.0,) if opts.get("judges") else (1.0, 0.1)):
        for seed in (0, 1):
            try:
                _, exact, lap, ep = _against_oracle(kind, opts, f, seed)
            except Exception as exc:   # the oracle refuses blocks it cannot integrate to its tolerance
                if type(exc).__name__.startswith("Oracle"):
                    continue
                raise
            checked += 1
            assert abs(ep - exact.loglik) < bound, (f, seed, ep, exact.loglik)
            assert abs(ep - exact.loglik) * 3 < abs(lap - exact.loglik), (f, seed, ep, lap, exact.loglik)
    assert checked >= 1


def _reversed(rows: list, flip) -> list:
    """Pair records with the ones `flip` picks written the other way round: the same observations."""
    out = []
    for n, row in enumerate(rows):
        row = dict(row)
        if flip(n, row):
            other = {"a": "b", "b": "a"}
            row["a"], row["b"] = row["b"], row["a"]
            row["verdict"] = other.get(row["verdict"], row["verdict"])
            row["shown_first"] = other.get(row["shown_first"], row["shown_first"])
        out.append(row)
    return out


@pytest.mark.parametrize("subset", ["all", "wins", "losses", "ties", "shown_b", "random", "repeated"])
def test_ep_is_invariant_to_the_orientation_of_pair_records(subset):
    """Recording a pair as (b, a) with the verdict and the order shown seen from b is the same observation.
    Reversing any subset of the records (wins, losses, ties, either order shown; `repeated` reverses one
    of each set of repeated identical comparisons) must leave the EP value, its gradient and the
    conditional moments unchanged: the arm means, biases and position effect at the conditional mean,
    and each pair's predictor (mean up to its sign, variance). The reference is the build from the
    records as simulated; the reversal happens in the records, before `model.build`."""
    sim = S.generate(specs.get("pairs", inputs=4), 0)
    rng = np.random.default_rng(5)
    pick = rng.random(len(sim.pairs)) < 0.5
    seen: set = set()

    def first_repeat(n, row):
        key = (row["judge"], row["a"], row["b"], row["shown_first"], row["verdict"])
        if key in seen:
            return False
        seen.add(key)
        return sum(1 for r in sim.pairs if (r["judge"], r["a"], r["b"], r["shown_first"], r["verdict"]) == key) > 1

    flip = {"all": lambda n, r: True, "wins": lambda n, r: r["verdict"] == "a",
            "losses": lambda n, r: r["verdict"] == "b", "ties": lambda n, r: r["verdict"] == "tie",
            "shown_b": lambda n, r: r["shown_first"] == "b", "random": lambda n, r: bool(pick[n]),
            "repeated": first_repeat}[subset]
    rows = _reversed(sim.pairs, flip)
    assert rows != sim.pairs
    results = []
    for records in (sim.pairs, rows):
        study = C.Study(Path("."), sim.runs, sim.scores, records, sim.items)
        prob = EL.Problem(M.build(study, sim.rubric, sim.judges, allow_simulated=True).data)
        theta = np.clip(prob.params.init + 0.3 * np.random.default_rng(2).standard_normal(prob.params.size),
                        prob.params.lo, prob.params.hi)
        res = prob.evaluate(theta)
        assert res["mode"]["ok"] and prob.inner == "ep"
        c = prob.coefficients(prob.params.unpack(theta))
        zg, zb = res["mode"]["zg"], res["mode"]["zb"]
        mean = EL.outputs(prob, res["nu"], zg[None], zb[None])
        pair_rows = prob.r_mode == 3
        eta = prob.eta(c, zg, zb)[pair_rows]
        var = prob.marginals(c, res["mode"]["factor"])[0][pair_rows]
        results.append((res["value"], res["grad"], mean, eta, var))
    (v0, g0, m0, e0, s0), (v1, g1, m1, e1, s1) = results
    assert v1 == pytest.approx(v0, abs=1e-8)
    assert np.abs(g1 - g0).max() < 1e-6
    for name in ("mu", "bias", "position"):
        assert np.allclose(m1[name], m0[name], atol=1e-7, equal_nan=True), name
    assert np.abs(np.abs(e1) - np.abs(e0)).max() < 1e-7
    assert np.abs(s1 - s0).max() < 1e-7


# ---------------------------------------------------------------------------------------------------
# The gradient of the outer objective

@pytest.mark.parametrize("name,inner", [("linear", None), ("levels-3", "ep"), ("binary-checks", "ep"), ("pairs", "ep"),
                                        ("case-items", "ep"), ("halo", "ep"), ("panel-vs-per-component", "ep"),
                                        ("own-family", "ep"), ("mixed-rubric", "ep"), ("levels-3", "laplace"),
                                        ("binary-checks", "laplace"), ("pairs", "laplace"), ("case-items", "laplace")])
def test_outer_gradient_matches_finite_differences(name, inner):
    """The analytic gradient against central differences of the value, for both inner steps. The EP
    gradient holds only at the EP fixed point (it treats the sites as fixed); the differences start EP
    from the sites at theta, so this also checks that a warm start reaches the same fixed point."""
    design, _ = catalog_design(name, inputs=4)
    prob = EL.Problem(design.data)
    rng = np.random.default_rng(11)
    theta = np.clip(prob.params.init + 0.3 * rng.standard_normal(prob.params.size), prob.params.lo,
                    prob.params.hi)
    res = prob.evaluate(theta, inner=inner)
    assert res["mode"]["ok"]
    cold = prob.evaluate(theta, grad=False, inner=inner)["value"]
    assert res["value"] == pytest.approx(cold, abs=1e-9)
    h = 1e-5
    for k in range(len(theta)):
        e = np.zeros(len(theta))
        e[k] = h
        fd = (prob.evaluate(theta + e, res["mode"], grad=False, inner=inner)["value"]
              - prob.evaluate(theta - e, res["mode"], grad=False, inner=inner)["value"]) / (2 * h)
        assert abs(fd - res["grad"][k]) <= 1e-4 * max(1.0, abs(fd)), (name, prob.params.labels[k], res["grad"][k], fd)


# ---------------------------------------------------------------------------------------------------
# The fit

def test_factor_structure_is_refused_by_name():
    design, _ = catalog_design("linear", inputs=4, structure="factor:1")
    with pytest.raises(NotImplementedError, match="factor"):
        EL.Problem(design.data)


def test_boundary_names_an_sd_already_at_its_bound_and_one_the_data_cannot_see():
    """`linear` has no family effect: setting family_sd to its lower bound makes it 'at the boundary'
    without a trial; the cheap judge's call noise is measured (repeated calls), so it is not named."""
    design, _ = catalog_design("linear", inputs=10, seed=3)
    prob = EL.Problem(design.data)
    P = prob.params
    fitted = EL.fit_theta(prob)
    theta = fitted["theta"].copy()
    k = P.labels.index("family_sd[0]")
    theta[k] = P.lo[k]
    at, unchecked = EL.boundary(prob, theta)
    assert "family_sd[0]" in at and not unchecked
    assert "sigma[2, 0]" not in at


# ---------------------------------------------------------------------------------------------------
# Split-normal theta draws

class _Objective:
    """A stand-in for `Problem` in `split_axes`: `evaluate` gives a known function of theta, and a value
    that is not finite (which `_attempt` refuses) where `refuse(theta)` is true. Keeps the points asked."""

    def __init__(self, f, lo, hi, refuse=lambda theta: False):
        self.f, self.refuse, self.asked = f, refuse, []
        self.params = type("Bounds", (), {"lo": np.asarray(lo, dtype=float), "hi": np.asarray(hi, dtype=float)})()

    def evaluate(self, theta, start=None, grad=True, **kw):
        self.asked.append(np.array(theta, dtype=float))
        value = np.nan if self.refuse(theta) else float(self.f(theta))
        return {"value": value, "grad": np.zeros(len(theta)), "laplace": value,
                "mode": {"bad_cells": 0, "zg": np.zeros(1), "zb": np.zeros((1, 1))}}


def test_split_normal_values_follow_its_distribution_function_and_moments():
    """t = F^-1(Phi(u)): checked against F written out by sides (each tail from its own formula), the closed
    forms of the mean sqrt(2/pi) (b - a) and the second moment (a^3 + b^3) / (a + b), the share b / (a + b)
    above 0, and the identity at equal scales of 1, out to 9 sds."""
    a, b = np.array([0.3, 1.0, 2.5, 1.0]), np.array([1.7, 1.0, 0.4, 3.0])
    u = np.linspace(-8.0, 8.0, 321)[:, None]
    t = EL.split_normal(u, a, b)
    assert np.all(np.diff(t, axis=0) > 0)
    below = 2 * a / (a + b) * special.ndtr(np.minimum(t, 0) / a)          # P(T <= t) for t < 0
    above = 2 * b / (a + b) * special.ndtr(-np.maximum(t, 0) / b)         # P(T > t) for t >= 0
    lower = np.where(t < 0, below, 1.0 - above)
    upper = np.where(t < 0, 1.0 - below, above)
    tail_u = np.where(u < 0, special.ndtr(u), special.ndtr(-u))
    tail_t = np.where(u < 0, lower, upper)
    np.testing.assert_allclose(tail_t, np.broadcast_to(tail_u, tail_t.shape), rtol=1e-9)
    grid = special.ndtri((np.arange(400000) + 0.5) / 400000)[:, None]   # quantiles: a near-exact average
    x = EL.split_normal(grid, a, b)
    np.testing.assert_allclose(x.mean(axis=0), np.sqrt(2 / np.pi) * (b - a), atol=2e-4)
    np.testing.assert_allclose((x ** 2).mean(axis=0), (a ** 3 + b ** 3) / (a + b), rtol=2e-4)
    np.testing.assert_allclose((x > 0).mean(axis=0), b / (a + b), atol=1e-5)
    z = np.linspace(-9.0, 9.0, 181)
    np.testing.assert_allclose(EL.split_normal(z, np.ones_like(z), np.ones_like(z)), z, rtol=0, atol=1e-12)


def test_split_axes_recovers_the_side_scales_of_a_split_gaussian_objective():
    """On an objective that is a half-Gaussian on each side of each axis (rotated, unequal sds), the side
    scales come back exactly, from points 2 sds out on each axis."""
    rng = np.random.default_rng(4)
    V, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    axes = V * np.array([0.3, 1.0, 2.0])
    m, top = np.array([0.4, -1.0, 2.0]), 3.7
    a, b = np.array([0.6, 1.0, 1.7]), np.array([1.4, 0.8, 1.0])

    def f(theta):
        t = np.linalg.solve(axes, theta - m)
        return top - float(np.sum(t ** 2 / (2 * np.where(t < 0, a, b) ** 2)))

    stub = _Objective(f, m - 50, m + 50)
    got = EL.split_axes(stub, m, axes, None, top)
    np.testing.assert_allclose(got["minus"], a, rtol=1e-10)
    np.testing.assert_allclose(got["plus"], b, rtol=1e-10)
    assert (got["evaluations"], got["refused"], got["kept"], got["held"]) == (6, 0, 0, 0)
    want = sorted(tuple(np.round(m + s * EL.SPLIT_REACH * axes[:, k], 12)) for k in range(3) for s in (-1, 1))
    assert sorted(tuple(np.round(p, 12)) for p in stub.asked) == want
    nothing = EL.split_axes(stub, m, axes, None, np.nan)   # no value at theta-hat: every side keeps 1
    assert nothing["kept"] == 6 and nothing["evaluations"] == 0 and np.all(nothing["minus"] == 1)


def test_split_axes_cuts_to_the_bounds_and_keeps_or_holds_a_side_it_cannot_measure():
    """Axis-aligned: room for 1 sd only (measured there, still exact), room under `SPLIT_LEAST` (kept at
    1, not evaluated), a value that rises (held at the upper limit), a side so steep its scale is under
    the lower limit (held), a point refused at 2 sds and taken at 1 (exact), and refused at both (kept)."""
    s = np.array([0.5, 1.0, 2.0])
    m, top = np.zeros(3), 0.0
    a, b = np.array([0.1, 1.3, 0.7]), np.array([1.6, 0.9, 1.0])   # b[2] unused: that side rises

    def f(theta):
        t = theta / s
        value = 0.0
        for k in range(3):
            if k == 2 and t[k] > 0:
                value += 0.1 * t[k] ** 2    # rises above the top on this side
            else:
                value -= t[k] ** 2 / (2 * (a[k] if t[k] < 0 else b[k]) ** 2)
        return top + value

    lo, hi = np.array([-50.0, -0.3, -50.0]), np.array([1.0 * s[0], 50.0, 50.0])
    refuse = lambda theta: theta[2] / s[2] < -1.5 or theta[1] / s[1] > 0.4
    stub = _Objective(f, lo, hi, refuse)
    got = EL.split_axes(stub, m, np.diag(s), None, top)
    low, high = EL.SPLIT_RANGE
    np.testing.assert_allclose(got["minus"], [low, 1.0, a[2]], rtol=1e-10)
    np.testing.assert_allclose(got["plus"], [b[0], 1.0, high], rtol=1e-10)
    assert (got["evaluations"], got["refused"], got["kept"], got["held"]) == (7, 3, 2, 2)
    assert any(np.allclose(p, [1.0 * s[0], 0, 0]) for p in stub.asked)          # cut to the room: 1 sd
    assert not any(p[1] < 0 for p in stub.asked)                                # no room: not evaluated


def test_split_normal_draws_put_a_prior_only_sd_near_its_exact_mean():
    """The log of a half-normal(1) sd that only its prior informs: log density x - e^(2x) / 2, mode 0, sd
    1 / sqrt(2). Exact: E[sd] = sqrt(2 / pi), E[sd^2] = 1. Gaussian draws in x give e^(1/4) = 1.284 for
    E[sd]; the split normal of `split_axes` must land within 6% and cut the error of E[sd^2] fivefold."""
    f = lambda x: float(x[0] - np.exp(2 * x[0]) / 2)
    axes = np.array([[1 / np.sqrt(2)]])
    got = EL.split_axes(_Objective(f, [-20.0], [5.0]), np.zeros(1), axes, None, f(np.zeros(1)))
    u = special.ndtri((np.arange(200000) + 0.5) / 200000)[None, :]
    split = EL._theta_draw(np.zeros(1), (axes, got["minus"], got["plus"]), u)[0]
    gauss = EL._theta_draw(np.zeros(1), axes, u)[0]
    exact1, exact2 = np.sqrt(2 / np.pi), 1.0
    assert np.mean(np.exp(gauss)) == pytest.approx(np.exp(0.25), rel=1e-3)
    assert abs(np.mean(np.exp(split)) / exact1 - 1) < 0.06
    assert abs(np.mean(np.exp(2 * split)) - exact2) < abs(np.mean(np.exp(2 * gauss)) - exact2) / 5


def test_sample_draws_theta_from_the_split_normal_by_default_and_names_it():
    """Same fit either way (theta-hat, Hessian); the split normal is the default, the Gaussian is kept on
    request, the `theta_hessian` check names the draws and the split counts, and a wrong name is refused."""
    design, _ = catalog_design("binary-checks", inputs=8)
    call = dict(seed=3, draws=40, chains=2, theta_points=16, find_boundary=False, cores=1)
    split = EL.sample(design.data, **call)
    gauss = EL.sample(design.data, theta_draws="gaussian", **call)
    assert split["theta"] == gauss["theta"]
    hs, hg = split["checks"][2], gauss["checks"][2]
    assert hs["theta_draws"] == "split" and hg["theta_draws"] == "gaussian" and "split_evaluations" not in hg
    n = len(split["theta"]["labels"])
    assert hs["split_kept"] + hs["split_evaluations"] - hs["split_refused"] == 2 * n
    low, high = EL.SPLIT_RANGE
    assert low <= hs["split_scale_min"] <= 1.0 <= hs["split_scale_max"] <= high and hs["ok"]
    assert all(check["ok"] for check in split["checks"]), split["checks"]
    assert not np.array_equal(split["draws"]["sd_input"], gauss["draws"]["sd_input"])
    with pytest.raises(ValueError, match="theta_draws"):
        EL.sample(design.data, theta_draws="student", **call)


def test_sample_returns_the_shape_of_engine_pymc_and_checks():
    design, _ = catalog_design("pairs", inputs=8)
    res = EL.sample(design.data, seed=2, draws=60, chains=2, theta_points=12)
    d = design.data
    A, D, J, N, R = d["A"], d["D"], d["J"], d["F"], d["R"]
    shapes = {"mu": (A, D), "sd_input": (D,), "sd_run": (D,), "corr_input": (D, D), "corr_run": (D, D),
              "q_run": (R, D), "bias": (J, D), "tau": (J, D), "sigma": (J, D), "halo_target": (J,),
              "halo_call": (J,), "family": (J, N, D), "sd_judge_input": (J, D), "sigma_pair": (J, D),
              "position": (J,), "tie_width": (J,)}
    assert set(res["draws"]) == set(engine_pymc.OUTPUTS)
    for name, shape in shapes.items():
        assert res["draws"][name].shape == (2, 60) + shape, name
    assert res["divergences"] == 0 and res["approximate"] and "approximate draws" in res["version"]
    assert {check["name"] for check in res["checks"]} == {"optimizer", "inner_newton", "theta_hessian", "cell_rule"}
    assert res["checks"][1]["inner"] == "ep"   # pairs are cut readings: EP, and groups of them need the cell rule
    assert all(check["ok"] for check in res["checks"]), res["checks"]
    # pairs-only dimension: arm means sum to 0
    only = np.flatnonzero(design.data["pairs_only"])
    assert np.allclose(res["draws"]["mu"][..., only].sum(axis=2), 0.0)


@pytest.mark.parametrize("helper", ["runs", "no PYTHONPATH", "fails", "other copy"])
def test_theta_points_in_a_helper_process_give_the_draws_of_one_process(monkeypatch, helper):
    """Theta points split over 2 processes give exactly the draws and labels one process gives; a helper
    that fails, or that imported another copy of the engine than the caller, has its share run in the
    caller, with the same draws. Without PYTHONPATH the helper still finds the caller's copy, ahead of any
    other installed one."""
    design, _ = catalog_design("pairs", inputs=8)
    call = dict(seed=2, draws=60, chains=2, theta_points=12, cores=2, find_boundary=False)
    monkeypatch.setattr(EL, "THETA_PARALLEL_SECONDS", float("inf"))
    alone = EL.sample(design.data, **call)
    monkeypatch.setattr(EL, "THETA_PARALLEL_SECONDS", 0.0)
    if helper == "no PYTHONPATH":
        monkeypatch.delenv("PYTHONPATH", raising=False)
    if helper == "fails":
        monkeypatch.setattr(EL, "_THETA_WORKER", "import sys; sys.exit(3)")
    if helper == "other copy":   # the helper is told the caller's engine is another file
        monkeypatch.setattr(EL, "_THETA_WORKER", "import sys\nsys.argv[2] = '/elsewhere/engine_laplace.py'\n"
                            + EL._THETA_WORKER)
    split = EL.sample(design.data, **call)
    assert alone["checks"][1]["theta_helpers"] == 0
    assert split["checks"][1]["theta_helpers"] == (1 if helper in ("runs", "no PYTHONPATH") else 0)
    np.testing.assert_array_equal(alone["theta_point"], split["theta_point"])
    assert set(alone["draws"]) == set(split["draws"])
    for name in alone["draws"]:
        np.testing.assert_array_equal(alone["draws"][name], split["draws"][name], err_msg=name)


def _helper_setup(monkeypatch, worker: str, seed: int = 0, jobs_seed: int = 0):
    """The review's helper case: a linear design of 8 inputs (simulation `seed`), 7 theta-point jobs of 2 draws
    (generators spawned from `jobs_seed`), 2 workers, helpers whatever the time, each running `worker`.
    Returns (problem, theta, chol, start, jobs, the helper processes started, as handles)."""
    design, _ = catalog_design("linear", inputs=8, seed=seed)
    prob = EL.Problem(design.data)
    prob.workers = 2
    theta = prob.params.init
    start = prob.evaluate(theta)["mode"]
    monkeypatch.setattr(EL, "THETA_PARALLEL_SECONDS", 0.0)
    monkeypatch.setattr(EL, "_THETA_WORKER", worker)
    procs: list = []

    class Recorded(subprocess.Popen):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            procs.append(self)

    monkeypatch.setattr(subprocess, "Popen", Recorded)
    jobs = [(gen, 2) for gen in np.random.default_rng(jobs_seed).spawn(7)]
    return prob, theta, np.eye(len(theta)) * 0.01, start, jobs, procs


@pytest.mark.parametrize("failure", ["interrupt while waiting", "wait fails", "pipe thread cannot start"])
def test_a_failure_while_a_helper_runs_kills_and_reaps_it_and_keeps_the_exception(monkeypatch, failure):
    """The review's cleanup cases, with a helper that holds its share for a minute: a real SIGINT while the
    caller waits for the helper, an exception from that wait, or a pipe thread that cannot start. The
    original exception reaches the caller at once; the helper was killed and reaped by its handle (its
    return code is set without the test polling it); no pipe thread is left and its pipes are closed; the
    cell rule has its threads back."""
    prob, theta, chol, start, jobs, procs = _helper_setup(monkeypatch, "import time\ntime.sleep(60)")
    real_join, real_start, timers = threading.Thread.join, threading.Thread.start, []

    def join(self, timeout=None):
        if self.name == "theta-helper" and not timers:
            if failure == "wait fails":
                timers.append(None)
                raise RuntimeError("the wait failed")
            timers.append(threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT)))   # while blocked below
            timers[0].start()
        return real_join(self, timeout)

    def start_thread(self):
        if self.name == "theta-helper":
            raise RuntimeError("can't start new thread")
        return real_start(self)

    if failure == "pipe thread cannot start":
        monkeypatch.setattr(threading.Thread, "start", start_thread)
    else:
        monkeypatch.setattr(threading.Thread, "join", join)
    raised = {"interrupt while waiting": (KeyboardInterrupt, None), "wait fails": (RuntimeError, "the wait failed"),
              "pipe thread cannot start": (RuntimeError, "can't start new thread")}[failure]
    handler = signal.signal(signal.SIGINT, signal.default_int_handler)
    began = time.time()
    try:
        with pytest.raises(raised[0], match=raised[1]):
            EL._theta_points(prob, theta, chol, start, jobs)
        took = time.time() - began
        codes = [proc.returncode for proc in procs]
        threads = [t for t in threading.enumerate() if t.name == "theta-helper"]
        pipes = [pipe.closed for proc in procs for pipe in (proc.stdin, proc.stdout, proc.stderr)]
    finally:
        for timer in timers:
            if timer is not None:
                timer.cancel()
        signal.signal(signal.SIGINT, handler)
        for proc in procs:   # a net for this test only: the asserts below have already read the state
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert codes == [-signal.SIGKILL] and threads == [] and pipes == [True] * 3
    assert took < 20 and prob.workers == 2


def test_a_helper_whose_pipe_fails_is_over_before_its_share_runs_here(monkeypatch):
    """The review's case: linear, 8 inputs, simulation seed 11, 7 jobs of 2 draws from generator seed 19, 2
    workers, the real helper, and its `communicate` raising before the payload is sent. The pipe thread keeps
    nothing, so the share is run here and the call returns normally: the results are those of one process
    and no helper counts as used. On return, before any net of the test, the helper was killed and reaped
    (its return code is set without the test polling it), no pipe thread is left and its pipes are closed."""
    prob, theta, chol, start, jobs, procs = _helper_setup(monkeypatch, EL._THETA_WORKER, seed=11, jobs_seed=19)
    prob.workers = 1
    alone = [EL._theta_point(prob, theta, chol, start, gen, n)
             for gen, n in [(gen, 2) for gen in np.random.default_rng(19).spawn(7)]]
    prob.workers = 2

    def broken(self, *args, **kw):
        raise OSError("injected pipe failure")

    monkeypatch.setattr(subprocess.Popen, "communicate", broken)   # the recording class of `_helper_setup`
    try:
        results, used = EL._theta_points(prob, theta, chol, start, jobs)
        codes = [proc.returncode for proc in procs]
        threads = [t for t in threading.enumerate() if t.name == "theta-helper"]
        pipes = [pipe.closed for proc in procs for pipe in (proc.stdin, proc.stdout, proc.stderr)]
    finally:
        for proc in procs:   # a net for this test only: the asserts below have already read the state
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert used == 0 and len(results) == len(alone) == 7
    for got, want in zip(results, alone):
        assert got[1:] == want[1:] and set(got[0]) == set(want[0])
        for name in want[0]:
            np.testing.assert_array_equal(got[0][name], want[0][name], err_msg=name)
    assert codes == [-signal.SIGKILL] and threads == [] and pipes == [True] * 3 and prob.workers == 2


_KILLED_CALLER = """
import subprocess
from pathlib import Path
import numpy as np
from loopmath.eval import contracts as C, engine_laplace as EL, model as M, simulate as S, specs
sim = S.generate(specs.get("linear", inputs=8), 0)
design = M.build(C.Study(Path("."), sim.runs, sim.scores, sim.pairs, sim.items), sim.rubric, sim.judges,
                 allow_simulated=True)
prob = EL.Problem(design.data)
prob.workers = 2
theta = prob.params.init
start = prob.evaluate(theta)["mode"]
EL.THETA_PARALLEL_SECONDS = 0.0
EL._THETA_WORKER = HELD_SOURCE
real = subprocess.Popen
class Told(real):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        print(self.pid, flush=True)
subprocess.Popen = Told
jobs = [(gen, 2) for gen in np.random.default_rng(0).spawn(7)]
EL._theta_points(prob, theta, np.eye(len(theta)) * 0.01, start, jobs)
"""
# the real helper, whose every point holds for a minute once it has read its share
_HELD_WORKER = ("import os, sys, time\nif sys.argv[1] not in sys.path: sys.path.insert(0, sys.argv[1])\n"
                "import loopmath.eval.engine_laplace as E\n"
                "def held(*args, **kw):\n    open(os.environ['THETA_READY'], 'w').close()\n    time.sleep(60)\n"
                "E._theta_point = held\nE._theta_worker()")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_a_helper_exits_by_itself_when_its_caller_is_killed(tmp_path):
    """A caller killed outright (SIGKILL: no cleanup of its own can run) leaves no helper behind. The helper
    has read its share and is holding it for a minute; it sees its parent change and exits within a few
    `PARENT_POLL_SECONDS`."""
    ready, script = tmp_path / "ready", tmp_path / "caller.py"
    script.write_text(_KILLED_CALLER.replace("HELD_SOURCE", repr(_HELD_WORKER)))
    env = {**os.environ, "PYTHONPATH": str(Path(EL.__file__).parents[2]), "THETA_READY": str(ready),
           "OMP_NUM_THREADS": "1", "LOOPMATH_NO_UPDATE_CHECK": "1"}
    caller = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)
    helper = None
    try:
        line = caller.stdout.readline()
        assert line.strip().isdigit(), caller.stderr.read()
        helper = int(line)
        deadline = time.time() + 60
        while not ready.exists():
            assert caller.poll() is None and time.time() < deadline, caller.stderr.read()
            time.sleep(0.05)
        caller.kill()
        caller.wait()
        killed = time.time()
        while _alive(helper) and time.time() < killed + 15:
            time.sleep(0.05)
        assert not _alive(helper), "the helper outlived its caller"
    finally:
        if caller.poll() is None:
            caller.kill()
        caller.communicate()
        if helper is not None and _alive(helper):   # a net for this test only: our helper, by its pid
            told = subprocess.run(["ps", "-o", "command=", "-p", str(helper)], capture_output=True, text=True)
            if "_theta_worker" in told.stdout:
                os.kill(helper, signal.SIGKILL)


_SPOIL_CALL = dict(seed=2, draws=60, chains=2, theta_points=12, cores=1)


@functools.lru_cache(maxsize=1)
def _clean_pairs_fit():
    design, _ = catalog_design("pairs", inputs=8)
    return design, EL.sample(design.data, **_SPOIL_CALL)


def _spoil(monkeypatch, phase: str, *, calls: set | None = None, field: str = "logZ", doubled: bool = False) -> dict:
    """Make `field[0]` of every `cell_integral` result NaN while the engine function `phase` runs: in every
    call, or with `calls` only within those evaluations (`Problem.evaluate`, counted from 1 in the phase).
    With `doubled`, only in the doubled rule of the cell rule check. Returns the state; with `calls`, its
    `seen` lists the theta of each evaluation in the phase."""
    rule, evaluate = EL.cell_integral, EL.Problem.evaluate
    state = {"on": False, "spoil": False, "n": 0, "seen": []}

    def spoiled(*args, **kw):
        out = rule(*args, **kw)
        if state["spoil"] and field in out and (not doubled or kw.get("panels") == 2 * EL.PANELS):
            out = dict(out, **{field: out[field].copy()})
            out[field].flat[0] = np.nan
        return out

    def counted(self, *args, **kw):
        if not state["on"] or calls is None:
            return evaluate(self, *args, **kw)
        state["n"] += 1
        state["seen"].append(np.array(args[0], dtype=float))
        state["spoil"] = state["n"] in calls
        try:
            return evaluate(self, *args, **kw)
        finally:
            state["spoil"] = False

    inner = getattr(EL, phase)

    def wrapped(*args, **kw):
        state["on"], state["spoil"] = True, calls is None
        try:
            return inner(*args, **kw)
        finally:
            state["on"], state["spoil"] = False, False

    monkeypatch.setattr(EL, "cell_integral", spoiled)
    monkeypatch.setattr(EL.Problem, "evaluate", counted)
    monkeypatch.setattr(EL, phase, wrapped)
    return state


_SPOILS = {"one boundary trial": ("boundary", {2}, False, []),
           "boundary scan": ("boundary", None, False, []),
           "one theta draw": ("_theta_points", {1}, False, []),
           "polish": ("polish", None, False, ["optimizer"]),
           "six theta attempts": ("_theta_points", set(range(1, 7)), False, ["inner_newton"]),
           "cell rule check": ("cell_rule_check", None, True, ["cell_rule"])}


@pytest.mark.parametrize("case", list(_SPOILS))
def test_a_cell_value_that_is_not_finite_is_rejected_or_fails_a_check(monkeypatch, case):
    """One group's log Z made NaN in one phase of the fit. A trial point that meets it is rejected and
    another tried, which alone fails no check: one boundary trial (the next level is tried), the whole
    boundary scan (its sds are reported unchecked), one theta draw (drawn again). A result the fit needs
    fails a check: the evaluation at theta-hat in polish fails `optimizer`; the first theta point with all
    six attempts spoiled, theta-hat included, fails `inner_newton`, and its draws are NaN rather than drawn
    from a rejected result; the doubled rule fails `cell_rule`. No check value is NaN or infinite."""
    design, clean = _clean_pairs_fit()
    assert all(c["ok"] for c in clean["checks"])
    base_newton = {c["name"]: c for c in clean["checks"]}["inner_newton"]
    assert base_newton["bad_cells"] == 0 and base_newton["lost_theta_points"] == 0
    phase, calls, doubled, expect = _SPOILS[case]
    state = _spoil(monkeypatch, phase, calls=calls, doubled=doubled)
    res = EL.sample(design.data, **_SPOIL_CALL)
    json.dumps(res["checks"], allow_nan=False)   # a fit file holds no NaN or infinity
    checks = {c["name"]: c for c in res["checks"]}
    assert sorted(name for name, c in checks.items() if not c["ok"]) == expect, res["checks"]
    newton, rule = checks["inner_newton"], checks["cell_rule"]
    if case == "cell rule check":
        assert rule["bad_groups"] >= 1 and rule["value"] is None and newton["bad_cells"] == 0
    else:
        assert newton["bad_cells"] >= 1 and rule["bad_groups"] == 0
    mu = res["draws"]["mu"]
    if case == "six theta attempts":
        assert newton["lost_theta_points"] == 1 and newton["redrawn_theta"] == base_newton["redrawn_theta"] + 6
        gone = ~np.isfinite(mu).all(axis=(2, 3))
        assert gone.any() and not gone.all() and np.isnan(mu[gone]).all()
        assert np.all(res["theta_point"][gone] == res["theta_points"])   # the theta-hat label
        checked = {**FT.existing(design), "halo_call": design.masks["panel"]}
        told = FT.diagnose(res["draws"], checked, 0, 0.0, clusters=res["theta_point"])
        assert not told["converged"] and told["checks"][-1]["name"] == "finite_draws"
        assert told["checks"][-1]["value"] > 0 and not told["checks"][-1]["ok"]
    else:
        assert np.isfinite(mu).all() and newton["lost_theta_points"] == 0
    if case == "one theta draw":
        assert newton["redrawn_theta"] == base_newton["redrawn_theta"] + 1
    if case == "polish":
        assert checks["optimizer"]["finite"] is False and checks["optimizer"]["value"] is None
        assert checks["theta_hessian"]["ok"]
    if case == "one boundary trial":   # evaluations 1, 2, 3: theta-hat, the spoiled trial, its retry
        base, first, second = state["seen"][:3]
        k = np.flatnonzero(first != base)
        assert len(k) == 1 and list(np.flatnonzero(second != base)) == list(k) and second[k[0]] > first[k[0]]
        # the same sd at the next level (1e-3 of the prior scale), which here decides as the first level did
        assert res["boundary_unchecked"] == clean["boundary_unchecked"] and res["boundary"] == clean["boundary"]
    if case == "boundary scan":   # theta-hat itself was spoiled: every sd not at its bound is unchecked
        assert len(res["boundary_unchecked"]) > 0 and set(res["boundary"]) <= set(clean["boundary"])


@pytest.mark.parametrize("field", ["logZ", "mean", "var", "tail", "dlo", "dhi"])
def test_a_cell_field_that_is_not_finite_stops_ep_at_its_fixed_point(monkeypatch, field):
    """Warm from a converged EP evaluation (the review's case: levels-3, 3 inputs), one cell's field made NaN:
    EP must not take it into an update or a convergence decision. The evaluation is refused: no value, no
    gradient, the inner step not ok. Bound derivatives are asked for only with the gradient, so the value
    alone stays as it was."""
    design, _ = catalog_design("levels-3", inputs=3, seed=2)
    prob = EL.Problem(design.data)
    prob.workers = 1
    theta = prob.params.init
    base = prob.evaluate(theta)
    assert base["mode"]["inner"] == "ep" and base["mode"]["ok"] and EL._usable(base)
    rule, seen = EL.cell_integral, [0]

    def spoiled(*args, **kw):
        out = rule(*args, **kw)
        if field in out:
            seen[0] += 1
            out = dict(out, **{field: out[field].copy()})
            out[field].flat[0] = np.nan
        return out

    monkeypatch.setattr(EL, "cell_integral", spoiled)
    res = prob.evaluate(theta, base["mode"])
    assert seen[0] >= 1
    assert not res["mode"]["ok"] and res["mode"]["bad_cells"] >= 1 and not EL._usable(res)
    assert np.isnan(res["value"]) and np.isnan(res["grad"]).all()
    if field in ("dlo", "dhi"):
        alone = prob.evaluate(theta, base["mode"], grad=False)
        assert alone["mode"]["ok"] and alone["value"] == pytest.approx(base["value"], rel=0, abs=1e-8)


def test_polish_refuses_a_value_that_is_not_finite_even_with_a_finite_gradient(monkeypatch):
    """The review's polish case: the objective at theta-hat NaN while its gradient is finite (and the
    Newton decrement tiny). Polish must not report that point as converged."""
    design, _ = catalog_design("levels-3", inputs=3, seed=2)
    prob = EL.Problem(design.data)
    prob.workers = 1
    theta = prob.params.init
    good = prob.evaluate(theta)
    H = np.eye(len(theta)) * 1e12   # any step gains next to nothing
    assert EL.polish(prob, theta, H, good["mode"])["ok"]
    evaluate = EL.Problem.evaluate

    def nan_value(self, *args, **kw):
        out = evaluate(self, *args, **kw)
        return dict(out, value=np.nan)

    monkeypatch.setattr(EL.Problem, "evaluate", nan_value)
    told = EL.polish(prob, theta, H, good["mode"])
    assert not told["ok"] and told["res"] is None and not np.isfinite(told["decrement"])

def test_a_fit_whose_final_evaluation_is_not_finite_is_written_and_not_converged(monkeypatch, tmp_path):
    """The review's polish probe: levels-3, 3 inputs, simulation seed 2, fit seed 9, 4 chains, 1000 saved
    draws, 1 core, one group's log Z NaN only while polish runs. The objective at theta-hat is then not
    finite: `optimizer` fails (no decrement), the fit is not converged, and the fit file is still written."""
    design, _ = catalog_design("levels-3", inputs=3, seed=2)
    _spoil(monkeypatch, "polish")
    meta, draws = FT.fit_design(design, seed=9, cores=1, engine="laplace")
    checks = {c["name"]: c for c in meta["diagnostics"]["checks"]}
    assert not meta["diagnostics"]["converged"]
    assert [name for name, c in checks.items() if not c["ok"]] == ["optimizer"]
    assert checks["optimizer"]["finite"] is False and checks["optimizer"]["value"] is None
    path = F.write_fit(tmp_path / "fit.json", meta, draws)
    assert F.read_fit(path, allow_unconverged=True).meta["diagnostics"]["converged"] is False


@pytest.mark.parametrize("lost", ["the first point", "every point"])
def test_a_fit_with_lost_theta_points_is_written_and_not_converged(monkeypatch, tmp_path, lost):
    """The review's lost-point probe: levels-3, 3 inputs, simulation seed 2, fit seed 9, 4 chains, 1000 saved
    draws, 1 core, one group's log Z NaN in all six attempts of the first theta point, or of every point (no
    usable draw row at all). The lost rows are NaN, so no summary that reads them is given: each stat and
    each correlation summary is null, never an estimate from the other rows. The fit fails `inner_newton`
    and `finite_draws`, is not converged, is written, and reads back only with `allow_unconverged`."""
    design, _ = catalog_design("levels-3", inputs=3, seed=2)
    _spoil(monkeypatch, "_theta_points", calls=set(range(1, 7)) if lost == "the first point" else None)
    meta, draws = FT.fit_design(design, seed=9, cores=1, engine="laplace")
    diagnostics = meta["diagnostics"]
    checks = {c["name"]: c for c in diagnostics["checks"]}
    assert diagnostics["converged"] is False
    assert not checks["inner_newton"]["ok"] and not checks["finite_draws"]["ok"]
    assert checks["inner_newton"]["lost_theta_points"] == (1 if lost == "the first point" else EL.THETA_POINTS)
    rows = np.isfinite(draws["mu"]).all(axis=(1, 2))   # saved draw rows, chains flattened
    assert not rows.all() and (rows.any() if lost == "the first point" else not rows.any())
    for level in ("input", "run"):
        assert meta["levels"][level]["corr"] is None
        assert all(one is None for one in meta["levels"][level]["sd"])
    assert all(row[name] is None for arm in meta["arms"] for row in arm["dims"] for name in ("reported", "fitting"))
    cells = [row for target in meta["targets"] for row in target["dims"]]
    assert cells and all(row["pooled"] is None and row["unpooled"] is None for row in cells)
    path = F.write_fit(tmp_path / "fit.json", meta, draws)
    with pytest.raises(C.ContractError, match="did not converge"):
        F.read_fit(path)
    back = F.read_fit(path, allow_unconverged=True)
    assert back.converged is False and back.meta["levels"] == meta["levels"]
    assert {c["name"]: c["ok"] for c in back.meta["diagnostics"]["checks"]} == {n: c["ok"] for n, c in checks.items()}
    for name in draws:
        np.testing.assert_array_equal(back.draws[name], draws[name], err_msg=name)


def test_a_correlation_summary_may_be_null_only_in_a_fit_that_did_not_converge(tmp_path):
    """A null correlation summary stands for draws a lost theta point left NaN. In a fit marked converged it
    is refused: such a fit has every draw."""
    design, _ = catalog_design("linear", inputs=4, seed=3)
    meta, draws = FT.fit_design(design, seed=1, cores=1, engine="laplace")
    assert meta["diagnostics"]["converged"] and meta["levels"]["run"]["corr"] is not None
    gone = dict(meta, levels={**meta["levels"], "run": {**meta["levels"]["run"], "corr": None}})
    with pytest.raises(C.ContractError, match="levels.run.corr"):
        F.write_fit(tmp_path / "fit.json", gone, draws)
    failed = dict(gone, diagnostics={**meta["diagnostics"], "converged": False})
    assert F.read_fit(F.write_fit(tmp_path / "fit.json", failed, draws), allow_unconverged=True).converged is False


def test_the_s3_pilot_case_fits_without_overflow_or_an_untrusted_cell():
    """panel-vs-per-component, seed 1 (the S3 pilot): the boundary scan moved cavity means hundreds of
    curvature scales from the cell rule's warm start; the mode search stopped short, exp overflowed, and
    the NaN was absorbed while every check passed. Now no overflow, no group that is not ok."""
    import warnings

    sim = S.generate(specs.get("panel-vs-per-component"), 1)
    design = M.build(C.Study(Path("."), sim.runs, sim.scores, sim.pairs, sim.items), sim.rubric, sim.judges,
                     allow_simulated=True)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        res = EL.sample(design.data, seed=1, draws=50, chains=2, theta_points=20, cores=1)
    numeric = [str(g.message) for g in got if issubclass(g.category, RuntimeWarning)]
    assert numeric == []
    checks = {c["name"]: c for c in res["checks"]}
    assert checks["inner_newton"]["bad_cells"] == 0 and checks["cell_rule"]["bad_groups"] == 0
    assert all(c["ok"] for c in res["checks"]), res["checks"]


def test_fit_recovers_a_linear_study_and_names_absent_terms_at_the_boundary():
    """`linear` simulates no family effect and no judge-by-input term: both family sds are at the
    boundary. The cheap judge's disagreement (true 0.08) is not. Arm means and their difference sit
    within 3 posterior sds of the truth (a single study: coverage itself is S3's bench)."""
    design, sim = catalog_design("linear", inputs=40, seed=0)
    res = EL.sample(design.data, seed=0)
    assert all(check["ok"] for check in res["checks"])
    edge = set(res["boundary"])
    assert {"family_sd[0]", "family_sd[1]"} <= edge
    assert not {"tau[2, 0]", "tau[2, 1]", "sigma[2, 0]", "sigma[2, 1]"} & edge
    mu = res["draws"]["mu"].reshape(-1, 2, 2)
    true_mu = np.asarray(sim.truth["mu"], dtype=float)
    assert np.all(np.abs(mu.mean(axis=0) - true_mu) < 3 * mu.std(axis=0)), (mu.mean(axis=0), true_mu)
    diff = mu[:, 1] - mu[:, 0]
    true_diff = true_mu[1] - true_mu[0]
    assert np.all(np.abs(diff.mean(axis=0) - true_diff) < 3 * diff.std(axis=0)), (diff.mean(axis=0), true_diff)
    tau = res["draws"]["tau"].reshape(-1, 3, 2).mean(axis=0)
    assert np.allclose(tau[1:], np.asarray(sim.truth["tau"], dtype=float)[1:], atol=0.03)


def test_fit_design_with_the_laplace_engine_writes_a_fit_marked_approximate(tmp_path):
    design, _ = catalog_design("linear", inputs=10, seed=1)
    meta, draws = FT.fit_design(design, seed=1, engine="laplace")
    assert meta["engine"]["name"] == "laplace" and "approximate draws" in meta["engine"]["version"]
    assert meta["options"]["engine"] == "laplace" and meta["options"]["approximate_draws"] is True
    names = [check["name"] for check in meta["diagnostics"]["checks"]]
    assert {"optimizer", "inner_newton", "theta_hessian", "boundary"} <= set(names)
    assert meta["diagnostics"]["converged"] and next(iter(draws.values())).shape[0] == 1000
    path = F.write_fit(tmp_path / "fit.json", meta, draws)
    fit = F.read_fit(path)
    assert fit.meta["engine"]["name"] == "laplace"
    with pytest.raises(C.ContractError):
        FT.fit_design(design, engine="stan")
    # the effective sample size counts theta points, not saved draws: a function of theta alone (the sds)
    # has at most as many independent values as there are theta points
    ess = next(check for check in meta["diagnostics"]["checks"] if check["name"] == "ess")
    assert ess["theta_points"] == meta["options"]["theta_points"] == EL.THETA_POINTS
    assert ESS_MIN_FOR_TEST <= meta["diagnostics"]["ess_min"] <= ess["theta_points"]


ESS_MIN_FOR_TEST = 100.0   # the gate of `fit.py`, written here so a change there shows in this file


def test_cluster_ess_against_the_spread_of_the_mean_over_repeated_studies():
    """K random theta points, n draws each, x = b_k + w (between sd 1, within sd 2). The effective size
    of the mean is its total variance over the variance of the mean, measured here over 3000 repeated
    simulations. `cluster_ess` from one simulation must sit near it (median within 10%), and a
    function of theta alone (w = 0) must get exactly K, for 10 draws per point and for 1000."""
    rng = np.random.default_rng(3)
    K, n, reps = 20, 10, 3000
    b = rng.standard_normal((reps, K, 1))
    w = 2.0 * rng.standard_normal((reps, K, n))
    x = (b + w).reshape(reps, K * n)
    measured = 5.0 / x.mean(axis=1).var()
    labels = np.repeat(np.arange(K), n)[None, :]
    got = [FT.cluster_ess(x[r][None, :, None], labels)[0] for r in range(200)]
    assert abs(np.median(got) / measured - 1) < 0.10, (np.median(got), measured)
    for repeats in (10, 1000):
        theta_only = np.repeat(rng.standard_normal(K), repeats)
        lab = np.repeat(np.arange(K), repeats)
        assert FT.cluster_ess(theta_only[None, :, None], lab[None])[0] == pytest.approx(K)


@pytest.mark.parametrize("draws", [250, 1000])
def test_four_theta_points_fail_the_effective_sample_size_however_many_draws(draws):
    """The review's case: 4 theta points must fail the gate, and more saved draws per point must not
    raise the effective sample size above the number of points."""
    sim = S.generate(specs.get("linear", inputs=8), 11)
    design = M.build(C.Study(Path("."), sim.runs, sim.scores, sim.pairs, sim.items), sim.rubric, sim.judges,
                     allow_simulated=True)
    res = EL.sample(design.data, seed=9, draws=draws, chains=4, theta_points=4, find_boundary=False)
    labels = res["theta_point"]
    assert labels.shape == (4, draws) and len(np.unique(labels)) <= 4
    checked = {**FT.existing(design), "halo_call": design.masks["panel"]}
    told = FT.diagnose(res["draws"], checked, 0, 0.0, clusters=labels)
    assert not told["converged"] and told["ess_min"] <= 4
    sd = np.asarray(res["draws"]["sd_run"]).reshape(4, draws, -1)
    assert FT.cluster_ess(sd, labels).max() <= 4 + 1e-9


def test_eval_fit_engine_laplace_on_the_command_line(tmp_path, capsys):
    folder = tmp_path / "study"
    assert cli.main(["eval", "simulate", "one-component", "--out", str(folder), "--seed", "4"]) == 0
    code = cli.main(["eval", "fit", str(folder), "--rubric", str(folder / "rubric.json"), "--judges",
                     str(folder / "judges.json"), "--allow-simulated", "--engine", "laplace"])
    assert code == 0
    meta = json.loads((folder / "fit.json").read_text())
    assert meta["engine"]["name"] == "laplace" and meta["diagnostics"]["converged"]
    assert "converged: yes" in capsys.readouterr().out
    assert cli.main(["eval", "judges", str(folder / "fit.json")]) == 0


# ---------------------------------------------------------------------------------------------------
# Slow: the same study fitted with PyMC

@slow
@pytest.mark.skipif(not engine_pymc.available(), reason="needs the [bayes] extra")
def test_laplace_and_pymc_agree_on_a_linear_study():
    design, _ = catalog_design("linear", inputs=40, seed=1)
    lap = EL.sample(design.data, seed=1)["draws"]
    mc = engine_pymc.sample(design.data, seed=1, draws=500, chains=2, cores=2)["draws"]
    for name in ("mu", "bias", "sigma"):
        a, b = lap[name].reshape(-1, *lap[name].shape[2:]), mc[name].reshape(-1, *mc[name].shape[2:])
        fixed = b.std(axis=0) < 1e-12   # an anchored entry (the first judge's bias) is 0 in every draw of both
        assert np.all(np.abs(a[:, fixed] - b[:, fixed].mean(axis=0)) < 1e-12), name
        sd = b.std(axis=0)[~fixed]
        assert np.all(np.abs(a.mean(axis=0)[~fixed] - b.mean(axis=0)[~fixed]) < 0.5 * sd), name
        assert np.all(np.abs(a.std(axis=0)[~fixed] / sd - 1.0) < 0.35), name
