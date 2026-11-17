"""The exact oracle for tiny designs (`laplace_oracle.py`), checked against values computed another way.

The references: the normal density of every reading written (Gaussian scales); closed forms for one call
on probit scales and Owen's T for two calls; scipy's adaptive quadrature in one dimension; scipy's
multivariate normal rectangle probabilities (Genz) over the latent value of every reading, item and pair;
a Laplace approximation taken with scipy's optimiser and finite differences; and the oracle's own two
quadratures (adaptive Gauss-Kronrod and Gauss-Hermite) against each other and at more nodes.

Every reference is built from the rows the tiny study wrote and the model as PRD section 7 states it,
not from the layout of `model.build`, so the oracle's reading of the layout is checked too. All data is
synthetic.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, linalg, optimize, special, stats
from scipy.stats import qmc

import laplace_oracle as O

HERE = Path(__file__).parent
INF = math.inf


# ---------------------------------------------------------------------------------------------------
# references built from the rows written

def latent_model(b: O.Tiny):
    """The latent value of every reading (one per item of an items call) and pair the tiny study wrote,
    as `mean + L @ n` over independent unit normals n, with its observed interval or Gaussian value.

    Returns `(mean, L, lo, hi, y, const, Q, q_mean)`: `y` is NaN for a rectangle observation; `const` the
    log binomial coefficients; `Q`, `q_mean` the run qualities on the same normals."""
    th = b.theta
    inputs: dict = {}
    run_input = [inputs.setdefault(run["input"], len(inputs)) for run in b.study.runs]
    R = len(b.study.runs)
    mu, bias, tau, sigma = (th[name][:, 0] for name in ("mu", "bias", "tau", "sigma"))
    sd_in, sd_run = float(th["sd_input"][0]), float(th["sd_run"][0])
    index: dict = {}
    obs: list = []
    const = 0.0

    def col(name) -> int:
        return index.setdefault(name, len(index))

    def add(*parts: dict) -> dict:
        out: dict = {}
        for part in parts:
            for key, value in part.items():
                out[key] = out.get(key, 0.0) + value
        return out

    def quality(r: int, sign: float = 1.0) -> dict:
        return {col(("v", run_input[r])): sign * sd_in, col(("u", r)): sign * sd_run}

    levels = len(b.rubric["components"][0]["scale"].get("levels", [])) or 2
    cuts = np.concatenate([[-INF], (np.arange(levels - 1) + 0.5) / (levels - 1), [INF]])
    for n, (r, j, value) in enumerate(b.readings):
        view = add(quality(r), {col(("t", j, r)): tau[j]})
        call = {col(("e", n)): sigma[j]}
        m = mu[r] + bias[j]
        if b.kind in ("linear", "fraction"):
            obs.append((m, add(view, call), -INF, INF, value))
        elif b.kind == "binary":
            obs.append((m, add(view, call), *((0.0, INF) if value >= 0.5 else (-INF, 0.0)), math.nan))
        elif b.kind == "levels":
            obs.append((m, add(view, call), cuts[value], cuts[value + 1], math.nan))
        else:
            k, items = value
            const += math.log(math.comb(items, k))
            for i in range(items):
                bounds = (0.0, INF) if i < k else (-INF, 0.0)
                obs.append((m, add(view, call, {col(("item", n, i)): 1.0}), *bounds, math.nan))
    for n, (ra, rb, j, o, verdict) in enumerate(b.pairs):
        coefs = add(quality(ra), quality(rb, -1.0), {col(("t", j, ra)): tau[j]}, {col(("t", j, rb)): -tau[j]},
                    {col(("pair", n)): th["sigma_pair"][j, 0]})
        tie = th["tie_width"][j]
        bounds = {0: (tie, INF), 1: (-tie, tie), 2: (-INF, -tie)}[verdict]
        obs.append((mu[ra] - mu[rb] + th["position"][j] * o, coefs, *bounds, math.nan))
    Q = np.zeros((R, len(index) + 2 * R))
    width = Q.shape[1]
    L = np.zeros((len(obs), width))
    for n, (_, coefs, *_rest) in enumerate(obs):
        for key, value in coefs.items():
            L[n, key] = value
    for r in range(R):
        for key, value in quality(r).items():
            Q[r, key] = value
    mean = np.array([x[0] for x in obs], dtype=float)
    lo, hi, y = (np.array([x[k] for x in obs], dtype=float) for k in (2, 3, 4))
    return mean, L, lo, hi, y, const, Q, mu.copy()


def log_rect(mean, cov, lo, hi, seed: int = 0, abseps: float = 1e-12) -> float:
    """log P(lo < X < hi) for X ~ N(mean, cov): a closed form in one dimension, Genz's method above
    (`abseps=0` asks for relative accuracy alone, which a tiny probability needs)."""
    if len(mean) == 1:
        s = math.sqrt(cov[0, 0])
        a, b = (lo[0] - mean[0]) / s, (hi[0] - mean[0]) / s
        return float(np.log(stats.norm.cdf(b) - stats.norm.cdf(a))) if a <= 0 else \
            float(np.log(stats.norm.sf(a) - stats.norm.sf(b)))
    value = stats.multivariate_normal.cdf(hi, mean=mean, cov=cov, lower_limit=lo, maxpts=400_000 * len(mean),
                                          abseps=abseps, releps=1e-9, rng=np.random.default_rng(seed))
    return float(np.log(value))


def log_phi_interval(a, b):
    """log(Phi(b) - Phi(a)) elementwise for a < b, from scipy's log_ndtr on the side of 0 that keeps
    the digits (this file's own, not the oracle's)."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    up = a > 0
    big = special.log_ndtr(np.where(up, -a, b))
    small = special.log_ndtr(np.where(up, -b, a))
    with np.errstate(divide="ignore", invalid="ignore"):
        return big + np.log1p(-np.exp(small - big))


def log_rect_small(mean, cov, lo, hi) -> float:
    """log P(lo < X < hi) for a normal X of one or two coordinates, to about 1e-12 at any size of the
    probability: the interval in one; in two, scipy quad over x1 of its density times the conditional
    interval of x2, over the range where that is within e^-80 of its top (found on a grid of 20001
    points), with 40 breakpoints across it."""
    if len(mean) == 1:
        s = math.sqrt(cov[0, 0])
        return float(log_phi_interval((lo[0] - mean[0]) / s, (hi[0] - mean[0]) / s))
    s1 = math.sqrt(cov[0, 0])
    beta = cov[0, 1] / cov[0, 0]
    s2 = math.sqrt(cov[1, 1] - beta * cov[0, 1])

    def logf(x):
        m2 = mean[1] + beta * (np.asarray(x) - mean[0])
        return stats.norm.logpdf(x, mean[0], s1) + log_phi_interval((lo[1] - m2) / s2, (hi[1] - m2) / s2)

    a, b = max(lo[0], mean[0] - 40 * s1), min(hi[0], mean[0] + 40 * s1)
    xs = np.linspace(a, b, 20001)
    lv = logf(xs)
    top = float(lv.max())
    keep = xs[lv > top - 80.0]
    a, b = max(a, keep.min() - (xs[1] - xs[0])), min(b, keep.max() + (xs[1] - xs[0]))
    points = sorted(set(np.linspace(a, b, 42)[1:-1]) | {float(xs[int(lv.argmax())])})
    value, _ = integrate.quad(lambda x: math.exp(float(logf(x)) - top), a, b, points=points, epsabs=0.0,
                              epsrel=1e-12, limit=5000)
    return top + math.log(value)


def log_rect_sov(mean, cov, lo, hi, seed: int = 0, m: int = 15, reps: int = 16):
    """log P(lo < X < hi) for X ~ N(mean, cov) by Genz's separation of variables with Genz-Bretz
    ordering, every interval and every inverse taken in log space (so a probability of e^-400 keeps
    its digits, where scipy's Genz returns 0), over `reps` scrambled Sobol sets of 2^m points of 64
    bits (the default 30 bits leave a bias of about 3e-10 in a deep tail). Returns the estimate and the
    standard error of the replicates in log units: their spread, not a bound on numerical bias, so a
    check against it keeps an absolute floor."""
    n = len(mean)
    a, b = np.asarray(lo, dtype=float) - mean, np.asarray(hi, dtype=float) - mean
    S = np.array(cov, dtype=float)
    L = np.zeros((n, n))
    y = np.zeros(n)
    for i in range(n):                      # order: the least likely interval next, given the ones before
        best, best_p = i, np.inf
        for k in range(i, n):
            sd = math.sqrt(S[k, k] - L[k, :i] @ L[k, :i])
            mu = L[k, :i] @ y[:i]
            p = float(log_phi_interval((a[k] - mu) / sd, (b[k] - mu) / sd))
            if p < best_p:
                best, best_p = k, p
        for arr in (a, b):
            arr[[i, best]] = arr[[best, i]]
        S[[i, best]] = S[[best, i]]
        S[:, [i, best]] = S[:, [best, i]]
        L[[i, best]] = L[[best, i]]
        L[i, i] = math.sqrt(S[i, i] - L[i, :i] @ L[i, :i])
        for k in range(i + 1, n):
            L[k, i] = (S[k, i] - L[k, :i] @ L[i, :i]) / L[i, i]
        mu = L[i, :i] @ y[:i]
        al, be = (a[i] - mu) / L[i, i], (b[i] - mu) / L[i, i]
        lz = float(log_phi_interval(al, be))
        y[i] = sum(sign * math.exp(-0.5 * x * x - 0.5 * math.log(2 * math.pi) - lz)
                   for sign, x in ((1.0, al), (-1.0, be)) if np.isfinite(x))
    est = []
    for r in range(reps):
        U = qmc.Sobol(d=max(1, n - 1), scramble=True, seed=seed * 1000 + r, bits=64).random_base2(m)
        z = np.zeros((U.shape[0], n))
        logw = np.zeros(U.shape[0])
        for i in range(n):
            mu = z[:, :i] @ L[i, :i]
            al, be = (a[i] - mu) / L[i, i], (b[i] - mu) / L[i, i]
            logw += log_phi_interval(al, be)
            if i < n - 1:
                w = np.clip(U[:, i], 1e-300, 1.0 - 1e-16)
                with np.errstate(divide="ignore"):
                    low = special.ndtri_exp(np.logaddexp(np.log1p(-w) + special.log_ndtr(al),
                                                         np.log(w) + special.log_ndtr(be)))
                    upp = -special.ndtri_exp(np.logaddexp(np.log1p(-w) + special.log_ndtr(-al),
                                                          np.log(w) + special.log_ndtr(-be)))
                z[:, i] = np.where(al > 0, upp, low)
        top = float(logw.max())
        est.append(top + math.log(float(np.mean(np.exp(logw - top)))))
    est = np.array(est)
    p = np.exp(est - est.max())
    return float(est.max() + math.log(p.mean())), float(p.std(ddof=1) / math.sqrt(reps) / p.mean())


def reference_loglik(b: O.Tiny, seed: int = 0, abseps: float = 1e-12) -> float:
    """Gaussian readings by their normal density, the rest by the rectangle probability given them."""
    mean, L, lo, hi, y, const, _, _ = latent_model(b)
    cov = L @ L.T
    g = ~np.isnan(y)
    out = const
    rest = ~g
    m_r, c_r = mean[rest], cov[np.ix_(rest, rest)]
    if g.any():
        cgg = cov[np.ix_(g, g)]
        out += float(stats.multivariate_normal.logpdf(y[g], mean[g], cgg))
        if rest.any():
            gain = np.linalg.solve(cgg, cov[np.ix_(g, rest)]).T
            m_r = m_r + gain @ (y[g] - mean[g])
            c_r = c_r - gain @ cov[np.ix_(g, rest)]
    if rest.any():
        out += log_rect(m_r, c_r, lo[rest], hi[rest], seed, abseps)
    return out


def gaussian_posterior_q(b: O.Tiny):
    """E and sd of the run qualities given every reading, by normal conditioning on the reading level."""
    mean, L, _, _, y, _, Q, q_mean = latent_model(b)
    Q = Q[:, :L.shape[1]]
    cyy, cqy = L @ L.T, Q @ L.T
    m = q_mean + cqy @ np.linalg.solve(cyy, y - mean)
    var = np.diag(Q @ Q.T - cqy @ np.linalg.solve(cyy, cqy.T))
    return m, np.sqrt(var)


def bvn_cdf(h: float, k: float, rho: float) -> float:
    """P(X <= h, Y <= k), standard bivariate normal with correlation rho, by Owen's T (h, k not 0)."""
    r = math.sqrt(1.0 - rho * rho)
    beta = 0.0 if h * k > 0 or (h * k == 0 and h + k >= 0) else 0.5
    return float(0.5 * special.ndtr(h) + 0.5 * special.ndtr(k) - special.owens_t(h, (k - rho * h) / (h * r))
                 - special.owens_t(k, (h - rho * k) / (k * r)) - beta)


def truncated(alpha: float, beta: float):
    """log P(alpha < X < beta), E X and Var X of a standard normal truncated there (scipy.stats)."""
    dist = stats.truncnorm(alpha, beta)
    return float(np.log(stats.norm.cdf(beta) - stats.norm.cdf(alpha))), float(dist.mean()), float(dist.var())


def with_theta(kind: str, **kw):
    theta = kw.pop("theta", None)
    return O.tiny(kind, theta=theta, **kw)


# ---------------------------------------------------------------------------------------------------
# the oracle imports no engine

def test_the_oracle_imports_no_engine_and_no_kernel():
    tree = ast.parse((HERE / "laplace_oracle.py").read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [node.module or ""] + [f"{node.module}.{alias.name}" for alias in node.names]
    for banned in ("engine_laplace", "laplace_kernels", "engine_pymc", "pymc", "pytensor"):
        assert not any(banned in name.split(".") for name in names), (banned, names)


# ---------------------------------------------------------------------------------------------------
# Gaussian scales: closed forms

GAUSSIAN_GRID = [(kind, runs, judges, calls, shared) for kind in ("linear", "fraction")
                 for runs, judges, calls in itertools.product((1, 2, 3), (1, 2), (1, 2, 3))
                 for shared in (False, True) if not (shared and runs == 1)]


@pytest.mark.parametrize("kind,runs,judges,calls,shared", GAUSSIAN_GRID)
def test_gaussian_designs_equal_the_normal_density_of_every_reading(kind, runs, judges, calls, shared):
    b = O.tiny(kind, runs=runs, judges=judges, calls=calls, inputs=[0] * runs if shared else None,
               seed=runs * 100 + judges * 10 + calls)
    out = O.exact(b.data, b.theta)
    assert out.loglik == pytest.approx(reference_loglik(b), abs=1e-9)
    m, sd = gaussian_posterior_q(b)
    np.testing.assert_allclose(out.posterior["q_run"]["mean"][:, 0], m, atol=1e-9)
    np.testing.assert_allclose(out.posterior["q_run"]["sd"][:, 0], sd, atol=1e-9)


@pytest.mark.parametrize("runs,judges,calls,shared", [(1, 1, 2, False), (2, 2, 2, False), (3, 2, 1, True),
                                                      (2, 1, 3, True), (3, 1, 2, False)])
@pytest.mark.parametrize("method", ["auto", "gh"])
def test_gaussian_readings_through_the_quadrature_equal_the_closed_form(runs, judges, calls, shared, method):
    """The same designs with the Gaussian readings as quadrature factors: the reduction to the directions
    the readings span, the blocks and the map back to the random effects, against exact values."""
    b = O.tiny("linear", runs=runs, judges=judges, calls=calls, inputs=[0] * runs if shared else None, seed=7)
    closed = O.exact(b.data, b.theta)
    quad = O.exact(b.data, b.theta, gaussian="quadrature", method=method)
    assert quad.blocks and closed.blocks == []
    assert quad.loglik == pytest.approx(closed.loglik, abs=1e-8)
    for name in ("v", "u", "q_run"):
        np.testing.assert_allclose(quad.posterior[name]["mean"], closed.posterior[name]["mean"], atol=1e-8)
        np.testing.assert_allclose(quad.posterior[name]["sd"], closed.posterior[name]["sd"], atol=1e-8)


def test_random_arm_means_and_biases_are_integrated_with_their_priors():
    """`random=("mu", "bias")`: the readings' normal density with the priors' variance added."""
    b = O.tiny("linear", runs=2, judges=2, calls=2, seed=11)
    out = O.exact(b.data, b.theta, random=("mu", "bias"))
    mean, L, _, _, y, _, _, _ = latent_model(b)
    s = float(b.data["scale"][0])
    loc = float(b.data["mu_loc"][0])
    w = b.data["bias_w"][:, 0]
    # the same readings with mu ~ N(loc, (2s)^2) per arm and bias = s (z - w.z) per judge added
    readings = b.readings
    extra = np.zeros((len(readings), 2 + 2))
    base = np.zeros(len(readings))
    for n, (r, j, _) in enumerate(readings):
        extra[n, r] = 2 * s
        extra[n, 2:] = s * ((np.arange(2) == j) - w)
        base[n] = loc
    cov = L @ L.T + extra @ extra.T
    assert out.loglik == pytest.approx(float(stats.multivariate_normal.logpdf(y, base, cov)), abs=1e-9)


def test_the_pymc_constant_is_what_the_potentials_drop():
    """A linear cell of 2 calls and an items reading of 3, all random effects at 0: exact minus the
    potentials of `engine_pymc` (written out here) is `pymc_constant`."""
    zero = {"sd_input": 0.0, "sd_run": 0.0}
    b = O.tiny("linear", calls=2, theta=zero, seed=3)
    th = b.theta
    y = np.array([value for _, _, value in b.readings])
    s2, own, m = th["sigma"][0, 0] ** 2, th["tau"][0, 0] ** 2, th["mu"][0, 0] + th["bias"][0, 0]
    exact = float(stats.multivariate_normal.logpdf(y, np.full(2, m), own + s2 * np.eye(2)))
    var, ss = own + s2 / 2, float(((y - y.mean()) ** 2).sum())
    potential = -0.5 * math.log(var) - 0.5 * (y.mean() - m) ** 2 / var - 0.5 * math.log(s2) - 0.5 * ss / s2
    assert O.exact_loglik(b.data, th) == pytest.approx(exact, abs=1e-12)
    assert exact - potential == pytest.approx(O.pymc_constant(b.data), abs=1e-12)
    b = O.tiny("items", items=3, theta={**zero, "tau": 0.0, "sigma": 0.0}, seed=4)
    (k, n), m = b.readings[0][2], b.theta["mu"][0, 0] + b.theta["bias"][0, 0]
    potential = k * special.log_ndtr(m) + (n - k) * special.log_ndtr(-m)
    exact = float(stats.binom.logpmf(k, n, special.ndtr(m)))
    assert O.exact_loglik(b.data, b.theta) == pytest.approx(exact, abs=1e-12)
    assert exact - potential == pytest.approx(O.pymc_constant(b.data), abs=1e-12)


# ---------------------------------------------------------------------------------------------------
# probit scales: closed forms for one and two calls

SETTINGS = [{}, {"sigma": 0.3, "tau": 0.9}, {"sd_run": 1.5, "sigma": 0.2}, {"mu": 1.4, "bias": -0.3}]


def one_call_sd(th) -> float:
    return math.sqrt(th["sd_input"][0] ** 2 + th["sd_run"][0] ** 2 + th["tau"][0, 0] ** 2 + th["sigma"][0, 0] ** 2)


@pytest.mark.parametrize("setting", range(len(SETTINGS)))
@pytest.mark.parametrize("kind", ["binary", "levels"])
def test_one_call_on_a_cut_scale_is_a_normal_interval(kind, setting):
    theta = dict(SETTINGS[setting])
    if kind == "levels":
        theta = {k: v / 3 if k != "bias" else v / 10 for k, v in theta.items()}
        theta.setdefault("mu", 0.5)
    for seed in range(3):
        b = O.tiny(kind, theta=theta, seed=seed)
        th = b.theta
        s, m = one_call_sd(th), th["mu"][0, 0] + th["bias"][0, 0]
        _, _, lo, hi, _, _, _, _ = latent_model(b)
        logp, ex, vx = truncated((lo[0] - m) / s, (hi[0] - m) / s)
        out = O.exact(b.data, th)
        assert out.loglik == pytest.approx(logp, abs=1e-10)
        vq = float(th["sd_input"][0] ** 2 + th["sd_run"][0] ** 2)
        assert out.posterior["q_run"]["mean"][0, 0] == pytest.approx(th["mu"][0, 0] + vq / s * ex, abs=1e-9)
        assert out.posterior["q_run"]["sd"][0, 0] == pytest.approx(math.sqrt(vq - vq ** 2 / s ** 2 * (1 - vx)),
                                                                   abs=1e-9)


@pytest.mark.parametrize("setting", range(len(SETTINGS)))
def test_one_items_reading_is_a_binomial_probit_integral(setting):
    """n = 3 by scipy's quad over the call's latent value; n = 1 by the closed form Phi(m / sqrt(1 + V))."""
    for seed in range(3):
        b = O.tiny("items", items=3, theta=SETTINGS[setting], seed=seed)
        th = b.theta
        (k, n), s, m = b.readings[0][2], one_call_sd(th), th["mu"][0, 0] + th["bias"][0, 0]
        ref, err = integrate.quad(lambda x: stats.binom.pmf(k, n, special.ndtr(x)) * stats.norm.pdf(x, m, s),
                                  m - 12 * s, m + 12 * s, epsabs=0, epsrel=1e-13, limit=500)
        assert O.exact_loglik(b.data, th) == pytest.approx(math.log(ref), abs=1e-9)
        b = O.tiny("items", items=1, theta=SETTINGS[setting], seed=seed)
        k = b.readings[0][2][0]
        p = special.ndtr(m / math.sqrt(1 + s ** 2))
        assert O.exact_loglik(b.data, b.theta) == pytest.approx(math.log(p if k else 1 - p), abs=1e-10)


@pytest.mark.parametrize("seed", range(6))
def test_one_pair_is_a_normal_interval_of_the_difference(seed):
    b = O.tiny("pairs", runs=2, pair_calls=1, theta={"tie_width": 0.4 + 0.3 * (seed % 2)}, seed=seed)
    th = b.theta
    ra, rb, j, o, verdict = b.pairs[0]
    var_w = 2 * th["sd_run"][0] ** 2 + 2 * th["tau"][j, 0] ** 2      # the shared input cancels
    s = math.sqrt(var_w + th["sigma_pair"][j, 0] ** 2)
    m = th["mu"][ra, 0] - th["mu"][rb, 0] + th["position"][j] * o
    tie = th["tie_width"][j]
    lo, hi = {0: (tie, INF), 1: (-tie, tie), 2: (-INF, -tie)}[verdict]
    logp, ex, vx = truncated((lo - m) / s, (hi - m) / s)
    out = O.exact(b.data, th)
    assert out.loglik == pytest.approx(logp, abs=1e-10)
    cov = th["sd_run"][0] ** 2                                      # Cov(q_a, W)
    vq = th["sd_input"][0] ** 2 + th["sd_run"][0] ** 2
    assert out.posterior["q_run"]["mean"][ra, 0] == pytest.approx(th["mu"][ra, 0] + cov / s * ex, abs=1e-9)
    assert out.posterior["q_run"]["sd"][ra, 0] == pytest.approx(math.sqrt(vq - cov ** 2 / s ** 2 * (1 - vx)),
                                                                abs=1e-9)


def test_owens_t_formula_matches_scipy_in_two_dimensions():
    for h, k, rho in [(0.3, -0.7, 0.4), (-1.2, -0.4, 0.85), (1.1, 2.0, -0.3), (-0.5, 0.9, 0.95)]:
        cov = np.array([[1.0, rho], [rho, 1.0]])
        assert bvn_cdf(h, k, rho) == pytest.approx(float(stats.multivariate_normal.cdf([h, k], cov=cov)), abs=1e-10)


@pytest.mark.parametrize("pattern", [(1, 1), (1, 0), (0, 0)])
@pytest.mark.parametrize("setting", range(len(SETTINGS)))
def test_two_binary_calls_on_one_output_equal_owens_t(pattern, setting):
    b = O.tiny("binary", calls=2, theta=SETTINGS[setting], seed=setting)
    data = dict(b.data)
    data["g_y"] = np.array(pattern, dtype=float)
    th = b.theta
    vc = th["sd_input"][0] ** 2 + th["sd_run"][0] ** 2 + th["tau"][0, 0] ** 2
    total = vc + th["sigma"][0, 0] ** 2
    h = (th["mu"][0, 0] + th["bias"][0, 0]) / math.sqrt(total)
    signs = [1.0 if y else -1.0 for y in pattern]
    expected = bvn_cdf(signs[0] * h, signs[1] * h, signs[0] * signs[1] * vc / total)
    assert O.exact_loglik(data, th) == pytest.approx(math.log(expected), abs=1e-10)


def test_the_review_3_case_two_top_readings_of_three_levels():
    """Quality 0.5, judge-by-output sd 0.1, call sd 0.01, cut 0.75: P = 0.00541994 (lane S2's statement of
    review 3), and scipy's quad to 1e-12 relative. The Laplace value is 28% low here."""
    b = O.tiny("levels", calls=2, seed=0)
    data = dict(b.data)
    assert list(data["g_cell"]) == [0, 0]                           # the cell term is an explicit unknown
    data["g_lo"], data["g_hi"] = np.full(2, 0.75), np.full(2, 1.0e6)
    theta = {"mu": 0.5, "sd_input": 0.0, "sd_run": 0.0, "bias": 0.0, "tau": 0.1, "sigma": 0.01}
    out = O.exact(data, theta)
    assert math.exp(out.loglik) == pytest.approx(0.00541994, rel=1e-6)

    def f(t):
        return stats.norm.pdf(t, 0, 0.1) * special.ndtr((0.5 + t - 0.75) / 0.01) ** 2

    ref, _ = integrate.quad(f, -1, 1, points=[0.25], epsabs=0, epsrel=1e-13, limit=500)
    assert math.exp(out.loglik) == pytest.approx(ref, rel=1e-12)
    assert math.exp(out.laplace) == pytest.approx(0.75 * ref, rel=0.05)
    with pytest.raises(O.OracleNotConverged):                       # Gauss-Hermite alone says it is not done
        O.exact(data, theta, method="gh", tol=1e-9)


# ---------------------------------------------------------------------------------------------------
# larger designs: multivariate normal rectangles

RECTANGLES = [("binary", dict(calls=3)), ("binary", dict(judges=2, calls=2)),
              ("binary", dict(runs=2, calls=3, inputs=[0, 0])), ("binary", dict(runs=2, judges=2, calls=1)),
              ("levels", dict(calls=3)), ("levels", dict(runs=2, calls=2, inputs=[0, 0])),
              ("levels", dict(judges=2, calls=3)), ("items", dict(calls=2, items=3)),
              ("items", dict(runs=2, items=3, inputs=[0, 0])), ("items", dict(judges=2, calls=1, items=2)),
              ("pairs", dict(runs=2, pair_calls=3)), ("pairs", dict(runs=3, pair_calls=2)),
              ("pairs", dict(runs=2, judges=2, pair_calls=2)), ("levels", dict(runs=2, calls=1, pair_calls=2)),
              ("binary", dict(runs=2, calls=2, pair_calls=1)), ("linear", dict(runs=2, calls=2, pair_calls=2)),
              ("linear", dict(runs=3, judges=2, calls=1, pair_calls=1))]


@pytest.mark.parametrize("kind,options", RECTANGLES, ids=[f"{k}-{i}" for i, (k, _) in enumerate(RECTANGLES)])
def test_designs_equal_multivariate_normal_rectangles(kind, options):
    """Each reading, item and pair as an interval of its latent value: one rectangle probability of a
    normal vector, by Genz's method (two seeds; its error is about 1e-8 absolute here)."""
    for seed in (1, 2):
        b = O.tiny(kind, seed=seed, **options)
        out = O.exact(b.data, b.theta)
        assert all(block.error <= 1e-8 for block in out.blocks)
        refs = [reference_loglik(b, seed=s) for s in (0, 1)]
        spread = abs(refs[0] - refs[1])
        assert out.loglik == pytest.approx(refs[0], abs=max(2e-6, 5 * spread)), (out.loglik, refs)


def test_a_block_too_large_is_refused():
    b = O.tiny("items", runs=2, judges=2, calls=3, items=2, inputs=[0, 0], seed=1)
    with pytest.raises(O.OracleTooLarge):                           # 12 directions with explicit item noise
        O.exact(b.data, b.theta, items="explicit")
    assert O.exact(b.data, b.theta).items == "summed"                # 4 once each call's noise is summed out


@pytest.mark.parametrize("options", [dict(calls=3, items=4), dict(runs=2, calls=2, items=2),
                                     dict(runs=2, judges=2, calls=1, items=3, inputs=[0, 0]),
                                     dict(judges=2, calls=2, items=5)])
def test_item_noise_summed_inside_its_factor_equals_it_as_a_direction(options):
    b = O.tiny("items", seed=6, **options)
    explicit = O.exact(b.data, b.theta, items="explicit")
    summed = O.exact(b.data, b.theta, items="summed")
    assert explicit.items == "explicit" and summed.items == "summed"
    assert summed.loglik == pytest.approx(explicit.loglik, abs=1e-8)
    assert summed.laplace == pytest.approx(explicit.laplace, abs=1e-12)   # always the explicit layout's
    assert "e" in explicit.posterior and "e" not in summed.posterior
    for name in ("q_run", "cell"):
        np.testing.assert_allclose(summed.posterior[name]["mean"], explicit.posterior[name]["mean"], atol=1e-7)
        np.testing.assert_allclose(summed.posterior[name]["sd"], explicit.posterior[name]["sd"], atol=1e-7)


def test_a_summed_items_call_equals_scipy_quad_over_its_noise():
    """The factor alone: log of the integral over e of Phi(w + t e)^k Phi(-w - t e)^(n - k) phi(e), by
    scipy's quad on the log-shifted integrand (split at its peak, where the probit argument is 0 and 3 and
    8 widths 1 / t either side), and its derivatives by finite differences. The reported error covers the
    difference; a tighter inner tolerance moves the value by less than 1e-12."""
    for w, t, k, n in itertools.product((-4.0, -1.3, 0.0, 0.7, 2.5), (0.05, 0.4, 1.0, 2.5, 30.0, 1000.0),
                                        (0, 1, 2, 5), (2, 5, 7)):
        if k > n:
            continue

        def log_g(e):
            x = w + t * e
            return k * special.log_ndtr(x) + (n - k) * special.log_ndtr(-x) - 0.5 * e * e

        grid = np.linspace(-12, 12, 4801)
        top = float(np.max(log_g(grid)))
        peak = float(grid[np.argmax(log_g(grid))])
        cuts = [peak] + [(-w + x) / t for x in (-8.0, -3.0, 0.0, 3.0, 8.0)]
        edges = [-12.0] + sorted(c for c in set(cuts) if -12.0 < c < 12.0) + [12.0]
        ref = sum(integrate.quad(lambda e: math.exp(log_g(e) - top), lo, hi, epsabs=1e-15, epsrel=1e-12,
                                 limit=1000)[0] for lo, hi in zip(edges, edges[1:]))
        ref = math.log(ref) + top - 0.5 * O.LOG_2PI
        got, d1, d2, err = (float(v[0]) for v in O.noisy_binomial(np.array([w]), t, k, n, with_error=True))
        assert got == pytest.approx(ref, abs=1e-9), (w, t, k, n)
        assert abs(got - ref) <= err + 2e-12, (w, t, k, n, got - ref, err)
        assert err <= O.INNER_TOL
        assert float(O.noisy_binomial(np.array([w]), t, k, n, derivs=False, tol=1e-13)[0]) == pytest.approx(got, abs=1e-12)
        h = 1e-4 * max(1.0, t)
        up, down = (float(O.noisy_binomial(np.array([w + x]), t, k, n, derivs=False)[0]) for x in (h, -h))
        assert d1 == pytest.approx((up - down) / (2 * h), abs=1e-6)
        assert d2 == pytest.approx((up - 2 * got + down) / h ** 2, abs=1e-4)
        assert d2 < 0                                                # log-concave in w


@pytest.mark.parametrize("sigma", [5.0, 30.0, 100.0])
def test_a_summed_none_of_three_call_is_the_trivariate_normal_orthant(sigma):
    """One items call of three items, none passed, nothing else random: the probability is
    E Phi(-sigma Z)^3, the orthant probability of three normals of correlation rho = sigma^2 / (1 + sigma^2),
    which is 1/8 + 3 asin(rho) / (4 pi) exactly. The reported error is positive and covers the difference."""
    b = O.tiny("items", theta={"mu": 0.0, "sd_input": 0.0, "sd_run": 0.0, "tau": 0.0, "sigma": sigma}, seed=0)
    b.data["g_k"][:] = 0
    assert list(b.data["g_n"]) == [3]
    out = O.exact(b.data, b.theta, items="summed")
    rho = sigma * sigma / (1.0 + sigma * sigma)
    ref = math.log(0.125 + 3.0 * math.asin(rho) / (4.0 * math.pi))
    if sigma == 30.0:
        assert ref == pytest.approx(-0.7159016050648754, abs=1e-15)
    assert 0.0 < out.error <= 1e-8
    assert abs(out.loglik - ref) <= out.error + 1e-14


def test_large_item_call_noise_on_the_default_path_equals_nested_quad():
    """Two runs on one input, three calls each of three items, none passed; sd_input 0.4 and sigma 30,
    nothing else random. The default path sums the call noise out (the review's case). The reference
    integrates each call's noise by quad inside a quad over the input effect: -4.293624424218564."""
    sigma = 30.0
    b = O.tiny("items", runs=2, judges=1, calls=3, inputs=[0, 0], seed=0,
               theta={"mu": 0.0, "sd_input": 0.4, "sd_run": 0.0, "tau": 0.0, "sigma": sigma})
    b.data["g_k"][:] = 0
    out = O.exact(b.data, b.theta)
    assert out.items == "summed"

    def call(c):
        def f(e):
            return special.ndtr(-c - sigma * e) ** 3 * math.exp(-e * e / 2) / math.sqrt(2 * math.pi)
        cuts = sorted({-12.0, -1.0, 0.0, 1.0, 12.0, -c / sigma, (-c - 8) / sigma, (-c + 8) / sigma})
        return sum(integrate.quad(f, lo, hi, epsabs=1e-15, epsrel=1e-12, limit=300)[0]
                   for lo, hi in zip(cuts, cuts[1:]))

    ref, _ = integrate.quad(lambda z: call(0.4 * z) ** 6 * math.exp(-z * z / 2) / math.sqrt(2 * math.pi), -10, 10,
                            epsabs=1e-15, epsrel=1e-11, limit=300)
    assert math.log(ref) == pytest.approx(-4.293624424218564, abs=1e-10)
    assert out.loglik == pytest.approx(-4.293624424218564, abs=1e-10)
    assert out.error <= 1e-8 and abs(out.loglik - math.log(ref)) <= out.error + 1e-10


def _theta_times(b, factor, names):
    theta = dict(b.theta)
    for name in names:
        theta[name] = np.asarray(theta[name], dtype=float) * factor
    return theta


@pytest.mark.parametrize("s", [1.0, 1e-3, 1e-6, 1e-9, 1e-10, 1e-11])
def test_a_common_change_of_units_changes_nothing(s):
    """Binary, one run, one judge, the reading 1: with every scale of theta times s, the probability is
    Phi(0.3 / sqrt(1.57)) for every s, and q given it is a truncated-normal projection whose mean and sd
    are s times those at s = 1."""
    b = O.tiny("binary", theta={"mu": 0.3 * s, "sd_input": 0.4 * s, "sd_run": 0.5 * s, "tau": 0.4 * s, "sigma": s},
               seed=0)
    b.data["g_y"][:] = 1.0
    out = O.exact(b.data, b.theta)
    sd_x, cov = math.sqrt(1.57), 0.41
    a = 0.3 / sd_x
    lam = math.exp(special.log_ndtr(a) * -1.0) * math.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    assert out.loglik == pytest.approx(float(special.log_ndtr(a)), abs=1e-13)
    assert out.loglik == pytest.approx(-0.5198455314716747, abs=1e-13)
    assert len(out.blocks) == 1 and out.error <= 1e-8
    mean = 0.3 + cov / sd_x * lam
    var = cov - cov ** 2 / sd_x ** 2 * (a * lam + lam * lam)
    assert float(out.posterior["q_run"]["mean"][0, 0]) / s == pytest.approx(mean, rel=1e-10)
    assert float(out.posterior["q_run"]["sd"][0, 0]) / s == pytest.approx(math.sqrt(var), rel=1e-9)


SCALED = [("binary", dict(judges=2, calls=2), ("mu", "sd_input", "sd_run", "tau", "sigma", "bias")),
          ("pairs", dict(runs=3, judges=1, pair_calls=1),
           ("mu", "sd_input", "sd_run", "tau", "sigma_pair", "tie_width", "position", "bias")),
          ("pairs", dict(runs=2, judges=2, pair_calls=2),
           ("mu", "sd_input", "sd_run", "tau", "sigma_pair", "tie_width", "position", "bias"))]


@pytest.mark.parametrize("kind,options,names", SCALED, ids=[f"{k}-{i}" for i, (k, _, _) in enumerate(SCALED)])
def test_blocks_of_several_directions_do_not_depend_on_units(kind, options, names):
    """Binary and pair designs read signs and widths only: every scale of theta times s leaves the
    likelihood alone and multiplies the posterior means and sds of q, the cell terms and the halo
    factors' loadings by s (z_jt is in sd units, so it stays). Down to s = 1e-11."""
    b = O.tiny(kind, seed=3, **options)
    base = O.exact(b.data, b.theta)
    for s in (1e-3, 1e-7, 1e-10, 1e-11):
        out = O.exact(b.data, _theta_times(b, s, names))
        assert out.loglik == pytest.approx(base.loglik, abs=1e-10), s
        assert [block.dim for block in out.blocks] == [block.dim for block in base.blocks]
        for name in ("q_run", "cell"):
            np.testing.assert_allclose(out.posterior[name]["mean"] / s, base.posterior[name]["mean"], atol=1e-7)
            np.testing.assert_allclose(out.posterior[name]["sd"] / s, base.posterior[name]["sd"], atol=1e-7)
        np.testing.assert_allclose(out.posterior["z_jt"]["mean"], base.posterior["z_jt"]["mean"], atol=1e-7)


# ---------------------------------------------------------------------------------------------------
# quadrature convergence

CONVERGENCE = [("binary", dict(calls=1)), ("binary", dict(calls=3)), ("binary", dict(judges=2, calls=2)),
               ("levels", dict(calls=2)), ("levels", dict(judges=2, calls=3)), ("items", dict(calls=3, items=4)),
               ("items", dict(calls=1, items=5)), ("pairs", dict(runs=2, pair_calls=3)),
               ("pairs", dict(runs=3, judges=1, pair_calls=1))]


@pytest.mark.parametrize("kind,options", CONVERGENCE, ids=[f"{k}-{i}" for i, (k, _) in enumerate(CONVERGENCE)])
def test_the_quadrature_converges_and_the_two_rules_agree(kind, options):
    b = O.tiny(kind, seed=5, **options)
    out = O.exact(b.data, b.theta)
    tight = O.exact(b.data, b.theta, tol=1e-11)
    gh = O.exact(b.data, b.theta, method="gh")
    assert out.loglik == pytest.approx(tight.loglik, abs=1e-8)
    assert out.loglik == pytest.approx(gh.loglik, abs=2e-8)
    for block in gh.blocks:
        changes = [abs(block.history[i][1] - block.history[i - 1][1]) for i in range(1, len(block.history))]
        assert changes[-1] <= 1e-8 and changes[-2] <= 1e-8
        assert changes[0] > changes[-1]                              # it moved and settled
    for name in ("q_run", "cell", "e"):
        np.testing.assert_allclose(out.posterior[name]["mean"], gh.posterior[name]["mean"], atol=1e-6)
        np.testing.assert_allclose(out.posterior[name]["sd"], gh.posterior[name]["sd"], atol=1e-6)
    biggest = max(gh.blocks, key=lambda block: block.dim)
    fixed = O.exact(b.data, b.theta, method="gh", nodes=[max(n for n in O.NODES if n ** biggest.dim <= 300_000)])
    assert fixed.loglik == pytest.approx(gh.loglik, abs=1e-8)


def judges_times(b: O.Tiny, factor: float) -> O.Tiny:
    """b with its judge noise (tau and the call noise, or the pair noise) times `factor`."""
    names = ("tau", "sigma_pair") if b.kind == "pairs" else ("tau", "sigma")
    return dataclasses.replace(b, theta=_theta_times(b, factor, names))


def test_the_review_seed_8_case_with_sharp_judges_converges_by_the_nested_rule():
    """Binary, one run, two judges, two calls, tau and sigma times 0.03: Gauss-Hermite does not settle
    (and refuses rather than report), the nested rule converges to the review's value, which Genz's
    method brackets too."""
    b = judges_times(O.tiny("binary", runs=1, judges=2, calls=2, seed=8), 0.03)
    out = O.exact(b.data, b.theta)
    assert out.loglik == pytest.approx(-1.7767059692894638, abs=1e-10)
    assert 0.0 < out.error <= 1e-8
    assert [(block.method, block.sharp) for block in out.blocks if block.dim > 1] == [("nested", 1)]
    refs = [reference_loglik(b, seed=s) for s in (0, 1)]
    assert out.loglik == pytest.approx(refs[0], abs=max(2e-6, 5 * abs(refs[0] - refs[1])))
    with pytest.raises(O.OracleNotConverged):
        O.exact(b.data, b.theta, method="gh")


NESTED = [("binary", dict(runs=1, judges=2, calls=2), 8), ("binary", dict(runs=2, judges=2, calls=2), 222),
          ("pairs", dict(runs=2, judges=2, pair_calls=1), 221), ("pairs", dict(runs=3, judges=1, pair_calls=1), 311)]


@pytest.mark.parametrize("factor", [1.0, 0.3])
@pytest.mark.parametrize("kind,options,seed", NESTED, ids=[f"{k}-{i}" for i, (k, _, _) in enumerate(NESTED)])
def test_the_nested_rule_equals_gauss_hermite_where_both_settle(kind, options, seed, factor):
    """With the default judge noise and at 0.3 of it, both rules converge: the log-likelihood and every
    posterior mean and sd agree."""
    b = judges_times(O.tiny(kind, seed=seed, **options), factor)
    gh = O.exact(b.data, b.theta, method="gh")
    nested = O.exact(b.data, b.theta, method="nested")
    assert all(block.method == "nested" for block in nested.blocks if block.dim > 1)
    assert nested.loglik == pytest.approx(gh.loglik, abs=2e-8)
    for name in gh.posterior:
        for key in ("mean", "sd"):
            np.testing.assert_allclose(nested.posterior[name][key], gh.posterior[name][key], atol=1e-6)


# ---------------------------------------------------------------------------------------------------
# the tolerance contract: tol bounds the whole reported error, or the oracle refuses

def one_binary(mu: float, sd_input: float) -> O.Tiny:
    """One binary reading of 1 with no run, judge or call terms but a unit call noise: log Phi(mu /
    sqrt(1 + sd_input^2)) exactly."""
    b = O.tiny("binary", theta={"mu": mu, "sd_input": sd_input, "sd_run": 0.0, "tau": 0.0, "sigma": 1.0}, seed=0)
    b.data["g_y"][:] = 1.0
    return b


CONTRACT = [(0.3, 1.0, 1e-12, "pass"), (0.3, 1.0, 1e-13, "pass"), (0.3, 1.0, 1e-14, "refuse"),
            (-100.0, 5e-8, 1e-12, "refuse"), (-100.0, 5e-8, 1e-10, "pass"),
            (-10000.0, 5e-8, 1e-8, "refuse"), (-10000.0, 5e-8, 1e-6, "pass"), (-10000.0, 5e-8, 1e-5, "pass")]


@pytest.mark.parametrize("mu,sd_input,tol,outcome", CONTRACT)
def test_every_answer_meets_the_requested_tol_or_the_oracle_refuses(mu, sd_input, tol, outcome):
    """The review's cases (mu 0.3 at tol 1e-12; mu -100 with sd_input 5e-8 at 1e-12; mu -10000 at the
    default tol) and their neighbours. A value of size 5005 or 5e7 cannot be certified below its
    roundoff (16 ulps a unit of the log terms), so those requests are refused, by `exact_loglik` too;
    every answer has its whole error within tol and that error covers the closed form. At mu -10000 and
    tol 1e-5 the input direction is dropped (what it could move fits its allowance) and the error
    still covers the difference it leaves."""
    b = one_binary(mu, sd_input)
    ref = float(special.log_ndtr(mu / math.sqrt(1.0 + sd_input ** 2)))
    if outcome == "refuse":
        with pytest.raises(O.OracleNotConverged):
            O.exact(b.data, b.theta, tol=tol)
        with pytest.raises(O.OracleNotConverged):
            O.exact_loglik(b.data, b.theta, tol=tol)
        return
    out = O.exact(b.data, b.theta, tol=tol)
    assert 0.0 < out.error <= tol
    assert abs(out.loglik - ref) <= out.error
    assert sum(block.error for block in out.blocks) <= out.error


@pytest.mark.parametrize("tol", [1e-6, 1e-8, 1e-10, 1e-11])
@pytest.mark.parametrize("kind,options", [("binary", dict(runs=2, judges=2, calls=2)), ("levels", dict(runs=3, judges=1, calls=2)),
                                          ("items", dict(runs=2, judges=1, calls=2, items=3)),
                                          ("pairs", dict(runs=3, judges=1, pair_calls=2))])
def test_tol_is_global_over_the_blocks(kind, options, tol):
    """tol bounds the sum over blocks, flat parts, cut links and roundoff, which is `error`; each answer
    is within its error of one at tol 1e-12."""
    b = O.tiny(kind, seed=7, **options)
    out = O.exact(b.data, b.theta, tol=tol)
    assert out.error <= tol
    assert sum(block.error for block in out.blocks) <= out.error
    tight = O.exact(b.data, b.theta, tol=1e-12)
    assert abs(out.loglik - tight.loglik) <= out.error + tight.error


def test_fixed_nodes_are_a_diagnostic_that_does_not_refuse():
    """`nodes=` fixes the Gauss-Hermite rule with no convergence loop and no refusal: an explicit
    diagnostic. The same block at that tol without it refuses."""
    b = judges_times(O.tiny("binary", runs=1, judges=2, calls=2, seed=8), 0.03)
    out = O.exact(b.data, b.theta, method="gh", nodes=[3], tol=1e-14)
    assert np.isfinite(out.loglik)
    with pytest.raises(O.OracleNotConverged):
        O.exact(b.data, b.theta, method="gh", tol=1e-14)


def test_the_error_carries_a_roundoff_allowance():
    """Pairs, 2 runs, 2 pair calls, tau and sigma_pair times 0.03: log p is -289. The review's 70-digit
    Decimal integral of its one-dimensional contrast gives -289.1146078789756178092186; the double
    result is about one ulp off, and the reported error covers it."""
    b = judges_times(O.tiny("pairs", runs=2, pair_calls=2, seed=212), 0.03)
    out = O.exact(b.data, b.theta)
    assert abs(out.loglik - (-289.1146078789756178092186)) <= out.error <= 1e-8
    assert out.error >= 16 * np.finfo(float).eps * 289


def _orthant_pair(rho: float) -> float:
    return math.log(0.25 + math.asin(rho) / (2 * math.pi))


SOV_CASES = [
    # the review's case: a correlated pair at 0 and an independent tail at 12
    (np.array([[1, .8, 0], [.8, 1, 0], [0, 0, 1.0]]), [0, 0, 12], math.log(special.ndtr(-12.0)) + _orthant_pair(0.8)),
    # two correlated pairs and a tail at 30
    (linalg.block_diag([[1, .8], [.8, 1]], [[1, -.6], [-.6, 1]], [[1.0]]), [0, 0, 0, 0, 30],
     _orthant_pair(0.8) + _orthant_pair(-0.6) + float(special.log_ndtr(-30.0))),
    # six coordinates of correlation 1/2 (P all > 0 = 1/7) and a tail at 20
    (linalg.block_diag(0.5 * np.eye(6) + 0.5, [[1.0]]), [0] * 6 + [20], math.log(1 / 7) + float(special.log_ndtr(-20.0))),
]


@pytest.mark.parametrize("case", range(len(SOV_CASES)))
def test_the_log_space_sov_reference_equals_closed_forms(case):
    """The sharp grid's reference for 3 to 9 latents against orthant closed forms (1/4 + asin(rho) / 2 pi
    for a pair, 1/(n + 1) for n coordinates of correlation 1/2) with independent tails, down to e^-460;
    within 5 replicate standard errors or 1e-9."""
    cov, lower, exact = SOV_CASES[case]
    lower = np.asarray(lower, dtype=float)
    est, se = log_rect_sov(np.zeros(len(lower)), cov, lower, np.full(len(lower), np.inf), seed=3)
    assert est == pytest.approx(exact, abs=max(1e-9, 5 * se))


@pytest.mark.parametrize("kind,calls", [("binary", 2), ("binary", 3), ("levels", 2), ("levels", 3)])
def test_posterior_moments_in_one_direction_equal_scipy_quad(kind, calls):
    """One output read `calls` times: the readings see only c = q + bias + part. E and Var of c by quad;
    q and the cell term follow from normal conditioning on c."""
    theta = {"tau": 0.15, "sigma": 0.06} if kind == "levels" else {"tau": 0.7, "sigma": 0.4}
    b = O.tiny(kind, calls=calls, theta=theta, seed=calls)
    th = b.theta
    mean, L, lo, hi, _, _, _, _ = latent_model(b)
    vq = float(th["sd_input"][0] ** 2 + th["sd_run"][0] ** 2)
    vt = float(th["tau"][0, 0] ** 2)
    s = float(th["sigma"][0, 0])
    mc, vc = mean[0], vq + vt

    def weight(c):
        lik = np.prod(stats.norm.cdf((hi - c) / s) - stats.norm.cdf((lo - c) / s))
        return stats.norm.pdf(c, mc, math.sqrt(vc)) * lik

    span = (mc - 10 * math.sqrt(vc), mc + 10 * math.sqrt(vc))
    z0, z1, z2 = (integrate.quad(lambda c, p=p: c ** p * weight(c), *span, epsabs=0, epsrel=1e-13, limit=500)[0]
                  for p in (0, 1, 2))
    ec, var_c = z1 / z0, z2 / z0 - (z1 / z0) ** 2
    out = O.exact(b.data, th)
    assert out.loglik == pytest.approx(math.log(z0), abs=1e-9)
    for name, v, centre in (("q_run", vq, th["mu"][0, 0]), ("cell", vt, 0.0)):
        got = out.posterior[name]
        beta = v / vc
        assert got["mean"].ravel()[0] == pytest.approx(centre + beta * (ec - mc), abs=1e-8)
        assert got["sd"].ravel()[0] == pytest.approx(math.sqrt(v - beta * v + beta ** 2 * var_c), abs=1e-8)


# ---------------------------------------------------------------------------------------------------
# the Laplace value

def laplace_reference(b: O.Tiny) -> float:
    """The Laplace approximation over the random effects the layout keeps (inputs, runs, the term of every
    cell with 2 or more readings or a pair, the call noise of an items call of 2 or more items), found
    with scipy's optimiser and a finite-difference Hessian, from the rows written."""
    th = b.theta
    inputs: dict = {}
    run_input = [inputs.setdefault(run["input"], len(inputs)) for run in b.study.runs]
    R = len(b.study.runs)
    per_cell: dict = {}
    for r, j, _ in b.readings:
        per_cell[(j, r)] = per_cell.get((j, r), 0) + 1
    explicit = sorted({key for key, n in per_cell.items() if n >= 2}
                      | {(j, r) for ra, rb, j, _, _ in b.pairs for r in (ra, rb)})
    items = [n for n, (_, _, value) in enumerate(b.readings) if b.kind == "items" and value[1] >= 2]
    names = [("v", i) for i in range(len(inputs))] + [("u", r) for r in range(R)] + [("t", key) for key in explicit] \
        + [("e", n) for n in items]
    at = {name: k for k, name in enumerate(names)}
    sd_in, sd_run = float(th["sd_input"][0]), float(th["sd_run"][0])
    levels = len(b.rubric["components"][0]["scale"].get("levels", [])) or 2
    cuts = np.concatenate([[-INF], (np.arange(levels - 1) + 0.5) / (levels - 1), [INF]])

    def log_int(lo, hi, m, s):
        if lo == -INF:
            return float(stats.norm.logcdf((hi - m) / s))
        if hi == INF:
            return float(stats.norm.logsf((lo - m) / s))
        return float(np.log(stats.norm.cdf((hi - m) / s) - stats.norm.cdf((lo - m) / s)))

    def joint(z):
        out = float(stats.norm.logpdf(z).sum())

        def view(r, j):
            value = th["mu"][r, 0] + sd_in * z[at[("v", run_input[r])]] + sd_run * z[at[("u", r)]]
            return value + (th["tau"][j, 0] * z[at[("t", (j, r))]] if ("t", (j, r)) in at else 0.0)

        for n, (r, j, value) in enumerate(b.readings):
            eta = view(r, j) + th["bias"][j, 0]
            var = th["sigma"][j, 0] ** 2 + (0 if ("t", (j, r)) in at else th["tau"][j, 0] ** 2)
            if b.kind == "binary":
                out += log_int(*((0.0, INF) if value else (-INF, 0.0)), eta, math.sqrt(var))
            elif b.kind == "levels":
                out += log_int(cuts[value], cuts[value + 1], eta, math.sqrt(var))
            else:
                k, nn = value
                x = eta + math.sqrt(var) * z[at[("e", n)]] if ("e", n) in at else eta / math.sqrt(1 + var)
                out += float(stats.binom.logpmf(k, nn, special.ndtr(x)))
        for ra, rb, j, o, verdict in b.pairs:
            w = view(ra, j) - view(rb, j) + th["position"][j] * o
            tie = th["tie_width"][j]
            out += log_int(*{0: (tie, INF), 1: (-tie, tie), 2: (-INF, -tie)}[verdict], w, th["sigma_pair"][j, 0])
        return out

    p = len(names)

    def grad(z, h=1e-6):
        return np.array([(joint(z + h * e) - joint(z - h * e)) / (2 * h) for e in np.eye(p)])

    def hessian(z, h=1e-4):
        H = np.zeros((p, p))
        for i, j in itertools.product(range(p), repeat=2):
            if j < i:
                continue
            ei, ej = h * np.eye(p)[i], h * np.eye(p)[j]
            H[i, j] = H[j, i] = (joint(z + ei + ej) - joint(z + ei - ej) - joint(z - ei + ej) + joint(z - ei - ej)) / (4 * h * h)
        return H

    z = optimize.minimize(lambda x: -joint(x), np.zeros(p), jac=lambda x: -grad(x), method="BFGS",
                          options={"gtol": 1e-9}).x
    for _ in range(3):                                               # polish with Newton steps
        z = z - np.linalg.solve(hessian(z), grad(z))
    sign, logdet = np.linalg.slogdet(-hessian(z))
    assert sign > 0
    return joint(z) + 0.5 * p * O.LOG_2PI - 0.5 * logdet          # binom.logpmf keeps C(n, k)


LAPLACE = [("binary", dict(calls=1)), ("binary", dict(calls=3)), ("binary", dict(judges=2, calls=2)),
           ("levels", dict(runs=2, calls=2)), ("items", dict(calls=2, items=3)), ("items", dict(items=1)),
           ("pairs", dict(runs=2, pair_calls=3)), ("pairs", dict(runs=3, judges=2, pair_calls=1)),
           ("levels", dict(runs=2, calls=1, pair_calls=2))]


@pytest.mark.parametrize("kind,options", LAPLACE, ids=[f"{k}-{i}" for i, (k, _) in enumerate(LAPLACE)])
def test_the_laplace_value_is_the_laplace_approximation_over_the_layouts_random_effects(kind, options):
    b = O.tiny(kind, seed=9, **options)
    assert O.laplace_loglik(b.data, b.theta) == pytest.approx(laplace_reference(b), abs=2e-6)


# ---------------------------------------------------------------------------------------------------
# the factors the oracle writes for itself

def test_the_oracle_factors_match_scipy_and_their_derivatives_match_finite_differences():
    w = np.linspace(-3.0, 3.0, 13)
    cases = [(O.INTERVAL, 0.0, INF, 0.7, 0, 0), (O.INTERVAL, -INF, 0.2, 1.3, 0, 0), (O.INTERVAL, -0.4, 0.5, 0.3, 0, 0),
             (O.BINOMIAL, 0, 0, 1.0, 2, 5), (O.BINOMIAL, 0, 0, 1.6, 0, 3), (O.GAUSS, 0.3, 0, 0.4, 0, 0)]
    for kind, lo, hi, s, k, n in cases:
        args = [np.full_like(w, x, dtype=float) for x in (kind, lo, hi, s, k, n)]
        args[0] = args[0].astype(np.int64)
        lp, d1, d2 = O.factor_values(*args, w)
        if kind == O.INTERVAL:
            a, b = (lo - w) / s, (hi - w) / s                        # on the side of 0 that keeps its digits
            with np.errstate(divide="ignore"):
                ref = np.where(a > 0, np.log(stats.norm.sf(a) - stats.norm.sf(b)),
                               np.log(stats.norm.cdf(b) - stats.norm.cdf(a)))
        elif kind == O.BINOMIAL:
            ref = stats.binom.logpmf(k, n, special.ndtr(w / s)) - math.log(math.comb(n, k))
        else:
            ref = stats.norm.logpdf(lo, w, s)
        np.testing.assert_allclose(lp, ref, atol=1e-12)
        h = 1e-5
        up = O.factor_values(*args, w + h, derivs=False)
        down = O.factor_values(*args, w - h, derivs=False)
        np.testing.assert_allclose(d1, (up - down) / (2 * h), atol=1e-6)
        np.testing.assert_allclose(d2, (up - 2 * lp + down) / h ** 2, atol=1e-3)
    # far in a tail the interval keeps its digits
    assert float(O.log_interval(np.array([30.0]), np.array([INF]))[0]) == pytest.approx(
        float(stats.norm.logsf(30.0)), rel=1e-12)


# ---------------------------------------------------------------------------------------------------
# the full grid of tiny designs (slow)

FULL = [(kind, runs, judges, calls) for kind in ("binary", "levels", "items")
        for runs, judges, calls in itertools.product((1, 2, 3), (1, 2), (1, 2, 3))] + \
       [("pairs", runs, judges, calls) for runs, judges, calls in itertools.product((2, 3), (1, 2), (1, 2, 3))]


@pytest.mark.slow
@pytest.mark.parametrize("kind,runs,judges,calls", FULL)
def test_every_tiny_design_converges_and_small_ones_equal_genz(kind, runs, judges, calls):
    options = dict(runs=runs, judges=judges, items=2)
    options.update(pair_calls=calls) if kind == "pairs" else options.update(calls=calls)
    b = O.tiny(kind, seed=runs * 100 + judges * 10 + calls, **options)
    out = O.exact(b.data, b.theta)
    assert all(block.error <= 1e-8 for block in out.blocks)
    tight = O.exact(b.data, b.theta, tol=1e-10)
    assert out.loglik == pytest.approx(tight.loglik, abs=1e-8)
    mean, L, *_ = latent_model(b)
    if len(mean) <= 9:
        refs = [reference_loglik(b, seed=s) for s in (0, 1)]
        assert out.loglik == pytest.approx(refs[0], abs=max(2e-6, 5 * abs(refs[0] - refs[1])))


SHARP = [(factor, *design) for factor in (0.1, 0.03) for design in FULL]
COSTLY = ("pairs", 3, 2)          # two sharp and two smooth coordinates: about 6 CPU-minutes a design


@pytest.mark.slow
@pytest.mark.parametrize("factor,kind,runs,judges,calls", SHARP)
def test_every_tiny_design_with_sharp_judges_converges_and_equals_an_independent_value(factor, kind, runs, judges,
                                                                                         calls):
    """The full grid with the judge noise times 0.1 and 0.03, where Laplace is far off and plain
    Gauss-Hermite does not settle: every block within 1e-8, and a run at tol 1e-10 agrees. The
    probabilities go down to e^-460, where scipy's Genz returns 0 or loses digits (-32.7856 for
    -32.7780), so the independent values are this file's own: up to 2 latents an interval or a quad of
    one (to 1e-8); up to 9, separation of variables in log space (to 5 standard errors).
    Pairs of 3 runs and 2 judges take minutes each and run only with LOOPMATH_EVAL_COSTLY=1 as well."""
    if (kind, runs, judges) == COSTLY and os.environ.get("LOOPMATH_EVAL_COSTLY") != "1":
        pytest.skip("two sharp and two smooth coordinates: minutes a design; set LOOPMATH_EVAL_COSTLY=1")
    options = dict(runs=runs, judges=judges, items=2)
    options.update(pair_calls=calls) if kind == "pairs" else options.update(calls=calls)
    b = judges_times(O.tiny(kind, seed=runs * 100 + judges * 10 + calls, **options), factor)
    out = O.exact(b.data, b.theta)
    assert all(block.error <= 1e-8 for block in out.blocks)
    tight = O.exact(b.data, b.theta, tol=1e-10)
    assert out.loglik == pytest.approx(tight.loglik, abs=1e-8)
    mean, L, lo, hi, y, const, _, _ = latent_model(b)
    assert np.isnan(y).all()                                      # no Gaussian reading on these kinds
    if len(mean) <= 2:
        assert out.loglik == pytest.approx(const + log_rect_small(mean, L @ L.T, lo, hi), abs=1e-8)
    elif len(mean) <= 9:
        ref, se = log_rect_sov(mean, L @ L.T, lo, hi)
        assert out.loglik == pytest.approx(const + ref, abs=max(2e-6, 5 * se))
