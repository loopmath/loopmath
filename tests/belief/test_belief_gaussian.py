"""Head engines: exact evidence gradients, Laplace gradients, rank-one updates (spec 04 section 4)."""

from __future__ import annotations

import numpy as np
import pytest
import simdata

from loopmath.belief import fit as F
from loopmath.belief.forest import Forest
from loopmath.belief.gaussian import GaussianHead, LogisticHead, gaussian_rank_one, logistic_linearized


@pytest.fixture(scope="module")
def rows():
    docs, _ = simdata.simulate(250, seed=5, q=0.98)
    heads, *_ = F.collect_rows(docs)
    return heads


def _numeric(f, theta, eps=1e-5):
    out = np.zeros(len(theta))
    for i in range(len(theta)):
        e = np.zeros(len(theta))
        e[i] = eps
        out[i] = (f(theta + e)[0] - f(theta - e)[0]) / (2 * eps)
    return out


def _head(hr, q_one=False):
    X, ids = F.build_matrix(hr.rows, Forest())
    prior = F.make_prior(ids, hr.kind)
    if hr.engine == "gaussian":
        return GaussianHead(X, np.array(hr.y), np.array(hr.w), prior), prior
    q = np.ones(len(hr.w)) if q_one else np.array(hr.w)
    return LogisticHead(X, np.array(hr.y), q, prior), prior


def test_gaussian_evidence_gradient_matches_finite_differences(rows):
    h, prior = _head(rows["cost"])
    theta = np.concatenate([np.log(prior.default_phi) + 0.2, [np.log(h.sigma0)]])
    _, g = h.objective(theta)
    np.testing.assert_allclose(g, _numeric(h.objective, theta), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("head,q_one", [("gate", True), ("success", False)])
def test_laplace_gradient_matches_finite_differences(rows, head, q_one):
    h, prior = _head(rows[head], q_one)
    theta = np.log(prior.default_phi)
    _, g = h.objective(theta)
    np.testing.assert_allclose(g, _numeric(h.objective, theta, 1e-4), rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("q", [0.7, 0.98])
def test_logistic_laplace_uses_the_observed_posterior_hessian(q):
    """D68: with q < 1 the Fisher and observed Hessians differ; the covariance and evidence use the observed one."""
    from scipy import sparse
    from scipy.special import expit

    from loopmath.belief.gaussian import Prior

    n, var = 20, 25.0
    h = LogisticHead(sparse.csr_matrix(np.ones((n, 1))), np.ones(n), np.full(n, q),
                     Prior(np.array([-1]), np.array([var]), [], np.array([])))
    res = h.fit(eb=False)
    x, eps = res.mean[0], 1e-3

    def negpost(v):
        return -n * np.log((1 - q) + (2 * q - 1) * expit(v)) + v * v / (2 * var)

    precision = (negpost(x + eps) - 2 * negpost(x) + negpost(x - eps)) / eps ** 2
    assert res.info["hessian"] == "observed"
    assert res.cov()[0, 0] == pytest.approx(1 / precision, rel=1e-4)
    # Laplace log evidence: log p(z | b*) + log N(b*; 0, var) + (1/2) log(2 pi / precision); the 2 pi terms cancel
    laplace = -negpost(x) - 0.5 * np.log(var) - 0.5 * np.log(precision)
    assert res.log_evidence == pytest.approx(laplace, abs=1e-5)


def test_a_non_positive_definite_observed_hessian_falls_back_to_fisher(monkeypatch):
    from scipy import sparse
    from scipy.special import expit

    from loopmath.belief.gaussian import Prior

    n, var, q = 20, 25.0, 0.7
    h = LogisticHead(sparse.csr_matrix(np.ones((n, 1))), np.ones(n), np.full(n, q),
                     Prior(np.array([-1]), np.array([var]), [], np.array([])))
    monkeypatch.setattr(h, "_observed", lambda eta: np.full(len(eta), -1.0))  # precision 1/25 - 20
    res = h.fit(eb=False)
    s = expit(res.mean[0])
    pi, slope = (1 - q) + (2 * q - 1) * s, (2 * q - 1) * s * (1 - s)
    assert res.info["hessian"] == "fisher"
    assert res.cov()[0, 0] == pytest.approx(1 / (n * slope ** 2 / (pi * (1 - pi)) + 1 / var), rel=1e-6)
    assert np.isfinite(h.objective(np.array([]))[0]) and h.fisher_fallbacks == 1


def test_empirical_bayes_recovers_the_noise_scale(rows):
    h, _ = _head(rows["cost"])
    res = h.fit()
    assert res.info["converged"]
    assert abs(res.sigma - simdata.SIGMA["cost"]) < 0.05
    np.testing.assert_allclose(res.cov(), np.linalg.inv(h.A / res.sigma ** 2 + np.diag(1 / h.prior.lam(res.phi))),
                               rtol=1e-6, atol=1e-10)


def test_gaussian_rank_one_equals_a_refit_with_the_row():
    rng = np.random.default_rng(0)
    from scipy import sparse

    from loopmath.belief.gaussian import Prior

    p, n = 6, 30
    X = rng.standard_normal((n, p))
    y = X @ rng.standard_normal(p) + 0.3 * rng.standard_normal(n)
    prior = Prior(np.zeros(p, dtype=int), np.zeros(p), ["g"], np.array([1.0]))
    base = GaussianHead(sparse.csr_matrix(X), y, np.ones(n), prior).fit(eb=False, phi=np.array([1.0]), sigma=0.3)
    x_new, y_new, tau2 = rng.standard_normal(p), 1.7, 0.3 ** 2 / 2.0
    mean, U = gaussian_rank_one(base.mean, base.cov_factor, x_new, y_new, tau2)
    ref = GaussianHead(sparse.csr_matrix(np.vstack([X, x_new])), np.append(y, y_new), np.append(np.ones(n), 2.0),
                       prior).fit(eb=False, phi=np.array([1.0]), sigma=0.3)
    np.testing.assert_allclose(mean, ref.mean, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(U @ U.T, ref.cov(), rtol=1e-8, atol=1e-12)


def test_logistic_linearized_adds_the_fisher_information():
    rng = np.random.default_rng(1)
    p = 5
    A = rng.standard_normal((p, p))
    cov = A @ A.T + np.eye(p)
    U = np.linalg.cholesky(cov)
    mean, x = 0.3 * rng.standard_normal(p), rng.standard_normal(p)
    for q in (1.0, 0.8):
        m1, U1 = logistic_linearized(mean, U, x, 1.0, q)
        m0, _ = logistic_linearized(mean, U, x, 0.0, q)
        s = 1 / (1 + np.exp(-x @ mean))
        a = 2 * q - 1
        pi = (1 - q) + a * s
        info = (a * s * (1 - s)) ** 2 / (pi * (1 - pi))
        np.testing.assert_allclose(np.linalg.inv(U1 @ U1.T), np.linalg.inv(cov) + info * np.outer(x, x), rtol=1e-8)
        assert x @ m1 > x @ mean > x @ m0  # a pass moves the predictor up, a fail down
