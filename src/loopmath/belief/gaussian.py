"""Closed-form Gaussian heads, Laplace logistic heads, empirical Bayes scales.

Spec 04 sections 2 and 4. Every head has effects `b` with prior `N(0, Lambda)`, where
`Lambda` is diagonal: `phi_level^2` for forest nodes, a fixed wide variance for fixed effects.

- Gaussian heads (cost, tokens, scores): `y ~ N(Xb, sigma^2 / w)`. The posterior is closed
  form in parameter space (`P = X'WX / sigma^2 + Lambda^-1`, Cholesky). The log marginal
  likelihood is exact through the Woodbury and determinant lemmas, with an analytic gradient
  in `log phi` per level and `log sigma` (Fisher's identity).
- Logistic heads (success, gate): `P(z = 1) = q s(eta) + (1 - q)(1 - s(eta))` (paper Remark
  4.10; `q = 1` for gates). The mode is found by Newton iterations with the expected
  (Fisher) Hessian, which equals the Newton Hessian when `q = 1` and stays positive definite
  when `q < 1`. The Laplace covariance and evidence use the observed Hessian of the log
  posterior at the mode (spec 04, D68); if that is not positive definite the head falls back
  to the Fisher precision, and the fit's diagnostics say so (`info["hessian"]`).
- Scales: L-BFGS-B over `log phi` (and `log sigma`) with a log-normal hyperprior on each
  `phi` (sd 0.7 around the spec defaults) so that sparse levels do not collapse to zero.
- Prior factors (benchmark priors) are Gaussian pseudo-observations `F b ~ N(f, V)` with a
  fixed variance, added to the precision and the shift of either kind of head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, optimize, sparse
from scipy.special import expit

from .forest import HYPER_SD

N_DRAWS = 400
LOG_PHI_BOUNDS = (math.log(1e-3), math.log(20.0))
SIGMA_PRIOR_SD = 1.0  # weak log-normal prior on sigma, centred at the starting value


@dataclass
class Factors:
    """Gaussian pseudo-observations on the effects: F b ~ N(f, diag(v))."""

    F: sparse.csr_matrix
    f: np.ndarray
    v: np.ndarray

    @property
    def A(self) -> np.ndarray:
        Fv = self.F.multiply(1.0 / self.v[:, None]).tocsr()
        return np.asarray((self.F.T @ Fv).todense())

    @property
    def c(self) -> np.ndarray:
        return np.asarray(self.F.T @ (self.f / self.v)).ravel()


@dataclass
class Prior:
    """Diagonal prior: per node a scale group (index into `groups`) or -1 with a fixed variance."""

    group_idx: np.ndarray
    fixed_var: np.ndarray
    groups: list[str]
    default_phi: np.ndarray

    def lam(self, phi: np.ndarray) -> np.ndarray:
        out = self.fixed_var.copy()
        grouped = self.group_idx >= 0
        out[grouped] = phi[self.group_idx[grouped]] ** 2
        return out


@dataclass
class HeadFit:
    kind: str  # "gaussian" | "logistic"
    mean: np.ndarray
    cov_factor: np.ndarray  # U with U U' = posterior covariance (upper triangular)
    phi: np.ndarray
    sigma: float = 1.0
    log_evidence: float = float("nan")
    iterations: int = 0
    info: dict = field(default_factory=dict)
    precision: sparse.csr_matrix | None = None  # the matrix whose Cholesky factor gives U (D94)

    def draws(self, seed: int, n: int = N_DRAWS) -> np.ndarray:
        z = np.random.default_rng(seed).standard_normal((len(self.mean), n))
        return self.mean[:, None] + self.cov_factor @ z

    def cov(self) -> np.ndarray:
        return self.cov_factor @ self.cov_factor.T


def _cho(P: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor, with a small jitter if P is not numerically positive definite."""
    jitter = 0.0
    for _ in range(6):
        try:
            # np.tril: the factor's other triangle is not guaranteed to be zero
            return np.tril(linalg.cholesky(P + jitter * np.eye(len(P)), lower=True, check_finite=False))
        except linalg.LinAlgError:
            jitter = max(1e-10, jitter * 10 or 1e-10 * float(np.mean(np.diag(P))))
    raise linalg.LinAlgError("posterior precision is not positive definite")


def cov_factor(R: np.ndarray) -> np.ndarray:
    """U = R^-T, upper triangular with U U' = P^-1 for the lower Cholesky factor R of P."""
    return np.ascontiguousarray(_inv_lower(R).T)


def _inv_lower(R: np.ndarray) -> np.ndarray:
    Rinv, info = linalg.lapack.dtrtri(R, lower=1)
    if info != 0:
        Rinv = linalg.solve_triangular(R, np.eye(len(R)), lower=True)
    return np.tril(Rinv)


# ---------------------------------------------------------------- Gaussian heads

class GaussianHead:
    """y ~ N(Xb, sigma^2 / w), b ~ N(0, Lambda(phi)), optional prior factors."""

    def __init__(self, X: sparse.csr_matrix, y: np.ndarray, w: np.ndarray, prior: Prior,
                 factors: Factors | None = None):
        self.X = X.tocsr()
        self.y = np.asarray(y, float)
        self.w = np.asarray(w, float)
        self.prior = prior
        self.n, self.p = self.X.shape
        Xw = self.X.multiply(self.w[:, None]).tocsr()
        self.A = np.asarray((self.X.T @ Xw).todense())
        self.c = np.asarray(self.X.T @ (self.w * self.y)).ravel()
        self.yWy = float(np.sum(self.w * self.y * self.y))
        self.slw = float(np.sum(np.log(self.w)))
        self.factors = factors
        self.Af = factors.A if factors is not None else np.zeros((self.p, self.p))
        self.cf = factors.c if factors is not None else np.zeros(self.p)
        self.fVf = float(np.sum(factors.f ** 2 / factors.v)) if factors is not None else 0.0
        self.slv = float(np.sum(np.log(factors.v))) if factors is not None else 0.0
        self.sigma0 = self._sigma_start()

    def _sigma_start(self) -> float:
        if self.n < 2:
            return 1.0
        mu = np.sum(self.w * self.y) / np.sum(self.w)
        sd = math.sqrt(float(np.sum(self.w * (self.y - mu) ** 2) / np.sum(self.w)))
        return max(0.05, min(5.0, sd if sd > 0 else 1.0))

    def _precision(self, lam: np.ndarray, s2: float) -> np.ndarray:
        return self.A / s2 + self.Af + np.diag(1.0 / lam)

    def _solve(self, phi: np.ndarray, sigma: float):
        lam = self.prior.lam(phi)
        s2 = sigma * sigma
        R = _cho(self._precision(lam, s2))
        ctot = self.c / s2 + self.cf
        m = linalg.cho_solve((R, True), ctot, check_finite=False)
        return lam, s2, R, ctot, m

    def objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        G = len(self.prior.groups)
        log_phi, log_sigma = theta[:G], theta[G]
        phi, sigma = np.exp(log_phi), math.exp(log_sigma)
        lam, s2, R, ctot, m = self._solve(phi, sigma)
        logdetP = 2.0 * float(np.sum(np.log(np.diag(R))))
        logp = -0.5 * (self.yWy / s2 + self.fVf - float(ctot @ m) + self.n * math.log(s2) - self.slw + self.slv
                       + float(np.sum(np.log(lam))) + logdetP + self.n * math.log(2 * math.pi))
        Rinv = _inv_lower(R)
        diag_S = np.sum(Rinv * Rinv, axis=0)
        per_node = (m * m + diag_S) / lam - 1.0
        grad_phi = np.zeros(G)
        grouped = self.prior.group_idx >= 0
        np.add.at(grad_phi, self.prior.group_idx[grouped], per_node[grouped])
        trAfS = float(np.sum((Rinv @ self.Af) * Rinv)) if self.factors is not None else 0.0
        trAS = s2 * (self.p - trAfS - float(np.sum(diag_S / lam)))
        rss = self.yWy - 2.0 * float(m @ self.c) + float(m @ self.A @ m)
        grad_sigma = (rss + trAS) / s2 - self.n
        dphi = log_phi - np.log(self.prior.default_phi)
        logp += -0.5 * float(np.sum(dphi ** 2)) / HYPER_SD ** 2
        grad_phi -= dphi / HYPER_SD ** 2
        dsig = log_sigma - math.log(self.sigma0)
        logp += -0.5 * dsig ** 2 / SIGMA_PRIOR_SD ** 2
        grad_sigma -= dsig / SIGMA_PRIOR_SD ** 2
        return -logp, -np.concatenate([grad_phi, [grad_sigma]])

    def fit(self, *, eb: bool = True, phi: np.ndarray | None = None, sigma: float | None = None,
            maxiter: int = 200) -> HeadFit:
        G = len(self.prior.groups)
        phi0 = self.prior.default_phi.copy() if phi is None else np.asarray(phi, float)
        sig0 = self.sigma0 if sigma is None else float(sigma)
        iterations = 0
        info: dict = {}
        if eb and self.n > 0:
            theta0 = np.concatenate([np.log(phi0), [math.log(sig0)]])
            bounds = [LOG_PHI_BOUNDS] * G + [(math.log(1e-3), math.log(20.0))]
            res = optimize.minimize(self.objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": maxiter})
            theta = res.x if np.all(np.isfinite(res.x)) else theta0
            phi0, sig0 = np.exp(theta[:G]), math.exp(theta[G])
            iterations = int(res.nit)
            info = {"converged": bool(res.success), "message": str(res.message)}
        lam, s2, R, ctot, m = self._solve(phi0, sig0)
        ev = -self.objective(np.concatenate([np.log(phi0), [math.log(sig0)]]))[0]
        return HeadFit("gaussian", m, cov_factor(R), phi0, sig0, ev, iterations, info,
                       sparse.csr_matrix(self._precision(lam, s2)))


# ---------------------------------------------------------------- logistic heads

class LogisticHead:
    """P(z = 1) = q s(Xb) + (1 - q)(1 - s(Xb)), b ~ N(0, Lambda(phi)), Laplace at the mode."""

    def __init__(self, X: sparse.csr_matrix, z: np.ndarray, q: np.ndarray, prior: Prior,
                 factors: Factors | None = None):
        self.X = X.tocsr()
        self.z = np.asarray(z, float)
        self.q = np.clip(np.asarray(q, float), 0.5, 1.0)
        self.prior = prior
        self.n, self.p = self.X.shape
        self.factors = factors
        self.Af = factors.A if factors is not None else np.zeros((self.p, self.p))
        self.cf = factors.c if factors is not None else np.zeros(self.p)
        self._b = np.zeros(self.p)
        self.fisher_fallbacks = 0  # evidence evaluations whose observed Hessian was not positive definite

    def _loglik_terms(self, eta: np.ndarray):
        s = expit(eta)
        a = 2.0 * self.q - 1.0
        pi = np.clip((1.0 - self.q) + a * s, 1e-12, 1 - 1e-12)
        dpi = a * s * (1.0 - s)
        ll = float(np.sum(self.z * np.log(pi) + (1.0 - self.z) * np.log1p(-pi)))
        g_eta = dpi * (self.z / pi - (1.0 - self.z) / (1.0 - pi))
        info = np.maximum(dpi * dpi / (pi * (1.0 - pi)), 1e-12)
        return ll, g_eta, info

    def _dinfo(self, eta: np.ndarray) -> np.ndarray:
        """d info / d eta for the expected information a^2 r^2 / (pi (1 - pi)), r = s (1 - s)."""
        s = expit(eta)
        a = 2.0 * self.q - 1.0
        r = s * (1.0 - s)
        pi = np.clip((1.0 - self.q) + a * s, 1e-12, 1 - 1e-12)
        pp = pi * (1.0 - pi)
        return a * a * r * r * (2.0 * (1.0 - 2.0 * s) * pp - a * r * (1.0 - 2.0 * pi)) / (pp * pp)

    def _observed(self, eta: np.ndarray) -> np.ndarray:
        """Observed information -d^2 loglik / d eta^2 per row (differs from the expected one when q < 1)."""
        s = expit(eta)
        a = 2.0 * self.q - 1.0
        r = s * (1.0 - s)
        pi = np.clip((1.0 - self.q) + a * s, 1e-12, 1 - 1e-12)
        d1, d2 = a * r, a * r * (1.0 - 2.0 * s)
        return np.where(self.z > 0.5, (d1 * d1 - d2 * pi) / (pi * pi), (d2 * (1.0 - pi) + d1 * d1) / (1.0 - pi) ** 2)

    def _dobserved(self, eta: np.ndarray) -> np.ndarray:
        """d observed information / d eta = -d^3 loglik / d eta^3 per row.

        With f = pi (z = 1) or 1 - pi (z = 0): l''' = f'''/f - 3 f' f''/f^2 + 2 (f'/f)^3, where
        pi' = a r, pi'' = a r (1 - 2s), pi''' = a r (1 - 6r).
        """
        s = expit(eta)
        a = 2.0 * self.q - 1.0
        r = s * (1.0 - s)
        pi = np.clip((1.0 - self.q) + a * s, 1e-12, 1 - 1e-12)
        sign = np.where(self.z > 0.5, 1.0, -1.0)
        f = np.where(self.z > 0.5, pi, 1.0 - pi)
        f1, f2, f3 = sign * a * r, sign * a * r * (1.0 - 2.0 * s), sign * a * r * (1.0 - 6.0 * r)
        g1 = f1 / f
        return -(f3 / f - 3.0 * g1 * f2 / f + 2.0 * g1 ** 3)

    def _precision(self, b: np.ndarray, lam_inv: np.ndarray) -> tuple[np.ndarray, str, np.ndarray]:
        """Cholesky factor of the Laplace precision at `b`: the observed Hessian of the log posterior,
        or the Fisher one when the observed Hessian is not positive definite (D68); and that precision."""
        eta = self.X @ b
        XO = self.X.multiply(self._observed(eta)[:, None]).tocsr()
        H = np.asarray((self.X.T @ XO).todense()) + np.diag(lam_inv) + self.Af
        try:
            return np.tril(linalg.cholesky(H, lower=True, check_finite=False)), "observed", H
        except linalg.LinAlgError:
            _, _, info = self._loglik_terms(eta)
            XI = self.X.multiply(info[:, None]).tocsr()
            F = np.asarray((self.X.T @ XI).todense()) + np.diag(lam_inv) + self.Af
            return _cho(F), "fisher", F

    def _logpost(self, b: np.ndarray, lam_inv: np.ndarray) -> float:
        ll = self._loglik_terms(self.X @ b)[0]
        return ll - 0.5 * float(b @ (lam_inv * b)) - 0.5 * float(b @ self.Af @ b) + float(self.cf @ b)

    def mode(self, lam: np.ndarray, b0: np.ndarray | None = None, iters: int = 60):
        lam_inv = 1.0 / lam
        b = self._b.copy() if b0 is None else b0.copy()
        lp = self._logpost(b, lam_inv)
        R = None
        for _ in range(iters):
            eta = self.X @ b
            _, g_eta, info = self._loglik_terms(eta)
            grad = np.asarray(self.X.T @ g_eta).ravel() - lam_inv * b - self.Af @ b + self.cf
            XI = self.X.multiply(info[:, None]).tocsr()
            H = np.asarray((self.X.T @ XI).todense()) + np.diag(lam_inv) + self.Af
            R = _cho(H)
            step = linalg.cho_solve((R, True), grad, check_finite=False)
            t = 1.0
            while True:
                nb = b + t * step
                nlp = self._logpost(nb, lam_inv)
                if nlp >= lp - 1e-10 or t < 1e-4:
                    break
                t *= 0.5
            b, lp = nb, nlp
            if float(np.max(np.abs(t * step))) < 1e-7:
                break
        R, kind, H = self._precision(b, lam_inv)
        self._b = b
        return b, R, lp, kind, H

    def objective(self, log_phi: np.ndarray) -> tuple[float, np.ndarray]:
        phi = np.exp(log_phi)
        lam = self.prior.lam(phi)
        b, R, lp, kind, _ = self.mode(lam)
        logdetH = 2.0 * float(np.sum(np.log(np.diag(R))))
        logev = lp - 0.5 * float(np.sum(np.log(lam))) - 0.5 * logdetH
        Rinv = _inv_lower(R)
        diag_S = np.sum(Rinv * Rinv, axis=0)
        per_node = (b * b + diag_S) / lam - 1.0
        # implicit term: the mode moves with lam (through the observed Hessian) and the
        # information in log|H| moves with the mode
        XR = np.asarray(self.X @ Rinv.T)
        v = np.sum(XR * XR, axis=1)
        eta = self.X @ b
        if kind == "observed":
            u = np.asarray(self.X.T @ (v * self._dobserved(eta))).ravel()
            Hu = Rinv.T @ (Rinv @ u)
        elif np.all(self.q >= 1.0):
            self.fisher_fallbacks += 1
            u = np.asarray(self.X.T @ (v * self._dinfo(eta))).ravel()
            Hu = Rinv.T @ (Rinv @ u)
        else:
            self.fisher_fallbacks += 1
            u = np.asarray(self.X.T @ (v * self._dinfo(eta))).ravel()
            XO = self.X.multiply(self._observed(eta)[:, None]).tocsr()
            Hobs = np.asarray((self.X.T @ XO).todense()) + np.diag(1.0 / lam) + self.Af
            try:
                Hu = linalg.solve(Hobs, u, assume_a="sym", check_finite=False)
            except linalg.LinAlgError:
                Hu = Rinv.T @ (Rinv @ u)
        per_node -= Hu * b / lam
        grad = np.zeros(len(self.prior.groups))
        grouped = self.prior.group_idx >= 0
        np.add.at(grad, self.prior.group_idx[grouped], per_node[grouped])
        dphi = log_phi - np.log(self.prior.default_phi)
        logev += -0.5 * float(np.sum(dphi ** 2)) / HYPER_SD ** 2
        grad -= dphi / HYPER_SD ** 2
        return -logev, -grad

    def fit(self, *, eb: bool = True, phi: np.ndarray | None = None, maxiter: int = 100) -> HeadFit:
        phi0 = self.prior.default_phi.copy() if phi is None else np.asarray(phi, float)
        iterations, info = 0, {}
        if eb and self.n > 0 and len(self.prior.groups):
            res = optimize.minimize(self.objective, np.log(phi0), jac=True, method="L-BFGS-B",
                                    bounds=[LOG_PHI_BOUNDS] * len(self.prior.groups), options={"maxiter": maxiter})
            if np.all(np.isfinite(res.x)):
                phi0 = np.exp(res.x)
            iterations = int(res.nit)
            info = {"converged": bool(res.success), "message": str(res.message)}
        lam = self.prior.lam(phi0)
        b, R, lp, kind, H = self.mode(lam)
        logev = lp - 0.5 * float(np.sum(np.log(lam))) - float(np.sum(np.log(np.diag(R))))
        info = {**info, "hessian": kind, "fisher_fallbacks": self.fisher_fallbacks}
        return HeadFit("logistic", b, cov_factor(R), phi0, 1.0, logev, iterations, info, sparse.csr_matrix(H))


# ---------------------------------------------------------------- updates (look-ahead, conditioned)

def gaussian_rank_one(mean: np.ndarray, U: np.ndarray, x: np.ndarray, y: float, tau2: float):
    """Posterior after one more observation y ~ N(x'b, tau2): (mean', U') with U'U'^T = Sigma'."""
    v = U.T @ x
    t = float(v @ v)
    s = t + tau2
    Sx = U @ v
    new_mean = mean + Sx * (y - float(x @ mean)) / s
    if t <= 0:
        return new_mean, U
    alpha = (1.0 - math.sqrt(tau2 / s)) / t
    return new_mean, U - alpha * np.outer(Sx, v)


def logistic_linearized(mean: np.ndarray, U: np.ndarray, x: np.ndarray, z: float, q: float):
    """One more logistic observation, linearized at the current mode (one Newton step).

    Precision gains `I x x'` (Fisher information at the mode); the mean moves by
    `Sigma' x g`, with `g` the score of the observation at the mode.
    """
    eta = float(x @ mean)
    s = float(expit(eta))
    a = 2.0 * q - 1.0
    pi = min(max((1.0 - q) + a * s, 1e-12), 1 - 1e-12)
    dpi = a * s * (1.0 - s)
    g = dpi * (z / pi - (1.0 - z) / (1.0 - pi))
    info = max(dpi * dpi / (pi * (1.0 - pi)), 1e-12)
    v = U.T @ x
    t = float(v @ v)
    Sx = U @ v
    new_mean = mean + Sx * g / (1.0 + info * t)
    if t <= 0:
        return new_mean, U
    tau2 = 1.0 / info
    alpha = (1.0 - math.sqrt(tau2 / (t + tau2))) / t
    return new_mean, U - alpha * np.outer(Sx, v)
