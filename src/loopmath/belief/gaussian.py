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
from scipy import linalg, sparse
from scipy.special import expit

from .block import BlockFactor, leaf_split
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
    phi: np.ndarray
    sigma: float = 1.0
    log_evidence: float = float("nan")
    iterations: int = 0
    info: dict = field(default_factory=dict)
    precision: sparse.csr_matrix | None = None  # the matrix whose Cholesky factor gives U (D94)
    _U: np.ndarray | None = field(default=None, repr=False)

    @property
    def cov_factor(self) -> np.ndarray:
        """U with U U' = posterior covariance (upper triangular), from the precision on first use;
        the fit itself never needs it."""
        if self._U is None:
            self._U = cov_factor(_cho(self.precision.toarray()))
        return self._U

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

def _factor_precision(factors: "Factors | None", p: int) -> sparse.csr_matrix:
    """F' V^-1 F as a sparse matrix (zero without factors)."""
    if factors is None:
        return sparse.csr_matrix((p, p))
    return (factors.F.T @ factors.F.multiply(1.0 / factors.v[:, None])).tocsr()


def _unperm(perm: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[perm] = x
    return out


class GaussianHead:
    """y ~ N(Xb, sigma^2 / w), b ~ N(0, Lambda(phi)), optional prior factors.

    The precision is factored by blocks (`block.py`): the leaf nodes (every task) are eliminated
    in closed form and only the core's Schur complement is dense."""

    def __init__(self, X: sparse.csr_matrix, y: np.ndarray, w: np.ndarray, prior: Prior,
                 factors: Factors | None = None):
        self.X = X.tocsr()
        self.y = np.asarray(y, float)
        self.w = np.asarray(w, float)
        self.prior = prior
        self.n, self.p = self.X.shape
        Xw = self.X.multiply(self.w[:, None]).tocsr()
        self.As = (self.X.T @ Xw).tocsr()
        self.c = np.asarray(self.X.T @ (self.w * self.y)).ravel()
        self.yWy = float(np.sum(self.w * self.y * self.y))
        self.slw = float(np.sum(np.log(self.w)))
        self.factors = factors
        self.Afs = _factor_precision(factors, self.p)
        self.cf = factors.c if factors is not None else np.zeros(self.p)
        self.fVf = float(np.sum(factors.f ** 2 / factors.v)) if factors is not None else 0.0
        self.slv = float(np.sum(np.log(factors.v))) if factors is not None else 0.0
        self.sigma0 = self._sigma_start()
        self.perm, self.l = leaf_split(abs(self.As) + abs(self.Afs))
        perm, l = self.perm, self.l
        Ap, Fp = self.As[perm][:, perm].tocsr(), self.Afs[perm][:, perm].tocsr()
        self._Ab = (Ap.diagonal()[:l], Ap[l:, :l].tocsr(), Ap[l:, l:].toarray())
        self._Fb = (Fp.diagonal()[:l], Fp[l:, :l].tocsr(), Fp[l:, l:].toarray())
        self.Fp = factors.F[:, perm].tocsr() if factors is not None else None

    @property
    def A(self) -> np.ndarray:
        return self.As.toarray()

    @property
    def Af(self) -> np.ndarray:
        return self.Afs.toarray()

    def _sigma_start(self) -> float:
        if self.n < 2:
            return 1.0
        mu = np.sum(self.w * self.y) / np.sum(self.w)
        sd = math.sqrt(float(np.sum(self.w * (self.y - mu) ** 2) / np.sum(self.w)))
        return max(0.05, min(5.0, sd if sd > 0 else 1.0))

    def _precision(self, lam: np.ndarray, s2: float) -> sparse.csr_matrix:
        P = (self.As / s2 + self.Afs + sparse.diags(1.0 / lam)).tocsr()
        P.eliminate_zeros()
        return P

    def _factor(self, lam: np.ndarray, s2: float) -> BlockFactor:
        lp, l = lam[self.perm], self.l
        (ad, aB, aC), (fd, fB, fC) = self._Ab, self._Fb
        PCC = aC / s2 + fC
        PCC[np.diag_indices_from(PCC)] += 1.0 / lp[l:]
        return BlockFactor.factor(ad / s2 + fd + 1.0 / lp[:l], (aB / s2 + fB).tocsr(), PCC)

    def _solve(self, phi: np.ndarray, sigma: float):
        lam = self.prior.lam(phi)
        s2 = sigma * sigma
        fac = self._factor(lam, s2)
        ctot = self.c / s2 + self.cf
        m = _unperm(self.perm, fac.solve(ctot[self.perm]))
        return lam, s2, fac, ctot, m

    def objective(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        G = len(self.prior.groups)
        log_phi, log_sigma = theta[:G], theta[G]
        phi, sigma = np.exp(log_phi), math.exp(log_sigma)
        lam, s2, fac, ctot, m = self._solve(phi, sigma)
        logdetP = fac.logdet
        logp = -0.5 * (self.yWy / s2 + self.fVf - float(ctot @ m) + self.n * math.log(s2) - self.slw + self.slv
                       + float(np.sum(np.log(lam))) + logdetP + self.n * math.log(2 * math.pi))
        diag_S = _unperm(self.perm, fac.diag_inv())
        per_node = (m * m + diag_S) / lam - 1.0
        grad_phi = np.zeros(G)
        grouped = self.prior.group_idx >= 0
        np.add.at(grad_phi, self.prior.group_idx[grouped], per_node[grouped])
        trAfS = float(np.sum(fac.quad(self.Fp) / self.factors.v)) if self.factors is not None else 0.0
        trAS = s2 * (self.p - trAfS - float(np.sum(diag_S / lam)))
        rss = self.yWy - 2.0 * float(m @ self.c) + float(m @ (self.As @ m))
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
            from scipy import optimize  # only the fit needs it (about 0.1 s to import)

            res = optimize.minimize(self.objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": maxiter})
            theta = res.x if np.all(np.isfinite(res.x)) else theta0
            phi0, sig0 = np.exp(theta[:G]), math.exp(theta[G])
            iterations = int(res.nit)
            info = {"converged": bool(res.success), "message": str(res.message)}
        lam, s2, _, _, m = self._solve(phi0, sig0)
        ev = -self.objective(np.concatenate([np.log(phi0), [math.log(sig0)]]))[0]
        return HeadFit("gaussian", m, phi0, sig0, ev, iterations, info, self._precision(lam, s2))


# ---------------------------------------------------------------- logistic heads

class LogisticHead:
    """P(z = 1) = q s(Xb) + (1 - q)(1 - s(Xb)), b ~ N(0, Lambda(phi)), Laplace at the mode.

    Precisions are factored by blocks, as in GaussianHead."""

    def __init__(self, X: sparse.csr_matrix, z: np.ndarray, q: np.ndarray, prior: Prior,
                 factors: Factors | None = None):
        self.X = X.tocsr()
        self.z = np.asarray(z, float)
        self.q = np.clip(np.asarray(q, float), 0.5, 1.0)
        self.prior = prior
        self.n, self.p = self.X.shape
        self.factors = factors
        self.Afs = _factor_precision(factors, self.p)
        self.cf = factors.c if factors is not None else np.zeros(self.p)
        self._b = np.zeros(self.p)
        self.fisher_fallbacks = 0  # evidence evaluations whose observed Hessian was not positive definite
        aX = abs(self.X)
        self.perm, self.l = leaf_split((aX.T @ aX) + abs(self.Afs))
        perm, l = self.perm, self.l
        self.Xp = self.X[:, perm].tocsr()
        self._XL, self._XC = self.Xp[:, :l].tocsr(), self.Xp[:, l:].tocsr()
        self._XL2 = self._XL.multiply(self._XL).T.tocsr()
        Fp = self.Afs[perm][:, perm].tocsr()
        self._Fb = (Fp.diagonal()[:l], Fp[l:, :l].tocsr(), Fp[l:, l:].toarray())

    @property
    def Af(self) -> np.ndarray:
        return self.Afs.toarray()

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

    def _blocks(self, o: np.ndarray, lam_inv: np.ndarray):
        """(d, B, P_CC) of X' diag(o) X + diag(lam_inv) + Af in the leaf-first order."""
        l = self.l
        li = lam_inv[self.perm]
        fd, fB, fC = self._Fb
        d = np.asarray(self._XL2 @ o).ravel() + li[:l] + fd
        XCo = self._XC.multiply(o[:, None]).tocsr()
        B = (XCo.T @ self._XL).tocsr() + fB
        PCC = np.asarray((self._XC.T @ XCo).todense())
        PCC[np.diag_indices_from(PCC)] += li[l:]
        return d, B.tocsr(), PCC + fC

    def _hessian(self, o: np.ndarray, lam_inv: np.ndarray) -> sparse.csr_matrix:
        H = (self.X.T @ self.X.multiply(o[:, None]) + sparse.diags(lam_inv) + self.Afs).tocsr()
        H.eliminate_zeros()
        return H

    def _precision(self, b: np.ndarray, lam_inv: np.ndarray) -> tuple[BlockFactor, str, np.ndarray]:
        """Factor of the Laplace precision at `b`: the observed Hessian of the log posterior, or the
        Fisher one when the observed Hessian is not positive definite (D68); and its row weights."""
        eta = self.X @ b
        o = self._observed(eta)
        try:
            return BlockFactor.factor(*self._blocks(o, lam_inv), strict=True), "observed", o
        except linalg.LinAlgError:
            _, _, info = self._loglik_terms(eta)
            return BlockFactor.factor(*self._blocks(info, lam_inv)), "fisher", info

    def _logpost(self, b: np.ndarray, lam_inv: np.ndarray) -> float:
        ll = self._loglik_terms(self.X @ b)[0]
        return ll - 0.5 * float(b @ (lam_inv * b)) - 0.5 * float(b @ (self.Afs @ b)) + float(self.cf @ b)

    def mode(self, lam: np.ndarray, b0: np.ndarray | None = None, iters: int = 60):
        lam_inv = 1.0 / lam
        b = self._b.copy() if b0 is None else b0.copy()
        lp = self._logpost(b, lam_inv)
        for _ in range(iters):
            eta = self.X @ b
            _, g_eta, info = self._loglik_terms(eta)
            grad = np.asarray(self.X.T @ g_eta).ravel() - lam_inv * b - self.Afs @ b + self.cf
            # Newton on the observed Hessian when it is positive definite: with q < 1 Fisher scoring
            # converges linearly and stopped at `iters` with max|grad| near 3e-5 on real data
            try:
                fac = BlockFactor.factor(*self._blocks(self._observed(eta), lam_inv), strict=True)
            except linalg.LinAlgError:
                fac = BlockFactor.factor(*self._blocks(info, lam_inv))
            step = _unperm(self.perm, fac.solve(grad[self.perm]))
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
        fac, kind, weights = self._precision(b, lam_inv)
        self._b = b
        return b, fac, lp, kind, weights

    def _solve_indefinite(self, o: np.ndarray, lam_inv: np.ndarray, u: np.ndarray) -> np.ndarray:
        """H^-1 u for the observed Hessian when it is not positive definite: blocks, S by LU."""
        d, B, PCC = self._blocks(o, lam_inv)
        if np.any(d == 0):
            raise linalg.LinAlgError("singular leaf block")
        l, up = self.l, u[self.perm]
        Bd = B.multiply(1.0 / d[None, :]).tocsr()
        S = PCC - (Bd @ B.T).toarray() if l else PCC
        xC = linalg.solve(S, up[l:] - Bd @ up[:l], assume_a="sym", check_finite=False) if len(S) else up[l:]
        xL = (up[:l] - B.T @ xC) / d
        return _unperm(self.perm, np.concatenate([xL, xC]))

    def objective(self, log_phi: np.ndarray) -> tuple[float, np.ndarray]:
        phi = np.exp(log_phi)
        lam = self.prior.lam(phi)
        b, fac, lp, kind, _ = self.mode(lam)
        logev = lp - 0.5 * float(np.sum(np.log(lam))) - 0.5 * fac.logdet
        diag_S = _unperm(self.perm, fac.diag_inv())
        per_node = (b * b + diag_S) / lam - 1.0
        # implicit term: the mode moves with lam (through the observed Hessian) and the
        # information in log|H| moves with the mode
        v = fac.quad(self.Xp)
        eta = self.X @ b
        if kind == "observed":
            u = np.asarray(self.X.T @ (v * self._dobserved(eta))).ravel()
            Hu = _unperm(self.perm, fac.solve(u[self.perm]))
        elif np.all(self.q >= 1.0):
            self.fisher_fallbacks += 1
            u = np.asarray(self.X.T @ (v * self._dinfo(eta))).ravel()
            Hu = _unperm(self.perm, fac.solve(u[self.perm]))
        else:
            self.fisher_fallbacks += 1
            u = np.asarray(self.X.T @ (v * self._dinfo(eta))).ravel()
            try:
                Hu = self._solve_indefinite(self._observed(eta), 1.0 / lam, u)
            except linalg.LinAlgError:
                Hu = _unperm(self.perm, fac.solve(u[self.perm]))
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
            from scipy import optimize  # only the fit needs it (about 0.1 s to import)

            res = optimize.minimize(self.objective, np.log(phi0), jac=True, method="L-BFGS-B",
                                    bounds=[LOG_PHI_BOUNDS] * len(self.prior.groups), options={"maxiter": maxiter})
            if np.all(np.isfinite(res.x)):
                phi0 = np.exp(res.x)
            iterations = int(res.nit)
            info = {"converged": bool(res.success), "message": str(res.message)}
        lam = self.prior.lam(phi0)
        b, fac, lp, kind, weights = self.mode(lam)
        logev = lp - 0.5 * float(np.sum(np.log(lam))) - 0.5 * fac.logdet
        info = {**info, "hessian": kind, "fisher_fallbacks": self.fisher_fallbacks}
        return HeadFit("logistic", b, phi0, 1.0, logev, iterations, info, self._hessian(weights, 1.0 / lam))


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
