"""The screen's model: a Bayesian linear model of each pair's difference on its fold-over row.

For one head (success, or log cost), pair i's difference d_i (arm A minus arm B) is x_i . delta + e_i, with x_i
the pair's +-1 row and delta_j edit j's on-versus-off effect (averaged over the other edits being on half the
time). The prior is delta_j ~ N(0, tau^2) independently, a shrinkage prior that pulls small, noisy effects to 0,
and e_i ~ N(0, sigma^2). Given (sigma, tau) the posterior of delta is normal in closed form. sigma and tau are not
plugged in: the fit weighs a grid of (sigma, tau) by marginal likelihood times prior, so the ranges carry the
doubt about both (a plugged-in scale undercovers, as loopmath's F1 review found). Draws from that mixture give
each edit's mean, 80% range and P(helps), and any subset's predicted gain: the sum of its edits' effects.

Edits are rarely random in sign: a batch written to cut cost mostly cuts it a little. So each effect is a shared
mean plus its own deviation, delta_j = mu + eta_j, with mu ~ N(0, s_mu^2): the prior covariance is
tau^2 I + s_mu^2 J, still normal and closed form. In a fold-over design mu is read from how many more edits one arm
has on than the other, so it is well measured, and a subset's predicted gain is not shrunk once per edit (without
it, five small same-direction effects that together cut cost by about a quarter read as a cut of a third
of that in simulation).

A common tau shrinks one large effect among many null ones hard (a true -0.5 among 7 nulls read as -0.13 to
-0.54 over 24 pairs). So when the pairs so far form an orthogonal design (X'X = m I, as a finished
Plackett-Burman or Hadamard screen does) with 2 or more spare pairs, the prior is sparse instead: each delta_j is
"near zero" (N(0, (kappa tau)^2)) with probability 1 - pi and N(0, tau^2) with probability pi. An orthogonal
design makes the edits' least-squares estimates independent given sigma, so the posterior is still closed form
per edit on a (sigma, tau, pi) grid. Other designs (a screen part way, a supersaturated one) use the common tau.

`kg` prices one more pair: the expected rise in the value of the best subset after it, per future run (the
knowledge gradient of choosing each edit on or off), from how much a random balanced row would narrow each
effect. It is a guide, not calibrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scale:
    """A head's grid of noise (sigma) and effect (tau) scales, log-spaced, with the prior on them: log-uniform
    on sigma, half-Cauchy(0, tau_prior) on the common tau, half-Cauchy(0, slab_prior) on the sparse prior's slab
    scale, whose range `slab` starts at the size of an effect worth finding."""

    name: str
    sigma: tuple[float, float]
    tau: tuple[float, float]
    tau_prior: float
    slab: tuple[float, float]
    slab_prior: float
    points: int = 24

    def scaled(self, k: float) -> "Scale":
        """The same grid for a head measured in units k times larger (a raw score)."""
        return Scale(self.name, (self.sigma[0] * k, self.sigma[1] * k), (self.tau[0] * k, self.tau[1] * k),
                     self.tau_prior * k, (self.slab[0] * k, self.slab[1] * k), self.slab_prior * k, self.points)


COST_SCALE = Scale("log cost", sigma=(0.02, 2.0), tau=(0.005, 1.0), tau_prior=0.2, slab=(0.03, 1.5),
                   slab_prior=0.3)  # natural log of USD
SUCCESS_SCALE = Scale("success", sigma=(0.02, 1.5), tau=(0.002, 0.6), tau_prior=0.1, slab=(0.03, 0.8),
                      slab_prior=0.25)  # 0/1 accepted


@dataclass
class HeadFit:
    """The posterior of one head's edit effects: a mixture over the (sigma, tau) grid of normals."""

    means: np.ndarray  # (K, p) the posterior mean at each grid point that carries weight
    covs: np.ndarray  # (K, p, p)
    weights: np.ndarray  # (K,) sums to 1
    sigmas: np.ndarray  # (K,)
    taus: np.ndarray  # (K,)
    n: int  # pairs
    ceiling: bool  # every difference is 0: the head cannot rank the edits (all pass, or all fail)
    evidence: float = float("nan")  # log marginal likelihood of the differences, the grid's prior normalised

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.means

    def var(self) -> np.ndarray:
        m1 = self.weights @ self.means
        m2 = self.weights @ (np.diagonal(self.covs, axis1=1, axis2=2) + self.means ** 2)
        return m2 - m1 ** 2

    def draws(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = rng.choice(len(self.weights), size=n, p=self.weights)
        chol = np.linalg.cholesky(self.covs + 1e-12 * np.eye(self.covs.shape[1]))
        z = rng.standard_normal((n, self.covs.shape[1]))
        return self.means[k] + np.einsum("nij,nj->ni", chol[k], z)

    def kg_sd(self) -> np.ndarray:
        """How far one more pair, with a random balanced row, would move each effect's posterior mean (its
        standard deviation, averaged over the grid): sum_k S_jk^2 / (trace S + sigma^2)."""
        num = (self.covs ** 2).sum(axis=2)  # (K, p)
        den = np.trace(self.covs, axis1=1, axis2=2) + self.sigmas ** 2  # (K,)
        return np.sqrt(self.weights @ (num / den[:, None]))

    def scales(self) -> dict:
        """The posterior means of sigma and tau, to show what the fit read from the pairs."""
        return {"sigma": float(self.weights @ self.sigmas), "tau": float(self.weights @ self.taus)}


def fit_head(rows: np.ndarray, d: np.ndarray, scale: Scale = COST_SCALE, prior: str = "auto", shared: bool = True):
    """The posterior of the edit effects from the pairs' rows (m by p, +-1) and differences (m). With `prior`
    "auto": when the rows are orthogonal with 2 or more spare pairs, `fit_sparse` and `fit_ridge` averaged by their
    evidence (`MixFit`), else `fit_ridge`. `shared` adds the edits' shared mean."""
    X = np.asarray(rows, dtype=float)
    m, p = X.shape
    orthogonal = m >= p + 2 and np.array_equal(X.T @ X, m * np.eye(p))
    if prior == "sparse":
        return fit_sparse(X, d, scale, shared=shared)
    if prior == "ridge" or not orthogonal:
        return fit_ridge(X, d, scale, shared=shared)
    fits = [fit_sparse(X, d, scale, shared=shared), fit_ridge(X, d, scale, shared=shared)]
    ev = np.array([f.evidence for f in fits])
    probs = np.exp(ev - ev.max())
    return MixFit(fits=fits, probs=probs / probs.sum())


def fit_ridge(rows: np.ndarray, d: np.ndarray, scale: Scale = COST_SCALE, shared: bool = True) -> HeadFit:
    """The common-tau posterior: normal given (sigma, tau), mixed over the grid. With `shared`, the prior
    covariance is tau^2 I + s_mu^2 J (s_mu the scale's slab_prior)."""
    X = np.asarray(rows, dtype=float)
    d = np.asarray(d, dtype=float)
    m, p = X.shape
    if m < 1:
        raise ValueError("no scored pair yet")
    sig = np.geomspace(*scale.sigma, scale.points)
    tau = np.geomspace(*scale.tau, scale.points)
    S2, T2 = np.meshgrid(sig ** 2, tau ** 2, indexing="ij")
    s2, t2 = S2.ravel(), T2.ravel()
    xtx, xtd, dtd = X.T @ X, X.T @ d, float(d @ d)
    mu2 = scale.slab_prior ** 2 if shared else 0.0
    # (tau^2 I + mu2 J)^-1 = I / tau^2 - mu2 J / (tau^2 (tau^2 + p mu2)), and its log determinant
    prior_prec = (np.eye(p)[None] / t2[:, None, None]
                  - (mu2 / (t2 * (t2 + p * mu2)))[:, None, None] * np.ones((p, p))[None])
    logdet_prior = p * np.log(t2) + np.log1p(p * mu2 / t2)
    A = xtx[None] / s2[:, None, None] + prior_prec
    cov = np.linalg.inv(A)
    b = xtd[None] / s2[:, None]  # (K, p)
    mean = np.einsum("kij,kj->ki", cov, b)
    _, logdet_a = np.linalg.slogdet(A)
    quad = dtd / s2 - np.einsum("ki,ki->k", b, mean)
    loglik = -0.5 * (m * np.log(2 * math.pi) + m * np.log(s2) + logdet_prior + logdet_a + quad)
    t = np.sqrt(t2)
    logprior = np.log(t) - np.log1p((t / scale.tau_prior) ** 2)  # half-Cauchy on tau, times tau on a log grid
    logw = loglik + logprior
    w = np.exp(logw - logw.max())
    keep = w > 1e-9 * w.max()
    w = w[keep] / w[keep].sum()
    return HeadFit(means=mean[keep], covs=cov[keep], weights=w, sigmas=np.sqrt(s2[keep]), taus=t[keep], n=m,
                   ceiling=bool(np.all(d == 0)), evidence=_evidence(logw, logprior))


def _lse(x: np.ndarray) -> float:
    top = float(np.max(x))
    return top + float(np.log(np.exp(x - top).sum()))


def _evidence(logw: np.ndarray, logprior: np.ndarray) -> float:
    """log of the marginal likelihood averaged over the grid's prior (normalised over the grid)."""
    return _lse(logw) - _lse(logprior)


KAPPA = 0.05  # the "near zero" component's scale, as a fraction of tau
PIS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])  # the grid of the share of edits that matter, uniform prior


@dataclass
class SparseFit:
    """The spike-and-slab posterior on an orthogonal design: per grid point (sigma, tau, pi) and edit, a mixture
    of two normals (near zero, and slab)."""

    comp_w: np.ndarray  # (K, p, 2) the weight of each component per grid point and edit
    comp_mean: np.ndarray  # (K, p, 2)
    comp_var: np.ndarray  # (K, p, 2)
    weights: np.ndarray  # (K,)
    sigmas: np.ndarray
    taus: np.ndarray
    pis: np.ndarray
    mus: np.ndarray  # the shared mean at each grid point
    n: int
    ceiling: bool
    evidence: float = float("nan")

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ (self.comp_w * self.comp_mean).sum(axis=2)

    def var(self) -> np.ndarray:
        """Each edit's posterior variance."""
        m1 = (self.comp_w * self.comp_mean).sum(axis=2)
        m2 = (self.comp_w * (self.comp_var + self.comp_mean ** 2)).sum(axis=2)
        return self.weights @ m2 - (self.weights @ m1) ** 2

    def draws(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        k = rng.choice(len(self.weights), size=n, p=self.weights)
        p = self.comp_w.shape[1]
        slab = rng.random((n, p)) < self.comp_w[k, :, 1]
        idx = slab.astype(int)
        mu = np.take_along_axis(self.comp_mean[k], idx[..., None], axis=2)[..., 0]
        sd = np.sqrt(np.take_along_axis(self.comp_var[k], idx[..., None], axis=2)[..., 0])
        return mu + sd * rng.standard_normal((n, p))

    def kg_sd(self) -> np.ndarray:
        v = self.var()
        sigma2 = float(self.weights @ self.sigmas ** 2)
        return np.sqrt(v ** 2 / (v.sum() + sigma2))

    def scales(self) -> dict:
        return {"sigma": float(self.weights @ self.sigmas), "tau": float(self.weights @ self.taus),
                "pi": float(self.weights @ self.pis), "shared_mean": float(self.weights @ self.mus)}

    def p_matters(self) -> np.ndarray:
        """Each edit's posterior chance of being in the slab (an effect beyond near zero)."""
        return self.weights @ self.comp_w[:, :, 1]


def _norm_logpdf(x: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (np.log(2 * math.pi * var) + x ** 2 / var)


MUS = np.linspace(-2.5, 2.5, 41)  # the shared mean's grid, in units of s_mu, with a N(0, 1) prior


def fit_sparse(rows: np.ndarray, d: np.ndarray, scale: Scale = COST_SCALE, shared: bool = True) -> SparseFit:
    """The spike-and-slab posterior for an orthogonal design (X'X = m I): the least-squares estimate b_j is
    N(delta_j, sigma^2 / m) independently given sigma, and the residual sum of squares carries sigma. With
    `shared`, an edit is either near zero (it does nothing) or in a slab around the shared mean mu (edits that do
    something share a direction), mu on a grid."""
    X = np.asarray(rows, dtype=float)
    d = np.asarray(d, dtype=float)
    m, p = X.shape
    b = X.T @ d / m
    rss = float(((d - X @ b) ** 2).sum())
    df = m - p
    pts = max(12, scale.points // 2)
    sig = np.geomspace(*scale.sigma, pts)
    tau = np.geomspace(*scale.slab, pts)
    mus = MUS * scale.slab_prior if shared else np.zeros(1)
    S, T, P, U = np.meshgrid(sig, tau, PIS, mus, indexing="ij")
    s, t, pi, mu = S.ravel(), T.ravel(), P.ravel(), U.ravel()
    v = (s ** 2 / m)[:, None]  # (K, 1) the variance of each b_j
    var_c = np.stack([np.broadcast_to((KAPPA * t) ** 2, s.shape), t ** 2], axis=1)[:, None, :]  # (K, 1, 2)
    logprior_c = np.log(np.stack([1 - pi, pi], axis=1))[:, None, :]  # (K, 1, 2)
    centre = np.stack([np.zeros_like(mu), mu], axis=1)[:, None, :]  # (K, 1, 2) near zero at 0, the slab at mu
    dev = b[None, :, None] - centre  # (K, p, 2) each estimate's distance from each component's centre
    marg = logprior_c + _norm_logpdf(dev, v[..., None] + var_c)  # (K, p, 2)
    top = marg.max(axis=2, keepdims=True)
    log_edit = (top + np.log(np.exp(marg - top).sum(axis=2, keepdims=True)))[..., 0]  # (K, p)
    comp_w = np.exp(marg - log_edit[..., None])
    shrink = var_c / (var_c + v[..., None])  # (K, 1, 2)
    comp_mean = centre + shrink * dev
    comp_var = np.broadcast_to(shrink * v[..., None], comp_w.shape)
    # the density of d: the p estimates (z = sqrt(m) b, so log p(z) = log p(b) - log(m) / 2) and the residual
    loglik = (log_edit.sum(axis=1) - 0.5 * p * math.log(m) - 0.5 * df * np.log(2 * math.pi * s ** 2)
              - rss / (2 * s ** 2))
    logprior = np.log(t) - np.log1p((t / scale.slab_prior) ** 2)  # half-Cauchy on the slab, times tau on a log grid
    if shared:
        logprior = logprior - 0.5 * (mu / scale.slab_prior) ** 2
    logw = loglik + logprior
    w = np.exp(logw - logw.max())
    keep = w > 1e-9 * w.max()
    w = w[keep] / w[keep].sum()
    return SparseFit(comp_w=comp_w[keep], comp_mean=comp_mean[keep], comp_var=comp_var[keep], weights=w,
                     sigmas=s[keep], taus=t[keep], pis=pi[keep], mus=mu[keep], n=m, ceiling=bool(np.all(d == 0)),
                     evidence=_evidence(logw, logprior))


@dataclass
class MixFit:
    """Two views weighed by their evidence: a few edits matter (sparse) and all matter a little (dense)."""

    fits: list
    probs: np.ndarray  # each view's posterior probability, equal prior
    names: tuple = ("sparse", "dense")

    @property
    def n(self) -> int:
        return self.fits[0].n

    @property
    def ceiling(self) -> bool:
        return self.fits[0].ceiling

    @property
    def weights(self) -> np.ndarray:
        return np.concatenate([q * f.weights for q, f in zip(self.probs, self.fits)])

    @property
    def mean(self) -> np.ndarray:
        return sum(q * f.mean for q, f in zip(self.probs, self.fits))

    def var(self) -> np.ndarray:
        m = self.mean
        return sum(q * (f.var() + (f.mean - m) ** 2) for q, f in zip(self.probs, self.fits))

    def draws(self, n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, self.probs)
        parts = [f.draws(int(c), seed + 7919 * (i + 1)) for i, (f, c) in enumerate(zip(self.fits, counts)) if c]
        out = np.concatenate(parts)
        return out[rng.permutation(n)]

    def kg_sd(self) -> np.ndarray:
        v = self.var()
        sigma2 = sum(q * float(f.weights @ f.sigmas ** 2) for q, f in zip(self.probs, self.fits))
        return np.sqrt(v ** 2 / (v.sum() + sigma2))

    def scales(self) -> dict:
        out = {f"p_{name}": float(q) for name, q in zip(self.names, self.probs)}
        for name, f in zip(self.names, self.fits):
            out.update({f"{name}_{k}": v for k, v in f.scales().items()})
        return out


def interval(x: np.ndarray, level: float = 0.8) -> dict:
    """{"mean", "lo", "hi"}: the mean and the central `level` range of draws."""
    lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
    return {"mean": float(np.mean(x)), "lo": float(lo), "hi": float(hi)}


BEST_MIN_P = 0.6  # an edit joins the best subset at P(helps) of at least this: a coin-flip edit adds nothing


def best_subset(util: np.ndarray, min_p: float = BEST_MIN_P) -> dict:
    """The predicted-best subset from utility draws (draws by edits): every edit whose expected utility is above
    0 and whose P(helps) is at least `min_p`, so an edit with no effect stays off. Its predicted gain, in utility
    units, with an 80% range and P(gain > 0)."""
    util = np.asarray(util, dtype=float)
    on = [int(j) for j in np.flatnonzero((util.mean(axis=0) > 0) & ((util > 0).mean(axis=0) >= min_p))]
    gain = util[:, on].sum(axis=1) if on else np.zeros(util.shape[0])
    return {"on": on, "gain": {**interval(gain), "p_positive": float((gain > 0).mean())}}


def _f(z: np.ndarray) -> np.ndarray:
    """z Phi(z) + phi(z): the expected positive part of a standard normal shifted by z."""
    from scipy.stats import norm

    return z * norm.cdf(z) + norm.pdf(z)


def kg(mean: np.ndarray, sd: np.ndarray) -> float:
    """The knowledge gradient of choosing each edit on or off: sum_j sd_j f(-|mean_j| / sd_j), in the
    utility's units per future run."""
    mean, sd = np.asarray(mean, dtype=float), np.asarray(sd, dtype=float)
    ok = sd > 0
    return float((sd[ok] * _f(-np.abs(mean[ok]) / sd[ok])).sum())


def kg_per_run(fit) -> float:
    """One head's knowledge gradient (its own units per future run)."""
    return kg(fit.mean, fit.kg_sd())
