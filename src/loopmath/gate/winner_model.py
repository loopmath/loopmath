"""The posterior of a search loop's winning gain: the screen's shrinkage, applied to one search.

N candidates were scored and the loop kept the best. Each score is y_i = theta_i + e_i with a known standard
error s_i, and the parent's is y_p = theta_p + e_p. The candidates' true scores are theta_i = theta_p + mu + eta_i:
a shared mean child effect mu ~ N(0, s_mu^2) plus each candidate's own deviation eta_i, independent under one
of two views, as `loopmath screen` has them:

- ridge: eta_i ~ N(0, tau^2);
- sparse: eta_i ~ (1 - pi) N(0, (kappa tau)^2) + pi N(0, tau^2), most candidates near the centre and a share
  pi spread out.

tau has a half-Cauchy prior on a log grid and pi a uniform prior on a grid. Neither is plugged in (a
plugged-in scale undercovers), and `auto` weighs the two views by their marginal likelihoods.

Candidates whose scores are unknown, but known to be at most some bound (a best-so-far curve, or "best of 64"
with only the best printed), enter through the normal CDF: k log F(bound).

With nu = theta_p + mu, the candidates are independent given nu, and theta_p integrates out in closed form.
So the posterior is a sum over a grid of (view, tau, pi, nu). At each grid point and mixture part the
winner's gain D = theta_w - theta_p is normal, so the posterior of D is a finite mixture of normals, with
exact means and tail probabilities.

Under an exchangeable prior, the posterior of a candidate picked by any rule on these scores is the same as
for one fixed in advance (Dawid 1994, "Selection paradoxes of Bayesian inference"). So this shrinkage is
the winner's-curse correction, provided that every score the selection used is in the model. Unreported
candidates are in it, as censored scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtr

KAPPA = 0.05  # the near-centre part's scale, as a fraction of tau (the screen's)
PIS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])  # the share of candidates spread out, uniform prior (the screen's)
DEFAULT_POINTS = (24, 61)  # tau points, and nu points in the second pass
VIEWS = ("auto", "ridge", "sparse")
_LOG2PI = math.log(2.0 * math.pi)
_BLOCK = 4_000_000  # array cells per block of candidates


@dataclass(frozen=True)
class Prior:
    """The hyperprior's scales, in score units: half-Cauchy(0, tau_prior) on tau over [tau_lo, tau_hi], and
    N(0, s_mu^2) on the mean child effect."""

    tau_prior: float
    s_mu: float
    tau_lo: float
    tau_hi: float


def default_prior(scale: str, ses: np.ndarray, values: np.ndarray) -> Prior:
    """On fractions, tau_prior = s_mu = 0.1. On any other scale, both are 3 times the median standard error.
    The tau grid runs from 1% of the median error (no spread at all, in effect) to well past the scores' range."""
    s_med = float(np.median(ses))
    tau_prior = s_mu = 0.1 if scale == "fraction" else 3.0 * s_med
    spread = float(np.ptp(values)) if len(values) > 1 else 0.0
    return Prior(tau_prior=tau_prior, s_mu=s_mu, tau_lo=0.01 * s_med,
                 tau_hi=max(5.0 * spread, 10.0 * s_med, 3.0 * tau_prior))


@dataclass(frozen=True)
class Data:
    """One search, higher is better: the parent, the scored candidates (the winner at index `w`), and the
    censored groups (`count` candidates each, every one at most `bound`)."""

    y_p: float
    s_p: float
    y: np.ndarray
    s: np.ndarray
    w: int
    bound: np.ndarray
    count: np.ndarray
    s_c: np.ndarray


@dataclass
class Mixture:
    """The posterior of the winner's gain: weights, means and variances of normal parts, plus what the fit
    read about the hyperparameters."""

    w: np.ndarray
    m: np.ndarray
    v: np.ndarray
    log_evidence: float
    hyper: dict

    @property
    def mean(self) -> float:
        return float(self.w @ self.m)

    def sf(self, x: float) -> float:
        """P(D > x)."""
        return float(self.w @ ndtr((self.m - x) / np.sqrt(self.v)))

    def quantiles(self, qs) -> list[float]:
        qs = np.asarray(qs, dtype=float)
        sd = np.sqrt(self.v)
        lo = np.full(qs.shape, float(np.min(self.m - 9.0 * sd)))
        hi = np.full(qs.shape, float(np.max(self.m + 9.0 * sd)))
        for _ in range(40):  # 2^-40 of an 18 SD bracket
            mid = 0.5 * (lo + hi)
            cdf = ndtr((mid[:, None] - self.m[None, :]) / sd[None, :]) @ self.w
            below = cdf < qs
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return [float(x) for x in 0.5 * (lo + hi)]

    def sample(self, rng: np.random.Generator, k: int) -> np.ndarray:
        idx = rng.choice(len(self.w), size=k, p=self.w / self.w.sum())
        return rng.normal(self.m[idx], np.sqrt(self.v[idx]))


def _log_tau_prior(tau: np.ndarray, prior: Prior) -> np.ndarray:
    """The half-Cauchy's log density on log tau, unnormalized."""
    return np.log(tau) - np.log1p((tau / prior.tau_prior) ** 2)


def _log_tau_norm(prior: Prior) -> float:
    """The log of that density's integral over [tau_lo, tau_hi], so a zoomed grid keeps the full range's prior
    (and the evidence stays comparable across spreads)."""
    g = np.linspace(math.log(prior.tau_lo), math.log(prior.tau_hi), 4001)
    return float(logsumexp(_log_tau_prior(np.exp(g), prior)) + math.log(g[1] - g[0]))


def _hyper_grid(prior: Prior, view: str, tau: np.ndarray, log_norm: float):
    """The (view, tau, pi) rows with their log prior mass, and each row's two mixture parts (log weight,
    variance). `tau` is a log-spaced grid inside [tau_lo, tau_hi]."""
    n_tau = len(tau)
    # a point's mass is the density on log tau times the step in log tau, so the log mass adds log(step)
    lp_tau = _log_tau_prior(tau, prior) + math.log(math.log(tau[1] / tau[0])) - log_norm
    views = ("ridge", "sparse") if view == "auto" else (view,)
    kind, taus, pis, lp = [], [], [], []
    for v in views:
        if v == "ridge":
            kind.append(np.zeros(n_tau, int))
            taus.append(tau)
            pis.append(np.ones(n_tau))
            lp.append(lp_tau - math.log(len(views)))
        else:
            for p in PIS:
                kind.append(np.ones(n_tau, int))
                taus.append(tau)
                pis.append(np.full(n_tau, p))
                lp.append(lp_tau - math.log(len(views)) - math.log(len(PIS)))
    kind, taus, pis, lp = (np.concatenate(a) for a in (kind, taus, pis, lp))
    sparse = kind == 1
    with np.errstate(divide="ignore"):
        logw = np.stack([np.where(sparse, np.log1p(-pis), -np.inf), np.where(sparse, np.log(pis), 0.0)], axis=1)
    var = np.stack([np.where(sparse, (KAPPA * taus) ** 2, taus ** 2), taus ** 2], axis=1)
    return kind, taus, pis, lp, logw, var


def _loglik(nu: np.ndarray, d: Data, taus: np.ndarray, views: tuple[str, ...], s_mu: float) -> np.ndarray:
    """log p(scores | view, tau, pi, nu) on the grid: (rows, nu points), rows in `_hyper_grid`'s order.

    For each tau the two normal parts' log densities are computed once: the ridge row uses the wide part, and
    each sparse row mixes the narrow and the wide part with its own pi."""
    T, K = len(taus), len(nu)
    rows = {v: [] for v in views}
    lp_pi = [(math.log1p(-p), math.log(p)) for p in PIS]
    out = {"ridge": np.zeros((T, K)), "sparse": [np.zeros((T, K)) for _ in PIS]}
    v_wide, v_narrow = taus ** 2, (KAPPA * taus) ** 2
    groups = [(d.y, d.s, None)] + ([(d.bound, d.s_c, d.count)] if len(d.bound) else [])
    block = max(1, _BLOCK // (T * K))
    for vals, ses, counts in groups:
        for i in range(0, len(vals), block):
            x, s2 = vals[i:i + block], ses[i:i + block] ** 2
            z = x[None, :] - nu[:, None]  # (K, b)
            parts = []
            for var in ((v_narrow, v_wide) if "sparse" in views else (v_wide,)):
                tot = var[:, None] + s2[None, :]  # (T, b)
                if counts is None:
                    parts.append(-0.5 * (_LOG2PI + np.log(tot))[:, None, :]
                                 - (z * z)[None, :, :] * (0.5 / tot)[:, None, :])  # (T, K, b)
                else:
                    parts.append(log_ndtr(z[None, :, :] / np.sqrt(tot)[:, None, :]))
            wide = parts[-1]
            weigh = (lambda a: a.sum(axis=2)) if counts is None else (lambda a: (a * counts[i:i + block]).sum(axis=2))
            if "ridge" in views:
                out["ridge"] += weigh(wide)
            if "sparse" in views:
                for j, (l0, l1) in enumerate(lp_pi):
                    out["sparse"][j] += weigh(np.logaddexp(l0 + parts[0], l1 + wide))
    pv = d.s_p ** 2 + s_mu ** 2  # theta_p integrated out: y_p | nu ~ N(nu, s_p^2 + s_mu^2)
    parent = (-0.5 * (_LOG2PI + math.log(pv) + (d.y_p - nu) ** 2 / pv))[None, :]
    for v in views:
        rows[v] = [out["ridge"]] if v == "ridge" else out["sparse"]
    return np.concatenate([r for v in views for r in rows[v]], axis=0) + parent


ZOOM_NU = 41  # nu points in each zoom pass
RESOLVED = 12  # a pass is fine enough once nu's posterior spans this many of its points


def _nu_range(d: Data, prior: Prior, lp: np.ndarray, taus: np.ndarray, views: tuple[str, ...],
              n: int, start: tuple[float, float] | None = None) -> np.ndarray:
    """Where nu's posterior lives: a wide pass (or `start`, the last fit's span), widened while mass sits at an
    edge and narrowed onto the mass until a pass resolves it; the final grid covers that span with n points."""
    if start is None:
        vals = np.concatenate([d.y, [d.y_p], d.bound])
        centre = float(np.median(np.concatenate([d.y, [d.y_p]])))
        half = 8.0 * max(float(np.ptp(vals)), float(np.max(d.s)), d.s_p, prior.s_mu)
        lo, hi = centre - half, centre + half
    else:
        lo, hi = start
    for _ in range(12):
        grid = np.linspace(lo, hi, ZOOM_NU)
        marg = logsumexp(_loglik(grid, d, taus, views, prior.s_mu) + lp[:, None], axis=0)
        keep = np.flatnonzero(marg - marg.max() > math.log(1e-12))
        step = grid[1] - grid[0]
        if keep[0] == 0 or keep[-1] == ZOOM_NU - 1:  # mass at an edge: widen around it
            width = hi - lo
            lo, hi = (lo - 2 * width, hi) if keep[0] == 0 else (lo, hi)
            lo, hi = (lo, hi + 2 * width) if keep[-1] == ZOOM_NU - 1 else (lo, hi)
            continue
        lo, hi = grid[keep[0]] - step, grid[keep[-1]] + step
        if len(keep) >= RESOLVED:
            break
    return np.linspace(lo, hi, n)


def fit(d: Data, *, scale: str, view: str = "auto", points: tuple[int, int] = DEFAULT_POINTS,
        prior: Prior | None = None) -> Mixture:
    """The posterior of the winner's gain D = theta_w - theta_p, as a mixture of normals."""
    if view not in VIEWS:
        raise ValueError(f"prior must be one of {', '.join(VIEWS)} (got {view!r})")
    n_tau, n_nu = points
    if prior is None:
        prior = default_prior(scale, np.concatenate([d.s, [d.s_p], d.s_c]),
                              np.concatenate([d.y, [d.y_p], d.bound]))
    views = ("ridge", "sparse") if view == "auto" else (view,)
    log_norm = _log_tau_norm(prior)
    tau, nu = np.geomspace(prior.tau_lo, prior.tau_hi, n_tau), None
    for _ in range(8):  # zoom the tau grid onto its posterior until a step is at most its SD on log tau
        kind, taus, pis, lp, logw, var = _hyper_grid(prior, view, tau, log_norm)
        nu = _nu_range(d, prior, lp, tau, views, n_nu, None if nu is None else (nu[0], nu[-1]))
        post = _loglik(nu, d, tau, views, prior.s_mu) + lp[:, None]  # (H, K)
        by_tau = logsumexp(logsumexp(post, axis=1).reshape(-1, n_tau), axis=0)
        lt, step = np.log(tau), math.log(tau[1] / tau[0])
        pt = np.exp(by_tau - logsumexp(by_tau))
        centre = float(pt @ lt)
        sd = math.sqrt(max(float(pt @ (lt - centre) ** 2), 0.0))
        if step <= sd:
            break
        half = 8.0 * max(sd, step / 2)
        lo, hi = max(centre - half, math.log(prior.tau_lo)), min(centre + half, math.log(prior.tau_hi))
        tau = np.exp(np.linspace(lo, hi, n_tau))
    log_z = float(logsumexp(post) + math.log(nu[1] - nu[0]))
    post = np.exp(post - logsumexp(post))

    # The winner's part, given (row, nu): which mixture part it is in, then normal.
    y_w, s_w = float(d.y[d.w]), float(d.s[d.w])
    tot = var[:, None, :] + s_w ** 2  # (H, 1, 2)
    z = y_w - nu[None, :, None]
    lr = logw[:, None, :] - 0.5 * (np.log(tot) + z * z / tot)
    resp = np.exp(lr - logsumexp(lr, axis=2, keepdims=True))  # (H, K, 2)
    b = var[:, None, :] / tot
    sp2, sm2 = d.s_p ** 2, prior.s_mu ** 2
    m_p = (d.y_p * sm2 + nu * sp2) / (sp2 + sm2)  # theta_p | nu, y_p
    v_p = sp2 * sm2 / (sp2 + sm2)
    mean = nu[None, :, None] + b * z - m_p[None, :, None]
    vv = np.broadcast_to(b * s_w ** 2 + v_p, mean.shape)
    weight = post[:, :, None] * resp
    flat = np.sort(weight, axis=None)
    cut = flat[np.searchsorted(np.cumsum(flat), 1e-12 * flat.sum(), side="right") - 1] if flat[0] < 1e-12 else 0.0
    keep = weight > cut  # the parts holding all but 1e-12 of the mass
    w = weight[keep]

    sparse = kind == 1
    row_w = post.sum(axis=1)
    nu_w = post.sum(axis=0)
    hyper = {
        "views": {"ridge": float(row_w[~sparse].sum()), "sparse": float(row_w[sparse].sum())},
        "tau": float(row_w @ taus),
        "pi": float(row_w[sparse] @ pis[sparse] / row_w[sparse].sum()) if row_w[sparse].sum() > 0 else None,
        "mu": float(nu_w @ (nu - m_p)),
        "theta_p": float(nu_w @ m_p),  # the parent's true score, posterior mean (higher-is-better space)
        "tau_prior": prior.tau_prior,
        "s_mu": prior.s_mu,
        "grid": {"tau": int(n_tau), "pi": int(len(PIS)), "nu": int(n_nu)},
    }
    return Mixture(w=w / w.sum(), m=mean[keep], v=vv[keep], log_evidence=log_z, hyper=hyper)


def combine(parts: list[Mixture], log_prior: np.ndarray) -> tuple[Mixture, np.ndarray]:
    """One mixture over several fits (a stated range of spreads), each weighed by its evidence times its prior.
    Returns the mixture and the weights."""
    lw = np.array([p.log_evidence for p in parts]) + np.asarray(log_prior, dtype=float)
    a = np.exp(lw - logsumexp(lw))
    w = np.concatenate([ai * p.w for ai, p in zip(a, parts)])
    m = np.concatenate([p.m for p in parts])
    v = np.concatenate([p.v for p in parts])
    hyper = dict(parts[int(np.argmax(a))].hyper)
    hyper["views"] = {k: float(sum(ai * p.hyper["views"][k] for ai, p in zip(a, parts))) for k in ("ridge", "sparse")}
    hyper["tau"] = float(sum(ai * p.hyper["tau"] for ai, p in zip(a, parts)))
    hyper["mu"] = float(sum(ai * p.hyper["mu"] for ai, p in zip(a, parts)))
    hyper["theta_p"] = float(sum(ai * p.hyper["theta_p"] for ai, p in zip(a, parts)))
    pis = [(ai, p.hyper["pi"]) for ai, p in zip(a, parts) if p.hyper["pi"] is not None]
    hyper["pi"] = float(sum(ai * x for ai, x in pis) / sum(ai for ai, _ in pis)) if pis else None
    return Mixture(w=w / w.sum(), m=m, v=v, log_evidence=float(logsumexp(lw)), hyper=hyper), a
