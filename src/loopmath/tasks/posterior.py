"""The candidate's ability and its suite score S, from its own runs and the history fit.

S is the mean over the suite of the candidate's chance of acceptance per task, p_t. Per task,
p_t = expit(a_t (theta - b_t) + e_t), with e_t ~ N(0, a_t^2 se_t^2 + sigma_u^2): the task fit's own uncertainty
plus the arm-by-task part. theta lives on a grid; e_t is integrated by Gauss-Hermite. Given theta, the tasks are
independent, and S given theta is taken as normal with the summed moments (a central limit over tasks). The
posterior of S is the grid mixture of those normals; its ranges are the mixture's quantiles on [0, 1].

With `arm_slope_sd` above 0 the candidate also has its own slope: p_t = expit(lam a_t (theta - b_t) + e_t), with
log lam ~ N(0, arm_slope_sd^2) on Gauss-Hermite nodes. A candidate whose chance of acceptance rises with task
ease more slowly (or faster) than the history arms' does is then in the model: its range widens until its runs
rule that out, and the expected-loss rule values the easy and hard tasks that tell. At 0 (lam = 1) every number
is as without it.

Left out of the ranges: the slopes' uncertainty, the correlation between task difficulties, the uncertainty of
tau, sigma_u and the ability prior, and task-fit error shared with other arms.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit, logsumexp, ndtr, ndtri

from .fit import TaskFit, gauss_hermite

GRID = 161
NODES = 16
SLOPES = 7
SPAN = 6.0
LEVELS = (0.8, 0.95)


def _quantiles(weights, mu, sd, probs):
    """Quantiles of the mixture sum_g w_g N(mu_g, sd_g^2) censored to [0, 1], by bisection on its CDF.

    S lies in [0, 1], but a normal component can put mass outside. Censoring piles the mass below 0 at 0 and
    the mass above 1 at 1: a quantile at or below the mass under 0 is 0, one at or above the mass under 1 is 1,
    and any other has its root inside (0, 1), where the censored CDF is the mixture's own, continuous and
    increasing. So the bisection on [0, 1] always ends on the censored quantile."""
    sd = np.maximum(sd, 1e-9)
    probs = np.asarray(probs, float)
    lo, hi = np.zeros_like(probs), np.ones_like(probs)
    cdf = lambda x: np.sum(weights[:, None] * ndtr((x[None, :] - mu[:, None]) / sd[:, None]), axis=0)
    below0, above1 = cdf(np.zeros(1))[0], cdf(np.ones(1))[0]
    for _ in range(60):
        mid = (lo + hi) / 2
        c = cdf(mid)
        lo, hi = np.where(c < probs, mid, lo), np.where(c < probs, hi, mid)
    out = (lo + hi) / 2
    out = np.where(probs <= below0, 0.0, out)
    return np.where(probs >= above1, 1.0, out)


class CandidatePosterior:
    """The candidate's posterior given `results`: per task, (accepted runs, runs); accepted may be fractional."""

    def __init__(self, fit: TaskFit, results: dict[str, tuple[float, float]], *, prior_mean: float | None = None,
                 prior_sd: float | None = None, grid: int = GRID, nodes: int = NODES, arm_slope_sd: float = 0.0,
                 slopes: int = SLOPES):
        self.fit = fit
        T = len(fit.tasks)
        idx = {t: j for j, t in enumerate(fit.tasks)}
        s, n = np.zeros(T), np.zeros(T)
        for task, (k, m) in results.items():
            if task not in idx:
                raise ValueError(f"task {task!r} is not in the suite")
            s[idx[task]] += k
            n[idx[task]] += m
        self.s, self.n = s, n
        mu = fit.theta_mean if prior_mean is None else prior_mean
        sd = fit.theta_sd if prior_sd is None else prior_sd
        self.prior_mean, self.prior_sd = float(mu), float(sd)
        th = mu + sd * np.linspace(-SPAN, SPAN, grid)
        self.arm_slope_sd = float(arm_slope_sd)
        if arm_slope_sd > 0:  # the grid is (theta, lam) pairs, flattened
            z, wz = gauss_hermite(slopes)
            self.theta, self.lam = np.repeat(th, slopes), np.tile(np.exp(arm_slope_sd * z), grid)
            self.logprior = -0.5 * ((self.theta - mu) / sd) ** 2 + np.tile(np.log(wz), grid)
        else:
            self.theta, self.lam = th, np.ones(grid)
            self.logprior = -0.5 * ((self.theta - mu) / sd) ** 2
        x, w = gauss_hermite(nodes)
        self.v = np.sqrt(fit.a**2 * fit.b_se**2 + fit.sigma_u**2)
        eta = (self.lam[:, None, None] * fit.a[None, :, None] * (self.theta[:, None, None] - fit.b[None, :, None])
               + self.v[None, :, None] * x)
        lw = s[None, :, None] * log_expit(eta) + (n - s)[None, :, None] * log_expit(-eta) + np.log(w)
        L = logsumexp(lw, axis=-1)  # per grid point and task: log P(the task's runs | theta)
        r = np.exp(lw - L[..., None])  # per node: its posterior weight
        p = expit(eta)
        self.m = np.sum(r * p, axis=-1)  # E[p_t | theta, runs]
        self.q = np.sum(r * p * p, axis=-1)  # E[p_t^2 | theta, runs]
        self.c = np.sum(r * p * p * p, axis=-1)  # E[p_t^3 | theta, runs], for the next run's update
        self.L = L  # per grid point and task: the log-likelihood of the task's runs (for batch replays)
        logpost = self.logprior + L.sum(1)
        self.pi = np.exp(logpost - logsumexp(logpost))
        self.T = T

    # ------------------------------------------------------------------ the score
    @property
    def sessions(self) -> int:
        return int(round(float(self.n.sum())))

    @property
    def passes(self) -> float:
        return float(self.s.sum())

    def seen(self) -> np.ndarray:
        return self.n > 0

    def _score_parts(self):
        mu = self.m.mean(1)
        var = np.clip(self.q - self.m**2, 0.0, None).sum(1) / self.T**2
        return mu, var

    def score(self, levels=LEVELS) -> dict:
        """The posterior mean of S, its SD, and the central ranges at `levels`."""
        mu, var = self._score_parts()
        mean = float(np.sum(self.pi * mu))
        total = float(np.sum(self.pi * (var + mu**2)) - mean**2)
        out = {"mean": mean, "sd": float(np.sqrt(max(total, 0.0)))}
        probs = [p for lv in levels for p in ((1 - lv) / 2, (1 + lv) / 2)]
        qs = _quantiles(self.pi, mu, np.sqrt(var), probs)
        for i, lv in enumerate(levels):
            out[f"range{int(round(lv * 100))}"] = [float(qs[2 * i]), float(qs[2 * i + 1])]
        return out

    def p_above(self, bar) -> float:
        """P(S > bar): a fixed number, or another arm's posterior (independent given the task fit)."""
        mu, var = self._score_parts()
        if isinstance(bar, CandidatePosterior):
            mu2, var2 = bar._score_parts()
            d = mu[:, None] - mu2[None, :]
            sd = np.sqrt(var[:, None] + var2[None, :])
            w = self.pi[:, None] * bar.pi[None, :]
            return float(np.sum(w * ndtr(d / np.maximum(sd, 1e-9))))
        return float(np.sum(self.pi * ndtr((mu - float(bar)) / np.maximum(np.sqrt(var), 1e-9))))

    def full_run_halfwidth(self, level: float = 0.8) -> float:
        """The half-width a full one-run-per-task suite would give at `level`: z sqrt(sum p(1 - p)) / T, with p
        the candidate's posterior rates."""
        ep = np.sum(self.pi[:, None] * self.m, axis=0)
        ep2 = np.sum(self.pi[:, None] * self.q, axis=0)
        return float(ndtri((1 + level) / 2) * np.sqrt(np.clip(ep - ep2, 0, None).sum()) / self.T)

    def rates(self) -> np.ndarray:
        """Per task: the candidate's posterior mean rate."""
        return np.sum(self.pi[:, None] * self.m, axis=0)

    def ability(self) -> tuple[float, float]:
        mean = float(np.sum(self.pi * self.theta))
        return mean, float(np.sqrt(max(np.sum(self.pi * self.theta**2) - mean**2, 0.0)))

    # ------------------------------------------------------------------ values of one more run
    def var_cut(self) -> np.ndarray:
        """Per task: the expected cut in Var(S) from one more run on it (the expected-loss rule)."""
        mu, var = self._score_parts()
        now = float(np.sum(self.pi * (var + mu**2)) - np.sum(self.pi * mu) ** 2)
        m, q, c, T = self.m, self.q, self.c, self.T
        m_ = np.clip(m, 1e-12, 1 - 1e-12)
        p1 = np.sum(self.pi[:, None] * m, axis=0)  # P(next run accepted)
        out = np.zeros(T)
        for y, (pw, mt, qt) in {1: (m_, q / m_, c / m_), 0: (1 - m_, (m - q) / (1 - m_), (q - c) / (1 - m_))}.items():
            wy = self.pi[:, None] * pw
            wy = wy / np.maximum(wy.sum(0, keepdims=True), 1e-300)
            mu_y = mu[:, None] + (mt - m) / T
            var_y = var[:, None] + (np.clip(qt - mt**2, 0, None) - np.clip(q - m**2, 0, None)) / T**2
            v_y = np.sum(wy * (var_y + mu_y**2), axis=0) - np.sum(wy * mu_y, axis=0) ** 2
            out += (p1 if y == 1 else 1 - p1) * v_y
        return np.maximum(now - out, 0.0)

    def information(self, at_prior: bool = False) -> np.ndarray:
        """Per task: Fisher information about theta of one run, at the posterior mean ability (classic adaptive
        testing), or averaged over the ability prior (`at_prior`, for one fixed order)."""
        f = self.fit
        k = 1 / np.sqrt(1 + np.pi * self.v**2 / 8)  # the logistic-normal slope shrink
        a = f.a * k
        if at_prior:
            w = np.exp(self.logprior - self.logprior.max())
            w /= w.sum()
            al = self.lam[:, None] * a[None, :]
            p = expit(al * (self.theta[:, None] - f.b[None, :]))
            return np.sum(w[:, None] * al**2 * p * (1 - p), axis=0)
        th = self.ability()[0]
        al = a * float(np.sum(self.pi * self.lam))
        p = expit(al * (th - f.b))
        return al**2 * p * (1 - p)

    def prior_rates(self) -> np.ndarray:
        """Per task: the rate predicted for a new arm from the history alone (prior ability, no runs)."""
        f = self.fit
        w = np.exp(self.logprior - self.logprior.max())
        w /= w.sum()
        x, gw = gauss_hermite(NODES)
        eta = (self.lam[:, None, None] * f.a[None, :, None] * (self.theta[:, None, None] - f.b[None, :, None])
               + self.v[None, :, None] * x)
        return np.sum(w[:, None] * np.sum(gw * expit(eta), axis=-1), axis=0)
