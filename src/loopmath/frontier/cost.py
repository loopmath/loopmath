"""The cost model behind the frontier: expected USD per run of each setting, and draws of each setting's mean cost.

`CostModel` is the one interface between the command and the cost prediction. The picker divides KG by
`per_run()` and `next` checks the budget with it; the lookahead worlds and `report` read `sample_mean()`.

- `StudyCost`: the study's cost head moved out of `Posterior` unchanged (plug-in sigma_s^2 shrunk to sigma0, exact
  Gaussian for mu, per-run mean `exp(mc + diag(Sc)/2 + sig2/2)`), so study-mode parity holds.
- `FrontierCost`: mean-preserving prior centred at `log(anchor) - (v + s2)/2`, empirical-Bayes widths;
  the default once its checks pass. Its first version reads the median in `per_run()`.

Settings are indexed 0 to S-1 in the frontier file's order; costs are USD per run.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CostModel(Protocol):
    def add(self, s: int, usd: float) -> None:
        """Record one run of setting `s` that cost `usd` (> 0). Takes effect at the next `fit()`."""

    def fit(self) -> None:
        """Refresh the posterior from every run added so far."""

    def refit(self) -> dict | None:
        """Empirical-Bayes widths, called with the step refit (every 50 trials). Returns a summary, or None when
        this model has nothing to refit yet."""

    def per_run(self) -> np.ndarray:
        """Expected USD per run, per setting (shape S): the KG divisor and the budget check."""

    def sample_mean(self, K: int, rng: np.random.Generator) -> np.ndarray:
        """K x S draws of each setting's mean cost per run, for the lookahead worlds and the report."""


class StudyCost:
    """Study Gaussian log-cost head, preserved for exact replay.

    Quality normals precede cost normals. Pool adjustment consumes quality
    normals, then cost normals, even when only its cost component is used.
    Extra methods support study replay; the public CostModel stays unchanged.
    """

    def __init__(self, prior: dict) -> None:
        self.prior = prior
        self.mc0, self.Sc0 = prior["mc0"], prior["Sc0"]
        self.Pc0 = np.linalg.inv(self.Sc0)
        self.Pc0mc0 = self.Pc0 @ self.mc0
        self.sigma0, self.nu0 = prior["sigma0"], prior["nu0"]
        self.S = len(self.mc0)
        self.nc = np.zeros(self.S)
        self.sx = np.zeros(self.S)
        self.sxx = np.zeros(self.S)
        self.csum = np.zeros(self.S)
        self.pool_N = None
        self.fit()


    def add(self, s: int, usd: float) -> None:
        if not np.isfinite(usd) or usd <= 0:
            raise ValueError("cost must be finite and positive")
        x = np.log(usd)
        self.nc[s] += 1
        self.sx[s] += x
        self.sxx[s] += x * x
        self.csum[s] += usd

    def fit(self):
        # cost: plug-in sigma_s^2 (inverse-gamma style shrink to sigma0), then exact Gaussian for mu (closed form)
        n = self.nc
        xbar = np.where(n > 0, self.sx / np.maximum(n, 1), 0.0)
        ss = np.maximum(self.sxx - n * xbar ** 2, 0.0)
        self.sig2 = (self.nu0 * self.sigma0 ** 2 + ss) / (self.nu0 + np.maximum(n - 1, 0))
        Pc = self.Pc0 + np.diag(n / self.sig2)
        self.Sc = np.linalg.inv(Pc)
        self.Sc = (self.Sc + self.Sc.T) / 2
        self.mc = self.Sc @ (self.Pc0mc0 + self.sx / self.sig2)
        self._Lc = None

    def refit(self) -> dict | None:
        from .model import fit_step_spreads, step_prior
        steps = self.prior.get("steps")
        if steps is None:
            return None
        xbar = np.where(self.nc > 0, self.sx / np.maximum(self.nc, 1), 0.0)
        spread, info, _ = fit_step_spreads(steps["c"], "c", (xbar, self.nc, self.sig2),
                                            getattr(self, "step_spread", None))
        self.step_spread = spread
        self.mc0, self.Sc0 = step_prior(steps["c"], spread)
        self.Pc0 = np.linalg.inv(self.Sc0)
        self.Pc0mc0 = self.Pc0 @ self.mc0
        self.fit()
        return info

    def per_run(self) -> np.ndarray:
        return np.exp(self.mc + np.diag(self.Sc) / 2 + self.sig2 / 2)

    def sample_mean(self, K: int, rng: np.random.Generator) -> np.ndarray:
        return np.exp(self.sample_latent(K, rng) + self.sig2 / 2)

    def sample_latent(self, K: int, rng: np.random.Generator) -> np.ndarray:
        if self._Lc is None:
            self._Lc = np.linalg.cholesky(self.Sc + 1e-10 * np.eye(self.S))
        return self.mc + rng.standard_normal((K, self.S)) @ self._Lc.T

    def set_pool(self, n_runs: np.ndarray) -> None:
        self.pool_N = np.asarray(n_runs, float)

    def pool_adjust(self, q: np.ndarray, M: np.ndarray, k: np.ndarray, n: np.ndarray,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.pool_N is None:
            return q, M
        N = self.pool_N
        rem = np.maximum(N - n, 0)
        zq = rng.standard_normal(q.shape)
        qp = (k + rem * q + np.sqrt(rem * q * (1 - q)) * zq) / N
        qp = np.clip(qp, 0.5 / N, 1 - 0.5 / N)
        c_sup = np.exp(M + self.sig2 / 2)
        sd_run = c_sup * np.sqrt(np.expm1(self.sig2))
        zc = rng.standard_normal(q.shape)
        m_rem = c_sup + sd_run / np.sqrt(np.maximum(rem, 1)) * zc
        cp = (self.csum + rem * np.maximum(m_rem, 1e-9)) / N
        return qp, np.log(cp) - self.sig2 / 2

def anchors(tb: np.ndarray, prices: np.ndarray | None = None, *, level: float | None = None,
            hint: float | np.ndarray | None = None, fallback: float = 1.0) -> np.ndarray:
    """Choose USD anchors: measured TB transfer, price ratio times a shared level, hint, fallback.

    Inputs are aligned with the frontier settings. Unknown TB values or effective
    token prices are NaN. ``prices`` may be relative rates or effective $/token
    rates; ``level`` then has the corresponding units. When omitted, the shared
    level is the geometric mean of TB/rate across matched settings. No target
    outcomes enter this calculation. The caller adjusts TB values and prices
    for effort and supplies wider covariance for settings without transfer data.
    """
    tb = np.asarray(tb, float)
    if tb.ndim != 1 or not len(tb):
        raise ValueError("TB anchors must be a nonempty vector")
    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError("fallback must be finite and positive")
    valid = np.isfinite(tb) & (tb > 0)
    out = np.full(tb.shape, float(fallback))
    if hint is not None:
        try:
            h = np.broadcast_to(np.asarray(hint, float), tb.shape)
        except ValueError as exc:
            raise ValueError("hint must be a scalar or a vector matching TB anchors") from exc
        out = np.where(np.isfinite(h) & (h > 0), h, out)
    if prices is not None:
        price = np.asarray(prices, float)
        if price.shape != tb.shape:
            raise ValueError("prices must match TB anchors")
        good = np.isfinite(price) & (price > 0)
        if level is None and np.any(good & valid):
            level = float(np.exp(np.mean(np.log(tb[good & valid]) - np.log(price[good & valid]))))
        if level is not None:
            if not np.isfinite(level) or level <= 0:
                raise ValueError("level must be finite and positive")
            out = np.where(good, price * level, out)
    elif level is not None and (not np.isfinite(level) or level <= 0):
        raise ValueError("level must be finite and positive")
    return np.where(valid, tb, out)


def _gauss_evidence(m0: np.ndarray, K: np.ndarray, xbar: np.ndarray,
                    nc: np.ndarray, sig2: np.ndarray) -> float:
    """Exact Gaussian evidence on observed settings, as in the study's gauss_evidence."""
    O = np.flatnonzero(nc > 0)
    C = K[np.ix_(O, O)] + np.diag(sig2[O] / nc[O])
    L = np.linalg.cholesky(C)
    y = np.linalg.solve(L, xbar[O] - m0[O])
    return float(-.5 * y @ y - np.log(np.diag(L)).sum())


class FrontierCost:
    """Mean-preserving Gaussian log-cost model, independent of the success model.

    ``anchor`` is each setting's expected USD per run before target observations.
    ``components`` optionally supplies named covariance matrices for the shared
    benchmark level, transfer errors and effort-step spreads. Their positive SD
    multipliers are fitted together by Gaussian evidence. A full ``covariance``
    can be supplied instead and is fitted as one component. Defaults share a
    benchmark level (SD .5) and independent transfer errors (SD .8).

    Run variance is shrunk to sigma0 using the study's sufficient statistics.
    Width refits rebuild the mean-preserving centre before fitting the posterior.
    """

    def __init__(self, anchor: np.ndarray, covariance: np.ndarray | None = None, *,
                 sigma0: float = .6, nu0: float = 5.0,
                 components: dict[str, np.ndarray] | None = None) -> None:
        self.anchor = np.asarray(anchor, float).copy()
        if self.anchor.ndim != 1 or not len(self.anchor) or not np.all(np.isfinite(self.anchor)) \
                or np.any(self.anchor <= 0):
            raise ValueError("anchor must be a nonempty vector of finite positive costs")
        self.S = len(self.anchor)
        if not np.isfinite(sigma0) or sigma0 <= 0 or not np.isfinite(nu0) or nu0 <= 0:
            raise ValueError("sigma0 and nu0 must be finite and positive")
        if covariance is not None and components is not None:
            raise ValueError("supply covariance or components, not both")
        if components is None:
            components = ({"transfer": covariance} if covariance is not None else
                          {"level": np.full((self.S, self.S), .25), "transfer": np.eye(self.S) * .64})
        if not components:
            raise ValueError("at least one covariance component is required")
        self.components = {str(k): np.asarray(v, float).copy() for k, v in components.items()}
        for v in self.components.values():
            if v.shape != (self.S, self.S) or not np.all(np.isfinite(v)) or not np.allclose(v, v.T):
                raise ValueError("covariance components must be finite symmetric S x S matrices")
            if np.linalg.eigvalsh(v).min() < -1e-9:
                raise ValueError("covariance components must be positive semidefinite")
        self.sigma0, self.nu0 = float(sigma0), float(nu0)
        self.widths = np.ones(len(self.components))
        self.nc = np.zeros(self.S)
        self.sx = np.zeros(self.S)
        self.sxx = np.zeros(self.S)
        self.csum = np.zeros(self.S)
        self._set_widths(self.widths)
        self.fit()

    def _prior(self, widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        K = sum(w * w * c for w, c in zip(widths, self.components.values())) + np.eye(self.S) * 1e-9
        return np.log(self.anchor) - (np.diag(K) + self.sigma0 ** 2) / 2, K

    def _set_widths(self, widths: np.ndarray) -> None:
        self.widths = np.asarray(widths, float).copy()
        self.mc0, self.Sc0 = self._prior(self.widths)

    def add(self, s: int, usd: float) -> None:
        if not isinstance(s, (int, np.integer)) or isinstance(s, (bool, np.bool_)) or not 0 <= s < self.S:
            raise ValueError("setting index is out of range")
        if not np.isfinite(usd) or usd <= 0:
            raise ValueError("run cost must be finite and positive")
        x = float(np.log(usd))
        self.nc[s] += 1
        self.sx[s] += x
        self.sxx[s] += x * x
        self.csum[s] += usd

    def _statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Log means and centered sums of squares, including fractional evidence.

        A discounted observation keeps its value: only its precision changes.
        Use one as the denominator for empty cells, never for positive counts.
        Integer-count sufficient statistics retain their original arithmetic.
        """
        xbar = self.sx / np.where(self.nc > 0, self.nc, 1)
        return xbar, np.maximum(self.sxx - self.nc * xbar ** 2, 0)

    def fit(self) -> None:
        xbar, ss = self._statistics()
        self.sig2 = (self.nu0 * self.sigma0 ** 2 + ss) / (self.nu0 + np.maximum(self.nc - 1, 0))
        O = np.flatnonzero(self.nc > 0)
        if not len(O):
            self.mc, self.Sc = self.mc0.copy(), self.Sc0.copy()
            return
        # Condition on observed means directly. Inverting a nearly singular
        # shared-level prior loses precision when its kernels have low rank.
        cross = self.Sc0[:, O]
        C = self.Sc0[np.ix_(O, O)] + np.diag(self.sig2[O] / self.nc[O])
        solved = np.linalg.solve(C, np.column_stack((xbar[O] - self.mc0[O], cross.T)))
        self.mc = self.mc0 + cross @ solved[:, 0]
        self.Sc = self.Sc0 - cross @ solved[:, 1:]
        self.Sc = (self.Sc + self.Sc.T) / 2

    def refit(self) -> dict | None:
        """Fit all cost covariance components jointly; leave the prior alone before data."""
        if not self.nc.any():
            return None
        from scipy.optimize import minimize

        self.fit()
        xbar, _ = self._statistics()

        def objective(x):
            m, K = self._prior(np.exp(x))
            return -_gauss_evidence(m, K, xbar, self.nc, self.sig2) + .5 * float(x @ x)

        result = minimize(objective, np.log(self.widths), method="L-BFGS-B",
                          bounds=[(np.log(.1), np.log(4.))] * len(self.widths),
                          options={"maxiter": 80})
        if np.isfinite(result.fun) and result.fun <= objective(np.log(self.widths)):
            self._set_widths(np.exp(result.x))
        self.fit()
        return {"runs": int(self.nc.sum()), "widths": dict(zip(self.components, self.widths.tolist())),
                "log_evidence": _gauss_evidence(self.mc0, self.Sc0, xbar, self.nc, self.sig2),
                "converged": bool(result.success)}

    def per_run(self) -> np.ndarray:
        """Posterior expected cost per run, the KG divisor and budget estimate."""
        return np.exp(self.mc + np.diag(self.Sc) / 2 + self.sig2 / 2)

    def sample_mean(self, K: int, rng: np.random.Generator) -> np.ndarray:
        """Draw each setting's mean cost; do not add future-run observation noise."""
        M = self.mc + rng.standard_normal((K, self.S)) @ np.linalg.cholesky(self.Sc).T
        return np.exp(M + self.sig2 / 2)
