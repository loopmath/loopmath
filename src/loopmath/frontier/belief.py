"""What `adaptive-blocks` believes: the package's own success and cost models, held apart and read for planning.

Success model (`task-laplace`): `TaskPosterior` (2PL per task, learned effort steps, Laplace over abilities and
task difficulties) with an inert cost head, so that it stands alone. Read as:

  p()        the posterior mean pass chance of every pair: expit integrated over the Laplace Gaussian of the pair's
             logit (Gauss-Hermite, 32 nodes). Deterministic.
  q()        p() averaged with the task weights.
  normal()   mu = q(); Sigma = J Cz J', Cz the Laplace covariance of (ability, difficulty), J the expected slope of
             q: J[s, s] = sum_t w_t a_t E[p_st (1 - p_st)], J[s, S + t] = -w_t a_t E[p_st (1 - p_st)].
  lookahead  the Gaussian update of (ability, difficulty) for planned runs, read through the same J. A run of
             (s, t) is the linear observation y = E[p] + a_t E[p(1-p)] (d - E d) + e, d = ability - difficulty,
             Var(e) = E[p](1 - E[p]) - (a_t E[p(1-p)])^2 Var(d), so one run has the predictive variance p(1-p).

Cost model (`frontier-separate`): `FrontierCost` from `commands.frontier_cost`, fed and refitted apart from the
success model. c() is the posterior expected mean cost per run; c_interval is exact for the lognormal mean.

Schedule. Each run is added and the success posterior fitted at once. At the end of a block: if `refit_every` or
more runs came in since the last full refit, a full refit (effort-step spreads, 2PL slopes and tau, cost widths);
the cost head is fitted; the misfit monitor closes its epoch. `refit()` is the same full refit on demand; `report`
calls it on a copy, so a report never changes what `next` does.

Ported from the design study behind `adaptive-blocks` (its success and cost heads and its refit schedule),
arithmetic unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .misfit import Monitor
from .model import TaskPosterior

GH_NODES = 32
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(GH_NODES)
_GH_W = _GH_W / _GH_W.sum()
REFIT_EVERY = 20


class InertCost:
    """A cost head that learns nothing, so that TaskPosterior stands alone as a success model."""

    def __init__(self, S: int):
        self.S = S
        self.mc, self.Sc, self.sig2 = np.zeros(S), np.eye(S), np.ones(S)

    def add(self, s, usd):
        pass

    def fit(self):
        pass

    def refit(self):
        return None

    def per_run(self):
        return np.ones(self.S)

    def sample_mean(self, K, rng):
        raise NotImplementedError("the success model has no cost")


class AdaptiveBelief:
    """The success posterior `post`, the cost head `cost` and the misfit `monitor`, fed run by run."""

    def __init__(self, prior, weights, cost, models, *, tau: float = 1.5, refit_every: int = REFIT_EVERY):
        W = np.asarray(weights, float)
        S, T = W.shape
        self.post = TaskPosterior(prior, W, tau=tau, cost=InertCost(S))
        self.cost = cost
        self.monitor = Monitor(list(models), T)
        self.refit_every = int(refit_every)
        self.n = np.zeros((S, T), dtype=int)
        self.n_runs = 0
        self.last_full = 0
        self.refits = []
        self._cache = {}

    @property
    def S(self):
        return self.post.S_

    @property
    def T(self):
        return self.post.T

    # ---- feeding

    def add(self, s, t, y, usd, probe=None):
        """One recorded run. probe=p_pred for a run whose task was drawn at equal chance: the monitor scores it."""
        s, t, y = int(s), int(t), int(y)
        if probe is not None:
            self.monitor.probe(s, t, y, probe)
        self.post.add(s, y, 1.0, t)
        self.post.fit()
        self.cost.add(s, float(usd))
        self.n[s, t] += 1
        self.n_runs += 1
        self._cache = {}

    def block_end(self):
        """A block has run to its end: refit everything if due, fit the cost head, close the misfit epoch."""
        if self.n_runs - self.last_full >= self.refit_every:
            self.refit()
        self.cost.fit()
        self.monitor.tick()

    def refit(self):
        """Full refit of both models from every run taken in; nothing to do without new runs."""
        if self.n_runs == self.last_full:                      # no run since the last refit (or none at all)
            return
        post, before = self.post, self.post.n_full
        out = dict(draws=self.n_runs, steps=post.refit_steps(), tasks=post.refit(min_runs=0))
        if post.n_full == before:
            post._ready = False
            post.fit()
        self.cost.fit()
        out["c"] = self.cost.refit()
        self.cost.fit()
        self.refits.append(out)
        self.last_full = self.n_runs
        self._cache = {}

    # ---- reading the success model

    def _mom(self):
        """Per pair: E[p], E[p(1-p)] and the variance of d = ability - difficulty, under the Laplace Gaussian."""
        if "mom" not in self._cache:
            post = self.post
            S = post.S_
            z, a, Cz = post.z, post.a, post.Cz
            dg = np.diag(Cz)
            vd = np.maximum(dg[:S, None] + dg[None, S:] - 2 * Cz[:S, S:], 0.0)
            m = a[None, :] * (z[:S, None] - z[None, S:])
            e = m[:, :, None] + (np.abs(a)[None, :] * np.sqrt(vd))[:, :, None] * _GH_X
            p = 1 / (1 + np.exp(-e))
            self._cache["mom"] = (p @ _GH_W, (p * (1 - p)) @ _GH_W, vd)
        return self._cache["mom"]

    def _lin(self):
        """J Cz (S, S + T) and Sigma = J Cz J' (S, S)."""
        if "lin" not in self._cache:
            post = self.post
            S = post.S_
            _, beta, _ = self._mom()
            d = post.W * beta * post.a[None, :]
            J = np.zeros((S, S + post.T))
            J[np.arange(S), np.arange(S)] = d.sum(1)
            J[:, S:] = -d
            G = J @ post.Cz
            Sig = G @ J.T
            self._cache["lin"] = (G, (Sig + Sig.T) / 2)
        return self._cache["lin"]

    def p(self):
        return self._mom()[0]

    def q(self):
        return (self.post.W * self._mom()[0]).sum(1)

    def normal(self):
        return self.q(), self._lin()[1].copy()

    def q_sd(self):
        return np.sqrt(np.maximum(np.diag(self._lin()[1]), 0.0))

    def lookahead(self, s, tasks):
        """(shift (S,), var_after (S,)) for one run of setting s on each of `tasks` (repeats allowed)."""
        post = self.post
        S = post.S_
        s = int(s)
        G, Sig = self._lin()
        var = np.diag(Sig).copy()
        tt, cnt = np.unique(np.asarray(tasks, dtype=int), return_counts=True)
        pbar, beta, vd = self._mom()
        slope = beta[s, tt] * post.a[tt]                       # d E[y] / d d, the expected slope
        resid = np.maximum(pbar[s, tt] * (1 - pbar[s, tt]) - slope ** 2 * vd[s, tt], 1e-300)
        w = cnt * slope ** 2 / resid                           # information of the planned runs on d_t
        keep = w > 1e-14
        if not keep.any():
            return np.zeros(S), var
        tt, w = tt[keep], w[keep]
        Cz = post.Cz
        X = G[:, [s]] - G[:, S + tt]                           # J Cz U, U the columns e_s - e_(S+t)
        UCU = Cz[s, s] - Cz[s, S + tt][None, :] - Cz[S + tt, s][:, None] + Cz[np.ix_(S + tt, S + tt)]
        Y = np.linalg.solve(UCU + np.diag(1.0 / w), X.T)       # (m, S)
        col = X @ Y[:, s]                                      # D[:, s], D = Sigma - Sigma_after
        dss = float(col[s])
        shift = col / np.sqrt(dss) if dss > 0 else np.zeros(S)
        return shift, np.maximum(var - np.einsum("sm,ms->s", X, Y), 0.0)

    def lookahead_pairs(self, s, tasks):
        """(shift (n, S), var_after (n, S)): row i is lookahead(s, [tasks[i]]), all rows in one pass."""
        post = self.post
        S = post.S_
        s = int(s)
        tasks = np.asarray(tasks, dtype=int)
        G, Sig = self._lin()
        var = np.diag(Sig)
        if not len(tasks):
            return np.zeros((0, S)), np.zeros((0, S))
        pbar, beta, vd = self._mom()
        slope = beta[s, tasks] * post.a[tasks]
        resid = np.maximum(pbar[s, tasks] * (1 - pbar[s, tasks]) - slope ** 2 * vd[s, tasks], 1e-300)
        w = slope ** 2 / resid
        keep = w > 1e-14
        Cz = post.Cz
        X = G[:, [s]] - G[:, S + tasks]                        # (S, n)
        ucu = Cz[s, s] - Cz[s, S + tasks] - Cz[S + tasks, s] + Cz[S + tasks, S + tasks]
        with np.errstate(divide="ignore"):
            Y = np.where(keep, X / (ucu + 1.0 / w), 0.0)       # (S, n)
        dss = X[s] * Y[s]
        ok = keep & (dss > 0)
        shift = np.where(ok, X * Y[s] / np.sqrt(np.where(ok, dss, 1.0)), 0.0)
        return shift.T.copy(), np.maximum(var[:, None] - X * Y, 0.0).T.copy()

    def misfit(self):
        return self.monitor.misfit()

    # ---- reading the cost model

    def c(self):
        return np.asarray(self.cost.per_run(), float)

    def c_pair(self):
        return np.repeat(self.c()[:, None], self.T, axis=1)

    def c_interval(self, level):
        """Equal-tailed interval of each setting's mean cost per run (exact for the lognormal mean)."""
        h = self.cost
        z = norm.ppf(0.5 + level / 2)
        sd = np.sqrt(np.maximum(np.diag(h.Sc), 0.0))
        return np.exp(h.mc - z * sd + h.sig2 / 2), np.exp(h.mc + z * sd + h.sig2 / 2)

    def p_fit(self, B):
        """(J, S): the chance that each setting's mean cost per run is at most each budget in B."""
        h = self.cost
        sd = np.sqrt(np.maximum(np.diag(h.Sc), 0.0))
        x = (np.log(np.asarray(B, float))[:, None] - h.sig2[None, :] / 2 - h.mc[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, norm.cdf(x / np.where(sd > 0, sd, 1.0)), (x >= 0).astype(float))

    def worlds(self):
        """A TaskPosterior view whose cost is the cost head, for the shipped sampling and check code."""
        view = object.__new__(TaskPosterior)
        view.__dict__.update(self.post.__dict__)
        view.cost = self.cost
        view.mc, view.Sc, view.sig2 = self.cost.mc, self.cost.Sc, self.cost.sig2
        view._Lq = view._Lc = view._Lz = None
        return view
