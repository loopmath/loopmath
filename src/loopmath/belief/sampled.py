"""Full Bayes for the success head: NUTS over the effects and the log scales together (`belief.success_head =
"sampled"`).

The default head keeps a Gaussian at the mode for the effects given its plugged-in scales, and integrating the
scales by importance sampling with that Gaussian at each one keeps it too. Near the ceiling, with q < 1, the
likelihood flattens for large effects: an effect's posterior is right skewed, its mode below its mean, and the
Gaussian at the mode cuts the right tail. On paired stores with a true effect of 1.0 log-odds (100 pairs, seeds 1
to 6) this head puts the mean of the means at 0.83 and 5 of 6 80% ranges on the truth (0.64 and 13 of 20 over seeds
1 to 20); the Gaussian at the mode, 0.64 to 0.70 and 3 of 6, with the scale integrated or plugged in (plugged in,
0.57 and 10 of 20).

- `Joint`: the log posterior of x = (v, log phi) and its gradient, each node partly non-centred: b = s^w v with
  v ~ N(0, s^(2 - 2w)), s the node's scale (its group's phi, or the fixed sd), log phi ~ N(log default_phi,
  HYPER_SD^2); the head's likelihood and its factor terms (`Afs`, `cf`) as `LogisticHead` has them. w = 1 is the
  non-centred form (b = s u, u ~ N(0, 1)), w = 0 the centred one (b ~ N(0, s^2)). Any w is the same posterior of
  (b, log phi), in other coordinates. Under `param="partial"` each node's w is 1 / (1 + s^2 tau), tau the node's
  own precision from the rows and factor terms at the EB mode (its precision's diagonal less 1 / s^2): the optimal
  weight of the normal model's partially non-centred form (Papaspiliopoulos, Roberts and Skold 2007), used here as a
  rule of thumb for the scale form (b = s^w v, as in Gorinova, Moore and Hoffman 2020). A node the rows pin down is
  near centred, a node the prior holds near non-centred. With every node non-centred (`param="noncentred"`), the
  curvature in u of a pinned node grows as s^2: a trajectory that moves its scale up meets a stiffness the adapted
  step never saw. On a 218-node store and on simulations of 1,200 runs the non-centred form diverged 1 to 5 times a
  fit at a target acceptance of 0.9 (still 1 or 2 at 0.95), and the partial one did not.
- `NUTS`: the no-U-turn sampler with multinomial trajectory sampling and Stan's windowed adaptation (75 fast, slow
  windows doubling from 25, 50 fast) of a diagonal metric, or of a dense one (`metric="dense"`) from the first window
  with at least twice as many draws as dimensions (a shorter window's covariance is singular; it sets the diagonal).
- `rhat`, `ess_bulk` and `ess_tail`: rank-normalized split R-hat, bulk and tail effective sample sizes (Vehtari et
  al. 2021), as arviz computes them, so a fit records its own convergence without arviz.
- `SampledLogisticHead`: 4 chains of 1,000 warmup and 1,000 draws from the EB mode; N_DRAWS of the draws, evenly
  spaced, stored with the fit (integrated.py; `D`, each draw's effects; `S`, each draw's scales), and a
  HeadFit whose mean is the stored draws' mean and whose precision is the Laplace one at the scales' root mean
  square, the one Gaussian the look-ahead reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import fft, linalg, stats

from .forest import HYPER_SD
from .gaussian import N_DRAWS, HeadFit, LogisticHead

CHAINS = 4
WARMUP = 1000  # per chain
DRAWS = 1000  # per chain; every (CHAINS x DRAWS / N_DRAWS)th draw is stored, the diagnostics read them all (100 per
# chain left R-hat at 1.05 on a 40-pair store, 500 left it at 1.011 to 1.012 on a paired and a chain store)
MAX_DEPTH = 10
METRIC = "diag"
PARAM = "partial"  # each node's non-centring weight from the EB mode (module docstring); "noncentred": every w 1
DELTA = 0.9  # target acceptance
RHAT_MAX = 1.01  # a fit above it, or with a divergence or a bulk or tail ESS under ESS_MIN, says so in its info
ESS_MIN = 400.0  # 100 per chain (Vehtari et al. 2021)


class Joint:
    """log p(v, log phi | z) and its gradient for a LogisticHead, b = s^w v (module docstring); `w` per node,
    1 (non-centred) when not given. A fixed node's scale is fixed, so its w is 1."""

    def __init__(self, head: LogisticHead, w: np.ndarray | None = None):
        self.head = head
        prior = head.prior
        self.p, self.k = head.p, len(prior.groups)
        self.grouped = prior.group_idx >= 0
        self.gidx = np.where(self.grouped, prior.group_idx, 0)
        self.fixed_sd = np.sqrt(np.where(self.grouped, 1.0, prior.fixed_var))
        self.th0 = np.log(prior.default_phi)
        self.dim = self.p + self.k
        self.w = np.ones(self.p) if w is None else np.where(self.grouped, np.clip(w, 0.0, 1.0), 1.0)

    def scales(self, th: np.ndarray) -> np.ndarray:
        s = self.fixed_sd.copy()
        s[self.grouped] = np.exp(th)[self.gidx[self.grouped]]
        return s

    def b_of(self, x: np.ndarray) -> np.ndarray:
        return self.scales(x[self.p:]) ** self.w * x[:self.p]

    def v_of(self, b: np.ndarray, th: np.ndarray) -> np.ndarray:
        return b / self.scales(th) ** self.w

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        h = self.head
        v, th = x[:self.p], x[self.p:]
        ls = np.log(self.scales(th))
        sw = np.exp(self.w * ls)  # s^w
        prec = np.exp(-2.0 * (1.0 - self.w) * ls)  # v's prior precision, s^(2w - 2)
        b = sw * v
        lik, g_eta, _ = h._loglik_terms(h.X @ b)
        Ab = h.Afs @ b
        lp = lik - 0.5 * float(b @ Ab) + float(h.cf @ b) - 0.5 * float(v @ (prec * v))
        lp -= float(np.sum(((1.0 - self.w) * ls)[self.grouped]))  # v's normalizing constant, a function of phi
        lp -= 0.5 * float(np.sum((th - self.th0) ** 2)) / HYPER_SD ** 2
        gb = np.asarray(h.X.T @ g_eta).ravel() - Ab + h.cf
        gv = sw * gb - prec * v
        dth = self.w * b * gb + (1.0 - self.w) * (prec * v * v - 1.0)  # each node's part of d lp / d log phi
        gth = np.bincount(self.gidx[self.grouped], weights=dth[self.grouped], minlength=self.k)
        gth -= (th - self.th0) / HYPER_SD ** 2
        return lp, np.concatenate([gv, gth])


@dataclass
class _Tree:
    xl: np.ndarray
    rl: np.ndarray
    gl: np.ndarray
    xr: np.ndarray
    rr: np.ndarray
    gr: np.ndarray
    xs: np.ndarray  # the tree's sample, with its log density and gradient
    lps: float
    gs: np.ndarray
    logw: float
    rho: np.ndarray  # the sum of the momenta
    n: int
    acc: float
    stop: bool
    div: bool


class NUTS:
    """Multinomial NUTS for a log density `f(x) -> (lp, grad)`. `inv_m` is the inverse metric: its diagonal, or
    under `metric="dense"` once a window is long enough, the matrix (`_L` its Cholesky factor)."""

    def __init__(self, f, dim: int, rng: np.random.Generator, *, max_depth: int = MAX_DEPTH, delta: float = DELTA,
                 metric: str = METRIC):
        self.f, self.dim, self.rng = f, dim, rng
        self.max_depth, self.delta, self.metric = max_depth, delta, metric
        self.inv_m = np.ones(dim)
        self._L: np.ndarray | None = None

    def _sharp(self, r):
        return self.inv_m * r if self._L is None else self.inv_m @ r

    def _kinetic(self, r) -> float:
        return 0.5 * float(np.sum(self.inv_m * r * r)) if self._L is None else 0.5 * float(r @ (self.inv_m @ r))

    def _momentum(self):
        z = self.rng.standard_normal(self.dim)
        if self._L is None:
            return z / np.sqrt(self.inv_m)
        return linalg.solve_triangular(self._L, z, lower=True, trans="T")  # covariance inv_m^-1

    def _leap(self, x, r, g, eps):
        r = r + 0.5 * eps * g
        x = x + eps * self.inv_m * r if self._L is None else x + eps * (self.inv_m @ r)
        lp, g = self.f(x)
        r = r + 0.5 * eps * g
        return x, r, g, lp

    def _crit(self, a, b, rho) -> bool:
        return float(self._sharp(a) @ rho) > 0.0 and float(self._sharp(b) @ rho) > 0.0

    def _turned(self, L: _Tree, R: _Tree, rho) -> bool:
        # the generalized no-U-turn check on the whole tree and on the two joins (Stan's extra checks)
        return not (self._crit(L.rl, R.rr, rho) and self._crit(L.rl, R.rl, L.rho + R.rl)
                    and self._crit(L.rr, R.rr, R.rho + L.rr))

    def _build(self, x, r, g, v, depth, eps, H0) -> _Tree:
        if depth == 0:
            x1, r1, g1, lp1 = self._leap(x, r, g, v * eps)
            hh = lp1 - self._kinetic(r1)
            if not np.isfinite(hh):
                hh = -np.inf
            div = (H0 - hh) > 1000.0
            acc = float(min(1.0, math.exp(hh - H0))) if np.isfinite(hh) else 0.0
            return _Tree(x1, r1, g1, x1, r1, g1, x1, lp1, g1, hh - H0, r1, 1, acc, div, div)
        a = self._build(x, r, g, v, depth - 1, eps, H0)
        if a.stop:
            return a
        e = (a.xr, a.rr, a.gr) if v == 1 else (a.xl, a.rl, a.gl)
        b = self._build(*e, v, depth - 1, eps, H0)
        n, acc = a.n + b.n, a.acc + b.acc
        if b.stop:
            b.n, b.acc = n, acc
            return b
        logw = float(np.logaddexp(a.logw, b.logw))
        xs, lps, gs = (b.xs, b.lps, b.gs) if math.log(self.rng.random()) < b.logw - logw else (a.xs, a.lps, a.gs)
        L, R = (a, b) if v == 1 else (b, a)
        rho = L.rho + R.rho
        return _Tree(L.xl, L.rl, L.gl, R.xr, R.rr, R.gr, xs, lps, gs, logw, rho, n, acc, self._turned(L, R, rho), False)

    def step(self, x, lp, g, eps):
        r0 = self._momentum()
        H0 = lp - self._kinetic(r0)
        t = _Tree(x, r0, g, x, r0, g, x, lp, g, 0.0, r0.copy(), 0, 0.0, False, False)
        n = acc = 0
        div, depth = False, 0
        for depth in range(self.max_depth):
            v = 1 if self.rng.random() < 0.5 else -1
            e = (t.xr, t.rr, t.gr) if v == 1 else (t.xl, t.rl, t.gl)
            s = self._build(*e, v, depth, eps, H0)
            n, acc = n + s.n, acc + s.acc
            div = div or s.div
            if s.stop:
                break
            if math.log(self.rng.random()) < s.logw - t.logw:  # biased progressive sampling
                t.xs, t.lps, t.gs = s.xs, s.lps, s.gs
            L, R = (t, s) if v == 1 else (s, t)
            rho = L.rho + R.rho
            t = _Tree(L.xl, L.rl, L.gl, R.xr, R.rr, R.gr, t.xs, t.lps, t.gs, float(np.logaddexp(t.logw, s.logw)),
                      rho, 0, 0.0, False, False)
            if self._turned(L, R, rho):
                break
        return t.xs, t.lps, t.gs, {"accept": acc / max(n, 1), "leapfrog": n, "depth": depth + 1, "div": div}

    def _find_eps(self, x, lp, g) -> float:
        eps = 0.1
        r = self._momentum()
        H0 = lp - self._kinetic(r)

        def logratio(e):
            _, r1, _, lp1 = self._leap(x, r, g, e)
            hh = lp1 - self._kinetic(r1)
            return hh - H0 if np.isfinite(hh) else -np.inf

        up = logratio(eps) > math.log(0.5)
        for _ in range(50):
            eps = eps * 2.0 if up else eps / 2.0
            if (logratio(eps) > math.log(0.5)) != up:
                break
        return eps

    def run(self, x0: np.ndarray, *, warmup: int, draws: int):
        """(draws, dim) samples after `warmup` adapting iterations, the per-draw stats and the final step size."""
        x = x0.copy()
        lp, g = self.f(x)
        eps = self._find_eps(x, lp, g)
        init, term, base = 75, 50, 25  # Stan's windows
        ends, start, w = [], init, base
        while start + w < warmup - term:
            end = start + w
            if end + 2 * w > warmup - term:
                end = warmup - term
            ends.append(end)
            start, w = end, 2 * w
        mu, hbar, leb, t = math.log(10 * eps), 0.0, 0.0, 0
        buf, out, stats_ = [], [], []
        for it in range(warmup + draws):
            x, lp, g, st = self.step(x, lp, g, eps)
            if it < warmup:  # dual averaging of the step size toward `delta` acceptance
                t += 1
                eta = 1.0 / (t + 10)
                hbar = (1 - eta) * hbar + eta * (self.delta - st["accept"])
                le = mu - math.sqrt(t) / 0.05 * hbar
                wk = t ** -0.75
                leb = wk * le + (1 - wk) * leb
                eps = math.exp(le)
                if init <= it < (ends[-1] if ends else 0):
                    buf.append(x.copy())
                if ends and it + 1 in ends:  # a slow window ends: the metric from its draws, shrunk as Stan does
                    B = np.array(buf)
                    nb = len(B)
                    if self.metric == "dense" and nb >= 2 * self.dim:
                        self.inv_m = (nb / (nb + 5.0)) * np.cov(B, rowvar=False) + 1e-3 * (5.0 / (nb + 5.0)) * np.eye(
                            self.dim)
                        self._L = np.linalg.cholesky(self.inv_m)
                    else:
                        self.inv_m = (nb / (nb + 5.0)) * B.var(axis=0) + 1e-3 * (5.0 / (nb + 5.0))
                    buf = []
                    eps = self._find_eps(x, lp, g)
                    mu, hbar, leb, t = math.log(10 * eps), 0.0, 0.0, 0
                if it + 1 == warmup:
                    eps = math.exp(leb)
            else:
                out.append(x.copy())
                stats_.append(st)
        return np.array(out), stats_, eps


# ---------------------------------------------------------------- convergence (as arviz)


def _split(x: np.ndarray) -> np.ndarray:
    half = x.shape[1] // 2
    return np.vstack([x[:, :half], x[:, -half:]])


def _z(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x, method="average").reshape(x.shape)
    return stats.norm.ppf((r - 0.375) / (x.size + 0.25))


def _rhat_plain(x: np.ndarray) -> float:
    n = x.shape[1]
    between = n * np.var(x.mean(axis=1), ddof=1)
    within = np.mean(np.var(x, axis=1, ddof=1))
    return float(np.sqrt((between / within + n - 1) / n))


def rhat(x: np.ndarray) -> float:
    """Rank-normalized split R-hat of one quantity's (chains, draws), the larger of the bulk and the folded one."""
    s = _split(np.asarray(x, float))
    if np.ptp(s) < np.finfo(float).resolution:
        return 1.0
    return max(_rhat_plain(_z(s)), _rhat_plain(_z(np.abs(s - np.median(s)))))


def _autocov(x: np.ndarray) -> np.ndarray:
    n = x.shape[1]
    m = fft.next_fast_len(2 * n)
    x = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(x, n=m, axis=1)
    return np.fft.irfft(f * np.conjugate(f), n=m, axis=1)[:, :n] / n


def _ess(x: np.ndarray) -> float:
    """Geyer's initial monotone sequence over the chains' mean autocorrelation (arviz's `_ess`)."""
    c, n = x.shape
    if np.ptp(x) < np.finfo(float).resolution:
        return float(x.size)
    acov = _autocov(x)
    mean_var = np.mean(acov[:, 0]) * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n
    if c > 1:
        var_plus += np.var(x.mean(axis=1), ddof=1)
    rho = np.zeros(n)
    even, odd = 1.0, 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho[0], rho[1] = even, odd
    t = 1
    while t < n - 3 and even + odd > 0.0:
        even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if even + odd >= 0:
            rho[t + 1], rho[t + 2] = even, odd
        t += 2
    max_t = t - 2
    if even > 0:
        rho[max_t + 1] = even
    t = 1
    while t <= max_t - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = rho[t + 2] = (rho[t - 1] + rho[t]) / 2.0
        t += 2
    tau = -1.0 + 2.0 * float(np.sum(rho[:max_t + 1])) + float(np.sum(rho[max_t + 1:max_t + 2]))
    return float(x.size / max(tau, 1.0 / math.log10(x.size)))


def ess_bulk(x: np.ndarray) -> float:
    """Bulk effective sample size of one quantity's (chains, draws): rank-normalized, split."""
    return _ess(_z(_split(np.asarray(x, float))))


def ess_tail(x: np.ndarray) -> float:
    """Tail effective sample size of one quantity's (chains, draws): the smaller of the split ESS of being at most
    the 5% and the 95% quantile."""
    x = np.asarray(x, float)
    return min(_ess(_split((x <= np.quantile(x, p)).astype(float))) for p in (0.05, 0.95))


# ---------------------------------------------------------------- the head


class SampledLogisticHead:
    """The success head by NUTS (module docstring). `fit(seed)` returns a HeadFit, `n` draws (p, n) and each draw's
    scales (groups, n)."""

    def __init__(self, head: LogisticHead):
        self.head = head

    def fit(self, *, seed: int, eb: HeadFit | None = None, n: int = N_DRAWS, chains: int = CHAINS,
            warmup: int = WARMUP, draws: int = DRAWS, metric: str = METRIC,
            param: str = PARAM) -> tuple[HeadFit, np.ndarray, np.ndarray]:
        h = self.head
        if eb is None:
            eb = h.fit(eb=True)
        f = Joint(h)
        if h.n == 0:
            return eb, eb.draws(seed, n), np.repeat(eb.phi[:, None], n, axis=1)
        th = np.log(np.maximum(eb.phi, 1e-3)) if f.k else np.zeros(0)
        if param == "partial" and eb.precision is not None:
            s0 = f.scales(th)
            tau = np.maximum(np.asarray(eb.precision.diagonal()).ravel() - 1.0 / s0 ** 2, 0.0)
            f = Joint(h, 1.0 / (1.0 + s0 ** 2 * tau))
        elif param not in ("partial", "noncentred"):
            raise ValueError(f"param {param!r}: partial or noncentred")
        self.joint = f  # its `w` and `b_of`, for checks that read the chains
        u0 = f.v_of(eb.mean, th)
        rng = np.random.default_rng(seed)
        X, sts = [], []
        for c in range(chains):  # each chain from the EB mode, jittered
            x0 = np.concatenate([u0 + 0.3 * rng.standard_normal(f.p), th + 0.3 * rng.standard_normal(f.k)])
            sampler = NUTS(f, f.dim, np.random.default_rng([seed, c]), metric=metric)
            out, st, eps = sampler.run(x0, warmup=warmup, draws=draws)
            X.append(out)
            sts.append((st, eps))
        Xs = np.stack(X)  # (chains, draws, dim)
        B = np.stack([np.stack([f.b_of(x) for x in chain]) for chain in Xs])  # (chains, draws, p), the effects
        TH = Xs[:, :, f.p:]
        r = [rhat(B[:, :, j]) for j in range(f.p)] + [rhat(TH[:, :, i]) for i in range(f.k)]
        e = [ess_bulk(B[:, :, j]) for j in range(f.p)] + [ess_bulk(TH[:, :, i]) for i in range(f.k)]
        et = [ess_tail(B[:, :, j]) for j in range(f.p)] + [ess_tail(TH[:, :, i]) for i in range(f.k)]
        div = int(sum(s_["div"] for st, _ in sts for s_ in st))
        hits = int(sum(s_["depth"] >= MAX_DEPTH for st, _ in sts for s_ in st))
        keep = np.linspace(0, chains * draws - 1, n).round().astype(int)  # evenly spaced over the chains
        D = B.reshape(-1, f.p).T[:, keep]
        S = np.exp(TH.reshape(-1, f.k)).T[:, keep] if f.k else np.zeros((0, n))
        phi_rms = np.sqrt(np.mean(S ** 2, axis=1)) if f.k else eb.phi
        lam = h.prior.lam(phi_rms)
        _, _, _, _, weights = h.mode(lam)
        max_rhat, min_ess, min_tail = float(np.nanmax(r)), float(np.nanmin(e)), float(np.nanmin(et))
        wg = f.w[f.grouped]
        info = {**eb.info, "chains": chains, "warmup": warmup, "draws_per_chain": draws, "metric": metric,
                "param": param, "w_quantiles": [round(float(x), 3) for x in np.quantile(wg, [0, 0.5, 1])] if len(wg)
                else [],
                "max_rhat": round(max_rhat, 4),
                "min_ess_bulk": round(min_ess, 1), "min_ess_tail": round(min_tail, 1), "divergences": div,
                "max_depth_hits": hits,
                "step_size": [round(float(s_), 5) for _, s_ in sts],
                "mean_leapfrog": round(float(np.mean([s_["leapfrog"] for st, _ in sts for s_ in st])), 1),
                "converged_draws": bool(max_rhat <= RHAT_MAX and min(min_ess, min_tail) >= ESS_MIN and div == 0)}
        res = HeadFit("logistic", D.mean(axis=1), phi_rms, 1.0, eb.log_evidence, eb.iterations, info,
                      h._hessian(weights, 1.0 / lam))
        return res, D, S
