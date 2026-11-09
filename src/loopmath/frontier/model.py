"""Prior construction and Gaussian (Laplace) posterior for the DeepSWE frontier study.

Math: FRAMEWORK.md sections 2 to 4. In short:
  eta_s = logit quality of setting s, mu_s = location of its lognormal per-run cost, mean cost c_s = exp(mu_s + sigma_s^2 / 2).
  Option A: eta_s = a + b z_m + o_f + w_m + d_f(e) + u_s and mu_s = a_c + b_c v_m + o^c_f + w^c_m + d^c_f(e) + u^c_s,
            z_m and v_m are TB abilities (logit accuracy and log cost per run) from a Gaussian fit of TB config means
            whose likelihood is raised to the power alpha. The prior over (eta, mu) is the moment-matched Gaussian.
  Option B: A with b = b_c = 1 fixed.
  Option C: independent settings, centred on the model's TB value with n0 = alpha * N0 pseudo-runs, no sharing.
Posterior: Laplace for eta (Bernoulli counts), exact Gaussian for mu given plug-in sigma_s (lognormal per run).
"""
from dataclasses import dataclass, field, asdict

import numpy as np

from .cost import StudyCost
from loopmath.tasks import fit as task_fit

N_EFFORT = 5

def family(model):
    """Family pooling used by the frozen transfer prior."""
    for prefix, name in (("claude", "claude"), ("gpt-6", "gpt6"), ("gpt-5", "gpt5"),
                         ("gemini", "gemini"), ("deepseek", "deepseek"), ("glm", "glm"),
                         ("grok", "grok"), ("kimi", "kimi"), ("qwen", "qwen"), ("muse", "muse")):
        if str(model).startswith(prefix):
            return name
    return "other"

LOGIT_FLAT_VAR = np.pi ** 2 / 3  # variance of logit(U), U uniform


@dataclass
class ModelConfig:
    option: str = "A"
    alpha: float = 1.0
    # DeepSWE level and transfer (quality, logit scale)
    a0: float = 0.0
    sa: float = 1.5
    b0: float = 1.0
    sb: float = 0.5
    s_fam: float = 0.5
    tau_w: float = 0.5
    g_mu: float = 0.3
    g_sd: float = 0.3
    h_sd: float = 0.2
    tau_u: float = 0.3
    # cost (log USD per run)
    ac0: float = 0.0
    sac: float = 2.0
    bc0: float = 1.0
    sbc: float = 0.5
    sc_fam: float = 0.5
    tauc_w: float = 0.5
    gc_mu: float = 0.4
    gc_sd: float = 0.3
    hc_sd: float = 0.2
    tauc_u: float = 0.3
    nu0: float = 5.0
    sigma0: float = 1.05
    # TB fit
    tb_pop: float = 1.0
    tb_fam: float = 0.5
    tb_h: float = 0.5
    tb_e: float = 0.3
    tb_bench_sd: float = 3.0
    n0_C: float = 20.0
    shuffle_tb_seed: int = -1
    # effort steps (FRAMEWORK.md section 18): "fixed" is the original prior above (g, h); "learned" replaces it with a
    # common step mean mu (a coefficient, hyperprior N(mu0, mu_sd^2), learned in the posterior), per-rung deviations
    # g_j, family deviations h_fam,j and one per-model step deviation d_m, whose spreads are fitted by empirical Bayes
    effort_prior: str = "fixed"
    mu_step0: float = 0.1
    mu_step_sd: float = 0.5
    step_g0: float = 0.3
    step_h0: float = 0.2
    step_d0: float = 0.2
    muc_step0: float = 0.1
    muc_step_sd: float = 0.5
    stepc_g0: float = 0.3
    stepc_h0: float = 0.2
    stepc_d0: float = 0.2
    step_hyper_sd: float = 1.0      # sd of log spread around its starting value (the hyperprior)
    step_r_sd: float = 2.0          # sd of log r_m, a ladder model's own multiplier on d (hyperprior centred at 1);
                                    # at 1.0 the hyperprior, not the data, held Luna's r near 3 (effort-fix/scratch/rsd.out)

    def as_dict(self):
        return asdict(self)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def tb_abilities(tb, models, cfg, what="quality"):
    """Gaussian fit of TB config means; returns (mean, cov) of the ability of each name in `models`.

    y_c = bench_b + z_m + h_harness + e_c,  z_m ~ N(mu_fam, tb_fam^2), mu_fam ~ N(0, tb_pop^2), h ~ N(0, tb_h^2),
    e_c ~ N(0, tb_e^2 + sampling var). Each observation's precision is multiplied by alpha (power prior).
    """
    tb = tb.copy()
    if what == "quality":
        tb = tb[tb.acc.notna() & (tb.n > 0)]
        p = np.clip(tb.acc.values, 1e-3, 1 - 1e-3)
        y = _logit(p)
        var = 1.0 / (tb.n.values * p * (1 - p)) + cfg.tb_e ** 2
    else:
        tb = tb[tb.cost.notna() & (tb.cost > 0)]
        y = np.log(tb.cost.values)
        var = 0.05 + cfg.tb_e ** 2
        var = np.full(len(y), var)
    if cfg.shuffle_tb_seed >= 0 and len(tb):
        names = sorted(set(tb.model))
        perm = np.random.default_rng([cfg.shuffle_tb_seed, 31]).permutation(len(names))
        mp = {n: names[i] for n, i in zip(names, perm)}
        tb["model"] = tb.model.map(mp)
    all_models = sorted(set(models) | set(tb.model))
    fams = sorted({family(m) for m in all_models})
    benches = sorted(set(tb.bench)) or ["none"]
    harn = sorted(set(tb.harness.fillna("unknown")))
    ib = {b: i for i, b in enumerate(benches)}
    i_f = {f: len(benches) + i for i, f in enumerate(fams)}
    i_m = {m: len(benches) + len(fams) + i for i, m in enumerate(all_models)}
    i_h = {h: len(benches) + len(fams) + len(all_models) + i for i, h in enumerate(harn)}
    P = len(benches) + len(fams) + len(all_models) + len(harn)
    # prior precision: bench, family means, z_m | mu_fam (as a joint Gaussian), harness
    prec = np.zeros((P, P))
    for b in benches:
        prec[ib[b], ib[b]] = 1 / cfg.tb_bench_sd ** 2
    for f in fams:
        prec[i_f[f], i_f[f]] += 1 / cfg.tb_pop ** 2
    for m in all_models:
        i, j = i_m[m], i_f[family(m)]
        t = 1 / cfg.tb_fam ** 2
        prec[i, i] += t
        prec[j, j] += t
        prec[i, j] -= t
        prec[j, i] -= t
    for h in harn:
        prec[i_h[h], i_h[h]] = 1 / cfg.tb_h ** 2
    rhs = np.zeros(P)
    if cfg.alpha > 0:
        for (b, m, h), yy, vv in zip(zip(tb.bench, tb.model, tb.harness.fillna("unknown")), y, var):
            a = np.zeros(P)
            a[[ib[b], i_m[m], i_h[h]]] = 1.0
            w = cfg.alpha / vv
            prec += w * np.outer(a, a)
            rhs += w * a * yy
    cov = np.linalg.inv(prec)
    mean = cov @ rhs
    idx = [i_m[m] for m in models]
    return mean[idx], cov[np.ix_(idx, idx)]


def _components(pool):
    """Design pieces shared by quality and cost: family, model, effort-step incidence."""
    S = pool.S
    fams = sorted(set(pool.fam))
    models = sorted(set(pool.model))
    Ff = np.zeros((S, len(fams)))
    Fm = np.zeros((S, len(models)))
    E = np.zeros((S, N_EFFORT - 1))                      # generic steps j -> j+1, j = 0..3
    Ef = np.zeros((S, len(fams) * (N_EFFORT - 1)))       # family steps
    for s in range(S):
        f = fams.index(pool.fam[s])
        Ff[s, f] = 1
        Fm[s, models.index(pool.model[s])] = 1
        for j in range(pool.effort[s], N_EFFORT - 1):
            E[s, j] = -1
            Ef[s, f * (N_EFFORT - 1) + j] = -1
    return models, Ff, Fm, E, Ef


def build_prior(pool, tb, cfg):
    """Gaussian prior over (eta, mu) for the pool's settings. Returns dict m0, S0, mc0, Sc0, sigma0."""
    S = pool.S
    models, Ff, Fm, E, Ef = _components(pool)
    Mi = np.array([models.index(m) for m in pool.model])
    if cfg.option in ("A", "B"):
        zq, Vq = tb_abilities(tb, models, cfg, "quality")
        zc, Vc = tb_abilities(tb, models, cfg, "cost")
        if cfg.option == "A":
            b0, sb, bc0, sbc = cfg.b0, cfg.sb, cfg.bc0, cfg.sbc
        else:
            b0, sb, bc0, sbc = 1.0, 0.0, 1.0, 0.0
        out = {}
        steps = {}
        for key, z, V, a0, sa, bb, sbb, sf, tw, gm, gs, hs, tu in (
                ("q", zq, Vq, cfg.a0, cfg.sa, b0, sb, cfg.s_fam, cfg.tau_w, cfg.g_mu, cfg.g_sd, cfg.h_sd, cfg.tau_u),
                ("c", zc, Vc, cfg.ac0, cfg.sac, bc0, sbc, cfg.sc_fam, cfg.tauc_w, cfg.gc_mu, cfg.gc_sd, cfg.hc_sd, cfg.tauc_u)):
            zs = z[Mi]
            Vs = V[np.ix_(Mi, Mi)]
            if key == "c":
                # centre TB log cost so the DeepSWE level a_c carries the level and b_c only the spread
                zs = zs - zs.mean()
            if cfg.effort_prior == "learned":
                base_m = a0 + bb * zs
                base_S = (sa ** 2 * np.ones((S, S)) + sbb ** 2 * np.outer(zs, zs) + (bb ** 2 + sbb ** 2) * Vs
                          + sf ** 2 * Ff @ Ff.T + tw ** 2 * Fm @ Fm.T + tu ** 2 * np.eye(S))
                st = step_parts(base_m, base_S, E, Ef, Fm, cfg, key)
                steps[key] = st
                out[key] = step_prior(st, st["spread0"])
                continue
            m0 = a0 + bb * zs + E.sum(axis=1) * gm
            S0 = (sa ** 2 * np.ones((S, S)) + sbb ** 2 * np.outer(zs, zs) + (bb ** 2 + sbb ** 2) * Vs
                  + sf ** 2 * Ff @ Ff.T + tw ** 2 * Fm @ Fm.T + gs ** 2 * E @ E.T + hs ** 2 * Ef @ Ef.T
                  + tu ** 2 * np.eye(S))
            out[key] = (m0, S0)
        (m0, S0), (mc0, Sc0) = out["q"], out["c"]
        if steps:
            return dict(m0=m0, S0=S0, mc0=mc0, Sc0=Sc0, sigma0=cfg.sigma0, nu0=cfg.nu0, steps=steps)
    elif cfg.option == "C":
        n0 = cfg.alpha * cfg.n0_C
        # hand-fixed map: centre each TB config's logit accuracy (and log cost) on its benchmark's mean, then average
        # per model, so TB sets ranks, not the DeepSWE level, and TB 2.1 and 4.0 levels do not mix
        t = tb.dropna(subset=["acc"]).copy()
        t["lq"] = _logit(t.acc.values)
        t["lq"] -= t.groupby("bench").lq.transform("mean")
        q_by = t.groupby("model").lq.mean()
        tc = tb[tb.cost > 0].copy()
        tc["lc"] = np.log(tc.cost)
        tc["lc"] -= tc.groupby("bench").lc.transform("mean")
        c_by = tc.groupby("model").lc.mean()
        if cfg.shuffle_tb_seed >= 0:
            q_by = q_by.iloc[np.random.default_rng([cfg.shuffle_tb_seed, 31]).permutation(len(q_by))].set_axis(q_by.index)
            c_by = c_by.iloc[np.random.default_rng([cfg.shuffle_tb_seed, 37]).permutation(len(c_by))].set_axis(c_by.index)
        q_c, c_c = q_by.to_dict(), c_by.to_dict()
        m0 = np.zeros(S)
        v0 = np.full(S, LOGIT_FLAT_VAR)
        mc0 = np.full(S, cfg.ac0)
        vc0 = np.full(S, cfg.sac ** 2)
        for s, m in enumerate(pool.model):
            if m in q_c and n0 > 0:
                p = 1 / (1 + np.exp(-q_c[m]))
                m0[s] = q_c[m]
                v0[s] = 1 / (1 / LOGIT_FLAT_VAR + n0 * p * (1 - p))
            if m in c_c and n0 > 0:
                mc0[s] = cfg.ac0 + c_c[m]
                vc0[s] = 1 / (1 / cfg.sac ** 2 + n0 / cfg.sigma0 ** 2)
        S0, Sc0 = np.diag(v0), np.diag(vc0)
    else:
        raise ValueError(f"unknown option {cfg.option}")
    return dict(m0=m0, S0=S0, mc0=mc0, Sc0=Sc0, sigma0=cfg.sigma0, nu0=cfg.nu0)


def step_parts(base_m, base_S, E, Ef, Fm, cfg, key):
    """Pieces of the learned effort prior for one head (FRAMEWORK.md section 18).

    eta_s = base_s + n_s mu + sum over rungs j climbed down of (g_j + h_fam,j + d_m) with n_s = -(rungs below max):
    mu ~ N(mu0, mu_sd^2) is a coefficient (its variance is in the base), g_j ~ N(0, g^2), h ~ N(0, h^2) and the
    per-model step deviation d_m ~ N(0, (d r_m)^2) on every rung of model m. r_m = 1 for a model with one effort in
    the pool; a model with two or more efforts (a ladder model) gets its own multiplier r_m, so the spread of d is a
    scale mixture with a heavier tail than one Gaussian and a steep model is not held to its family's curve.
    Returns what step_prior needs for any spreads (g, h, d, r_1, ..., r_M).
    """
    xs = E.sum(axis=1)                                   # minus the number of rungs below max
    Dm = Fm * xs[:, None]
    ladder = np.flatnonzero((Fm.sum(axis=0) >= 2) & (np.abs(Dm).sum(axis=0) > 0))
    single = np.setdiff1d(np.arange(Fm.shape[1]), ladder)
    if key == "q":
        mu0, mu_sd, s0 = cfg.mu_step0, cfg.mu_step_sd, (cfg.step_g0, cfg.step_h0, cfg.step_d0)
    else:
        mu0, mu_sd, s0 = cfg.muc_step0, cfg.muc_step_sd, (cfg.stepc_g0, cfg.stepc_h0, cfg.stepc_d0)
    return dict(m0=base_m + mu0 * xs, base=base_S + mu_sd ** 2 * np.outer(xs, xs),
                mats=(E @ E.T, Ef @ Ef.T, Dm[:, single] @ Dm[:, single].T),
                model_mats=[np.outer(Dm[:, m], Dm[:, m]) for m in ladder], ladder=ladder,
                spread0=np.concatenate([s0, np.ones(len(ladder))]), xs=xs, mu0=mu0, mu_sd=mu_sd,
                hyper_sd=cfg.step_hyper_sd, r_sd=cfg.step_r_sd)


def step_prior(st, spread):
    """(m0, S0) of one head at step spreads (g, h, d, r_1, ..., r_M)."""
    g, h, d = spread[:3]
    S0 = st["base"] + g ** 2 * st["mats"][0] + h ** 2 * st["mats"][1] + d ** 2 * st["mats"][2]
    for r, M in zip(spread[3:], st["model_mats"]):
        S0 = S0 + (d * r) ** 2 * M
    return st["m0"], S0


def laplace_evidence(m0, K, k, n, f=None):
    """Laplace approximation of log p(k | m0, K) for k_s ~ Binomial(n_s, expit(eta_s)), eta ~ N(m0, K), without the
    binomial coefficients (GPML algorithms 3.1 and 3.2 with a mean). Settings with n_s = 0 add nothing.
    Returns (logZ, mode f, a = K^-1 (f - m0), Cholesky L of B = I + sW K sW, sW)."""
    S = len(m0)
    f = m0.copy() if f is None else f.copy()

    def psi(f, a):
        p = 1 / (1 + np.exp(-f))
        return -0.5 * a @ (f - m0) + np.sum(k * np.log(p + 1e-300) + (n - k) * np.log1p(-p + 1e-300))

    a = np.linalg.solve(K + 1e-9 * np.eye(S), f - m0)
    old = psi(f, a)
    for _ in range(100):
        p = 1 / (1 + np.exp(-f))
        W = n * p * (1 - p)
        sW = np.sqrt(W)
        L = np.linalg.cholesky(np.eye(S) + sW[:, None] * K * sW[None, :])
        b = W * (f - m0) + (k - n * p)
        a_new = b - sW * _cho(L, sW * (K @ b))
        da = a_new - a
        for _ in range(30):                              # step halving on the Laplace objective
            a_try = a + da
            f_try = m0 + K @ a_try
            new = psi(f_try, a_try)
            if new >= old - 1e-10:
                break
            da = da / 2
        f, a = f_try, a_try
        if abs(new - old) < 1e-10:
            old = new
            break
        old = new
    p = 1 / (1 + np.exp(-f))
    sW = np.sqrt(n * p * (1 - p))
    L = np.linalg.cholesky(np.eye(S) + sW[:, None] * K * sW[None, :])
    return psi(f, a) - np.log(np.diag(L)).sum(), f, a, L, sW


def _cho(L, b):
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def gauss_evidence(m0, K, xbar, nc, sig2):
    """Exact log p(xbar_O) for observed settings O (nc > 0): xbar_O ~ N(m0_O, K_OO + diag(sig2_O / nc_O)), up to
    terms free of (m0, K). Returns (logZ, C^-1 y, Cholesky of C, O)."""
    O = np.flatnonzero(nc > 0)
    if len(O) == 0:
        return 0.0, np.zeros(0), np.zeros((0, 0)), O
    C = K[np.ix_(O, O)] + np.diag(sig2[O] / nc[O])
    L = np.linalg.cholesky(C)
    y = xbar[O] - m0[O]
    al = _cho(L, y)
    return -0.5 * y @ al - np.log(np.diag(L)).sum(), al, L, O


def fit_step_spreads(st, key, data, spread_start=None, f=None):
    """Empirical Bayes for one head's step spreads (g, h, d): maximise the evidence (Laplace for quality, exact
    Gaussian for cost) plus a log-normal hyperprior on each spread around its starting value. The mean step mu is a
    coefficient, so its posterior comes with the evidence. Returns (spread, info)."""
    from scipy.optimize import minimize
    s0 = np.log(st["spread0"])
    x0 = s0 if spread_start is None else np.log(spread_start)
    hsd = np.where(np.arange(len(s0)) < 3, st["hyper_sd"], st["r_sd"])
    bounds = [(np.log(0.01), np.log(5.0))] * 3 + [(np.log(0.05), np.log(20.0))] * (len(s0) - 3)
    state = dict(f=f)

    def ev(x):
        m0, K = step_prior(st, np.exp(x))
        if key == "q":
            k, n = data
            lz, ff, *_ = laplace_evidence(m0, K, k, n, state["f"])
            state["f"] = ff
        else:
            lz = gauss_evidence(m0, K, *data)[0]
        return lz - 0.5 * np.sum(((x - s0) / hsd) ** 2)

    r = minimize(lambda x: -ev(x), x0, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=60))
    spread = np.exp(r.x)
    m0, K = step_prior(st, spread)
    xs, mu0, mu_sd = st["xs"], st["mu0"], st["mu_sd"]
    if key == "q":
        lz, ff, a, L, sW = laplace_evidence(m0, K, *data, state["f"])
        v = sW * xs
        mu_mean = mu0 + mu_sd ** 2 * xs @ a
        mu_var = mu_sd ** 2 - mu_sd ** 4 * v @ _cho(L, v)
    else:
        lz, al, L, O = gauss_evidence(m0, K, *data)
        mu_mean = mu0 + mu_sd ** 2 * xs[O] @ al
        mu_var = mu_sd ** 2 - mu_sd ** 4 * xs[O] @ _cho(L, xs[O])
        ff = None
    info = dict(g=round(float(spread[0]), 4), h=round(float(spread[1]), 4), d=round(float(spread[2]), 4),
                r=[round(float(v), 3) for v in spread[3:]],
                mu=round(float(mu_mean), 4), mu_sd=round(float(np.sqrt(max(mu_var, 0.0))), 4),
                log_ev=round(float(lz), 3), nfev=int(r.nfev))
    return spread, info, ff


def prior_worth(prior):
    """Per setting: prior information on the quality logit, in Bernoulli runs at the prior mean quality."""
    m0, S0 = prior["m0"], prior["S0"]
    p = 1 / (1 + np.exp(-m0))
    return 1.0 / (np.diag(S0) * p * (1 - p))


class Posterior:
    """Gaussian posterior over (eta, mu) of S settings. add() records a draw; fit() refreshes the approximation.

    F1 switch `incremental` (FRAMEWORK.md section 16): after a draw, rank-1 (Sherman-Morrison) update of the Laplace
    covariance for the new run at the current mode, then `inc_steps` warm Newton steps using it; a full Newton to
    convergence with a fresh inverse every `full_every` draws or when the last step exceeds `step_tol`.
    Off: every fit() is a full Newton from the previous mode with a fresh inverse.
    """
    incremental = True
    full_every = 50
    inc_steps = 2
    step_tol = 1e-2          # logit; measured: pass-rate error at most 1.5e-4 over 1,500 draws (FRAMEWORK.md 16)

    def __init__(self, prior, cost=None):
        self.cost = StudyCost(prior) if cost is None else cost
        self.prior = prior
        self.m0, self.S0 = prior["m0"], prior["S0"]
        self.P0 = np.linalg.inv(self.S0)
        S = len(self.m0)
        self.S = S
        self.k = np.zeros(S)
        self.n = np.zeros(S)
        self.m = self.m0.copy()
        self._pending = []
        self._since_full = 0
        self.n_full = self.n_inc = 0
        self.fit()

    def set_prior(self, prior):
        """Swap in a new prior (after an effort-step refit); the next fit() is a full one from the current mode."""
        self.prior = prior
        self.m0, self.S0 = prior["m0"], prior["S0"]
        self.P0 = np.linalg.inv(self.S0)
        self._ready = False

    def refit_steps(self):
        steps = self.prior.get("steps")
        if steps is None:
            return self.cost.refit()
        if not hasattr(self, "step_spread"):
            self.step_spread, self._step_f = dict(q=None), None
        sq, iq, self._step_f = fit_step_spreads(steps["q"], "q", (self.k, self.n), self.step_spread["q"], self._step_f)
        ic = self.cost.refit()
        self.step_spread = dict(q=sq)
        m0, S0 = step_prior(steps["q"], sq)
        self.set_prior(dict(self.prior, m0=m0, S0=S0))
        self.fit()
        return dict(q=iq, c=ic)

    def add(self, s, y, cost):
        self._pending.append(s)
        self.k[s] += y
        self.n[s] += 1
        self.cost.add(s, cost)

    def _use_inc(self):
        return (self.incremental and getattr(self, "_ready", False)
                and self._since_full + len(self._pending) < self.full_every)

    def fit(self):
        if self._use_inc() and self._fit_inc():
            return
        self._fit_full()
        self._ready = True

    def _fit_inc(self):
        """F1: rank-1 covariance update per pending run, then warm Newton steps. False asks for a full fit."""
        Sq, m = self.Sq.copy(), self.m.copy()
        for s in self._pending:
            p = 1 / (1 + np.exp(-m[s]))
            w = p * (1 - p)
            col = Sq[:, s].copy()
            Sq -= np.outer(col, col) * (w / (1 + w * col[s]))
        step = np.zeros(1)
        for _ in range(self.inc_steps):
            p = 1 / (1 + np.exp(-m))
            step = Sq @ ((self.k - self.n * p) - self.P0 @ (m - self.m0))
            m = m + step
        if np.abs(step).max() > self.step_tol:
            return False
        self._since_full += len(self._pending)
        self._pending = []
        self.n_inc += 1
        self.m, self.Sq = m, (Sq + Sq.T) / 2
        self._fit_cost()
        return True

    def _fit_full(self):
        self._pending = []
        self._since_full = 0
        self.n_full += 1
        m = self.m.copy()
        for _ in range(200):
            p = 1 / (1 + np.exp(-m))
            g = (self.k - self.n * p) - self.P0 @ (m - self.m0)
            H = self.P0 + np.diag(self.n * p * (1 - p))
            step = np.linalg.solve(H, g)
            step *= min(1.0, 2.0 / max(np.abs(step).max(), 1e-12))   # damped: at most 2 logit units per step
            m = m + step
            if np.abs(step).max() < 1e-9:
                break
        p = 1 / (1 + np.exp(-m))
        H = self.P0 + np.diag(self.n * p * (1 - p))
        self.m = m
        self.Sq = np.linalg.inv(H)
        self.Sq = (self.Sq + self.Sq.T) / 2
        self._fit_cost()

    def _fit_cost(self):
        self.cost.fit()
        self._Lq = self._Lc = None
        if all(hasattr(self.cost, name) for name in ("mc", "Sc", "sig2")):
            self.mc, self.Sc, self.sig2 = self.cost.mc, self.cost.Sc, self.cost.sig2
        else:
            # A Gaussian view used only by the lookahead. Worlds draw from CostModel.
            draws = np.log(self.cost.sample_mean(4096, np.random.default_rng(7921)))
            self.Sc = np.atleast_2d(np.cov(draws, rowvar=False)) + 1e-10 * np.eye(self.S)
            self.sig2 = np.full(self.S, 0.25)
            self.mc = draws.mean(0) - self.sig2 / 2

    def chol(self):
        if self._Lq is None:
            self._Lq = np.linalg.cholesky(self.Sq + 1e-10 * np.eye(self.S))
            self._Lc = np.linalg.cholesky(self.Sc + 1e-10 * np.eye(self.S))
        return self._Lq, self._Lc

    def sample_latent(self, K, rng):
        Lq, Lc = self.chol()
        E = self.m + rng.standard_normal((K, self.S)) @ Lq.T
        M = (self.cost.sample_latent(K, rng) if isinstance(self.cost, StudyCost)
             else np.log(self.cost.sample_mean(K, rng)) - self.sig2 / 2)
        return E, M

    def to_qc(self, E, M):
        return 1 / (1 + np.exp(-E)), np.exp(M + self.sig2 / 2)

    def sample_worlds(self, K, rng):
        """K worlds for the reweighting lookahead: (q, c, rho, M). q, c: the targets (pool means when the pool
        correction is on); rho: each world's chance that the next run of s passes; M: latent log-cost locations."""
        Lq, Lc = self.chol()
        E = self.m + rng.standard_normal((K, self.S)) @ Lq.T
        M = (self.cost.sample_latent(K, rng) if isinstance(self.cost, StudyCost)
             else np.log(self.cost.sample_mean(K, rng)) - self.sig2 / 2)
        rho = 1 / (1 + np.exp(-E))
        if getattr(self, "pool_N", None) is None:
            return rho, np.exp(M + self.sig2 / 2), rho, M
        q, Mp = self._fpc(rho, M, rng)
        return q, np.exp(Mp + self.sig2 / 2), rho, M

    def point(self):
        """Plug-in values: posterior median quality and the mean cost at the posterior mean of mu."""
        return 1 / (1 + np.exp(-self.m)), np.exp(self.mc + self.sig2 / 2)

    def mean_draw_cost(self):
        return self.cost.per_run()


class PoolMixin:
    """Finite-pool correction (draws without replacement): truth is the pool mean, so remaining uncertainty is only
    over the runs not yet drawn. set_pool(N) turns it on; sample_latent then returns the pool mean's logit and log."""

    pool_N = None

    def set_pool(self, N):
        if not isinstance(self.cost, StudyCost):
            raise ValueError("finite pool correction requires StudyCost")
        self.pool_N = np.asarray(N, float)
        self.cost.set_pool(N)

    def _fpc(self, q, M, rng):
        return self.cost.pool_adjust(q, M, self.k, self.n, rng)


class SettingPosterior(PoolMixin, Posterior):
    """Setting-level Bernoulli model (Posterior) with the finite-pool correction available."""

    def __init__(self, prior):
        self.csum = np.zeros(len(prior["m0"]))
        super().__init__(prior)

    def add(self, s, y, cost, t=None):
        super().add(s, y, cost)
        self.csum[s] += cost

    def sample_latent(self, K, rng):
        E, M = super().sample_latent(K, rng)
        if self.pool_N is None:
            return E, M
        q, M = self._fpc(1 / (1 + np.exp(-E)), M, rng)
        return np.log(q / (1 - q)), M




class TaskPosterior(PoolMixin, Posterior):
    """Task-level quality (loopmath.tasks model): p_st = expit(a_t (theta_s - b_t)), b_t ~ N(0, tau^2).

    Slopes a_t and tau are plug-ins from loopmath.tasks.fit.fit_bank (2PL, tau by empirical Bayes over its grid),
    refit on the runs drawn so far every refit_every draws (refit()); before the first refit a_t = 1, tau = tau.

    Setting quality q_s = sum_t w_st p_st with w_st the setting's share of runs on task t in the pool (equal shares
    when unknown). theta's prior is the option's structured Gaussian rescaled from the marginal-logit scale by
    k = sqrt(1 + pi tau^2 / 8) (logistic-normal approximation). Laplace over (theta, b); cost as in Posterior.
    m and Sq expose a delta-method Gaussian view of logit q_s for the KG lookahead.
    """

    def __init__(self, prior, weights, tau=1.5, slope=None, cost=None):
        self.W = np.asarray(weights, float)                       # S x T, rows sum to 1
        self.S_, self.T = self.W.shape
        S, T = self.S_, self.T
        self.prior_ = prior
        self.a = np.ones(T) if slope is None else np.asarray(slope, float)
        self.Pz = np.zeros((S + T, S + T))
        self._set_hyper(tau, self.a)
        self.z0 = np.concatenate([self.th0, np.zeros(T)])
        self.kst = np.zeros((S, T))
        self.nst = np.zeros((S, T))
        self.z = self.z0.copy()
        self.csum = np.zeros(S)
        super().__init__(prior, cost=cost)

    def _set_hyper(self, tau, a):
        """theta prior: the option's Gaussian on the marginal-logit scale, rescaled by k = sqrt(1 + pi (a tau)^2 / 8)
        with a the mean slope (logistic-normal approximation); task block N(0, tau^2)."""
        S = self.S_
        self.tau, self.a = float(tau), np.asarray(a, float)
        kk = np.sqrt(1 + np.pi * (self.a.mean() * tau) ** 2 / 8)
        self.th0 = self.prior_["m0"] * kk
        self.Pz[:, :] = 0.0
        self.Pz[:S, :S] = np.linalg.inv(self.prior_["S0"] * kk ** 2)
        self.Pz[S:, S:] = np.eye(self.T) / tau ** 2
        if hasattr(self, "z0"):
            self.z0[:S] = self.th0
        self._ready = False                     # new hyperparameters: next fit() is a full one

    def set_prior(self, prior):
        """New option prior (effort-step refit): rescale it into theta's prior at the current slopes and tau."""
        super().set_prior(prior)
        self.prior_ = prior
        self._set_hyper(self.tau, self.a)

    def refit(self, model="2pl", min_runs=100):
        """Plug in slopes and tau from loopmath.tasks.fit.fit_bank on the drawn runs. Returns its summary or None."""
        seen = self.nst.sum(1) > 0
        if seen.sum() < 2 or self.nst.sum() < min_runs:
            return None
        lm = task_fit
        tasks = [str(t) for t in range(self.T)]
        bank = lm.Bank([str(s) for s in np.flatnonzero(seen)], tasks, self.kst[seen], self.nst[seen])
        f = lm.fit_bank(bank, model=model)
        self._set_hyper(f.tau, f.a)
        self.fit()
        return f.summary()

    def add(self, s, y, cost, t=None):
        Posterior.add(self, s, y, cost)
        self._pending[-1] = (s, t)
        self.csum[s] += cost
        self.kst[s, t] += y
        self.nst[s, t] += 1

    def _p(self, z):
        S = self.S_
        return 1 / (1 + np.exp(-self.a[None, :] * (z[:S, None] - z[None, S:])))

    def fit(self):
        if not hasattr(self, "kst"):            # called from Posterior.__init__ before task arrays exist
            self._pending = []
            return
        if self._use_inc() and self._fit_inc():
            return
        self._fit_full()
        self._ready = True

    def _fit_inc(self):
        """F1 for the task model: a run on cell (s, t) adds w a_t^2 u u^T to the precision, u = e_s - e_(S+t), so the
        covariance takes a rank-1 Sherman-Morrison update; then warm Newton steps with it. False asks for a full fit."""
        S = self.S_
        Cz, z = self.Cz.copy(), self.z.copy()
        for s, t in self._pending:
            j = S + t
            p = 1 / (1 + np.exp(-self.a[t] * (z[s] - z[j])))
            w = p * (1 - p) * self.a[t] ** 2
            v = Cz[:, s] - Cz[:, j]
            Cz -= np.outer(v, v) * (w / (1 + w * (v[s] - v[j])))
        step = np.zeros(1)
        for _ in range(self.inc_steps):
            p = self._p(z)
            r = (self.kst - self.nst * p) * self.a[None, :]
            step = Cz @ (np.concatenate([r.sum(1), -r.sum(0)]) - self.Pz @ (z - self.z0))
            z = z + step
        if np.abs(step).max() > self.step_tol:
            return False
        self._since_full += len(self._pending)
        self._pending = []
        self.n_inc += 1
        self.z, self.Cz = z, (Cz + Cz.T) / 2
        self._view()
        return True

    def _fit_full(self):
        self._pending = []
        self._since_full = 0
        self.n_full += 1
        S, T = self.S_, self.T
        z = self.z.copy()
        for _ in range(200):
            p = self._p(z)
            r = (self.kst - self.nst * p) * self.a[None, :]
            g = np.concatenate([r.sum(1), -r.sum(0)]) - self.Pz @ (z - self.z0)
            H = self._hess(p)
            step = np.linalg.solve(H, g)
            step *= min(1.0, 2.0 / max(np.abs(step).max(), 1e-12))   # damped: at most 2 logit units per step
            z = z + step
            if np.abs(step).max() < 1e-8:
                break
        self.z = z
        p = self._p(z)
        self.Cz = np.linalg.inv(self._hess(p))
        self.Cz = (self.Cz + self.Cz.T) / 2
        self._view()

    def _view(self):
        S, T = self.S_, self.T
        p = self._p(self.z)
        self._Lz = None
        # delta-method view of logit q_s
        q = (self.W * p).sum(1)
        d = self.W * p * (1 - p) * self.a[None, :]                 # dq_s / d theta_s per task
        J = np.zeros((S, S + T))
        J[np.arange(S), np.arange(S)] = d.sum(1)
        J[:, S:] = -d
        D = 1 / (q * (1 - q))
        self.m = np.log(q / (1 - q))
        self.Sq = (D[:, None] * (J @ self.Cz @ J.T)) * D[None, :]
        self.Sq = (self.Sq + self.Sq.T) / 2 + 1e-9 * np.eye(S)
        self._fit_cost()

    def _hess(self, p):
        S = self.S_
        Wf = self.nst * p * (1 - p) * self.a[None, :] ** 2
        H = self.Pz.copy()
        H[np.arange(S), np.arange(S)] += Wf.sum(1)
        H[S + np.arange(self.T), S + np.arange(self.T)] += Wf.sum(0)
        H[:S, S:] -= Wf
        H[S:, :S] -= Wf.T
        return H

    def _fit_cost(self):
        self.cost.fit()
        self._Lq = self._Lc = None
        if all(hasattr(self.cost, name) for name in ("mc", "Sc", "sig2")):
            self.mc, self.Sc, self.sig2 = self.cost.mc, self.cost.Sc, self.cost.sig2
        else:
            # A Gaussian view used only by the lookahead. Worlds draw from CostModel.
            draws = np.log(self.cost.sample_mean(4096, np.random.default_rng(7921)))
            self.Sc = np.atleast_2d(np.cov(draws, rowvar=False)) + 1e-10 * np.eye(self.S)
            self.sig2 = np.full(self.S, 0.25)
            self.mc = draws.mean(0) - self.sig2 / 2

    def sample_latent(self, K, rng):
        S = self.S_
        if self._Lz is None:
            self._Lz = np.linalg.cholesky(self.Cz + 1e-10 * np.eye(len(self.z)))
        _, Lc = self.chol()
        Z = self.z + rng.standard_normal((K, len(self.z))) @ self._Lz.T
        M = (self.cost.sample_latent(K, rng) if isinstance(self.cost, StudyCost)
             else np.log(self.cost.sample_mean(K, rng)) - self.sig2 / 2)
        P = 1 / (1 + np.exp(-self.a * (Z[:, :S, None] - Z[:, None, S:])))  # K x S x T
        if self.pool_N is not None and self.cell_N is not None:
            rem = np.maximum(self.cell_N - self.nst, 0)            # S x T runs not yet drawn
            mean = (self.kst + rem * P).sum(2)
            sd = np.sqrt((rem * P * (1 - P)).sum(2))
            q = (mean + sd * rng.standard_normal((K, S))) / self.pool_N
            q = np.clip(q, 0.5 / self.pool_N, 1 - 0.5 / self.pool_N)
            _, M = self._fpc(q, M, rng)                           # cost part only; q already the pool mean
            qret = q
        else:
            qret = (self.W[None] * P).sum(2)
        return np.log(qret / (1 - qret)), M

    cell_N = None
    cell_cap = np.inf

    def set_cells(self, cell_N, cap=np.inf):
        self.cell_N = np.asarray(cell_N, float)
        self.cell_cap = cap
        self.set_pool(self.cell_N.sum(1))

    def open_shares(self):
        """S x T: chance the next draw of s lands on task t (uniform over open cells; pool shares with replacement)."""
        if self.cell_N is None:
            return self.W
        op = (self.nst < np.minimum(self.cell_N, self.cell_cap)).astype(float)
        tot = op.sum(1, keepdims=True)
        return np.where(tot > 0, op / np.maximum(tot, 1), self.W)

    def sample_worlds_tasks(self, K, rng):
        """As Posterior.sample_worlds but through the task terms (K x S x T work); rho integrates the task of the
        next draw over the open cells. The KG default uses the inherited delta-method view (m, Sq), K x S work."""
        S = self.S_
        if self._Lz is None:
            self._Lz = np.linalg.cholesky(self.Cz + 1e-10 * np.eye(len(self.z)))
        _, Lc = self.chol()
        Z = self.z + rng.standard_normal((K, len(self.z))) @ self._Lz.T
        M = (self.cost.sample_latent(K, rng) if isinstance(self.cost, StudyCost)
             else np.log(self.cost.sample_mean(K, rng)) - self.sig2 / 2)
        P = 1 / (1 + np.exp(-self.a * (Z[:, :S, None] - Z[:, None, S:])))
        rho = np.einsum("kst,st->ks", P, self.open_shares())
        if self.pool_N is not None and self.cell_N is not None:
            rem = np.maximum(self.cell_N - self.nst, 0)
            mean = (self.kst + rem * P).sum(2)
            sd = np.sqrt((rem * P * (1 - P)).sum(2))
            q = np.clip((mean + sd * rng.standard_normal((K, S))) / self.pool_N, 0.5 / self.pool_N, 1 - 0.5 / self.pool_N)
            _, Mp = self._fpc(q, M, rng)
        else:
            q, Mp = (self.W[None] * P).sum(2), M
        return q, np.exp(Mp + self.sig2 / 2), rho, M

    def point(self):
        return 1 / (1 + np.exp(-self.m)), np.exp(self.mc + self.sig2 / 2)
