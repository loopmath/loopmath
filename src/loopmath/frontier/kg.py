"""Stop rule and pickers. FRAMEWORK.md sections 5 to 7.

Stop rule: draw K posterior worlds, take candidate returned sets (the plug-in frontier at the posterior point and the
most frequent frontiers among the draws), score each by P(match), keep the best. Stop when it reaches conf.
KG: expected one-step gain in P(best guess matches the world); per dollar divides by the expected cost of the
lookahead draws. Two engines: "reweight" (default; FRAMEWORK.md section 16) reweights one shared set of K posterior
worlds by each hypothetical outcome's likelihood, no refits and no new frontiers; "matheron" moves 128 shared worlds
by a Gaussian update per candidate and outcome and recomputes their frontiers.
"""
from dataclasses import dataclass, asdict

import numpy as np

from .pareto import frontier_mask, frontier_mask_batch, match_batch, best_q_within


@dataclass
class MatchSpec:
    kind: str = "two_sided"
    eps_q: float = 0.05
    eps_c: float = 0.10
    floor: float = -1.0          # < 0 means no floor
    conf: float = 0.95
    tau_R: float = 0.5           # ci rule: returned set = settings on the frontier in at least tau_R of the draws
    ci_stop: str = "joint"       # ci rule: all at once at conf (default), or per point and per exclusion at conf
    kg_target: str = "stop"      # KG target: "stop" = the stop rule's probability; "exact" = P(exact frontier)

    @property
    def fl(self):
        return None if self.floor < 0 else self.floor

    def as_dict(self):
        return asdict(self)


def p_match(post, spec, K, rng, n_freq=3):
    """Best candidate returned set and its posterior match probability. Returns (R, p, candidates scored).

    Candidates: plug-in frontier, inclusive sets {s: P(s on frontier) >= tau} for tau in TAUS, and the n_freq most
    frequent frontiers among the draws. The same draws score every candidate, so the max is biased up by a little
    (selection over at most 7 candidates); the calibration check measures the net effect.
    """
    E, M = post.sample_latent(K, rng)
    q, c = post.to_qc(E, M)
    F = frontier_mask_batch(c, q, spec.fl)
    qh, ch = post.point()
    cands = [frontier_mask(ch, qh, spec.fl)]
    freq = F.mean(axis=0)
    for tau in TAUS:
        cands.append(freq >= tau)
    uniq, counts = np.unique(F, axis=0, return_counts=True)
    for i in np.argsort(-counts)[:n_freq]:
        cands.append(uniq[i])
    best_R, best_p = None, -1.0
    seen = set()
    for R in cands:
        key = R.tobytes()
        if key in seen:
            continue
        seen.add(key)
        p = match_batch(R, c, q, spec.kind, spec.eps_q, spec.eps_c, spec.fl, F=F).mean()
        if p > best_p:
            best_R, best_p = R.copy(), float(p)
    return best_R, best_p, len(seen)


def ci_parts(q, c, spec, F=None):
    """Pieces of the stop rule (kind "ci") from posterior draws q, c (K x S).

    R = settings on the frontier in at least tau_R of the draws. Each r in R needs its conf-level central interval of
    pass@1 to have half-width at most eps_q. Each s outside R must be ruled out with posterior probability at least
    conf: q_s <= (best quality in R at cost <= (1 + eps_c) c_s, 0 if none) + eps_q, i.e. dominated by or within
    tolerance of a returned frontier point. Returns R, half-widths (S), ruled-out probabilities (S).
    """
    if F is None:
        F = frontier_mask_batch(c, q, spec.fl)
    R = F.mean(axis=0) >= spec.tau_R
    a = (1 - spec.conf) / 2
    lo, hi = np.quantile(q, [a, 1 - a], axis=0)
    half = (hi - lo) / 2
    qa = q if spec.fl is None else np.where(q >= spec.fl, q, -np.inf)
    reach = np.maximum(best_q_within(c, np.where(R, qa, -np.inf), (1 + spec.eps_c) * c), 0.0)
    out = (q <= reach + spec.eps_q + 1e-12).mean(axis=0)
    return R, half, out


def reach_R(c, q, R, eps_c):
    """K x S: best q among the members of R (one set for all rows, or one per row) costing at most (1 + eps_c) c_s
    in the same world, 0 if none. Direct K x |R| x S comparison; R is small (the frontier guess)."""
    R = np.asarray(R, bool)
    K, S = c.shape
    if R.ndim == 1:
        idx = np.flatnonzero(R)
        if not len(idx):
            return np.zeros((K, S))
        cr, qr = c[:, idx], q[:, idx]
        ok = cr[:, :, None] <= (1 + eps_c) * c[:, None, :]
        return np.maximum(np.where(ok, qr[:, :, None], 0.0).max(axis=1), 0.0)
    return np.maximum(best_q_within(c, np.where(R, q, -np.inf), (1 + eps_c) * c), 0.0)


def ci_match(q, c, spec, F=None, H=1):
    """Joint ci match per draw, H stacked blocks of K draws: the best guess (R, qhat) matches a world when every
    r in R has |q_r - qhat_r| <= eps_q and every s outside R has q_s <= reach_R(s) + eps_q in that world.
    R and qhat come from each block's own draws (R: frontier frequency >= tau_R; qhat: mean). Returns (H, K) bools.
    """
    if F is None:
        F = frontier_mask_batch(c, q, spec.fl)
    HK, S = q.shape
    K = HK // H
    q3 = q.reshape(H, K, S)
    R = F.reshape(H, K, S).mean(axis=1) >= spec.tau_R
    qhat = q3.mean(axis=1)
    Rrows = np.repeat(R, K, axis=0)
    qa = q if spec.fl is None else np.where(q >= spec.fl, q, -np.inf)
    reach = np.maximum(best_q_within(c, np.where(Rrows, qa, -np.inf), (1 + spec.eps_c) * c), 0.0).reshape(H, K, S)
    near = np.abs(q3 - qhat[:, None, :]) <= spec.eps_q
    out = q3 <= reach + spec.eps_q + 1e-12
    ok = np.where(R[:, None, :], near, out)
    return ok.all(axis=2)


def ci_check(post, spec, K, rng):
    """(R, p_out, info): stop iff info['stop']. p_out = smallest ruled-out probability outside R."""
    E, M = post.sample_latent(K, rng)
    q, c = post.to_qc(E, M)
    R, half, out = ci_parts(q, c, spec)
    p_out = float(out[~R].min()) if (~R).any() else 1.0
    maxhalf = float(half[R].max()) if R.any() else 1.0
    qhat = q.mean(axis=0)
    p_joint = float(ci_match(q, c, spec).mean())
    if spec.ci_stop == "joint":
        stop, p = p_joint >= spec.conf, p_joint
    else:
        stop, p = maxhalf <= spec.eps_q and p_out >= spec.conf, p_out
    return R, p, dict(stop=bool(stop), maxhalf=maxhalf, qhat=qhat, n_R=int(R.sum()), p_joint=p_joint, p_out=p_out)


def ci_loss(q, c, spec, F=None, H=1):
    """KG target for kind ci, for H stacked blocks of K draws each (q, c: (H K) x S). Returns H losses.

    Loss = sum over R of the half-width excess over eps_q (normal width, in units of eps_q) plus sum outside R of the
    ruled-out shortfall below conf. 0 when the stop rule holds.
    """
    from scipy.stats import norm
    if F is None:
        F = frontier_mask_batch(c, q, spec.fl)
    HK, S = q.shape
    K = HK // H
    R = F.reshape(H, K, S).mean(axis=1) >= spec.tau_R                       # H x S
    Rrows = np.repeat(R, K, axis=0)
    qa = q if spec.fl is None else np.where(q >= spec.fl, q, -np.inf)
    reach = np.maximum(best_q_within(c, np.where(Rrows, qa, -np.inf), (1 + spec.eps_c) * c), 0.0)
    out = (q <= reach + spec.eps_q + 1e-12).reshape(H, K, S).mean(axis=1)    # H x S
    half = norm.ppf(0.5 + spec.conf / 2) * q.reshape(H, K, S).std(axis=1)
    loss = (np.where(R, np.maximum(half - spec.eps_q, 0), 0).sum(1) / spec.eps_q
            + np.where(~R, np.maximum(spec.conf - out, 0), 0).sum(1))
    return loss


TAUS = (0.5, 0.3, 0.1)   # inclusive candidates: every setting on the frontier in at least tau of the draws
TAU_KG = 0.3             # the returned-set rule KG optimises (a frontier member is always sound, so include liberally)
GH3 = (np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)]), np.array([1 / 6, 2 / 3, 1 / 6]))


def kg_values(post, spec, cands, L, K, rng, per_dollar=True, EM=None):
    """KG value of drawing L runs from each candidate setting. Returns array aligned with cands.

    EM: optional (E, M) posterior draws to reuse (the relevance screen's)."""
    S = post.S
    E, M = post.sample_latent(K, rng) if EM is None else EM
    q, c = post.to_qc(E, M)
    F0 = frontier_mask_batch(c, q, spec.fl)
    if spec.kg_target == "exact":
        R0 = F0.mean(axis=0) >= 0.5
        base = (F0 == R0).all(axis=1).mean()
    elif spec.kind == "ci":
        base = ci_match(q, c, spec, F0).mean()
    else:
        R0 = F0.mean(axis=0) >= TAU_KG
        base = match_batch(R0, c, q, spec.kind, spec.eps_q, spec.eps_c, spec.fl, F=F0).mean()
    sig2 = post.sig2
    rowsE, rowsM, weights, owner = [], [], [], []
    for ci, s in enumerate(cands):
        ms, vs = post.m[s], post.Sq[s, s]
        p = 1 / (1 + np.exp(-ms))
        w = L * p * (1 - p)
        g = post.Sq[:, s] / (vs + 1 / w)
        if L == 1:
            pbar = 1 / (1 + np.exp(-ms / np.sqrt(1 + np.pi * vs / 8)))
            ks, wk = np.array([0.0, 1.0]), np.array([1 - pbar, pbar])
        else:
            ks = L / (1 + np.exp(-(ms + np.sqrt(vs) * GH3[0])))
            wk = GH3[1]
        noise_q = rng.standard_normal(K) / np.sqrt(w)
        # cost: the lookahead moves mu_s by a predictive draw of the mean log cost of L runs; take its mean,
        # which keeps the variance shrink (the part that changes P(match)) and drops a symmetric shift
        vcs = post.Sc[s, s] + sig2[s] / L
        gc = post.Sc[:, s] / vcs
        noise_c = rng.standard_normal(K) * np.sqrt(sig2[s] / L)
        Mn = M + (post.mc[s] - M[:, s] - noise_c)[:, None] * gc[None, :]
        for kk, wkk in zip(ks, wk):
            Y = ms + (kk - L * p) / w
            En = E + (Y - E[:, s] - noise_q)[:, None] * g[None, :]
            rowsE.append(En)
            rowsM.append(Mn)
            weights.append(wkk)
            owner.append(ci)
    H = len(rowsE)
    En = np.concatenate(rowsE)
    Mn = np.concatenate(rowsM)
    qn, cn = 1 / (1 + np.exp(-En)), np.exp(Mn + sig2 / 2)
    Fn = frontier_mask_batch(cn, qn, spec.fl)
    if spec.kg_target == "exact":
        F3 = Fn.reshape(H, K, S)
        Rn = F3.mean(axis=1) >= 0.5
        ok = (F3 == Rn[:, None, :]).all(axis=2).mean(axis=1)
    elif spec.kind == "ci":
        ok = ci_match(qn, cn, spec, Fn, H).mean(axis=1)
    else:
        Rn = Fn.reshape(H, K, S).mean(axis=1) >= TAU_KG             # H x S
        Rrows = np.repeat(Rn, K, axis=0)
        ok = match_batch(Rrows, cn, qn, spec.kind, spec.eps_q, spec.eps_c, spec.fl, F=Fn).reshape(H, K).mean(axis=1)
    val = np.zeros(len(cands))
    for h in range(H):
        val[owner[h]] += weights[h] * ok[h]
    val -= base
    if per_dollar:
        val = val / (L * post.cost.per_run()[cands])
    return val, base


class WorldSet:
    """One shared set of K posterior worlds and its frontiers, scored under any weights w (K, sum 1).

    The stop-rule match of a world depends on the weights only through R (frontier frequency >= tau_R) and qhat
    (weighted mean); the exclusion part is cached per distinct R, so a new weight vector costs two K x S products."""

    def __init__(self, post, spec, K, rng):
        self.spec = spec
        self.q, self.c, self.rho, self.M = post.sample_worlds(K, rng)
        self.K, self.S = self.q.shape
        self.F = frontier_mask_batch(self.c, self.q, spec.fl).astype(float)
        self.qa = self.q if spec.fl is None else np.where(self.q >= spec.fl, self.q, -np.inf)
        self._out = {}

    def _excl(self, R):
        key = R.tobytes()
        if key not in self._out:
            sp = self.spec
            if sp.kg_target == "exact":
                v = (self.F.astype(bool) == R[None, :]).all(axis=1)
            elif sp.kind == "ci":
                reach = reach_R(self.c, self.qa, R, sp.eps_c)
                v = ((self.q <= reach + sp.eps_q + 1e-12) | R[None, :]).all(axis=1)
            else:
                v = match_batch(R, self.c, self.q, sp.kind, sp.eps_q, sp.eps_c, sp.fl, F=self.F.astype(bool))
            self._out[key] = v
        return self._out[key]

    def value(self, w):
        """P(stop rule met) under weights w: R and qhat from the weighted worlds, matched in the same worlds."""
        sp = self.spec
        tau = 0.5 if sp.kg_target == "exact" else (sp.tau_R if sp.kind == "ci" else TAU_KG)
        R = (w @ self.F) >= tau
        ok = self._excl(R)
        if sp.kind == "ci" and sp.kg_target != "exact" and R.any():
            qhat = w @ self.q[:, R]
            ok = ok & (np.abs(self.q[:, R] - qhat) <= sp.eps_q).all(axis=1)
        return float(w @ ok)

    def values(self, W):
        """value() for every column of W (K x H) at once; R groups share the exclusion part and the near test."""
        sp = self.spec
        H = W.shape[1]
        tau = 0.5 if sp.kg_target == "exact" else (sp.tau_R if sp.kind == "ci" else TAU_KG)
        Rs = (W.T @ self.F) >= tau                                     # H x S
        out = np.empty(H)
        keys, inv = np.unique(Rs, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        near_on = sp.kind == "ci" and sp.kg_target != "exact"
        for g, R in enumerate(keys):
            hs = np.flatnonzero(inv == g)
            ok = self._excl(R).astype(float)                          # K
            if near_on and R.any():
                Q = self.q[:, R]                                        # K x r
                qhat = W[:, hs].T @ Q                                   # n x r
                near = (np.abs(Q[None, :, :] - qhat[:, None, :]) <= sp.eps_q).all(axis=2)   # n x K
                out[hs] = (near * ok[None, :] * W[:, hs].T).sum(axis=1)
            else:
                out[hs] = ok @ W[:, hs]
        return out

    def screen(self, available):
        """Settings whose next run can still change the stop decision: frontier members in some world, or settings
        not ruled out in every world (call 3). Uniform weights."""
        w = np.full(self.K, 1 / self.K)
        R = (w @ self.F) >= self.spec.tau_R
        reach = reach_R(self.c, self.qa, R, self.spec.eps_c)
        not_out = (self.q > reach + self.spec.eps_q).any(axis=0)
        keep = ((self.F.sum(axis=0) > 0) | not_out | R) & available
        return np.flatnonzero(keep)


def kg_reweight(post, spec, cands, WS, L=1, per_dollar=True, min_ess=0.25, cost_nodes=False):
    """KG of L runs from each candidate by importance reweighting of the shared worlds WS (no refits).

    Hypothetical outcomes: passes k (L = 1: 0 or 1; L > 1: 3 Gauss-Hermite nodes of the predictive). World weight
    rho^k (1 - rho)^(L - k), all candidates and outcomes at once (one K x H weight matrix). cost_nodes=True also
    crosses 3 Gauss-Hermite nodes of the mean log cost (weight: its normal likelihood given the world's mu_s); off by
    default, because early on a setting's cost posterior is wider than one run's likelihood and the weights collapse.
    Returns (values, base, indices into cands whose weights have ESS below min_ess K)."""
    K = WS.K
    base = WS.value(np.full(K, 1 / K))
    C = len(cands)
    cols, owner, prob = [], [], []
    for i, s in enumerate(cands):
        rho = np.clip(WS.rho[:, s], 1e-9, 1 - 1e-9)
        if L == 1:
            pb = rho.mean()
            ks, wk = np.array([0.0, 1.0]), np.array([1 - pb, pb])
        else:
            ms, vs = post.m[s], post.Sq[s, s]
            ks = L / (1 + np.exp(-(ms + np.sqrt(vs) * GH3[0])))
            wk = GH3[1]
        lr = np.log(rho)[:, None] * ks[None, :] + np.log1p(-rho)[:, None] * (L - ks[None, :])
        if cost_nodes:
            sig2 = post.sig2[s] / L
            xs = post.mc[s] + np.sqrt(post.Sc[s, s] + sig2) * GH3[0]
            lc = -(xs[None, :] - WS.M[:, s:s + 1]) ** 2 / (2 * sig2)
            lr = (lr[:, :, None] + lc[:, None, :]).reshape(K, -1)
            wk = (wk[:, None] * GH3[1][None, :]).ravel()
        cols.append(lr)
        owner += [i] * lr.shape[1]
        prob.append(wk)
    LW = np.concatenate(cols, axis=1)                                  # K x H
    W = np.exp(LW - LW.max(axis=0))
    W /= W.sum(axis=0)
    owner, prob = np.array(owner), np.concatenate(prob)
    ess = 1 / (W * W).sum(axis=0)
    P = WS.values(W)
    val = np.bincount(owner, weights=prob * P, minlength=C) - base
    low = sorted(set(owner[ess < min_ess * K].tolist()))
    if per_dollar:
        val = val / (L * post.cost.per_run()[cands])
    return val, base, low


def relevant(post, spec, K, rng, max_cand, delta=0.02, EM=None):
    """Settings that could matter for the match: near the frontier in some posterior draw. Ranked by relevance x sd."""
    E, M = post.sample_latent(K, rng) if EM is None else EM
    q, c = post.to_qc(E, M)
    qa = q if spec.fl is None else np.where(q >= spec.fl, q, -np.inf)
    bq = best_q_within(c, qa, c)
    near = (q >= bq - spec.eps_q - delta).mean(axis=0)
    sdq = q.std(axis=0)
    score = np.where(near > 0.01, near * sdq, 0.0)
    order = np.argsort(-score)
    return [int(s) for s in order[:max_cand] if score[s] > 0], score


class Picker:
    """KG per dollar or uniform setting selection among available settings."""

    def __init__(self, kind, S, seed, K_kg=128, max_cand=20, Ls=(1, 10, 50), engine="reweight", K_rw=2000):
        if kind not in ("kg_dollar", "uniform"):
            raise ValueError("picker must be kg_dollar or uniform")
        self.kind = kind
        self.engine, self.K_rw = engine, K_rw
        self.min_ess, self.ess_fallback = 0.25, "matheron"
        self.screen = True                      # F3: score only settings that can still change the stop decision
        self.S = S
        self.rng = np.random.default_rng([seed, 11])
        self.order = np.random.default_rng([seed, 13]).permutation(S)
        self.pos = 0
        self.K_kg, self.max_cand, self.Ls = K_kg, max_cand, Ls
        self.fallbacks = {"widened": 0, "uncertainty": 0, "low_ess": 0}
        self.oracle_set = None

    def pick(self, post, spec, available):
        if self.kind == "uniform":
            return int(self.rng.choice(np.flatnonzero(available)))
        if self.engine == "reweight":
            return self._pick_reweight(post, spec, available)
        EM = post.sample_latent(self.K_kg, self.rng)
        cands, score = relevant(post, spec, self.K_kg, self.rng, self.max_cand if self.screen else post.S, EM=EM)
        if not self.screen:
            cands, score = list(np.flatnonzero(available)), np.sqrt(np.diag(post.Sq))
        cands = [s for s in cands if available[s]]
        if not cands:
            self.fallbacks["uncertainty"] += 1
            sd = np.sqrt(np.diag(post.Sq)) * available
            return int(np.argmax(sd))
        tol = 1.0 / self.K_kg
        for i, L in enumerate(self.Ls):
            val, _ = kg_values(post, spec, cands, L, self.K_kg, self.rng, per_dollar=(self.kind == "kg_dollar"),
                               EM=EM)
            raw = val * (L * post.cost.per_run()[cands]) if self.kind == "kg_dollar" else val
            if raw.max() > tol:
                if i > 0:
                    self.fallbacks["widened"] += 1
                return cands[int(np.argmax(val))]
        self.fallbacks["uncertainty"] += 1
        return cands[int(np.argmax(score[cands]))]

    def _pick_reweight(self, post, spec, available):
        WS = WorldSet(post, spec, self.K_rw, self.rng)
        cands = WS.screen(available) if self.screen else np.flatnonzero(available)
        if not len(cands):
            self.fallbacks["uncertainty"] += 1
            return int(np.argmax(np.sqrt(np.diag(post.Sq)) * available))
        # with shared weights the first-order Monte Carlo terms cancel over outcomes (sum_y P(y) w_y = 1/K per world),
        # exactly at L = 1 and only approximately for L > 1 (3 Gauss-Hermite nodes), so a gain is nonzero only when
        # R or qhat moves; any positive gain counts as signal (FRAMEWORK.md sections 9 and 17)
        tol = 1e-6
        per_dollar = self.kind == "kg_dollar"
        for i, L in enumerate(self.Ls):
            val, _, low = kg_reweight(post, spec, cands, WS, L, per_dollar, self.min_ess)
            if low and self.ess_fallback == "matheron":
                # weights too uneven: those candidates get the Gaussian (Matheron) lookahead instead
                self.fallbacks["low_ess"] += 1
                sub = [int(cands[j]) for j in low]
                v2, _ = kg_values(post, spec, sub, L, self.K_kg, self.rng, per_dollar)
                val[low] = v2
            raw = val * (L * post.cost.per_run()[cands]) if per_dollar else val
            if raw.max() > tol:
                if i > 0:
                    self.fallbacks["widened"] += 1
                return int(cands[int(np.argmax(val))])
        self.fallbacks["uncertainty"] += 1
        sd = WS.q[:, cands].std(axis=0)
        return int(cands[int(np.argmax(sd))])
