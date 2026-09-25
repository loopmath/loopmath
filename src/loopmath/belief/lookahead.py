"""One-step look-ahead gain (paper Remark 5.9; spec 04 section 6), and `conditioned`.

`lookahead(state, task, explore, goal, candidates)` simulates one more run of `explore` on
`task`:
- outcomes: `z` in {0, 1} (reliability `LOOKAHEAD_Q`), or a grid on the score when `g` comes
  from a score head; times a grid on a composite log-cost observation (the mean log cost of
  explore's pieces in round 1, noise `sigma^2 / n`). The grids are the trapezoid rule on the
  standard normal, step 0.2 over 6 sd each way: `min_c ell_after(c)` has kinks where the best
  candidate changes, and Gauss-Hermite rules converge slowly there (5 points overstated `G` by
  15 to 20 percent against brute force; tests/belief/test_belief_lookahead.py);
- updates: exact rank-one Gaussian updates of the cost head and of a score head. The gate head
  is not updated. Nodes a head has not seen take part as independent `N(0, phi^2)` effects;
- the success head's update is exact Bayes over its joint draws: every candidate's chance
  `s(eta)` in each of the 400 draws (the known nodes' draws plus each unseen node's own draws,
  shared by every row that has it), times Gauss-Hermite nodes on an unseen task's effect, which
  every row shares (so a new task's id does not move the gain). The verdict reweights the
  draws by its likelihood, `q s + (1 - q)(1 - s)` or its complement, and `g` after it is the
  reweighted mean. `g` now, `P(z)` and `g` after come from the same draws, so each candidate's
  `g` after, averaged over `z`, is its `g` now. A Newton step linearized at the mean did not
  hold this: with a success logit variance near 10 it lowered every related candidate's
  expected `g` by 6 points and turned `G` negative on a prior-only store;
- `ell` for every candidate is recomputed after each outcome, the cost in closed form:
  `E[C] = sum over rows of width * E[r(k)] * exp(mu + (v + sigma^2) / 2)` (E[r(k)] from the
  gate draws, which the update leaves alone, with an unseen task's gate effect integrated out
  as in the prediction means), and `g` as above or `p_reach`. `ell_now` uses the same formulas,
  so no candidate's expected `ell` after the run is above its `ell` now (the cost grid's cut
  tails can only lower it), and `G` is not negative before its clip.

`G = max(0, min_c ell_now(c) - E[min_c ell_after(c)])`. `p_beats_goal` is the share of the
400 joint draws in which `explore` has lower `ell` than `goal`. `posterior_shift` gives the
10th and 90th percentiles, over the simulated outcomes, of explore's `g` and expected cost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.special import expit, ndtr

from ..types import AcceptanceRule, Configuration, Interval, LookaheadResult, Task
from .design import cost_rest
from .state import (MIN_SCORE_SWITCH, N_DRAWS, PREDICT_SOURCE, FitState, HeadState, _GW, _GX, inverse_transform,
                    transform)

LOOKAHEAD_Q = 0.95  # reliability of the simulated run's verdict (tier "reported")
QUAD_STEP = 0.2
_QX = np.arange(-6.0, 6.0 + 1e-9, QUAD_STEP)  # standard normal outcomes
_QW = np.exp(-0.5 * _QX ** 2)
_QW = _QW / _QW.sum()


def _info(eta: float, q: float = LOOKAHEAD_Q) -> float:
    """Fisher information of one verdict with reliability q at linear predictor eta."""
    s = float(expit(eta))
    a = 2.0 * q - 1.0
    pi1 = min(max((1.0 - q) + a * s, 1e-12), 1 - 1e-12)
    dpi = a * s * (1.0 - s)
    return max(dpi * dpi / (pi1 * (1.0 - pi1)), 1e-12)


def _probit(mu: np.ndarray, v: np.ndarray) -> np.ndarray:
    return expit(mu / np.sqrt(1.0 + math.pi * np.maximum(v, 0.0) / 8.0))


@dataclass
class _Rows:
    """Rows of one head in the augmented space: known nodes through U, unseen nodes diagonal."""

    head: HeadState
    mu: np.ndarray  # (n,)
    B: np.ndarray  # (n, p): x' U
    Xv: sparse.csr_matrix  # (n, q) unseen-node values
    vcol: dict[str, int]
    vphi2: np.ndarray  # (q,)
    var: np.ndarray  # (n,)

    @classmethod
    def build(cls, head: HeadState, rows: Sequence[Sequence[tuple]]) -> "_Rows":
        Xk, Xv, vnodes, vphi = head.matrices(rows)
        mu = np.asarray(Xk @ head.mean).ravel() if Xk.nnz else np.zeros(len(rows))
        B = np.asarray(Xk @ head.U) if head.nodes else np.zeros((len(rows), 0))
        vphi2 = vphi ** 2
        var = np.sum(B * B, axis=1) + np.asarray(Xv.multiply(Xv) @ vphi2).ravel()
        return cls(head, mu, B, Xv.tocsc(), {v: i for i, v in enumerate(vnodes)}, vphi2, var)

    def observation(self, terms: Sequence[tuple]) -> tuple[float, float, np.ndarray]:
        """Mean, variance, and covariance with every row, of one observation row `x'b`."""
        head = self.head
        kx = np.zeros(len(head.nodes))
        vpart: dict[str, float] = {}
        for node, _parent, value in terms:
            j = head.index.get(node)
            if j is not None:
                kx[j] += value
            elif head.node_scale(node) > 0:
                vpart[node] = vpart.get(node, 0.0) + value
        nz = np.flatnonzero(kx)  # a few nodes: only their rows of U
        b = kx[nz] @ head.U[nz] if head.nodes else np.zeros(0)
        mu = float(kx @ head.mean) if head.nodes else 0.0
        var = float(b @ b)
        kappa = self.B @ b if head.nodes else np.zeros(len(self.mu))
        for node, value in vpart.items():
            phi2 = head.node_scale(node) ** 2
            var += phi2 * value * value
            col = self.vcol.get(node)
            if col is not None:
                kappa = kappa + phi2 * value * self.Xv[:, col].toarray().ravel()
        return mu, var, kappa


@dataclass
class _Prepared:
    configs: list[Configuration]
    index: dict[str, int]
    cost: _Rows
    coef: np.ndarray  # (n_cost_rows,): width * E[r(k)]
    owner: sparse.csr_matrix  # (n_configs, n_cost_rows)
    success: _Rows
    score: _Rows | None
    score_name: str | None
    chance: np.ndarray | None = None  # (n_configs, m): s(eta) per joint draw and task node, when g is success
    weight: np.ndarray | None = None  # (m,): each column's weight, summing to 1


def _success_draws(state: FitState, task: Task, plans: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Every configuration's success chance in each joint draw of the success head, and the draws' weights.

    A row's draws hold the known nodes' draws and each unseen node's own draws, which every row with that node
    shares. The task part is common to every row; its unseen effect (a new task's node, say) is drawn by the
    task's id, so here it is a shared Gauss-Hermite dimension instead: one more axis of the joint sample, not a
    per-row integral, and a new task's id then does not move the gain."""
    head = state.heads["success"]
    tp = state._task_parts(task)["success"]
    eta = np.stack([tp.known + state._row(head, plan["run"]).draws for plan in plans])  # (n, draws)
    n, d = eta.shape
    if tp.s2 > 0:
        eta = (eta[:, None, :] + math.sqrt(tp.s2) * _GX[None, :, None]).reshape(n, -1)  # column k * d + j
        return expit(eta), np.repeat(_GW, d) / d
    return expit(eta), np.full(d, 1.0 / d)


def _success_update(prep: _Prepared, ei: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Bayes for one simulated verdict on configuration `ei`: `P(z)` for z = 1, 0 (2,), every
    configuration's `g` now (n,) and after each verdict (2, n). Each draw's weight splits between the two
    verdicts in proportion to their likelihoods, which sum to 1, so `P(z) @ g_after` is `g_now` exactly."""
    p, w = prep.chance, prep.weight
    q = LOOKAHEAD_Q
    pe = p[ei]
    joint = np.stack([q * pe + (1.0 - q) * (1.0 - pe), (1.0 - q) * pe + q * (1.0 - pe)]) * w  # (2, m)
    pz = joint.sum(axis=1)
    return pz, p @ w, (joint @ p.T) / pz[:, None]


def _prepare(state: FitState, task: Task, configs: list[Configuration], rule: AcceptanceRule | None) -> _Prepared:
    key = ("lookahead", tuple(state.task_terms(task)), tuple(c.id for c in configs),
           repr(rule))
    cached = state._cache.get(key)
    if cached is not None:
        return cached
    out = state._predict_many(task, configs, rule, None)
    tt = state.task_terms(task)
    cost_rows, coef, owner_r, owner_c, run_rows, plans = [], [], [], [], [], []
    for ci, (cfg, (_, d)) in enumerate(zip(configs, out)):
        plan = state._plan(cfg)
        plans.append(plan)
        st = plan["st"]
        reach = d["reach"]  # E[r(k)] with the unseen task effect on the gates integrated out
        for (piece, k), rest in plan["cost"].items():
            owner_r.append(ci)
            owner_c.append(len(cost_rows))
            cost_rows.append(tt + list(rest))
            coef.append(st.widths[piece] * reach[(piece, k)])
        run_rows.append(tt + list(plan["run"]))
    owner = sparse.csr_matrix((np.ones(len(owner_r)), (owner_r, owner_c)), shape=(len(configs), len(cost_rows)))
    score_name = None
    if rule is not None and rule.score is not None and f"score:{rule.score.name}" in state.heads:
        if state._score_support(rule.score.name, task) >= MIN_SCORE_SWITCH:
            score_name = rule.score.name
    prep = _Prepared(configs, {c.id: i for i, c in enumerate(configs)},
                     _Rows.build(state.heads["cost"], cost_rows), np.array(coef), owner,
                     _Rows.build(state.heads["success"], run_rows),
                     _Rows.build(state.heads[f"score:{score_name}"], run_rows) if score_name else None, score_name)
    if score_name is None:
        prep.chance, prep.weight = _success_draws(state, task, plans)
    state._cache[key] = prep
    return prep


def _explore_cost_obs(state: FitState, task: Task, config: Configuration) -> tuple[list[tuple], int]:
    """Composite observation: the mean of the round-1 cost rows of the configuration's pieces,
    and the number of pieces (the observation's noise variance is sigma^2 / n)."""
    plan = state._plan(config)
    st = plan["st"]
    tt = state.task_terms(task)
    acc: dict[str, list] = {}
    n = len(st.pieces)
    for piece in st.pieces:
        for node, parent, value in tt + list(cost_rest(st, piece, 1, source=PREDICT_SOURCE)):
            if node in acc:
                acc[node][1] += value / n
            else:
                acc[node] = [parent, value / n]
    terms = [(node, p, v) for node, (p, v) in acc.items()]
    return terms, n


def _score_g(state: FitState, rows: _Rows, name: str, rule: AcceptanceRule, mu, var) -> np.ndarray:
    head = rows.head
    info = state.score_info(name)
    scale = info.get("scale") or "linear"
    better = rule.score.better or info.get("better") or "higher"
    tt = transform(float(rule.score.target), scale)
    if tt is None:
        return np.zeros_like(mu)
    z = (head.center + head.scale * mu - tt) / (head.scale * np.sqrt(np.maximum(var, 0.0) + head.sigma ** 2))
    return ndtr(z) if better == "higher" else ndtr(-z)


@dataclass
class _Outcomes:
    """One simulated run of `explore`: the outcomes' weights `W` (x, o) over the cost grid x and the verdict or
    score grid o, and every candidate's cost, `g` and `ell` now (c,) and after each outcome."""

    ei: int
    W: np.ndarray
    pz: np.ndarray  # (o,)
    cost_now: np.ndarray
    cost_after: np.ndarray  # (x, c)
    g_now: np.ndarray
    g_after: np.ndarray  # (o, c)
    ell_now: np.ndarray
    ell: np.ndarray  # (x, o, c)
    score_now: np.ndarray | None = None
    s_after: np.ndarray | None = None  # (o, c)

    def raw_gain(self) -> float:
        """`min_c ell_now(c) - E[min_c ell_after(c)]`, before `G` clips it at 0."""
        return float(self.ell_now.min() - np.sum(self.W * self.ell.min(axis=2)))


def _outcomes(state: FitState, task: Task, explore: Configuration, goal: Configuration,
              candidates: Sequence[Configuration], rule: AcceptanceRule | None, rescue_usd: float | None) -> _Outcomes:
    configs: list[Configuration] = []
    seen: set[str] = set()
    for c in list(candidates) + [explore, goal]:
        if c.id not in seen:
            seen.add(c.id)
            configs.append(c)
    prep = _prepare(state, task, configs, rule)
    R = float(rescue_usd or 0.0)
    cost = prep.cost
    sig2 = cost.head.sigma ** 2

    def expected_cost(mu, var):
        """Expected cost per candidate; a 2-D `mu` (rows, k), one column per outcome, gives (k, candidates)."""
        e = prep.coef[:, None] * np.exp(mu.reshape(len(mu), -1) + 0.5 * (var + sig2)[:, None])
        out = np.asarray(prep.owner @ e).T
        return out if mu.ndim == 2 else out[0]

    # the observation of explore's cost
    obs_terms, n_obs = _explore_cost_obs(state, task, explore)
    _, o_var, kappa = cost.observation(obs_terms)
    s_c = o_var + sig2 / n_obs
    var_after_c = cost.var - kappa * kappa / s_c
    cost_now = expected_cost(cost.mu, cost.var)
    cost_after = expected_cost(cost.mu[:, None] + np.outer(kappa, _QX) / math.sqrt(s_c), var_after_c)  # (x, c)

    # the outcome of explore's run: weights (o,), g after (o, c), score after (o, c) or None
    ei = prep.index[explore.id]
    score_now = s_after = None
    if prep.score is not None:
        sr = prep.score
        kap = _row_cov(sr, ei)
        s_s = float(sr.var[ei]) + sr.head.sigma ** 2
        g_now = _score_g(state, sr, prep.score_name, rule, sr.mu, sr.var)
        score_now = _score_raw(state, sr, prep.score_name, sr.mu)
        v_after = sr.var - kap * kap / s_s
        mu_after = sr.mu[None, :] + np.outer(_QX, kap) / math.sqrt(s_s)
        pz = _QW
        g_after = _score_g(state, sr, prep.score_name, rule, mu_after, v_after)
        s_after = _score_raw(state, sr, prep.score_name, mu_after)
    else:
        pz, g_now, g_after = _success_update(prep, ei)

    ell_now = cost_now + (1.0 - g_now) * R
    W = _QW[:, None] * pz[None, :]  # (x, o)
    ell = cost_after[:, None, :] + (1.0 - g_after[None, :, :]) * R  # (x, o, c)
    return _Outcomes(ei, W, pz, cost_now, cost_after, g_now, g_after, ell_now, ell, score_now, s_after)


def lookahead(state: FitState, task: Task, explore: Configuration, goal: Configuration,
              candidates: Sequence[Configuration], rule: AcceptanceRule | None = None, *,
              rescue_usd: float | None = None) -> LookaheadResult:
    o = _outcomes(state, task, explore, goal, candidates, rule, rescue_usd)
    W, ell, g_now, g_after, cost_now, cost_after = o.W, o.ell, o.g_now, o.g_after, o.cost_now, o.cost_after
    best_now = int(np.argmin(o.ell_now))
    j = np.argmin(ell, axis=2)
    ix, io = np.indices(j.shape)
    exp_min = float(np.sum(W * ell[ix, io, j]))
    exp_g = float(np.sum(W * g_after[io, j]))
    exp_c = float(np.sum(W * cost_after[ix, j]))
    gain = max(0.0, float(o.ell_now[best_now]) - exp_min)
    gains: dict[str, float | None] = {
        "usd": gain,
        "success_pp": 100.0 * (exp_g - float(g_now[best_now])),
        "cost_pct": 100.0 * (exp_c - float(cost_now[best_now])) / float(cost_now[best_now])
        if cost_now[best_now] > 0 else None,
        "score": None,
    }
    if o.score_now is not None:
        gains["score"] = float(np.sum(W * o.s_after[io, j])) - float(o.score_now[best_now])
    ell_d = state.ell_draws(task, [explore, goal], rule, rescue_usd=rescue_usd)
    p_beats = float(np.mean(ell_d[0] < ell_d[1]))
    shift = {"p_success": _spread(o.pz, g_after[:, o.ei]), "cost_usd": _spread(_QW, cost_after[:, o.ei])}
    return LookaheadResult(gain_per_run=gains, p_beats_goal=p_beats, posterior_shift=shift)


def _row_cov(rows: _Rows, i: int) -> np.ndarray:
    """Covariance of every row with row i (known part through U, unseen part diagonal)."""
    kap = rows.B @ rows.B[i] if rows.B.shape[1] else np.zeros(len(rows.mu))
    if rows.Xv.shape[1]:
        xi = rows.Xv[[i], :].toarray().ravel()
        kap = kap + np.asarray(rows.Xv @ (rows.vphi2 * xi)).ravel()
    return kap


def _score_raw(state: FitState, rows: _Rows, name: str, mu: np.ndarray) -> np.ndarray:
    """Mean score in raw units (the transformed mean mapped back)."""
    scale = state.score_info(name).get("scale") or "linear"
    return np.asarray(inverse_transform(rows.head.center + rows.head.scale * mu, scale), float)


def _spread(w: np.ndarray, v: np.ndarray) -> Interval:
    """Weighted mean and 10th and 90th percentiles of an outcome-weighted quantity."""
    w = np.asarray(w, float) / max(float(np.sum(w)), 1e-300)
    v = np.asarray(v, float)
    order = np.argsort(v)
    cdf = np.cumsum(w[order])
    lo, hi = np.interp([0.1, 0.9], cdf - 0.5 * w[order], v[order])
    return Interval(mean=float(np.sum(w * v)), lo=float(lo), hi=float(hi))


# ---------------------------------------------------------------- conditioned

def _condition_head(head: HeadState, terms: Sequence[tuple], tau2: float) -> HeadState:
    """Variance-only rank-one update of one head for one observation row with noise `tau2`."""
    kx: dict[int, float] = {}
    new_nodes: list[str] = []
    new_vals: list[float] = []
    for node, _parent, value in terms:
        j = head.index.get(node)
        if j is not None:
            kx[j] = kx.get(j, 0.0) + value
        elif head.node_scale(node) > 0:
            if node in new_nodes:
                new_vals[new_nodes.index(node)] += value
            else:
                new_nodes.append(node)
                new_vals.append(value)
    p, q = len(head.nodes), len(new_nodes)
    phis = np.array([head.node_scale(n) for n in new_nodes])
    mean = np.concatenate([head.mean, np.zeros(q)])
    U = np.zeros((p + q, p + q))
    if p:
        U[:p, :p] = head.U
    U[p:, p:] = np.diag(phis)
    x = np.zeros(p + q)
    for j, v in kx.items():
        x[j] = v
    x[p:] = new_vals
    v = U.T @ x
    t = float(v @ v)
    if t > 0:
        alpha = (1.0 - math.sqrt(tau2 / (t + tau2))) / t
        U = U - alpha * np.outer(U @ v, v)
    Z = np.zeros((p + q, N_DRAWS))
    if p:
        Z[:p] = head.unit
    for i, node in enumerate(new_nodes):
        Z[p + i] = head.virtual_unit(node)
    draws = mean[:, None] + U @ Z
    meta = dict(head.meta) or {"engine": head.engine, "kind": head.kind, "sigma": head.sigma}
    out = HeadState(head.name, meta, head.nodes + new_nodes, None, head.fit_id, mean=mean, U=U, draws=draws,
                    unit=Z)
    out.fitted = head.fitted
    out.phi = dict(head.phi)
    return out


def conditioned(state: FitState, task: Task, config: Configuration, rule: AcceptanceRule | None = None) -> FitState:
    heads = dict(state.heads)
    terms, n_obs = _explore_cost_obs(state, task, config)
    for name in ("cost", "tokens"):
        heads[name] = _condition_head(state.heads[name], terms, state.heads[name].sigma ** 2 / n_obs)
    run_terms = state.task_terms(task) + list(state._plan(config)["run"])
    succ = state.heads["success"]
    heads["success"] = _condition_head(succ, run_terms, 1.0 / _info(float(_Rows.build(succ, [run_terms]).mu[0])))
    if rule is not None and rule.score is not None and f"score:{rule.score.name}" in state.heads:
        sh = state.heads[f"score:{rule.score.name}"]
        heads[sh.name] = _condition_head(sh, run_terms, sh.sigma ** 2)
    return FitState(state.path, meta=state.meta, heads=heads, design=state.design)
