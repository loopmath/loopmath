"""BeliefState: load fits/latest and answer predict, node_summary, lookahead, support.

Spec 04 sections 3 and 6, spec 03 section 2. `load_latest(home)` reads `meta.json` and opens
`state.npz` lazily, so loading stays well under 300 ms; `design.json` (support counts) is read
on first use. `state.npz` holds each head's sparse posterior precision (D94); the Cholesky
factor, the covariance factor `U` and the 400 draws are rebuilt from it on first use, the draws
from the fit's seed. Fits written before D94 hold `U` and the draws, and load as they are.

Prediction:
- A row's effect is the sum of its nodes' effects. Nodes the head has seen use the fit's 400
  joint posterior draws. Nodes it has not seen (a new task, repo or model version) are virtual:
  mean 0 and standard deviation the fitted scale of their level, with draws seeded from the
  node id, so every configuration that shares the node shares its draws. Unseen fixed
  effects (a round or control term without data) contribute 0.
- The task part of every row is computed once per task and head; the configuration part
  (setting, role, topology, position, round, control) is cached per fit, so `predict_many`
  over thousands of candidates reuses rows.
- The composition (compose.py) runs over the draws. Cost intervals are predictive (D35).
- Means (D98) do not sample the unseen nodes: their effects are normal, so each cost and
  tokens row's expected cost is log-normal in closed form over the known draws, the unseen
  task effect on the gates (one normal shared by every gate row of the task) and on success
  or a score are integrated by Gauss-Hermite, and a score's reach chance is a normal CDF.
  The `ell.tokens` mean prices the rescue at each draw's own tokens per dollar, as its draws
  do: `E[1 / C_run]` is integrated over every unseen cost effect, the task's and each piece's
  (a new model, role or position), in closed form when the pieces share them and otherwise
  over their differences, by Gauss-Hermite up to four axes and a fixed Sobol set beyond, never
  more than 256 points (D106, D108).
  A new task's means therefore do not depend on its id, and the Monte Carlo error of 400
  seeded draws of a large unseen effect (task, repo, org) leaves them. Intervals, the draws
  (`success_draws`, `ell_draws`) and the simulated runs keep the sampled effects. Unseen
  configuration nodes (a new model version) in gate rows stay sampled in the means too.
- `g` comes from the success head, or from a score head when the rule is a score target and
  that head has at least 5 runs in the task's type (spec 04 section 2).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import linalg, sparse
from scipy.special import expit, ndtr, ndtri

from ..types import (
    AcceptanceRule, BeliefState, Configuration, Interval, LookaheadResult, Money, NodeSummary, PiecePrediction,
    Prediction, ScorePrediction, Task,
)
from .compose import Intervals, RunDraws, compose, cost_per_round, interval
from .design import Structure, cost_rest, gate_rest, run_rest, structure, task_terms
from .forest import LEVEL_ALIASES, default_scale, display_level, level_of, node_key, scale_group
from .gaussian import _cho, cov_factor

N_DRAWS = 400
PREDICT_SOURCE = "user"
MIN_SCORE_SWITCH = 5  # spec 04 section 2: g = p_reach once the score head has 5 runs in the task's type
BASE_HEADS = {"cost": ("gaussian", "cost"), "tokens": ("gaussian", "cost"), "gate": ("logistic", "logit"),
              "success": ("logistic", "logit")}
TASK_CHAIN = ("task", "subtype", "repo", "type")
FIXED_LEVELS = ("fixed", "round", "control")
SCORE_CLAMP = (0.005, 0.995)
_CACHE_LIMIT = 200_000
MEAN_NODES = 12  # D98: Gauss-Hermite nodes over an unseen normal effect
_GX, _GW = hermegauss(MEAN_NODES)
_GW = _GW / _GW.sum()


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("/".join(parts).encode()).digest()[:8], "little")


def _iv(t: tuple[float, float, float]) -> Interval:
    return Interval(mean=t[0], lo=t[1], hi=t[2])


def transform(value: float, scale: str) -> float | None:
    """Score response scale (spec 04 section 2): identity, natural log, or clamped logit."""
    if value is None or not math.isfinite(value):
        return None
    if scale == "log":
        return math.log(value) if value > 0 else None
    if scale == "fraction":
        v = min(max(value, SCORE_CLAMP[0]), SCORE_CLAMP[1])
        return math.log(v / (1.0 - v))
    return float(value)


def inverse_transform(t, scale: str):
    if scale == "log":
        return np.exp(t)
    if scale == "fraction":
        return expit(t)
    return t


class Row(NamedTuple):
    """One row's linear predictor: posterior mean, joint draws, the draws of its known nodes
    only, and the variance of its unseen nodes' effect (normal with mean 0)."""

    mu: float
    draws: np.ndarray
    known: np.ndarray
    s2: float


def normal_mean(f: Callable[[np.ndarray], np.ndarray], eta: np.ndarray, s2: float) -> np.ndarray:
    """`E f(eta + e)` per draw for `e ~ N(0, s2)`, by Gauss-Hermite (D98)."""
    if s2 <= 0:
        return f(eta)
    return _GW @ f(eta[None, :] + math.sqrt(s2) * _GX[:, None])


MAX_POINTS = 256  # D108: the most points `_rule` gives, whatever the number of axes
_GRID_NODES = {1: MEAN_NODES, 2: 8, 3: 6, 4: 4}  # Gauss-Hermite nodes per axis; nodes ** k <= MAX_POINTS
_SOBOL_SEED = 106
_CHUNK = 64  # points evaluated at once in `_inv_cost`


@lru_cache(maxsize=32)
def _rule(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Points (Q, k) and weights (Q,) for `E f(z)`, `z ~ N(0, I_k)`, with `Q <= MAX_POINTS` (D108):
    a Gauss-Hermite product grid up to four axes; from five on, a scrambled Sobol set with a fixed
    seed, whose leading coordinates are the most even (callers put the largest axis first)."""
    nq = _GRID_NODES.get(k)
    if k == 0:
        return np.zeros((1, 0)), np.ones(1)
    if nq is not None:
        x, w = hermegauss(nq)
        w = w / w.sum()
        pts = np.array(list(itertools.product(x, repeat=k)))
        wts = np.prod(np.array(list(itertools.product(w, repeat=k))), axis=1)
        return pts, wts
    from scipy.stats import qmc  # only here: importing scipy.stats takes about 0.3 s

    u = qmc.Sobol(d=k, scramble=True, seed=_SOBOL_SEED).random(MAX_POINTS)
    return ndtri(u), np.full(MAX_POINTS, 1.0 / MAX_POINTS)


class HeadState:
    """One head's posterior: node ids, mean, covariance factor and joint draws (lazy).

    `loader(part)` returns a stored array or None when the fit did not store it."""

    def __init__(self, name: str, meta: dict, nodes: Sequence[str], loader: Callable[[str], np.ndarray] | None,
                 fit_id: str, *, mean=None, U=None, draws=None, unit=None):
        base = BASE_HEADS.get(name, ("gaussian", "score"))
        self.name = name
        self.engine = meta.get("engine") or base[0]
        self.kind = meta.get("kind") or base[1]
        self.sigma = float(meta.get("sigma", 1.0))
        self.center = float(meta.get("center", 0.0))
        self.scale = float(meta.get("scale", 1.0))
        self.baseline = float(meta.get("baseline", 0.0))
        self.phi: dict[str, float] = dict(meta.get("phi") or {})
        self.seed = int(meta.get("seed", 0))
        self.meta = meta
        self.fit_id = fit_id
        self.nodes = list(nodes)
        self.index = {n: j for j, n in enumerate(self.nodes)}
        self._loader = loader
        self._mean, self._U, self._draws, self._unit = mean, U, draws, unit
        self._chol: np.ndarray | None = None
        self._virtual: dict[str, np.ndarray] = {}
        self.fitted = bool(meta)

    def _load(self, part: str) -> np.ndarray | None:
        if self._loader is None:
            p = len(self.nodes)
            return np.zeros(p) if part == "mean" else np.zeros((p, p if part == "U" else N_DRAWS))
        return self._loader(part)

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            self._mean = self._load("mean")
        return self._mean

    @property
    def chol(self) -> np.ndarray:
        """Lower Cholesky factor of the posterior precision, as the fit computed it (D94)."""
        if self._chol is None:
            p = len(self.nodes)
            P = sparse.csr_matrix((self._load("P_data"), self._load("P_indices"), self._load("P_indptr")),
                                  shape=(p, p))
            self._chol = _cho(P.toarray())
        return self._chol

    @property
    def U(self) -> np.ndarray:
        if self._U is None:
            U = self._load("U")
            self._U = U if U is not None else cov_factor(self.chol)
        return self._U

    @property
    def draws(self) -> np.ndarray:
        if self._draws is None:
            D = self._load("draws")
            if D is None:  # mean + U unit, as a triangular solve (U = chol^-T)
                D = self.mean[:, None] + linalg.solve_triangular(self.chol, self.unit, trans="T", lower=True,
                                                                  check_finite=False)
            self._draws = D
        return self._draws

    @property
    def unit(self) -> np.ndarray:
        """The standard normals behind `draws` (draws = mean + U unit), as the fit drew them."""
        if self._unit is None:
            self._unit = np.random.default_rng(self.seed).standard_normal((len(self.nodes), N_DRAWS))
        return self._unit

    def node_scale(self, node: str) -> float:
        group = scale_group(node)
        if group is None:
            return 0.0
        return float(self.phi.get(group, default_scale(group, self.kind)))

    def virtual_unit(self, node: str) -> np.ndarray:
        """Standard normal draws of an unseen node, fixed by the fit, the head and the node id."""
        z = self._virtual.get(node)
        if z is None:
            z = np.random.default_rng(_seed(self.fit_id, self.name, node)).standard_normal(N_DRAWS)
            self._virtual[node] = z
        return z

    def matrices(self, rows: Sequence[Sequence[tuple]]):
        """Known-node matrix, virtual-node matrix, virtual node ids and their scales."""
        n = len(rows)
        ri, ci, vals, vr, vc, vv = [], [], [], [], [], []
        vindex: dict[str, int] = {}
        index = self.index
        for r, terms in enumerate(rows):
            for node, _parent, value in terms:
                j = index.get(node)
                if j is not None:
                    ri.append(r)
                    ci.append(j)
                    vals.append(value)
                elif scale_group(node) is not None:
                    q = vindex.get(node)
                    if q is None:
                        q = vindex[node] = len(vindex)
                    vr.append(r)
                    vc.append(q)
                    vv.append(value)
        Xk = sparse.csr_matrix((vals, (ri, ci)), shape=(n, len(self.nodes)))
        vnodes = list(vindex)
        Xv = sparse.csr_matrix((vv, (vr, vc)), shape=(n, len(vnodes)))
        vphi = np.array([self.node_scale(v) for v in vnodes])
        return Xk, Xv, vnodes, vphi

    def eval_rows(self, rows: Sequence[Sequence[tuple]]) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean (n,) and joint draws (n, N_DRAWS) of each row's linear predictor."""
        mu, D, _, _ = self.eval_parts(rows)
        return mu, D

    def eval_parts(self, rows: Sequence[Sequence[tuple]]):
        """`eval_rows`, plus the draws of the known nodes only (n, N_DRAWS) and the variance of
        the unseen nodes' effect (n,)."""
        Xk, Xv, vnodes, vphi = self.matrices(rows)
        n = len(rows)
        mu = np.zeros(n)
        D = np.zeros((n, N_DRAWS))
        s2 = np.zeros(n)
        if Xk.nnz:
            mu = np.asarray(Xk @ self.mean).ravel()
            D = np.asarray(Xk @ self.draws)
        known = D
        if vnodes:
            V = np.stack([self.virtual_unit(v) for v in vnodes]) * vphi[:, None]
            D = D + np.asarray(Xv @ V)
            s2 = np.asarray(Xv.multiply(Xv) @ (vphi ** 2)).ravel()
        return mu, D, known, s2


class FitState:
    """The belief state of one fit (types.BeliefState)."""

    def __init__(self, path: Path, *, meta: dict | None = None, heads: dict[str, HeadState] | None = None,
                 design: dict | None = None):
        self.path = Path(path)
        self.meta = meta if meta is not None else json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.fit_id: str = self.meta.get("fit") or self.path.name
        self.created_at: str = self.meta.get("created_at", "")
        self._design = design
        self._npz = None
        self._cache: dict = {}
        self.sim_seed = _seed(self.fit_id, "simulate")
        if heads is None:
            heads = {}
            for name, hmeta in (self.meta.get("heads") or {}).items():
                key = name.replace(":", "@")
                ids = [str(x) for x in self._array(f"{key}__ids")]
                heads[name] = HeadState(name, hmeta, ids, self._loader(key), self.fit_id)
        for name in BASE_HEADS:
            if name not in heads:
                heads[name] = HeadState(name, {}, [], None, self.fit_id)
        self.heads = heads

    # ------------------------------------------------------------ loading

    def _file(self):
        if self._npz is None:
            self._npz = np.load(self.path / "state.npz", allow_pickle=False)
        return self._npz

    def _array(self, key: str) -> np.ndarray:
        return self._file()[key]

    def _loader(self, key: str) -> Callable[[str], np.ndarray | None]:
        def load(part: str) -> np.ndarray | None:
            npz = self._file()
            name = f"{key}__{part}"
            return npz[name] if name in npz.files else None

        return load

    @property
    def design(self) -> dict:
        if self._design is None:
            p = self.path / "design.json"
            self._design = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
        return self._design

    @property
    def n_runs(self) -> int:
        return int(sum((self.meta.get("runs_by_source") or {}).values()))

    @property
    def score_heads(self) -> list[str]:
        return [h[len("score:"):] for h in self.heads if h.startswith("score:")]

    # ------------------------------------------------------------ rows

    def _cached(self, head: HeadState, keys: Sequence[tuple]) -> list[Row]:
        cache = self._cache
        missing = [k for k in dict.fromkeys(keys) if (head.name, k) not in cache]
        if missing:
            if len(cache) + len(missing) > _CACHE_LIMIT:
                cache.clear()
            mu, D, known, s2 = head.eval_parts(missing)
            for i, k in enumerate(missing):
                cache[(head.name, k)] = Row(float(mu[i]), D[i], known[i], float(s2[i]))
        return [cache[(head.name, k)] for k in keys]

    def _row(self, head: HeadState, rest: tuple) -> Row:
        return self._cached(head, [rest])[0]

    def _plan(self, config: Configuration) -> dict:
        st = structure(config)
        cost = {}
        for piece in st.pieces:
            rounds = st.k_max if st.piece_loop.get(piece) is not None else 1
            for k in range(1, rounds + 1):
                cost[(piece, k)] = cost_rest(st, piece, k)
        gate = {}
        for gi, g in enumerate(st.gates):
            for k in range(1, (st.k_max if st.gate_loop.get(gi) is not None else 1) + 1):
                gate[(gi, k)] = gate_rest(st, g, k)
        return {"config": config, "st": st, "cost": cost, "gate": gate, "run": run_rest(st)}

    def _task_parts(self, task: Task) -> dict:
        key = tuple(task_terms(task, PREDICT_SOURCE))
        return {name: self._cached(head, [key])[0] for name, head in self.heads.items()}

    def _warm(self, plans: list[dict]) -> None:
        """Evaluate every configuration part the plans need, in one batch per head."""
        cost_keys = [k for p in plans for k in p["cost"].values()]
        gate_keys = [k for p in plans for k in p["gate"].values()]
        run_keys = [p["run"] for p in plans]
        self._cached(self.heads["cost"], cost_keys)
        self._cached(self.heads["tokens"], cost_keys)
        if gate_keys:
            self._cached(self.heads["gate"], gate_keys)
        for name, head in self.heads.items():
            if name == "success" or name.startswith("score:"):
                self._cached(head, run_keys)

    def _eta(self, head: HeadState, task_part: Row, rest: tuple) -> tuple[float, np.ndarray]:
        r = self._row(head, rest)
        return task_part.mu + r.mu, task_part.draws + r.draws

    def _known(self, head: HeadState, task_part: Row, rest: tuple) -> tuple[np.ndarray, float]:
        """A row's known-node draws and its unseen nodes' variance (task and configuration parts)."""
        r = self._row(head, rest)
        return task_part.known + r.known, task_part.s2 + r.s2

    def _unseen(self, head: HeadState, rest: tuple) -> tuple[tuple[str, float], ...]:
        """The unseen nodes of a row's configuration part, each with its value times its level's
        scale: the row's unseen effect is the sum of these times independent standard normals."""
        key = ("unseen", head.name, rest)
        hit = self._cache.get(key)
        if hit is None:
            load: dict[str, float] = {}
            for node, _parent, value in rest:
                if node not in head.index and scale_group(node) is not None:
                    load[node] = load.get(node, 0.0) + value * head.node_scale(node)
            hit = self._cache[key] = tuple(sorted((n, b) for n, b in load.items() if b != 0.0))
        return hit

    # ------------------------------------------------------------ prediction

    def _score_support(self, name: str, task: Task) -> int:
        by_type = (self.design.get("score_support_by_type") or {}).get(f"score:{name}") or {}
        return int(by_type.get(task.type, 0))

    def _config_support(self, task: Task, config: Configuration) -> int:
        counts = (self.design.get("config_support") or {}).get(config.id) or {}
        if not counts:
            return 0
        chain = {level_of(n): n for n, _, _ in task_terms(task, PREDICT_SOURCE)}
        for level in TASK_CHAIN:
            node = chain.get(level)
            if node and counts.get(node):
                return int(counts[node])
        return int(counts.get("all", 0))

    def score_info(self, name: str) -> dict:
        return (self.meta.get("scores") or {}).get(name) or {}

    def _scores(self, task: Task, plan: dict, rule: AcceptanceRule | None, tparts: dict, rng
                ) -> tuple[dict, dict, dict]:
        """Score predictions and, per score name, the draws of p_reach when the rule targets it and
        (D98) p_reach per known draw with the unseen effect integrated out."""
        out: dict[str, ScorePrediction] = {}
        reach: dict[str, np.ndarray] = {}
        reach_mean: dict[str, np.ndarray] = {}
        target = rule.score if rule is not None else None
        for name in self.score_heads:
            support = self._score_support(name, task)
            wanted = target is not None and target.name == name
            if support <= 0 and not wanted:
                continue
            head = self.heads[f"score:{name}"]
            _, eta = self._eta(head, tparts[head.name], plan["run"])
            sinfo = self.score_info(name)
            scale = sinfo.get("scale") or "linear"
            better = (target.better if wanted and target.better else sinfo.get("better")) or "higher"
            t_draws = head.center + head.scale * eta
            sd = head.scale * head.sigma
            sim = t_draws + sd * rng.standard_normal(len(t_draws))
            raw = np.asarray(inverse_transform(sim, scale), float)
            known, s2 = self._known(head, tparts[head.name], plan["run"])
            t_known = head.center + head.scale * known
            spread2 = head.scale ** 2 * s2  # the unseen effect on the transformed score
            p_reach = None
            if wanted:
                tt = transform(float(target.target), scale)
                if tt is not None:
                    z = (t_draws - tt) / max(sd, 1e-12)
                    reach[name] = ndtr(z) if better == "higher" else ndtr(-z)
                    zm = (t_known - tt) / max(math.sqrt(sd * sd + spread2), 1e-12)
                    reach_mean[name] = ndtr(zm) if better == "higher" else ndtr(-zm)
                    p_reach = float(np.mean(reach_mean[name]))
            # the mean of one run's score: the unseen effect and the run noise integrated out
            if scale == "log":
                mean_raw = float(np.mean(np.exp(t_known + 0.5 * (sd * sd + spread2))))
            elif scale == "fraction":
                mean_raw = float(np.mean(normal_mean(expit, t_known, sd * sd + spread2)))
            else:
                mean_raw = float(np.mean(t_known))
            out[name] = ScorePrediction(name=name, unit=sinfo.get("unit"), better=better,
                                        value=_iv(interval(raw, mean_raw)), p_reach=p_reach, support=support)
        return out, reach, reach_mean

    def _mean_run(self, plan: dict, tparts: dict) -> tuple[RunDraws, Callable[[np.ndarray], np.ndarray]]:
        """D98: the composition's expectations with the unseen effects integrated out, and the map
        from its arrays to one value per known draw.

        A cost or tokens row enters the composition only through `exp(eta)`, times weights that
        depend on the gates alone, so its unseen effect `N(0, s2)` becomes `eta + s2 / 2`. The
        unseen task effect on the gates is one normal added to every gate row, integrated by
        Gauss-Hermite (the loop's reach is not linear in it): the draws are tiled once per node.
        """
        heads = self.heads
        st: Structure = plan["st"]

        def lognormal(name: str, rest: tuple) -> np.ndarray:
            known, s2 = self._known(heads[name], tparts[name], rest)
            return known + 0.5 * s2

        cost = {k: lognormal("cost", r) for k, r in plan["cost"].items()}
        tok = {k: lognormal("tokens", r) for k, r in plan["cost"].items()}
        tg: Row = tparts["gate"]
        gate = {k: tg.known + self._row(heads["gate"], r).draws for k, r in plan["gate"].items()}
        nodes = MEAN_NODES if gate and tg.s2 > 0 else 1
        if nodes > 1:
            shift = math.sqrt(tg.s2) * _GX[:, None]
            gate = {k: (v[None, :] + shift).ravel() for k, v in gate.items()}
            cost = {k: np.tile(v, nodes) for k, v in cost.items()}
            tok = {k: np.tile(v, nodes) for k, v in tok.items()}
        run = compose(st, cost, tok, gate, sigma_usd=heads["cost"].sigma, sigma_tokens=heads["tokens"].sigma, rng=None)
        if nodes == 1:
            return run, lambda a: a
        return run, lambda a: _GW @ np.reshape(a, (nodes, -1))

    def _inv_cost(self, plan: dict, tparts: dict, mr: RunDraws) -> np.ndarray:
        """D106: `E[1 / C_run]` per entry of the mean composition `mr`, every unseen cost effect
        integrated out (the task's, and a new model's, role's or position's).

        `C_run = sum over pieces of A_p exp(u_p)`: `A_p` is the piece's expected cost at its known
        nodes (its `mr` cost without the log-normal factor), `u_p` its unseen effect. Every round of
        a piece has the same unseen nodes (round and control terms are fixed effects), and pieces
        with the same unseen nodes form one group. The task part is common to every group and
        gives `exp(s2 / 2)`. With one group, `E[1 / C] = exp(Var(u) / 2) / A` exactly. Otherwise,
        with `d_G = u_G - u_r` against a reference group `r`, a change of measure gives
        `E[1 / C] = exp(Var(u_r) / 2) E[1 / (A_r + sum A_G exp(d_G))]`, `d ~ N(-Cov(d, u_r), Cov(d))`,
        integrated on the principal axes of `Cov(d)` by `_rule`: at most `MAX_POINTS` points,
        evaluated `_CHUNK` at a time, so memory and work stay bounded for any number of pieces
        (D108). The reference is the group with the largest cost, which bounds the integrand by
        `1 / A_r`.
        """
        head = self.heads["cost"]
        groups: dict[tuple, np.ndarray] = {}
        for piece in plan["st"].pieces:
            load = self._unseen(head, plan["cost"][(piece, 1)])
            a = mr.pieces[piece].exp_usd * math.exp(-0.5 * (tparts["cost"].s2 + sum(b * b for _, b in load)))
            groups[load] = groups[load] + a if load in groups else a
        loads = list(groups)
        A = np.stack([groups[g] for g in loads])
        names = {n: j for j, n in enumerate(sorted({n for g in loads for n, _ in g}))}
        L = np.zeros((len(loads), len(names)))
        for i, g in enumerate(loads):
            for n, b in g:
                L[i, names[n]] = b
        ref = int(np.argmax(A.mean(axis=1)))
        scale = math.exp(0.5 * (tparts["cost"].s2 + float(L[ref] @ L[ref])))
        others = [i for i in range(len(loads)) if i != ref]
        if not others:
            return scale * np.divide(1.0, A[ref], out=np.zeros_like(A[ref]), where=A[ref] > 0)
        D = L[others] - L[ref]
        w, V = np.linalg.eigh(D @ D.T)
        keep = w > 1e-12 * max(float(w.max()), 1e-300)
        pts, wts = _rule(int(keep.sum()))
        axes = (V[:, keep] * np.sqrt(w[keep]))[:, ::-1]  # the largest axis first
        mu = -(D @ L[ref])
        out = np.zeros(A.shape[1])
        for i in range(0, len(wts), _CHUNK):
            S = A[ref][None, :] + np.exp(mu[None, :] + pts[i:i + _CHUNK] @ axes.T) @ A[others]
            out += wts[i:i + _CHUNK] @ np.divide(1.0, S, out=np.zeros_like(S), where=S > 0)
        return scale * out

    def _predict(self, task: Task, plan: dict, tparts: dict, rule: AcceptanceRule | None,
                 rescue_usd: float | None) -> tuple[Prediction, dict]:
        st: Structure = plan["st"]
        cfg: Configuration = plan["config"]
        heads = self.heads
        cost_eta = {k: self._eta(heads["cost"], tparts["cost"], r)[1] for k, r in plan["cost"].items()}
        tok_eta = {k: self._eta(heads["tokens"], tparts["tokens"], r)[1] for k, r in plan["cost"].items()}
        gate_eta = {k: self._eta(heads["gate"], tparts["gate"], r)[1] for k, r in plan["gate"].items()}
        rng = np.random.default_rng(self.sim_seed)
        rd: RunDraws = compose(st, cost_eta, tok_eta, gate_eta, sigma_usd=heads["cost"].sigma,
                               sigma_tokens=heads["tokens"].sigma, rng=rng)
        _, s_eta = self._eta(heads["success"], tparts["success"], plan["run"])
        g = expit(s_eta)
        s_known, s_s2 = self._known(heads["success"], tparts["success"], plan["run"])
        g_mean = normal_mean(expit, s_known, s_s2)  # per known draw (D98)
        success_from = "success_head"
        scores, reach, reach_mean = self._scores(task, plan, rule, tparts, rng)
        if rule is not None and rule.score is not None and rule.score.name in reach:
            if self._score_support(rule.score.name, task) >= MIN_SCORE_SWITCH:
                g, g_mean = reach[rule.score.name], reach_mean[rule.score.name]
                success_from = "score_head"
        rescue = float(rescue_usd or 0.0)
        ell_usd = rd.exp_usd + (1.0 - g) * rescue
        per_usd = np.divide(rd.exp_tokens, rd.exp_usd, out=np.zeros_like(rd.exp_tokens), where=rd.exp_usd > 0)
        ell_tokens = rd.exp_tokens + (1.0 - g) * rescue * per_usd
        # D98: means with the unseen effects integrated out, per known draw, then averaged
        mr, per_draw = self._mean_run(plan, tparts)

        def m(a: np.ndarray) -> float:
            return float(np.mean(per_draw(a)))

        m_usd, m_tok = per_draw(mr.exp_usd), per_draw(mr.exp_tokens)
        # D106: the ell.tokens mean is the expectation of what its draws hold, the rescue priced at
        # the draw's own tokens per dollar. Given the gates, the cost and tokens heads' unseen
        # effects are independent, so E[T / C] = E[T] E[1 / C]: `mr` holds E[T], and `_inv_cost`
        # integrates 1 / C over every unseen cost effect. The gates' unseen task effect stays under
        # Gauss-Hermite. Without a rescue the ratio does not enter (D108).
        m_per_usd = per_draw(mr.exp_tokens * self._inv_cost(plan, tparts, mr)) if rescue > 0 else 0.0
        iv = Intervals()
        handles = {}
        for piece, pd in rd.pieces.items():
            mp = mr.pieces[piece]
            m_pu, m_pt, m_rounds = m(mp.exp_usd), m(mp.exp_tokens), m(mp.rounds)
            hu, ht = iv.add(pd.sim_usd, m_pu), iv.add(pd.sim_tokens, m_pt)
            if st.piece_loop.get(piece) is None:
                hpr = (hu, ht)  # runs once: its cost is its cost per round
            else:
                pr = cost_per_round(pd)
                if pr is None or m_rounds <= 0:
                    hpr = None
                elif pd.round_free:
                    hpr = (iv.add(pr[0][1], m(mp.one_exp_usd)), iv.add(pr[1][1], m(mp.one_exp_tokens)))
                else:
                    hpr = (iv.add(pr[0][1], m_pu / m_rounds), iv.add(pr[1][1], m_pt / m_rounds))
            hg = iv.add(pd.gate_pass, m(mp.gate_pass)) if pd.gate_pass is not None else None
            handles[piece] = (hu, ht, hg, iv.add(pd.rounds, m_rounds), hpr)
        h_run = (iv.add(g, float(np.mean(g_mean))), iv.add(rd.sim_usd, float(np.mean(m_usd))),
                 iv.add(rd.sim_tokens, float(np.mean(m_tok))),
                 iv.add(ell_usd, float(np.mean(m_usd + (1.0 - g_mean) * rescue))),
                 iv.add(ell_tokens, float(np.mean(m_tok + (1.0 - g_mean) * rescue * m_per_usd))),
                 iv.add(rd.rounds, m(mr.rounds)))
        res = [_iv(t) for t in iv.resolve()]
        per_piece = {  # D60: cost is the piece's full-run contribution; cost_per_round one execution
            piece: PiecePrediction(piece=piece, cost=Money(usd=res[hu], tokens=res[ht]),
                                   gate_pass=res[hg] if hg is not None else None, rounds=res[hr],
                                   cost_per_round=Money(usd=res[hpr[0]], tokens=res[hpr[1]]) if hpr else None)
            for piece, (hu, ht, hg, hr, hpr) in handles.items()}
        pred = Prediction(
            config=cfg.id,
            p_success=res[h_run[0]],
            cost=Money(usd=res[h_run[1]], tokens=res[h_run[2]]),
            ell=Money(usd=res[h_run[3]], tokens=res[h_run[4]]),
            rounds=res[h_run[5]],
            per_piece=per_piece,
            support=self._config_support(task, cfg),
            scores=scores,
            success_from=success_from,
        )
        reach_means = {k: m(v) for k, v in mr.reach.items()}
        return pred, {"g": g, "ell": ell_usd, "run": rd, "reach": reach_means, "success_from": success_from}

    def _predict_many(self, task: Task, configs: Sequence[Configuration], rule: AcceptanceRule | None,
                      rescue_usd: float | None) -> list[tuple[Prediction, dict]]:
        plans = [self._plan(c) for c in configs]
        tparts = self._task_parts(task)
        self._warm(plans)
        return [self._predict(task, p, tparts, rule, rescue_usd) for p in plans]

    def predict(self, task: Task, config: Configuration, rule: AcceptanceRule | None = None, *,
                rescue_usd: float | None = None) -> Prediction:
        return self._predict_many(task, [config], rule, rescue_usd)[0][0]

    def predict_many(self, task: Task, configs: Sequence[Configuration], rule: AcceptanceRule | None = None, *,
                     rescue_usd: float | None = None) -> list[Prediction]:
        return [p for p, _ in self._predict_many(task, configs, rule, rescue_usd)]

    def success_draws(self, task: Task, configs: Sequence[Configuration],
                      rule: AcceptanceRule | None = None) -> np.ndarray:
        """D8: the 400 joint draws of `g` per configuration, shape (len(configs), 400)."""
        out = self._predict_many(task, configs, rule, None)
        return np.stack([d["g"] for _, d in out]) if out else np.zeros((0, N_DRAWS))

    def ell_draws(self, task: Task, configs: Sequence[Configuration], rule: AcceptanceRule | None = None, *,
                  rescue_usd: float | None = None) -> np.ndarray:
        """The joint draws of `ell` per configuration (parameter draws, aligned with success_draws)."""
        out = self._predict_many(task, configs, rule, rescue_usd)
        return np.stack([d["ell"] for _, d in out]) if out else np.zeros((0, N_DRAWS))

    # ------------------------------------------------------------ look-ahead

    def lookahead(self, task: Task, explore: Configuration, goal: Configuration,
                  candidates: Sequence[Configuration], rule: AcceptanceRule | None = None, *,
                  rescue_usd: float | None = None) -> LookaheadResult:
        from .lookahead import lookahead

        return lookahead(self, task, explore, goal, candidates, rule, rescue_usd=rescue_usd)

    def conditioned(self, task: Task, config: Configuration, rule: AcceptanceRule | None = None,
                    rescue_usd: float | None = None) -> "FitState":
        """D10: the state after one more run of `config` on `task` observed at its predicted mean.

        Means do not move; variances shrink. `rescue_usd` is accepted for symmetry and unused.
        """
        from .lookahead import conditioned

        return conditioned(self, task, config, rule)

    # ------------------------------------------------------------ summaries

    def support(self, task: Task) -> dict[str, int]:
        """Runs per group level of the task's chain (the most runs any head has at that node)."""
        ts = self.design.get("task_support") or {}
        chain = {level_of(n): n for n, _, _ in task_terms(task, PREDICT_SOURCE)}
        out = {level: int(ts.get(chain[level], 0)) for level in ("task", "subtype", "repo", "type", "org")
               if level in chain}
        out["all"] = self.n_runs
        out["user"] = int((self.meta.get("runs_by_source") or {}).get("user", 0))
        return out

    def node_summary(self, level: str | None = None, head: str | None = None) -> list[NodeSummary]:
        """Every fitted node's effect: its own deviation plus its ancestors' (the node's belief)."""
        parents = {n["id"]: n.get("parent") for n in self.design.get("nodes") or []}
        support = self.design.get("support") or {}
        wanted = set(LEVEL_ALIASES.get(level, (level,))) if level else None
        out: list[NodeSummary] = []
        for name, hs in self.heads.items():
            if (head and name != head) or not hs.fitted or not hs.nodes:
                continue
            draws = hs.draws
            hsup = support.get(name) or {}
            for j, node in enumerate(hs.nodes):
                lvl = display_level(node)
                raw_level = level_of(node)
                if wanted is None:
                    if raw_level in FIXED_LEVELS:
                        continue
                elif not (lvl in wanted or raw_level in wanted or (level == "feature" and raw_level == "feature")):
                    continue
                chain = [j]
                cur = parents.get(node)
                while cur:
                    k = hs.index.get(cur)
                    if k is not None:
                        chain.append(k)
                    cur = parents.get(cur)
                effect, display = self._effect(hs, draws[chain].sum(axis=0))
                n_runs, mix = hsup.get(node) or [0, {}]
                out.append(NodeSummary(level=lvl, key=node_key(node), head=name, effect=effect, display=display,
                                       support=int(n_runs), parent=parents.get(node),
                                       source_mix={str(k): int(v) for k, v in (mix or {}).items()}))
        return out

    def _effect(self, hs: HeadState, eff: np.ndarray) -> tuple[Interval, Interval]:
        """cost: multiplier (x1.4); success and gate: log-odds shift, shown in pp at the baseline;
        score: shift on the modelled scale, shown in raw units (log: ratio; fraction: pp)."""
        if hs.kind == "cost":
            iv = _iv(interval(np.exp(eff)))
            return iv, iv
        if hs.kind == "logit":
            pp = 100.0 * (expit(hs.baseline + eff) - expit(hs.baseline))
            return _iv(interval(eff)), _iv(interval(pp))
        shift = hs.scale * eff
        scale = self.score_info(hs.name[len("score:"):]).get("scale") or "linear"
        if scale == "log":
            disp = np.exp(shift)
        elif scale == "fraction":
            disp = 100.0 * (expit(hs.center + shift) - expit(hs.center))
        else:
            disp = shift
        return _iv(interval(shift)), _iv(interval(disp))


def fit_dir(home: Path) -> Path | None:
    fits = Path(home) / "fits"
    latest = fits / "latest"
    if latest.exists():
        return latest.resolve()
    if not fits.is_dir():
        return None
    done = sorted(p for p in fits.glob("fit_*") if p.is_dir() and not p.name.endswith(".partial")
                  and (p / "meta.json").is_file())
    return done[-1] if done else None


def load_latest(home: Path) -> BeliefState | None:
    """The newest complete fit, or None when there is none."""
    path = fit_dir(home)
    if path is None or not (path / "meta.json").is_file():
        return None
    return FitState(path)


def load(path: Path) -> FitState:
    return FitState(Path(path))
