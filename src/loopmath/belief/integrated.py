"""The opt-in success head (config `belief.success_head = "sampled"`): its scales integrated out, not plugged in.

0.3.1's success head sets each scale phi by empirical Bayes and plugs it in as known. A scale that one factor node
or a few role nodes carry cannot be learned from them, and for binary outcomes the plugged-in scale is biased low,
so effects the data do measure are shrunk toward 0 and their ranges miss. Under the key:

- `integrate_success` refits the success head by full Bayes, NUTS over the effects and the log scales together
  (sampled.py), and stores its draws with each draw's scales (`scale_draws`), so an unseen node's draw uses the
  scale its draw came from. An importance sampler over the scales with a Laplace Gaussian at each (an earlier
  candidate) is not offered: near the ceiling the Gaussian at the mode puts a measured effect's mean low, and its
  held-out log loss was no better than the default head's.
- `node_baselines`: the linear predictor each node's success effect is shown at, the mean over the rows that carry
  the node of the predictor without the node's own effect (`own`) or its chain's (`chain`), instead of the head's
  mean over every row (a p of 0.995 on the shipped prior, where a log-odds unit is worth half a point).
- `identified`: whether some contrast among a node's own level that involves it is estimable from the head's rows,
  every other term free. A role inside its shape's span is not: only the prior splits the shape's effect.

With the key unset none of this runs, and every fit is 0.3.1's.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, sparse

from .forest import level_of
from .gaussian import LogisticHead

SUCCESS_HEAD_KEY = "belief.success_head"
METHODS = ("eb", "sampled")
FLAG_LEVELS = ("role", "role_family", "harness", "hfactor", "topology")


def success_method(config: dict) -> str:
    """The success head's method from a config.toml table: `eb` (0.3.1) unless `belief.success_head` says else."""
    value = (config.get("belief") or {}).get("success_head") if isinstance(config.get("belief"), dict) else None
    return value if value in METHODS else "eb"


# ---------------------------------------------------------------- the head


def integrate_success(h: dict, seed: int, parents: dict[str, str | None]) -> dict[str, np.ndarray]:
    """Refit the success head of `fit_head`'s result `h` by NUTS (sampled.py), in place (its fit, phi and
    baseline), and return the extra state arrays: the draws, each draw's scales, the display baselines and the
    identification flags, under the head's state key."""
    from .sampled import SampledLogisticHead

    X, z, q, prior, factors = h["design"]
    res, D, S = SampledLogisticHead(LogisticHead(X, z, q, prior, factors)).fit(seed=seed, eb=h["fit"])
    h["fit"] = res
    h["phi"] = {g: float(v) for g, v in zip(prior.groups, res.phi)}
    h["baseline"] = float(np.mean(X @ res.mean)) if X.shape[0] else 0.0
    own, chain = node_baselines(X, res.mean, h["node_ids"], parents)
    return {"success__draws": D, "success__scale_draws": S, "success__base_own": own, "success__base_chain": chain,
            "success__identified": identified(X, h["node_ids"])}


# ---------------------------------------------------------------- display baselines


def node_baselines(X: sparse.csr_matrix, mean: np.ndarray, node_ids: list[str],
                   parents: dict[str, str | None]) -> tuple[np.ndarray, np.ndarray]:
    """Per node, the mean over the rows that carry it of the linear predictor less its own effect (`own`) and
    less its chain's, itself and its ancestors in the head (`chain`). A node on no row gets the head's mean."""
    X = sparse.csc_matrix(X)
    eta = np.asarray(X @ mean).ravel()
    index = {n: j for j, n in enumerate(node_ids)}
    head_mean = float(eta.mean()) if len(eta) else 0.0
    own = np.full(len(node_ids), head_mean)
    chain = np.full(len(node_ids), head_mean)
    for j, node in enumerate(node_ids):
        rows = X.indices[X.indptr[j]:X.indptr[j + 1]]
        if not len(rows):
            continue
        sub = X[rows]
        own[j] = float(np.mean(eta[rows] - sub[:, j].toarray().ravel() * mean[j]))
        idx, cur = [j], parents.get(node)
        while cur:
            k = index.get(cur)
            if k is not None:
                idx.append(k)
            cur = parents.get(cur)
        chain[j] = float(np.mean(eta[rows] - np.asarray(sub[:, idx] @ mean[idx]).ravel()))
    return own, chain


# ---------------------------------------------------------------- identification


def identified(X: sparse.csr_matrix, node_ids: list[str], levels=FLAG_LEVELS) -> np.ndarray:
    """Per node: 1 when some contrast among the nodes of its level that involves it (the node alone included) is
    estimable from the rows, every other term free; 0 when none is; -1 for levels not checked.

    Estimable means in the row space of X. Row-disjoint columns outside the checked levels (tasks, mostly) are
    projected out first, which leaves the estimable functions of the other columns unchanged; the null space of
    what remains, restricted to a level, spans the directions no row can see."""
    p = len(node_ids)
    out = np.full(p, -1, dtype=np.int8)
    lv = [level_of(n) for n in node_ids]
    checked = np.array([x in levels for x in lv])
    if not checked.any() or X.shape[0] == 0:
        return out
    from .block import leaf_split

    X = sparse.csr_matrix(X)
    aX = abs(X)
    perm, nleaf = leaf_split(aX.T @ aX)
    leaves = [j for j in perm[:nleaf] if not checked[j]]
    core = [j for j in range(p) if j not in set(leaves)]
    Xc = X[:, core].toarray()
    Xl = sparse.csc_matrix(X[:, leaves])
    for c in range(len(leaves)):  # project each row-disjoint leaf column out
        rows = Xl.indices[Xl.indptr[c]:Xl.indptr[c + 1]]
        v = Xl.data[Xl.indptr[c]:Xl.indptr[c + 1]]
        vv = float(v @ v)
        if vv > 0:  # a column of stored zeros (a numeric term at 0 on its rows) has nothing to project out
            Xc[rows] -= np.outer(v, (v @ Xc[rows]) / vv)
    if Xc.shape[0] > Xc.shape[1]:  # R has X's row space and null space, at p x p size (no n x n U)
        Xc = np.linalg.qr(Xc, mode="r")
    _, s, Vt = linalg.svd(Xc, full_matrices=True, check_finite=False)
    tol = 1e-9 * (s[0] if len(s) and s[0] > 0 else 1.0)
    rank = int(np.sum(s > tol))
    N = Vt[rank:].T  # null space, core coordinates
    pos = {j: i for i, j in enumerate(core)}
    for level in levels:
        js = [j for j in range(p) if lv[j] == level]
        if not js:
            continue
        if N.shape[1] == 0:
            out[js] = 1
            continue
        NL = N[[pos[j] for j in js]]
        U, sl, _ = linalg.svd(NL, full_matrices=False, check_finite=False)
        r = int(np.sum(sl > 1e-9 * max(1.0, sl[0] if len(sl) else 1.0)))
        Q = U[:, :r]
        for i, j in enumerate(js):
            out[j] = int(1.0 - float(Q[i] @ Q[i]) > 1e-8)
    return out
