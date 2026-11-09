"""Pareto frontier and epsilon-frontier matching.

A setting is a point (cost c, quality q). Lower cost and higher quality are better.
Frontier: s is on it iff no other setting has c_t <= c_s and q_t >= q_s with at least one strict, and on an
exact tie in both the setting with the lower index stays (ties go to the cheaper; equal cost, to the better).
Settings below a quality floor are dropped before the frontier is taken.

Every function takes plain arrays. The *_batch functions take K x S arrays (K posterior draws or worlds)
and return K results, so the stop rule and KG never loop over draws in Python.

Match definitions (FRAMEWORK.md section 5), parameters eps_q (absolute quality) and eps_c (relative cost):
  exact:     returned set R equals the true frontier F.
  coverage:  every true frontier point f has some r in R with c_r <= (1 + eps_c) c_f and q_r >= q_f - eps_q,
             where "no r within that budget" counts as quality 0 (so a frontier point with q_f <= eps_q is free).
             This equals "budget regret at most eps_q": at every budget B the best quality R reaches with
             budget (1 + eps_c) B is within eps_q of the best the true frontier reaches with B.
  two_sided: coverage, and every r in R is sound: no setting s eps-dominates it, that is (1 + eps_c) c_s <= c_r and
             q_s >= q_r + eps_q, strict in at least one. With eps_q = eps_c = 0, two_sided is exact (up to exact
             duplicate points). A near-tie at gap 0 is never a soundness risk once eps_q > 0.
Quality and cost of R are always read in the same world as F (truth for the measures, a posterior draw for the
stop rule), never from the estimate that chose R.
"""
import numpy as np

MATCHES = ("exact", "coverage", "two_sided")


def frontier_mask(cost, quality, floor=None):
    """Boolean mask of frontier members for one world (1-D arrays)."""
    return frontier_mask_batch(np.asarray(cost, float)[None, :], np.asarray(quality, float)[None, :], floor)[0]


def frontier_mask_batch(cost, quality, floor=None):
    """K x S boolean frontier masks. Sort by (cost asc, quality desc, index asc); keep strict new maxima."""
    cost = np.asarray(cost, float)
    quality = np.asarray(quality, float)
    K, S = cost.shape
    q = quality if floor is None else np.where(quality >= floor, quality, -np.inf)
    idx = np.broadcast_to(np.arange(S), (K, S))
    order = np.lexsort((idx, -q, cost), axis=1)
    qs = np.take_along_axis(q, order, axis=1)
    run = np.maximum.accumulate(qs, axis=1)
    prev = np.concatenate([np.full((K, 1), -np.inf), run[:, :-1]], axis=1)
    on_sorted = (qs > prev) & np.isfinite(qs)
    out = np.zeros((K, S), bool)
    np.put_along_axis(out, order, on_sorted, axis=1)
    return out


def best_q_within(cost, quality, budget, strict=False):
    """K x B: max quality over settings with cost <= budget (cost < budget if strict), -inf if none.

    cost, quality: K x S (quality -inf marks excluded settings); budget: K x B. Costs must be positive.
    O(K S log S): sort each row by cost, prefix max, one flat searchsorted with per-row offsets on log cost.
    """
    cost = np.asarray(cost, float)
    quality = np.asarray(quality, float)
    budget = np.asarray(budget, float)
    K, S = cost.shape
    order = np.argsort(cost, axis=1, kind="stable")
    cs = np.take_along_axis(cost, order, axis=1)
    pm = np.maximum.accumulate(np.take_along_axis(quality, order, axis=1), axis=1)
    off = (np.arange(K) * 4000.0)[:, None]
    keys = (np.log(np.maximum(cs, 1e-300)) + off).ravel()
    bk = np.log(np.maximum(budget, 1e-300)) + off
    if budget.shape == cost.shape:
        # budgets here are a monotone map of each row's costs, so the cost order sorts them too; sorted queries make
        # searchsorted cache friendly (checked against the brute reference in the tests)
        bks = np.take_along_axis(bk, order, axis=1)
        if np.all(np.diff(bks, axis=1) >= 0):
            ps = np.searchsorted(keys, bks.ravel(), side="left" if strict else "right").reshape(budget.shape)
            pos = np.empty_like(ps)
            np.put_along_axis(pos, order, ps, axis=1)
        else:
            pos = np.searchsorted(keys, bk.ravel(), side="left" if strict else "right").reshape(budget.shape)
    else:
        pos = np.searchsorted(keys, bk.ravel(), side="left" if strict else "right").reshape(budget.shape)
    cnt = np.clip(pos - (np.arange(K) * S)[:, None], 0, S)
    pmx = np.concatenate([np.full((K, 1), -np.inf), pm], axis=1)
    return np.take_along_axis(pmx, cnt, axis=1)


def match_batch(R, cost, quality, kind="two_sided", eps_q=0.0, eps_c=0.0, floor=None, F=None):
    """K booleans: does R (length-S bool, or K x S) match the frontier of each world (rows of cost, quality).

    F: optional precomputed K x S frontier masks of those worlds.
    """
    cost = np.asarray(cost, float)
    quality = np.asarray(quality, float)
    K, S = cost.shape
    R = np.broadcast_to(np.asarray(R, bool), (K, S))
    if F is None:
        F = frontier_mask_batch(cost, quality, floor)
    if kind == "exact":
        return (R == F).all(axis=1)
    if kind not in MATCHES:
        raise ValueError(f"unknown match kind {kind}")
    qa = quality if floor is None else np.where(quality >= floor, quality, -np.inf)
    reach = best_q_within(cost, np.where(R, qa, -np.inf), (1.0 + eps_c) * cost)
    reach = np.maximum(reach, 0.0)                      # running nothing has quality 0
    cov = ((~F) | (reach >= quality - eps_q - 1e-12)).all(axis=1)
    if kind == "coverage":
        return cov
    # r is unsound iff some s eps-dominates it: (1 + eps_c) c_s <= c_r and q_s >= q_r + eps_q, strict in one
    b = cost / (1.0 + eps_c)
    beat = (best_q_within(cost, qa, b) > quality + eps_q + 1e-12) | \
           (best_q_within(cost, qa, b, strict=True) >= quality + eps_q - 1e-12)
    bad = R & beat
    if floor is not None:
        bad |= R & (quality < floor)
    return cov & ~bad.any(axis=1)


def match_batch_brute(R, cost, quality, kind="two_sided", eps_q=0.0, eps_c=0.0, floor=None):
    """Reference implementation by direct pairwise comparison (tests only)."""
    cost = np.asarray(cost, float)
    quality = np.asarray(quality, float)
    K, S = cost.shape
    R = np.broadcast_to(np.asarray(R, bool), (K, S))
    out = np.zeros(K, bool)
    for k in range(K):
        c, q, r = cost[k], quality[k], R[k]
        F = frontier_mask(c, q, floor)
        if kind == "exact":
            out[k] = (r == F).all()
            continue
        ok = True
        for f in np.flatnonzero(F):
            cand = [q[i] for i in np.flatnonzero(r) if c[i] <= (1 + eps_c) * c[f] and (floor is None or q[i] >= floor)]
            if max(cand + [0.0]) < q[f] - eps_q - 1e-12:
                ok = False
        if ok and kind == "two_sided":
            for i in np.flatnonzero(r):
                if floor is not None and q[i] < floor:
                    ok = False
                for j in range(S):
                    if floor is not None and q[j] < floor:
                        continue
                    if j == i:
                        continue
                    dom = (1 + eps_c) * c[j] <= c[i] and q[j] >= q[i] + eps_q - 1e-12
                    strict = (1 + eps_c) * c[j] < c[i] or q[j] > q[i] + eps_q + 1e-12
                    if dom and strict:
                        ok = False
        out[k] = ok
    return out


def match(R, cost, quality, kind="two_sided", eps_q=0.0, eps_c=0.0, floor=None):
    """Single-world version of match_batch."""
    return bool(match_batch(np.asarray(R, bool)[None, :], np.asarray(cost, float)[None, :],
                            np.asarray(quality, float)[None, :], kind, eps_q, eps_c, floor)[0])


def hypervolume_gap(R, cost, quality, F, cmin=None, cmax=None):
    """Area between the true frontier step curve and R's step curve, both read at true values, on log10 cost.

    The step curve of a set is Q(B) = best quality with cost <= B. Integrated over log10 B from the cheapest
    true frontier cost to the dearest one, so a unit is quality x decades. 0 means R reaches the truth everywhere.
    """
    cost = np.asarray(cost, float)
    quality = np.asarray(quality, float)
    F = np.asarray(F, bool)
    R = np.asarray(R, bool)
    lo = np.log10(cost[F].min() if cmin is None else cmin)
    hi = np.log10(cost[F].max() if cmax is None else cmax)
    if hi <= lo:
        return 0.0
    grid = np.linspace(lo, hi, 400)
    B = 10 ** grid
    def step(mask):
        ok = mask[None, :] & (cost[None, :] <= B[:, None])
        v = np.where(ok, quality[None, :], 0.0).max(axis=1)
        return v
    gap = np.clip(step(F) - step(R), 0.0, None)
    return float(np.trapezoid(gap, grid) if hasattr(np, "trapezoid") else np.trapz(gap, grid))


def precision_recall(R, F):
    R = np.asarray(R, bool)
    F = np.asarray(F, bool)
    tp = (R & F).sum()
    prec = tp / R.sum() if R.sum() else 0.0
    rec = tp / F.sum() if F.sum() else 1.0
    return float(prec), float(rec)
