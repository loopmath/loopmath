"""The value behind `adaptive-blocks`: what one run, or one block of runs, is worth for the picks over a grid of
budgets per run.

Definition. Settings are ordered by estimated cost, then index. For budget B_j the eligible set A_j is the settings
with a finite positive cost estimate at most B_j. The pick at budget j is the first setting in that order among
those in A_j with the largest mean pass rate mu_i. The value is V = sum_j w_j (mu of the pick - kappa sqrt(2/pi) sd
of the pick). A planned run (or block) moves the normal view of the pass rates: under it the means become
mu + shift Z, Z standard normal, and the variances var_after. The gain is G = E_Z V(mu + shift Z, var_after) -
V(mu, var). `adaptive-blocks` uses kappa = 0: no penalty term in the value that chooses runs.

Two exact engines compute G:

  gain_pairs     many single-run candidates at once. Budgets with the same eligible set form one group; a line that
                 could not win anywhere in |z| <= ZC is left out; a candidate keeps its value from one group to the
                 next when the larger eligible set adds no contender; the rest get the upper envelope from pairwise
                 bounds. gain_exact_slow is the plain reference by exhaustive breakpoints.
  block_gains    many block candidates at once, in closed form: eligible sets are nested in the budget, so one
                 running maximum and minimum over the cost order gives every line's interval for every budget.

Both integrate the winner's line over each interval with normal CDF and PDF differences. Ported from the design
study behind `adaptive-blocks` (the exact one-run gain and the closed-form block gain), arithmetic unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

ZC = 10.0          # a line that could only win beyond |z| = ZC is left out. What that drops is at most
#                    (largest difference of shifts) * 2 * pdf(ZC), about 1.5e-23 per unit of shift.
_SQ = np.sqrt(2.0 / np.pi)
_R2PI = 1.0 / np.sqrt(2.0 * np.pi)
_CHUNK = 2_000_000   # largest (groups * lines^2) handled in one piece
_ZBIG = 40.0         # the normal density and both tail masses are exactly 0.0 in double precision beyond this
_SQRT_2_OVER_PI = np.sqrt(2 / np.pi)
_INV_SQRT_2PI = 1 / np.sqrt(2 * np.pi)


# ---------------------------------------------------------------- picks

def pick(q: np.ndarray, c: np.ndarray, B: float, mask: np.ndarray | None = None) -> int:
    """The pick at budget B: among settings in `mask` (all if None) with finite q, finite c > 0 and c <= B, the
    largest q, then the smallest c, then the lowest index. -1 if there is no candidate."""
    ok = np.isfinite(q) & np.isfinite(c) & (c > 0) & (c <= B)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    idx = np.flatnonzero(ok)
    if not len(idx):
        return -1
    return int(idx[np.lexsort((idx, c[idx], -q[idx]))[0]])


def returned_default(q_hat: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
    """(S,) bool: the frontier of the estimates, the set the budget picks are taken from."""
    from .pareto import frontier_mask
    ok = np.isfinite(q_hat) & np.isfinite(c_hat) & (c_hat > 0)
    out = np.zeros(len(q_hat), bool)
    if ok.any():
        out[np.flatnonzero(ok)] = frontier_mask(np.asarray(c_hat, float)[ok], np.asarray(q_hat, float)[ok])
    return out


def _order(c_hat: np.ndarray) -> np.ndarray:
    """Settings by cost, then index: an argmax over this order breaks ties in mu as `pick` does."""
    c = np.where(np.isfinite(c_hat), c_hat, np.inf)
    return np.lexsort((np.arange(len(c)), c))


# ---------------------------------------------------------------- single runs: the exact envelope

def _pdf(z):
    with np.errstate(over="ignore"):
        return _R2PI * np.exp(-0.5 * np.square(z))


def _mass(lo, hi):
    """P(lo < Z < hi), accurate in both tails."""
    up = lo > 0
    return np.where(up, ndtr(-lo) - ndtr(-hi), ndtr(hi) - ndtr(lo))


def _layout(c_hat, B, w):
    """Lines in cost order, and the budgets grouped by how many of them are eligible.
    Returns (o, ks, W): o the setting index of each line; for group g the eligible set is lines [0, ks[g]) and W[g]
    the summed weight of its budgets. Groups with no eligible line or no weight are dropped."""
    c = np.asarray(c_hat, float)
    ok = np.isfinite(c) & (c > 0)
    o = np.lexsort((np.arange(len(c)), np.where(ok, c, np.inf)))
    o = o[ok[o]]
    k = np.searchsorted(c[o], np.asarray(B, float), side="right")
    ks, inv = np.unique(k, return_inverse=True)
    W = np.bincount(inv, weights=np.asarray(w, float), minlength=len(ks))
    keep = (ks > 0) & (W != 0)
    return o, ks[keep], W[keep]


def gain_exact_slow(mu, var, shift, var_after, c_hat, B, w, kappa: float = 0.0) -> float:
    """G for one candidate, by exhaustive breakpoints. The plain reference of `gain_pairs`."""
    o, ks, W = _layout(c_hat, B, w)
    a, b = np.asarray(mu, float)[o], np.asarray(shift, float)[o]
    pen0 = kappa * _SQ * np.sqrt(np.maximum(np.asarray(var, float)[o], 0.0))
    pen1 = kappa * _SQ * np.sqrt(np.maximum(np.asarray(var_after, float)[o], 0.0))
    return float(sum(wk * _one_group(a, b, pen0, pen1, int(k)) for k, wk in zip(ks, W)))


def _envelope(a, b, pen, a_r, pen_r):
    """For P groups of L lines each (rows are padded with a = -inf): sum over lines of
    (a_i - a_r - pen_i + pen_r) * P(line i is the pick) + b_i * E[Z; line i is the pick].
    Lines are in tie order (the earlier wins an exact tie). Returns (value (P,), total mass (P,))."""
    L = a.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        da = a[:, None, :] - a[:, :, None]            # [p, i, l] = a_l - a_i
        db = b[:, :, None] - b[:, None, :]            # [p, i, l] = b_i - b_l
        x = da / db
        lo = np.where(db > 0, x, -np.inf).max(2)      # i wins only where z >= every crossing with a flatter line
        hi = np.where(db < 0, x, np.inf).min(2)       # and z <= every crossing with a steeper line
        earlier = np.arange(L)[None, None, :] < np.arange(L)[None, :, None]
        dead = ((db == 0) & ((da > 0) | ((da == 0) & earlier))).any(2)
    live = ~dead & (hi > lo)
    lo, hi = np.where(live, lo, 0.0), np.where(live, hi, 0.0)
    mass = np.where(live, _mass(lo, hi), 0.0)
    ez = np.where(live, _pdf(lo) - _pdf(hi), 0.0)
    lead = np.where(live, a - a_r[:, None] - pen + pen_r[:, None], 0.0)
    return (lead * mass + np.where(live, b, 0.0) * ez).sum(1), mass.sum(1)


def gain_pairs(mu, var, shifts, var_after, c_hat, B, w, kappa: float = 0.0) -> np.ndarray:
    """(N,) the gain of each of N single-run candidates. shifts and var_after are (N, S): row n is the lookahead of
    candidate n. Equal to gain_exact_slow row by row."""
    shifts, var_after = np.atleast_2d(np.asarray(shifts, float)), np.atleast_2d(np.asarray(var_after, float))
    N = shifts.shape[0]
    out = np.zeros(N)
    o, ks, W = _layout(c_hat, B, w)
    if not len(ks) or not N:
        return out
    a = np.asarray(mu, float)[o]
    if not np.isfinite(a).all():
        raise ValueError("gain_pairs needs a finite mean for every setting with a cost estimate")
    b = shifts[:, o]
    pen0 = kappa * _SQ * np.sqrt(np.maximum(np.asarray(var, float)[o], 0.0))
    pen1 = kappa * _SQ * np.sqrt(np.maximum(var_after[:, o], 0.0))
    picks = np.array([int(np.argmax(a[:k])) for k in ks])
    for r in np.unique(picks):
        mine = np.flatnonzero(picks == r)                          # its groups, by growing eligible set
        kmax = int(ks[mine].max())
        # a line i can win somewhere in |z| <= ZC only if a_r - a_i <= ZC * |b_i - b_r|, and never at equal slope
        dbr = np.abs(b[:, :kmax] - b[:, [r]])
        cont = (dbr > 0) & ((a[r] - a[:kmax])[None, :] <= ZC * dbr * (1 + 1e-12))
        seen = np.cumsum(cont, axis=1)                             # contenders of each row among the first k lines
        val = pen0[r] - pen1[:, r]                                 # the pick stays: only its penalty changes
        before = np.zeros(N, dtype=seen.dtype)
        for g in mine:
            k = int(ks[g])
            now = seen[:, k - 1]
            new = np.flatnonzero(now != before)                    # rows whose contenders differ from the last group's
            before = now
            if len(new):
                val = val.copy()
                _fill(val, new, now[new], cont, r, k, a, b, pen0, pen1)
            out += W[g] * val
    return out


def _fill(val, rows, count, cont, r, k, a, b, pen0, pen1):
    """val[rows] for one eligible set (the first k lines, pick r): the rows are sorted into classes by their
    number of contenders, so that a row with one contender is not padded to the width of the widest."""
    width = 2
    while len(rows):
        take = count < width
        todo, rows, count = rows[take], rows[~take], count[~take]
        L, width = width, width * 2
        step = max(1, _CHUNK // (L * L))
        for i0 in range(0, len(todo), step):
            rr = todo[i0:i0 + step]
            mm = cont[rr, :k].copy()
            mm[:, r] = True
            idx = np.argsort(~mm, axis=1, kind="stable")[:, :L]    # the pick and the contenders, in tie order
            real = np.take_along_axis(mm, idx, 1)
            ag = np.where(real, a[idx], -np.inf)
            bg = np.where(real, b[rr[:, None], idx], 0.0)
            pg = np.where(real, pen1[rr[:, None], idx], 0.0)
            v, mass = _envelope(ag, bg, pg, np.full(len(rr), a[r]), np.full(len(rr), pen0[r]))
            for q in np.flatnonzero(np.abs(mass - 1.0) > 1e-9):   # never seen; kept as a guard
                v[q] = _one_group(a, b[rr[q]], pen0, pen1[rr[q]], k)
            val[rr] = v


def _one_group(a, b, pen0, pen1, k):
    """One candidate, one eligible set, by exhaustive breakpoints (the guard path of gain_pairs)."""
    ak, bk = a[:k], b[:k]
    r0 = int(np.argmax(ak))
    db = bk[:, None] - bk[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (ak[None, :] - ak[:, None]) / db
    cuts = np.unique(x[(db != 0) & np.isfinite(x)])
    edges = np.concatenate(([-np.inf], cuts, [np.inf]))
    ev = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        l, h = max(lo, -_ZBIG), min(hi, _ZBIG)
        if h <= l:
            continue                                      # a piece beyond |z| = 40 holds nothing a double can see
        mid = 0.5 * (l + h)
        r = 0                                             # the winner at mid, found pair by pair through
        for i in range(1, len(ak)):                       # differences, so that two lines that tie at the pick's
            if (ak[i] - ak[r]) + (bk[i] - bk[r]) * mid > 0.0:   # scale are still told apart; an exact tie stays
                r = i                                     # with the earlier line
        ev += (ak[r] - ak[r0] - pen1[r] + pen0[r0]) * float(_mass(lo, hi)) + bk[r] * float(_pdf(lo) - _pdf(hi))
    return ev


# ---------------------------------------------------------------- blocks: the closed form

def _phi(z):
    with np.errstate(over="ignore"):
        return _INV_SQRT_2PI * np.exp(-0.5 * z * z)       # 0 at +-inf


def _prefixes(c_hat, B, w):
    """Settings in cost order restricted to those that can ever be eligible; per distinct prefix length n > 0
    the summed weight of the budgets whose eligible set is the first n of them."""
    c = np.asarray(c_hat, float)
    o = _order(c)
    o = o[np.isfinite(c[o]) & (c[o] > 0)]
    n_j = np.searchsorted(c[o], np.asarray(B, float), side="right")       # c <= B, closed
    ns = np.unique(n_j[n_j > 0])
    wn = np.array([np.asarray(w, float)[n_j == n].sum() for n in ns])
    return o, ns, wn


def value(mu, var, c_hat, B, w, kappa: float = 0.0) -> float:
    """V(mu, var) as defined in the module docstring."""
    o, ns, wn = _prefixes(c_hat, B, w)
    if not len(ns):
        return 0.0
    m = np.asarray(mu, float)[o]
    pen = kappa * _SQRT_2_OVER_PI * np.sqrt(np.maximum(np.asarray(var, float)[o], 0.0))
    r = np.array([int(np.argmax(m[:n])) for n in ns])
    return float(wn @ (m[r] - pen[r]))


def block_gains(mu, var, shifts, var_afters, c_hat, B, w, kappa: float = 0.0) -> np.ndarray:
    """(C,) the gain of each of C candidate blocks. shifts and var_afters are (C, S): one lookahead per row.

    For one budget, V under Z is the upper envelope of the lines mu_i + shift_i z over the eligible settings. Each
    line is on top on one interval [lo_i, hi_i] (possibly empty): lo_i is the largest crossing with a flatter line,
    hi_i the smallest crossing with a steeper one; among lines of equal slope only the highest counts, ties going
    to the cheaper setting, then the lower index."""
    shifts, var_afters = np.atleast_2d(np.asarray(shifts, float)), np.atleast_2d(np.asarray(var_afters, float))
    C = shifts.shape[0]
    o, ns, wn = _prefixes(c_hat, B, w)
    if not len(ns):
        return np.zeros(C)
    N = len(o)
    a = np.asarray(mu, float)[o]                          # (N,)
    b = shifts[:, o]                                      # (C, N)
    pen = kappa * _SQRT_2_OVER_PI * np.sqrt(np.maximum(var_afters[:, o], 0.0))
    num = a[None, :] - a[:, None]                         # [i, k] = a_k - a_i
    db = b[:, :, None] - b[:, None, :]                    # [c, i, k] = b_i - b_k
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num[None] / db                            # where line i meets line k
    lo = np.maximum.accumulate(np.where(db > 0, ratio, -np.inf), axis=2)
    hi = np.minimum.accumulate(np.where(db < 0, ratio, np.inf), axis=2)
    before = np.arange(N)[None, :] < np.arange(N)[:, None]            # [i, k]: k comes before i in the order
    dom = np.logical_or.accumulate((db == 0) & ((num > 0) | ((num == 0) & before))[None], axis=2)
    cols = ns - 1                                         # prefix n reads column n - 1
    lo, hi, dom = lo[:, :, cols], hi[:, :, cols], dom[:, :, cols]     # (C, N, D)
    inside = np.arange(N)[:, None] < ns[None, :]                      # (N, D): setting i is in prefix n
    on = inside[None] & ~dom & (lo < hi)
    mass = np.where(on, ndtr(hi) - ndtr(lo), 0.0)
    tilt = np.where(on, _phi(lo) - _phi(hi), 0.0)
    ev = ((a[None, :, None] - pen[:, :, None]) * mass + b[:, :, None] * tilt).sum(1)   # (C, D)
    return ev @ wn - value(mu, var, c_hat, B, w, kappa)
