"""A fast engine for `eval fit`: Laplace marginal likelihood, empirical Bayes for the hyperparameters.

Not a command: `fit.py` calls `sample(design.data, ...)` when `eval fit --engine laplace` is given. The
result has the shape of `engine_pymc.sample`, so `fit.py`, `fitfile` and every report read it unchanged.
numpy and scipy only; it does not need the [bayes] extra.

The model is the one `engine_pymc` samples (PRD section 7, the module text of `engine_pymc.py`), with
the same priors. What differs is how it is fitted.

The method
----------
Every Gaussian unknown of the model is written in standard form, z ~ N(0, 1): arm means, biases, input
and run levels (through the Cholesky factor of their covariance), family effects, judge-by-input terms,
the halo factor of a target, the explicit judge-by-output parts, the shared part of a panel call, the
call noise of an `items` reading, and the position biases. The judge's view of a reading is then a
linear predictor `eta = X(theta) z + offset`, whose coefficients depend on the hyperparameters theta:
sds, correlations, halo shares, call noise, pair noise and tie widths. Writing the unknowns this way
keeps a variance near 0 harmless: its unknowns stay at their prior and the coefficient goes to 0.

1. Inner step. For fixed theta, the mode z-hat of `log p(y | z, theta) - |z|^2 / 2` by Newton steps
   with a step-halving line search (the log likelihood of every scale is concave in eta, so the
   objective is concave). The Hessian `H = I + X'WX` is sparse with an arrow shape: unknowns that belong
   to one input (its level, its runs' levels, judge-by-input terms, halo factors, explicit parts, panel
   calls, item noise) touch only that input and the few global unknowns (arm means, biases, family
   effects, positions). Each input block is factored on its own, then the global block through its
   Schur complement. The same factors give `log det H`, solves, draws, and the entries of `H^-1` that
   the gradient needs (the input blocks, the global block and the blocks between them).
2. Outer step. theta-hat maximises the Laplace approximation of `log p(y | theta)`,
   `log p(y | z-hat) - |z-hat|^2 / 2 - log det H / 2`, plus the log prior density of theta on its
   unconstrained scale (log of an sd, logit of a share, Fisher-z style partial correlations): penalised
   empirical Bayes. With the log scale's Jacobian in the penalty, an sd the data cannot tell from 0 is
   held off 0 by the penalty, never estimated as an exact 0. The gradient is exact: the envelope part at
   fixed z-hat, the trace of `H^-1 dH`, and an adjoint solve for the dependence of W on z-hat (third
   derivatives of the log likelihood). A projected BFGS with capped steps on bounded theta, then Newton
   steps with the Hessian of step 3 (`polish`).
   With cut readings (binary, levels, items, pairs) the inner step is expectation propagation (EP)
   by default, and the outer objective is the EP value of `log p(y | theta)`, not the Laplace value:
   plain Laplace is off by up to a log unit on one sharp judge-by-output cell. The cut readings that
   share a linear predictor (the calls of one judge on one output and component, the passed and failed
   items of one call, pairs shown the same way) form one site; its tilted moments are one-dimensional
   integrals, in closed form for a single reading and otherwise by `cell_integral` (graded
   Gauss-Legendre panels between the cuts and the mode, with a log-concave tail bound). Sites are
   updated in parallel sweeps with Anderson acceleration. The gradient treats the sites as fixed, which
   is exact at the EP fixed point. Where one site holds every reading of its unknowns, the EP value is
   the exact marginal likelihood; where sites share unknowns it is an approximation (measured against
   an exact oracle in the tests). `sample(inner="laplace")` keeps the Laplace step.
3. Draws. theta is drawn from N(theta-hat, Sigma), Sigma the inverse of minus the Hessian of step 2 at
   theta-hat on the unconstrained scale (forward differences of the exact gradient). For each of
   `theta_points` theta draws the inner step is solved again and z is drawn from N(z-hat, H^-1); the
   parameters of the fit file follow from (theta, z). The draws are spread over the theta points at
   random, about equally, so the hyperparameters of a fit take `theta_points` distinct values (250 by
   default) and every chain holds a random share of them.

What is approximate
-------------------
The draws are approximate posterior draws: Gaussian in z given theta (exact for `linear` and
`fraction` readings, the EP or Laplace Gaussian for cut readings and pairs) and Gaussian in theta on the
unconstrained scale. The fit file names the engine and says its draws are approximate (`fit.py`).

Likelihood kernels
------------------
Gaussian readings use `gaussian_kernel`. Every cut reading is an interval of a latent Gaussian: a
`binary` reading is the interval above or below 0, a `levels` reading the interval between two cut
points, an `items` count k of n two weighted intervals, a pair verdict the interval above the tie width,
below minus it, or between. All of them go through `interval_kernel`, written for the standardised
predictor (eta, bounds and coefficients divided by the noise sd), with derivatives to third order.

Limits of this spike
--------------------
`--structure factor:K` is not built here (`NotImplementedError`). An input with very many runs gives a
large dense block. The share at the boundary is reported by `boundary` (see `sample`).
"""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from scipy import linalg as sla
from scipy import optimize, special

from .. import __version__

NAME = "laplace"
DRAWS_NOTE = "approximate draws"
PERSON_SIGMA = 0.04   # as engine_pymc: the half-normal scale of a person's call noise on linear and fraction
LKJ_ETA = 2.0         # as engine_pymc
LINEAR, BINARY, LEVELS, ITEMS = 0, 1, 2, 3   # as model.KIND_CODE
OPEN = 1.0e6          # as model.OPEN
GAUSS, INTERVAL = 0, 1
LOG_2PI = float(np.log(2.0 * np.pi))
SQRT2 = float(np.sqrt(2.0))
CHUNK = 4096          # rows per pass when gathering entries of H^-1: bounds the memory
EP_TOL = 1e-10        # site change at which EP stops, in the fit (the gradient needs it tight)
VALUE_TOL = 1e-6      # where only the value is used (the boundary scan): its error is about 1e-12
DRAW_TOL = 1e-3       # at the theta points: the draws' mean and covariance move by about 1e-3 of an sd
ANDERSON = 4          # memory of the Anderson acceleration of EP sweeps
DECREMENT_MAX = 1e-4  # the optimiser has converged when a Newton step would gain less log likelihood than this
OUTPUTS = ("mu", "sd_input", "sd_run", "corr_input", "corr_run", "q_run", "bias", "tau", "sigma", "halo_target",
           "halo_call", "family", "sd_judge_input", "sigma_pair", "position", "tie_width")


def available() -> bool:
    """Always: numpy and scipy are core dependencies."""
    return True


def version() -> str:
    return f"loopmath {__version__}, {DRAWS_NOTE}"


# ---------------------------------------------------------------------------------------------------
# Kernels

def gaussian_kernel(eta: np.ndarray, y: np.ndarray, var: np.ndarray) -> dict:
    """`log N(y; eta, var)` and its derivatives in eta (to third order) and in var."""
    r = y - eta
    inv = 1.0 / var
    return {"logp": -0.5 * (LOG_2PI + np.log(var)) - 0.5 * r * r * inv, "d1": r * inv, "d2": -inv,
            "d3": np.zeros_like(eta), "dvar": -0.5 * inv + 0.5 * r * r * inv * inv, "d1_dvar": -r * inv * inv,
            "d2_dvar": inv * inv}


def _log_mills_inv(x: np.ndarray) -> np.ndarray:
    """`log(phi(x) / Phi(-x))` for x >= 0 (the log inverse Mills ratio), from `erfcx` without cancellation."""
    return -np.log(np.sqrt(np.pi / 2.0) * special.erfcx(x / SQRT2))


def _lower_tail(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a < b <= 0: `(log P, R_a, R_b)` with P = Phi(b) - Phi(a), R = phi / P at each end.

    Written with `lam = phi / Phi` at each end (from `erfcx`) and `r = Phi(a) / Phi(b)` from the ratio
    of the two, so nothing is the exponential of a difference of two large numbers."""
    log_lb, log_la = _log_mills_inv(-b), _log_mills_inv(-a)
    with np.errstate(over="ignore"):
        log_r = log_lb - log_la - 0.5 * (a - b) * (a + b)
    log_1mr = np.log(-np.expm1(log_r))
    logp = special.log_ndtr(b) + log_1mr
    rb = np.exp(log_lb - log_1mr)
    ra = np.exp(log_la + log_r - log_1mr)
    return logp, ra, rb


def _ratios(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(log(Phi(b) - Phi(a)), phi(a) / P, phi(b) / P)` for a < b, its digits kept in both tails."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    logp, ra, rb = np.empty(a.shape), np.empty(a.shape), np.empty(a.shape)
    upper, lower = a >= 0.0, b <= 0.0
    mid = ~(upper | lower)
    logp[lower], ra[lower], rb[lower] = _lower_tail(a[lower], b[lower])
    logp[upper], rb[upper], ra[upper] = _lower_tail(-b[upper], -a[upper])   # by symmetry: the ends swap
    am, bm = a[mid], b[mid]
    logp[mid] = np.log(0.5 * (special.erf(bm / SQRT2) - special.erf(am / SQRT2)))
    ra[mid] = np.exp(-0.5 * am * am - 0.5 * LOG_2PI - logp[mid])
    rb[mid] = np.exp(-0.5 * bm * bm - 0.5 * LOG_2PI - logp[mid])
    return logp, ra, rb


def log_interval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`log(Phi(b) - Phi(a))` for a < b, with its digits kept in both tails and for a narrow interval."""
    return _ratios(a, b)[0]


FAR = 3.0        # from this distance on, a one-sided tail uses the continued fraction
CF_DEPTH = 300   # terms of the continued fraction (converged to rounding for x >= FAR)


def _tail_terms(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For x >= FAR: `t_1, t_2, t_3` of the continued fraction of the inverse Mills ratio
    `lam(x) = phi(x) / Phi(-x) = x + 1 / t_1`, with `t_k = x + (k + 1) / t_(k + 1)`. Then
    `lam - x = 1 / t_1` and `1 - x (lam - x) = 2 / (t_1 t_2)` come without cancellation."""
    t = x.copy()
    keep: dict[int, np.ndarray] = {}
    for k in range(CF_DEPTH, 0, -1):
        t = x + (k + 1) / t
        if k <= 3:
            keep[k] = t
    return keep[1], keep[2], keep[3]


def interval_kernel(eta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """`log(Phi(hi - eta) - Phi(lo - eta))`: a unit-noise latent `eta + e` falls in `(lo, hi)`.

    Derivatives in eta to third order, and of the value, d1 and d2 in each bound. With `a = lo - eta`
    and `b = hi - eta`, `R_a = phi(a) / P` and `R_b = phi(b) / P` carry everything. d1, d2 and d3 are
    the mean, the variance minus 1 and the third central moment of a standard normal cut to (a, b), so
    d2 lies in [-1, 0] (it is clipped there against rounding). Far out in a one-sided tail the general
    formulas cancel; there they come from the continued fraction of `_tail_terms`."""
    a, b = lo - eta, hi - eta
    logp, ra, rb = _ratios(a, b)
    ara = np.where(ra > 0.0, a * ra, 0.0)    # a R_a is 0 at an open bound, not inf * 0
    brb = np.where(rb > 0.0, b * rb, 0.0)
    a2ra = np.where(ra > 0.0, a * ara, 0.0)
    b2rb = np.where(rb > 0.0, b * brb, 0.0)
    l_a, l_b = -ra, rb
    l_aa, l_bb, l_ab = ara - ra * ra, -brb - rb * rb, ra * rb
    l_aaa = ra - a2ra + 3.0 * ara * ra - 2.0 * ra ** 3
    l_bbb = -rb + b2rb + 3.0 * brb * rb + 2.0 * rb ** 3
    l_aab = -ara * rb + 2.0 * ra * ra * rb
    l_abb = -brb * ra - 2.0 * ra * rb * rb
    out = {"logp": logp, "d1": ra - rb, "d2": l_aa + 2.0 * l_ab + l_bb,
           "d3": -(l_aaa + 3.0 * l_aab + 3.0 * l_abb + l_bbb),
           "dlo": l_a, "dhi": l_b, "d1_dlo": -(l_aa + l_ab), "d1_dhi": -(l_ab + l_bb),
           "d2_dlo": l_aaa + 2.0 * l_aab + l_abb, "d2_dhi": l_aab + 2.0 * l_abb + l_bbb}
    # one-sided far tails: the far end carries nothing next to the near one
    below = (b <= -FAR) & (ra <= 1e-17 * rb)     # latent pushed far above hi: the finite end is hi
    above = (a >= FAR) & (rb <= 1e-17 * ra)      # latent pushed far below lo: the finite end is lo
    for side, x, sign in ((below, -b, -1.0), (above, a, 1.0)):
        if not side.any():
            continue
        xs = x[side]
        t1, t2, t3 = _tail_terms(xs)
        lam = xs + 1.0 / t1
        d1 = sign * lam
        d2 = -lam / t1
        d3 = -sign * 2.0 * lam * (2.0 / t2 - 3.0 / t3) / (t1 * t1 * t2)
        out["d1"][side], out["d2"][side], out["d3"][side] = d1, d2, d3
        end, other = ("dhi", "dlo") if sign < 0 else ("dlo", "dhi")
        out[end][side], out["d1_" + end][side], out["d2_" + end][side] = -d1, -d2, -d3
        out[other][side], out["d1_" + other][side], out["d2_" + other][side] = 0.0, 0.0, 0.0
    out["d2"] = np.clip(out["d2"], -1.0, 0.0)
    return out


# ---------------------------------------------------------------------------------------------------
# Partial correlations to the Cholesky factor of a correlation matrix

# ---- one-dimensional integrals of a cell ------------------------------------------------------------

GL_X, GL_W = np.polynomial.legendre.leggauss(6)
PANELS = 4         # Gauss-Legendre panels per half gap (measured: log Z within 1.5e-7 of quad on hard cells)
WINDOW = 36.0      # the integral runs where log h is within this of its maximum
STEP_RES = 0.4     # panel width next to a cut, in units of the call noise


def cell_integral(m: np.ndarray, s2: np.ndarray, lo: np.ndarray, hi: np.ndarray, w: np.ndarray, *,
                  panels: int = PANELS, bounds: bool = False, guess: Mapping[str, np.ndarray] | None = None) -> dict:
    """Per cell g: `log Z = log integral N(u; m, s2) prod_r (Phi(hi_r - u) - Phi(lo_r - u))^w_r du`, with the
    mean and variance of u under the normalised integrand.

    `lo`, `hi`, `w` are (cells, rows), padded with w = 0. The integrand h is log-concave. The integral
    runs over [A, B], where log h is within WINDOW of its maximum at the mode u*. Breakpoints are A, the
    finite bounds inside (A, B), u* and B. Every gap between breakpoints is split in two halves, and
    each half carries `panels` Gauss-Legendre panels, graded geometrically from its outer end (width
    about STEP_RES next to a cut, half the Laplace scale next to u*). A cut is a unit-width step, so
    near it the panels are fine; far from cuts the integrand is a smooth Gaussian piece and the panels
    grow. `tail` bounds the mass outside [A, B] relative to Z by log-concavity:
    integral beyond B <= h(B) / |(log h)'(B)|. With `bounds`, also d log Z / d lo_r and d hi_r.
    `guess` (`mode`, `A`, `B` of an earlier call on nearby arguments) only starts the searches: the
    mode and the window ends are found to the same tolerances either way."""
    m, s2 = np.asarray(m, float), np.asarray(s2, float)
    G, R = lo.shape
    live = w > 0
    lo_m, hi_m = np.where(live, lo, -OPEN), np.where(live, hi, OPEN)

    def logh(u):   # u: (G, K)
        a, b = lo_m[:, None, :] - u[:, :, None], hi_m[:, None, :] - u[:, :, None]
        lp = np.where(live[:, None, :], log_interval(a, b), 0.0)
        return (w[:, None, :] * lp).sum(axis=2) - 0.5 * (u - m[:, None]) ** 2 / s2[:, None]

    def derivs(u):  # u: (G,)
        k = interval_kernel(np.repeat(u, R), lo_m.ravel(), hi_m.ravel())
        d1 = (w * k["d1"].reshape(G, R)).sum(axis=1) - (u - m) / s2
        d2 = (w * k["d2"].reshape(G, R)).sum(axis=1) - 1.0 / s2
        f = (w * np.where(live, k["logp"].reshape(G, R), 0.0)).sum(axis=1) - 0.5 * (u - m) ** 2 / s2
        return f, d1, d2

    # the mode: Newton with a bracket (d1 falls in u)
    s = np.sqrt(s2)
    u = m.copy() if guess is None else np.asarray(guess["mode"], float).copy()
    left, right = np.full(G, -np.inf), np.full(G, np.inf)
    for _ in range(80):
        f, d1, d2 = derivs(u)
        left = np.where(d1 > 0, np.maximum(left, u), left)
        right = np.where(d1 <= 0, np.minimum(right, u), right)
        if np.all(np.abs(d1) * np.sqrt(-1.0 / d2) < 1e-9):
            break
        step = np.clip(-d1 / d2, -4.0 * np.maximum(1.0, s), 4.0 * np.maximum(1.0, s))
        new = u + step
        inside = (new > left) & (new < right)
        both = np.isfinite(left) & np.isfinite(right)
        u = np.where(inside | ~both, new, 0.5 * (left + right))
    f0, _, d2 = derivs(u)
    mode, scale = u, np.sqrt(-1.0 / d2)
    # the window ends: Newton on the convex F(v) = f0 - f(v) - WINDOW, from the quadratic guess
    reach = np.sqrt(2.0 * (WINDOW + 50.0)) * s + np.abs(mode - m) + 10.0
    ends = []
    for side in (-1.0, 1.0):
        v = mode + side * np.sqrt(2.0 * WINDOW) * scale
        if guess is not None:   # an earlier end, if it is still on its side of the mode
            old_end = np.asarray(guess["A" if side < 0 else "B"], float)
            v = np.where(side * (old_end - mode) > 0, old_end, v)
        for _ in range(60):
            f, d1, _ = derivs(v)
            F = f0 - f - WINDOW
            if np.all(np.abs(F) < 0.5):
                break
            slope = -d1
            step = np.where(np.abs(slope) > 1e-300, -F / np.where(slope == 0, 1.0, slope), side * reach)
            step = side * np.clip(side * step, -0.9 * np.abs(v - mode), reach)
            v = v + step
        ends.append(v)
    A, B = ends
    # breakpoints: A, cuts inside, the mode, B
    cuts = np.concatenate([np.where(live & (np.abs(lo) < OPEN), lo, A[:, None]),
                           np.where(live & (np.abs(hi) < OPEN), hi, A[:, None])], axis=1)
    cuts = np.sort(np.clip(cuts, A[:, None], B[:, None]), axis=1)
    cuts[:, 1:] = np.where(cuts[:, 1:] - cuts[:, :-1] > 1e-12, cuts[:, 1:], A[:, None])   # one copy of each cut
    cuts = cuts[:, np.any(cuts - A[:, None] > 1e-12, axis=0)]
    pts = np.sort(np.concatenate([A[:, None], cuts, mode[:, None], B[:, None]], axis=1), axis=1)
    is_cut = np.zeros(pts.shape, bool)
    for col in range(cuts.shape[1]):
        is_cut |= np.isclose(pts, cuts[:, col:col + 1], rtol=0, atol=1e-12) & \
            (np.abs(cuts[:, col:col + 1] - A[:, None]) > 1e-12)
    res = np.where(is_cut, STEP_RES, np.where(np.isclose(pts, mode[:, None], rtol=0, atol=1e-12),
                                              0.5 * scale[:, None], np.inf))
    a, b = pts[:, :-1], pts[:, 1:]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    t = np.arange(panels + 1) / panels
    edges = []
    for end, sign, r in ((a, 1.0, res[:, :-1]), (b, -1.0, res[:, 1:])):
        beta = np.log(np.maximum(half / np.where(np.isfinite(r), r, np.maximum(half, 1e-300)), 1.0))
        safe = np.maximum(beta, 1e-12)
        frac = np.where(beta[..., None] > 1e-9, np.expm1(safe[..., None] * t) / np.expm1(safe)[..., None], t)
        x = end[..., None] + sign * half[..., None] * frac      # from the outer end to the midpoint
        lo_e, hi_e = (x[..., :-1], x[..., 1:]) if sign > 0 else (x[..., 1:], x[..., :-1])
        edges.append((lo_e, hi_e))
    lo_e = np.concatenate([edges[0][0], edges[1][0]], axis=-1).reshape(G, -1)
    hi_e = np.concatenate([edges[0][1], edges[1][1]], axis=-1).reshape(G, -1)
    width = hi_e - lo_e
    nodes = (0.5 * (lo_e + hi_e))[..., None] + 0.5 * width[..., None] * GL_X
    weights = 0.5 * width[..., None] * GL_W
    nodes, weights = nodes.reshape(G, -1), weights.reshape(G, -1)
    lh = logh(nodes)
    top = lh.max(axis=1, keepdims=True)
    q = weights * np.exp(lh - top)
    Z = q.sum(axis=1)
    pw = q / Z[:, None]
    mean = (pw * nodes).sum(axis=1)
    var = (pw * (nodes - mean[:, None]) ** 2).sum(axis=1)
    logZ = np.log(Z) + top[:, 0] - 0.5 * np.log(2.0 * np.pi * s2)
    # tails beyond [A, B], relative to Z, by log-concavity
    tail = np.zeros(G)
    for v in (A, B):
        f, d1, _ = derivs(v)
        tail += np.exp(f - top[:, 0]) / np.maximum(np.abs(d1), 1e-300) / Z
    out = {"logZ": logZ, "mean": mean, "var": var, "tail": tail, "mode": mode, "scale": scale, "A": A, "B": B}
    if bounds:
        kk = interval_kernel(np.repeat(nodes.ravel(), R), np.tile(lo_m, (1, nodes.shape[1])).ravel(),
                             np.tile(hi_m, (1, nodes.shape[1])).ravel())
        shape = (G, nodes.shape[1], R)
        out["dlo"] = (pw[:, :, None] * kk["dlo"].reshape(shape)).sum(axis=1) * w
        out["dhi"] = (pw[:, :, None] * kk["dhi"].reshape(shape)).sum(axis=1) * w
    return out


def cpc_count(D: int) -> int:
    return D * (D - 1) // 2


def cpc_levels(D: int) -> np.ndarray:
    """The LKJ(eta) prior on canonical partial correlations: z at level j is Beta(b, b) on (-1, 1) with
    b = eta + (D - 2 - j) / 2 (Lewandowski, Kurowicka and Joe 2009). One entry per partial correlation."""
    return np.array([LKJ_ETA + (D - 2 - j) / 2.0 for i in range(1, D) for j in range(i)])


def cpc_chol(theta: np.ndarray, D: int) -> np.ndarray:
    """The Cholesky factor L of a correlation matrix from unconstrained theta (z = tanh theta), as Stan's
    transform: row i holds z_i0, then z_ij scaled by what row i has left. Works on complex theta too, so
    `cpc_jacobian` can take complex steps."""
    z = np.tanh(theta)
    L = np.zeros((D, D), dtype=z.dtype)
    L[0, 0] = 1.0
    k = 0
    for i in range(1, D):
        left = 1.0 + 0.0 * z[0]
        for j in range(i):
            L[i, j] = z[k] * np.sqrt(left)
            left = left - L[i, j] ** 2
            k += 1
        L[i, i] = np.sqrt(left)
    return L


def cpc_jacobian(theta: np.ndarray, D: int) -> np.ndarray:
    """dL / dtheta as an array [P, D, D], by complex steps (exact to rounding)."""
    out = np.zeros((len(theta), D, D))
    for k in range(len(theta)):
        step = theta.astype(complex)
        step[k] += 1e-30j
        out[k] = cpc_chol(step, D).imag / 1e-30
    return out


# ---------------------------------------------------------------------------------------------------
# The block factorisation of H = I + X'WX

class Factor:
    """H in three blocks: `gg` (G x G, global), `bg` (I x B x G), `bb` (I x B x B, per input).

    Per input `L_i L_i' = H_bb,i`; then the Schur complement of the global block. Gives log det H,
    solves, the entries of H^-1 on the three blocks, and draws from N(0, H^-1)."""

    def __init__(self, gg: np.ndarray, bg: np.ndarray, bb: np.ndarray):
        self.G, (self.I, self.B) = gg.shape[0], bb.shape[:2]
        self.bg = bg
        lbb = np.linalg.cholesky(bb)
        self.linv = np.linalg.inv(lbb)
        self.bb_inv = np.einsum("ikb,ikc->ibc", self.linv, self.linv)
        self.K = self.bb_inv @ bg                                   # I x B x G
        schur = gg - np.einsum("ibg,ibh->gh", bg, self.K)
        self.lgg = np.linalg.cholesky(0.5 * (schur + schur.T))
        self.logdet = 2.0 * float(np.log(np.diagonal(lbb, axis1=1, axis2=2)).sum()
                                  + np.log(np.diagonal(self.lgg)).sum())

    def solve(self, rg: np.ndarray, rb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = np.einsum("ibc,ic->ib", self.bb_inv, rb)
        xg = sla.cho_solve((self.lgg, True), rg - np.einsum("ibg,ib->g", self.bg, t))
        return xg, t - np.einsum("ibg,g->ib", self.K, xg)

    def selected(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The blocks of H^-1: global (G x G), between (I x B x G), per input (I x B x B)."""
        sgg = sla.cho_solve((self.lgg, True), np.eye(self.G))
        sbg = -self.K @ sgg
        sbb = self.bb_inv + np.einsum("ibg,icg->ibc", self.K @ sgg, self.K)
        return sgg, sbg, sbb

    def draw(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        """n draws from N(0, H^-1): `(global [n, G], per input [n, I, B])`."""
        eg = rng.standard_normal((n, self.G))
        eb = rng.standard_normal((n, self.I, self.B))
        xg = sla.solve_triangular(self.lgg, eg.T, lower=True, trans="T").T
        xb = np.einsum("ikb,nik->nib", self.linv, eb) - np.einsum("ibg,ng->nib", self.K, xg)
        return xg, xb


# ---------------------------------------------------------------------------------------------------
# The problem: unknowns, rows, hyperparameters

class _Latents:
    """Where each unknown lives: global (index in [0, G)) or local (input i, position p in its block)."""

    def __init__(self, I: int):
        self.G = 0
        self.fill = np.zeros(I, dtype=np.int64)
        self.families: dict[str, dict] = {}

    def add_global(self, name: str, n: int) -> None:
        self.families[name] = {"local": np.zeros(n, dtype=bool), "g": np.arange(self.G, self.G + n),
                               "i": np.zeros(n, dtype=np.int64), "p": np.zeros(n, dtype=np.int64)}
        self.G += n

    def add(self, name: str, inputs: np.ndarray) -> None:
        """Unknowns of one family; `inputs[k]` is the input that owns unknown k, or -1 for global."""
        inputs = np.asarray(inputs, dtype=np.int64)
        n = len(inputs)
        fam = {"local": inputs >= 0, "g": np.zeros(n, dtype=np.int64), "i": np.maximum(inputs, 0),
               "p": np.zeros(n, dtype=np.int64)}
        for k in range(n):
            if inputs[k] >= 0:
                fam["p"][k] = self.fill[inputs[k]]
                self.fill[inputs[k]] += 1
            else:
                fam["g"][k] = self.G
                self.G += 1
        self.families[name] = fam

    @property
    def B(self) -> int:
        return int(max(1, self.fill.max(initial=0)))


def _owner(n: int, pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """The input of each of n unknowns from (unknown index, input) pairs; -1 where rows of two inputs
    reach it (it becomes global) or where nothing does (it stays local to input 0 as a prior-only unknown)."""
    owner = np.full(n, -2, dtype=np.int64)
    for idx, inp in pairs:
        keep = idx >= 0
        for k, i in zip(idx[keep], inp[keep]):
            owner[k] = i if owner[k] in (-2, i) else -1
    owner[owner == -2] = 0
    return owner


class Problem:
    """The arrays of `model.Design.data` laid out for the Laplace engine."""

    def __init__(self, data: Mapping[str, Any]):
        d = data
        A, D, J, N, I, R = (int(d[name]) for name in ("A", "D", "J", "F", "I", "R"))
        self.A, self.D, self.J, self.N, self.I, self.R = A, D, J, N, I, R
        self.data = d
        if len(d["k_r"]):
            raise ValueError("engine_laplace: grouped cut readings are not supported; build the design with "
                             "loopmath.eval.model.build")
        form = d["structure"]
        if form == "full" and D < 2:
            form = "diag"
        if form == "factor":
            raise NotImplementedError("engine_laplace: --structure factor:K is not built in this engine yet; "
                                      "use --structure full or diag")
        self.form = form
        self.scale = np.asarray(d["scale"], dtype=float)
        self.mu_loc = np.asarray(d["mu_loc"], dtype=float)
        self.pairs_only = np.asarray(d["pairs_only"], dtype=bool)
        self.run_arm = np.asarray(d["run_arm"], dtype=np.int64)
        self.run_input = np.asarray(d["run_input"], dtype=np.int64)
        self.bias_w = np.asarray(d["bias_w"], dtype=float)
        self.sigma_w = np.asarray(d["sigma_w"], dtype=float)
        self.pair_w = np.asarray(d["pair_w"], dtype=float)
        self.halo_on = np.asarray(d["halo_on"], dtype=bool)
        self.family_on = np.asarray(d["family_on"], dtype=bool)
        self.pair_scale = np.asarray(d["pair_scale"], dtype=float)
        self.sigma_fixed = np.broadcast_to(np.asarray(d["sigma_fixed"], dtype=bool), (J, D))
        self.levels_dim = np.broadcast_to(np.asarray(d["levels_dim"], dtype=bool), (J, D))
        self.tight = np.asarray(d["person_tight"], dtype=bool)
        ri = self.run_input
        s_r, g_r = np.asarray(d["s_r"], dtype=np.int64), np.asarray(d["g_r"], dtype=np.int64)
        p_ra, p_rb = np.asarray(d["p_ra"], dtype=np.int64), np.asarray(d["p_rb"], dtype=np.int64)
        cell_suff = np.asarray(d["cell_suff"], dtype=np.int64)
        self.n_jt, self.n_ji, self.n_cell, self.n_pc, self.n_e = (int(d[k]) for k in ("n_jt", "n_ji", "n_cell",
                                                                                      "n_pc", "n_e"))

        # ---- unknowns ---------------------------------------------------------------------------
        lat = _Latents(I)
        lat.add_global("mu", A * D)
        lat.add_global("bias", J * D)
        fam_keys = [(j, dd) for j in range(J) for dd in range(D) if self.family_on[j, dd]]
        self.fam_keys = fam_keys
        lat.add_global("family", len(fam_keys) * N)
        pair_judges = np.zeros(J, dtype=bool)
        pair_judges[np.asarray(d["p_j"], dtype=np.int64)] = True
        self.pair_judges = pair_judges
        lat.add_global("position", J)
        width = D
        lat.add("v", np.repeat(np.arange(I), width))
        lat.add("u", np.repeat(ri, width))
        lat.add("ji", _owner(self.n_ji, [(np.asarray(d["s_ji"]), ri[s_r]), (np.asarray(d["g_ji"]), ri[g_r])]))
        lat.add("jt", _owner(self.n_jt, [(np.asarray(d["s_jt"]), ri[s_r]), (np.asarray(d["g_jt"]), ri[g_r]),
                                         (np.asarray(d["p_jta"]), ri[p_ra]), (np.asarray(d["p_jtb"]), ri[p_rb])]))
        linked = cell_suff >= 0
        lat.add("cell", _owner(self.n_cell, [(np.asarray(d["g_cell"]), ri[g_r]), (np.asarray(d["p_ca"]), ri[p_ra]),
                                             (np.asarray(d["p_cb"]), ri[p_rb]),
                                             (np.flatnonzero(linked), ri[s_r[cell_suff[linked]]] if linked.any()
                                              else np.zeros(0, dtype=np.int64))]))
        lat.add("pc", _owner(self.n_pc, [(np.asarray(d["g_pc"]), ri[g_r])]))
        lat.add("e", _owner(self.n_e, [(np.asarray(d["g_e"]), ri[g_r])]))
        self.lat = lat
        self.G, self.B = lat.G, lat.B

        # ---- slots: coefficients that depend on theta ---------------------------------------------
        off = {}
        at = 0
        for name, size in (("Lin", D * D), ("Lrun", D * D), ("famsd", D), ("sdji", J * D), ("ajt", J * D),
                           ("acell", J * D), ("apc", J * D)):
            off[name] = at
            at += size
        self.slot_off, self.n_slots = off, at

        # ---- rows ---------------------------------------------------------------------------------
        rows: dict[str, list] = {k: [] for k in ("kind", "y", "w", "o", "lo", "hi", "lot", "hit", "j", "d", "vt", "n",
                                                 "x", "sh", "mode")}
        ents: dict[str, list] = {k: [] for k in ("row", "fam", "idx", "base", "slot", "e")}
        self._rows, self._ents = rows, ents
        n_rows = 0

        def add_rows(count: int, **cols) -> np.ndarray:
            nonlocal n_rows
            ids = np.arange(n_rows, n_rows + count)
            defaults = {"kind": GAUSS, "y": 0.0, "w": 1.0, "o": 0.0, "lo": -OPEN, "hi": OPEN, "lot": 0.0, "hit": 0.0,
                        "j": 0, "d": 0, "vt": 0, "n": 1.0, "x": 0.0, "sh": 0.0, "mode": 0}
            for key, default in defaults.items():
                value = cols.get(key, default)
                rows[key].append(np.broadcast_to(np.asarray(value), (count,)).copy())
            n_rows += count
            return ids

        def ent(row_ids, fam: str, idx, base, slot, e=False) -> None:
            row_ids = np.asarray(row_ids, dtype=np.int64)
            idx = np.broadcast_to(np.asarray(idx, dtype=np.int64), row_ids.shape)
            keep = idx >= 0
            ents["row"].append(row_ids[keep])
            ents["fam"].append(np.full(int(keep.sum()), list(lat.families).index(fam), dtype=np.int64))
            ents["idx"].append(idx[keep])
            ents["base"].append(np.broadcast_to(np.asarray(base, dtype=float), row_ids.shape)[keep])
            ents["slot"].append(np.broadcast_to(np.asarray(slot, dtype=np.int64), row_ids.shape)[keep])
            ents["e"].append(np.full(int(keep.sum()), bool(e)))

        def quality(row_ids, r, dd, sign=1.0, with_v=True, with_mu=True):
            """q[r, d] = mu[arm, d] + v[input, d] + u[r, d]."""
            if with_mu:
                arm = self.run_arm[r]
                po = self.pairs_only[dd]
                for a in range(A):
                    base = np.where(po, 2.0 * self.scale[dd] * ((a == arm) - 1.0 / A),
                                    np.where(a == arm, 2.0 * self.scale[dd], 0.0))
                    ent(row_ids, "mu", np.where(base != 0.0, a * D + dd, -1), sign * base, -1)
            for k in range(D):
                slot_in = off["Lin"] + dd * D + k
                slot_run = off["Lrun"] + dd * D + k
                use = k <= dd if self.form == "full" else k == dd
                if with_v:
                    ent(row_ids, "v", np.where(use, self.run_input[r] * D + k, -1), sign, slot_in)
                ent(row_ids, "u", np.where(use, r * D + k, -1), sign, slot_run)

        def bias(row_ids, j, dd):
            """bias[j, d] = s (z[j, d] - sum over k of w[k, d] z[k, d])."""
            for jj in range(J):
                base = self.scale[dd] * ((jj == j) - self.bias_w[jj, dd])
                ent(row_ids, "bias", np.where(base != 0.0, jj * D + dd, -1), base, -1)

        fam_pos = {key: n for n, key in enumerate(fam_keys)}

        def family(row_ids, j, f, dd, sign=1.0):
            pos = np.array([fam_pos.get((int(a), int(b)), -1) for a, b in zip(np.atleast_1d(j), np.atleast_1d(dd))],
                           dtype=np.int64)
            pos = np.broadcast_to(pos, np.shape(row_ids)) if pos.size == 1 else pos
            for ff in range(N):
                base = sign * ((ff == f) - 1.0 / N)
                ent(row_ids, "family", np.where(pos >= 0, pos * N + ff, -1), base, off["famsd"] + dd)

        def view(row_ids, r, dd, j, f, jt, ji=None, sign=1.0, with_v=True, with_mu=True):
            quality(row_ids, r, dd, sign, with_v, with_mu)
            if ji is not None:
                bias(row_ids, j, dd)
                ent(row_ids, "ji", ji, sign, off["sdji"] + j * D + dd)
            family(row_ids, j, f, dd, sign)
            ent(row_ids, "jt", jt, sign, off["ajt"] + j * D + dd)

        # suff cells: Gaussian on the mean of the cell's calls
        s_n = np.asarray(d["s_n"], dtype=float)
        n_s = len(s_r)
        self.s_n, self.s_ss = s_n, np.asarray(d["s_ss"], dtype=float)
        self.s_j, self.s_d = np.asarray(d["s_j"], dtype=np.int64), np.asarray(d["s_d"], dtype=np.int64)
        if n_s:
            link_of = np.full(n_s, -1, dtype=np.int64)
            link_of[cell_suff[linked]] = np.flatnonzero(linked)
            sj, sd = self.s_j, self.s_d
            ids = add_rows(n_s, kind=GAUSS, y=np.asarray(d["s_ybar"], dtype=float), o=self.mu_loc[sd], j=sj, d=sd,
                           vt=1, n=s_n, x=(link_of >= 0).astype(float), mode=0)
            view(ids, s_r, sd, sj, np.asarray(d["s_f"]), np.asarray(d["s_jt"]), np.asarray(d["s_ji"]))
            ent(ids, "cell", link_of, 1.0, off["acell"] + sj * D + sd)

        # readings one by one
        n_g = len(g_r)
        if n_g:
            kind = np.asarray(d["g_kind"], dtype=np.int64)
            gj, gd = np.asarray(d["g_j"], dtype=np.int64), np.asarray(d["g_d"], dtype=np.int64)
            g_cell, g_pc, g_e = (np.asarray(d[k], dtype=np.int64) for k in ("g_cell", "g_pc", "g_e"))
            g_k, g_n = np.asarray(d["g_k"], dtype=float), np.asarray(d["g_n"], dtype=float)
            explicit, shared = (g_cell >= 0).astype(float), (g_pc >= 0).astype(float)
            pieces = []
            lin = np.flatnonzero(kind == LINEAR)
            pieces.append((lin, dict(kind=GAUSS, y=np.asarray(d["g_y"], dtype=float)[lin], mode=0), None))
            cut = np.flatnonzero((kind == BINARY) | (kind == LEVELS))
            pieces.append((cut, dict(kind=INTERVAL, lo=np.asarray(d["g_lo"], dtype=float)[cut],
                                     hi=np.asarray(d["g_hi"], dtype=float)[cut], mode=1), None))
            for has_e, mode in ((True, 0), (False, 2)):
                it = np.flatnonzero((kind == ITEMS) & ((g_e >= 0) == has_e))
                for upper in (True, False):
                    weight = g_k[it] if upper else g_n[it] - g_k[it]
                    keep = weight > 0
                    sel = it[keep]
                    pieces.append((sel, dict(kind=INTERVAL, w=weight[keep], lo=0.0 if upper else -OPEN,
                                             hi=OPEN if upper else 0.0, mode=mode), has_e))
            for sel, cols, has_e in pieces:
                if not len(sel):
                    continue
                ids = add_rows(len(sel), o=self.mu_loc[gd[sel]], j=gj[sel], d=gd[sel], vt=2, x=explicit[sel],
                               sh=shared[sel], **cols)
                view(ids, g_r[sel], gd[sel], gj[sel], np.asarray(d["g_f"])[sel], np.asarray(d["g_jt"])[sel],
                     np.asarray(d["g_ji"])[sel])
                ent(ids, "cell", g_cell[sel], 1.0, off["acell"] + gj[sel] * D + gd[sel])
                ent(ids, "pc", g_pc[sel], 1.0, off["apc"] + gj[sel] * D + gd[sel])
                if has_e:
                    ent(ids, "e", g_e[sel], 1.0, -1, e=True)

        # pairs: z = eta_a - eta_b + position o against the tie width, noise sigma_pair
        n_p = len(p_ra)
        self.has_pairs = bool(n_p)
        if n_p:
            pj, pd = np.asarray(d["p_j"], dtype=np.int64), np.asarray(d["p_d"], dtype=np.int64)
            v = np.asarray(d["p_v"], dtype=np.int64)
            lo = np.where(v == 0, 0.0, np.where(v == 2, -OPEN, 0.0))
            hi = np.where(v == 0, OPEN, np.where(v == 2, 0.0, 0.0))
            lot = np.where(v == 0, 1.0, np.where(v == 2, 0.0, -1.0))
            hit = np.where(v == 0, 0.0, np.where(v == 2, -1.0, 1.0))
            ids = add_rows(n_p, kind=INTERVAL, lo=lo, hi=hi, lot=lot, hit=hit, j=pj, d=pd, vt=0, mode=3)
            # v cancels: the two runs of a pair share their input (model.build refuses others)
            view(ids, p_ra, pd, pj, np.asarray(d["p_fa"]), np.asarray(d["p_jta"]), None, 1.0, with_v=False)
            view(ids, p_rb, pd, pj, np.asarray(d["p_fb"]), np.asarray(d["p_jtb"]), None, -1.0, with_v=False)
            ent(ids, "cell", np.asarray(d["p_ca"]), 1.0, off["acell"] + pj * D + pd)
            ent(ids, "cell", np.asarray(d["p_cb"]), -1.0, off["acell"] + pj * D + pd)
            ent(ids, "position", pj, self.pair_scale[pj] * np.asarray(d["p_o"], dtype=float), -1)

        self.n_rows = n_rows
        for key, parts in rows.items():
            setattr(self, "r_" + key, np.concatenate(parts) if parts else np.zeros(0))
        self.r_kind, self.r_j, self.r_d, self.r_vt, self.r_mode = (getattr(self, "r_" + k).astype(np.int64) for k in
                                                                    ("kind", "j", "d", "vt", "mode"))
        self.gauss = self.r_kind == GAUSS
        self.all_gauss = bool(self.gauss.all())
        self._pack_entries()
        self._groups()
        self.params = Params(self)
        self.inner = "laplace" if self.all_gauss else "ep"

    def _pack_entries(self) -> None:
        """Entries to padded arrays per row: global entries (n x Kg) and local ones (n x Kl)."""
        e = {k: np.concatenate(v) if v else np.zeros(0) for k, v in self._ents.items()}
        names = list(self.lat.families)
        fam = e["fam"].astype(np.int64)
        idx = e["idx"].astype(np.int64)
        local = np.zeros(len(idx), dtype=bool)
        g = np.zeros(len(idx), dtype=np.int64)
        inp = np.zeros(len(idx), dtype=np.int64)
        p = np.zeros(len(idx), dtype=np.int64)
        for k, name in enumerate(names):
            m = fam == k
            f = self.lat.families[name]
            local[m], g[m], inp[m], p[m] = f["local"][idx[m]], f["g"][idx[m]], f["i"][idx[m]], f["p"][idx[m]]
        row = e["row"].astype(np.int64)
        n = self.n_rows
        self.row_input = np.zeros(n, dtype=np.int64)
        seen = np.full(n, -1, dtype=np.int64)
        for rr, ii in zip(row[local], inp[local]):
            if seen[rr] not in (-1, ii):
                raise ValueError("engine_laplace: a row reaches unknowns of two inputs")
            seen[rr] = ii
        self.row_input = np.maximum(seen, 0)

        def pad(mask: np.ndarray, where: np.ndarray):
            r = row[mask]
            order = np.argsort(r, kind="stable")
            r = r[order]
            counts = np.bincount(r, minlength=n)
            width = int(max(1, counts.max(initial=0)))
            start = np.concatenate([[0], np.cumsum(counts)[:-1]])
            col = np.arange(len(r)) - start[r]
            out_idx = np.zeros((n, width), dtype=np.int64)
            out_base = np.zeros((n, width))
            out_slot = np.full((n, width), -1, dtype=np.int64)
            out_e = np.zeros((n, width), dtype=bool)
            out_idx[r, col] = where[mask][order]
            out_base[r, col] = e["base"][mask][order]
            out_slot[r, col] = e["slot"][mask][order].astype(np.int64)
            out_e[r, col] = e["e"][mask][order].astype(bool)
            return out_idx, out_base, out_slot, out_e

        self.eg_idx, self.eg_base, self.eg_slot, self.eg_e = pad(~local, g)
        self.el_idx, self.el_base, self.el_slot, self.el_e = pad(local, p)
        if self.eg_e.any():
            raise ValueError("engine_laplace: an item noise unknown spans two inputs")

    # ---- coefficients and row parameters -----------------------------------------------------------

    def slots(self, nu: dict) -> np.ndarray:
        D, J = self.D, self.J
        lin = nu["sd_in"][:, None] * nu["Lc_in"]
        lrun = nu["sd_run"][:, None] * nu["Lc_run"]
        rh, r1h, rc = np.sqrt(nu["h"]), np.sqrt(1.0 - nu["h"]), np.sqrt(nu["hc"])
        return np.concatenate([lin.ravel(), lrun.ravel(), nu["famsd"], nu["sdji"].ravel(),
                               (nu["tau"] * rh[:, None]).ravel(), (nu["tau"] * r1h[:, None]).ravel(),
                               (nu["sigma"] * rc[:, None]).ravel(), [1.0]])

    def row_var(self, nu: dict) -> np.ndarray:
        j, d = self.r_j, self.r_d
        sig2 = nu["sigma"][j, d] ** 2
        own = nu["tau"][j, d] ** 2 * (1.0 - nu["h"][j])
        s2 = np.where(self.r_vt == 1, sig2 / self.r_n, sig2 * (1.0 - nu["hc"][j] * self.r_sh))
        var = s2 + own * (1.0 - self.r_x)
        return np.where(self.r_vt == 0, 1.0, var)

    def row_rho(self, nu: dict, var: np.ndarray) -> np.ndarray:
        m = self.r_mode
        rho = np.ones(self.n_rows)
        rho = np.where(m == 1, 1.0 / np.sqrt(var), rho)
        rho = np.where(m == 2, 1.0 / np.sqrt(1.0 + var), rho)
        return np.where(m == 3, 1.0 / nu["sp"][self.r_j, self.r_d], rho)

    def coefficients(self, nu: dict) -> dict:
        gam = self.slots(nu)
        var = self.row_var(nu)
        rho = self.row_rho(nu, var)
        cg = rho[:, None] * self.eg_base * gam[self.eg_slot]
        cl = rho[:, None] * self.el_base * gam[self.el_slot] * np.where(self.el_e, np.sqrt(var)[:, None], 1.0)
        tie = nu["tie"][self.r_j]
        lo = self.r_lo + self.r_lot * tie
        hi = self.r_hi + self.r_hit * tie
        return {"gam": gam, "var": var, "rho": rho, "cg": cg, "cl": cl, "o": rho * self.r_o, "lo_u": lo, "hi_u": hi,
                "lo": rho * lo, "hi": rho * hi}

    # ---- the linear predictor and the Hessian ----------------------------------------------------

    def eta(self, c: dict, zg: np.ndarray, zb: np.ndarray) -> np.ndarray:
        return (c["o"] + (c["cg"] * zg[self.eg_idx]).sum(axis=1)
                + (c["cl"] * zb[self.row_input[:, None], self.el_idx]).sum(axis=1))

    def transpose(self, c: dict, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X' r as (global, per input)."""
        rg = np.bincount(self.eg_idx.ravel(), (c["cg"] * r[:, None]).ravel(), minlength=self.G)
        flat = (self.row_input[:, None] * self.B + self.el_idx).ravel()
        rb = np.bincount(flat, (c["cl"] * r[:, None]).ravel(), minlength=self.I * self.B).reshape(self.I, self.B)
        return rg, rb

    def hessian(self, c: dict, W: np.ndarray) -> Factor:
        G, I, B = self.G, self.I, self.B
        cg, cl, ig, il, ri = c["cg"], c["cl"], self.eg_idx, self.el_idx, self.row_input
        wg = cg * W[:, None]
        gg = np.bincount((ig[:, :, None] * G + ig[:, None, :]).ravel(), (wg[:, :, None] * cg[:, None, :]).ravel(),
                         minlength=G * G).reshape(G, G)
        flat = ((ri[:, None, None] * B + il[:, :, None]) * G + ig[:, None, :]).ravel()
        bg = np.bincount(flat, (cl[:, :, None] * wg[:, None, :]).ravel(), minlength=I * B * G).reshape(I, B, G)
        wl = cl * W[:, None]
        flat = ((ri[:, None, None] * B + il[:, :, None]) * B + il[:, None, :]).ravel()
        bb = np.bincount(flat, (wl[:, :, None] * cl[:, None, :]).ravel(), minlength=I * B * B).reshape(I, B, B)
        gg[np.diag_indices(G)] += 1.0
        bb += np.eye(B)[None]
        return Factor(gg, bg, bb)

    def kernel(self, c: dict, eta: np.ndarray) -> dict:
        """Per row: w log p and its derivatives (all weighted by the row's count)."""
        n = self.n_rows
        out = {k: np.zeros(n) for k in ("logp", "d1", "d2", "d3", "dvar", "d1_dvar", "d2_dvar", "dlo", "dhi",
                                        "d1_dlo", "d1_dhi", "d2_dlo", "d2_dhi")}
        g = self.gauss
        if g.any():
            k = gaussian_kernel(eta[g], self.r_y[g], c["var"][g])
            for name, value in k.items():
                out[name][g] = value
        t = ~g
        if t.any():
            k = interval_kernel(eta[t], c["lo"][t], c["hi"][t])
            for name, value in k.items():
                out[name][t] = value
        w = self.r_w
        return {name: value * w for name, value in out.items()}

    # ---- the inner step --------------------------------------------------------------------------

    def mode(self, c: dict, zg: np.ndarray, zb: np.ndarray, tol: float = 1e-10, max_iter: int = 100) -> dict:
        """Newton steps to z-hat. Returns z-hat, the kernel and the factor of H at z-hat, and the steps."""
        def objective(zg, zb):
            eta = self.eta(c, zg, zb)
            k = self.kernel(c, eta)
            return float(k["logp"].sum() - 0.5 * (zg @ zg) - 0.5 * (zb * zb).sum()), eta, k

        f, eta, k = objective(zg, zb)
        steps, ok, fac, fresh = 0, False, None, False
        for steps in range(1, max_iter + 1):
            fac, fresh = self.hessian(c, -k["d2"]), True
            rg, rb = self.transpose(c, k["d1"])
            rg, rb = rg - zg, rb - zb
            dg, db = fac.solve(rg, rb)
            decrement = float(rg @ dg + (rb * db).sum())
            if self.all_gauss:   # a quadratic: one full step is the mode, and H does not move
                zg, zb = zg + dg, zb + db
                f, eta, k = objective(zg, zb)
                return {"zg": zg, "zb": zb, "eta": eta, "k": k, "factor": fac, "f": f, "steps": steps, "ok": True}
            if decrement < 2.0 * tol:
                # one last full step: Newton is quadratic here, so it takes the mode error from about
                # sqrt(decrement) to its square; the log determinant and the gradient are first order in it
                ng, nb = zg + dg, zb + db
                nf, neta, nk = objective(ng, nb)
                if np.isfinite(nf) and nf >= f - 1e-12 * abs(f):
                    zg, zb, f, eta, k, fresh = ng, nb, nf, neta, nk, False
                ok = True
                break
            step = 1.0
            for _ in range(40):
                ng, nb = zg + step * dg, zb + step * db
                nf, neta, nk = objective(ng, nb)
                if np.isfinite(nf) and nf >= f - 1e-12 * abs(f):
                    break
                step *= 0.5
            else:
                break
            zg, zb, f, eta, k, fresh = ng, nb, nf, neta, nk, False
        if not fresh:
            fac = self.hessian(c, -k["d2"])
        return {"zg": zg, "zb": zb, "eta": eta, "k": k, "factor": fac, "f": f, "steps": steps, "ok": ok}

    # ---- the outer objective and its gradient ------------------------------------------------------

    def suff_noise(self, nu: dict) -> tuple[float, np.ndarray]:
        """The part of the likelihood of cells of independent calls that is not in their mean: the spread
        of the calls around it. `(value, d value / d sigma[s])` per suff row."""
        if not len(self.s_n):
            return 0.0, np.zeros(0)
        sig = nu["sigma"][self.s_j, self.s_d]
        n1 = self.s_n - 1.0
        value = float((-0.5 * n1 * (LOG_2PI + 2.0 * np.log(sig)) - 0.5 * self.s_ss / sig ** 2
                       - 0.5 * np.log(self.s_n)).sum())
        return value, -n1 / sig + self.s_ss / sig ** 3

    def _groups(self) -> None:
        """Cut readings that share one linear predictor form one site of the EP inner step: the calls of a
        cell, the passed and failed items of a call, the pairs a judge saw shown the same way. Within a
        site the readings are integrated exactly (`cell_integral`), so repeated sharp calls are not
        counted as separate evidence."""
        t = np.flatnonzero(~self.gauss)
        n = self.n_rows
        self.g_of_row = np.full(n, -1, dtype=np.int64)
        if not len(t):
            self.n_groups, self.g_rep, self.g_rows = 0, np.zeros(0, np.int64), np.zeros((0, 1), np.int64)
            self.g_simple = np.zeros(0, bool)
            return
        cols = [self.row_input, self.r_o, self.r_mode, self.r_j, self.r_d, self.r_vt, self.r_n, self.r_x, self.r_sh,
                self.r_kind]
        mats = [self.eg_idx, self.eg_base, self.eg_slot, self.el_idx, self.el_base, self.el_slot, self.el_e]
        key = np.column_stack([np.asarray(col, float)[t] for col in cols] +
                              [np.asarray(mat, float)[t].reshape(len(t), -1) for mat in mats])
        _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inv = np.asarray(inv).ravel()
        ng = len(first)
        self.n_groups = ng
        self.g_rep = t[first]
        self.g_of_row[t] = inv
        size = np.bincount(inv, minlength=ng)
        order = np.argsort(inv, kind="stable")
        pos = np.arange(len(t)) - np.repeat(np.cumsum(size) - size, size)
        rows = np.full((ng, int(size.max())), -1, dtype=np.int64)
        rows[inv[order], pos] = t[order]
        self.g_rows = rows
        self.g_simple = (size == 1) & (self.r_w[self.g_rep] == 1.0)

    def marginals(self, c: dict, fac: Factor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per row, the variance of eta under N(., H^-1), and `X H^-1` on the row's entries (global, local)."""
        sgg, sbg, sbb = fac.selected()
        n, ri = self.n_rows, self.row_input
        xs_g = np.zeros(self.eg_idx.shape)
        xs_l = np.zeros(self.el_idx.shape)
        for start_row in range(0, n, CHUNK):
            sl = slice(start_row, min(n, start_row + CHUNK))
            ig, il, i = self.eg_idx[sl], self.el_idx[sl], ri[sl]
            cg, cl = c["cg"][sl], c["cl"][sl]
            s_gg = sgg[ig[:, :, None], ig[:, None, :]]
            s_lg = sbg[i[:, None, None], il[:, :, None], ig[:, None, :]]
            s_ll = sbb[i[:, None, None], il[:, :, None], il[:, None, :]]
            xs_g[sl] = np.einsum("ne,nef->nf", cg, s_gg) + np.einsum("ne,nef->nf", cl, s_lg)
            xs_l[sl] = np.einsum("ne,nfe->nf", cg, s_lg) + np.einsum("ne,nef->nf", cl, s_ll)
        v = (c["cg"] * xs_g).sum(axis=1) + (c["cl"] * xs_l).sum(axis=1)
        return v, xs_g, xs_l

    def tilted(self, c: dict, muc: np.ndarray, sc2: np.ndarray, *, bounds: bool = False, panels: int = PANELS,
               guess: dict | None = None) -> dict:
        """Per group: log Z, mean and variance of the group's predictor under cavity times readings.
        `guess` (the `guess` of an earlier result) starts the searches of the cell rule."""
        ng = self.n_groups
        out = {"logZ": np.zeros(ng), "mean": np.zeros(ng), "var": np.zeros(ng), "tail": np.zeros(ng)}
        if bounds:
            out["dlo"], out["dhi"] = np.zeros(self.g_rows.shape), np.zeros(self.g_rows.shape)
        one = self.g_simple
        if one.any():
            rep = self.g_rep[one]
            sd = np.sqrt(1.0 + sc2[one])
            k = interval_kernel(muc[one] / sd, c["lo"][rep] / sd, c["hi"][rep] / sd)
            out["logZ"][one] = k["logp"]
            out["mean"][one] = muc[one] + sc2[one] * k["d1"] / sd
            out["var"][one] = sc2[one] * (1.0 + sc2[one] * k["d2"] / sd ** 2)
            if bounds:
                out["dlo"][one, 0], out["dhi"][one, 0] = k["dlo"] / sd, k["dhi"] / sd
        many = ~one
        if many.any():
            rows = self.g_rows[many]
            use = rows >= 0
            rr = np.where(use, rows, 0)
            q = cell_integral(muc[many], sc2[many], np.where(use, c["lo"][rr], -OPEN), np.where(use, c["hi"][rr], OPEN),
                              np.where(use, self.r_w[rr], 0.0), panels=panels, bounds=bounds, guess=guess)
            for name in ("logZ", "mean", "var", "tail"):
                out[name][many] = q[name]
            out["guess"] = {"mode": q["mode"], "A": q["A"], "B": q["B"]}
            if bounds:
                out["dlo"][many], out["dhi"][many] = q["dlo"], q["dhi"]
        return out

    def ep(self, c: dict, sites: tuple[np.ndarray, np.ndarray] | None = None, *, grad: bool = False,
           tol: float = 1e-10, max_iter: int = 500, damping: float = 0.7) -> dict:
        # `tol` bounds the relative change of every site in the last sweep. The EP value is stationary in
        # the sites, so its error is about the square of that; the gradient's is first order in it.
        """Expectation propagation over the same unknowns, one site per group of cut readings (`_groups`).

        A Gaussian reading's site is exact. A group's site is a Gaussian on its predictor, fitted so that
        the posterior's moments of the predictor match those of the cavity times the group's readings;
        those moments are one-dimensional integrals, in closed form for a single reading and by
        `cell_integral` otherwise. Parallel damped updates, from `sites` (a warm start) or flat. Returns
        the posterior mean (`zg`, `zb`), the factor of its precision, the sites, the EP value of
        log p(y | theta) (without the spread of suff cells), and with `grad` the row terms the gradient
        needs."""
        n, w, g = self.n_rows, self.r_w, self.gauss
        ng, rep = self.n_groups, self.g_rep
        var = c["var"]
        tau_g, nu_g = np.zeros(ng), np.zeros(ng)
        guess = None
        if sites is not None and len(sites[0]) == ng:
            tau_g, nu_g = sites[0].copy(), sites[1].copy()
            guess = sites[2] if len(sites) > 2 else None
        W_row, N_row = np.zeros(n), np.zeros(n)
        W_row[g], N_row[g] = w[g] / var[g], w[g] * self.r_y[g] / var[g]
        ok, steps, change, last = False, 0, np.inf, np.inf
        hist_g: list[np.ndarray] = []
        hist_f: list[np.ndarray] = []
        for steps in range(1, max_iter + 1):
            W_row[rep], N_row[rep] = tau_g, nu_g
            fac = self.hessian(c, W_row)
            rg, rb = self.transpose(c, N_row - W_row * c["o"])
            mg, mb = fac.solve(rg, rb)
            mu = self.eta(c, mg, mb)
            v, xs_g, xs_l = self.marginals(c, fac)
            if not ng:
                ok = True
                break
            vt, mut = v[rep], mu[rep]
            tc = np.maximum(1.0 / vt - tau_g, 1e-12 / vt)       # cavity precision
            sc2 = 1.0 / tc
            muc = (mut / vt - nu_g) * sc2
            tl = self.tilted(c, muc, sc2, guess=guess)
            guess = tl.get("guess")
            vt_new = np.maximum(tl["var"], 1e-300)
            tau_new = np.maximum(1.0 / vt_new - tc, 0.0)
            nu_new = tl["mean"] / vt_new - muc * tc
            change = max(float(np.max(np.abs(tau_new - tau_g) / (1.0 + tau_g), initial=0.0)),
                         float(np.max(np.abs(nu_new - nu_g) / (1.0 + np.abs(nu_g)), initial=0.0)))
            if change < tol or steps == max_iter:   # stop before an update, so the value matches the sites
                ok = change < tol
                break
            # Anderson acceleration of the sweep map (memory ANDERSON), restarted with a damped step
            # whenever a sweep grows the change
            x_old, x_new = np.concatenate([tau_g, nu_g]), np.concatenate([tau_new, nu_new])
            if change >= last:
                hist_g, hist_f = [], []
                damping = max(0.05, damping * 0.5)
            hist_g.append(x_new)
            hist_f.append(x_new - x_old)
            del hist_g[:-(ANDERSON + 1)], hist_f[:-(ANDERSON + 1)]
            last = change
            if len(hist_f) >= 2:
                sc = 1.0 / (1.0 + np.abs(x_old))
                dF = np.stack([b - a for a, b in zip(hist_f[:-1], hist_f[1:])], axis=1) * sc[:, None]
                dG = np.stack([b - a for a, b in zip(hist_g[:-1], hist_g[1:])], axis=1)
                gamma = np.linalg.lstsq(dF, hist_f[-1] * sc, rcond=None)[0]
                x = x_new - dG @ gamma
            else:
                x = (1.0 - damping) * x_old + damping * x_new
            tau_g, nu_g = np.maximum(x[:ng], 0.0), x[ng:]
        value = -0.5 * fac.logdet + 0.5 * float(rg @ mg + (rb * mb).sum()) \
            + float((-0.5 * W_row * c["o"] ** 2 + N_row * c["o"]).sum())
        value += float((w[g] * (-0.5 * (LOG_2PI + np.log(var[g])) - 0.5 * self.r_y[g] ** 2 / var[g])).sum())
        out = {"zg": mg, "zb": mb, "eta": mu, "factor": fac, "sites": (tau_g, nu_g, guess), "steps": steps, "ok": ok,
               "change": change, "inner": "ep", "v": v, "xs_g": xs_g, "xs_l": xs_l, "tail": 0.0}
        if ng:
            value += float((tl["logZ"] - 0.5 * np.log(tc * vt) - 0.5 * (mut ** 2 / vt - muc ** 2 * tc)).sum())
            out.update(tail=float(tl["tail"].max()), cavity=(muc, sc2))
        out["f"] = value
        if grad:
            out["g1"], out["h2"] = N_row - W_row * mu, -W_row
            dlo, dhi = np.zeros(n), np.zeros(n)
            if ng:
                tb = self.tilted(c, muc, sc2, bounds=True, guess=guess)
                use = self.g_rows >= 0
                dlo[self.g_rows[use]] = tb["dlo"][use]
                dhi[self.g_rows[use]] = tb["dhi"][use]
            out["dlo"], out["dhi"] = dlo, dhi
        return out

    def evaluate(self, theta: np.ndarray, start: Any = None, *, grad: bool = True, inner: str | None = None,
                 tol: float | None = None) -> dict:
        """The penalised objective at theta, with its gradient when `grad`.

        `inner` is "laplace" (the Laplace approximation at the mode) or "ep" (expectation propagation);
        default `self.inner`. `start` warm-starts the inner step: an earlier result's `mode`, a pair
        (zg, zb), or None."""
        inner = inner or self.inner
        P = self.params
        nu = P.unpack(theta)
        c = self.coefficients(nu)
        zg, zb, sites = np.zeros(self.G), np.zeros((self.I, self.B)), None
        if isinstance(start, dict):
            zg, zb, sites = start["zg"], start["zb"], start.get("sites")
        elif start is not None:
            zg, zb = start
        ss_value, ss_grad = self.suff_noise(nu)
        prior, prior_grad = P.log_prior(theta)
        if inner == "ep" and not self.all_gauss:
            m = self.ep(c, sites, grad=grad, tol=EP_TOL if tol is None else tol)
            fac = m["factor"]
            laplace = m["f"] + ss_value
        else:
            m = self.mode(c, zg, zb)
            m["inner"] = "laplace"
            fac = m["factor"]
            laplace = m["f"] - 0.5 * fac.logdet + ss_value
        out = {"value": laplace + prior, "laplace": laplace, "prior": prior, "mode": m, "nu": nu, "c": c}
        if not grad:
            return out
        zg, zb = m["zg"], m["zb"]
        n, ri, w = self.n_rows, self.row_input, self.r_w
        if m["inner"] == "ep":
            # at the EP fixed point the value is stationary in the sites and in the row marginals: the
            # gradient is that of a Gaussian model with the sites as observations, plus each site's
            # log normaliser at its fixed cavity for the bounds and the call noise
            v, xs_g, xs_l = m["v"], m["xs_g"], m["xs_l"]
            mu = m["eta"]
            g1, h2 = m["g1"], m["h2"]
            G_row = g1
            lam_g, lam_b, xlam = np.zeros(self.G), np.zeros((self.I, self.B)), np.zeros(n)
            var = c["var"]
            dvar_k = np.where(self.gauss, w * (-0.5 / var + 0.5 * ((self.r_y - mu) ** 2 + v) / var ** 2), 0.0)
            dlo, dhi = m["dlo"], m["dhi"]
        else:
            k = m["k"]
            v, xs_g, xs_l = self.marginals(c, fac)   # posterior variance of eta per row
            g1, h2, t3 = k["d1"], k["d2"], k["d3"]
            r = 0.5 * v * t3
            if np.any(r != 0.0):
                lam_g, lam_b = fac.solve(*self.transpose(c, r))
                xlam = self.eta({**c, "o": np.zeros(n)}, lam_g, lam_b)
            else:
                lam_g, lam_b, xlam = np.zeros(self.G), np.zeros((self.I, self.B)), np.zeros(n)
            G_row = g1 + r + xlam * h2
            dvar_k = k["dvar"] + 0.5 * v * k["d2_dvar"] + xlam * k["d1_dvar"]
            dlo = k["dlo"] + 0.5 * v * k["d2_dlo"] + xlam * k["d1_dlo"]
            dhi = k["dhi"] + 0.5 * v * k["d2_dhi"] + xlam * k["d1_dhi"]
        # d value / d coefficient, per entry
        dcg = G_row[:, None] * zg[self.eg_idx] + g1[:, None] * lam_g[self.eg_idx] + h2[:, None] * xs_g
        zl = zb[ri[:, None], self.el_idx]
        laml = lam_b[ri[:, None], self.el_idx]
        dcl = G_row[:, None] * zl + g1[:, None] * laml + h2[:, None] * xs_l
        rho = c["rho"]
        # through rho: every coefficient, the offset and both bounds of a row are scaled by it
        drho = ((dcg * c["cg"]).sum(axis=1) + (dcl * c["cl"]).sum(axis=1)) / rho + G_row * self.r_o \
            + dlo * c["lo_u"] + dhi * c["hi_u"]
        # slots
        gam = c["gam"]
        dgam = np.bincount(self.eg_slot.ravel() % len(gam), (dcg * rho[:, None] * self.eg_base).ravel(),
                           minlength=len(gam))
        lfac = np.where(self.el_e, np.sqrt(c["var"])[:, None], 1.0)
        dgam += np.bincount(self.el_slot.ravel() % len(gam), (dcl * rho[:, None] * self.el_base * lfac).ravel(),
                            minlength=len(gam))
        # row variances
        var = c["var"]
        dvar = np.where(self.gauss, dvar_k, 0.0)
        dvar += np.where(self.r_mode == 1, drho * (-0.5) * var ** -1.5, 0.0)
        dvar += np.where(self.r_mode == 2, drho * (-0.5) * (1.0 + var) ** -1.5, 0.0)
        e_part = np.where(self.el_e, dcl * rho[:, None] * self.el_base * gam[self.el_slot], 0.0).sum(axis=1)
        dvar += e_part * 0.5 / np.sqrt(var)
        dnu = P.zeros()
        J, D = self.J, self.D
        dsp = np.where(self.r_mode == 3, drho * (-rho * rho), 0.0)
        dnu["sp"] += np.bincount(self.r_j * D + self.r_d, dsp, minlength=J * D).reshape(J, D)
        dnu["tie"] += np.bincount(self.r_j, (dlo * self.r_lot + dhi * self.r_hit) * rho, minlength=J)
        self._var_pullback(nu, dvar, dnu)
        if len(self.s_n):
            dnu["sigma"] += np.bincount(self.s_j * D + self.s_d, ss_grad, minlength=J * D).reshape(J, D)
        self._slot_pullback(nu, dgam, dnu)
        out["grad"] = P.pullback(theta, nu, dnu) + prior_grad
        return out

    def _var_pullback(self, nu: dict, dvar: np.ndarray, dnu: dict) -> None:
        j, d, D, J = self.r_j, self.r_d, self.D, self.J
        live = self.r_vt > 0
        sig, tau, h, hc = nu["sigma"][j, d], nu["tau"][j, d], nu["h"][j], nu["hc"][j]
        suff = self.r_vt == 1
        dsig = np.where(suff, 2.0 * sig / self.r_n, 2.0 * sig * (1.0 - hc * self.r_sh)) * dvar
        dtau = 2.0 * tau * (1.0 - h) * (1.0 - self.r_x) * dvar
        dh = -tau ** 2 * (1.0 - self.r_x) * dvar
        dhc = np.where(suff, 0.0, -sig ** 2 * self.r_sh) * dvar
        flat = j * D + d
        dnu["sigma"] += np.bincount(flat[live], dsig[live], minlength=J * D).reshape(J, D)
        dnu["tau"] += np.bincount(flat[live], dtau[live], minlength=J * D).reshape(J, D)
        dnu["h"] += np.bincount(j[live], dh[live], minlength=J)
        dnu["hc"] += np.bincount(j[live], dhc[live], minlength=J)

    def _slot_pullback(self, nu: dict, dgam: np.ndarray, dnu: dict) -> None:
        D, J, off = self.D, self.J, self.slot_off
        dlin = dgam[off["Lin"]:off["Lin"] + D * D].reshape(D, D)
        dlrun = dgam[off["Lrun"]:off["Lrun"] + D * D].reshape(D, D)
        dnu["sd_in"] += (dlin * nu["Lc_in"]).sum(axis=1)
        dnu["Lc_in"] += dlin * nu["sd_in"][:, None]
        dnu["sd_run"] += (dlrun * nu["Lc_run"]).sum(axis=1)
        dnu["Lc_run"] += dlrun * nu["sd_run"][:, None]
        dnu["famsd"] += dgam[off["famsd"]:off["famsd"] + D]
        dnu["sdji"] += dgam[off["sdji"]:off["sdji"] + J * D].reshape(J, D)
        dajt = dgam[off["ajt"]:off["ajt"] + J * D].reshape(J, D)
        dacell = dgam[off["acell"]:off["acell"] + J * D].reshape(J, D)
        dapc = dgam[off["apc"]:off["apc"] + J * D].reshape(J, D)
        rh, r1h, rc = np.sqrt(nu["h"]), np.sqrt(1.0 - nu["h"]), np.sqrt(nu["hc"])
        dnu["tau"] += dajt * rh[:, None] + dacell * r1h[:, None]
        dnu["sigma"] += dapc * rc[:, None]
        dnu["sqrt_h"] += (dajt * nu["tau"]).sum(axis=1)
        dnu["sqrt_1mh"] += (dacell * nu["tau"]).sum(axis=1)
        dnu["sqrt_hc"] += (dapc * nu["sigma"]).sum(axis=1)


# ---------------------------------------------------------------------------------------------------
# Hyperparameters: unconstrained theta, natural values, priors

HN, GAMMA, REL, BETA12, CPC = 0, 1, 2, 3, 4


class Params:
    """theta: one unconstrained number per hyperparameter that reaches the likelihood.

    Prior codes (with the Jacobian of the transform, so the penalty is the log density of theta):
    `HN` log of an sd with a half-normal(c) prior; `GAMMA` log of an sd with a Gamma(2, rate c) prior;
    `REL` a relative log sd with a N(0, 0.5) prior; `BETA12` logit of a share with a Beta(1, 2) prior;
    `CPC` atanh of a canonical partial correlation with an LKJ(2) prior (c holds the Beta parameter)."""

    def __init__(self, prob: Problem):
        D, J = prob.D, prob.J
        s = prob.scale
        wide = np.broadcast_to(s, (J, D))
        self.D, self.J = D, J
        self.labels: list[str] = []
        codes, consts, inits = [], [], []
        self.idx: dict[str, np.ndarray] = {}

        def add(name: str, shape: tuple, use: np.ndarray, code, const, init) -> None:
            use = np.broadcast_to(np.asarray(use, dtype=bool), shape)
            code = np.broadcast_to(np.asarray(code), shape)
            const = np.broadcast_to(np.asarray(const, dtype=float), shape)
            init = np.broadcast_to(np.asarray(init, dtype=float), shape)
            idx = np.full(shape, -1, dtype=np.int64)
            for pos in zip(*np.nonzero(use)):
                idx[pos] = len(codes)
                codes.append(int(code[pos]))
                consts.append(float(const[pos]))
                inits.append(float(init[pos]))
                self.labels.append(f"{name}{list(int(p) for p in pos)}")
            self.idx[name] = idx

        has_score = np.zeros((J, D), dtype=bool)
        used_tau = np.zeros((J, D), dtype=bool)
        live = prob.r_vt > 0
        has_score[prob.r_j[live], prob.r_d[live]] = True
        used_tau |= has_score
        cell_j, cell_d = np.asarray(prob.data["cell_j"], dtype=np.int64), np.asarray(prob.data["cell_d"], dtype=np.int64)
        used_tau[cell_j, cell_d] = True
        jt_judges = np.zeros(J, dtype=bool)
        for key in ("s", "g"):
            sel = np.asarray(prob.data[f"{key}_jt"]) >= 0
            jt_judges[np.asarray(prob.data[f"{key}_j"], dtype=np.int64)[sel]] = True
        if prob.has_pairs:
            sel = (np.asarray(prob.data["p_jta"]) >= 0) | (np.asarray(prob.data["p_jtb"]) >= 0)
            jt_judges[np.asarray(prob.data["p_j"], dtype=np.int64)[sel]] = True
            used_tau[np.asarray(prob.data["p_j"], dtype=np.int64)[sel], np.asarray(prob.data["p_d"],
                                                                                   dtype=np.int64)[sel]] = True
        pc_judges = np.zeros(J, dtype=bool)
        sel = np.asarray(prob.data["g_pc"]) >= 0
        pc_judges[np.asarray(prob.data["g_j"], dtype=np.int64)[sel]] = True
        ji_used = np.zeros((J, D), dtype=bool)
        ji_used[np.asarray(prob.data["ji_j"], dtype=np.int64), np.asarray(prob.data["ji_d"], dtype=np.int64)] = True
        pair_jd = np.zeros((J, D), dtype=bool)
        if prob.has_pairs:
            pair_jd[np.asarray(prob.data["p_j"], dtype=np.int64), np.asarray(prob.data["p_d"], dtype=np.int64)] = True
        self.has_score, self.pair_jd = has_score, pair_jd

        add("sd_input", (D,), True, HN, s, np.log(0.5 * s))
        add("sd_run", (D,), True, HN, s, np.log(0.5 * s))
        P = cpc_count(D) if prob.form == "full" else 0
        add("cpc_input", (P,), True, CPC, cpc_levels(D) if P else [], 0.0)
        add("cpc_run", (P,), True, CPC, cpc_levels(D) if P else [], 0.0)
        add("tau", (J, D), used_tau, HN, wide, np.log(0.5 * wide))
        fixed = prob.sigma_fixed
        sig_code = np.where(fixed, REL, np.where(prob.levels_dim, GAMMA, HN))
        sig_const = np.where(fixed, 0.0, np.where(prob.levels_dim, 2.0 / (0.8 * wide),
                                                  np.where(prob.tight, PERSON_SIGMA, wide)))
        sig_init = np.where(fixed, 0.0, np.where(prob.levels_dim, np.log(0.8 * wide),
                                                 np.log(np.where(prob.tight, PERSON_SIGMA, 0.5 * wide))))
        add("sigma", (J, D), has_score, sig_code, sig_const, sig_init)
        add("halo_target", (J,), prob.halo_on & jt_judges, BETA12, 0.0, -np.log(2.0))
        add("halo_call", (J,), pc_judges, BETA12, 0.0, -np.log(2.0))
        add("family_sd", (D,), prob.family_on.any(axis=0), HN, 0.4 * s, np.log(0.2 * s))
        add("sd_judge_input", (J, D), ji_used, HN, 0.6 * wide, np.log(0.3 * wide))
        only = np.broadcast_to(prob.pairs_only, (J, D))
        add("sigma_pair", (J, D), pair_jd, np.where(only, REL, GAMMA), np.where(only, 0.0, 2.0 / (1.6 * wide)),
            np.where(only, 0.0, np.log(1.6 * wide)))
        add("tie_width", (J,), prob.pair_judges, HN, prob.pair_scale, np.log(0.5 * prob.pair_scale))
        self.code = np.asarray(codes, dtype=np.int64)
        self.const = np.asarray(consts, dtype=float)
        self.init = np.asarray(inits, dtype=float)
        self.size = len(codes)
        self.scale = s
        self.pair_scale = prob.pair_scale
        self.sigma_w, self.pair_w = prob.sigma_w, prob.pair_w
        self.fixed, self.only = fixed, only
        self.tight, self.levels_dim = prob.tight, prob.levels_dim
        self.lo, self.hi = self._bounds()

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = np.empty(self.size), np.empty(self.size)
        code, const = self.code, self.const
        logs = (code == HN) | (code == GAMMA)
        centre = np.where(code == HN, np.log(np.maximum(const, 1e-300)), np.log(2.0 / np.maximum(const, 1e-300)))
        lo[logs], hi[logs] = centre[logs] - 10.0, centre[logs] + 5.0
        for c, (a, b) in ((REL, (-8.0, 8.0)), (BETA12, (-14.0, 14.0)), (CPC, (-6.0, 6.0))):
            lo[code == c], hi[code == c] = a, b
        return lo, hi

    def zeros(self) -> dict:
        D, J = self.D, self.J
        return {"sd_in": np.zeros(D), "sd_run": np.zeros(D), "Lc_in": np.zeros((D, D)), "Lc_run": np.zeros((D, D)),
                "tau": np.zeros((J, D)), "sigma": np.zeros((J, D)), "h": np.zeros(J), "hc": np.zeros(J),
                "sqrt_h": np.zeros(J), "sqrt_1mh": np.zeros(J), "sqrt_hc": np.zeros(J), "famsd": np.zeros(D),
                "sdji": np.zeros((J, D)), "sp": np.zeros((J, D)), "tie": np.zeros(J)}

    def _take(self, theta: np.ndarray, name: str, default: float | np.ndarray) -> np.ndarray:
        idx = self.idx[name]
        out = np.array(np.broadcast_to(default, idx.shape), dtype=float)
        use = idx >= 0
        out[use] = theta[idx[use]]
        return out

    def unpack(self, theta: np.ndarray) -> dict:
        D, J, s = self.D, self.J, self.scale
        wide = np.broadcast_to(s, (J, D))
        nu = {"sd_in": np.exp(self._take(theta, "sd_input", np.log(0.5 * s))),
              "sd_run": np.exp(self._take(theta, "sd_run", np.log(0.5 * s)))}
        for name, key in (("cpc_input", "Lc_in"), ("cpc_run", "Lc_run")):
            t = theta[self.idx[name]] if len(self.idx[name]) else np.zeros(0)
            nu[key] = cpc_chol(t, D) if len(t) else np.eye(D)
            nu["J_" + key] = cpc_jacobian(t, D) if len(t) else np.zeros((0, D, D))
        nu["tau"] = np.exp(self._take(theta, "tau", np.log(0.5 * wide)))
        raw = self._take(theta, "sigma", 0.0)
        rel = np.exp(raw - (raw * self.sigma_w).sum(axis=0, keepdims=True))
        nu["sigma"] = np.where(self.fixed, rel, np.where(self.idx["sigma"] >= 0, np.exp(raw), 0.5 * wide))
        nu["h"] = special.expit(self._take(theta, "halo_target", -np.inf))
        nu["hc"] = special.expit(self._take(theta, "halo_call", -np.inf))
        nu["famsd"] = np.exp(self._take(theta, "family_sd", np.log(0.2 * s)))
        nu["sdji"] = np.exp(self._take(theta, "sd_judge_input", np.log(0.3 * wide)))
        raw = self._take(theta, "sigma_pair", 0.0)
        rel = np.exp(raw - (raw * self.pair_w).sum(axis=0, keepdims=True))
        nu["sp"] = np.where(self.only, rel, np.where(self.idx["sigma_pair"] >= 0, np.exp(raw), 1.6 * wide))
        nu["tie"] = np.exp(self._take(theta, "tie_width", np.log(0.5 * self.pair_scale)))
        return nu

    def pullback(self, theta: np.ndarray, nu: dict, dnu: dict) -> np.ndarray:
        """d / d theta from d / d nu (the chain rule through `unpack`)."""
        out = np.zeros(self.size)

        def put(name: str, values: np.ndarray) -> None:
            idx = self.idx[name]
            use = idx >= 0
            np.add.at(out, idx[use], values[use])

        put("sd_input", dnu["sd_in"] * nu["sd_in"])
        put("sd_run", dnu["sd_run"] * nu["sd_run"])
        for name, key in (("cpc_input", "Lc_in"), ("cpc_run", "Lc_run")):
            if len(self.idx[name]):
                put(name, np.einsum("kab,ab->k", nu["J_" + key], dnu[key]))
        put("tau", dnu["tau"] * nu["tau"])
        g = dnu["sigma"] * nu["sigma"]
        rel = g - self.sigma_w * np.where(self.fixed, g, 0.0).sum(axis=0, keepdims=True)
        put("sigma", np.where(self.fixed, rel, g))
        for name, key, sq, sq1 in (("halo_target", "h", "sqrt_h", "sqrt_1mh"), ("halo_call", "hc", "sqrt_hc", None)):
            h = nu[key]
            total = dnu[key] * h * (1.0 - h) + dnu[sq] * 0.5 * np.sqrt(h) * (1.0 - h)
            if sq1:
                total = total - dnu[sq1] * 0.5 * h * np.sqrt(1.0 - h)
            put(name, total)
        put("family_sd", dnu["famsd"] * nu["famsd"])
        put("sd_judge_input", dnu["sdji"] * nu["sdji"])
        g = dnu["sp"] * nu["sp"]
        rel = g - self.pair_w * np.where(self.only, g, 0.0).sum(axis=0, keepdims=True)
        put("sigma_pair", np.where(self.only, rel, g))
        put("tie_width", dnu["tie"] * nu["tie"])
        return out

    def log_prior(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        code, c = self.code, self.const
        lp, g = np.zeros(self.size), np.zeros(self.size)
        m = code == HN
        x = np.exp(theta[m]) / c[m]
        lp[m], g[m] = theta[m] - 0.5 * x * x, 1.0 - x * x
        m = code == GAMMA
        x = c[m] * np.exp(theta[m])
        lp[m], g[m] = 2.0 * theta[m] - x, 2.0 - x
        m = code == REL
        lp[m], g[m] = -2.0 * theta[m] ** 2, -4.0 * theta[m]
        m = code == BETA12
        s = special.expit(theta[m])
        lp[m], g[m] = special.log_expit(theta[m]) + 2.0 * special.log_expit(-theta[m]), 1.0 - 3.0 * s
        m = code == CPC
        t = theta[m]
        lp[m] = -2.0 * c[m] * (np.abs(t) + np.log1p(np.exp(-2.0 * np.abs(t))) - np.log(2.0))
        g[m] = -2.0 * c[m] * np.tanh(t)
        return float(lp.sum()), g


# ---------------------------------------------------------------------------------------------------
# Fitting and drawing

def fit_theta(prob: Problem, *, max_iter: int = 600, gtol: float = 1e-5, max_step: float = 1.0) -> dict:
    """The outer step: theta-hat by a projected BFGS on minus the penalised Laplace objective.

    Each step is capped at `max_step` on the unconstrained scale (a factor e on an sd), and the line
    search halves it on any trial point the inner step cannot take (a factorisation that fails or a
    value that is not finite), so the search never jumps to the far end of a bound. A hyperparameter at
    a bound whose gradient points out of the box is held there. `polish` finishes with Newton steps."""
    P = prob.params
    lo, hi = P.lo, P.hi
    state = {"start": None, "calls": 0, "failed": 0}

    def fun(theta):
        state["calls"] += 1
        try:
            res = prob.evaluate(theta, state["start"])
        except np.linalg.LinAlgError:
            state["failed"] += 1
            return None
        if not (np.isfinite(res["value"]) and np.all(np.isfinite(res["grad"]))):
            state["failed"] += 1
            return None
        return res

    if P.size == 0:
        res = prob.evaluate(np.zeros(0))
        return {"theta": np.zeros(0), "value": res["value"], "ok": True, "message": "no hyperparameters",
                "iterations": 0, "calls": 1, "failed_points": 0, "grad_max": 0.0, "at_bound": [],
                "start": res["mode"]}
    x = np.clip(P.init, lo, hi)
    res = fun(x)
    if res is None:
        raise RuntimeError("engine_laplace: the model cannot be evaluated at its starting point")
    state["start"] = res["mode"]
    f, g = -res["value"], -res["grad"]
    n = len(x)
    eye = np.eye(n)
    Hinv = eye / max(1.0, float(np.abs(g).max()))
    updates, stalled, message, it = 0, 0, "iteration limit", 0
    for it in range(1, max_iter + 1):
        held = ((x <= lo + 1e-12) & (g > 0)) | ((x >= hi - 1e-12) & (g < 0))
        idx = np.flatnonzero(~held)
        if not len(idx) or np.abs(g[idx]).max() < gtol:
            message = "projected gradient below tolerance"
            break
        d = np.zeros(n)
        d[idx] = -Hinv[np.ix_(idx, idx)] @ g[idx]
        if g[idx] @ d[idx] >= 0.0:   # not a descent direction: start the curvature over
            Hinv, updates = eye / max(1.0, float(np.abs(g).max())), 0
            d[idx] = -Hinv[np.ix_(idx, idx)] @ g[idx]
        longest = float(np.abs(d).max())
        if longest > max_step:
            d *= max_step / longest
        t, new = 1.0, None
        for _ in range(40):
            trial = np.clip(x + t * d, lo, hi)
            if np.array_equal(trial, x):
                break
            new = fun(trial)
            if new is not None and -new["value"] <= f + 1e-4 * float(g @ (trial - x)):
                break
            new = None
            t *= 0.5
        if new is None:
            message = "line search made no progress"
            break
        g_new = -new["grad"]
        step, y = trial - x, g_new - g
        sy = float(step @ y)
        if sy > 1e-10 * np.linalg.norm(step) * np.linalg.norm(y):
            if updates == 0:
                Hinv = eye * (sy / float(y @ y))
            r = 1.0 / sy
            V = eye - r * np.outer(step, y)
            Hinv = V @ Hinv @ V.T + r * np.outer(step, step)
            updates += 1
        gain = f + new["value"]
        x, f, g, res = trial, -new["value"], g_new, new
        state["start"] = res["mode"]
        stalled = stalled + 1 if gain <= 1e-12 * max(1.0, abs(f)) else 0
        if stalled >= 3:
            message = "objective stalled"
            break
    grad = res["grad"]
    free = ~(((x <= lo + 1e-9) & (grad < 0)) | ((x >= hi - 1e-9) & (grad > 0)))
    grad_max = float(np.abs(grad[free]).max(initial=0.0))
    edge = (x <= lo + 1e-9) | (x >= hi - 1e-9)
    return {"theta": x, "value": float(res["value"]), "ok": bool(grad_max < 1e-3 and res["mode"]["ok"]),
            "message": message, "iterations": it, "calls": state["calls"], "failed_points": state["failed"],
            "grad_max": grad_max, "at_bound": [P.labels[k] for k in np.flatnonzero(edge)],
            "start": res["mode"]}


def theta_hessian(prob: Problem, theta: np.ndarray, start=None, step: float = 1e-4) -> np.ndarray:
    """Minus the Hessian of the penalised objective at theta, by forward differences of the exact gradient
    (backward next to an upper bound): n + 1 evaluations. The error is about `step` times the third
    derivative, far below what the Hessian is used for (the theta covariance and Newton steps)."""
    n = len(theta)
    H = np.zeros((n, n))
    g0 = prob.evaluate(theta, start)["grad"]
    for k in range(n):
        e = np.zeros(n)
        e[k] = step if theta[k] + step <= prob.params.hi[k] else -step
        H[:, k] = -(prob.evaluate(theta + e, start)["grad"] - g0) / e[k]
    return 0.5 * (H + H.T)


def polish(prob: Problem, theta: np.ndarray, H: np.ndarray, start=None, *, rounds: int = 5,
           tol: float = 1e-6) -> dict:
    """Newton steps on theta with a fixed Hessian H (minus the Hessian of the objective), after the BFGS of `fit_theta`.

    A hyperparameter at a bound with its gradient pointing out stays there. Stops when the Newton
    decrement `g' H^-1 g / 2` (the gain a full step would bring, in log likelihood units) is below `tol`.
    Returns theta, the last evaluation, the decrement and the largest free gradient."""
    P = prob.params
    res = prob.evaluate(theta, start)
    decrement = np.inf
    for _ in range(rounds + 1):
        g = res["grad"]
        free = ~(((theta <= P.lo + 1e-9) & (g < 0)) | ((theta >= P.hi - 1e-9) & (g > 0)))
        idx = np.flatnonzero(free)
        step = np.zeros_like(theta)
        if len(idx):
            sub = H[np.ix_(idx, idx)]
            try:
                step[idx] = np.linalg.solve(sub, g[idx])
            except np.linalg.LinAlgError:
                step[idx] = np.linalg.lstsq(sub, g[idx], rcond=None)[0]
        decrement = 0.5 * float(g[idx] @ step[idx]) if len(idx) else 0.0
        if not np.isfinite(decrement) or decrement < tol or _ == rounds:
            break
        size = 1.0
        for _try in range(30):
            trial = np.clip(theta + size * step, P.lo, P.hi)
            new = prob.evaluate(trial, res["mode"])
            if np.isfinite(new["value"]) and new["value"] >= res["value"] - 1e-9:
                break
            size *= 0.5
        else:
            break
        theta, res = trial, new
    g = res["grad"]
    free = ~(((theta <= P.lo + 1e-9) & (g < 0)) | ((theta >= P.hi - 1e-9) & (g > 0)))
    return {"theta": theta, "res": res, "decrement": float(decrement),
            "grad_max": float(np.abs(g[free]).max(initial=0.0))}


def _cov_from(H: np.ndarray) -> tuple[np.ndarray, bool]:
    """The covariance of theta from minus the Hessian; eigenvalues below a floor are raised to it."""
    if H.size == 0:
        return H, True
    vals, vecs = np.linalg.eigh(H)
    ok = bool(vals.min() > 0)
    floor = max(1e-8 * float(np.abs(vals).max(initial=1.0)), 1e-10)
    vals = np.maximum(vals, floor)
    return (vecs / vals) @ vecs.T, ok


def boundary(prob: Problem, theta: np.ndarray, start=None, drop: float = 1.92) -> tuple[list[str], list[str]]:
    """The sds the data cannot tell from 0: the Laplace log likelihood (without the penalty) falls by
    less than `drop` (half the 95% point of chi-square with 1 degree of freedom) when the sd is set to
    5e-5 times its prior scale and the rest is held at theta-hat. Holding the rest makes this list
    short rather than long: an sd it names is at the boundary. An sd already that small is named.

    Where the inner step cannot take the trial point (a call noise that small makes the Hessian too
    ill-conditioned to factor), the trial is 1e-3 times the prior scale instead; an sd neither trial can
    take is returned in the second list, unchecked. Returns `(at the boundary, unchecked)`."""
    P = prob.params
    base = prob.evaluate(theta, start, grad=False, tol=VALUE_TOL)["laplace"]
    out, unchecked = [], []
    for k in np.flatnonzero(P.code == HN):
        near_zero = P.lo[k] + 0.2            # about 5e-5 of the prior scale
        if theta[k] <= near_zero:
            out.append(P.labels[k])
            continue
        value = None
        for level in (near_zero, P.lo[k] + 3.1):   # then about 1e-3 of the prior scale
            if level >= theta[k]:
                continue
            trial = theta.copy()
            trial[k] = level
            try:
                value = prob.evaluate(trial, start, grad=False, tol=VALUE_TOL)["laplace"]
                break
            except np.linalg.LinAlgError:
                value = None
        if value is None or not np.isfinite(value):
            unchecked.append(P.labels[k])
        elif base - value < drop:
            out.append(P.labels[k])
    return out, unchecked


def outputs(prob: Problem, nu: dict, zg: np.ndarray, zb: np.ndarray) -> dict:
    """The parameters of the fit file for draws z (`zg [n, G]`, `zb [n, I, B]`) at one theta."""
    A, D, J, N, I, R = prob.A, prob.D, prob.J, prob.N, prob.I, prob.R
    n = zg.shape[0]
    fams = prob.lat.families
    s = prob.scale

    def glob(name: str) -> np.ndarray:
        return zg[:, fams[name]["g"]]

    def loc(name: str) -> np.ndarray:
        f = fams[name]
        out = np.where(f["local"][None], zb[:, f["i"], f["p"]], zg[:, f["g"]])
        return out

    zmu = glob("mu").reshape(n, A, D)
    mu = prob.mu_loc + 2.0 * s * zmu
    if prob.pairs_only.any():
        centred = mu - mu.mean(axis=1, keepdims=True)
        mu = np.where(prob.pairs_only[None, None, :], centred, mu)
    zb_ = glob("bias").reshape(n, J, D)
    bias = s * (zb_ - (zb_ * prob.bias_w[None]).sum(axis=1, keepdims=True))
    lin = nu["sd_in"][:, None] * nu["Lc_in"]
    lrun = nu["sd_run"][:, None] * nu["Lc_run"]
    v = loc("v").reshape(n, I, D) @ lin.T
    u = loc("u").reshape(n, R, D) @ lrun.T
    q = mu[:, prob.run_arm] + v[:, prob.run_input] + u
    family = np.zeros((n, J, N, D))
    if prob.fam_keys:
        raw = glob("family").reshape(n, len(prob.fam_keys), N)
        raw = raw - raw.mean(axis=2, keepdims=True)
        for k, (j, d) in enumerate(prob.fam_keys):
            family[:, j, :, d] = raw[:, k] * nu["famsd"][d]
    position = prob.pair_scale[None] * glob("position")

    def rep(x):
        return np.broadcast_to(x, (n,) + np.shape(x)).copy()

    return {"mu": mu, "sd_input": rep(nu["sd_in"]), "sd_run": rep(nu["sd_run"]),
            "corr_input": rep(nu["Lc_in"] @ nu["Lc_in"].T), "corr_run": rep(nu["Lc_run"] @ nu["Lc_run"].T),
            "q_run": q, "bias": bias, "tau": rep(nu["tau"]), "sigma": rep(nu["sigma"]), "halo_target": rep(nu["h"]),
            "halo_call": rep(nu["hc"]), "family": family, "sd_judge_input": rep(nu["sdji"]),
            "sigma_pair": rep(nu["sp"]), "position": position, "tie_width": rep(nu["tie"])}


CELL_RULE_MAX = 1e-4   # log units, summed over the groups integrated by the cell rule: far below what moves a fit


def cell_rule_check(prob: "Problem", theta: np.ndarray, start: Any) -> dict | None:
    """At theta-hat, the error of the cell rule: every group integrated by `cell_integral` again with
    twice the panels, at the final cavities; the check sums |log Z(PANELS) - log Z(2 PANELS)| and the
    tail bounds over the groups. None when no group needs the rule."""
    if prob.inner != "ep" or not isinstance(start, dict) or "cavity" not in start or prob.g_simple.all():
        return None
    c = prob.coefficients(prob.params.unpack(theta))
    muc, sc2 = start["cavity"]
    base, fine = prob.tilted(c, muc, sc2), prob.tilted(c, muc, sc2, panels=2 * PANELS)
    many = ~prob.g_simple
    diff = np.abs(base["logZ"] - fine["logZ"])[many]
    value = float(diff.sum() + base["tail"][many].sum())
    return {"name": "cell_rule", "value": value, "limit": CELL_RULE_MAX, "ok": bool(value <= CELL_RULE_MAX),
            "groups": int(many.sum()), "max_group": float(diff.max(initial=0.0))}


def sample(data: Mapping[str, Any], *, seed: int = 0, draws: int = 250, tune: int | None = None, chains: int = 4,
           cores: int | None = None, target_accept: float | None = None, theta_points: int = 250,
           find_boundary: bool = True, inner: str | None = None) -> dict:
    """Fit the model for the arrays of `model.Design.data` and draw from the Laplace approximation.

    `inner` picks how the unknowns below theta are integrated: "ep" (expectation propagation, one site
    per group of cut readings that share a predictor, each integrated exactly; the default when there
    are cut readings) or "laplace" (the Laplace approximation at the mode; exact when every reading is
    Gaussian, and the default then).

    Returns what `engine_pymc.sample` returns, `{"draws": {name: array[chains, draws, ...]},
    "divergences", "run_seconds", "version"}`, plus `approximate` (true), `checks` (the optimiser's
    convergence, the inner steps and the Hessian of theta, each `{"name", "value", "limit", "ok"}`),
    `boundary` (labels of sds the data cannot tell from 0; `boundary_unchecked` those the scan could not
    test) and `theta` (labels, estimates and sds on the
    unconstrained scale). `divergences` is 0: there is no sampler. `tune`, `cores` and `target_accept`
    are accepted for the same call shape and not used."""
    del tune, cores, target_accept
    started = time.time()
    rng = np.random.default_rng(seed)
    prob = Problem(data)
    if inner is not None:
        if inner not in ("ep", "laplace"):
            raise ValueError(f"inner is 'ep' or 'laplace', not {inner!r}")
        prob.inner = inner if not prob.all_gauss else "laplace"
    P = prob.params
    fitted = fit_theta(prob)
    theta = fitted["theta"]
    start = fitted.get("start")
    decrement, grad_max = 0.0, fitted["grad_max"]
    if P.size:
        hess = theta_hessian(prob, theta, start)
        polished = polish(prob, theta, hess, start)
        if np.max(np.abs(polished["theta"] - theta), initial=0.0) > 1e-3:   # moved: the Hessian moves too
            theta = polished["theta"]
            hess = theta_hessian(prob, theta, start)
            polished = polish(prob, theta, hess, start)
        theta, decrement, grad_max = polished["theta"], polished["decrement"], polished["grad_max"]
        start = polished["res"]["mode"]
        fitted["value"] = polished["res"]["value"]
        inner_fit = bool(polished["res"]["mode"]["ok"])
    else:
        hess = np.zeros((0, 0))
        inner_fit = True
    cov, hess_ok = _cov_from(hess)
    edge, unchecked = boundary(prob, theta, start) if find_boundary and P.size else ([], [])
    total = int(draws) * int(chains)
    M = max(1, min(int(theta_points), total))
    chol = np.linalg.cholesky(cov) if P.size else np.zeros((0, 0))
    out = {name: None for name in OUTPUTS}
    which = rng.permutation(np.arange(total) % M)   # theta points spread over chains and positions at random
    inner_ok, redrawn = True, 0
    for m in range(M):
        picks = np.flatnonzero(which == m)
        if not len(picks):
            continue
        res = None
        for attempt in range(6):   # a theta draw the inner step cannot take is drawn again; theta-hat last
            t = theta if m == 0 or attempt == 5 else np.clip(theta + chol @ rng.standard_normal(len(theta)), P.lo,
                                                              P.hi)
            try:
                res = prob.evaluate(t, start, grad=False, tol=DRAW_TOL)
            except np.linalg.LinAlgError:
                res = None
            if res is not None and np.isfinite(res["value"]):
                break
            redrawn += 1
        inner_ok &= bool(res["mode"]["ok"])
        dg, db = res["mode"]["factor"].draw(rng, len(picks))
        values = outputs(prob, res["nu"], res["mode"]["zg"] + dg, res["mode"]["zb"] + db)
        for name, x in values.items():
            if out[name] is None:
                out[name] = np.zeros((total,) + x.shape[1:])
            out[name][picks] = x
    result = {name: np.asarray(x).reshape((int(chains), int(draws)) + np.shape(x)[1:]) for name, x in out.items()}
    if not prob.has_pairs:
        for name in ("sigma_pair", "position", "tie_width"):
            result[name] = np.full(result[name].shape, np.nan)
    checks = [{"name": "optimizer", "value": decrement, "limit": DECREMENT_MAX,
               "ok": bool(decrement <= DECREMENT_MAX and inner_fit), "grad_max": grad_max,
               "message": fitted["message"], "iterations": fitted["iterations"]},
              {"name": "inner_newton", "value": int(not inner_ok), "limit": 0, "ok": bool(inner_ok),
               "redrawn_theta": int(redrawn)},
              {"name": "theta_hessian", "value": float(np.linalg.eigvalsh(hess).min()) if P.size else 0.0,
               "limit": 0.0, "ok": bool(hess_ok)}]
    checks[1]["inner"] = prob.inner
    rule = cell_rule_check(prob, theta, start)
    if rule is not None:
        checks.append(rule)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None)) if P.size else np.zeros(0)
    return {"draws": result, "divergences": 0, "run_seconds": float(time.time() - started), "version": version(),
            "approximate": True, "checks": checks, "boundary": edge, "boundary_unchecked": unchecked,
            "theta": {"labels": list(P.labels), "estimate": theta.tolist(), "sd": sd.tolist()},
            "objective": float(fitted["value"]), "theta_points": M}
