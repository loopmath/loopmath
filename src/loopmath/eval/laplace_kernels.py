"""Likelihood kernels and one-dimensional cell integrals for the Laplace engine (numpy and scipy only).

Not a command: the Laplace engine (`engine_laplace.py`) calls `kernel`, `kernel_full` and `cell_integral`.

Kernels
-------
`kernel(kind, eta, obs, params) -> (logp, d1, d2)` gives, per observation, the log-likelihood and its first
and second derivatives in the linear predictor `eta`. `kernel_full` gives the same plus the third derivative
`d3`, for every parameter of the kernel the derivatives of `logp`, `d1` and `d2` in that parameter (what the
gradient of a Laplace marginal likelihood needs: the envelope term, the shift of the mode and the change of
the Hessian), and `logp_err`, an allowance for the floating-point error of `logp`. All arrays broadcast
against `eta`; the result has the broadcast shape. Every `logp` is a normalised log probability (a log
density for the Gaussian kinds), constants included.

`kind` is a name below, or the integer codes of `model.KIND_CODE` (0 linear, 1 binary, 2 levels, 3 items).

    kind        obs                       params               the reading, given eta
    linear      y                         sd                   y ~ N(eta, sd^2)            ("fraction" is the same)
    linear_cell n, ybar, ss               sd, tau              n calls y_c = eta + t + e_c, t ~ N(0, tau^2),
                                                               e_c ~ N(0, sd^2), entered through their mean
                                                               ybar and sum of squares ss around it (exact)
    binary      y (0 or 1)                sd                   1 with chance Phi(eta / sd)
    levels      lo, hi                    sd                   the level whose interval (lo, hi) holds
                                                               eta + e, e ~ N(0, sd^2); open ends below
    items       k, n                      sd                   k of n items pass. n = 1: Phi(x) with
                                                               x = eta / sqrt(1 + sd^2), the call noise sd
                                                               summed out (exact). n > 1: the n items of a
                                                               call share its noise, so the count is not
                                                               binomial at the mean chance; sd must be 0 and
                                                               the noise an explicit unknown (in eta, or the
                                                               cell term of `cell_integral`); k ~ Binomial(n,
                                                               Phi(eta)). sd > 0 with n > 1 raises ValueError
    pairs       v (0 a, 1 tie, 2 b), o    sd, width, position  z = eta + position o + e, e ~ N(0, sd^2); verdict
                (+1 a shown first, -1 b,                       a when z > width, b when z < -width, else tie.
                0 unknown)                                     eta is eta_a - eta_b: add d1 to a, subtract
                                                               from b; d2 enters (a,a), (b,b) and -d2 (a,b)

Open ends of levels. A lower cut of -inf, or at or below -OPEN_FROM (1e5), and an upper cut of +inf, or at or
above +OPEN_FROM, are open: the interval has no wall there, in the likelihood and in the cell integrator
alike (`model.OPEN` = 1e6 is the model's sentinel). Every other cut is a finite cut, whatever its size.

`KernelResult.dparam[name] = (dlogp, dd1, dd2)` with derivatives in the parameter as written (sd, not log sd
or the variance). A kernel parameter that is a function of the hyperparameters (for example a reading's sd
made of sigma, tau and the halo share) is the engine's to chain.

Numerics. Probit terms use `log Phi` (scipy's `log_ndtr`) and the ratio `lambda = phi / Phi` through `erfcx`,
never `log(ndtr)`. In the far lower tail (x < -10) the second and third derivatives cancel catastrophically in
their textbook forms (`-lambda (x + lambda)` loses all digits by x = -1e8), so there they come from the
asymptotic series of the Mills ratio, summed to 30 terms (truncation below 1e-19 at x = -10). An interval
probability is `Phi(alpha) - Phi(beta)` taken on the side of 0 where it keeps its digits, as
`log Phi(alpha) + log(1 - exp(log Phi(beta) - log Phi(alpha)))`. An interval that is narrow on the scale of
the normal density there (half-width times max(1, |centre|) below `NARROW`, in noise sds) comes instead from
the series `P = 2 h phi(c) sum_m He_2m(c) h^2m / (2m + 1)!` in its centre c and half-width h, with the
derivatives in the centre and in the half-width taken from the series directly: the two ends' derivatives
are large and nearly cancel there, and their difference (the tie width derivative) would lose every digit.
An items count of 0 < k < n uses the saddle-point form of the binomial (Stirling remainders and the deviance
terms `x log(x / M) + M - x`), not `log C(n, k) + k log p + (n - k) log q`, whose terms are of order n and
cancel to order log n (the coefficient alone was off by 2e-7 at n = 1e8 with `gammaln`).

Cell integrals
--------------
`cell_integral(kind, m, obs, params, cell, tau, ...)` integrates a one-dimensional latent term per cell: the
readings i of cell c have `eta_i = m_i + tau_c z`, z ~ N(0, 1) shared by the cell (the judge-by-output term of
repeated calls on one output, or the call noise of one `items` call). It returns, per cell,
`logl = log integral phi(z) prod_i p(obs_i | m_i + tau_c z) dz` with an error estimate, and the derivatives the
inner Newton step and the outer optimisation need, each with its own error estimate (see `CellResult`).

Three methods:

- `agh`: adaptive Gauss-Hermite, centred at the mode of the integrand with the curvature scale, `nodes`
  points. The error estimate is the difference to the same rule with about half the points (`low_nodes`).
  That difference is a heuristic, not a bound (`err_kind` "heuristic"): both rules can miss a feature
  together. It fails when the call noise is small against the term: the integrand is then a normal density
  cut by near-hard walls, and a Gaussian-weight rule converges slowly (two top readings of a three-level
  scale with call noise a tenth of the term are off by 1.9 percent at 15 points).
- `composite`: Gauss-Kronrod (7, 15) panels over the interval where the log integrand is within `DROP` of its
  maximum, split at the mode, at the points where it has dropped by `SPLIT_DROPS`, and at every wall of the
  readings (its location in z and a few wall widths either side). The estimate (`err_kind` "kronrod") is the
  sum over panels of |K15 - G7|, plus a bound on the two tails outside the interval, which holds because the
  integrand is log-concave for every kernel here. It is not a proof either: a feature that no panel boundary
  marks could be missed by both rules; every feature the kernels have is marked.
- `auto` (default): `agh` for every cell, then `composite` for every cell whose `agh` estimate is above `tol`,
  whose derivative estimates miss `dtol` (`deriv_ok`), or that has a wall narrower than `AGH_SMOOTH` curvature
  scales anywhere the integrand is not negligible: there both `agh` rules can miss the wall together (a wall
  five latent sds out left a 5e-7 error under a 6e-11 estimate). Walls are cut points, verdict edges, the
  sign change of a binary reading, and for an items count the two places where the powered terms
  `k log Phi(eta)` and `(n - k) log Phi(-eta)` turn from steep to flat (at eta = +-a, a = -Phi^-1(1 / (count +
  1)), width 1 / max(a, 1)): for k = n = 1e5 that is eta = 4.26, not 0. Cells that `composite` leaves outside
  `tol` or `dtol` by quadrature (not rounding) error are redone with every panel halved, up to `refine` times.

Every error estimate has a quadrature part (the rule difference above) and a fixed part that finer panels do
not shrink: floating-point error (the kernels' `logp_err` and the rounding of eta = m + tau z and of the nodes,
carried through the posterior weights, and the rounding of each sum; posterior means divide by the weights'
own sum, which is 1 only to the rounding of logl), and the part of the integrand beyond the two drop-`DROP`
ends, bounded by the exponential envelope that log-concavity gives there, with bounds on how fast the kernel
derivatives can grow into it (|d2| <= 1/sd^2 for every probit kind, the variance of a truncated normal being
below 1). Refinement stops when the fixed part alone misses `tol` or `dtol`.

Derivatives. For a common shift s of a cell's m (every call of a judge-by-output cell has the same m) and for
tau > 0, integration by parts gives exact identities in the posterior moments of z:
`shift_d1 = E[z] / tau`, `shift_d2 = (Var z - 1) / tau^2`, `shift_d3 = kappa3(z) / tau^3`,
`dtau = (E[z^2] - 1) / tau`. They avoid the kernel derivatives, which on a sharp wall are huge and cancel in
the moment forms (`E[L''] + Var L'`); the moment forms are accurate where the identities cancel instead (tau
small against the noise). Both are computed with error estimates and the smaller estimate wins, per cell and
per quantity. A cell of one reading uses the shift quantities for that reading's d1 and Hessian. Per-reading
d1 and Hessian entries of cells with several readings come from the moment forms. Parameter derivatives use
pointwise identities where there is one: the heat identity `d p / d sd = sd d2 p / d eta2` gives
`d logl / d sd_i = sd_i (H_ii + d1_i^2)` (every kind but `linear_cell`), `d logl / d position_i = o_i d1_i`,
and the tie width of an a or b verdict moves its one edge as eta does (`-d1`, `d1`); the width of a tie and
the parameters of `linear_cell` are posterior means. `deriv_ok` says whether every derivative of the cell met
`dtol` (see `CellResult`); False means its accuracy could not be established, not that it is wrong.

`rule=` re-evaluates at other arguments with the nodes of an earlier result held fixed (frozen). Its
derivatives are the exact derivatives of the frozen sum (moment forms), which is what finite differences of
frozen sums need; its value and derivatives carry no error estimate (`err` NaN, `err_kind` "none",
`method` "frozen", `deriv_ok` False), because the nodes were placed for other arguments.

Pairs. A pair reads the terms of two cells (the judge's view of output a and of output b). When the repeated
verdicts of one judge on one pair of outputs are the only readings that touch those two terms, only their
difference enters, a normal with sd sqrt(tau_a^2 + tau_b^2): pass kind "pairs", m = eta_a - eta_b without the
two terms, and that sd as tau. A judge whose pairs share outputs couples the terms into a larger block, which
is not a one-dimensional integral and is not handled here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy import special

__all__ = ["KINDS", "KernelResult", "CellResult", "kernel", "kernel_full", "cell_integral", "log_ndtr_derivs",
           "hermite_rule", "kronrod_rule"]

KINDS = ("linear", "fraction", "linear_cell", "binary", "levels", "items", "pairs")
CODE_KIND = {0: "linear", 1: "binary", 2: "levels", 3: "items"}   # model.KIND_CODE
PARAMS = {"linear": ("sd",), "fraction": ("sd",), "linear_cell": ("sd", "tau"), "binary": ("sd",),
          "levels": ("sd",), "items": ("sd",), "pairs": ("sd", "width", "position")}
OBS = {"linear": ("y",), "fraction": ("y",), "linear_cell": ("n", "ybar", "ss"), "binary": ("y",),
       "levels": ("lo", "hi"), "items": ("k", "n"), "pairs": ("v", "o")}
HEAT_KINDS = ("linear", "fraction", "binary", "levels", "items", "pairs")   # d p / d sd = sd d2 p / d eta2

LOG_2PI = float(np.log(2.0 * np.pi))
SQRT2 = float(np.sqrt(2.0))
SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
EPS = float(np.finfo(float).eps)
OPEN_FROM = 1.0e5        # a lower cut at or below -OPEN_FROM or an upper cut at or above it is open (model.OPEN 1e6)
TAIL = 10.0              # below -TAIL the derivatives of log Phi come from the asymptotic series
NARROW = 0.25            # an interval with half-width * max(1, |centre|) below this (noise sds) uses the series
NARROW_TERMS = 12
DROP = 30.0              # composite: the integration interval is where log f >= max - DROP
SPLIT_DROPS = (2.0, 8.0, 18.0)
WALL_OFFSETS = (1.5, 4.0, 7.0)  # composite: breakpoints at a wall and this many wall widths either side
AGH_SMOOTH = 3.0         # auto: AGH only when every wall in range is at least this many curvature scales wide


# ---- log Phi and its derivatives ---------------------------------------------------------------------

def _series_coefficients(terms: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact integer coefficients of the asymptotic series of the Mills ratio, in u = 1/x^2.

    R(t) = (1 - Phi(t)) / phi(t) = (1/t) S(u), S = sum_k c_k u^k, c_k = (-1)^k (2k - 1)!!.
    Returns, low order first, the coefficients of A = 1 - S, B = t^2 A - 1, 2A + B and 3A + B."""
    c = [1]
    df = 1
    for k in range(1, terms + 2):
        df *= 2 * k - 1
        c.append((-1) ** k * df)
    a = [0] + [-c[k] for k in range(1, terms + 1)]
    b = [0] + [-c[m + 1] for m in range(1, terms + 1)]
    two_a_b = [0] + [-(2 * c[m] + c[m + 1]) for m in range(1, terms + 1)]
    three_a_b = [0] + [-(3 * c[m] + c[m + 1]) for m in range(1, terms + 1)]   # the u term is exactly 0
    return tuple(np.array([float(x) for x in seq]) for seq in (a, b, two_a_b, three_a_b))  # type: ignore


_SA, _SB, _S2AB, _S3AB = _series_coefficients()


def log_ndtr_derivs(x: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`log Phi(x)` and its first three derivatives, elementwise, stable for any real x (and +-inf)."""
    x = np.asarray(x, dtype=float)
    l0 = special.log_ndtr(x)
    l1 = np.zeros_like(x)
    l2 = np.zeros_like(x)
    l3 = np.zeros_like(x)
    direct = np.isfinite(x) & (x >= -TAIL)
    if direct.any():
        xd = x[direct]
        lam = SQRT_2_OVER_PI / special.erfcx(-xd / SQRT2)
        g = xd + lam
        d2 = -lam * g
        l1[direct] = lam
        l2[direct] = d2
        with np.errstate(over="ignore", invalid="ignore"):
            l3[direct] = np.where(lam > 0.0, lam * (g * g - (1.0 + d2)), 0.0)
    tail = np.isfinite(x) & (x < -TAIL)
    if tail.any():
        t = -x[tail]
        u = 1.0 / (t * t)
        A = np.polynomial.polynomial.polyval(u, _SA)
        B = np.polynomial.polynomial.polyval(u, _SB)
        ab3 = np.polynomial.polynomial.polyval(u, _S3AB)
        S = 1.0 - A
        l1[tail] = t / S
        l2[tail] = -(1.0 + B) / (S * S)
        l3[tail] = t * (ab3 + A * (B - A)) / (S * S * S)
    low = x == -np.inf
    if low.any():
        l1[low] = np.inf
        l2[low] = -1.0
        l3[low] = 0.0
    return l0, l1, l2, l3


def _log1mexp(d: np.ndarray) -> np.ndarray:
    """log(1 - exp(d)) for d <= 0, accurate at both ends."""
    out = np.empty_like(d)
    near = d > -np.log(2.0)
    out[near] = np.log(-np.expm1(d[near]))
    out[~near] = np.log1p(-np.exp(d[~near]))
    return out


def _interval_core(alpha: np.ndarray, beta: np.ndarray) -> dict:
    """G = log(Phi(alpha) - Phi(beta)), alpha > beta, with T^k G (k = 1..3, T = d/dalpha + d/dbeta) and the
    partials d/dalpha T^k G, d/dbeta T^k G (k = 0..2). Either end may be infinite (alpha = +inf, beta = -inf),
    not both."""
    flip = (alpha + beta) > 0.0
    a = np.where(flip, -beta, alpha)
    b = np.where(flip, -alpha, beta)
    a0, a1, a2, a3 = log_ndtr_derivs(a)
    fb = np.isfinite(b)
    b0 = np.full_like(b, -np.inf)
    b1 = np.zeros_like(b)
    b2 = np.zeros_like(b)
    b3 = np.zeros_like(b)
    if fb.any():
        r = log_ndtr_derivs(b[fb])
        b0[fb], b1[fb], b2[fb], b3[fb] = r
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        delta = b0 - a0
        F0 = _log1mexp(np.minimum(delta, 0.0))
        F1 = -1.0 / np.expm1(-delta)          # F'(delta) = -rho / (1 - rho), rho = exp(delta)
        one_m_F1 = -1.0 / np.expm1(delta)     # 1 - F' = 1 / (1 - rho)
        F2 = F1 * one_m_F1
        F3 = F2 * (1.0 - 2.0 * F1)
        D1, D2, D3 = b1 - a1, b2 - a2, b3 - a3
        G = a0 + F0
        T1 = a1 + F1 * D1
        T2 = a2 + F2 * D1 * D1 + F1 * D2
        T3 = a3 + F3 * D1 ** 3 + 3.0 * F2 * D1 * D2 + F1 * D3
        Pa = [a1 * one_m_F1,
              a2 * one_m_F1 - F2 * a1 * D1,
              a3 * one_m_F1 - F3 * a1 * D1 * D1 - 2.0 * F2 * a2 * D1 - F2 * a1 * D2]
        Pb = [F1 * b1,
              F2 * b1 * D1 + F1 * b2,
              F3 * b1 * D1 * D1 + 2.0 * F2 * D1 * b2 + F2 * b1 * D2 + F1 * b3]
    sign = np.where(flip, -1.0, 1.0)
    out = {"G": G, "T1": sign * T1, "T2": T2, "T3": sign * T3}
    for k in range(3):
        s = sign ** (k + 1)   # flipped: d/dalpha T^k G = (-1)^(k+1) Pb'_k, d/dbeta T^k G = (-1)^(k+1) Pa'_k
        out[f"Pa{k}"] = np.where(flip, s * Pb[k], Pa[k])
        out[f"Pb{k}"] = np.where(flip, s * Pa[k], Pb[k])
    return out


def _narrow_core(c: np.ndarray, h: np.ndarray) -> dict:
    """G = log(Phi(c + h) - Phi(c - h)) for h max(1, |c|) < NARROW, from P = 2 h phi(c) S,
    S = sum_m He_2m(c) h^2m / (2m + 1)!. Returns G, its derivatives in c (Gc1..Gc3) and the derivatives in h
    of G, d/dc G and d2/dc2 G (Gh0..Gh2), each from the series (no difference of the two ends)."""
    n_max = 2 * NARROW_TERMS + 1
    g = [np.ones_like(c), c * h]            # g_n = He_n(c) h^n, scaled so that nothing overflows
    for j in range(1, n_max):
        g.append(c * h * g[j] - j * h * h * g[j - 1])
    zero = np.zeros_like(c)
    S, S1, S2, S3, Sh, Sh1, Sh2 = (zero.copy() for _ in range(7))
    for m in range(NARROW_TERMS + 1):
        f = 1.0 / math.factorial(2 * m + 1)
        t = 2 * m
        S += g[t] * f
        if m >= 1:
            S1 += t * g[t - 1] * f
            S2 += t * (t - 1) * g[t - 2] * f
            Sh += t * g[t] * f
            Sh1 += t * t * g[t - 1] * f
            Sh2 += t * t * (t - 1) * g[t - 2] * f
        if m >= 2:
            S3 += t * (t - 1) * (t - 2) * g[t - 3] * f
    S1, S2, S3, Sh, Sh2 = S1 * h, S2 * h * h, S3 * h ** 3, Sh / h, Sh2 * h
    L1 = S1 / S
    L2 = S2 / S - L1 * L1
    L3 = S3 / S - 3.0 * L1 * S2 / S + 2.0 * L1 ** 3
    Lh = Sh / S
    Lh1 = Sh1 / S - Sh * S1 / (S * S)
    Lh2 = Sh2 / S - (2.0 * Sh1 * S1 + Sh * S2) / (S * S) + 2.0 * Sh * S1 * S1 / S ** 3
    return {"G": np.log(2.0 * h) - 0.5 * c * c - 0.5 * LOG_2PI + np.log(S),
            "Gc1": -c + L1, "Gc2": -1.0 + L2, "Gc3": L3, "Gh0": 1.0 / h + Lh, "Gh1": Lh1, "Gh2": Lh2}


# ---- kernels ---------------------------------------------------------------------------------------------

@dataclass
class KernelResult:
    """Per observation: `logp`, its derivatives `d1`, `d2`, `d3` in eta, `dparam[name] = (dlogp, dd1, dd2)`,
    and `logp_err`, an allowance for the absolute floating-point error of `logp`."""

    logp: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    d3: np.ndarray
    dparam: dict = field(default_factory=dict)
    logp_err: Any = None


def _kind_name(kind: Any) -> str:
    if isinstance(kind, (int, np.integer)) and int(kind) in CODE_KIND:
        return CODE_KIND[int(kind)]
    if isinstance(kind, str) and kind in KINDS:
        return kind
    raise ValueError(f"laplace_kernels: unknown kind {kind!r}; use one of {KINDS} or a model.KIND_CODE value")


def _arrays(kind: str, eta: Any, obs: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[np.ndarray, dict, dict]:
    missing = [k for k in OBS[kind] if k not in obs] + [k for k in PARAMS[kind] if k not in params]
    if missing:
        raise ValueError(f"laplace_kernels: kind {kind!r} needs {', '.join(missing)}")
    names_o, names_p = OBS[kind], PARAMS[kind]
    arrays = np.broadcast_arrays(np.asarray(eta, dtype=float), *[np.asarray(obs[k], dtype=float) for k in names_o],
                                 *[np.asarray(params[k], dtype=float) for k in names_p])
    eta_b = np.array(arrays[0], dtype=float)
    o = {k: np.array(v, dtype=float) for k, v in zip(names_o, arrays[1:1 + len(names_o)])}
    p = {k: np.array(v, dtype=float) for k, v in zip(names_p, arrays[1 + len(names_o):])}
    return eta_b, o, p


def _gauss(eta: np.ndarray, y: np.ndarray, sd: np.ndarray) -> KernelResult:
    r = y - eta
    s2 = sd * sd
    logp = -0.5 * LOG_2PI - np.log(sd) - 0.5 * r * r / s2
    d1 = r / s2
    d2 = -1.0 / s2
    dsd = (-1.0 / sd + r * r / (s2 * sd), -2.0 * r / (s2 * sd), 2.0 / (s2 * sd))
    err = 8.0 * EPS * (1.0 + np.abs(np.log(sd)) + 0.5 * r * r / s2 + np.abs(d1) * (np.abs(y) + np.abs(eta)))
    return KernelResult(logp, d1, np.broadcast_to(d2, eta.shape).copy(), np.zeros_like(eta), {"sd": dsd}, err)


def _gauss_cell(eta: np.ndarray, n: np.ndarray, ybar: np.ndarray, ss: np.ndarray, sd: np.ndarray,
                tau: np.ndarray) -> KernelResult:
    r = ybar - eta
    s2 = sd * sd
    V = s2 + n * tau * tau
    parts = (0.5 * n * LOG_2PI, (n - 1.0) * np.log(sd), 0.5 * np.log(V), 0.5 * ss / s2, 0.5 * n * r * r / V)
    logp = -parts[0] - parts[1] - parts[2] - parts[3] - parts[4]
    d1 = n * r / V
    d2 = -n / V
    V2 = V * V
    dsd = (-(n - 1.0) / sd - sd / V + ss / (s2 * sd) + n * r * r * sd / V2, -2.0 * n * r * sd / V2, 2.0 * n * sd / V2)
    dtau = (-n * tau / V + n * n * r * r * tau / V2, -2.0 * n * n * r * tau / V2, 2.0 * n * n * tau / V2)
    err = 8.0 * EPS * (1.0 + sum(np.abs(q) for q in parts) + np.abs(d1) * (np.abs(ybar) + np.abs(eta)))
    return KernelResult(logp, d1, d2, np.zeros_like(eta), {"sd": dsd, "tau": dtau}, err)


def _one_sided(x_sign: np.ndarray, eta: np.ndarray, scale: np.ndarray) -> tuple:
    """log Phi(sgn eta / scale) and derivatives in eta; derivatives in scale as (dlogp, dd1, dd2); logp_err."""
    x = x_sign * eta / scale
    l0, l1, l2, l3 = log_ndtr_derivs(x)
    d1 = x_sign * l1 / scale
    d2 = l2 / scale ** 2
    d3 = x_sign * l3 / scale ** 3
    dscale = (-x * l1 / scale, -x_sign * (x * l2 + l1) / scale ** 2, -(x * l3 + 2.0 * l2) / scale ** 3)
    err = 8.0 * EPS * (np.maximum(1.0, np.abs(l0)) + np.abs(x * l1))
    return l0, d1, d2, d3, dscale, err


def _interval(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, sd: np.ndarray) -> tuple:
    """log P(lo < x + e < hi), e ~ N(0, sd^2), lo < hi, either end possibly infinite. Returns the result (with
    dparam["sd"] and logp_err) and the derivatives, each (dlogp, dd1, dd2), in lo, in hi, and in the
    half-width at a fixed centre (moving lo down and hi up together)."""
    with np.errstate(invalid="ignore"):
        alpha = (x - lo) / sd
        beta = (x - hi) / sd
    core = _interval_core(alpha, beta)
    G = core["G"]
    T = [core["T1"], core["T2"], core["T3"]]
    Pa = [core[f"Pa{k}"] for k in range(3)]
    Pb = [core[f"Pb{k}"] for k in range(3)]
    fa, fb = np.isfinite(alpha), np.isfinite(beta)
    a_safe, b_safe = np.where(fa, alpha, 0.0), np.where(fb, beta, 0.0)
    with np.errstate(invalid="ignore"):
        spread = [Pa[k] - Pb[k] for k in range(3)]
        moved = [a_safe * Pa[k] + b_safe * Pb[k] for k in range(3)]
        gerr = 8.0 * EPS * (np.maximum(1.0, np.abs(G)) + np.abs(a_safe * Pa[0]) + np.abs(b_safe * Pb[0]) + 1.0)
        centre = (x - 0.5 * (lo + hi)) / sd
        half = 0.5 * (hi - lo) / sd
        narrow = fa & fb & (half > 0.0) & (half * np.maximum(1.0, np.abs(centre)) < NARROW)
    if narrow.any():
        c, h = centre[narrow], half[narrow]
        s = _narrow_core(c, h)
        G[narrow] = s["G"]
        for k in range(3):
            T[k][narrow] = s[f"Gc{k + 1}"]
            Pa[k][narrow] = 0.5 * (s[f"Gc{k + 1}"] + s[f"Gh{k}"])
            Pb[k][narrow] = 0.5 * (s[f"Gc{k + 1}"] - s[f"Gh{k}"])
            spread[k][narrow] = s[f"Gh{k}"]
            moved[k][narrow] = c * s[f"Gc{k + 1}"] + h * s[f"Gh{k}"]
        gerr[narrow] = 8.0 * EPS * (np.maximum(1.0, np.abs(s["G"])) + np.abs(c * s["Gc1"]) + np.abs(h * s["Gh0"])
                                    + 1.0)
    d1, d2, d3 = T[0] / sd, T[1] / sd ** 2, T[2] / sd ** 3
    dlo = tuple(-Pa[k] / sd ** (k + 1) for k in range(3))
    dhi = tuple(-Pb[k] / sd ** (k + 1) for k in range(3))
    dhalf = tuple(spread[k] / sd ** (k + 1) for k in range(3))
    dsd = tuple((-k * dk / sd if k else 0.0) - moved[k] / sd ** (k + 1) for k, dk in enumerate((G, d1, d2)))
    err = gerr + 4.0 * EPS * np.abs(x * d1)
    return KernelResult(G, d1, d2, d3, {"sd": dsd}, err), dlo, dhi, dhalf


def _stirlerr(n: np.ndarray) -> np.ndarray:
    """log(n!) - ((n + 1/2) log n - n + log(2 pi) / 2), n >= 1."""
    out = np.empty_like(n)
    big = n >= 15.0
    nb = n[big]
    u = 1.0 / (nb * nb)
    out[big] = (1 / 12 - u * (1 / 360 - u * (1 / 1260 - u * (1 / 1680 - u * (1 / 1188 - u * 691 / 360360))))) / nb
    ns = n[~big]
    out[~big] = special.gammaln(ns + 1.0) - (ns + 0.5) * np.log(ns) + ns - 0.5 * LOG_2PI
    return out


def _bd0(x: np.ndarray, d: np.ndarray, logratio: np.ndarray) -> np.ndarray:
    """x log(x / M) + M - x >= 0 for M = x - d, given d = x - M and logratio = log(x / M); a series in
    v = d / (x + M) where d is small, so that nothing cancels."""
    s = 2.0 * x - d
    out = x * logratio - d
    near = np.abs(d) < 0.1 * s
    if near.any():
        v = d[near] / s[near]
        v2 = v * v
        term = v.copy()
        acc = np.zeros_like(v)
        for j in range(1, 11):
            term = term * v2
            acc += term / (2 * j + 1)
        out[near] = d[near] * v + 2.0 * x[near] * acc
    return out


def _items_logp(k: np.ndarray, n: np.ndarray, x: np.ndarray, u0: np.ndarray,
                v0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log of C(n, k) p^k q^(n - k), p = Phi(x), q = Phi(-x), u0 = log p, v0 = log q, and a rounding allowance.
    0 < k < n: Stirling remainders and deviance terms (the textbook terms are of order n and cancel)."""
    with np.errstate(invalid="ignore"):
        logp = np.where(k > 0, k * u0, 0.0) + np.where(n - k > 0, (n - k) * v0, 0.0)
    err = 8.0 * EPS * (np.abs(logp) + 1.0)
    mid = (k > 0) & (k < n)
    if mid.any():
        km, nm, xm, um, vm = k[mid], n[mid], x[mid], u0[mid], v0[mid]
        jm = nm - km
        low = xm <= 0.0
        small = special.ndtr(np.where(low, xm, -xm))       # the smaller of p and q, to full relative precision
        d = np.where(low, km - nm * small, nm * small - jm)   # k - n p = n q - (n - k)
        b1 = _bd0(km, d, np.log(km / nm) - um)    # log(k / n), log((n - k) / n): one rounding each, no 1 - k/n
        b2 = _bd0(jm, -d, np.log(jm / nm) - vm)
        st = _stirlerr(nm) - _stirlerr(km) - _stirlerr(jm)
        lead = -0.5 * (LOG_2PI + np.log(km) + np.log(jm / nm))
        logp[mid] = st + lead - b1 - b2
        err[mid] = 8.0 * EPS * (np.abs(st) + np.abs(lead) + b1 + b2 + np.abs(d) * (np.abs(um) + np.abs(vm) + 1.0)
                                + 1.0)
    return logp, err


def kernel_full(kind: Any, eta: Any, obs: Mapping[str, Any], params: Mapping[str, Any]) -> KernelResult:
    """Log-likelihood per observation with derivatives in eta (up to the third) and in each parameter."""
    kind = _kind_name(kind)
    eta, o, p = _arrays(kind, eta, obs, params)
    shape = eta.shape
    r = _kernel_flat(kind, eta.ravel(), {k: v.ravel() for k, v in o.items()}, {k: v.ravel() for k, v in p.items()})
    fix = lambda a: np.reshape(a, shape)
    return KernelResult(fix(r.logp), fix(r.d1), fix(r.d2), fix(r.d3),
                        {n: tuple(fix(a) for a in v) for n, v in r.dparam.items()}, fix(r.logp_err))


def _kernel_flat(kind: str, eta: np.ndarray, o: dict, p: dict) -> KernelResult:
    if kind in ("linear", "fraction"):
        return _gauss(eta, o["y"], p["sd"])
    if kind == "linear_cell":
        return _gauss_cell(eta, o["n"], o["ybar"], o["ss"], p["sd"], p["tau"])
    if kind == "binary":
        sgn = np.where(o["y"] >= 0.5, 1.0, -1.0)
        l0, d1, d2, d3, dsd, err = _one_sided(sgn, eta, p["sd"])
        return KernelResult(l0, d1, d2, d3, {"sd": dsd}, err)
    if kind == "levels":
        lo = np.where(o["lo"] <= -OPEN_FROM, -np.inf, o["lo"])
        hi = np.where(o["hi"] >= OPEN_FROM, np.inf, o["hi"])
        return _interval(eta, lo, hi, p["sd"])[0]
    if kind == "items":
        k, n, sd = o["k"], o["n"], p["sd"]
        if np.any((n > 1) & (sd != 0.0)):
            raise ValueError("laplace_kernels: items with n > 1 need sd = 0: the items of one call share its "
                             "noise, so integrate it as an explicit unknown (a cell term), not as sqrt(1 + sd^2)")
        if np.any((k < 0) | (k > n)):
            raise ValueError("laplace_kernels: items need 0 <= k <= n")
        c = np.sqrt(1.0 + sd * sd)
        x = eta / c
        u0, u1, u2, u3 = log_ndtr_derivs(x)
        v0, v1, v2, v3 = log_ndtr_derivs(-x)
        logp, err = _items_logp(k, n, x, u0, v0)
        d1 = (k * u1 - (n - k) * v1) / c
        d2 = (k * u2 + (n - k) * v2) / c ** 2
        d3 = (k * u3 - (n - k) * v3) / c ** 3
        f = sd / (c * c)
        dsd = (-f * eta * d1, -f * (eta * d2 + d1), -f * (eta * d3 + 2.0 * d2))
        return KernelResult(logp, d1, d2, d3, {"sd": dsd}, err + 4.0 * EPS * np.abs(eta * d1))
    # pairs
    v, orient, w, sd, pos = o["v"], o["o"], p["width"], p["sd"], p["position"]
    x = eta + pos * orient
    lo = np.where(v == 0, w, np.where(v == 1, -w, -np.inf))
    hi = np.where(v == 0, np.inf, np.where(v == 1, w, -w))
    res, dlo, dhi, dhalf = _interval(x, lo, hi, sd)
    # a: lo = width; tie: the half-width of (-width, width); b: hi = -width
    res.dparam["width"] = tuple(np.where(v == 0, dlo[k], np.where(v == 1, dhalf[k], -dhi[k])) for k in range(3))
    res.dparam["position"] = (orient * res.d1, orient * res.d2, orient * res.d3)
    return res


def kernel(kind: Any, eta: Any, obs: Mapping[str, Any],
           params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(logp, d1, d2)` per observation: the log-likelihood and its first two derivatives in eta."""
    r = kernel_full(kind, eta, obs, params)
    return r.logp, r.d1, r.d2


# ---- quadrature rules ------------------------------------------------------------------------------------

def hermite_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for integral exp(-x^2) f(x) dx."""
    return np.polynomial.hermite.hermgauss(int(n))


# Gauss-Kronrod (7, 15) on [-1, 1] (QUADPACK qk15). Index 1, 3, 5 of the positive nodes and 0 are Gauss nodes.
_XK = np.array([0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                0.207784955007898467600689403773245, 0.0])
_WK = np.array([0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                0.204432940075298892414161999234649, 0.209482141084727828012999174891714])
_WG = np.array([0.0, 0.129484966168869693270611432679082, 0.0, 0.279705391489276667901467771423780,
                0.0, 0.381830050505118944950369775488975, 0.0, 0.417959183673469387755102040816327])


def kronrod_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 15 Kronrod nodes on [-1, 1], their weights, and the 7-point Gauss weights on the same nodes (0
    off the Gauss nodes)."""
    x = np.concatenate([-_XK[:-1], _XK[::-1]])
    wk = np.concatenate([_WK[:-1], _WK[::-1]])
    wg = np.concatenate([_WG[:-1], _WG[::-1]])
    return x, wk, wg


# ---- cell integrals --------------------------------------------------------------------------------------

@dataclass
class CellResult:
    """What `cell_integral` returns. C cells, N readings (in the caller's order).

    logl[C]           log integral phi(z) prod_i p(obs_i | m_i + tau_c z) dz
    err[C]            estimated absolute error of logl (relative error of the integral), rounding included;
                      NaN for a frozen rule
    err_kind[C]       "kronrod" (composite), "heuristic" (agh: a difference of two rules, not a bound) or
                      "none" (frozen rule)
    method[C]         "agh", "composite" or "frozen", per cell
    evals[C]          integrand evaluations used (nodes of the returned rule)
    mode[C], scale[C] where the rule was placed: the mode of the integrand in z and its curvature scale
                      (-h'')^(-1/2) at the arguments it was built for (for a frozen rule, those of the
                      earlier call, not of the current integrand)
    d1[N]             d logl / d m_i
    hess_i, hess_j, hess_v
                      d2 logl / d m_i d m_j for every ordered pair (i, j) of readings of one cell, i = j
                      included (reading indices in the caller's order)
    shift_d1, shift_d2, shift_d3 [C]
                      derivatives of logl in a shift added to every m_i of the cell
    dtau[C]           d logl / d tau_c
    dparam[name][N]   d logl / d params[name]_i (sum over the cell's readings for a shared parameter)
    deriv_err         estimated absolute errors, rounding included, of the derivatives above, in arrays of
                      the same shapes: "d1", "hess" (aligned with hess_v), "shift_d1", "shift_d2", "shift_d3",
                      "dtau", and "dparam" (a dict by name); NaN for a frozen rule
    deriv_ok[C]       every derivative estimate of the cell is within dtol of its scale: D = |shift_d1| +
                      |shift_d2|^(1/2) + D0 for the cell (shift_dk to dtol D^k, dtau to dtol D (1 + |E z|)) and
                      D_i = |d1_i| + |H_ii|^(1/2) + D0 for reading i (d1_i to dtol D_i, H_ij to dtol D_i D_j, a
                      parameter derivative v to dtol (|v| + D_i + 3 |theta| D_i^2)), with the floor
                      D0 = 1 / max(tau, w), w the narrowest feature width of the cell's readings, so that a
                      derivative near 0 is judged on the cell's curvature scale, not on itself; False when that
                      could not be established, and for every frozen rule
    rule              the nodes (z, log weight, cell) behind every number here, with placement metadata; pass
                      it back as `rule=`
    """

    logl: np.ndarray
    err: np.ndarray
    err_kind: np.ndarray
    method: np.ndarray
    evals: np.ndarray
    mode: np.ndarray
    scale: np.ndarray
    d1: np.ndarray
    hess_i: np.ndarray
    hess_j: np.ndarray
    hess_v: np.ndarray
    shift_d1: np.ndarray
    shift_d2: np.ndarray
    shift_d3: np.ndarray
    dtau: np.ndarray
    dparam: dict
    deriv_err: dict
    deriv_ok: np.ndarray
    rule: dict


class _Cells:
    """Readings sorted by cell, with the per-cell sums the integrand needs."""

    def __init__(self, kind: str, m: Any, obs: Mapping[str, Any], params: Mapping[str, Any], cell: Any, tau: Any,
                 n_cells: int | None) -> None:
        m = np.asarray(m, dtype=float).ravel()
        cell = np.asarray(cell, dtype=np.int64).ravel()
        if cell.shape != m.shape:
            raise ValueError("laplace_kernels: cell and m must have one entry per reading")
        C = int(n_cells if n_cells is not None else (cell.max() + 1 if len(cell) else 0))
        tau = np.broadcast_to(np.asarray(tau, dtype=float), (C,)).astype(float)
        if len(cell) and (cell.min() < 0 or cell.max() >= C):
            raise ValueError("laplace_kernels: a cell index is outside 0..n_cells-1")
        if np.any(tau < 0) or not np.all(np.isfinite(tau)):
            raise ValueError("laplace_kernels: tau must be finite and >= 0")
        self.kind, self.C, self.N = kind, C, len(m)
        self.order = np.argsort(cell, kind="stable")
        self.cell = cell[self.order]
        self.m = m[self.order]
        self.tau = tau
        self.obs = {k: np.broadcast_to(np.asarray(obs[k], dtype=float), m.shape)[self.order] for k in OBS[kind]}
        self.params = {k: np.broadcast_to(np.asarray(params[k], dtype=float), m.shape)[self.order]
                       for k in PARAMS[kind]}
        self._sums()

    def _sums(self) -> None:
        self.count = np.bincount(self.cell, minlength=self.C)
        self.start = np.concatenate([[0], np.cumsum(self.count)[:-1]]).astype(np.int64)
        n_c = self.count[self.cell]                         # ordered pairs (i, j) of readings of one cell
        self.pi = np.repeat(np.arange(self.N), n_c)
        self.pj = (self.start[self.cell[self.pi]] + np.arange(len(self.pi))
                   - np.repeat(np.cumsum(n_c) - n_c, n_c)).astype(np.int64)
        self.pair_start = np.concatenate([[0], np.cumsum(self.count ** 2)[:-1]]).astype(np.int64)

    def pair_index(self, gi: np.ndarray, gj: np.ndarray) -> np.ndarray:
        """Position of the ordered pair of (sorted) readings gi, gj of one cell in the pair arrays."""
        c = self.cell[gi]
        return self.pair_start[c] + (gi - self.start[c]) * self.count[c] + (gj - self.start[c])

    def subset(self, which: np.ndarray) -> tuple["_Cells", np.ndarray, np.ndarray]:
        """The cells in `which` only, renumbered from 0; their old indices; their readings' old positions."""
        old = np.flatnonzero(which)
        new_of = np.full(self.C, -1, dtype=np.int64)
        new_of[old] = np.arange(len(old))
        rows = which[self.cell]
        sub = object.__new__(_Cells)
        sub.kind, sub.C, sub.N = self.kind, len(old), int(rows.sum())
        sub.order = np.arange(sub.N)
        sub.cell = new_of[self.cell[rows]]
        sub.m = self.m[rows]
        sub.tau = self.tau[old]
        sub.obs = {k: v[rows] for k, v in self.obs.items()}
        sub.params = {k: v[rows] for k, v in self.params.items()}
        sub._sums()
        return sub, old, np.flatnonzero(rows)

    def h(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """log integrand per cell at z[C] and its first two derivatives in z."""
        eta = self.m + self.tau[self.cell] * z[self.cell]
        r = kernel_full(self.kind, eta, self.obs, self.params)
        t = self.tau
        h0 = -0.5 * z * z - 0.5 * LOG_2PI + np.bincount(self.cell, r.logp, minlength=self.C)
        h1 = -z + t * np.bincount(self.cell, r.d1, minlength=self.C)
        h2 = -1.0 + t * t * np.bincount(self.cell, r.d2, minlength=self.C)
        return h0, h1, h2

    def mode(self, *, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mode of each cell's integrand (strictly concave in z) by Newton steps with step halving."""
        z = np.zeros(self.C)
        h0, h1, h2 = self.h(z)
        active = np.ones(self.C, dtype=bool)
        for _ in range(max_iter):
            step = -h1 / h2
            active &= np.abs(step) > 1e-12 * (1.0 + np.abs(z))
            active &= np.abs(h1) * np.sqrt(-1.0 / h2) > 1e-13
            if not active.any():
                break
            trial = np.where(active, z + step, z)
            t0, t1, t2 = self.h(trial)
            for _ in range(60):
                bad = active & ~(t0 >= h0 - 1e-12 * np.abs(h0))
                if not bad.any():
                    break
                step = np.where(bad, 0.5 * step, step)
                trial = np.where(bad, z + step, trial)
                t0, t1, t2 = self.h(trial)
            z, h0, h1, h2 = trial, t0, t1, t2
        return z, h0, h2

    def level(self, z0: np.ndarray, h0: np.ndarray, scale: np.ndarray, drop: float, side: float,
              *, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The point on one side of the mode where log f = h0 - drop, approached from beyond it (Newton on a
        concave function from that side never crosses back). Returns the point, log f and its slope there."""
        target = h0 - drop
        y = scale * np.sqrt(2.0 * drop)
        f0, f1, _ = self.h(z0 + side * y)
        for _ in range(max_iter):
            g = f0 - target
            done = (g <= 0.0) & (g > -1e-3 * max(1.0, drop))
            if done.all():
                break
            dy = -g / (side * f1)
            y = np.where(done, y, np.maximum(y + dy, 0.5 * y))
            f0, f1, _ = self.h(z0 + side * y)
        return z0 + side * y, f0, f1

    def walls(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Locations in eta and widths of the features of each reading (cut points, turns), with the reading."""
        k, o, p = self.kind, self.obs, self.params
        idx = np.arange(self.N)
        if k in ("linear", "fraction"):
            return o["y"], p["sd"], idx
        if k == "linear_cell":
            return o["ybar"], p["sd"] / np.sqrt(np.maximum(o["n"], 1.0)), idx
        if k == "binary":
            return np.zeros(self.N), p["sd"], idx
        if k == "items":   # where k log Phi(x) and (n - k) log Phi(-x) turn from steep to flat, x = eta / c
            c = np.sqrt(1.0 + p["sd"] ** 2)
            locs, wids, rds = [], [], []
            for count, sign in ((o["k"], 1.0), (o["n"] - o["k"], -1.0)):
                has = count >= 1
                a = np.maximum(-special.ndtri(1.0 / (count[has] + 1.0)), 0.0)
                locs.append(sign * c[has] * a)
                wids.append(c[has] / np.maximum(a, 1.0))
                rds.append(idx[has])
            return np.concatenate(locs), np.concatenate(wids), np.concatenate(rds)
        if k == "pairs":   # the verdict edges +-width, moved by the position effect
            w, v, shift = p["width"], o["v"], p["position"] * o["o"]
            lo = np.where(v == 0, w, np.where(v == 1, -w, -np.inf)) - shift
            hi = np.where(v == 0, np.inf, np.where(v == 1, w, -w)) - shift
            keep = np.concatenate([np.isfinite(lo), np.isfinite(hi)])
        else:
            lo, hi = o["lo"], o["hi"]
            keep = np.concatenate([np.isfinite(lo) & (lo > -OPEN_FROM), np.isfinite(hi) & (hi < OPEN_FROM)])
        loc = np.concatenate([lo, hi])
        wid = np.concatenate([p["sd"], p["sd"]])
        rd = np.concatenate([idx, idx])
        return loc[keep], wid[keep], rd[keep]


def _agh_rule(z0: np.ndarray, scale: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, w = hermite_rule(n)
    C = len(z0)
    z = (z0[:, None] + SQRT2 * scale[:, None] * x[None, :]).ravel()
    lw = (np.log(SQRT2 * scale)[:, None] + (np.log(w) + x * x)[None, :]).ravel()
    return z, lw, np.repeat(np.arange(C), n)


def _ends(cells: _Cells, z0: np.ndarray, h0: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, ...]:
    """The two points where log f has dropped by DROP from its maximum, with log f and its slope at each."""
    return cells.level(z0, h0, scale, DROP, -1.0) + cells.level(z0, h0, scale, DROP, 1.0)


def _composite_rule(cells: _Cells, z0: np.ndarray, h0: np.ndarray, scale: np.ndarray, ends: tuple,
                    split: int = 1) -> tuple[np.ndarray, ...]:
    """Kronrod panels for every cell between its two ends: nodes, Kronrod and Gauss log weights, cell and panel.
    Every panel is cut into `split` equal pieces."""
    zl, hl, sl, zr, hr, sr = ends
    pts_z = [zl, zr, z0]
    pts_c = [np.arange(cells.C)] * 3
    for d in SPLIT_DROPS:
        for side in (-1.0, 1.0):
            pts_z.append(cells.level(z0, h0, scale, d, side, max_iter=20)[0])
            pts_c.append(np.arange(cells.C))
    loc, wid, rd = cells.walls()
    c_of = cells.cell[rd]
    t = cells.tau[c_of]
    ok = t > 0
    if ok.any():
        zc = (loc[ok] - cells.m[rd[ok]]) / t[ok]
        om = wid[ok] / t[ok]
        for off in (0.0,) + WALL_OFFSETS + tuple(-x for x in WALL_OFFSETS):
            pts_z.append(zc + off * om)
            pts_c.append(c_of[ok])
    pz = np.concatenate(pts_z)
    pc = np.concatenate(pts_c)
    keep = np.isfinite(pz) & (pz >= zl[pc]) & (pz <= zr[pc])
    pz, pc = pz[keep], pc[keep]
    order = np.lexsort((pz, pc))
    pz, pc = pz[order], pc[order]
    same = pc[1:] == pc[:-1]
    span = (zr - zl)[pc[1:]]
    a, b, cc = pz[:-1][same], pz[1:][same], pc[1:][same]
    good = (b - a) > 1e-13 * np.maximum(span[same], 1e-300)
    a, b, cc = a[good], b[good], cc[good]
    if split > 1:
        f = np.arange(split) / split
        a, b = (a[:, None] + (b - a)[:, None] * f).ravel(), (a[:, None] + (b - a)[:, None] * (f + 1.0 / split)).ravel()
        cc = np.repeat(cc, split)
    x, wk, wg = kronrod_rule()
    half = 0.5 * (b - a)
    z = ((a + b)[:, None] * 0.5 + half[:, None] * x[None, :]).ravel()
    with np.errstate(divide="ignore"):
        lwk = (np.log(half)[:, None] + np.log(wk)[None, :]).ravel()
        lwg = (np.log(half)[:, None] + np.log(np.where(wg > 0, wg, 0.0))[None, :]).ravel()
    node_cell = np.repeat(cc, len(x))
    panel = np.repeat(np.arange(len(a)), len(x))
    return z, lwk, lwg, node_cell, panel


def _sharpness(cells: _Cells, scale: np.ndarray, zl: np.ndarray, zr: np.ndarray) -> np.ndarray:
    """Per cell, the smallest width (in units of the curvature scale) of a wall that lies within a few of its own
    widths of the interval [zl, zr] where the integrand is not negligible; inf when there is none."""
    loc, wid, rd = cells.walls()
    out = np.full(cells.C, np.inf)
    c_of = cells.cell[rd]
    t = cells.tau[c_of]
    ok = t > 0
    if ok.any():
        c_of, rd = c_of[ok], rd[ok]
        zc = (loc[ok] - cells.m[rd]) / t[ok]
        om = wid[ok] / t[ok]
        near = (zc >= zl[c_of] - 3.0 * om) & (zc <= zr[c_of] + 3.0 * om)
        np.minimum.at(out, c_of[near], (om / scale[c_of])[near])
    return out


def _default_low(n: int) -> int:
    low = max(1, n // 2)
    return low + 1 if low % 2 == 0 else low


def _sup_derivs(cells: _Cells) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per reading, bounds on |d2|, |d3|, |d4| in eta over all eta, and on |d (d logp / d width) / d eta| (pairs).
    For log Phi and log(Phi(a) - Phi(b)) the second derivative lies in (-1, 0) (a truncated normal has variance
    below 1); the third and fourth are its cumulants, with measured suprema 0.30 and 0.38 (bounds 0.5 and 1);
    the mixed width derivative is an end's or the half-width's slope, measured supremum 1.0 (bound 1)."""
    k, o, p = cells.kind, cells.obs, cells.params
    zero = np.zeros(cells.N)
    if k in ("linear", "fraction"):
        return 1.0 / p["sd"] ** 2, zero, zero, zero
    if k == "linear_cell":
        return o["n"] / (p["sd"] ** 2 + o["n"] * p["tau"] ** 2), zero, zero, zero
    if k == "items":
        c = np.sqrt(1.0 + p["sd"] ** 2)
        return o["n"] / c ** 2, 0.5 * o["n"] / c ** 3, o["n"] / c ** 4, zero
    s = p["sd"]
    return 1.0 / s ** 2, 0.5 / s ** 3, 1.0 / s ** 4, (1.0 / s ** 2 if k == "pairs" else zero)


def _evaluate(cells: _Cells, z: np.ndarray, lw: np.ndarray, nc: np.ndarray, tails: tuple | None, *,
              lwg: np.ndarray | None = None, panel: np.ndarray | None = None) -> dict:
    """Evaluate a rule (nodes z with log weights lw, sorted by cell) on every cell: logl and the posterior
    expectations behind every derivative, each as (mean, quadrature error, fixed error). The quadrature error
    is the per-panel Kronrod-Gauss difference when the Gauss weights `lwg` and `panel` are given (a composite
    rule), else None (the caller differences two rules). The fixed error does not shrink with finer panels:
    floating-point error, and the part of the integrand outside the drop-DROP ends `tails` (zl, hl, sl, zr, hr,
    sr: each end, log f there and its slope), bounded by the exponential envelope that log-concavity gives."""
    C, N = cells.C, cells.N
    Kc = np.bincount(nc, minlength=C)
    node_start = np.concatenate([[0], np.cumsum(Kc)[:-1]]).astype(np.int64)
    rep = Kc[cells.cell]
    T = int(rep.sum())
    r_idx = np.repeat(np.arange(N), rep)                  # (reading, node) layout, nodes in rule order
    n_idx = node_start[cells.cell[r_idx]] + np.arange(T) - np.repeat(np.cumsum(rep) - rep, rep)
    tz = cells.tau[cells.cell[r_idx]] * z[n_idx]
    eta = cells.m[r_idx] + tz
    eta_err = 2.0 * EPS * (np.abs(cells.m[r_idx]) + np.abs(tz))   # eta and the node itself, rounded
    obs_r = {k: v[r_idx] for k, v in cells.obs.items()}
    par_r = {k: v[r_idx] for k, v in cells.params.items()}
    kr = kernel_full(cells.kind, eta, obs_r, par_r)
    M = len(z)
    d1, d2, d3 = kr.d1, kr.d2, kr.d3
    h = -0.5 * z * z - 0.5 * LOG_2PI + np.bincount(n_idx, kr.logp, minlength=M)
    t = lw + h
    mx = np.full(C, -np.inf)
    np.maximum.at(mx, nc, t)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    with np.errstate(divide="ignore"):
        logl = mx + np.log(np.bincount(nc, np.exp(t - mx[nc]), minlength=C))
    w = np.exp(t - logl[nc])                              # posterior weight of each node
    delta = (np.bincount(n_idx, kr.logp_err + np.abs(d1) * eta_err, minlength=M) + 4.0 * EPS * (np.abs(h) + 1.0))
    out: dict = {"logl": logl}

    # tails: per cell and end, exp(log f(end) - logl), the envelope's decay rate, and the end
    ends = []
    if tails is not None:
        zl, hl, sl, zr, hr, sr = tails
        M2s, M3s, M4s, Mws = _sup_derivs(cells)
        for ze, he, be in ((zl, hl, sl), (zr, hr, -sr)):
            with np.errstate(over="ignore", invalid="ignore"):
                rel = np.exp(he - logl)
                rate = np.where(be > 0.0, be, np.nan)
            eta_e = cells.m + cells.tau[cells.cell] * ze[cells.cell]
            ke = kernel_full(cells.kind, eta_e, cells.obs, cells.params)
            ends.append({"z": ze, "rel": rel, "b": rate, "k": ke, "eta": eta_e})
    else:
        M2s = M3s = M4s = Mws = np.zeros(N)

    def tail(polys: list, owner_cell: np.ndarray) -> Any:
        """sum over the two ends of the envelope integral of a polynomial bound sum_j c_j u^j on |g - E g|."""
        if tails is None:
            return 0.0
        tot = 0.0
        for e, coefs in zip(ends, polys):
            b = e["b"][owner_cell]
            rel = e["rel"][owner_cell]
            s = sum(c * math.factorial(j) / b ** (j + 1) for j, c in enumerate(coefs))
            with np.errstate(invalid="ignore"):   # no envelope (an end not found, a slope not outward): unbounded
                tot = tot + np.where(np.isnan(rel) | np.isnan(s), np.inf, np.where(rel > 0.0, rel * s, 0.0))
        return tot

    with np.errstate(invalid="ignore", divide="ignore"):
        tail_mass = tail([[1.0], [1.0]], np.arange(C))
    logl_fixed = (np.log1p(np.bincount(nc, w * np.expm1(np.minimum(delta, 50.0)), minlength=C))
                  + 4.0 * EPS * (np.abs(logl) + Kc) + tail_mass)
    comp = lwg is not None
    if comp:
        with np.errstate(invalid="ignore"):
            rho = -np.expm1(lwg - lw)                     # 1 - w_G / w_K: (K15 - G7) per node, relative
        P = int(panel.max()) + 1 if len(panel) else 0
        pcell = np.zeros(P, dtype=np.int64)
        pcell[panel] = nc
        logl_quad = np.bincount(pcell, np.abs(np.bincount(panel, w * rho, minlength=P)), minlength=C)
    else:
        logl_quad = None
    out["logl_e"] = (logl, logl_quad, logl_fixed)

    def _quad(dev: np.ndarray, wts: np.ndarray, owner: np.ndarray, nodes: np.ndarray, n_out: int) -> Any:
        """Sum over panels of |sum (w_K - w_G) f dev| / I, per owner; owners' elements are contiguous by panel."""
        if not comp:
            return None
        if len(dev) == 0:
            return np.zeros(n_out)
        key = panel[nodes]
        brk = np.flatnonzero((owner[1:] != owner[:-1]) | (key[1:] != key[:-1])) + 1
        starts = np.concatenate([[0], brk])
        sums = np.add.reduceat(wts * rho[nodes] * dev, starts)
        return np.bincount(owner[starts], np.abs(sums), minlength=n_out)

    def expect(g: np.ndarray, gerr: Any, owner: np.ndarray, nodes: np.ndarray, n_out: int,
               owner_cell: np.ndarray, polys: Any) -> tuple:
        """(mean, quadrature error, fixed error) of E g; polys(mean) gives the tail bound's polynomials."""
        wts = w[nodes]
        # divide by the weights' own sum: it is 1 only to the rounding of logl (eps |logl|), which would add
        # that much times |g| (a cell at z = 333 with logl = -5.6e4 had E z off by 1.2e-9)
        mean = np.bincount(owner, wts * g, minlength=n_out) / np.bincount(owner, wts, minlength=n_out)
        dev = g - mean[owner]
        rnd = np.bincount(owner, wts * (8.0 * EPS * np.abs(g) + gerr + 2.0 * delta[nodes] * np.abs(dev)),
                          minlength=n_out)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            fixed = rnd + 4.0 * EPS * np.abs(mean) + tail(polys(np.abs(mean)), owner_cell)
        return mean, _quad(dev, wts, owner, nodes, n_out), fixed

    nodes_all = np.arange(M)
    cells_all = np.arange(C)
    rc = cells.cell
    tau_r = cells.tau[rc]
    # per end: |d1| there, and its growth into the tail (tau sup|d2|), per reading and summed per cell
    a_r = [np.abs(e["k"].d1) for e in ends]
    b_r = tau_r * M2s
    A = [np.bincount(rc, a, minlength=C) for a in a_r]
    B = np.bincount(rc, b_r, minlength=C)
    M2c, M3c = np.bincount(rc, M2s, minlength=C), np.bincount(rc, M3s, minlength=C)
    ze = [np.abs(e["z"]) for e in ends]
    L1 = np.bincount(n_idx, d1, minlength=M)
    S2 = np.bincount(n_idx, d2, minlength=M)
    S3 = np.bincount(n_idx, d3, minlength=M)
    A1 = np.bincount(n_idx, np.abs(d1), minlength=M)
    A2 = np.bincount(n_idx, np.abs(d2), minlength=M)
    A3 = np.bincount(n_idx, np.abs(d3), minlength=M)
    E2 = np.bincount(n_idx, np.abs(d2) * eta_err, minlength=M)      # what eta's rounding does to each sum
    E3 = np.bincount(n_idx, np.abs(d3) * eta_err, minlength=M)
    E4 = np.bincount(n_idx, M4s[r_idx] * eta_err, minlength=M)
    aL = np.abs(L1)
    out["L1"] = expect(L1, 16.0 * EPS * A1 + E2, nc, nodes_all, C, cells_all,
                       lambda g: [[A[e] + g, B] for e in range(len(ends))])
    out["Q2"] = expect(S2 + L1 * L1, 16.0 * EPS * (A2 + A1 * A1) + E3 + 2.0 * aL * E2, nc, nodes_all, C, cells_all,
                       lambda g: [[M2c + A[e] ** 2 + g, 2.0 * A[e] * B, B * B] for e in range(len(ends))])
    out["Q3"] = expect(S3 + 3.0 * L1 * S2 + L1 ** 3,
                       16.0 * EPS * (A3 + 3.0 * A1 * A2 + A1 ** 3) + E4 + 3.0 * (np.abs(S2) + L1 * L1) * E2
                       + 3.0 * aL * E3, nc, nodes_all, C, cells_all,
                       lambda g: [[M3c + 3.0 * M2c * A[e] + A[e] ** 3 + g, 3.0 * M2c * B + 3.0 * A[e] ** 2 * B,
                                   3.0 * A[e] * B * B, B ** 3] for e in range(len(ends))])
    out["zL1"] = expect(z * L1, 16.0 * EPS * np.abs(z) * A1 + np.abs(z) * E2, nc, nodes_all, C, cells_all,
                        lambda g: [[ze[e] * A[e] + g, ze[e] * B + A[e], B] for e in range(len(ends))])
    out["z"] = expect(z, 0.0, nc, nodes_all, C, cells_all, lambda g: [[ze[e] + g, 1.0] for e in range(len(ends))])
    zbar = out["z"][0]
    dzc = [np.abs(e["z"] - zbar) for e in ends]
    dz = z - zbar[nc]
    out["V"] = expect(dz * dz, 0.0, nc, nodes_all, C, cells_all,
                      lambda g: [[dzc[e] ** 2 + g, 2.0 * dzc[e], 1.0] for e in range(len(ends))])
    out["K3"] = expect(dz ** 3, 0.0, nc, nodes_all, C, cells_all,
                       lambda g: [[dzc[e] ** 3 + g, 3.0 * dzc[e] ** 2, 3.0 * dzc[e], 1.0] for e in range(len(ends))])
    out["d1"] = expect(d1, 16.0 * EPS * np.abs(d1) + np.abs(d2) * eta_err, r_idx, n_idx, N, rc,
                       lambda g: [[a_r[e] + g, b_r] for e in range(len(ends))])
    out["e2"] = expect(d2, 16.0 * EPS * np.abs(d2) + np.abs(d3) * eta_err, r_idx, n_idx, N, rc,
                       lambda g: [[M2s + g] for e in range(len(ends))])
    out["dparam"] = {}
    if cells.kind == "pairs":
        dw = kr.dparam["width"]
        w_r = [np.abs(e["k"].dparam["width"][0]) for e in ends]
        out["dparam"]["width"] = expect(dw[0], 16.0 * EPS * np.abs(dw[0]) + np.abs(dw[1]) * eta_err, r_idx, n_idx,
                                        N, rc, lambda g: [[w_r[e] + g, tau_r * Mws] for e in range(len(ends))])
    if cells.kind == "linear_cell":   # Gaussian in the residual r = ybar - eta: quadratic growth into the tail
        n_, ss, sd, tp = cells.obs["n"], cells.obs["ss"], cells.params["sd"], cells.params["tau"]
        V = sd * sd + n_ * tp * tp
        for name in ("sd", "tau"):
            dv = kr.dparam[name]
            polys = []
            for e in ends:
                r_e = np.abs(cells.obs["ybar"] - e["eta"])
                if name == "sd":
                    c0 = (n_ - 1.0) / sd + sd / V + ss / sd ** 3 + n_ * sd * r_e ** 2 / V ** 2
                    f = n_ * sd / V ** 2
                else:
                    c0 = n_ * tp / V + n_ * n_ * tp * r_e ** 2 / V ** 2
                    f = n_ * n_ * tp / V ** 2
                polys.append((c0, 2.0 * f * r_e * tau_r, f * tau_r ** 2))
            out["dparam"][name] = expect(dv[0], 16.0 * EPS * np.abs(dv[0]) + np.abs(dv[1]) * eta_err, r_idx, n_idx,
                                         N, rc, lambda g, polys=polys: [[p[0] + g, p[1], p[2]] for p in polys])
    # Cov(d1_i, d1_j) over the ordered pairs of one cell, on a (pair, node) layout
    pi, pj = cells.pi, cells.pj
    Kp = Kc[rc[pi]]
    pair_rep = np.repeat(np.arange(len(pi)), Kp)
    k_off = np.arange(int(Kp.sum())) - np.repeat(np.cumsum(Kp) - Kp, Kp)
    block = np.concatenate([[0], np.cumsum(rep)[:-1]]).astype(np.int64)
    ai = block[pi[pair_rep]] + k_off
    aj = block[pj[pair_rep]] + k_off
    dhat = out["d1"][0]
    c1 = d1 - dhat[r_idx]
    prod = c1[ai] * c1[aj]
    perr = (16.0 * EPS * (np.abs(prod) + np.abs(c1[ai] * d1[aj]) + np.abs(d1[ai] * c1[aj]))
            + np.abs(c1[aj] * d2[ai]) * eta_err[ai] + np.abs(c1[ai] * d2[aj]) * eta_err[aj])
    pc = [[a_r[e][pi] + np.abs(dhat[pi]), a_r[e][pj] + np.abs(dhat[pj])] for e in range(len(ends))]
    out["cov"] = expect(prod, perr, pair_rep, n_idx[ai], len(pi), rc[pi],
                        lambda g: [[pc[e][0] * pc[e][1] + g, pc[e][0] * b_r[pj] + pc[e][1] * b_r[pi],
                                    b_r[pi] * b_r[pj]] for e in range(len(ends))])
    return out


_COMPONENTS = ("logl_e", "L1", "Q2", "Q3", "zL1", "z", "V", "K3", "d1", "e2", "cov")


def _agh_quad(hi: dict, lo: dict) -> dict:
    """The high rule's results with the difference to the low rule as every quadrature error (a heuristic)."""
    out = dict(hi)
    for key in _COMPONENTS:
        out[key] = (hi[key][0], np.abs(hi[key][0] - lo[key][0]), hi[key][2])
    out["dparam"] = {n: (v[0], np.abs(v[0] - lo["dparam"][n][0]), v[2]) for n, v in hi["dparam"].items()}
    return out


def _err(*terms: tuple, extra: Any = 0.0) -> tuple:
    """(total, rounding part) of the error of a combination: terms are (coefficient, (mean, quad, rnd))."""
    q = sum(c * t[1] for c, t in terms)
    r = sum(c * t[2] for c, t in terms) + extra
    return q + r, r


def _pick(a: tuple, b: tuple) -> tuple:
    """Of two candidates (value, (total, rounding)), the one with the smaller total error, per element."""
    use_a = a[1][0] <= b[1][0]
    return (np.where(use_a, a[0], b[0]), (np.where(use_a, a[1][0], b[1][0]), np.where(use_a, a[1][1], b[1][1])))


def _param_derivs(cells: _Cells, ev: dict, d1: np.ndarray, hii: np.ndarray, e_d1: list, e_hii: list) -> tuple:
    """d logl / d params per reading, with (total, fixed) errors. The noise sd of every kind but linear_cell by
    the heat identity, pointwise d p / d sd = sd d2 p / d eta2, so d logl / d sd_i = sd_i (H_ii + d1_i^2); the
    position as o d1; the tie width of an a or b verdict as -d1 or d1 (its one edge moves as eta does); the
    rest (the width of a tie, linear_cell) as posterior means."""
    dparam, e_par = {}, {}
    for name in PARAMS[cells.kind]:
        if name == "position":
            o = cells.obs["o"]
            dparam[name], e_par[name] = o * d1, tuple(np.abs(o) * e for e in e_d1)
        elif name == "width":
            t_ = ev["dparam"][name]
            v = cells.obs["v"]
            edge = v != 1
            e_mom = _err((1.0, t_))
            dparam[name] = np.where(edge, np.where(v == 0, -d1, d1), t_[0])
            e_par[name] = tuple(np.where(edge, ed, em) for ed, em in zip(e_d1, e_mom))
        elif name == "sd" and cells.kind in HEAT_KINDS:
            sd = cells.params["sd"]
            dparam[name] = sd * (hii + d1 * d1)
            e_par[name] = tuple(np.abs(sd) * (eh + (2.0 * np.abs(d1) + ed) * ed + 4.0 * EPS * (np.abs(hii) + d1 * d1))
                                for eh, ed in zip(e_hii, e_d1))
        else:
            t_ = ev["dparam"][name]
            dparam[name], e_par[name] = t_[0], _err((1.0, t_))
    return dparam, e_par


def _finish(cells: _Cells, ev: dict, dtol: float, tol: float, *, frozen: bool = False) -> dict:
    """Choose, per cell and quantity, between the identities and the moment forms; propagate the errors;
    check every derivative against dtol; mark for refinement the cells whose quadrature error (not rounding)
    misses tol or dtol. Frozen: moment forms (exact for the frozen sum), no errors."""
    if frozen:   # the values only: zero quadrature errors, so the combinations below go through
        ev = _agh_quad(ev, ev)
    tau = cells.tau
    s1m_t = ev["L1"]
    EQ2, EQ3 = ev["Q2"][0], ev["Q3"][0]
    s1m = s1m_t[0]
    mom = {
        "shift_d1": (s1m, _err((1.0, s1m_t))),
        "shift_d2": (EQ2 - s1m * s1m, _err((1.0, ev["Q2"]), (2.0 * np.abs(s1m), s1m_t),
                                           extra=4.0 * EPS * (np.abs(EQ2) + s1m * s1m))),
        "shift_d3": (EQ3 - 3.0 * EQ2 * s1m + 2.0 * s1m ** 3,
                     _err((1.0, ev["Q3"]), (3.0 * np.abs(s1m), ev["Q2"]), (3.0 * np.abs(EQ2) + 6.0 * s1m * s1m, s1m_t),
                          extra=4.0 * EPS * (np.abs(EQ3) + 3.0 * np.abs(EQ2 * s1m) + 2.0 * np.abs(s1m) ** 3))),
        "dtau": (ev["zL1"][0], _err((1.0, ev["zL1"]))),
    }
    d1_t, e2_t, cov_t = ev["d1"], ev["e2"], ev["cov"]
    pi, pj = cells.pi, cells.pj
    diag = pi == pj
    hess = cov_t[0] + np.where(diag, e2_t[0][pi], 0.0)
    out: dict = {"logl": ev["logl"]}
    if frozen:
        out.update({k: v[0] for k, v in mom.items()})
        zr = [np.zeros(cells.N)] * 2
        dparam = _param_derivs(cells, ev, d1_t[0], hess[diag], zr, zr)[0]
        out.update({"d1": d1_t[0], "hess": hess, "dparam": dparam})
        nan_c, nan_r = np.full(cells.C, np.nan), np.full(cells.N, np.nan)
        out["err"] = nan_c
        out["deriv_err"] = {"d1": nan_r, "hess": np.full(len(pi), np.nan), "shift_d1": nan_c, "shift_d2": nan_c,
                            "shift_d3": nan_c, "dtau": nan_c, "dparam": {n: nan_r for n in dparam}}
        out["deriv_ok"] = np.zeros(cells.C, dtype=bool)
        out["refine"] = np.zeros(cells.C, dtype=bool)
        return out
    lq, lr = ev["logl_e"][1], ev["logl_e"][2]
    out["err"] = lq + lr
    pos = tau > 0
    ti = np.where(pos, tau, 1.0)
    inf = np.where(pos, 0.0, np.inf)
    zb, Vv, k3 = ev["z"][0], ev["V"][0], ev["K3"][0]
    ez = _err((1.0, ev["z"]))
    e_ez2 = _err((1.0, ev["V"]), (2.0 * np.abs(zb), ev["z"]), extra=4.0 * EPS * np.maximum(Vv + zb * zb, 1.0))
    ibp = {
        "shift_d1": (zb / ti, (ez[0] / ti + inf, ez[1] / ti + inf)),
        "shift_d2": ((Vv - 1.0) / ti ** 2, tuple(e / ti ** 2 + inf for e in
                                                  _err((1.0, ev["V"]), extra=4.0 * EPS * np.maximum(Vv, 1.0)))),
        "shift_d3": (k3 / ti ** 3, tuple(e / ti ** 3 + inf for e in _err((1.0, ev["K3"]), (3.0 * Vv, ev["z"])))),
        "dtau": ((Vv + zb * zb - 1.0) / ti, tuple(e / ti + inf for e in e_ez2)),
    }
    chosen = {k: _pick(ibp[k], mom[k]) for k in mom}
    for k, (v, e) in chosen.items():
        out[k] = v
    d1 = d1_t[0].copy()
    e_d1 = list(_err((1.0, d1_t)))
    e_h = list(_err((1.0, cov_t)))
    e2e = _err((1.0, e2_t))
    for j in range(2):
        e_h[j] = e_h[j] + np.where(diag, e2e[j][pi], 0.0)
    single = cells.count[cells.cell] == 1                 # a cell of one reading: the shift quantities are its own
    cs = cells.cell[single]
    d1[single] = chosen["shift_d1"][0][cs]
    for j in range(2):
        e_d1[j][single] = chosen["shift_d1"][1][j][cs]
    sp = single[pi]                                       # the one pair (i, i) of such a cell
    hess[sp] = chosen["shift_d2"][0][cells.cell[pi[sp]]]
    for j in range(2):
        e_h[j][sp] = chosen["shift_d2"][1][j][cells.cell[pi[sp]]]
    dparam, e_par = _param_derivs(cells, ev, d1, hess[diag], e_d1, [e[diag] for e in e_h])
    out.update({"d1": d1, "hess": hess, "dparam": dparam})
    out["deriv_err"] = {"d1": e_d1[0], "hess": e_h[0], "dparam": {n: e[0] for n, e in e_par.items()},
                        **{k: chosen[k][1][0] for k in chosen}}

    loc, wid, rd = cells.walls()
    wmin = np.full(cells.C, np.inf)
    np.minimum.at(wmin, cells.cell[rd], wid)
    with np.errstate(divide="ignore"):
        D0 = 1.0 / np.maximum(tau, wmin)                  # the curvature scale of the cell: prior or narrowest feature

    def check(j: int) -> np.ndarray:
        """Every derivative of a cell within dtol of its scale, with the total (j = 0) or fixed (1) errors."""
        D = np.abs(out["shift_d1"]) + np.sqrt(np.abs(out["shift_d2"])) + D0
        ok = np.ones(cells.C, dtype=bool)
        for k in (1, 2, 3):
            ok &= chosen[f"shift_d{k}"][1][j] <= dtol * D ** k
        ok &= chosen["dtau"][1][j] <= dtol * D * (1.0 + np.abs(zb))
        Di = np.abs(d1) + np.sqrt(np.abs(hess[diag])) + D0[cells.cell]   # diag pairs are in reading order
        ok_r = e_d1[j] <= dtol * Di
        for name, e in e_par.items():
            theta = np.abs(cells.params[name]) if name in cells.params else 0.0
            ok_r &= e[j] <= dtol * (np.abs(dparam[name]) + Di + 3.0 * theta * Di * Di)
        ok_p = e_h[j] <= dtol * Di[pi] * Di[pj]
        ok &= np.bincount(cells.cell, ~ok_r, minlength=cells.C) == 0
        ok &= np.bincount(cells.cell[pi], ~ok_p, minlength=cells.C) == 0
        return ok

    with np.errstate(invalid="ignore"):
        out["deriv_ok"] = check(0)
        fixed_ok = check(1)
    out["refine"] = (~out["deriv_ok"] & fixed_ok) | ((lq > tol) & (lr <= tol))
    return out


class _Out:
    """Results per cell, per reading and per pair (sorted reading order), filled group of cells by group."""

    def __init__(self, cells: _Cells) -> None:
        C, N, P = cells.C, cells.N, len(cells.pi)
        self.cells = cells
        self.c = {k: np.full(C, np.nan) for k in ("logl", "err", "shift_d1", "shift_d2", "shift_d3", "dtau")}
        self.ce = {k: np.full(C, np.nan) for k in ("shift_d1", "shift_d2", "shift_d3", "dtau")}
        self.method = np.full(C, "", dtype=object)
        self.err_kind = np.full(C, "", dtype=object)
        self.deriv_ok = np.zeros(C, dtype=bool)
        self.d1, self.e_d1 = np.full(N, np.nan), np.full(N, np.nan)
        self.hess, self.e_hess = np.full(P, np.nan), np.full(P, np.nan)
        self.dparam = {n: np.full(N, np.nan) for n in PARAMS[cells.kind]}
        self.e_dparam = {n: np.full(N, np.nan) for n in PARAMS[cells.kind]}
        self.chunks: list = []
        self.last = np.full(C, -1, dtype=np.int64)

    def put(self, sub: _Cells, old: np.ndarray, rows: np.ndarray, fin: dict, method: str, err_kind: str,
            nodes: tuple) -> None:
        for k in self.c:
            self.c[k][old] = fin[k]
        for k in self.ce:
            self.ce[k][old] = fin["deriv_err"][k]
        self.method[old] = method
        self.err_kind[old] = err_kind
        self.deriv_ok[old] = fin["deriv_ok"]
        self.d1[rows] = fin["d1"]
        self.e_d1[rows] = fin["deriv_err"]["d1"]
        at = self.cells.pair_index(rows[sub.pi], rows[sub.pj])
        self.hess[at] = fin["hess"]
        self.e_hess[at] = fin["deriv_err"]["hess"]
        for n in self.dparam:
            self.dparam[n][rows] = fin["dparam"][n]
            self.e_dparam[n][rows] = fin["deriv_err"]["dparam"][n]
        z, lw, nc = nodes
        self.chunks.append((z, lw, old[nc]))
        self.last[old] = len(self.chunks) - 1

    def result(self, inv: np.ndarray, mode: np.ndarray, scale: np.ndarray, extra: dict | None = None) -> CellResult:
        keep = [(z[self.last[c] == i], lw[self.last[c] == i], c[self.last[c] == i])
                for i, (z, lw, c) in enumerate(self.chunks)]
        z, lw, nc = (np.concatenate([k[j] for k in keep]) if keep else np.zeros(0) for j in range(3))
        nc = nc.astype(np.int64)
        order = np.argsort(nc, kind="stable")
        z, lw, nc = z[order], lw[order], nc[order]
        cells = self.cells
        rule = {"z": z, "lw": lw, "cell": nc, "method": self.method, "mode": mode, "scale": scale}
        if extra:
            rule.update(extra)
        so = cells.order
        return CellResult(
            logl=self.c["logl"], err=self.c["err"], err_kind=self.err_kind, method=self.method,
            evals=np.bincount(nc, minlength=cells.C), mode=mode, scale=scale, d1=self.d1[inv],
            hess_i=so[cells.pi], hess_j=so[cells.pj], hess_v=self.hess, shift_d1=self.c["shift_d1"],
            shift_d2=self.c["shift_d2"], shift_d3=self.c["shift_d3"], dtau=self.c["dtau"],
            dparam={n: v[inv] for n, v in self.dparam.items()},
            deriv_err={"d1": self.e_d1[inv], "hess": self.e_hess, **self.ce,
                       "dparam": {n: v[inv] for n, v in self.e_dparam.items()}},
            deriv_ok=self.deriv_ok, rule=rule)


def cell_integral(kind: Any, m: Any, obs: Mapping[str, Any], params: Mapping[str, Any], cell: Any, tau: Any, *,
                  method: str = "auto", nodes: int = 9, low_nodes: int | None = None, tol: float = 1.0e-7,
                  dtol: float = 1.0e-6, refine: int = 2, n_cells: int | None = None,
                  rule: Mapping[str, Any] | None = None) -> CellResult:
    """Integrate the one-dimensional latent term of every cell; see the module docstring and `CellResult`."""
    kind = _kind_name(kind)
    if method not in ("auto", "agh", "composite"):
        raise ValueError(f"laplace_kernels: unknown method {method!r}; use auto, agh or composite")
    cells = _Cells(kind, m, obs, params, cell, tau, n_cells)
    C = cells.C
    inv = np.empty(cells.N, dtype=np.int64)
    inv[cells.order] = np.arange(cells.N)
    out = _Out(cells)
    every = np.arange(C)
    rows_all = np.arange(cells.N)
    if rule is not None:
        z, lw, nc = (np.asarray(rule[k]) for k in ("z", "lw", "cell"))
        z, lw, nc = z.astype(float), lw.astype(float), nc.astype(np.int64)
        fin = _finish(cells, _evaluate(cells, z, lw, nc, None), dtol, tol, frozen=True)
        out.put(cells, every, rows_all, fin, "frozen", "none", (z, lw, nc))
        z0 = np.asarray(rule.get("mode", np.full(C, np.nan)), dtype=float)
        sc = np.asarray(rule.get("scale", np.full(C, np.nan)), dtype=float)
        return out.result(inv, z0, sc)
    z0, h0, h2 = cells.mode()
    scale = 1.0 / np.sqrt(-h2)
    todo = np.ones(C, dtype=bool)
    ends = _ends(cells, z0, h0, scale)
    if method in ("auto", "agh"):
        za, lwa, nca = _agh_rule(z0, scale, nodes)
        low = _default_low(nodes) if low_nodes is None else int(low_nodes)
        hi = _evaluate(cells, za, lwa, nca, ends)
        lo = _evaluate(cells, *_agh_rule(z0, scale, low), ends)
        fin = _finish(cells, _agh_quad(hi, lo), dtol, tol)
        out.put(cells, every, rows_all, fin, "agh", "heuristic", (za, lwa, nca))
        if method == "agh":
            return out.result(inv, z0, scale)
        # AGH and its estimate miss a wall narrower than the curvature scale, even one out in the tail
        todo = ((fin["err"] > tol) | ~np.isfinite(fin["err"]) | ~fin["deriv_ok"]
                | (_sharpness(cells, scale, ends[0], ends[3]) < AGH_SMOOTH))
    split = 1
    for _ in range(refine + 1):
        if not todo.any():
            break
        sub, old, rows = cells.subset(todo)
        sub_ends = tuple(e[old] for e in ends)
        zc, lwk, lwg, ncc, panel = _composite_rule(sub, z0[old], h0[old], scale[old], sub_ends, split)
        fin = _finish(sub, _evaluate(sub, zc, lwk, ncc, sub_ends, lwg=lwg, panel=panel), dtol, tol)
        out.put(sub, old, rows, fin, "composite", "kronrod", (zc, lwk, ncc))
        todo = np.zeros(C, dtype=bool)
        todo[old[fin["refine"]]] = True
        split *= 2
    return out.result(inv, z0, scale)
