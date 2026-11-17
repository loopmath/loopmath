"""Likelihood kernels and one-dimensional cell integrals for the Laplace engine (numpy and scipy only).

Not a command: the Laplace engine (`engine_laplace.py`) calls `kernel`, `kernel_full` and `cell_integral`.

Kernels
-------
`kernel(kind, eta, obs, params) -> (logp, d1, d2)` gives, per observation, the log-likelihood and its first
and second derivatives in the linear predictor `eta`. `kernel_full` gives the same plus the third derivative
`d3` and, for every parameter of the kernel, the derivatives of `logp`, `d1` and `d2` in that parameter (what
the gradient of a Laplace marginal likelihood needs: the envelope term, the shift of the mode and the change
of the Hessian). All arrays broadcast against `eta`; the result has the broadcast shape. Every `logp` is a
normalised log probability (a log density for the Gaussian kinds), constants included.

`kind` is a name below, or the integer codes of `model.KIND_CODE` (0 linear, 1 binary, 2 levels, 3 items).

    kind        obs                       params               the reading, given eta
    linear      y                         sd                   y ~ N(eta, sd^2)            ("fraction" is the same)
    linear_cell n, ybar, ss               sd, tau              n calls y_c = eta + t + e_c, t ~ N(0, tau^2),
                                                               e_c ~ N(0, sd^2), entered through their mean
                                                               ybar and sum of squares ss around it (exact)
    binary      y (0 or 1)                sd                   1 with chance Phi(eta / sd)
    levels      lo, hi                    sd                   the level whose interval (lo, hi) holds
                                                               eta + e, e ~ N(0, sd^2); an open end is +-inf
                                                               or +-model.OPEN
    items       k, n                      sd                   k ~ Binomial(n, Phi(x)), x = eta / sqrt(1 + sd^2):
                                                               sd is the call noise integrated out (pass 0
                                                               when the noise is an explicit unknown in eta)
    pairs       v (0 a, 1 tie, 2 b), o    sd, width, position  z = eta + position o + e, e ~ N(0, sd^2); verdict
                (+1 a shown first, -1 b,                       a when z > width, b when z < -width, else tie.
                0 unknown)                                     eta is eta_a - eta_b: add d1 to a, subtract
                                                               from b; d2 enters (a,a), (b,b) and -d2 (a,b)

`KernelResult.dparam[name] = (dlogp, dd1, dd2)` with derivatives in the parameter as written (sd, not log sd
or the variance). A kernel parameter that is a function of the hyperparameters (for example a reading's sd
made of sigma, tau and the halo share) is the engine's to chain.

Numerics. Probit terms use `log Phi` (scipy's `log_ndtr`) and the ratio `lambda = phi / Phi` through `erfcx`,
never `log(ndtr)`. In the far lower tail (x < -10) the second and third derivatives cancel catastrophically in
their textbook forms (`-lambda (x + lambda)` loses all digits by x = -1e8), so there they come from the
asymptotic series of the Mills ratio, summed to 30 terms (truncation below 1e-19 at x = -10). An interval
probability is `Phi(alpha) - Phi(beta)` taken on the side of 0 where it keeps its digits, as
`log Phi(alpha) + log(1 - exp(log Phi(beta) - log Phi(alpha)))`. A very narrow interval (tie width or level
width below about 1e-6 of the noise sd) keeps only part of its relative precision in `d1`, `d2`.

Cell integrals
--------------
`cell_integral(kind, m, obs, params, cell, tau, ...)` integrates a one-dimensional latent term per cell: the
readings i of cell c have `eta_i = m_i + tau_c z`, z ~ N(0, 1) shared by the cell (the judge-by-output term of
repeated calls on one output, or the call noise of one `items` call). It returns, per cell,
`logl = log integral phi(z) prod_i p(obs_i | m_i + tau_c z) dz` with an error estimate, and the derivatives the
inner Newton step and the outer optimisation need (see `CellResult`).

Three methods:

- `agh`: adaptive Gauss-Hermite, centred at the mode of the integrand with the curvature scale, `nodes`
  points. The error estimate is the difference to the same rule with about half the points (`low_nodes`).
  It is accurate when the readings are smooth on the scale of the latent term, and it is not when the call
  noise is small against the term: the integrand is then a normal density cut by near-hard walls at the cut
  points, and a Gaussian-weight rule converges slowly (two top readings of a
  three-level scale with call noise a tenth of the term are off by 1.9 percent at 15 points).
- `composite`: Gauss-Kronrod (7, 15) panels over the interval where the log integrand is within `DROP` of its
  maximum, split at the mode, at the points where it has dropped by `SPLIT_DROPS`, and at every wall of the
  readings (a cut point, its location in z and a few wall widths either side). The error estimate is the sum
  over panels of |K15 - G7| plus a bound on the two tails outside the interval, which holds because the
  integrand is log-concave for every kernel here (the log of a Gaussian density plus log-concave terms).
- `auto` (default): `agh` for every cell, then `composite` for the cells whose `agh` estimate is above `tol`
  and for every cell with a wall narrower than `AGH_SMOOTH` curvature scales anywhere the integrand is not
  negligible: there both `agh` rules can miss the wall together (a wall five latent sds out, beyond the
  outermost node, left a 5e-7 error under a 6e-11 estimate).

Derivatives are exact for the quadrature with its nodes held where they were placed (`CellResult.rule`); pass
`rule=` to evaluate at other arguments with the same nodes. With nodes re-placed they agree to the size of
the quadrature error.

Pairs. A pair reads the terms of two cells (the judge's view of output a and of output b). When the repeated
verdicts of one judge on one pair of outputs are the only readings that touch those two terms, only their
difference enters, a normal with sd sqrt(tau_a^2 + tau_b^2): pass kind "pairs", m = eta_a - eta_b without the
two terms, and that sd as tau. A judge whose pairs share outputs couples the terms into a larger block, which
is not a one-dimensional integral and is not handled here.
"""

from __future__ import annotations

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

LOG_2PI = float(np.log(2.0 * np.pi))
SQRT2 = float(np.sqrt(2.0))
SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
OPEN_FROM = 1.0e5        # a cut at or beyond this magnitude is an open end (model.OPEN is 1e6): not a wall
TAIL = 10.0              # below -TAIL the derivatives of log Phi come from the asymptotic series
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


# ---- kernels ---------------------------------------------------------------------------------------------

@dataclass
class KernelResult:
    """Per observation: `logp`, its derivatives `d1`, `d2`, `d3` in eta, and `dparam[name] = (dlogp, dd1, dd2)`."""

    logp: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    d3: np.ndarray
    dparam: dict = field(default_factory=dict)


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
    return KernelResult(logp, d1, np.broadcast_to(d2, eta.shape).copy(), np.zeros_like(eta), {"sd": dsd})


def _gauss_cell(eta: np.ndarray, n: np.ndarray, ybar: np.ndarray, ss: np.ndarray, sd: np.ndarray,
                tau: np.ndarray) -> KernelResult:
    r = ybar - eta
    s2 = sd * sd
    V = s2 + n * tau * tau
    logp = -0.5 * n * LOG_2PI - (n - 1.0) * np.log(sd) - 0.5 * np.log(V) - 0.5 * ss / s2 - 0.5 * n * r * r / V
    d1 = n * r / V
    d2 = -n / V
    V2 = V * V
    dsd = (-(n - 1.0) / sd - sd / V + ss / (s2 * sd) + n * r * r * sd / V2, -2.0 * n * r * sd / V2, 2.0 * n * sd / V2)
    dtau = (-n * tau / V + n * n * r * r * tau / V2, -2.0 * n * n * r * tau / V2, 2.0 * n * n * tau / V2)
    return KernelResult(logp, d1, d2, np.zeros_like(eta), {"sd": dsd, "tau": dtau})


def _one_sided(x_sign: np.ndarray, eta: np.ndarray, scale: np.ndarray) -> tuple:
    """log Phi(sgn eta / scale) and derivatives in eta; derivatives in scale as (dlogp, dd1, dd2)."""
    x = x_sign * eta / scale
    l0, l1, l2, l3 = log_ndtr_derivs(x)
    d1 = x_sign * l1 / scale
    d2 = l2 / scale ** 2
    d3 = x_sign * l3 / scale ** 3
    dscale = (-x * l1 / scale, -x_sign * (x * l2 + l1) / scale ** 2, -(x * l3 + 2.0 * l2) / scale ** 3)
    return l0, d1, d2, d3, dscale


def _interval(x: np.ndarray, lo: np.ndarray, hi: np.ndarray, sd: np.ndarray) -> tuple[KernelResult, tuple, tuple]:
    """log P(lo < x + e < hi), e ~ N(0, sd^2). Returns the result (with dparam["sd"]) and the derivatives in
    lo and in hi, each (dlogp, dd1, dd2)."""
    with np.errstate(invalid="ignore"):
        alpha = (x - lo) / sd
        beta = (x - hi) / sd
    c = _interval_core(alpha, beta)
    d1, d2, d3 = c["T1"] / sd, c["T2"] / sd ** 2, c["T3"] / sd ** 3
    dlo = tuple(-c[f"Pa{k}"] / sd ** (k + 1) for k in range(3))
    dhi = tuple(-c[f"Pb{k}"] / sd ** (k + 1) for k in range(3))
    fa, fb = np.isfinite(alpha), np.isfinite(beta)
    dsd = []
    a_safe, b_safe = np.where(fa, alpha, 0.0), np.where(fb, beta, 0.0)
    for k, dk in enumerate((c["G"], d1, d2)):
        moved = a_safe * c[f"Pa{k}"] + b_safe * c[f"Pb{k}"]
        dsd.append((-k * dk if k else 0.0) / sd - moved / sd ** (k + 1))
    return KernelResult(c["G"], d1, d2, d3, {"sd": tuple(dsd)}), dlo, dhi


def kernel_full(kind: Any, eta: Any, obs: Mapping[str, Any], params: Mapping[str, Any]) -> KernelResult:
    """Log-likelihood per observation with derivatives in eta (up to the third) and in each parameter."""
    kind = _kind_name(kind)
    eta, o, p = _arrays(kind, eta, obs, params)
    if kind in ("linear", "fraction"):
        return _gauss(eta, o["y"], p["sd"])
    if kind == "linear_cell":
        return _gauss_cell(eta, o["n"], o["ybar"], o["ss"], p["sd"], p["tau"])
    if kind == "binary":
        sgn = np.where(o["y"] >= 0.5, 1.0, -1.0)
        l0, d1, d2, d3, dsd = _one_sided(sgn, eta, p["sd"])
        return KernelResult(l0, d1, d2, d3, {"sd": dsd})
    if kind == "levels":
        res, _, _ = _interval(eta, o["lo"], o["hi"], p["sd"])
        return res
    if kind == "items":
        k, n, sd = o["k"], o["n"], p["sd"]
        c = np.sqrt(1.0 + sd * sd)
        x = eta / c
        u0, u1, u2, u3 = log_ndtr_derivs(x)
        v0, v1, v2, v3 = log_ndtr_derivs(-x)
        logc = special.gammaln(n + 1.0) - special.gammaln(k + 1.0) - special.gammaln(n - k + 1.0)
        logp = logc + k * u0 + (n - k) * v0
        d1 = (k * u1 - (n - k) * v1) / c
        d2 = (k * u2 + (n - k) * v2) / c ** 2
        d3 = (k * u3 - (n - k) * v3) / c ** 3
        f = sd / (c * c)
        dsd = (-f * eta * d1, -f * (eta * d2 + d1), -f * (eta * d3 + 2.0 * d2))
        return KernelResult(logp, d1, d2, d3, {"sd": dsd})
    # pairs
    v, orient, w, sd, pos = o["v"], o["o"], p["width"], p["sd"], p["position"]
    x = eta + pos * orient
    lo = np.where(v == 0, w, np.where(v == 1, -w, -np.inf))
    hi = np.where(v == 0, np.inf, np.where(v == 1, w, -w))
    res, dlo, dhi = _interval(x, lo, hi, sd)
    dlo_w = np.where(v == 0, 1.0, np.where(v == 1, -1.0, 0.0))
    dhi_w = np.where(v == 0, 0.0, np.where(v == 1, 1.0, -1.0))
    dwidth = tuple(np.where(dlo_w != 0, dlo_w * dlo[k], 0.0) + np.where(dhi_w != 0, dhi_w * dhi[k], 0.0)
                   for k in range(3))
    res.dparam["width"] = dwidth
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


@dataclass
class CellResult:
    """What `cell_integral` returns. C cells, N readings (in the caller's order).

    logl[C]           log integral phi(z) prod_i p(obs_i | m_i + tau_c z) dz
    err[C]            estimated absolute error of logl (relative error of the integral)
    method[C]         "agh" or "composite", per cell
    evals[C]          integrand evaluations used (nodes of the returned rule)
    mode[C], scale[C] mode of the integrand in z and its curvature scale (-h'')^(-1/2)
    d1[N]             d logl / d m_i
    hess_i, hess_j, hess_v
                      d2 logl / d m_i d m_j for every ordered pair (i, j) of readings of one cell, i = j
                      included (reading indices in the caller's order)
    shift_d1, shift_d2, shift_d3 [C]
                      derivatives of logl in a shift added to every m_i of the cell
    dtau[C]           d logl / d tau_c
    dparam[name][N]   d logl / d params[name]_i (sum over the cell's readings for a shared parameter)
    rule              the nodes (z, log weight, cell) behind every number here; pass it back as `rule=`
    """

    logl: np.ndarray
    err: np.ndarray
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
        self.count = np.bincount(self.cell, minlength=C)
        self.start = np.concatenate([[0], np.cumsum(self.count)[:-1]]).astype(np.int64)

    def subset(self, which: np.ndarray) -> tuple["_Cells", np.ndarray]:
        """The cells in `which` only, renumbered from 0, and their old indices."""
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
        sub.count = np.bincount(sub.cell, minlength=sub.C)
        sub.start = np.concatenate([[0], np.cumsum(sub.count)[:-1]]).astype(np.int64)
        return sub, old

    def h(self, z: np.ndarray, *, need: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        """Locations in eta and widths of the features of each reading (cut points, centres), with the reading."""
        k, o, p = self.kind, self.obs, self.params
        idx = np.arange(self.N)
        if k in ("linear", "fraction"):
            return o["y"], p["sd"], idx
        if k == "linear_cell":
            return o["ybar"], p["sd"] / np.sqrt(np.maximum(o["n"], 1.0)), idx
        if k == "binary":
            return np.zeros(self.N), p["sd"], idx
        if k == "items":
            return np.zeros(self.N), np.sqrt(1.0 + p["sd"] ** 2), idx
        if k == "pairs":   # the verdict edges +-width, moved by the position effect
            w, v, shift = p["width"], o["v"], p["position"] * o["o"]
            lo = np.where(v == 0, w, np.where(v == 1, -w, -np.inf)) - shift
            hi = np.where(v == 0, np.inf, np.where(v == 1, w, -w)) - shift
            loc = np.concatenate([lo, hi])
        else:
            loc = np.concatenate([o["lo"], o["hi"]])
        wid = np.concatenate([p["sd"], p["sd"]])
        rd = np.concatenate([idx, idx])
        keep = np.isfinite(loc) & (np.abs(loc) < OPEN_FROM)
        return loc[keep], wid[keep], rd[keep]


def _rule_eval(cells: _Cells, z: np.ndarray, lw: np.ndarray, node_cell: np.ndarray, *, full: bool,
               gauss_lw: np.ndarray | None = None, panel: np.ndarray | None = None) -> dict:
    """Evaluate a rule (nodes z with log weights lw, sorted by cell) on every cell. Value only unless `full`."""
    C = cells.C
    K = np.bincount(node_cell, minlength=C)
    node_start = np.concatenate([[0], np.cumsum(K)[:-1]]).astype(np.int64)
    rep = K[cells.cell]
    total = int(rep.sum())
    r_idx = np.repeat(np.arange(cells.N), rep)
    offs = np.arange(total) - np.repeat(np.cumsum(rep) - rep, rep)
    n_idx = node_start[cells.cell[r_idx]] + offs
    eta = cells.m[r_idx] + cells.tau[cells.cell[r_idx]] * z[n_idx]
    obs = {k: v[r_idx] for k, v in cells.obs.items()}
    par = {k: v[r_idx] for k, v in cells.params.items()}
    kr = kernel_full(cells.kind, eta, obs, par)
    M = len(z)
    h = -0.5 * z * z - 0.5 * LOG_2PI + np.bincount(n_idx, kr.logp, minlength=M)
    t = lw + h
    mx = np.full(C, -np.inf)
    np.maximum.at(mx, node_cell, t)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    total_w = np.bincount(node_cell, np.exp(t - mx[node_cell]), minlength=C)
    with np.errstate(divide="ignore"):
        logl = mx + np.log(total_w)
    out = {"logl": logl, "h": h}
    if gauss_lw is not None:   # composite: per-panel |K15 - G7| relative to the integral
        rel = np.exp(h - logl[node_cell])
        diff = (np.exp(lw) - np.exp(gauss_lw)) * rel
        per_panel = np.abs(np.bincount(panel, diff))
        first = np.unique(panel, return_index=True)[1]
        out["kg_err"] = np.bincount(node_cell[first], per_panel[np.unique(panel)], minlength=C)
    if not full:
        return out
    w = np.exp(t - logl[node_cell])                        # posterior weight of each node
    wp = w[n_idx]                                          # ... per (reading, node)
    mu1 = np.bincount(r_idx, wp * kr.d1, minlength=cells.N)
    e2 = np.bincount(r_idx, wp * kr.d2, minlength=cells.N)
    c1 = kr.d1 - mu1[r_idx]
    # Hessian in m: E[d2_i] delta_ij + Cov(d1_i, d1_j), over ordered pairs (i, j) of one cell
    n_c = cells.count[cells.cell]
    pi = np.repeat(np.arange(cells.N), n_c)
    pj_off = np.arange(len(pi)) - np.repeat(np.cumsum(n_c) - n_c, n_c)
    pj = cells.start[cells.cell[pi]] + pj_off
    Kp = K[cells.cell[pi]]
    pair_rep = np.repeat(np.arange(len(pi)), Kp)
    k_off = np.arange(int(Kp.sum())) - np.repeat(np.cumsum(Kp) - Kp, Kp)
    # position of (reading, node) in the r_idx layout: block start of the reading + node offset
    block = np.concatenate([[0], np.cumsum(rep)[:-1]]).astype(np.int64)
    ai = block[pi[pair_rep]] + k_off
    aj = block[pj[pair_rep]] + k_off
    cov = np.bincount(pair_rep, wp[ai] * c1[ai] * c1[aj], minlength=len(pi))
    hess_v = cov + np.where(pi == pj, e2[pi], 0.0)
    # shift derivatives: cumulants of the summed derivatives under the node weights
    L1 = np.bincount(n_idx, kr.d1, minlength=M)
    L2 = np.bincount(n_idx, kr.d2, minlength=M)
    L3 = np.bincount(n_idx, kr.d3, minlength=M)
    m1 = np.bincount(node_cell, w * L1, minlength=C)
    m2 = np.bincount(node_cell, w * L2, minlength=C)
    m3 = np.bincount(node_cell, w * L3, minlength=C)
    c = L1 - m1[node_cell]
    k2 = np.bincount(node_cell, w * c * c, minlength=C)
    k3 = np.bincount(node_cell, w * c ** 3, minlength=C)
    cov12 = np.bincount(node_cell, w * c * (L2 - m2[node_cell]), minlength=C)
    out.update({
        "d1": mu1, "hess_i": pi, "hess_j": pj, "hess_v": hess_v,
        "shift_d1": m1, "shift_d2": m2 + k2, "shift_d3": m3 + 3.0 * cov12 + k3,
        "dtau": np.bincount(node_cell, w * z * L1, minlength=C),
        "dparam": {name: np.bincount(r_idx, wp * vals[0], minlength=cells.N) for name, vals in kr.dparam.items()},
    })
    return out


def _agh_rule(z0: np.ndarray, scale: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, w = hermite_rule(n)
    C = len(z0)
    z = (z0[:, None] + SQRT2 * scale[:, None] * x[None, :]).ravel()
    lw = (np.log(SQRT2 * scale)[:, None] + (np.log(w) + x * x)[None, :]).ravel()
    return z, lw, np.repeat(np.arange(C), n)


def _composite_rule(cells: _Cells, z0: np.ndarray, h0: np.ndarray, scale: np.ndarray,
                    which: np.ndarray, ends: tuple | None = None) -> tuple[np.ndarray, ...]:
    """Kronrod panels for the cells in `which`: nodes, Kronrod and Gauss log weights, cell, panel, tail bound.
    `ends` is `_ends(...)` when the caller has it already."""
    zl, hl, sl, zr, hr, sr = _ends(cells, z0, h0, scale) if ends is None else ends
    with np.errstate(divide="ignore", over="ignore"):
        tail = np.exp(hl - h0) / np.abs(sl) + np.exp(hr - h0) / np.abs(sr)   # times exp(h0); scaled below
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
    keep = which[pc] & np.isfinite(pz) & (pz >= zl[pc]) & (pz <= zr[pc])
    pz, pc = pz[keep], pc[keep]
    order = np.lexsort((pz, pc))
    pz, pc = pz[order], pc[order]
    same = pc[1:] == pc[:-1]
    span = (zr - zl)[pc[1:]]
    a, b, cc = pz[:-1][same], pz[1:][same], pc[1:][same]
    good = (b - a) > 1e-13 * np.maximum(span[same], 1e-300)
    a, b, cc = a[good], b[good], cc[good]
    x, wk, wg = kronrod_rule()
    half = 0.5 * (b - a)
    z = ((a + b)[:, None] * 0.5 + half[:, None] * x[None, :]).ravel()
    with np.errstate(divide="ignore"):
        lwk = (np.log(half)[:, None] + np.log(wk)[None, :]).ravel()
        lwg = (np.log(half)[:, None] + np.log(np.where(wg > 0, wg, 0.0))[None, :]).ravel()
    node_cell = np.repeat(cc, len(x))
    panel = np.repeat(np.arange(len(a)), len(x))
    return z, lwk, lwg, node_cell, panel, tail


def _ends(cells: _Cells, z0: np.ndarray, h0: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, ...]:
    """The two points where log f has dropped by DROP from its maximum, with log f and its slope at each."""
    return cells.level(z0, h0, scale, DROP, -1.0) + cells.level(z0, h0, scale, DROP, 1.0)


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


def cell_integral(kind: Any, m: Any, obs: Mapping[str, Any], params: Mapping[str, Any], cell: Any, tau: Any, *,
                  method: str = "auto", nodes: int = 9, low_nodes: int | None = None, tol: float = 1.0e-7,
                  n_cells: int | None = None, rule: Mapping[str, Any] | None = None) -> CellResult:
    """Integrate the one-dimensional latent term of every cell; see the module docstring and `CellResult`."""
    kind = _kind_name(kind)
    if method not in ("auto", "agh", "composite"):
        raise ValueError(f"laplace_kernels: unknown method {method!r}; use auto, agh or composite")
    cells = _Cells(kind, m, obs, params, cell, tau, n_cells)
    C = cells.C
    inv = np.empty(cells.N, dtype=np.int64)
    inv[cells.order] = np.arange(cells.N)
    if rule is not None:
        z, lw, nc = (np.asarray(rule[k]) for k in ("z", "lw", "cell"))
        res = _rule_eval(cells, z.astype(float), lw.astype(float), nc.astype(np.int64), full=True)
        err = np.asarray(rule.get("err", np.full(C, np.nan)), dtype=float)
        meth = np.asarray(rule.get("method", np.full(C, "given")), dtype=object)
        z0 = np.asarray(rule.get("mode", np.full(C, np.nan)), dtype=float)
        sc = np.asarray(rule.get("scale", np.full(C, np.nan)), dtype=float)
        return _result(cells, inv, res, err, meth, np.bincount(nc, minlength=C), z0, sc,
                       {"z": z, "lw": lw, "cell": nc, "err": err, "method": meth, "mode": z0, "scale": sc})
    z0, h0, h2 = cells.mode()
    scale = 1.0 / np.sqrt(-h2)
    method_c = np.full(C, "agh", dtype=object)
    err = np.zeros(C)
    if method in ("auto", "agh"):
        za, lwa, nca = _agh_rule(z0, scale, nodes)
        low = _default_low(nodes) if low_nodes is None else int(low_nodes)
        zb, lwb, ncb = _agh_rule(z0, scale, low)
        hi_v = _rule_eval(cells, za, lwa, nca, full=False)["logl"]
        lo_v = _rule_eval(cells, zb, lwb, ncb, full=False)["logl"]
        err = np.abs(hi_v - lo_v)
        use_comp = np.zeros(C, dtype=bool)
    else:
        use_comp = np.ones(C, dtype=bool)
        za = lwa = nca = None
    ends = None
    if method == "auto":
        # AGH and its estimate miss a wall narrower than the curvature scale, even one out in the tail
        ends = _ends(cells, z0, h0, scale)
        use_comp = (err > tol) | ~np.isfinite(err) | (_sharpness(cells, scale, ends[0], ends[3]) < AGH_SMOOTH)
    parts_z, parts_lw, parts_c = [], [], []
    if za is not None:
        keep = ~use_comp[nca]
        parts_z.append(za[keep])
        parts_lw.append(lwa[keep])
        parts_c.append(nca[keep])
    if use_comp.any():
        sub, old = cells.subset(use_comp)
        zs, hs, ss = z0[old], h0[old], scale[old]
        zc, lwk, lwg, ncc, panel, tail = _composite_rule(sub, zs, hs, ss, np.ones(sub.C, dtype=bool),
                                                         None if ends is None else tuple(e[old] for e in ends))
        ev = _rule_eval(sub, zc, lwk, ncc, full=False, gauss_lw=lwg, panel=panel)
        with np.errstate(over="ignore", invalid="ignore"):
            tail_rel = tail * np.exp(hs - ev["logl"])
        err[old] = ev["kg_err"] + tail_rel
        method_c[old] = "composite"
        parts_z.append(zc)
        parts_lw.append(lwk)
        parts_c.append(old[ncc])
    z = np.concatenate(parts_z)
    lw = np.concatenate(parts_lw)
    nc = np.concatenate(parts_c)
    order = np.argsort(nc, kind="stable")
    z, lw, nc = z[order], lw[order], nc[order]
    res = _rule_eval(cells, z, lw, nc, full=True)
    evals = np.bincount(nc, minlength=C)
    return _result(cells, inv, res, err, method_c, evals, z0, scale,
                   {"z": z, "lw": lw, "cell": nc, "err": err, "method": method_c, "mode": z0, "scale": scale})


def _result(cells: _Cells, inv: np.ndarray, res: dict, err: np.ndarray, method: np.ndarray, evals: np.ndarray,
            z0: np.ndarray, scale: np.ndarray, rule: dict) -> CellResult:
    order = cells.order
    return CellResult(
        logl=res["logl"], err=err, method=method, evals=evals, mode=z0, scale=scale,
        d1=res["d1"][inv], hess_i=order[res["hess_i"]], hess_j=order[res["hess_j"]], hess_v=res["hess_v"],
        shift_d1=res["shift_d1"], shift_d2=res["shift_d2"], shift_d3=res["shift_d3"], dtau=res["dtau"],
        dparam={k: v[inv] for k, v in res["dparam"].items()}, rule=rule)
