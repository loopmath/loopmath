"""Exact marginal likelihood of tiny `eval fit` designs: a test oracle for the Laplace engine.

A test helper, not part of the package. It reads the arrays of `model.Design.data` (the dict every
engine is given) and a parameter set `theta`, and computes by numerical integration

    log p(y | theta) = log of the integral, over every random effect x, of p(y | x, theta) p(x | theta)

with the posterior mean and sd of every random effect given y and theta. It imports numpy, scipy and,
to build tiny designs, `loopmath.eval.contracts` and `model`. It never imports
`engine_laplace`, `laplace_kernels` or `engine_pymc`, so it can check them.

    import laplace_oracle as O                      # tests/eval is on the path of its own tests
    built = O.tiny("binary", runs=2, judges=1, calls=2, seed=1)
    out = O.exact(built.data, built.theta)          # loglik, error, laplace, posterior, blocks
    O.exact_loglik(built.data, built.theta)         # the number alone

The model
---------
The one `engine_pymc` samples (its docstring, PRD section 7). Quality `q_r = mu_a + v_i + u_r`; a
judge's view of a cell `eta = q + bias + family + w_ji + tau sqrt(h) z_jt + part`; a reading adds call
noise of sd `sigma`, of which a `panel` call shares `sqrt(halo_call) sigma z_pc`. The layout of
`model.build` sums some terms out exactly (a `linear` cell enters through its mean and sum of squares;
a cell with one reading and no pair has its `part` in the reading's noise); this module follows the
layout, so it integrates exactly what an engine given the same arrays has to integrate.

Random effects (always integrated): `v` (inputs), `u` (runs), the shared halo factors `z_jt`, the
explicit cell terms `cell` (the `part` of a cell, in fitting units), the shared panel parts `z_pc`, the
call noise `e` of an `items` call of more than one item, the judge-by-input terms `ji`, and the family
effects `family` where the design has them on. Everything else is a value in `theta`. `random=` adds
`mu`, `bias` or `position` to the integrated set, with the priors of `engine_pymc` (arm means
N(mu_loc, (2 s)^2), centred over arms on a pairs-only dimension; bias raw N(0, s^2) minus its weighted
mean over `bias_w`; position N(0, pair_scale^2)).

theta
-----
A mapping with the names of `engine_pymc.OUTPUTS` where they are parameters, without draw axes:
`mu` (A, D), `sd_input` (D), `sd_run` (D), `corr_input` and `corr_run` (D, D), `bias` (J, D), `tau`
(J, D), `sigma` (J, D), `halo_target` (J), `halo_call` (J), `sd_judge_input` (J, D), `sigma_pair`
(J, D), `position` (J), `tie_width` (J); and `family_sd` (J), each judge's sd of its family effects
relative to the scale of each dimension: on dimension d the sd is `family_sd[j] * scale[d]` (the
effects themselves are random). A scalar is broadcast; a name left out takes `default_theta`. Values are
used as given: the oracle does not impose the units of `model.py` (sum-to-zero biases, an anchor's
`sigma` of 1); the engine's own parameterisation has to map onto these values.

The log-likelihood is normalised: it keeps the -log(2 pi)/2 of every Gaussian reading and the log
binomial coefficient of every `items` reading, which `engine_pymc`'s potentials drop.
`pymc_constant(data)` is that difference: potentials summed with the random effects integrated out
equal `exact_loglik - pymc_constant`.

Method
------
1. Every random effect is a linear form `c + a.z` in standard normals `z` (p of them).
2. Gaussian readings are integrated in closed form: their joint normal density, and `z` given them is
   normal (`gaussian="quadrature"` puts them in step 3 instead, as a check of the quadrature).
3. Every other reading is a factor of one linear form `w = c + a.z`: the probability that a normal
   with mean `w` and sd `s` falls in `(lo, hi)` (`binary`, `levels`, pair verdicts), or the
   binomial-probit term `C(n, k) Phi(w/s)^k Phi(-w/s)^(n-k)` (`items`). Factors whose forms are
   correlated given the Gaussian readings form a block; the integral is the product over blocks.
   The call noise `e` of an `items` call of n > 1 items touches one factor only, so it can be summed
   out inside that factor: `log of the integral over e ~ N(0, 1) of Phi(w + t e)^k Phi(-w - t e)^(n-k)`
   per value of w (`noisy_binomial`). That is still log-concave in w. The inner rule is composite
   Gauss-Legendre over e* +- 10 (e* the mode; the integrand there is below e^-50 of its top), with
   breakpoints at e* + 2^j sd for j = 0..24 each side and where the probit argument w + t e crosses
   0, +-2, .., +-10, and panels halved until two successive values agree to the inner tolerance (it
   refuses after `INNER_LEVELS`). Its error is the last change, and it enters the block's error.
   `items="summed"` does this for every such call, `items="explicit"` keeps `e` as a direction of the
   block, and `items="auto"` (the default) sums only when an explicit block would span more than
   `SUM_DIM` directions. The Laplace value is always that of the explicit layout, and `posterior`
   reports `e` only when it stayed explicit.
4. Decisions about the layout are made in units of each factor's own scale (its `s`, or
   `sqrt(1 + t^2)` for a summed call), so a common change of units changes nothing. Two factors are
   linked when the correlation of their forms passes `LINK_TOL`; a block keeps the directions whose
   standardised singular value passes `DROP_STD`. What the dropped directions and the cut links could
   move, to second order at the mode, is added to the block's error.
5. A block whose forms span d directions is integrated over those d coordinates (the rest of `z` does
   not touch it). Its coordinates are sharp where the standardised singular value is at least
   `SHARP_AT` (the factors change over a small part of the prior sd there).
   - One direction: scipy's adaptive quad over the range above e^-80 of the peak.
   - No sharp coordinate (or `method="gh"`): tensor-product Gauss-Hermite centred at the mode and
     scaled by the curvature there, the node count per coordinate through `NODES`.
   - One or two sharp coordinates (or `method="nested"`): the sharp ones by adaptive line rules with
     breakpoints at the mode (graded out geometrically) and where each factor turns, and in a plane
     at the vertices where two factors turn; the range cut where the log-concave integrand is 70
     below its top. The other coordinates by Gauss-Hermite from the Laplace marginal of those
     coordinates, through `NODES`. More than two sharp coordinates go to Gauss-Hermite in `auto`
     (which may refuse) and are refused by `nested`.
   The node count stops when the last two changes of log p plus the largest inner error are within
   what is left of the block's share of `tol` (below). With 1 node the rule is the Laplace
   approximation of the same integral, returned as `laplace`.
   The tolerance is global: `tol` bounds the reported `error`, which is the sum of every block's
   error (its rule's convergence term, its inner error, what its dropped directions could move and a
   roundoff allowance of 16 ulps a unit of the log terms it sums), the flat parts' errors (dropped
   directions, inner error, roundoff), the first-order effect of the links cut at LINK_TOL and the
   closed-form part's roundoff. `tol` is split in equal shares over the connected parts plus one for
   the links and the closed form; a block keeps more directions when its dropped ones would take more
   than DROP_SHARE of its share, its rule is asked for what is left after the dropped and roundoff
   terms, and the sum is checked against `tol` before anything is returned. A request that cannot be
   met (a share smaller than the roundoff of a value of size 5000, a rule that does not settle, a quad
   warning) raises `OracleNotConverged`; nothing is returned with an error above `tol`.
6. Posterior moments come from the same nodes, mapped back through steps 1 and 2.

A block of more than `MAX_DIM` directions is refused (`OracleTooLarge`), as is a block that does not
converge within `max_nodes` nodes, an inner rule that cannot reach its tolerance or a total error
above `tol` (`OracleNotConverged`). Fixed `nodes=` is the one exception: a diagnostic with no
convergence loop and no refusal, whose `error` is NaN when a block ran a single node count.

All data made here is synthetic.
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy import integrate, linalg, special
from scipy.sparse.csgraph import connected_components

from loopmath.eval import contracts as C
from loopmath.eval import model as M

LINEAR, BINARY, LEVELS, ITEMS = M.LINEAR, M.BINARY, M.LEVELS, M.ITEMS
INTERVAL, BINOMIAL, GAUSS, NOISY = 0, 1, 2, 3  # factor kinds (NOISY: binomial-probit, call noise summed out)
NODES = (3, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257)   # numpy's hermegauss fails at 513
MAX_DIM = 8
SUM_DIM = 4                                   # items="auto" sums item call noise out above this block size
GL_NODES = 8                                  # Gauss-Legendre nodes per panel in the breakpoint rules
INNER_LEVELS = 8                              # panel halvings a breakpoint rule may try before it refuses
INNER_TOL = 1.0e-12                           # relative accuracy asked of a NOISY factor by default
LOG_2PI = math.log(2.0 * math.pi)
DROP_STD = 1.0e-7                             # standardised singular values at or below this are dropped
SHARP_AT = 3.0                                # standardised singular values from this are sharp directions
NESTED_NODES = 2500                           # Gauss-Hermite nodes over the smooth directions of a nested block
# Breakpoints of the adaptive line rules. Around a mode: 0 and +-2^j local sds, a geometric grading, so
# no segment is much longer than the decay of a log-concave integrand at its inner end (two rules that
# both step over a narrow tail would agree on nothing). Around a factor's turn, in its widths: past 8 a
# probit is within 1e-15 of 0 or 1.
STEPS_MODE = (0.0,) + tuple(sign * 2.0 ** j for j in range(25) for sign in (-1.0, 1.0))
STEPS_WALL = (-8.0, -5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0, 8.0)
LINK_TOL = 1.0e-12                            # relative covariance below this does not link two factors
ROUND = 16.0 * np.finfo(float).eps            # roundoff allowance per unit of the log terms a value sums
DROP_SHARE = 0.25                             # of a part's share of tol, what dropped directions may take
RANDOM_EXTRAS = ("mu", "bias", "position")
CHUNK = 1 << 17


class OracleTooLarge(ValueError):
    """A block of the design spans more directions than dense quadrature can take."""


class OracleNotConverged(RuntimeError):
    """The node count reached `max_nodes` before two successive changes fell below `tol`."""


# ---------------------------------------------------------------------------------------------------
# theta

def sizes(data: Mapping[str, Any]) -> dict:
    return {name: int(data[name]) for name in ("A", "D", "J", "F", "I", "R")}


def shapes(data: Mapping[str, Any]) -> dict:
    s = sizes(data)
    A, D, J = s["A"], s["D"], s["J"]
    return {"mu": (A, D), "sd_input": (D,), "sd_run": (D,), "corr_input": (D, D), "corr_run": (D, D),
            "bias": (J, D), "tau": (J, D), "sigma": (J, D), "halo_target": (J,), "halo_call": (J,),
            "family_sd": (J,), "sd_judge_input": (J, D), "sigma_pair": (J, D), "position": (J,),
            "tie_width": (J,)}


def default_theta(data: Mapping[str, Any]) -> dict:
    """Plausible values at the scale of each dimension: the centre of the priors, sds at half the scale."""
    s = sizes(data)
    A, D, J = s["A"], s["D"], s["J"]
    scale = np.asarray(data["scale"], dtype=float)
    return {"mu": np.broadcast_to(np.asarray(data["mu_loc"], dtype=float), (A, D)).copy(),
            "sd_input": 0.5 * scale, "sd_run": 0.5 * scale, "corr_input": np.eye(D), "corr_run": np.eye(D),
            "bias": np.zeros((J, D)), "tau": np.broadcast_to(0.5 * scale, (J, D)).copy(),
            "sigma": np.broadcast_to(0.5 * scale, (J, D)).copy(), "halo_target": np.full(J, 0.3),
            "halo_call": np.full(J, 0.3), "family_sd": np.full(J, 0.4),
            "sd_judge_input": np.broadcast_to(0.3 * scale, (J, D)).copy(),
            "sigma_pair": np.broadcast_to(scale, (J, D)).copy(), "position": np.zeros(J),
            "tie_width": np.full(J, 0.3)}


def full_theta(data: Mapping[str, Any], theta: Mapping[str, Any] | None) -> dict:
    """`theta` with every name present at its shape: scalars broadcast, missing names from the defaults."""
    out = default_theta(data)
    want = shapes(data)
    for name, value in (theta or {}).items():
        if name not in want:
            raise KeyError(f"unknown theta name {name!r}; known: {', '.join(want)}")
        arr = np.asarray(value, dtype=float)
        if arr.shape != want[name]:
            arr = np.broadcast_to(arr, want[name]).copy()
        out[name] = arr
    return out


# ---------------------------------------------------------------------------------------------------
# the problem: random effects as linear forms in standard normals, readings as factors

@dataclass
class Problem:
    p: int
    effects: dict                                  # name -> (offset array, matrix with a last axis of p)
    g_rows: np.ndarray                             # Gaussian readings: w = g_off + g_rows z, y ~ N(w, g_var)
    g_off: np.ndarray
    g_y: np.ndarray
    g_var: np.ndarray
    f_rows: np.ndarray                             # other readings: one factor of w = f_off + f_rows z each
    f_off: np.ndarray
    f_kind: np.ndarray
    f_lo: np.ndarray
    f_hi: np.ndarray
    f_s: np.ndarray
    f_k: np.ndarray
    f_n: np.ndarray
    const: float                                   # log terms that hold no random effect


def _sqrt_cov(sd: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """A square root L of diag(sd) corr diag(sd) (L L' = cov), safe for a variance of 0."""
    cov = sd[:, None] * np.asarray(corr, dtype=float) * sd[None, :]
    vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
    return vecs * np.sqrt(np.clip(vals, 0.0, None))[None, :]


def build_problem(data: Mapping[str, Any], theta: Mapping[str, Any] | None = None, *,
                  random: Sequence[str] = (), gaussian: str = "closed", items: str = "explicit") -> Problem:
    """The design as standard normals, linear forms and factors. See the module docstring."""
    th = full_theta(data, theta)
    if items not in ("explicit", "summed"):
        raise ValueError("items= is 'explicit' or 'summed' here")
    for name in random:
        if name not in RANDOM_EXTRAS:
            raise ValueError(f"random= takes {', '.join(RANDOM_EXTRAS)}, not {name!r}")
    if gaussian not in ("closed", "quadrature"):
        raise ValueError("gaussian= is 'closed' or 'quadrature'")
    s = sizes(data)
    A, D, J, N, I, R = s["A"], s["D"], s["J"], s["F"], s["I"], s["R"]
    ints = {name: int(data[name]) for name in ("n_jt", "n_cell", "n_pc", "n_e", "n_ji")}
    family_on = np.asarray(data["family_on"], dtype=bool).reshape(J, D) & (N >= 2)
    has_pairs = len(data["p_ra"]) > 0

    count = 0

    def take(*shape: int) -> np.ndarray:
        nonlocal count
        n = int(np.prod(shape)) if shape else 1
        out = np.arange(count, count + n).reshape(shape)
        count += n
        return out

    zv, zu = take(I, D), take(R, D)
    zjt, zcell, zpc, ze, zji = (take(ints[k]) for k in ("n_jt", "n_cell", "n_pc", "n_e", "n_ji"))
    zfam = take(J, N, D) if family_on.any() else None
    zmu = take(A, D) if "mu" in random else None
    zbias = take(J, D) if "bias" in random else None
    zpos = take(J) if ("position" in random and has_pairs) else None
    p = count
    scale = np.asarray(data["scale"], dtype=float)

    # quality
    v_T = np.zeros((I, D, p))
    u_T = np.zeros((R, D, p))
    root_in = _sqrt_cov(th["sd_input"], th["corr_input"])
    root_run = _sqrt_cov(th["sd_run"], th["corr_run"])
    for i in range(I):
        v_T[i][:, zv[i]] = root_in
    for r in range(R):
        u_T[r][:, zu[r]] = root_run
    mu_T = np.zeros((A, D, p))
    if zmu is None:
        mu_off = th["mu"].copy()
    else:
        mu_off = np.broadcast_to(np.asarray(data["mu_loc"], dtype=float), (A, D)).copy()
        only = np.asarray(data["pairs_only"], dtype=bool)
        centre = np.eye(A) - np.full((A, A), 1.0 / A)
        for d in range(D):
            mix = centre if only[d] else np.eye(A)
            mu_off[:, d] = mix @ mu_off[:, d]
            for a in range(A):
                mu_T[a, d, zmu[:, d]] = 2.0 * scale[d] * mix[a]
    run_arm = np.asarray(data["run_arm"], dtype=np.int64)
    run_input = np.asarray(data["run_input"], dtype=np.int64)
    q_off = mu_off[run_arm]
    q_T = mu_T[run_arm] + v_T[run_input] + u_T

    # judge terms
    b_T = np.zeros((J, D, p))
    if zbias is None:
        b_off = th["bias"].copy()
    else:
        b_off = np.zeros((J, D))
        w = np.asarray(data["bias_w"], dtype=float)
        for d in range(D):
            for j in range(J):
                b_T[j, d, zbias[:, d]] = scale[d] * ((np.arange(J) == j) - w[:, d])
    fam_T = np.zeros((J, max(N, 1), D, p))
    if zfam is not None:
        for j in range(J):
            for d in range(D):
                if family_on[j, d]:
                    for f in range(N):
                        fam_T[j, f, d, zfam[j, :, d]] = th["family_sd"][j] * scale[d] * ((np.arange(N) == f) - 1.0 / N)
    sd_ji = th["sd_judge_input"]
    ji_j, ji_d = np.asarray(data["ji_j"], dtype=np.int64), np.asarray(data["ji_d"], dtype=np.int64)
    halo_on = np.asarray(data["halo_on"], dtype=bool)
    h = np.where(halo_on, th["halo_target"], 0.0)
    root_h = np.sqrt(h)
    tau, sigma = th["tau"], th["sigma"]
    own = tau ** 2 * (1.0 - h[:, None])
    root_call = np.sqrt(th["halo_call"])
    cell_j, cell_d = np.asarray(data["cell_j"], dtype=np.int64), np.asarray(data["cell_d"], dtype=np.int64)
    cell_suff = np.asarray(data["cell_suff"], dtype=np.int64)
    linked = {int(cell_suff[c]): c for c in range(len(cell_suff)) if cell_suff[c] >= 0}

    def view(r: int, d: int, j: int, f: int, jt: int, ji: int, reading: bool) -> tuple[float, np.ndarray]:
        c, a = float(q_off[r, d]), q_T[r, d].copy()
        if reading:
            c += float(b_off[j, d])
            a += b_T[j, d]
            if ji >= 0:
                a[zji[ji]] += sd_ji[ji_j[ji], ji_d[ji]]
        if zfam is not None:
            a += fam_T[j, f, d]
        if jt >= 0:
            a[zjt[jt]] += tau[j, d] * root_h[j]
        return c, a

    g: dict[str, list] = {"rows": [], "off": [], "y": [], "var": []}
    fac: dict[str, list] = {name: [] for name in ("rows", "off", "kind", "lo", "hi", "s", "k", "n")}
    const = 0.0

    def gaussian_reading(c: float, a: np.ndarray, y: float, var: float) -> None:
        if gaussian == "closed":
            for name, value in (("rows", a), ("off", c), ("y", y), ("var", var)):
                g[name].append(value)
        else:
            factor(c, a, GAUSS, y, 0.0, math.sqrt(var), 0, 0)

    def factor(c: float, a: np.ndarray, kind: int, lo: float, hi: float, sd: float, k: int, n: int) -> None:
        for name, value in (("rows", a), ("off", c), ("kind", kind), ("lo", lo), ("hi", hi), ("s", sd), ("k", k),
                            ("n", n)):
            fac[name].append(value)

    def bound(x: float) -> float:
        return -math.inf if x <= -M.OPEN / 2 else (math.inf if x >= M.OPEN / 2 else float(x))

    # linear cells by their sufficient statistics
    for n in range(len(data["s_r"])):
        r, d, j = int(data["s_r"][n]), int(data["s_d"][n]), int(data["s_j"][n])
        c, a = view(r, d, j, int(data["s_f"][n]), int(data["s_jt"][n]), int(data["s_ji"][n]), True)
        calls, s2 = int(data["s_n"][n]), float(sigma[j, d]) ** 2
        if n in linked:
            a[zcell[linked[n]]] += math.sqrt(own[j, d])
            var = s2 / calls
        else:
            var = own[j, d] + s2 / calls
        gaussian_reading(c, a, float(data["s_ybar"][n]), var)
        if calls > 1:
            const += -0.5 * (calls - 1) * (LOG_2PI + math.log(s2)) - 0.5 * math.log(calls) \
                - 0.5 * float(data["s_ss"][n]) / s2

    # readings one by one
    e_sd = np.zeros(ints["n_e"])
    for n in range(len(data["g_r"])):
        kind = int(data["g_kind"][n])
        r, d, j = int(data["g_r"][n]), int(data["g_d"][n]), int(data["g_j"][n])
        c, a = view(r, d, j, int(data["g_f"][n]), int(data["g_jt"][n]), int(data["g_ji"][n]), True)
        cell, pc, e = int(data["g_cell"][n]), int(data["g_pc"][n]), int(data["g_e"][n])
        if cell >= 0:
            a[zcell[cell]] += math.sqrt(own[j, d])
        s2 = float(sigma[j, d]) ** 2
        if pc >= 0:
            a[zpc[pc]] += float(sigma[j, d]) * root_call[j]
            s2 *= 1.0 - float(th["halo_call"][j])
        var = s2 + (0.0 if cell >= 0 else float(own[j, d]))
        y, lo, hi = float(data["g_y"][n]), bound(data["g_lo"][n]), bound(data["g_hi"][n])
        k, nn = int(data["g_k"][n]), int(data["g_n"][n])
        if kind == LINEAR:
            gaussian_reading(c, a, y, var)
        elif kind == BINARY:
            factor(c, a, INTERVAL, *((0.0, math.inf) if y >= 0.5 else (-math.inf, 0.0)), math.sqrt(var), 0, 0)
        elif kind == LEVELS:
            factor(c, a, INTERVAL, lo, hi, math.sqrt(var), 0, 0)
        elif kind == ITEMS:
            const += float(special.gammaln(nn + 1) - special.gammaln(k + 1) - special.gammaln(nn - k + 1))
            if e >= 0 and items == "summed":
                factor(c, a, NOISY, 0.0, 0.0, math.sqrt(var), k, nn)
            elif e >= 0:
                a[ze[e]] += math.sqrt(var)
                e_sd[e] = math.sqrt(var)
                factor(c, a, BINOMIAL, 0.0, 0.0, 1.0, k, nn)
            else:
                factor(c, a, BINOMIAL, 0.0, 0.0, math.sqrt(1.0 + var), k, nn)
        else:
            raise ValueError(f"unknown reading kind {kind}")

    # pairs
    pos_T = np.zeros((J, p))
    if zpos is not None:
        pair_scale = np.asarray(data["pair_scale"], dtype=float)
        for j in range(J):
            pos_T[j, zpos[j]] = pair_scale[j]
        pos_off = np.zeros(J)
    else:
        pos_off = th["position"]
    for n in range(len(data["p_ra"])):
        d, j = int(data["p_d"][n]), int(data["p_j"][n])
        ca, aa = view(int(data["p_ra"][n]), d, j, int(data["p_fa"][n]), int(data["p_jta"][n]), -1, False)
        cb, ab = view(int(data["p_rb"][n]), d, j, int(data["p_fb"][n]), int(data["p_jtb"][n]), -1, False)
        aa[zcell[int(data["p_ca"][n])]] += math.sqrt(own[j, d])
        ab[zcell[int(data["p_cb"][n])]] += math.sqrt(own[j, d])
        o = float(data["p_o"][n])
        c, a = ca - cb + pos_off[j] * o, aa - ab + pos_T[j] * o
        tie = float(th["tie_width"][j])
        lo, hi = {0: (tie, math.inf), 1: (-tie, tie), 2: (-math.inf, -tie)}[int(data["p_v"][n])]
        factor(c, a, INTERVAL, lo, hi, float(th["sigma_pair"][j, d]), 0, 0)

    # effects to report
    def rows_at(index: np.ndarray, coef: np.ndarray) -> np.ndarray:
        out = np.zeros(index.shape + (p,))
        for pos in np.ndindex(index.shape):
            out[pos + (int(index[pos]),)] = coef[pos]
        return out

    effects = {"v": (np.zeros((I, D)), v_T), "u": (np.zeros((R, D)), u_T), "q_run": (q_off, q_T),
               "cell": (np.zeros(len(zcell)), rows_at(zcell, np.sqrt(own[cell_j, cell_d]) if len(zcell) else zcell)),
               "z_jt": (np.zeros(len(zjt)), rows_at(zjt, np.ones(len(zjt)))),
               "z_pc": (np.zeros(len(zpc)), rows_at(zpc, np.ones(len(zpc)))),
               "e": (np.zeros(len(ze)), rows_at(ze, e_sd)),
               "ji": (np.zeros(len(zji)), rows_at(zji, sd_ji[ji_j, ji_d] if len(zji) else zji))}
    if items == "summed":
        del effects["e"]                           # summed out inside its factor
    if zfam is not None:
        effects["family"] = (np.zeros((J, N, D)), fam_T[:, :N])
    if zmu is not None:
        effects["mu"] = (mu_off, mu_T)
    if zbias is not None:
        effects["bias"] = (b_off, b_T)
    if zpos is not None:
        effects["position"] = (pos_off, pos_T)

    def arr(values: list, width: int | None = None) -> np.ndarray:
        if width is not None:
            return np.asarray(values, dtype=float).reshape(len(values), width)
        return np.asarray(values, dtype=float)

    return Problem(p=p, effects=effects, g_rows=arr(g["rows"], p), g_off=arr(g["off"]), g_y=arr(g["y"]),
                   g_var=arr(g["var"]), f_rows=arr(fac["rows"], p), f_off=arr(fac["off"]),
                   f_kind=np.asarray(fac["kind"], dtype=np.int64), f_lo=arr(fac["lo"]), f_hi=arr(fac["hi"]),
                   f_s=arr(fac["s"]), f_k=arr(fac["k"]), f_n=arr(fac["n"]), const=const)


# ---------------------------------------------------------------------------------------------------
# factors: log value and first and second derivative in w (written here, independent of the kernels)

def _log_phi(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return -0.5 * x * x - 0.5 * LOG_2PI


def log_interval(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log(Phi(b) - Phi(a)) for a < b, on the side of 0 where the difference keeps its digits."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.empty(a.shape)
    open_hi = np.isposinf(b)                  # one-sided: one log_ndtr, which keeps its digits on its own
    open_lo = np.isneginf(a) & ~open_hi
    both = ~(open_hi | open_lo)
    out[open_hi] = special.log_ndtr(-a[open_hi])
    out[open_lo] = special.log_ndtr(b[open_lo])
    a, b = a[both], b[both]
    upper = a > 0
    hi_tail = np.where(upper, -a, b)          # the larger of the two Phi values, as a Phi argument
    lo_tail = np.where(upper, -b, a)
    big = special.log_ndtr(hi_tail)
    small = special.log_ndtr(lo_tail)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[both] = big + np.log(-np.expm1(small - big))
    return out


def _binomial_g(x: np.ndarray, k: np.ndarray, n: np.ndarray):
    """k log Phi(x) + (n - k) log Phi(-x) and its first two derivatives in x."""
    up, down = special.log_ndtr(x), special.log_ndtr(-x)
    lam1, lam0 = np.exp(_log_phi(x) - up), np.exp(_log_phi(x) - down)
    return (k * up + (n - k) * down, k * lam1 - (n - k) * lam0,
            -k * lam1 * (x + lam1) - (n - k) * lam0 * (lam0 - x))


@functools.lru_cache(maxsize=16)
def _panel_rule(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on (0, 1), the unit cut into 2**level panels of GL_NODES each."""
    x, w = leggauss(GL_NODES)
    parts = 2 ** level
    frac = ((np.arange(parts)[:, None] + 0.5 * (x[None, :] + 1.0)) / parts).ravel()
    return frac, np.tile(0.5 * w / parts, parts)


def breakpoint_rule(P: np.ndarray, evaluate, top: np.ndarray, tol: float, what: str, q: int,
                    chunk: int = 1 << 20):
    """Integrals over (P[:, 0], P[:, -1]) of exp(L - top) per row, with breakpoints P (rows x B, sorted).

    `evaluate(E, R)` gives `(L, G, err)` at the points E of the rows R (flat arrays, R non-decreasing):
    the log integrand, q weight functions (a sequence of arrays like E) whose weighted integrals come
    back too, and the relative error of each integrand value (or None when the integrand is exact).
    Each segment of nonzero width between breakpoints gets Gauss-Legendre panels of GL_NODES nodes; the
    panels are halved until two successive values of every row agree to `tol` relative, and the finer
    one is kept. Returns `(value, G integrals (q x rows), error)`, the error of a row being its last
    change plus the largest integrand error met in it. Raises `OracleNotConverged` when a row has not
    settled after INNER_LEVELS halvings. `chunk` bounds the points of one `evaluate` call."""
    rows = P.shape[0]
    width = np.diff(P, axis=1)
    seg_r, seg_j = np.nonzero(width > 0)
    seg_a, seg_w = P[seg_r, seg_j], width[seg_r, seg_j]
    inner = np.zeros(rows)

    def at(level: int, todo: np.ndarray):
        frac, wts = _panel_rule(level)
        pick = np.flatnonzero(todo[seg_r])
        value, out = np.zeros(rows), np.zeros((q, rows))
        per = max(1, chunk // len(frac))
        for start in range(0, len(pick), per):
            seg = pick[start:start + per]
            E = (seg_a[seg, None] + seg_w[seg, None] * frac[None, :]).ravel()
            W = (seg_w[seg, None] * wts[None, :]).ravel()
            R = np.repeat(seg_r[seg], len(frac))
            L, G, err = evaluate(E, R)
            Pw = W * np.exp(L - top[R])
            r0, r1 = int(R[0]), int(R[-1]) + 1
            value[r0:r1] += np.bincount(R - r0, Pw, minlength=r1 - r0)
            for j in range(q):
                out[j, r0:r1] += np.bincount(R - r0, G[j] * Pw, minlength=r1 - r0)
            if err is not None:
                first = np.flatnonzero(np.r_[True, R[1:] != R[:-1]])
                inner[R[first]] = np.maximum(inner[R[first]], np.maximum.reduceat(err, first))
        return value, out

    todo = np.ones(rows, dtype=bool)
    v_prev, _ = at(0, todo)
    value, G, err = np.zeros(rows), np.zeros((q, rows)), np.full(rows, np.inf)
    for level in range(1, INNER_LEVELS + 1):
        v_new, g_new = at(level, todo)
        idx = np.flatnonzero(todo)
        change = np.abs(v_new[idx] - v_prev[idx]) / np.where(v_new[idx] > 0, v_new[idx], np.nan)
        value[idx], G[:, idx], err[idx] = v_new[idx], g_new[:, idx], change
        v_prev = v_new
        todo[idx[change <= tol]] = False
        if not todo.any():
            return value, G, err + inner
    raise OracleNotConverged(f"{what}: {int(todo.sum())} of {rows} integrals still change by up to "
                             f"{float(np.nanmax(err[todo])):.3g} after {INNER_LEVELS} halvings (tol {tol:.3g})")


def _bracketed_root(grad, lo: np.ndarray, hi: np.ndarray, start: np.ndarray, what: str) -> np.ndarray:
    """The root of a decreasing function per row: Newton steps kept inside the bracket (lo, hi), a
    bisection step otherwise. `grad(x)` returns (value, derivative)."""
    x = np.clip(start, lo, hi)
    for _ in range(200):
        g1, g2 = grad(x)
        lo = np.where(g1 > 0, x, lo)
        hi = np.where(g1 < 0, x, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = x - g1 / g2
        near = 1e-13 * (1.0 + np.abs(x))
        done = (g1 == 0) | (np.abs(newton - x) <= near) | (hi - lo <= 2.0 * near)
        if done.all():
            return np.where(np.abs(newton - x) <= near, newton, x)
        inside = np.isfinite(newton) & (newton >= lo) & (newton <= hi)
        x = np.where(done, x, np.where(inside, newton, 0.5 * (lo + hi)))
    raise OracleNotConverged(f"{what}: the mode was not found")


def noisy_binomial(w: np.ndarray, t: np.ndarray, k: np.ndarray, n: np.ndarray, derivs: bool = True,
                   tol: float = INNER_TOL, with_error: bool = False):
    """log of the integral over e ~ N(0, 1) of Phi(w + t e)^k Phi(-w - t e)^(n - k), with d/dw and d2/dw2
    (and, with `with_error`, the relative error estimate of each value).

    Per element: the mode e* of `h(e) = g(w + t e) - e^2 / 2` (g = k log Phi + (n - k) log Phi(-), so
    h'' <= -1 and the root of h' lies between 0 and h'(0)); the range e* +- 10, outside which h is at
    least 50 below its peak; breakpoints at e* + `STEPS_MODE` local sds and, when t > 1, where the
    probit argument w + t e crosses 0, +-2, ..., +-10 (the steps of width 1 / t that a large call noise
    makes); then `breakpoint_rule` to relative accuracy `tol`, refusing otherwise. The derivatives are the
    posterior mean of g' and the posterior mean of g'' plus the posterior variance of g'."""
    w, t, k, n = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(w, t, k, n))
    shape = w.shape
    w, t, k, n = (v.ravel().copy() for v in (w, t, k, n))
    lp, d1, d2, err = (np.zeros(w.size) for _ in range(4))
    flat = t <= 1e-300
    if flat.any():                                    # no noise: the binomial-probit term itself
        g, g1, g2 = _binomial_g(w[flat], k[flat], n[flat])
        lp[flat], d1[flat], d2[flat] = g, g1, g2
    idx = np.flatnonzero(~flat)
    if len(idx):
        ww, tt, kk, nn = w[idx], t[idx], k[idx], n[idx]

        def grad(e):
            _, g1, g2 = _binomial_g(ww + tt * e, kk, nn)
            return tt * g1 - e, tt * tt * g2 - 1.0

        h0 = grad(np.zeros(len(idx)))[0]
        e = _bracketed_root(grad, np.minimum(0.0, h0) - 1.0, np.maximum(0.0, h0) + 1.0, np.zeros(len(idx)),
                            "a summed items call")
        g0, _, g2 = _binomial_g(ww + tt * e, kk, nn)
        top = g0 - 0.5 * e * e
        sd = 1.0 / np.sqrt(1.0 - tt * tt * g2)
        cols = [e - 10.0, e + 10.0] + [e + m * sd for m in STEPS_MODE]
        steep = tt > 1.0
        for x in (-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            cols.append(np.where(steep, (x - ww) / np.where(steep, tt, 1.0), e))
        P = np.sort(np.clip(np.stack(cols, axis=1), (e - 10.0)[:, None], (e + 10.0)[:, None]), axis=1)

        def evaluate(E, R):
            x = ww[R] + tt[R] * E
            if not derivs:
                return kk[R] * special.log_ndtr(x) + (nn[R] - kk[R]) * special.log_ndtr(-x) - 0.5 * E * E, (), None
            g, g1, g2 = _binomial_g(x, kk[R], nn[R])
            return g - 0.5 * E * E, (g1, g2 + g1 * g1), None

        value, G, rel = breakpoint_rule(P, evaluate, top, tol, "a summed items call", 2 if derivs else 0)
        lp[idx] = top + np.log(value) - 0.5 * LOG_2PI
        err[idx] = rel
        if derivs:
            m1 = G[0] / value
            d1[idx], d2[idx] = m1, G[1] / value - m1 * m1
    lp, d1, d2, err = (v.reshape(shape) for v in (lp, d1, d2, err))
    out = (lp, d1, d2) if derivs else (lp,)
    if with_error:
        out = out + (err,)
    return out if len(out) > 1 else out[0]


def factor_values(kind: np.ndarray, lo: np.ndarray, hi: np.ndarray, s: np.ndarray, k: np.ndarray, n: np.ndarray,
                  w: np.ndarray, derivs: bool = True, with_error: bool = False, tol: float = INNER_TOL):
    """log f(w) and, with `derivs`, d/dw and d2/dw2, elementwise. The factor arrays broadcast over w. With
    `with_error` the relative error estimate of each value comes last (0 except for NOISY factors, whose
    inner integral is asked for relative accuracy `tol`)."""
    lp = np.zeros(np.broadcast(w, kind).shape)
    err = np.zeros_like(lp)
    d1 = np.zeros_like(lp)
    d2 = np.zeros_like(lp)
    kind = np.broadcast_to(kind, lp.shape)
    lo, hi, s, k, n = (np.broadcast_to(x, lp.shape) for x in (lo, hi, s, k, n))
    w = np.broadcast_to(w, lp.shape)
    m = kind == INTERVAL
    if m.any():
        a, b, sd = (lo[m] - w[m]) / s[m], (hi[m] - w[m]) / s[m], s[m]
        logp = log_interval(a, b)
        lp[m] = logp
        if derivs:
            ra, rb = np.exp(_log_phi(a) - logp), np.exp(_log_phi(b) - logp)
            g1 = (ra - rb) / sd
            aa, bb = np.where(np.isfinite(a), a, 0.0), np.where(np.isfinite(b), b, 0.0)
            d1[m] = g1
            d2[m] = (aa * ra - bb * rb) / sd ** 2 - g1 ** 2
    m = kind == BINOMIAL
    if m.any():
        sd = s[m]
        g, g1, g2 = _binomial_g(w[m] / sd, k[m], n[m])
        lp[m] = g
        if derivs:
            d1[m] = g1 / sd
            d2[m] = g2 / sd ** 2
    m = kind == NOISY
    if m.any():
        if derivs:
            lp[m], d1[m], d2[m], err[m] = noisy_binomial(w[m], s[m], k[m], n[m], tol=tol, with_error=True)
        else:
            lp[m], err[m] = noisy_binomial(w[m], s[m], k[m], n[m], derivs=False, tol=tol, with_error=True)
    m = kind == GAUSS
    if m.any():
        sd, y = s[m], lo[m]
        z = (y - w[m]) / sd
        lp[m] = -0.5 * z * z - np.log(sd) - 0.5 * LOG_2PI
        if derivs:
            d1[m] = z / sd
            d2[m] = -1.0 / sd ** 2
    d2 = np.minimum(d2, 0.0)       # every factor is log-concave; rounding far in a tail can say otherwise
    out = (lp, d1, d2) if derivs else (lp,)
    if with_error:
        out = out + (err,)
    return out if len(out) > 1 else out[0]


def factor_scale(kind: np.ndarray, s: np.ndarray) -> np.ndarray:
    """The width in w over which a factor changes: the noise sd s, or sqrt(1 + t^2) for a NOISY factor."""
    return np.where(np.asarray(kind) == NOISY, np.sqrt(1.0 + np.asarray(s, dtype=float) ** 2), s)


# ---------------------------------------------------------------------------------------------------
# adaptive Gauss-Hermite on one block

@dataclass
class Block:
    factors: np.ndarray            # positions in the problem's factor list
    dim: int
    method: str                    # "quad" (one direction, scipy), "gh" (Gauss-Hermite) or "nested"
    nodes: int                     # Gauss-Hermite nodes per (smooth) direction at the answer (0: none)
    loglik: float
    laplace: float
    error: float                   # quadrature + inner integrals + dropped directions, in log units
    history: list                  # (nodes per direction, log integral); for "quad" one entry (0, value)
    mean: np.ndarray               # posterior mean of the block's d coordinates
    cov: np.ndarray
    basis: np.ndarray              # p x d: the coordinates are basis' zeta, zeta ~ N(0, I) given Gaussian readings
    sharp: int = 0                 # coordinates integrated by adaptive line rules ("nested")
    slopes: np.ndarray | None = None   # |d log f / dw| times each factor's scale, at the mode


class _Factors:
    """The factor arrays of some rows of a problem, with their scales and the places where they turn."""

    def __init__(self, prob: Problem, rows: np.ndarray, tol: float):
        self.kind, self.lo, self.hi, self.s, self.k, self.n = (getattr(prob, f"f_{name}")[rows]
                                                               for name in ("kind", "lo", "hi", "s", "k", "n"))
        self.scale = factor_scale(self.kind, self.s)
        self.noisy = int(np.sum(self.kind == NOISY))
        # a NOISY factor is asked for a share of the block's tolerance (relative, so in log units)
        self.inner_tol = max(1e-13, min(1e-10, tol / (20.0 * max(1, self.noisy))))
        walls = []
        for i, kind in enumerate(self.kind):
            if kind == INTERVAL:
                walls += [(i, float(b)) for b in (self.lo[i], self.hi[i]) if np.isfinite(b)]
            elif kind == GAUSS:
                walls.append((i, float(self.lo[i])))
            else:
                walls.append((i, 0.0))
        self.walls = walls

    def values(self, W: np.ndarray, derivs: bool = True):
        """log f, (d1, d2,) and the relative error of each value at W (... x factors)."""
        return factor_values(self.kind, self.lo, self.hi, self.s, self.k, self.n, W, derivs=derivs,
                             with_error=True, tol=self.inner_tol)


def _modes(F: _Factors, C: np.ndarray, A: np.ndarray, what: str):
    """For each row of C (m x factors): the maximum over y (A's dh columns) of
    L(y) = sum log f(C + A y) - |y|^2 / 2, by Newton with backtracking per row (L is concave, L'' <= -I).
    Returns y, L, its gradient and Hessian there."""
    m, dh = C.shape[0], A.shape[1]
    eye = np.eye(dh)

    def h(y, idx):
        lp, d1, d2, _ = F.values(C[idx] + y @ A.T)
        return lp.sum(axis=1) - 0.5 * (y * y).sum(axis=1), d1 @ A - y, np.einsum("ri,ia,ib->rab", d2, A, A) - eye

    every = np.arange(m)
    y = np.zeros((m, dh))
    val, grad, hess = h(y, every)
    for _ in range(200):
        step = np.linalg.solve(-hess, grad[..., None])[..., 0]
        gain = (grad * step).sum(axis=1)
        idx = np.flatnonzero(gain > 1e-15 * np.maximum(1.0, np.abs(val)))
        if not len(idx):
            return y, val, grad, hess
        t = np.ones(len(idx))
        for _ in range(60):
            new = y[idx] + t[:, None] * step[idx]
            nval, ngrad, nhess = h(new, idx)
            ok = np.isfinite(nval) & (nval >= val[idx] - 1e-14 * np.abs(val[idx]))
            if ok.all():
                break
            t = np.where(ok, t, 0.5 * t)
        else:
            raise OracleNotConverged(f"{what}: the mode of a line was not found")
        y[idx], val[idx], grad[idx], hess[idx] = new, nval, ngrad, nhess
    raise OracleNotConverged(f"{what}: the mode of a line was not found in 200 Newton steps")




def _reach(grad: np.ndarray, y: np.ndarray):
    """Centre and half-width of the region outside which the integrand is below e^-60 of its top.
    Since L'' <= -I, L(x) <= L(y) + |g|^2 / 2 - |x - (y + g)|^2 / 2 for any y with gradient g."""
    g2 = (grad * grad).sum(axis=1)
    return y + grad, np.sqrt(120.0 + g2)


CUT = 70.0                                    # how far below its top a log-concave integrand ends a line rule


def _cut(points: np.ndarray, logv: np.ndarray, c0: np.ndarray, top: np.ndarray, reach: np.ndarray):
    """The range of a line rule around its mode c0: from the nearest candidate breakpoint on each side
    where the log integrand (log-concave, so it stays lower beyond) is CUT below its top, or else the
    reach bound of `_reach`. Also a bound on the mass left out, relative to e^top: beyond a cut point p
    the log integrand falls at least as fast as the chord from the mode, so the tail is at most
    e^(L(p) - top) |p - c0| / (top - L(p)); beyond the reach bound it is at most e^-60 / reach."""
    low = logv < (top - CUT)[:, None]
    lo, hi = c0 - reach, c0 + reach
    tail = np.zeros(len(c0))
    for side in (-1.0, 1.0):
        here = low & (side * (points - c0[:, None]) > 0)
        dist = np.where(here, side * (points - c0[:, None]), np.inf)
        k = np.argmin(dist, axis=1)
        found = np.isfinite(dist[np.arange(len(c0)), k]) & (dist[np.arange(len(c0)), k] < reach)
        p, lp = points[np.arange(len(c0)), k], logv[np.arange(len(c0)), k]
        drop = np.where(found, top - lp, 1.0)
        tail += np.where(found, np.exp(-drop) * np.abs(p - c0) / drop, math.exp(-60.0) / reach)
        if side < 0:
            lo = np.where(found, p, lo)
        else:
            hi = np.where(found, p, hi)
    return lo, hi, tail


def _line_integrals(F: _Factors, C: np.ndarray, a: np.ndarray, tol: float, what: str):
    """Per row of C (m x factors): log of the integral over y ~ N(0, 1) of prod f(C + a y), the posterior
    mean and second moment of y (m x 1 and m x 1 x 1) and the relative error. Breakpoints at the mode and
    where each factor turns (`walls`), the range cut where the integrand is CUT below its top (`_cut`),
    then `breakpoint_rule` to relative accuracy `tol`."""
    y, val, grad, hess = _modes(F, C, a[:, None], what)
    centre, reach = _reach(grad, y)
    c0 = centre[:, 0]
    sd = 1.0 / np.sqrt(-hess[:, 0, 0])
    cols = [c0 - reach, c0 + reach] + [c0 + k * sd for k in STEPS_MODE]
    for i, b in F.walls:
        if a[i] != 0.0:
            turn, width = (b - C[:, i]) / a[i], F.scale[i] / abs(a[i])
            cols += [turn + k * width for k in STEPS_WALL]
    cand = np.clip(np.stack(cols, axis=1), (c0 - reach)[:, None], (c0 + reach)[:, None])
    lp, _ = F.values(C[:, None, :] + cand[..., None] * a[None, None, :], derivs=False)
    lo, hi, tail = _cut(cand, lp.sum(axis=2) - 0.5 * cand * cand, c0, val, reach)
    P = np.sort(np.clip(cand, lo[:, None], hi[:, None]), axis=1)

    def evaluate(E, R):
        lp, err = F.values(C[R] + E[:, None] * a[None, :], derivs=False)
        return lp.sum(axis=1) - 0.5 * E * E, (E, E * E), (err.sum(axis=1) if F.noisy else None)

    value, G, rel = breakpoint_rule(P, evaluate, val, tol, what, 2, chunk=max(1024, (1 << 20) // len(a)))
    rel = rel + tail / value
    return (val + np.log(value) - 0.5 * LOG_2PI, (G[0] / value)[:, None], (G[1] / value)[:, None, None], rel)


def _plane_integrals(F: _Factors, C: np.ndarray, A: np.ndarray, tol: float, what: str):
    """As `_line_integrals` over y ~ N(0, I_2), A factors x 2: the outer coordinate by `breakpoint_rule`
    over line integrals in the inner one. The outer breakpoints: the mode, where a factor turns on the
    line through the mode, and the vertices where two factors turn, each with its width (the integral of
    a line has kinks where the set it cuts changes shape). Vertices where the integrand is below e^-80 of
    its top are skipped; the range is cut on the profile, the maximum over the inner coordinate (concave
    too, and the line integral is at most its exponential). Returns means m x 2 and second moments m x 2 x 2."""
    y, val, grad, hess = _modes(F, C, A, what)
    centre, reach = _reach(grad, y)
    c0, c1 = centre[:, 0], centre[:, 1]
    sd = np.sqrt(np.linalg.inv(-hess)[:, 0, 0])
    cols = [c0 - reach, c0 + reach] + [c0 + k * sd for k in STEPS_MODE]
    for i, b in F.walls:
        if A[i, 0] != 0.0:
            turn, width = (b - C[:, i] - A[i, 1] * c1) / A[i, 0], F.scale[i] / abs(A[i, 0])
            cols += [turn + k * width for k in STEPS_WALL]
    for one in range(len(F.walls)):
        for two in range(one + 1, len(F.walls)):
            (i, bi), (j, bj) = F.walls[one], F.walls[two]
            det = A[i, 0] * A[j, 1] - A[i, 1] * A[j, 0]
            if i == j or abs(det) <= 1e-12 * np.hypot(*A[i]) * np.hypot(*A[j]):
                continue
            ri, rj = bi - C[:, i], bj - C[:, j]
            v0, v1 = (ri * A[j, 1] - A[i, 1] * rj) / det, (A[i, 0] * rj - ri * A[j, 0]) / det
            lp, _ = F.values(C + v0[:, None] * A[None, :, 0] + v1[:, None] * A[None, :, 1], derivs=False)
            near = lp.sum(axis=1) - 0.5 * (v0 * v0 + v1 * v1) >= val - 80.0
            width = np.hypot(F.scale[i] * A[j, 1], F.scale[j] * A[i, 1]) / abs(det)
            cols += [np.where(near, v0 + k * width, c0) for k in STEPS_WALL]
    cand = np.clip(np.stack(cols, axis=1), (c0 - reach)[:, None], (c0 + reach)[:, None])
    m, K = cand.shape
    lines = (C[:, None, :] + cand[..., None] * A[None, None, :, 0]).reshape(m * K, -1)
    profile = _modes(F, lines, A[:, 1:2], what)[1].reshape(m, K) - 0.5 * cand * cand
    lo, hi, tail = _cut(cand, profile, c0, val, reach)
    P = np.sort(np.clip(cand, lo[:, None], hi[:, None]), axis=1)

    def evaluate(E, R):
        log_in, m_in, s_in, err = _line_integrals(F, C[R] + E[:, None] * A[None, :, 0], A[:, 1], tol, what)
        m_in, s_in = m_in[:, 0], s_in[:, 0, 0]
        return log_in - 0.5 * E * E, (E, E * E, m_in, E * m_in, s_in), err

    value, G, rel = breakpoint_rule(P, evaluate, val, tol, what, 5, chunk=2048)
    rel = rel + tail / value
    G = G / value
    mean = np.stack([G[0], G[2]], axis=1)
    second = np.stack([np.stack([G[1], G[3]], axis=1), np.stack([G[3], G[4]], axis=1)], axis=1)
    return val + np.log(value) - 0.5 * LOG_2PI, mean, second, rel


def _gh_grid(n: int, d: int, start: int, stop: int):
    """Nodes start..stop of the n^d tensor Gauss-Hermite grid (weight e^{-z^2/2}) and their log weights."""
    x, wts = hermegauss(n)
    logw = np.log(wts)
    idx = np.unravel_index(np.arange(start, stop), (n,) * d)
    return np.stack([x[i] for i in idx], axis=1), np.sum([logw[i] for i in idx], axis=0)


def _integrate_block(prob: Problem, part, *, tol: float, max_nodes: int, nodes: Sequence[int] | None,
                     method: str, laplace_only: bool = False):
    """The integral over one block's d coordinates y ~ N(0, I) of the product of its factors (with
    `laplace_only`, just its Laplace value). The block's error (its rule's convergence and inner terms,
    what its dropped directions could move and a roundoff allowance) must fit `part.share`; its rule is
    asked for what is left, and the block refuses when nothing is (unless `nodes` fixes the rule)."""
    rows, mu, Mb, basis = part.rows, part.mu, part.Mb, part.basis
    d = Mb.shape[1]
    share = min(tol, part.share)
    F = _Factors(prob, rows, share)

    def h(y: np.ndarray):
        lp, d1, d2, _ = F.values(mu + Mb @ y)
        return lp.sum() - 0.5 * y @ y, Mb.T @ d1 - y, (Mb.T * d2) @ Mb - np.eye(d)

    # the mode, by Newton with backtracking (every factor is log-concave in w, so the mode is unique)
    y = np.zeros(d)
    val, grad, hess = h(y)
    for _ in range(500):
        step = np.linalg.solve(-hess, grad)
        if float(grad @ step) < 1e-15 * max(1.0, abs(float(val))):     # the gain left is below rounding
            break
        t = 1.0
        while True:
            new = y + t * step
            nval, ngrad, nhess = h(new)
            if np.isfinite(nval) and nval >= val - 1e-14 * abs(val):
                break
            t *= 0.5
            if t < 1e-12:
                raise OracleNotConverged("the mode of a block was not found")
        y, val, grad, hess = new, nval, ngrad, nhess
    else:
        raise OracleNotConverged("the mode of a block was not found in 500 Newton steps")
    prec = -hess
    chol = np.linalg.cholesky(0.5 * (prec + prec.T))
    Lc = linalg.solve_triangular(chol, np.eye(d), lower=True).T     # Lc Lc' = prec^-1
    log_det = -float(np.log(np.diag(chol)).sum())
    laplace = log_det + float(val)
    if laplace_only:
        return laplace
    # what the dropped directions could move: second order in their (standardised) variance
    lp, d1, d2, _ = F.values(mu + Mb @ y)
    slopes = np.abs(d1) * F.scale
    dropped = 0.5 * part.drop * float(np.sum(np.abs(d2) * F.scale ** 2 + slopes ** 2))
    # roundoff: a few ulps of every log term the value sums (the factors at the mode, the prior, the
    # determinant), so a value of -289 is not reported to 1e-14
    rounding = ROUND * (float(np.abs(lp).sum()) + 0.5 * float(y @ y) + abs(log_det) + 1.0)
    left = share - dropped - rounding
    if left <= 0.0 and nodes is None:
        raise OracleNotConverged(f"a block of {d} directions cannot meet its share {share:.3g} of tol: its dropped "
                                 f"directions could move {dropped:.3g} and its roundoff is {rounding:.3g}")
    left = max(left, 0.0)

    sharp = int(np.sum(part.strength >= SHARP_AT))
    if method == "nested" or (method == "auto" and d >= 2 and nodes is None and 1 <= sharp <= 2):
        if sharp > 2:
            raise OracleTooLarge(f"a block has {sharp} sharp directions; the nested rule takes at most 2")
        block = _nested_block(F, part, y, prec, max(1, sharp), laplace, left, max_nodes)
    elif method == "auto" and d == 1 and nodes is None:
        block = _quad_block(F, part, float(y[0]), float(Lc[0, 0]), float(val), laplace, left)
    else:
        block = _gh_block(F, part, y, Lc, log_det, laplace, left, max_nodes, nodes)
    block.error += dropped + rounding
    block.slopes = slopes
    return block


def _gh_block(F: _Factors, part, y, Lc, log_det, laplace, tol, max_nodes, nodes) -> Block:
    """Tensor Gauss-Hermite centred at the mode and scaled by the curvature there, through `NODES` until
    the last two changes plus the largest inner error are within `tol`."""
    rows, mu, Mb, basis = part.rows, part.mu, part.Mb, part.basis
    d = Mb.shape[1]
    chunk = max(256, CHUNK // max(1, 4 * F.noisy))

    def rule(n: int):
        total = n ** d
        top, s0, inner = -np.inf, 0.0, 0.0
        s1, s2 = np.zeros(d), np.zeros((d, d))
        for start in range(0, total, chunk):
            Z, lw = _gh_grid(n, d, start, min(total, start + chunk))
            Y = y + Z @ Lc.T
            lp, err = F.values(mu + Y @ Mb.T, derivs=False)
            inner = max(inner, float(err.sum(axis=1).max()))
            val_n = lp.sum(axis=1) - 0.5 * (Y * Y).sum(axis=1) + 0.5 * (Z * Z).sum(axis=1) + lw
            local = float(np.max(val_n))
            if local > top:
                shrink = math.exp(top - local) if np.isfinite(top) else 0.0
                s0, s1, s2, top = s0 * shrink, s1 * shrink, s2 * shrink, local
            wts_n = np.exp(val_n - top)
            s0 += float(wts_n.sum())
            s1 += wts_n @ Y
            s2 += (Y * wts_n[:, None]).T @ Y
        logint = log_det - 0.5 * d * LOG_2PI + top + math.log(s0)
        mean = s1 / s0
        return logint, mean, s2 / s0 - np.outer(mean, mean), inner

    seq = list(nodes) if nodes is not None else [n for n in NODES if n ** d <= max_nodes]
    history: list = []
    found = None
    for n in seq:
        logint, mean, cov, inner = rule(n)
        history.append((n, logint))
        found = (n, logint, mean, cov, inner)
        if nodes is None and len(history) >= 3:
            last = max(abs(history[-1][1] - history[-2][1]), abs(history[-2][1] - history[-3][1]))
            if last + inner <= tol:
                break
    else:
        if nodes is None:
            changes = [abs(history[i][1] - history[i - 1][1]) for i in range(1, len(history))]
            raise OracleNotConverged(f"block of {d} directions: changes {changes} at nodes "
                                     f"{[n for n, _ in history]}, tol {tol}, max_nodes {max_nodes}")
    assert found is not None
    changes = [abs(history[i][1] - history[i - 1][1]) for i in range(1, len(history))]
    n, logint, mean, cov, inner = found
    error = (max(changes[-2:]) if changes else math.nan) + inner
    return Block(factors=rows, dim=d, method="gh", nodes=n, loglik=logint, laplace=laplace, error=error,
                 history=history, mean=mean, cov=cov, basis=basis)


def _quad_block(F: _Factors, part, mode: float, scale: float, top: float, laplace: float, tol: float) -> Block:
    """One direction: scipy's adaptive Gauss-Kronrod over the range where the integrand is above e^-80 of
    its peak, with a break point at the mode. Robust to the near-hard walls of a small call noise. The
    integral is asked for relative accuracy 0.4 tol, at most 1e-11 (or the integrand's roundoff, a few ulps
    of its log top, when that is larger) and at least the 2e-14 scipy accepts; quad's own error
    estimate plus the inner error must then be within tol, and quad warning (roundoff, subdivisions)
    refuses."""
    col, mu = part.Mb[:, 0], part.mu
    inner = [0.0]

    def hval(y: float) -> float:
        lp, err = F.values(mu + col * y, derivs=False)
        inner[0] = max(inner[0], float(err.sum()))
        return float(lp.sum()) - 0.5 * y * y

    def edge(sign: float) -> float:
        step = scale
        for _ in range(200):
            if hval(mode + sign * step) - top < -80.0:
                return mode + sign * step
            step *= 2.0
        raise OracleNotConverged("the integrand of a block does not decay")

    lo, hi = edge(-1.0), edge(1.0)
    points = [mode + k * scale for k in (-8, -2, 0, 2, 8) if lo < mode + k * scale < hi]
    # what quad can ask for: the integrand exp(h - top) carries a relative roundoff of a few ulps of |top|
    reach = max(1e-11, ROUND * (abs(top) + 1.0))
    found = []
    for power in (0, 1, 2):            # a moment near 0 gets an absolute tolerance scaled by the integral
        tol_abs = 0.0 if power == 0 else 1e-13 * found[0][0] * (abs(mode) + scale) ** power
        if power == 0:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", integrate.IntegrationWarning)
                value, err = integrate.quad(lambda y: math.exp(hval(y) - top), lo, hi, points=points, epsabs=0.0,
                                            epsrel=max(min(reach, 0.4 * tol), 2e-14), limit=2000)
            trouble = [str(w.message) for w in caught if issubclass(w.category, integrate.IntegrationWarning)]
            if trouble or not err / value + inner[0] <= tol:
                raise OracleNotConverged(f"a block of one direction: quad error {err / value:.3g} plus inner "
                                         f"{inner[0]:.3g} against {tol:.3g}" + (f" ({trouble[0][:80]})" if trouble else ""))
        else:
            value, err = integrate.quad(lambda y: (y ** power) * math.exp(hval(y) - top), lo, hi, points=points,
                                        epsabs=tol_abs, epsrel=reach, limit=2000)
        found.append((value, err))
    s0, e0 = found[0]
    mean = found[1][0] / s0
    var = found[2][0] / s0 - mean * mean
    logint = top - 0.5 * LOG_2PI + math.log(s0)
    return Block(factors=part.rows, dim=1, method="quad", nodes=0, loglik=logint, laplace=laplace,
                 error=e0 / s0 + inner[0], history=[(0, logint)], mean=np.array([mean]), cov=np.array([[var]]),
                 basis=part.basis)


def _nested_block(F: _Factors, part, y, prec, dh: int, laplace: float, tol: float, max_nodes: int) -> Block:
    """The first `dh` coordinates (the sharp ones: the factors change over a small part of their sd) by
    adaptive line rules with breakpoints where the factors turn (`_line_integrals`, `_plane_integrals`),
    the rest by Gauss-Hermite from the Laplace marginal of those coordinates, through `NODES` until the
    last two changes plus the largest inner error are within `tol`. The inner rules get tol / 20 each."""
    rows, mu, Mb, basis = part.rows, part.mu, part.Mb, part.basis
    d = Mb.shape[1]
    dg = d - dh
    Ah, Ag = Mb[:, :dh], Mb[:, dh:]
    inner_tol = tol / 20.0
    what = f"a block of {d} directions, {dh} sharp"

    def sharp(Yg):
        C = mu[None, :] + Yg @ Ag.T
        if dh == 1:
            return _line_integrals(F, C, Ah[:, 0], inner_tol, what)
        return _plane_integrals(F, C, Ah, inner_tol, what)

    if dg == 0:
        logint, mh, sh, err = sharp(np.zeros((1, 0)))
        mean = mh[0]
        return Block(factors=rows, dim=d, method="nested", nodes=0, loglik=float(logint[0]), laplace=laplace,
                     error=float(err[0]), history=[(0, float(logint[0]))], mean=mean,
                     cov=sh[0] - np.outer(mean, mean), basis=basis, sharp=dh)

    cov_all = np.linalg.inv(prec)
    Lg = np.linalg.cholesky(0.5 * (cov_all[dh:, dh:] + cov_all[dh:, dh:].T))
    log_det = float(np.log(np.diag(Lg)).sum())
    yg = y[dh:]

    def rule(n: int):
        Z, lw = _gh_grid(n, dg, 0, n ** dg)
        Yg = yg + Z @ Lg.T
        logJ, mh, sh, err = sharp(Yg)
        val_n = logJ - 0.5 * (Yg * Yg).sum(axis=1) - 0.5 * dg * LOG_2PI + 0.5 * (Z * Z).sum(axis=1) + lw
        top = float(np.max(val_n))
        wts = np.exp(val_n - top)
        s0 = float(wts.sum())
        w = wts / s0
        mean = np.concatenate([w @ mh, w @ Yg])
        second = np.zeros((d, d))
        second[:dh, :dh] = np.einsum("r,rab->ab", w, sh)
        second[dh:, dh:] = (Yg * w[:, None]).T @ Yg
        second[dh:, :dh] = (Yg * w[:, None]).T @ mh
        second[:dh, dh:] = second[dh:, :dh].T
        return log_det + top + math.log(s0), mean, second - np.outer(mean, mean), float(err.max())

    history: list = []
    found = None
    for n in [n for n in NODES if n ** dg <= min(max_nodes, NESTED_NODES)]:
        logint, mean, cov, inner = rule(n)
        history.append((n, logint))
        found = (n, logint, mean, cov, inner)
        if len(history) >= 3:
            last = max(abs(history[-1][1] - history[-2][1]), abs(history[-2][1] - history[-3][1]))
            if last + inner <= tol:
                break
    else:
        changes = [abs(history[i][1] - history[i - 1][1]) for i in range(1, len(history))]
        raise OracleNotConverged(f"{what}: changes {changes} at nodes {[n for n, _ in history]}, tol {tol}")
    assert found is not None
    n, logint, mean, cov, inner = found
    changes = [abs(history[i][1] - history[i - 1][1]) for i in range(1, len(history))]
    return Block(factors=rows, dim=d, method="nested", nodes=n, loglik=logint, laplace=laplace,
                 error=max(changes[-2:]) + inner, history=history, mean=mean, cov=cov, basis=basis, sharp=dh)


# ---------------------------------------------------------------------------------------------------
# the oracle

@dataclass
class Result:
    loglik: float                  # log p(y | theta), normalised
    error: float                   # error estimate on loglik (blocks, inner integrals, dropped and unlinked parts)
    laplace: float                 # the Laplace approximation of the same integral (1 node per direction)
    gaussian_loglik: float         # the closed-form part (Gaussian readings and terms free of random effects)
    blocks: list = field(default_factory=list)
    posterior: dict = field(default_factory=dict)   # name -> {"mean": array, "sd": array}
    z_mean: np.ndarray | None = None
    z_cov: np.ndarray | None = None
    items: str = "explicit"        # how the call noise of items calls was integrated


@dataclass
class _Part:
    rows: np.ndarray               # factors of the block
    mu: np.ndarray                 # their forms at zeta = 0
    Mb: np.ndarray                 # factors x d: the forms are mu + Mb y, y ~ N(0, I_d)
    basis: np.ndarray              # p x d: y = basis' zeta
    strength: np.ndarray           # d standardised singular values (Mb's columns over the factor scales)
    drop: float                    # sum of the squared standardised singular values dropped
    share: float = math.inf        # this part's share of the requested tol


@dataclass
class _Layout:
    gauss: float                   # the closed-form part
    gauss_round: float             # its roundoff allowance
    flat: float                    # factors left with no random direction
    flat_error: float              # their dropped-direction, inner and roundoff error
    mean: np.ndarray               # z given the Gaussian readings: mean + root zeta, zeta ~ N(0, I)
    root: np.ndarray
    parts: list                    # _Part per block
    cross: np.ndarray              # |covariance| of standardised forms in different blocks (unlinked)
    slopes: np.ndarray             # standardised |d log f / dw| of flat factors at their value (0 elsewhere)
    share: float                   # each part's share of tol; one more share is the links' and the closed form's


def _layout(prob: Problem, tol: float = 1e-8) -> _Layout:
    """Steps 2 and 3 of the method. Every rank and link decision is made on the forms standardised by
    their factor's scale (how many noise sds a factor moves per unit of zeta), so it does not depend on
    the units of theta: two factors are linked when their forms correlate above LINK_TOL, and a block
    keeps the directions whose standardised singular value exceeds DROP_STD and, below that, as many
    more as it takes for what the rest could move (second order, at the mean) to fit DROP_SHARE of the
    part's share of `tol`. A block with none is a product of values at the mean. `tol` is split in
    equal shares over the parts, plus one share for the cut links and the closed-form part."""
    p = prob.p
    gauss, gauss_terms = prob.const, 0.0
    mean, cov = np.zeros(p), np.eye(p)
    if len(prob.g_y):
        G = prob.g_rows
        Sg = G @ G.T + np.diag(prob.g_var)
        factor = linalg.cho_factor(Sg, lower=True)
        resid = prob.g_y - prob.g_off
        alpha = linalg.cho_solve(factor, resid)
        log_det = 2.0 * float(np.log(np.diag(factor[0])).sum())
        gauss += -0.5 * (len(resid) * LOG_2PI + log_det + float(resid @ alpha))
        gauss_terms = 0.5 * (len(resid) * LOG_2PI + abs(log_det) + abs(float(resid @ alpha)))
        mean = G.T @ alpha
        cov = np.eye(p) - G.T @ linalg.cho_solve(factor, G)
        cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    root = vecs * np.sqrt(np.clip(vals, 0.0, None))[None, :]       # root root' = cov
    nf = len(prob.f_off)
    flat, flat_error, parts = 0.0, 0.0, []
    cross, slopes = np.zeros((nf, nf)), np.zeros(nf)
    if nf:
        mu_w = prob.f_off + prob.f_rows @ mean
        scale = factor_scale(prob.f_kind, prob.f_s)
        Bs = (prob.f_rows @ root) / scale[:, None]
        cw = Bs @ Bs.T
        sd_w = np.sqrt(np.clip(np.diag(cw), 0.0, None))
        link = np.abs(cw) > LINK_TOL * np.outer(sd_w, sd_w)
        link &= (sd_w[:, None] > 0) & (sd_w[None, :] > 0)
        n_parts, label = connected_components(link, directed=False)
        cross = np.abs(cw) * (label[:, None] != label[None, :])
        share = tol / (n_parts + 1)
        for part in range(n_parts):
            rows = np.flatnonzero(label == part)
            U, S, Vt = np.linalg.svd(Bs[rows], full_matrices=False)
            F = _Factors(prob, rows, share)
            lp, d1, d2, err = F.values(mu_w[rows])
            curv = float(np.sum(np.abs(d2) * F.scale ** 2 + (d1 * F.scale) ** 2))
            keep = S > DROP_STD                 # then the next largest while the rest could move too much
            tail = np.cumsum((S ** 2)[::-1])[::-1]          # tail[k]: sum of S^2 from k on
            for k in range(len(S)):
                if not keep[k] and S[k] > 0 and 0.5 * tail[k] * curv > DROP_SHARE * share:
                    keep[k] = True
            drop = float(np.sum(S[~keep] ** 2))
            if keep.any():
                parts.append(_Part(rows=rows, mu=mu_w[rows], Mb=scale[rows, None] * (U[:, keep] * S[keep][None, :]),
                                   basis=Vt[keep].T, strength=S[keep], drop=drop, share=share))
                continue
            flat += float(lp.sum())
            slopes[rows] = np.abs(d1) * F.scale
            flat_error += (float(err.sum()) + 0.5 * drop * curv + ROUND * (float(np.abs(lp).sum()) + 1.0))
    else:
        share = tol
    return _Layout(gauss=gauss, gauss_round=ROUND * (abs(prob.const) + gauss_terms), flat=flat,
                   flat_error=flat_error, mean=mean, root=root, parts=parts, cross=cross, slopes=slopes, share=share)


def exact(data: Mapping[str, Any], theta: Mapping[str, Any] | None = None, *, random: Sequence[str] = (),
          tol: float = 1e-8, max_nodes: int = 2_000_000, nodes: Sequence[int] | None = None,
          gaussian: str = "closed", method: str = "auto", items: str = "auto") -> Result:
    """The exact log-likelihood, its error estimate, the Laplace value and the posterior of every random effect.

    `method="auto"` integrates a block of one direction with scipy's adaptive quadrature, a block with
    one or two sharp directions by the nested rule and other blocks by Gauss-Hermite; `method="gh"` uses
    Gauss-Hermite everywhere and `method="nested"` the nested rule everywhere (checks of one against the
    other). `nodes` fixes the Gauss-Hermite node counts per direction (no convergence loop; the last
    count is the answer). `items` is "auto", "explicit" or "summed" (the module docstring, step 3).

    `tol` is global, in log units: the returned `error` (every block, flat part, cut link and roundoff
    term) is at most `tol`, or `OracleNotConverged` is raised. Except with fixed `nodes`, which never
    refuses."""
    if method not in ("auto", "gh", "nested"):
        raise ValueError("method= is 'auto', 'gh' or 'nested'")
    if items not in ("auto", "explicit", "summed"):
        raise ValueError("items= is 'auto', 'explicit' or 'summed'")
    options = dict(tol=tol, max_nodes=max_nodes, nodes=nodes, method=method)
    prob = build_problem(data, theta, random=random, gaussian=gaussian, items="explicit")
    lay = _layout(prob, tol)
    has_e = len(prob.effects["e"][0]) > 0
    summed = has_e and (items == "summed" or (items == "auto" and
                                             max((part.Mb.shape[1] for part in lay.parts), default=0) > SUM_DIM))
    laplace = None
    if summed:                     # the Laplace value of the explicit layout, then integrate the summed one
        laplace = lay.gauss + lay.flat + sum(_integrate_block(prob, part, laplace_only=True, **options)
                                             for part in lay.parts)
        prob = build_problem(data, theta, random=random, gaussian=gaussian, items="summed")
        lay = _layout(prob, tol)
    p = prob.p
    loglik, lap, error = lay.gauss + lay.flat, lay.gauss + lay.flat, lay.flat_error
    slopes = lay.slopes.copy()
    blocks: list[Block] = []
    z_mean, inner = lay.mean.copy(), np.eye(p)
    for part in lay.parts:
        d = part.Mb.shape[1]
        if d > MAX_DIM:
            raise OracleTooLarge(f"a block of {len(part.rows)} factors spans {d} directions; the cap is {MAX_DIM}")
        block = _integrate_block(prob, part, **options)
        blocks.append(block)
        loglik += block.loglik
        lap += block.laplace
        error += block.error           # NaN only with fixed nodes and one count: then the error is unknown
        slopes[part.rows] = block.slopes
        z_mean += lay.root @ (part.basis @ block.mean)
        inner += part.basis @ (block.cov - np.eye(d)) @ part.basis.T
    error += 0.5 * float(slopes @ lay.cross @ slopes)      # first order in the covariance left unlinked
    error += lay.gauss_round
    if nodes is None and not error <= tol:
        raise OracleNotConverged(f"the error {error:.3g} is above tol {tol:.3g} (blocks "
                                 f"{[round(b.error, 15) for b in blocks]}, flat {lay.flat_error:.3g}, "
                                 f"closed form {lay.gauss_round:.3g})")
    z_cov = lay.root @ inner @ lay.root.T
    posterior = {}
    for name, (off, T) in prob.effects.items():
        m = off + T @ z_mean
        var = np.einsum("...i,ij,...j->...", T, z_cov, T)
        posterior[name] = {"mean": m, "sd": np.sqrt(np.clip(var, 0.0, None))}
    return Result(loglik=float(loglik), error=float(error), laplace=float(lap if laplace is None else laplace),
                  gaussian_loglik=float(lay.gauss), blocks=blocks, posterior=posterior, z_mean=z_mean, z_cov=z_cov,
                  items="summed" if summed else "explicit")


def exact_loglik(data: Mapping[str, Any], theta: Mapping[str, Any] | None = None, **options: Any) -> float:
    """log p(y | theta), every random effect integrated out. See `exact`."""
    return exact(data, theta, **options).loglik


def laplace_loglik(data: Mapping[str, Any], theta: Mapping[str, Any] | None = None, **options: Any) -> float:
    """The Laplace approximation of the same integral, at the joint mode of the random effects.

    Gaussian readings are integrated exactly first, which leaves the Laplace value unchanged (they are
    quadratic in the random effects). An engine that integrates the random effects of the layout by
    Laplace at the same theta should reproduce this number."""
    return exact(data, theta, **options).laplace


def posterior(data: Mapping[str, Any], theta: Mapping[str, Any] | None = None, **options: Any) -> dict:
    """Posterior mean and sd of every random effect given y and theta: name -> {"mean", "sd"}."""
    return exact(data, theta, **options).posterior


def pymc_constant(data: Mapping[str, Any]) -> float:
    """What the normalised log-likelihood keeps and `engine_pymc`'s potentials drop: -log(2 pi)/2 per
    Gaussian reading, -log(n)/2 per linear cell of n calls, and log C(n, k) per `items` reading."""
    out = 0.0
    for calls in np.asarray(data["s_n"], dtype=float):
        out += -0.5 * calls * LOG_2PI - 0.5 * math.log(calls)
    kind = np.asarray(data["g_kind"], dtype=np.int64)
    out += -0.5 * LOG_2PI * int(np.sum(kind == LINEAR))
    k, n = np.asarray(data["g_k"], dtype=float), np.asarray(data["g_n"], dtype=float)
    at = kind == ITEMS
    out += float(np.sum(special.gammaln(n[at] + 1) - special.gammaln(k[at] + 1) - special.gammaln(n[at] - k[at] + 1)))
    return float(out)


# ---------------------------------------------------------------------------------------------------
# tiny designs: a study in memory, the design `model.build` makes of it, and the theta it was drawn from

SCALES = {"linear": {"kind": "linear", "range": [0, 1]}, "fraction": {"kind": "fraction"},
          "binary": {"kind": "binary"}, "items": {"kind": "items"}, "pairs": {"kind": "linear", "range": [0, 1]}}
WHEN = "2026-01-15T09:00:00-08:00"


@dataclass
class Tiny:
    kind: str
    study: Any
    rubric: dict
    judges: dict
    design: Any
    data: dict
    theta: dict
    readings: list          # (run, judge, value on the fitting scale) per reading, in the order written
    pairs: list             # (run a, run b, judge, o, verdict 0 a / 1 tie / 2 b) per pair


def tiny_theta(kind: str, runs: int, judges: int) -> dict:
    """Values a tiny study of the kind is drawn at (all synthetic)."""
    unit = kind in ("linear", "fraction", "levels")
    mu = np.array([0.45, 0.6, 0.35])[:runs] if unit else np.array([0.3, -0.4, 0.6])[:runs]
    out = {"mu": mu[:, None], "bias": np.array([0.0, 0.05 if unit else 0.3])[:judges, None]}
    if unit:
        sig = [0.15, 0.2] if kind == "levels" else [0.05, 0.08]
        out.update(sd_input=0.08, sd_run=0.1, tau=np.array([0.06, 0.09])[:judges, None],
                   sigma=np.array(sig)[:judges, None])
    else:
        sig = [1.0, 0.8] if kind == "binary" else [0.5, 0.7]
        out.update(sd_input=0.4, sd_run=0.5, tau=np.array([0.4, 0.6])[:judges, None],
                   sigma=np.array(sig)[:judges, None])
    if kind == "pairs":
        out.update(mu=np.array([0.3, -0.3, 0.1])[:runs, None], sd_input=0.4, sd_run=0.5,
                   tau=np.array([0.4, 0.6])[:judges, None])
    out.update(sigma_pair=np.array([1.0, 1.3])[:judges, None], position=np.array([0.2, -0.1])[:judges],
               tie_width=np.array([0.3, 0.5])[:judges])
    return out


def tiny(kind: str, *, runs: int = 1, judges: int = 1, calls: int = 1, pair_calls: int = 0,
         inputs: Sequence[int] | None = None, levels: int = 3, items: int = 3,
         theta: Mapping[str, Any] | None = None, seed: int = 0) -> Tiny:
    """A tiny one-dimension study, drawn from the model at `tiny_theta` (updated by `theta`), and its design.

    `kind` is a scale kind (`linear`, `fraction`, `binary`, `levels`, `items`) or `pairs` (a linear
    component with pairs only). Every run is its own arm. `inputs` gives each run's input; the default
    is one input per run, and one shared input when there are pairs (a pair needs two runs of one
    input). Every judge reads every run `calls` times and compares every two runs `pair_calls` times,
    shown first in turn a, b, unknown. Values are clipped to the scale of `linear` and `fraction`."""
    if kind not in (*SCALES, "levels"):
        raise ValueError(f"unknown kind {kind!r}")
    if kind == "pairs":
        calls, pair_calls = 0, max(1, pair_calls)
    rng = np.random.default_rng(seed)
    scale = ({"kind": "levels", "levels": [f"l{n}" for n in range(levels)]} if kind == "levels"
             else dict(SCALES[kind]))
    rubric = C.validate_rubric({"schema": C.SCHEMA_RUBRIC, "id": "tiny", "version": "1",
                                "components": [{"id": "c", "title": "C", "scale": scale}]})
    judge_rows = [{"id": f"j{n + 1}", "kind": "model", "model": {"id": f"model-{n + 1}", "family": "alpha"}}
                  for n in range(judges)]
    judges_doc = C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": ["j1"], "min_anchor_targets": 1,
                                    "judges": judge_rows})
    if inputs is None:
        inputs = [0] * runs if pair_calls else list(range(runs))
    run_rows = [C.validate_run({"schema": C.SCHEMA_RUNS, "run": f"r{r + 1}", "input": f"i{inputs[r] + 1}",
                                "arm": f"a{r + 1}", "steps": [{"step": "final", "model": {"id": "m", "family": "gamma"}}]})
                for r in range(runs)]
    th = {**tiny_theta(kind, runs, judges), **dict(theta or {})}
    J, R = judges, runs
    n_in = max(inputs) + 1
    as_arr = {name: np.broadcast_to(np.asarray(th[name], dtype=float), shape) for name, shape in
              (("mu", (R, 1)), ("bias", (J, 1)), ("tau", (J, 1)), ("sigma", (J, 1)), ("sigma_pair", (J, 1)),
               ("position", (J,)), ("tie_width", (J,)), ("sd_input", (1,)), ("sd_run", (1,)))}
    q = (as_arr["mu"][:, 0] + as_arr["sd_input"][0] * rng.standard_normal(n_in)[inputs]
         + as_arr["sd_run"][0] * rng.standard_normal(R))
    cell = as_arr["tau"][:, 0][:, None] * rng.standard_normal((J, R))
    scores, readings, pair_rows, pairs = [], [], [], []
    cuts = (np.arange(levels - 1) + 0.5) / (levels - 1)
    for r in range(R):
        for j in range(J):
            for k in range(calls):
                eta = q[r] + as_arr["bias"][j, 0] + cell[j, r]
                noise = as_arr["sigma"][j, 0] * rng.standard_normal()
                row = {"schema": C.SCHEMA_SCORES, "id": f"s{len(scores):03d}", "run": f"r{r + 1}", "component": "c",
                       "judge": f"j{j + 1}", "call": f"j{j + 1}-r{r + 1}-k{k}", "rubric": "tiny@1", "observed_at": WHEN}
                if kind in ("linear", "fraction"):
                    y = float(np.clip(round(eta + noise, 6), 0.0, 1.0))
                    row["value"] = y
                    readings.append((r, j, y))
                elif kind == "binary":
                    y = int(eta + noise >= 0)
                    row["value"] = y
                    readings.append((r, j, float(y)))
                elif kind == "levels":
                    level = int(np.searchsorted(cuts, eta + noise))
                    row["value"] = f"l{level}"
                    readings.append((r, j, level))
                else:
                    passed = int(rng.binomial(items, special.ndtr(eta + noise)))
                    row["value"], row["of"] = passed, items
                    readings.append((r, j, (passed, items)))
                scores.append(C.validate_score(row))
    for j in range(J):
        for ra in range(R):
            for rb in range(ra + 1, R):
                for k in range(pair_calls):
                    shown = ("a", "b", None)[k % 3]
                    o = {"a": 1.0, "b": -1.0, None: 0.0}[shown]
                    z = (q[ra] + cell[j, ra] - q[rb] - cell[j, rb] + as_arr["position"][j] * o
                         + as_arr["sigma_pair"][j, 0] * rng.standard_normal())
                    tie = as_arr["tie_width"][j]
                    verdict = 0 if z > tie else (2 if z < -tie else 1)
                    pair_rows.append(C.validate_pair({
                        "schema": C.SCHEMA_PAIRS, "id": f"p{len(pair_rows):03d}", "a": f"r{ra + 1}", "b": f"r{rb + 1}",
                        "component": "c", "judge": f"j{j + 1}", "call": f"pj{j + 1}-{ra}-{rb}-k{k}",
                        "shown_first": shown, "verdict": ("a", "tie", "b")[verdict], "rubric": "tiny@1",
                        "observed_at": WHEN}))
                    pairs.append((ra, rb, j, o, verdict))
    study = C.Study(Path("."), run_rows, scores, pair_rows, [])
    design = M.build(study, rubric, judges_doc)
    data = design.data
    for axis, names in (("arms", [f"a{r + 1}" for r in range(R)]), ("judges", [f"j{j + 1}" for j in range(J)])):
        if list(design.index[axis]) != names:   # theta is laid out in this order
            raise AssertionError(f"model.build ordered the {axis} as {design.index[axis]}, not {names}")
    return Tiny(kind=kind, study=study, rubric=rubric, judges=judges_doc, design=design, data=data,
                theta=full_theta(data, th), readings=readings, pairs=pairs)
