"""Tests of the Laplace engine's likelihood kernels and cell integrals (`laplace_kernels`).

Every value is checked against something the module does not compute itself: scipy's distributions,
a multivariate normal density, a 90-digit decimal evaluation of the normal cdf, finite differences, and
`scipy.integrate.quad` over the latent term with an integrand written here from the model's definition.
"""
from __future__ import annotations

import functools
import math
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from loopmath.eval import laplace_kernels as K
from loopmath.eval import model

SHARP_CELL = 0.00541994   # two top readings of a 3-level scale, q 0.5, tau 0.1, sd 0.01, cut 0.75

# ---- independent references -------------------------------------------------------------------------------

_PI = Decimal("3.14159265358979323846264338327950288419716939937510"
               "58209749445923078164062862089986280348253421170679")


def _logphi_ref(x: float, prec: int = 90) -> tuple[float, float, float, float]:
    """log Phi(x) and its first three derivatives in 90-digit decimal arithmetic, as floats."""
    with localcontext() as ctx:
        ctx.prec = prec
        return tuple(float(v) for v in _logphi_dec(Decimal(x), prec))


def _logphi_dec(X: Decimal, prec: int = 90) -> tuple:
    """log Phi(X) and its first three derivatives as decimals (the caller sets the context precision).

    |x| < 8: Phi = 1/2 + phi(x) sum x^(2n+1) / (2n+1)!!. x <= -8: the continued fraction of the Mills ratio,
    R(t) = 1 / (t + 1 / (t + 2 / (t + 3 / ...))). Neither is the asymptotic series the module uses."""
    x = float(X)
    if x <= -8.0:
        t = -X
        f = t
        for k in range(6000, 0, -1):
            f = t + k / f
        lam = f                                   # phi / Phi = 1 / R(t)
        logphi = -(X * X) / 2 - (2 * _PI).sqrt().ln()
        l0 = logphi - lam.ln()
    else:
        pdf = (-(X * X) / 2).exp() / (2 * _PI).sqrt()
        term, s, n = X, X, 0
        while True:
            n += 1
            term = term * X * X / (2 * n + 1)
            s += term
            if n > 10 and abs(term) <= Decimal(10) ** (-prec) * abs(s):
                break
        cdf = Decimal("0.5") + pdf * s
        lam = pdf / cdf
        l0 = cdf.ln()
    g = X + lam
    l2 = -lam * g
    l3 = lam * (g * (X + 2 * lam) - 1)
    return l0, lam, l2, l3


def _interval_ref(eta, lo, hi, sd):
    """log P(lo < eta + e < hi) from scipy's logcdf / logsf, on the side where the difference keeps digits."""
    a, b = (lo - eta) / sd, (hi - eta) / sd
    upper = a > 0
    big = np.where(upper, stats.norm.logsf(a), stats.norm.logcdf(b))
    small = np.where(upper, stats.norm.logsf(b), stats.norm.logcdf(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        return big + np.where(np.isfinite(small), np.log1p(-np.exp(small - big)), 0.0)


def _reading_ref(kind, eta, ob, sd):
    """log p(reading | eta) for a scalar eta, from scipy's log_ndtr on the side where a difference keeps digits."""
    if kind == "levels":
        a, b = (ob[0] - eta) / sd, (ob[1] - eta) / sd
        if a > 0:
            big, small = special.log_ndtr(-a), special.log_ndtr(-b)
        else:
            big, small = special.log_ndtr(b), special.log_ndtr(a)
        return float(big) + (math.log1p(-math.exp(small - big)) if small > -math.inf else 0.0)
    if kind == "binary":
        return float(special.log_ndtr(eta / sd if ob == 1 else -eta / sd))
    k, n = ob
    return (k * float(special.log_ndtr(eta)) + (n - k) * float(special.log_ndtr(-eta))
            + math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


@functools.lru_cache(maxsize=None)
def _quad_ref(kind, m, tau, sd, readings):
    """log of integral phi(z) prod p(reading | m + tau z) dz by scipy.integrate.quad over the region where the
    integrand is within 60 log units of its maximum, cut into 80 pieces. `readings` is a tuple."""
    def H(z):
        out = -0.5 * z * z - 0.5 * math.log(2 * math.pi)
        return out + sum(_reading_ref(kind, m + tau * z, ob, sd) for ob in readings)

    zs = np.linspace(-80.0, 80.0, 1601)          # the integrand is log-concave: the grid argmax brackets the mode
    i = int(np.argmax([H(z) for z in zs]))
    r = optimize.minimize_scalar(lambda z: -H(z), bounds=(zs[max(i - 1, 0)], zs[min(i + 1, len(zs) - 1)]),
                                 method="bounded", options={"xatol": 1e-13})
    zh = r.x
    h0 = H(zh)

    def edge(sign):
        y = 1e-4
        while H(zh + sign * y) > h0 - 60.0:
            y *= 1.6
        return optimize.brentq(lambda v: H(zh + sign * v) - h0 + 60.0, 0.0, y, xtol=1e-13)

    pts = np.unique(np.concatenate([np.linspace(zh - edge(-1), zh + edge(1), 81), [zh]]))
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        total += integrate.quad(lambda z: math.exp(H(z) - h0), a, b, epsabs=0.0, epsrel=1e-13, limit=400)[0]
    return math.log(total) + h0


def _two_reading_exact(h, rho):
    """P(X1 < h, X2 < h) for standard normals with correlation rho, by Owen's T: Phi(h) - 2 T(h, a)."""
    return special.ndtr(h) - 2.0 * special.owens_t(h, np.sqrt((1.0 - rho) / (1.0 + rho)))


def _fd(f, x, h):
    """Five-point central difference of f at x with step h (elementwise)."""
    return (8.0 * (f(x + h) - f(x - h)) - (f(x + 2 * h) - f(x - 2 * h))) / (12.0 * h)


def _close(got, want, rtol, atol=0.0):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert np.all(np.isfinite(got)), got
    bad = np.abs(got - want) > atol + rtol * np.abs(want)
    assert not bad.any(), (got[bad], want[bad])


# ---- kernel values -----------------------------------------------------------------------------------------

def test_gaussian_kernel_matches_scipy_density():
    eta = np.linspace(-1.0, 2.0, 7)
    logp, d1, d2 = K.kernel("linear", eta, {"y": 0.3}, {"sd": 0.2})
    _close(logp, stats.norm.logpdf(0.3, eta, 0.2), 1e-13)
    assert K.kernel("fraction", eta, {"y": 0.3}, {"sd": 0.2})[0].tolist() == logp.tolist()


def test_gaussian_cell_matches_multivariate_normal_of_the_calls():
    rng = np.random.default_rng(3)
    for n in (1, 2, 5):
        y = rng.normal(0.5, 0.1, n)
        eta, sd, tau = 0.42, 0.05, 0.13
        cov = sd ** 2 * np.eye(n) + tau ** 2 * np.ones((n, n))
        want = stats.multivariate_normal.logpdf(y, np.full(n, eta), cov)
        got = K.kernel("linear_cell", eta, {"n": n, "ybar": y.mean(), "ss": ((y - y.mean()) ** 2).sum()},
                       {"sd": sd, "tau": tau})[0]
        _close(got, want, 1e-12)


def test_probit_kernels_match_scipy_in_the_body():
    eta = np.linspace(-2.0, 2.0, 9)
    _close(K.kernel("binary", eta, {"y": 1}, {"sd": 0.7})[0], stats.norm.logcdf(eta / 0.7), 1e-13)
    _close(K.kernel("binary", eta, {"y": 0}, {"sd": 0.7})[0], stats.norm.logsf(eta / 0.7), 1e-13)
    lv = K.kernel("levels", eta, {"lo": 0.25, "hi": 0.75}, {"sd": 0.6})[0]
    _close(lv, np.log(stats.norm.cdf((0.75 - eta) / 0.6) - stats.norm.cdf((0.25 - eta) / 0.6)), 1e-12)
    items = K.kernel("items", eta, {"k": 3, "n": 7}, {"sd": 0.0})[0]
    _close(items, stats.binom.logpmf(3, 7, stats.norm.cdf(eta)), 1e-12)
    # one item: the call noise sums out exactly, P(pass) = Phi(eta / sqrt(1 + sd^2))
    _close(K.kernel("items", eta, {"k": 1, "n": 1}, {"sd": 0.5})[0], stats.norm.logcdf(eta / np.sqrt(1.25)), 1e-13)
    _close(K.kernel("items", eta, {"k": 0, "n": 1}, {"sd": 0.5})[0], stats.norm.logsf(eta / np.sqrt(1.25)), 1e-13)


def test_pairs_kernel_matches_the_model_definition():
    eta = np.linspace(-2.0, 2.0, 9)
    sd, w, pos = 0.8, 0.3, 0.25
    for o in (-1.0, 0.0, 1.0):
        x = eta + pos * o
        want = {0: stats.norm.sf((w - x) / sd), 1: stats.norm.cdf((w - x) / sd) - stats.norm.cdf((-w - x) / sd),
                2: stats.norm.cdf((-w - x) / sd)}
        for v in (0, 1, 2):
            got = K.kernel("pairs", eta, {"v": v, "o": o}, {"sd": sd, "width": w, "position": pos})[0]
            _close(got, np.log(want[v]), 1e-12)


def test_binary_is_the_open_interval_and_codes_name_kinds():
    eta = np.array([-40.0, -3.0, 0.1, 5.0, 60.0])
    for y, lo, hi in ((1, 0.0, np.inf), (0, -np.inf, 0.0), (1, 0.0, model.OPEN), (0, -model.OPEN, 0.0)):
        b = K.kernel_full("binary", eta, {"y": y}, {"sd": 0.9})
        lv = K.kernel_full(model.KIND_CODE["levels"], eta, {"lo": lo, "hi": hi}, {"sd": 0.9})
        for name in ("logp", "d1", "d2", "d3"):
            _close(getattr(lv, name), getattr(b, name), 1e-12, 1e-300)
        for k in range(3):
            _close(lv.dparam["sd"][k], b.dparam["sd"][k], 1e-11, 1e-300)
    with pytest.raises(ValueError):
        K.kernel("poisson", eta, {}, {})
    with pytest.raises(ValueError):
        K.kernel("levels", eta, {"lo": 0.0}, {"sd": 1.0})


# ---- tails ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("x", [-1e8, -1e4, -200.0, -40.0, -15.0, -10.0001, -9.9999, -8.0, -5.0, -1.0, 0.0, 2.0, 6.0])
def test_log_ndtr_derivatives_against_90_digit_reference(x):
    want = _logphi_ref(x)
    got = [float(v[0]) for v in K.log_ndtr_derivs(np.array([x]))]
    tol = (1e-14, 1e-13, 1e-12, 1e-8)
    for g, w, t in zip(got, want, tol):
        assert abs(g - w) <= t * abs(w) + 1e-300, (x, got, want)


def test_reference_methods_agree_where_they_meet():
    a, b = _logphi_ref(-8.0), _logphi_ref(-7.999999999999)
    assert abs(a[1] - b[1]) < 1e-10 and abs(a[3] - b[3]) < 1e-10


def test_kernels_stay_finite_and_concave_far_in_the_tails():
    eta = np.array([-1e6, -1e3, -30.0, 30.0, 1e3, 1e6])
    for kind, obs, params in (("binary", {"y": 1}, {"sd": 1e-3}), ("binary", {"y": 0}, {"sd": 1e-3}),
                              ("levels", {"lo": 0.25, "hi": 0.75}, {"sd": 1e-3}),
                              ("levels", {"lo": 0.75, "hi": model.OPEN}, {"sd": 0.01}),
                              ("items", {"k": 4, "n": 9}, {"sd": 0.0}),
                              ("pairs", {"v": 1, "o": 1.0}, {"sd": 0.05, "width": 0.2, "position": 0.1})):
        r = K.kernel_full(kind, eta, obs, params)
        for name in ("logp", "d1", "d2", "d3"):
            assert np.all(np.isfinite(getattr(r, name))), (kind, name, getattr(r, name))
        assert np.all(r.d2 <= 0.0), (kind, r.d2)
        for vals in r.dparam.values():
            assert all(np.all(np.isfinite(v)) for v in vals), kind
    # far below the cut of a top reading: log P ~ -x^2/2 - log(-x) - log(2 pi)/2, slope -> -x / sd
    r = K.kernel_full("levels", np.array([0.75 - 0.01 * 1e4]), {"lo": 0.75, "hi": np.inf}, {"sd": 0.01})
    want = _logphi_ref(-1e4)
    _close(r.logp, want[0], 1e-14)
    _close(r.d1, want[1] / 0.01, 1e-12)
    _close(r.d2, want[2] / 0.01 ** 2, 1e-12)


# ---- derivatives against finite differences ------------------------------------------------------------------

CASES = [
    ("linear", {"y": 0.4}, {"sd": 0.15}, np.linspace(-0.5, 1.5, 9)),
    ("linear_cell", {"n": 3, "ybar": 0.6, "ss": 0.02}, {"sd": 0.07, "tau": 0.12}, np.linspace(-0.5, 1.5, 9)),
    ("binary", {"y": 1}, {"sd": 0.8}, np.array([-12.0, -8.5, -8.0, -3.0, -0.2, 0.0, 0.7, 4.0, 9.0])),
    ("binary", {"y": 0}, {"sd": 0.3}, np.array([-2.0, -0.1, 0.5, 2.9, 3.05, 6.0])),
    ("levels", {"lo": 0.25, "hi": 0.75}, {"sd": 0.05}, np.array([-0.4, 0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.3, 1.5])),
    ("levels", {"lo": 0.75, "hi": np.inf}, {"sd": 0.01}, np.array([0.5, 0.6, 0.65, 0.7, 0.749, 0.76, 0.8])),
    ("levels", {"lo": -model.OPEN, "hi": 0.25}, {"sd": 0.1}, np.array([-0.5, 0.0, 0.3, 0.9, 1.2, 1.4])),
    ("levels", {"lo": 0.4, "hi": 0.6}, {"sd": 2.0}, np.array([-3.0, 0.0, 0.5, 4.0])),
    ("items", {"k": 3, "n": 10}, {"sd": 0.0}, np.array([-6.0, -2.0, -0.4, 0.0, 1.0, 5.0])),
    ("items", {"k": 0, "n": 4}, {"sd": 0.0}, np.array([-3.0, 0.0, 3.0, 12.0])),
    ("items", {"k": 1, "n": 1}, {"sd": 0.6}, np.array([-6.0, -2.0, 0.0, 1.0, 5.0])),
    ("items", {"k": 0, "n": 1}, {"sd": 1.5}, np.array([-3.0, 0.0, 3.0, 12.0])),
    ("items", {"k": 600, "n": 1000}, {"sd": 0.0}, np.array([-0.5, 0.2, 0.25, 0.3, 1.0])),
    ("pairs", {"v": 0, "o": 1.0}, {"sd": 0.7, "width": 0.2, "position": 0.3}, np.array([-9.0, -1.0, 0.0, 2.0])),
    ("pairs", {"v": 1, "o": -1.0}, {"sd": 0.4, "width": 0.3, "position": 0.2}, np.array([-6.0, -0.5, 0.0, 0.4, 5.0])),
    ("pairs", {"v": 2, "o": 1.0}, {"sd": 1.1, "width": 0.05, "position": -0.4}, np.array([-2.0, 0.0, 3.0, 11.0])),
    ("pairs", {"v": 1, "o": 0.0}, {"sd": 0.02, "width": 0.01, "position": 0.0},
     np.array([-0.3, -0.01, 0.0, 0.02, 0.2])),
]


@pytest.mark.parametrize("kind,obs,params,eta", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_derivatives_in_eta_match_finite_differences(kind, obs, params, eta):
    r = K.kernel_full(kind, eta, obs, params)
    h = 1e-3 * min([v for k, v in params.items() if k in ("sd", "width") and v > 0] or [1.0])
    if obs.get("n", 1) > 100:
        h = 1e-6

    def part(name):
        return lambda e: getattr(K.kernel_full(kind, e, obs, params), name)

    for got, name in ((r.d1, "logp"), (r.d2, "d1"), (r.d3, "d2")):
        want = _fd(part(name), eta, h)
        _close(got, want, 2e-6, 1e-7 * (np.abs(want).max() + 1.0))


@pytest.mark.parametrize("kind,obs,params,eta", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_derivatives_in_parameters_match_finite_differences(kind, obs, params, eta):
    r = K.kernel_full(kind, eta, obs, params)
    assert set(r.dparam) == set(K.PARAMS[kind])
    for p, value in params.items():
        if kind == "items" and obs["n"] > 1:   # sd is pinned at 0 there; the shared-noise marginal is even in sd
            assert all(np.all(v == 0.0) for v in r.dparam["sd"])
            continue
        h = 1e-4 * max(abs(value), 1e-2)

        def at(v, name, p=p):
            q = dict(params)
            q[p] = v
            return getattr(K.kernel_full(kind, eta, obs, q), name)

        for k, name in enumerate(("logp", "d1", "d2")):
            want = _fd(lambda v, name=name: at(v, name), value, h)
            _close(r.dparam[p][k], want, 2e-6, 1e-7 * (np.abs(want).max() + 1.0))


def test_tail_series_derivatives_match_finite_differences_of_log_ndtr():
    x = np.array([-10.5, -14.0, -30.0, -300.0])
    l0, l1, l2, l3 = K.log_ndtr_derivs(x)
    for got, f in ((l1, lambda v: special.log_ndtr(v)), (l2, lambda v: K.log_ndtr_derivs(v)[1]),
                   (l3, lambda v: K.log_ndtr_derivs(v)[2])):
        _close(got, _fd(f, x, 1e-3), 1e-6, 1e-12)


# ---- cell integrals --------------------------------------------------------------------------------------------

def _cells(kind, specs):
    """specs: list of (m, tau, sd, readings). Returns the flat arguments of cell_integral."""
    m, cell, tau, obs, par = [], [], [], {}, {"sd": []}
    for c, (mm, t, sd, readings) in enumerate(specs):
        tau.append(t)
        for ob in readings:
            m.append(mm)
            cell.append(c)
            par["sd"].append(sd)
            if kind == "levels":
                obs.setdefault("lo", []).append(ob[0])
                obs.setdefault("hi", []).append(ob[1])
            elif kind == "binary":
                obs.setdefault("y", []).append(ob)
            else:
                obs.setdefault("k", []).append(ob[0])
                obs.setdefault("n", []).append(ob[1])
    return m, obs, par, cell, tau


TOP, MID, LOW = (0.75, np.inf), (0.25, 0.75), (-np.inf, 0.25)


def test_sharp_two_top_readings_cell_matches_the_numerical_integral():
    ref = math.exp(_quad_ref("levels", 0.5, 0.1, 0.01, (TOP, TOP)))
    assert abs(ref / SHARP_CELL - 1.0) < 1e-6
    v = 0.1 ** 2 + 0.01 ** 2   # both readings above the cut: a bivariate normal orthant, exact by Owen's T
    assert abs(ref / _two_reading_exact((0.5 - 0.75) / math.sqrt(v), 0.1 ** 2 / v) - 1.0) < 1e-11
    args = _cells("levels", [(0.5, 0.1, 0.01, [TOP, TOP])])
    for method in ("composite", "auto"):
        r = K.cell_integral("levels", *args, method=method)
        assert abs(math.exp(r.logl[0]) / ref - 1.0) < 1e-10
        assert r.method[0] == "composite"
        assert abs(math.expm1(r.logl[0] - math.log(ref))) <= r.err[0] < 1e-7
    # plain adaptive Gauss-Hermite is far off here and its own estimate says so, which is what auto reads
    for n in (5, 9, 15):
        r = K.cell_integral("levels", *args, method="agh", nodes=n)
        assert r.err[0] > 1e-3
        assert abs(math.exp(r.logl[0]) / ref - 1.0) > 1e-3


def test_two_reading_wall_cells_match_owens_t_exactly():
    """Two readings on the same side of one cut (both top, both bottom, both 1, both 0): the cell probability is
    a bivariate normal orthant with variance tau^2 + sd^2 and correlation tau^2 / (tau^2 + sd^2)."""
    rows = []
    for m in (0.05, 0.3, 0.5, 0.62, 0.9):
        for tau in (0.02, 0.1, 0.3, 1.0):
            for sd in (0.001, 0.003, 0.01, 0.03, 0.1):
                if sd <= tau:
                    rows.append((m, tau, sd))
    m, tau, sd = (np.array(x) for x in zip(*rows))
    v = tau ** 2 + sd ** 2
    rho = tau ** 2 / v
    C = len(rows)
    cell = np.repeat(np.arange(C), 2)
    cases = [("levels", {"lo": np.full(2 * C, 0.75), "hi": np.full(2 * C, np.inf)}, (m - 0.75) / np.sqrt(v)),
             ("levels", {"lo": np.full(2 * C, -np.inf), "hi": np.full(2 * C, 0.25)}, (0.25 - m) / np.sqrt(v)),
             ("binary", {"y": np.ones(2 * C)}, (m - 0.4) / np.sqrt(v)),
             ("binary", {"y": np.zeros(2 * C)}, -(m - 0.4) / np.sqrt(v))]
    for kind, obs, h in cases:
        mm = np.repeat(m - 0.4, 2) if kind == "binary" else np.repeat(m, 2)
        p = _two_reading_exact(h, rho)
        keep = (h > -8.0) & (p > 1e-3 * special.ndtr(h))   # Phi(h) - 2T keeps at least 13 digits
        assert keep.sum() > 0.6 * C
        want = np.log(np.where(keep, p, 1.0))
        for method in ("composite", "auto"):
            r = K.cell_integral(kind, mm, obs, {"sd": np.repeat(sd, 2)}, cell, tau, method=method)
            err = np.where(keep, np.abs(np.expm1(r.logl - want)), 0.0)
            assert err.max() < 1e-9, (kind, method, rows[int(err.argmax())], err.max())
            assert np.all((err <= r.err) | (err < 1e-12)), (kind, method)


GRID = [(kind, m, tau, sd, readings)
        for kind, m, tau, sd, readings in
        [("levels", m, tau, sd, r) for m in (0.1, 0.5, 0.9) for tau in (0.03, 0.3) for sd in (0.003, 0.03, 0.3)
         for r in ([TOP, TOP], [MID, MID, MID], [LOW] * 5, [TOP, MID], [LOW, TOP])]
        + [("binary", m, tau, sd, r) for m in (-3.0, 0.0, 1.5) for tau in (0.1, 3.0) for sd in (0.1, 1.0)
           for r in ([1, 1], [0, 0, 0, 0, 0], [1, 0])]
        + [("items", m, tau, 0.0, r) for m in (-2.0, 2.0) for tau in (0.3, 3.0) for r in ([(0, 10)], [(7, 20)])]]


def _grid_errors(method, nodes=9):
    out = []
    for kind in ("levels", "binary", "items"):
        specs = [g[1:] for g in GRID if g[0] == kind]
        r = K.cell_integral(kind, *_cells(kind, specs), method=method, nodes=nodes)
        for c, spec in enumerate(specs):
            ref = _quad_ref(kind, *spec[:3], tuple(spec[3]))
            out.append((abs(math.expm1(r.logl[c] - ref)), float(r.err[c]), kind, spec[:3]))
    return out


def test_composite_and_auto_meet_their_error_estimates_on_a_grid():
    for method in ("composite", "auto"):
        rows = _grid_errors(method)
        worst = max(rows)
        assert worst[0] < 1e-8, (method, worst)
        under = [row for row in rows if row[0] > row[1] and row[0] > 1e-11]   # the reference keeps about 12 digits
        assert not under, (method, under[:5])


def test_gaussian_cells_are_exact_for_any_method():
    """Linear readings: the integral is the multivariate normal of the calls (tau^2 shared, sd^2 own)."""
    y = [[0.3, 0.35], [0.9, 0.1, 0.5], [0.42]]
    m, tau, sd = [0.4, 0.5, 0.45], [0.2, 0.05, 0.3], 0.07
    flat_m, cell, ys = [], [], []
    for c, row in enumerate(y):
        for v in row:
            flat_m.append(m[c])
            cell.append(c)
            ys.append(v)
    for method in ("agh", "composite"):
        r = K.cell_integral("linear", flat_m, {"y": ys}, {"sd": sd}, cell, tau, method=method, nodes=5)
        for c, row in enumerate(y):
            n = len(row)
            want = stats.multivariate_normal.logpdf(row, np.full(n, m[c]), sd ** 2 * np.eye(n) + tau[c] ** 2)
            assert abs(r.logl[c] - want) < 1e-10, (method, c)


def _frozen_fd(kind, m, obs, par, cell, tau, r, h=1e-6):
    rule = r.rule

    def logl(mm=None, tt=None, pp=None):
        return K.cell_integral(kind, m if mm is None else mm, obs, par if pp is None else pp, cell,
                               tau if tt is None else tt, rule=rule).logl

    m = np.asarray(m, dtype=float)
    N, C = len(m), len(tau)
    grad = np.zeros(N)
    hess = np.zeros((N, N))
    for i in range(N):
        e = np.zeros(N)
        e[i] = h
        grad[i] = (logl(mm=m + e) - logl(mm=m - e))[cell[i]] / (2 * h)
        for j in range(N):
            if cell[j] != cell[i]:
                continue
            f = np.zeros(N)
            f[j] = h * 10
            hess[i, j] = ((logl(mm=m + e * 10 + f) - logl(mm=m + e * 10 - f) - logl(mm=m - e * 10 + f)
                           + logl(mm=m - e * 10 - f))[cell[i]] / (4 * 100 * h * h))
    tau = np.asarray(tau, dtype=float)
    dtau = np.array([(logl(tt=tau + h * np.eye(C)[c]) - logl(tt=tau - h * np.eye(C)[c]))[c] / (2 * h)
                     for c in range(C)])
    shift = [sum((logl(mm=m + s * h * 10) * w) for s, w in ((2, -1), (1, 8), (-1, -8), (-2, 1))) / (12 * h * 10)]
    return grad, hess, dtau, shift[0]


@pytest.mark.parametrize("method", ["agh", "composite"])
def test_cell_derivatives_are_exact_for_the_frozen_rule(method):
    kind = "levels"
    m = [0.55, 0.6, 0.5, 0.3, 0.32, 0.8]
    cell = [0, 0, 0, 1, 1, 2]
    obs = {"lo": [0.75, 0.25, 0.75, -np.inf, 0.25, 0.25], "hi": [np.inf, 0.75, np.inf, 0.25, 0.75, 0.75]}
    par = {"sd": [0.05, 0.05, 0.05, 0.02, 0.02, 0.1]}
    tau = [0.1, 0.2, 0.05]
    first = K.cell_integral(kind, m, obs, par, cell, tau, method=method, nodes=7)
    r = K.cell_integral(kind, m, obs, par, cell, tau, rule=first.rule)   # the frozen sum and its derivatives
    assert list(r.method) == ["frozen"] * 3 and not r.deriv_ok.any() and np.isnan(r.err).all()
    _close(r.logl, first.logl, 1e-14, 1e-14)
    grad, hess, dtau, shift = _frozen_fd(kind, m, obs, par, cell, tau, r)
    _close(r.d1, grad, 1e-6, 1e-6)
    got = np.zeros((6, 6))
    got[r.hess_i, r.hess_j] = r.hess_v
    _close(got, hess, 1e-4, 1e-3)
    _close(r.dtau, dtau, 1e-6, 1e-6)
    _close(r.shift_d1, shift, 1e-6, 1e-6)
    _close(r.shift_d1, np.bincount(cell, r.d1), 1e-12, 1e-12)
    _close(r.shift_d2, np.bincount(np.asarray(cell)[r.hess_i], r.hess_v), 1e-10, 1e-10)
    # shift_d3: the change of shift_d2 along a shift, frozen rule
    d2_at = lambda x: K.cell_integral(kind, np.asarray(m) + x, obs, par, cell, tau, rule=r.rule).shift_d2
    _close(r.shift_d3, _fd(d2_at, 0.0, 1e-4), 1e-6, 1e-4)
    # the noise sd of each reading
    sd = np.asarray(par["sd"])
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1e-6
        f = lambda s: K.cell_integral(kind, m, obs, {"sd": s}, cell, tau, rule=r.rule).logl[cell[i]]
        _close(r.dparam["sd"][i], (f(sd + e) - f(sd - e)) / 2e-6, 1e-5, 1e-6)


def test_replaced_nodes_change_derivatives_by_the_quadrature_error_only():
    kind = "levels"
    args = _cells(kind, [(0.5, 0.1, 0.01, [TOP, TOP]), (0.4, 0.2, 0.05, [MID, TOP, MID])])
    r = K.cell_integral(kind, *args, method="composite")
    m = np.asarray(args[0])
    h = 1e-5
    up = K.cell_integral(kind, m + h, *args[1:], method="composite").logl
    dn = K.cell_integral(kind, m - h, *args[1:], method="composite").logl
    _close(r.shift_d1, (up - dn) / (2 * h), 1e-5, 1e-5)


def test_items_cell_integrates_the_call_noise():
    """One items call with its call noise as the latent term: k ~ Binomial(n, Phi(m + sd z))."""
    for m, s, k, n in ((0.3, 1.2, 4, 9), (-2.0, 0.4, 0, 6), (1.0, 6.0, 15, 15)):
        r = K.cell_integral("items", [m], {"k": [k], "n": [n]}, {"sd": [0.0]}, [0], [s])
        want = _quad_ref("items", m, s, 0.0, ((k, n),))
        assert abs(r.logl[0] - want) < 1e-9, (m, s, k, n)


def test_cell_integral_refuses_bad_cells():
    with pytest.raises(ValueError):
        K.cell_integral("binary", [0.0, 1.0], {"y": [1, 0]}, {"sd": [1.0, 1.0]}, [0, 2], [1.0, 1.0])
    with pytest.raises(ValueError):
        K.cell_integral("binary", [0.0], {"y": [1]}, {"sd": [1.0]}, [0], [-1.0])


def _pair_interval(v, o, width, position):
    """The model's verdict rule as an interval for d + e, d = eta_a - eta_b plus the shared term."""
    lo = {0: width, 1: -width, 2: -np.inf}[v] - position * o
    hi = {0: np.inf, 1: width, 2: -width}[v] - position * o
    return (lo, hi)


PAIR_CELLS = [  # (m = eta_a - eta_b, tau of the difference, sigma_pair, width, position, verdicts as (v, o))
    (0.4, 0.3, 0.02, 0.05, 0.1, ((0, 1), (0, -1))), (0.0, 1.0, 0.03, 0.1, 0.2, ((1, 1), (1, -1), (1, 1))),
    (-0.2, 0.5, 0.01, 0.02, 0.0, ((0, 1), (2, 1))), (1.5, 2.0, 0.3, 0.4, -0.3, ((2, -1),)),
    (0.1, 0.05, 0.5, 0.3, 0.1, ((1, 0), (0, 1), (2, -1), (1, 1))), (-1.0, 3.0, 0.05, 0.01, 0.05, ((0, 1),) * 4),
]


def _pair_args(cells):
    m, cell, tau, obs, par = [], [], [], {"v": [], "o": []}, {"sd": [], "width": [], "position": []}
    for c, (mm, t, sd, w, pos, verdicts) in enumerate(cells):
        tau.append(t)
        for v, o in verdicts:
            m.append(mm)
            cell.append(c)
            obs["v"].append(v)
            obs["o"].append(o)
            par["sd"].append(sd)
            par["width"].append(w)
            par["position"].append(pos)
    return m, obs, par, cell, tau


def test_pair_cells_match_the_numerical_integral_and_owens_t():
    args = _pair_args(PAIR_CELLS)
    for method in ("composite", "auto"):
        r = K.cell_integral("pairs", *args, method=method)
        for c, (mm, t, sd, w, pos, verdicts) in enumerate(PAIR_CELLS):
            ref = _quad_ref("levels", mm, t, sd, tuple(_pair_interval(v, o, w, pos) for v, o in verdicts))
            err = abs(math.expm1(r.logl[c] - ref))
            assert err < 1e-9 and (err <= r.err[c] or err < 1e-11), (method, c, err, r.err[c])
    # two verdicts for a with a shown first, repeated: a bivariate normal orthant
    for mm, t, sd, w, pos in ((0.3, 0.4, 0.01, 0.05, 0.1), (-0.1, 1.0, 0.1, 0.2, 0.3), (0.5, 0.2, 0.05, 0.02, -0.1)):
        r = K.cell_integral("pairs", [mm, mm], {"v": [0, 0], "o": [1, 1]},
                            {"sd": sd, "width": w, "position": pos}, [0, 0], [t])
        v = t * t + sd * sd
        want = _two_reading_exact((mm + pos - w) / math.sqrt(v), t * t / v)
        assert abs(math.exp(r.logl[0]) / want - 1.0) < 1e-10


@pytest.mark.parametrize("method", ["agh", "composite"])
def test_pair_cell_derivatives_are_exact_for_the_frozen_rule(method):
    m, obs, par, cell, tau = _pair_args(PAIR_CELLS[:3])
    first = K.cell_integral("pairs", m, obs, par, cell, tau, method=method, nodes=9)
    r = K.cell_integral("pairs", m, obs, par, cell, tau, rule=first.rule)
    N = len(m)
    m = np.asarray(m, dtype=float)
    lg = lambda mm=m, pp=par, tt=tau: K.cell_integral("pairs", mm, obs, pp, cell, tt, rule=r.rule).logl
    h = 1e-6
    for i in range(N):
        e = np.zeros(N)
        e[i] = h
        _close(r.d1[i], (lg(mm=m + e) - lg(mm=m - e))[cell[i]] / (2 * h), 1e-6, 1e-6)
        for name in ("sd", "width", "position"):
            up, dn = dict(par), dict(par)
            up[name] = np.asarray(par[name], dtype=float) + e
            dn[name] = np.asarray(par[name], dtype=float) - e
            _close(r.dparam[name][i], (lg(pp=up) - lg(pp=dn))[cell[i]] / (2 * h), 1e-5, 1e-6)
    t = np.asarray(tau, dtype=float)
    for c in range(len(tau)):
        f = np.zeros(len(tau))
        f[c] = h
        _close(r.dtau[c], (lg(tt=t + f) - lg(tt=t - f))[c] / (2 * h), 1e-6, 1e-6)


# ---- open ends, frozen rules, items counts -----------------------------------------------------------------

def test_the_open_sentinel_is_an_open_end_at_any_predictor():
    for eta, sd in ((1e6, 0.01), (2e6, 1.0), (5e5, 3.0), (1e6 - 0.5, 0.2)):
        r = K.kernel_full("levels", eta, {"lo": 0.75, "hi": model.OPEN}, {"sd": sd})
        ref = K.kernel_full("levels", eta, {"lo": 0.75, "hi": np.inf}, {"sd": sd})
        assert r.logp == 0.0 and r.d1 == 0.0, (eta, sd, r.logp, r.d1)
        for name in ("logp", "d1", "d2", "d3"):
            assert getattr(r, name) == getattr(ref, name)
        low = K.kernel_full("levels", -eta, {"lo": -model.OPEN, "hi": 0.25}, {"sd": sd})
        assert low.logp == 0.0 and low.d1 == 0.0
    # a large finite cut stays a cut: below OPEN_FROM it is a wall
    r = K.kernel("levels", 6e4, {"lo": 0.75, "hi": 5e4}, {"sd": 1e3})[0]
    _close(r, stats.norm.logcdf(-10.0), 1e-13)
    # the cell integrator and the kernel agree that an OPEN end has no wall
    args = ([1e6 - 0.5], {"lo": [0.75], "hi": [model.OPEN]}, {"sd": [0.01]}, [0], [1.0])
    r = K.cell_integral("levels", *args)
    assert abs(r.logl[0]) <= r.err[0] + 1e-300 and r.err[0] < 1e-12


def test_a_frozen_rule_carries_no_error_estimate():
    first = K.cell_integral("linear", [0.0], {"y": [0.0]}, {"sd": 1.0}, [0], [1.0])
    exact0 = stats.norm.logpdf(0.0, 0.0, math.sqrt(2.0))
    assert abs(first.logl[0] - exact0) <= first.err[0] < 1e-9 and first.deriv_ok[0]
    moved = K.cell_integral("linear", [8.0], {"y": [0.0]}, {"sd": 1.0}, [0], [1.0], rule=first.rule)
    exact = stats.norm.logpdf(0.0, 8.0, math.sqrt(2.0))
    assert abs(moved.logl[0] - exact) > 1e-2            # the nodes were placed for m = 0, not 8
    assert np.isnan(moved.err[0]) and moved.err_kind[0] == "none" and moved.method[0] == "frozen"
    assert not moved.deriv_ok[0]
    errs = moved.deriv_err
    assert all(np.isnan(errs[k]).all() for k in ("d1", "hess", "shift_d1", "shift_d2", "shift_d3", "dtau"))
    assert np.isnan(errs["dparam"]["sd"]).all()
    assert moved.mode[0] == first.mode[0] and moved.scale[0] == first.scale[0]   # placement, not the new mode
    fresh = K.cell_integral("linear", [8.0], {"y": [0.0]}, {"sd": 1.0}, [0], [1.0])
    assert abs(fresh.logl[0] - exact) <= fresh.err[0] < 1e-9


def test_the_items_of_one_call_share_its_noise():
    with pytest.raises(ValueError):
        K.kernel("items", 0.0, {"k": 2, "n": 2}, {"sd": 1.0})
    with pytest.raises(ValueError):
        K.cell_integral("items", [0.0], {"k": [2], "n": [2]}, {"sd": [0.5]}, [0], [1.0])
    with pytest.raises(ValueError):
        K.kernel("items", 0.0, {"k": 3, "n": 2}, {"sd": 0.0})
    # eta = 0 and one shared N(0, 1) call error: Phi(e) is uniform on (0, 1), so the count is uniform on 0..n
    for k, n in ((2, 2), (0, 2), (1, 2), (20, 20), (7, 20), (0, 1), (1, 1)):
        r = K.cell_integral("items", [0.0], {"k": [k], "n": [n]}, {"sd": [0.0]}, [0], [1.0])
        assert abs(math.expm1(r.logl[0] + math.log(n + 1))) <= r.err[0] < 1e-9, (k, n, r.logl[0], r.err[0])


@pytest.mark.parametrize("n,k", [(1e8, 1), (1e8, 5e7), (1e8, 1e8 - 1), (1e8, 3e7), (1e5, 5e4), (1e5, 1),
                                 (1e12, 3e11), (1e14, 1e14), (1e14, 0)])
def test_the_uniform_count_is_exact_for_large_counts(n, k):
    r = K.cell_integral("items", [0.0], {"k": [k], "n": [n]}, {"sd": [0.0]}, [0], [1.0])
    err = abs(math.expm1(r.logl[0] + math.log1p(n)))
    assert err <= r.err[0] < 1e-8, (n, k, err, r.err[0], r.method[0])


def _log_factorial_dec(n: int) -> Decimal:
    """log n! as a decimal: the sum of logs up to 2000, the Stirling series above (error below 1e-40 there)."""
    if n <= 2000:
        return sum((Decimal(j).ln() for j in range(2, n + 1)), Decimal(0))
    N = Decimal(n)
    s = N * N.ln() - N + (2 * _PI * N).ln() / 2
    for num, den, p in ((1, 12, 1), (-1, 360, 3), (1, 1260, 5), (-1, 1680, 7), (1, 1188, 9)):
        s += Decimal(num) / (Decimal(den) * N ** p)
    return s


@pytest.mark.parametrize("n,k,eta", [(1000, 1, -3.0), (1000, 500, 0.01), (1000, 999, 3.1), (1000, 0, -0.5),
                                     (100000000, 1, -5.6), (100000000, 50000000, 1e-4),
                                     (100000000, 99999999, 5.6), (100000000, 30000000, -0.52),
                                     (100000000, 30000000, -0.5244)])
def test_items_kernel_against_a_decimal_reference(n, k, eta):
    with localcontext() as ctx:
        ctx.prec = 60
        X = Decimal(eta)
        ref = (_log_factorial_dec(n) - _log_factorial_dec(k) - _log_factorial_dec(n - k)
               + k * _logphi_dec(X, 60)[0] + (n - k) * _logphi_dec(-X, 60)[0])
    r = K.kernel_full("items", eta, {"k": k, "n": n}, {"sd": 0.0})
    err = abs(float(Decimal(float(r.logp)) - ref))
    assert err <= float(r.logp_err) < 1e-6, (n, k, eta, err, float(r.logp_err))


def _complement_ref(n, m, tau):
    """log P(k = n of n) for one items call with its noise as the cell term: log(1 - v), where the small
    complement v = integral phi(z) (1 - Phi(m + tau z)^n) dz is integrated on fixed panels by quad."""
    def f(z):
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) * -math.expm1(n * float(special.log_ndtr(m + tau * z)))

    pts = np.arange(-16.0, 16.01, 0.5)
    v = math.fsum(integrate.quad(f, a, b, epsabs=1e-22, epsrel=1e-11)[0] for a, b in zip(pts[:-1], pts[1:]))
    return math.log1p(-v)


def test_all_items_pass_cells_are_routed_where_the_powered_term_turns():
    """k = n: Phi(eta)^n turns from 0 to 1 at eta = -Phi^-1(1/n), not at 0. A large count there sent the cell to
    adaptive Gauss-Hermite, whose estimate understated its error (n = 1e5, m = 14.5, tau = 2)."""
    r = K.cell_integral("items", [14.5], {"k": [100000], "n": [100000]}, {"sd": 0.0}, [0], [2.0])
    ref = -2.8668848210239774e-7       # the complement on panels of width 1 and 0.25 agrees to 16 digits
    assert abs(math.expm1(r.logl[0] - ref)) <= r.err[0] < 1e-9 and r.method[0] == "composite"
    rows = [(n, t, float(-special.ndtri(1.0 / n) + h * t)) for n in (1e3, 1e5, 1e8, 1e14) for t in (0.1, 0.5, 2.0)
            for h in (3.0, 4.0, 5.0, 6.0)]
    n, t, m = (np.array(x) for x in zip(*rows))
    r = K.cell_integral("items", m, {"k": n, "n": n}, {"sd": 0.0}, np.arange(len(rows)), t)
    for c, (nn, tt, mm) in enumerate(rows):
        err = abs(math.expm1(r.logl[c] - _complement_ref(nn, mm, tt)))
        assert err <= r.err[c] + 1e-13, (rows[c], err, r.err[c], r.method[c])   # quad keeps about 13 digits here


def test_error_kinds_name_what_the_estimate_is():
    args = _cells("levels", [(0.5, 0.1, 0.01, [TOP, TOP]), (0.5, 0.1, 0.3, [MID])])
    assert list(K.cell_integral("levels", *args, method="agh").err_kind) == ["heuristic", "heuristic"]
    assert list(K.cell_integral("levels", *args, method="composite").err_kind) == ["kronrod", "kronrod"]
    r = K.cell_integral("levels", *args)
    assert list(r.method) == ["composite", "agh"] and list(r.err_kind) == ["kronrod", "heuristic"]


# ---- narrow intervals --------------------------------------------------------------------------------------

def test_narrow_tie_mixed_derivatives_are_exact():
    """A tie of width 1e-8 or 1e-9 at noise sd 0.01: by symmetry d1 and d(d1)/d width are 0; the second mixed
    derivative is from a 70-digit series of the integral of exp(-y^2 / (2 sd^2)) over (-width, width)."""
    for w, want in ((1e-8, 0.6666666666664889), (1e-9, 0.06666666666666649)):
        r = K.kernel_full("pairs", 0.0, {"v": 1, "o": 1}, {"sd": 0.01, "width": w, "position": 0.0})
        assert r.d1 == 0.0 and r.dparam["width"][1] == 0.0 and r.d3 == 0.0
        assert abs(r.dparam["width"][2] - want) < 1e-12 * want, (w, r.dparam["width"][2])
        _close(r.dparam["width"][0], 1.0 / w, 1e-12)
        r2 = K.kernel_full("pairs", np.array([-0.003, 0.003]), {"v": 1, "o": 1},
                           {"sd": 0.01, "width": w, "position": 0.0})
        assert r2.d1[0] == -r2.d1[1] and r2.dparam["width"][2][0] == r2.dparam["width"][2][1]


def _phi_cdf_dec(X: Decimal) -> Decimal:
    if X < -8:
        l0 = _logphi_dec(X, 80)[0]
        return l0.exp()
    if X > 8:
        return 1 - _phi_cdf_dec(-X)
    return _logphi_dec(X, 80)[0].exp()


def _tie_logp_dec(x: Decimal, w: Decimal, s: Decimal) -> Decimal:
    return (_phi_cdf_dec((x + w) / s) - _phi_cdf_dec((x - w) / s)).ln()


@pytest.mark.parametrize("x,w,s", [(0.003, 1e-7, 0.01), (0.0, 2e-6, 0.01), (0.05, 1e-4, 0.02), (-0.3, 0.008, 0.1),
                                   (-0.3, 0.01, 0.1), (0.4, 0.3, 0.5)])
def test_tie_kernel_matches_decimal_derivatives_near_and_across_the_narrow_switch(x, w, s):
    """Every derivative of a tie (in eta, width and sd) against central differences in 80-digit arithmetic."""
    with localcontext() as ctx:
        ctx.prec = 80
        X, W, S = Decimal(x), Decimal(w), Decimal(s)
        d = Decimal("1e-14") * S

        def f(a=X, b=W, c=S):
            return _tie_logp_dec(a, b, c)

        def dx(g, k):   # k-th central difference in x of g(a)
            if k == 0:
                return g(X)
            if k == 1:
                return (g(X + d) - g(X - d)) / (2 * d)
            if k == 2:
                return (g(X + d) - 2 * g(X) + g(X - d)) / (d * d)
            return (g(X + 2 * d) - 2 * g(X + d) + 2 * g(X - d) - g(X - 2 * d)) / (2 * d ** 3)

        want = {"d1": dx(lambda a: f(a=a), 1), "d2": dx(lambda a: f(a=a), 2), "d3": dx(lambda a: f(a=a), 3)}
        for name, var in (("width", "b"), ("sd", "c")):
            base = {"b": W, "c": S}[var]
            for k in range(3):
                up = dx(lambda a: f(a=a, **{var: base + d}), k)
                dn = dx(lambda a: f(a=a, **{var: base - d}), k)
                want[f"{name}{k}"] = (up - dn) / (2 * d)
    r = K.kernel_full("pairs", x, {"v": 1, "o": 1}, {"sd": s, "width": w, "position": 0.0})
    got = {"d1": r.d1, "d2": r.d2, "d3": r.d3}
    for name in ("width", "sd"):
        for k in range(3):
            got[f"{name}{k}"] = r.dparam[name][k]
    scale = {"d1": 1 / s, "d2": 1 / s ** 2, "d3": 1 / s ** 3}
    for key, g in got.items():
        wv = float(want[key])
        k = int(key[-1]) if key[-1].isdigit() else int(key[1])
        floor = 1e-10 * (1 / s) ** (k + 1) if key[:1] in ("w", "s") else 1e-10 * scale[key]
        assert abs(float(g) - wv) <= 1e-9 * abs(wv) + floor, (key, float(g), wv)


# ---- cell derivatives against exact convolutions -------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _conv_ref(m, lo, hi, S):
    """One reading with lo < m + N(0, S^2) < hi, as decimals: log P and its derivatives in m (1..3), in S, in lo
    and in hi, from phi and Phi in 80-digit arithmetic (exact convolution of the cell term and the noise)."""
    with localcontext() as ctx:
        ctx.prec = 80
        M, Sd = Decimal(m), Decimal(S)
        two_pi = (2 * _PI).sqrt()

        def end(c, sign):
            """Phi and 1 - Phi at the standardised end, the density there, the end, and its side (+-inf open)."""
            if math.isinf(c):
                one, zero = Decimal(1), Decimal(0)
                return (one, zero, zero, zero, math.inf) if sign > 0 else (zero, one, zero, zero, -math.inf)
            a = (M - Decimal(c)) / Sd
            return _phi_cdf_dec(a), _phi_cdf_dec(-a), (-(a * a) / 2).exp() / two_pi, a, float(a)

        Pa, Qa, fa, a, xa = end(lo, 1)
        Pb, Qb, fb, b, xb = end(hi, -1)
        P = Pa - Pb if xa + xb <= 0 else Qb - Qa       # the side of 0 where the difference keeps its digits
        P1 = (fa - fb) / Sd
        P2 = -(a * fa - b * fb) / Sd ** 2
        P3 = ((a * a - 1) * fa - (b * b - 1) * fb) / Sd ** 3
        L1 = P1 / P
        L2 = P2 / P - L1 * L1
        L3 = P3 / P - 3 * L1 * P2 / P + 2 * L1 ** 3
        LS = -(a * fa - b * fb) / Sd / P
        return tuple(float(v) for v in (P.ln(), L1, L2, L3, LS, -fa / Sd / P, fb / Sd / P))


SINGLE = ([("binary", m, t, sd, 1) for m in (-10.0, -3.0, -1.0, 0.0, 2.0) for t in (0.03, 1.0, 3.0)
           for sd in (1e-8, 1e-4, 0.01, 0.3, 3.0)]
          + [("binary", m, t, sd, 0) for m, t, sd in ((-1.0, 1.0, 1e-4), (0.5, 1e-7, 1.0), (0.0, 1e-7, 1e-7))]
          + [("levels", m, t, sd, r) for m in (0.1, 0.5, 0.9) for t in (0.03, 0.3) for sd in (1e-6, 0.003, 0.1)
             for r in (TOP, MID)]
          + [("pairs", m, t, sd, (v, o, w)) for m in (-0.3, 0.2) for t in (0.1, 2.0) for sd in (1e-5, 0.02, 0.5)
             for v in (0, 1, 2) for o in (-1.0, 1.0) for w in (0.01,)])


def _single_args(rows):
    m = [r[1] for r in rows]
    tau = [r[2] for r in rows]
    kind = rows[0][0]
    if kind == "binary":
        return m, {"y": [r[4] for r in rows]}, {"sd": [r[3] for r in rows]}, list(range(len(rows))), tau
    if kind == "levels":
        return (m, {"lo": [r[4][0] for r in rows], "hi": [r[4][1] for r in rows]}, {"sd": [r[3] for r in rows]},
                list(range(len(rows))), tau)
    return (m, {"v": [r[4][0] for r in rows], "o": [r[4][1] for r in rows]},
            {"sd": [r[3] for r in rows], "width": [r[4][2] for r in rows], "position": [0.1] * len(rows)},
            list(range(len(rows))), tau)


def _single_ref(row):
    kind, m, t, sd, ob = row
    S = math.sqrt(t * t + sd * sd)
    if kind == "binary":
        lo, hi, x = (0.0, math.inf, m) if ob == 1 else (-math.inf, 0.0, m)
    elif kind == "levels":
        lo, hi, x = ob[0], ob[1], m
    else:
        v, o, w = ob
        lo, hi = {0: (w, math.inf), 1: (-w, w), 2: (-math.inf, -w)}[v]
        x = m + 0.1 * o
    L0, L1, L2, L3, LS, Llo, Lhi = _conv_ref(x, lo, hi, S)
    ref = {"logl": L0, "d1": L1, "hess": L2, "shift_d1": L1, "shift_d2": L2, "shift_d3": L3,
           "dtau": LS * t / S, "sd": LS * sd / S}
    if kind == "pairs":
        ref["position"] = o * L1
        ref["width"] = {0: Llo, 1: Lhi - Llo, 2: -Lhi}[v]
    return ref


@pytest.mark.parametrize("kind", ["binary", "levels", "pairs"])
def test_single_reading_cell_derivatives_match_the_exact_convolution(kind):
    """One reading per cell: the integral is the reading at noise sqrt(tau^2 + sd^2). Every derivative, with the
    nodes placed afresh, is within its own error estimate of the exact one; deriv_ok holds where it says so."""
    rows = [r for r in SINGLE if r[0] == kind]
    res = K.cell_integral(kind, *_single_args(rows))
    for c, row in enumerate(rows):
        ref = _single_ref(row)
        assert abs(math.expm1(res.logl[c] - ref["logl"])) <= res.err[c], (row, "logl")
        got = {"d1": res.d1[c], "hess": res.hess_v[c], "shift_d1": res.shift_d1[c], "shift_d2": res.shift_d2[c],
               "shift_d3": res.shift_d3[c], "dtau": res.dtau[c]}
        est = {k: res.deriv_err[k][c] for k in got}
        for name in res.dparam:
            got[name] = res.dparam[name][c]
            est[name] = res.deriv_err["dparam"][name][c]
        for key, g in got.items():
            assert abs(g - ref[key]) <= est[key], (row, key, g, ref[key], est[key], res.method[c])
    assert res.deriv_ok.mean() > 0.9


def test_sharp_single_reading_derivatives_where_moment_forms_cancel():
    """Exact convolution values (log Phi(m / sqrt(tau^2 + sd^2))) where the moment forms lost every digit."""
    for m, t, sd, key, want in ((-1.0, 1.0, 1e-4, "shift_d3", 0.11693119404795756),
                                (-1.0, 1.0, 1e-5, "shift_d3", 0.11693119539246769),
                                (-10.0, 3.0, 1e-8, "shift_d2", -0.104328660798724)):
        r = K.cell_integral("binary", [m], {"y": [1]}, {"sd": sd}, [0], [t])
        got = getattr(r, key)[0]
        assert abs(got - want) <= r.deriv_err[key][0] < 1e-7 * abs(want), (m, t, sd, got, want)
        assert r.deriv_ok[0]


def _orthant_derivs(h, rho, S):
    """P(X1 < h, X2 < h), correlation rho, and its derivatives in the two means (each a shift of h by 1/S)."""
    P = _two_reading_exact(h, rho)
    q = math.sqrt(1.0 - rho * rho)
    u = h * math.sqrt((1.0 - rho) / (1.0 + rho))
    ph = stats.norm.pdf(h)
    P1 = ph * stats.norm.cdf(u) / S
    P11 = (-h * ph * stats.norm.cdf(u) - ph * stats.norm.pdf(u) * rho / q) / S ** 2
    P12 = math.exp(-h * h / (1.0 + rho)) / (2.0 * math.pi * q) / S ** 2
    d1 = P1 / P
    return d1, P11 / P - d1 * d1, P12 / P - d1 * d1


def test_two_reading_hessian_matches_the_bivariate_orthant():
    """Two readings on one side of a wall with the same m: d1, H_11 and H_12 from the orthant P = Phi(h) - 2 T(h, a)
    and its textbook derivatives. Each is within its error estimate (plus the double-precision reference's 1e-11)."""
    rows = [(m, t, sd) for m in (-0.5, 0.0, 0.3) for t in (0.1, 1.0) for sd in (0.01, 0.1, 0.5)]
    for kind, obs in (("binary", {"y": [1, 1] * len(rows)}),
                      ("levels", {"lo": [0.0, 0.0] * len(rows), "hi": [np.inf, np.inf] * len(rows)})):
        m = np.repeat([r[0] for r in rows], 2)
        sd = np.repeat([r[2] for r in rows], 2)
        r = K.cell_integral(kind, m, obs, {"sd": sd}, np.repeat(np.arange(len(rows)), 2), [x[1] for x in rows])
        H = np.zeros((2 * len(rows), 2 * len(rows)))
        E = np.zeros_like(H)
        H[r.hess_i, r.hess_j] = r.hess_v
        E[r.hess_i, r.hess_j] = r.deriv_err["hess"]
        for c, (mm, t, s) in enumerate(rows):
            S = math.sqrt(t * t + s * s)
            d1, h11, h12 = _orthant_derivs(mm / S, t * t / (S * S), S)
            i = 2 * c
            for got, want, est in ((r.d1[i], d1, r.deriv_err["d1"][i]), (r.d1[i + 1], d1, r.deriv_err["d1"][i + 1]),
                                   (H[i, i], h11, E[i, i]), (H[i, i + 1], h12, E[i, i + 1]),
                                   (H[i + 1, i], h12, E[i + 1, i])):
                assert abs(got - want) <= est + 1e-11 * (abs(want) + 1.0 / S ** 2), (kind, rows[c], got, want, est)
            assert abs(r.shift_d2[c] - (2 * h11 + 2 * h12)) <= r.deriv_err["shift_d2"][c] + 1e-11 / S ** 2



ONE_NODE_CELL = ("binary", [1.0, 1.0], {"y": [1, 1]}, {"sd": 3.0}, [0, 0], [1.0])


@pytest.mark.parametrize("method", ["auto", "agh", "composite"])
def test_a_rule_is_never_compared_with_itself(method):
    """An agh estimate is the difference to a lower rule, so that rule must exist and differ: rule orders are
    positive integers and 1 <= low_nodes < nodes. Three nodes against three would estimate 0 on a cell they get
    wrong in the sixth digit of the gradient; that and every other malformed order is refused."""
    for kw in ({"nodes": 3, "low_nodes": 3}, {"nodes": 3, "low_nodes": 5}, {"nodes": 5, "low_nodes": 0},
               {"nodes": 0}, {"nodes": -2}, {"nodes": 2.5}, {"nodes": 9.0}, {"nodes": True},
               {"nodes": 9, "low_nodes": 4.0}, {"nodes": 9, "low_nodes": False}, {"nodes": "9"}):
        with pytest.raises(ValueError):
            K.cell_integral(*ONE_NODE_CELL, method=method, **kw)
    r = K.cell_integral(*ONE_NODE_CELL, method="agh", nodes=np.int64(3), low_nodes=np.int64(1))
    assert r.err_kind[0] == "heuristic" and np.isfinite(r.err[0]) and r.err[0] > 0.0


def test_one_node_claims_nothing_and_auto_integrates_the_cell():
    """With one node there is no lower rule: agh returns the one-node values with no error claim, and auto
    sends the cell to composite. Two positive binary readings with call noise 3 and term 1 are two calls,
    bivariate normal with variance 10 and correlation 0.1: the reference is the orthant (Owen's T)."""
    S = math.sqrt(10.0)
    P = _two_reading_exact(1.0 / S, 0.1)
    assert abs(P - 0.4039752749432193) < 1e-15
    d1, h11, h12 = _orthant_derivs(1.0 / S, 0.1, S)
    raw = K.cell_integral(*ONE_NODE_CELL, method="agh", nodes=1)
    assert raw.method[0] == "agh" and raw.err_kind[0] == "none" and raw.evals[0] == 1
    assert np.isnan(raw.err[0]) and not raw.deriv_ok[0]
    for k in ("d1", "hess", "shift_d1", "shift_d2", "shift_d3", "dtau"):
        assert np.all(np.isnan(raw.deriv_err[k])), k
    assert all(np.all(np.isnan(v)) for v in raw.deriv_err["dparam"].values())
    assert abs(raw.d1[0] - d1) > 1e-3 and abs(raw.logl[0] - math.log(P)) > 1e-4   # far off, so it claims nothing
    for kw in ({}, {"method": "auto"}):
        r = K.cell_integral(*ONE_NODE_CELL, nodes=1, **kw)
        assert r.method[0] == "composite" and r.err_kind[0] == "kronrod"
        assert abs(math.expm1(r.logl[0] - math.log(P))) <= r.err[0] < 1e-7
        H = np.zeros((2, 2))
        E = np.zeros((2, 2))
        H[r.hess_i, r.hess_j] = r.hess_v
        E[r.hess_i, r.hess_j] = r.deriv_err["hess"]
        for got, want, est in ((r.d1[0], d1, r.deriv_err["d1"][0]), (r.d1[1], d1, r.deriv_err["d1"][1]),
                               (H[0, 0], h11, E[0, 0]), (H[1, 1], h11, E[1, 1]),
                               (H[0, 1], h12, E[0, 1]), (H[1, 0], h12, E[1, 0])):
            assert abs(got - want) <= est + 1e-12, (got, want, est)
        assert r.deriv_ok[0]


@pytest.mark.parametrize("kw", [{"nodes": 2}, {"nodes": 3}, {"nodes": 9}, {"nodes": 9, "low_nodes": 2}])
def test_agh_never_certifies_derivatives_it_gets_wrong(kw):
    """Two agh rules can miss the same wall and agree on wrong derivatives (the gradient near 0 where it is
    0.26), so agh certifies derivatives only where its value meets tol and no wall is sharp. Two positive binary
    readings, against the orthant; cells with probability below 1e-6 are left out (the double-precision
    orthant loses digits there)."""
    rows = [(m, t, sd) for m in (-0.5, 0.0, 0.5, 1.0, 2.5) for t in (0.1, 0.5, 1.0, 2.0, 4.0)
            for sd in (0.05, 0.3, 1.0, 3.0, 10.0)]
    m = np.repeat([r[0] for r in rows], 2)
    r = K.cell_integral("binary", m, {"y": [1] * len(m)}, {"sd": np.repeat([x[2] for x in rows], 2)},
                        np.repeat(np.arange(len(rows)), 2), [x[1] for x in rows], method="agh", **kw)
    H = np.zeros((len(m), len(m)))
    H[r.hess_i, r.hess_j] = r.hess_v
    wrong = 0
    for c, (mm, t, s) in enumerate(rows):
        S = math.sqrt(t * t + s * s)
        if _two_reading_exact(mm / S, t * t / (S * S)) < 1e-6:
            continue
        d1, h11, h12 = _orthant_derivs(mm / S, t * t / (S * S), S)
        Di = abs(d1) + math.sqrt(abs(h11)) + 1.0 / max(t, s)
        i = 2 * c
        bad = (abs(r.d1[i] - d1) > 1e-6 * Di or abs(H[i, i] - h11) > 1e-6 * Di * Di
               or abs(H[i, i + 1] - h12) > 1e-6 * Di * Di)
        wrong += bad
        assert not (bad and r.deriv_ok[c]), (rows[c], r.d1[i], d1)
    assert wrong > 0   # the grid does reach cells that agh gets wrong

@pytest.mark.slow
def test_grid_table_for_the_report(capsys):
    """Worst relative error of each method on the grid (printed, for the record)."""
    with capsys.disabled():
        for method, nodes in (("agh", 5), ("agh", 9), ("agh", 15), ("composite", 9), ("auto", 9)):
            rows = _grid_errors(method, nodes)
            worst = max(rows)
            under = sum(1 for row in rows if row[0] > row[1] and row[0] > 1e-13)
            print(f"{method}{nodes if method == 'agh' else ''}: worst {worst[0]:.3e} at {worst[2]} {worst[3]}, "
                  f"estimate below the error in {under} of {len(rows)}")
