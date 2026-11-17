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
    """log Phi(x) and its first three derivatives in 90-digit decimal arithmetic.

    |x| < 8: Phi = 1/2 + phi(x) sum x^(2n+1) / (2n+1)!!. x <= -8: the continued fraction of the Mills ratio,
    R(t) = 1 / (t + 1 / (t + 2 / (t + 3 / ...))). Neither is the asymptotic series the module uses."""
    with localcontext() as ctx:
        ctx.prec = prec
        X = Decimal(x)
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
        return float(l0), float(lam), float(l2), float(l3)


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
    items = K.kernel("items", eta, {"k": 3, "n": 7}, {"sd": 0.5})[0]
    _close(items, stats.binom.logpmf(3, 7, stats.norm.cdf(eta / np.sqrt(1.25))), 1e-12)


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
    ("items", {"k": 3, "n": 10}, {"sd": 0.6}, np.array([-6.0, -2.0, -0.4, 0.0, 1.0, 5.0])),
    ("items", {"k": 0, "n": 4}, {"sd": 1.5}, np.array([-3.0, 0.0, 3.0, 12.0])),
    ("pairs", {"v": 0, "o": 1.0}, {"sd": 0.7, "width": 0.2, "position": 0.3}, np.array([-9.0, -1.0, 0.0, 2.0])),
    ("pairs", {"v": 1, "o": -1.0}, {"sd": 0.4, "width": 0.3, "position": 0.2}, np.array([-6.0, -0.5, 0.0, 0.4, 5.0])),
    ("pairs", {"v": 2, "o": 1.0}, {"sd": 1.1, "width": 0.05, "position": -0.4}, np.array([-2.0, 0.0, 3.0, 11.0])),
    ("pairs", {"v": 1, "o": 0.0}, {"sd": 0.02, "width": 0.01, "position": 0.0},
     np.array([-0.3, -0.01, 0.0, 0.02, 0.2])),
]


@pytest.mark.parametrize("kind,obs,params,eta", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_derivatives_in_eta_match_finite_differences(kind, obs, params, eta):
    r = K.kernel_full(kind, eta, obs, params)
    h = 1e-3 * min(v for k, v in params.items() if k in ("sd", "width") and v > 0)

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
    r = K.cell_integral(kind, m, obs, par, cell, tau, method=method, nodes=7)
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
    r = K.cell_integral("pairs", m, obs, par, cell, tau, method=method, nodes=9)
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
