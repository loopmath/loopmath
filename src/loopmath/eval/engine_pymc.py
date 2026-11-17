"""The sampler: the only module of `loopmath.eval` that may import pymc.

Not a command: `fit.py` calls `available()` and `sample(design.data, ...)`. pymc is imported inside
`sample` only, so importing this module costs nothing and works without the [bayes] extra.

The model, as sampled
---------------------
Quality. `q_r = mu_a + v_i + u_r`, with `v` and `u` drawn through the Cholesky factor of their
covariance (`full`, an LKJ prior with eta 2), through K factors plus a diagonal (`factor:K`), or
independent per dimension (`diag`).

A judge's view of a cell (judge j, run r, dimension d):

    eta = q_r[d] + bias_j[d] + family_j,f[d] + w_j,i[d] + tau_j[d] (sqrt(h_j) z_j,t + sqrt(1 - h_j) z_j,t,d)

`h_j` is `halo_target`: the share of the judge-by-output variance that is shared by the dimensions of
one target. The shared factor `z_j,t` is an explicit unknown. The part of one dimension, `z_j,t,d`,
is summed out wherever that is exact (see `model.py`), and explicit otherwise.

Readings, given `eta` and the call noise `e ~ N(0, sigma_j[d]^2)`:

- `linear`, `fraction`: `y = eta + e`. Cells of independent calls enter through the mean of their calls
  and the sum of squares around it, which is exact.
- `binary`: 1 with chance `Phi(eta / sigma)`.
- `levels`: the level whose interval holds `eta + e`, with fixed cut points and open outer intervals.
- A `binary` or `levels` cell with repeats keeps its judge-by-output term as an explicit unknown, so
  the likelihood is the model's own, with no numerical integration. When `sigma` is small that unknown
  sits between near-hard walls and the sampler can diverge there; the fit's convergence checks count
  the divergences and mark such a fit as not converged.
- `items`: `k ~ Binomial(n, Phi(eta + e))`. For n above 1 the call noise is an explicit unknown.
- A call of a `panel` judge shares the part `sqrt(halo_call_j) sigma z_call` of its noise.
- A pair: `z = eta_a - eta_b + position_j o + e_pair`, verdict a when `z > tie_width_j`, b when
  `z < -tie_width_j`, else a tie. Bias and the judge-by-input term cancel. Where the judge also gave
  `linear` readings of an output, its view of that output is drawn from its exact normal law given
  those readings, so the pair adds one standard normal unknown per output.

Units. Biases sum to 0 over the anchors of a dimension (over the judges with readings when it is not
anchored). On a `binary` dimension the anchors' mean log `sigma` is 0. On a pairs-only dimension the
anchors' mean log `sigma_pair` is 0 and the arm means sum to 0.

Priors. With s = 0.25 on a [0, 1] dimension and s = 1 on a probit or pairs-only dimension: arm means
N(centre, 2s); every sd (`sd_input`, `sd_run`, `tau`, `sigma`) half-normal(s); a person's `sigma` on a
`linear` or `fraction` dimension half-normal(0.04), as in the flow fit; `sigma` on a `levels` dimension
Gamma(2) with mean 0.8 s, and `sigma_pair` Gamma(2) with mean 1.6 s (see below); bias N(0, s); the sd
of family effects half-normal(0.4 s) and the effects sum to 0 over families; the judge-by-input sd
half-normal(0.6 s); halo shares Beta(1, 2), sampled as an angle; relative log sds N(0, 0.5).

Two priors are not half-normal. A reading that is cut at fixed points (a level, a verdict) has a
likelihood with near-hard walls when its call noise goes to 0, and a small study cannot rule that out.
The sampler then diverges. A Gamma(2) prior has the same mean as the half-normal it replaces and no
mass at 0, which keeps the walls soft. With enough data the two give the same answer.

Every chain starts at the same point, the centre of the priors, with no jitter. A jittered start puts
biases far from the data, and the first steps can then throw a chain to where every sd is 0 and it never
moves again.

Compilation is pinned to pytensor's classic CVM backend for the call, as `loopmath.fit_bayes` does:
the default backend needs a numba build that matches the installed numpy.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .commands import not_built

INSTALL = ("loopmath eval fit needs the optional modelling extra (PyMC). "
           "Install it with: pip install 'loopmath[bayes]'")
NAME = "pymc"
PERSON_SIGMA = 0.04   # the half-normal scale of a person's call noise on a linear or fraction dimension
MAX_CORES = 4         # chains run side by side on at most this many cores unless the caller says otherwise
TINY = 1.0e-300
LINEAR, BINARY, LEVELS, ITEMS = 0, 1, 2, 3   # as model.KIND_CODE
OUTPUTS = ("mu", "sd_input", "sd_run", "corr_input", "corr_run", "q_run", "bias", "tau", "sigma", "halo_target",
           "halo_call", "family", "sd_judge_input", "sigma_pair", "position", "tie_width")
_EMPTY_ARCHIVE = b"!<arch>\n"


def run(args: argparse.Namespace) -> int:
    """Not a command. `eval fit` calls `sample`."""
    return not_built(args)


def available() -> bool:
    """Whether the [bayes] extra is installed."""
    return importlib.util.find_spec("pymc") is not None


def _prepare() -> None:
    """Before pytensor is imported, on macOS: put an empty `libd64.a` on the link path.

    pytensor passes `-ld64` to the compiler there, and a clang that no longer knows the flag reads it as
    `-l d64`. A user's own `gcc__cxxflags` in PYTENSOR_FLAGS is left alone."""
    if sys.platform != "darwin" or "pytensor" in sys.modules:
        return
    flags = os.environ.get("PYTENSOR_FLAGS", "")
    stub = Path(tempfile.gettempdir()) / "loopmath-eval-link-stub"
    if "gcc__cxxflags" in flags or any(ch.isspace() or ch in '",' for ch in str(stub)):
        return
    try:
        stub.mkdir(parents=True, exist_ok=True)
        if not (stub / "libd64.a").is_file():
            (stub / "libd64.a").write_bytes(_EMPTY_ARCHIVE)
    except OSError:
        return
    os.environ["PYTENSOR_FLAGS"] = ",".join(x for x in (flags, f'gcc__cxxflags="-L{stub}"') if x)


def version() -> str:
    import pymc

    return str(pymc.__version__)


def sample(data: Mapping[str, Any], *, seed: int = 0, draws: int = 1000, tune: int | None = None, chains: int = 4,
           cores: int | None = None, target_accept: float = 0.95) -> dict:
    """Sample the model for the arrays of `model.Design.data`.

    Returns `{"draws": {name: array[chains, draws, ...]}, "divergences", "run_seconds", "version"}` with
    one array per name of `OUTPUTS`. A parameter the design has no use for comes back as sampled from
    its prior (or as zeros): `fit.py` writes NaN where the parameter does not exist. `tune` defaults to
    `draws`; `cores` to the number of chains, at most `MAX_CORES` and the machine's count."""
    _prepare()
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    d = data
    A, D, J, N, I, R = (int(d[name]) for name in ("A", "D", "J", "F", "I", "R"))
    scale = np.asarray(d["scale"], dtype=float)
    wide = np.broadcast_to(scale, (J, D)).copy()
    form, K = d["structure"], int(d["K"])
    if form == "full" and D < 2:
        form = "diag"
    tune = int(draws if tune is None else tune)
    cores = int(min(chains, MAX_CORES, os.cpu_count() or 1) if cores is None else cores)
    started = time.time()

    def phi(x):
        return 0.5 * pt.erfc(-x / np.sqrt(2.0))

    def lcdf(x):
        return pm.logcdf(pm.Normal.dist(0.0, 1.0), x)

    def interval(lo, hi):
        """log of Phi(hi) - Phi(lo), taken on the side where the difference keeps its digits."""
        p = pt.switch(pt.gt(lo, 0.0), phi(-lo) - phi(-hi), phi(hi) - phi(lo))
        return pt.log(pt.maximum(p, TINY))

    def take(vector, positions: np.ndarray):
        """`vector[positions]`, and 0 where the position is -1."""
        return vector[np.maximum(positions, 0)] * (positions >= 0).astype(float)

    with pytensor.config.change_flags(mode="CVM"), pm.Model() as model:
        mu_raw = pm.Normal("mu_raw", 0.0, 1.0, shape=(A, D))
        mu = d["mu_loc"] + 2.0 * scale * mu_raw
        if d["pairs_only"].any():   # arm means sum to 0 there: centre over arms
            centre = np.eye(A) - np.full((A, A), 1.0 / A)
            mu = pt.switch(np.broadcast_to(d["pairs_only"], (A, D)), pt.dot(centre, mu), mu)
        mu = pm.Deterministic("mu", mu)

        def level(name: str, rows: int):
            z = pm.Normal(f"z_{name}", 0.0, 1.0, shape=(rows, D))
            if form == "full":
                chol, corr, sds = pm.LKJCholeskyCov(f"chol_{name}", n=D, eta=2.0,
                                                    sd_dist=pm.HalfNormal.dist(sigma=scale, shape=D),
                                                    compute_corr=True)
                pm.Deterministic(f"sd_{name}", sds)
                pm.Deterministic(f"corr_{name}", corr)
                return pt.dot(z, chol.T)
            if form == "factor":
                loadings = pm.Normal(f"loadings_{name}", 0.0, 1.0, shape=(D, K)) * scale[:, None]
                own = pm.HalfNormal(f"own_{name}", sigma=scale, shape=D)
                factors = pm.Normal(f"factors_{name}", 0.0, 1.0, shape=(rows, K))
                cov = pt.dot(loadings, loadings.T) + pt.diag(own ** 2)
                sd = pm.Deterministic(f"sd_{name}", pt.sqrt(pt.diag(cov)))
                pm.Deterministic(f"corr_{name}", cov / pt.outer(sd, sd))
                return pt.dot(factors, loadings.T) + z * own
            sd = pm.HalfNormal(f"sd_{name}", sigma=scale, shape=D)
            return z * sd

        v = level("input", I)
        u = level("run", R)
        q = pm.Deterministic("q_run", mu[d["run_arm"]] + v[d["run_input"]] + u)

        bias_raw = pm.Normal("bias_raw", 0.0, 1.0, shape=(J, D)) * scale
        bias = pm.Deterministic("bias", bias_raw - pt.sum(bias_raw * d["bias_w"], axis=0))
        tau = pm.HalfNormal("tau", sigma=wide, shape=(J, D))
        tight = np.asarray(d["person_tight"], dtype=bool)
        sigma_free = pm.HalfNormal("sigma_free", sigma=np.where(tight, PERSON_SIGMA, wide), shape=(J, D))
        sigma_log = pm.Normal("sigma_log", 0.0, 0.5, shape=(J, D))
        sigma_rel = pt.exp(sigma_log - pt.sum(sigma_log * d["sigma_w"], axis=0))
        fixed = np.broadcast_to(np.asarray(d["sigma_fixed"], dtype=bool), (J, D))
        sigma_cut = pm.Gamma("sigma_cut", alpha=2.0, beta=2.0 / (0.8 * wide), shape=(J, D))
        cut = np.broadcast_to(np.asarray(d["levels_dim"], dtype=bool), (J, D))
        sigma = pm.Deterministic("sigma", pt.switch(fixed, sigma_rel, pt.switch(cut, sigma_cut, sigma_free)))

        def share(name: str):
            """A share in (0, 1) with a Beta(1, 2) prior, sampled as an angle: share = sin(angle)^2.

            Returns `(sqrt(share), share)`. The square root of a share sampled directly has an infinite
            slope at 0, which the sampler cannot follow; the sine of an angle has none."""
            angle = pm.Uniform(f"{name}_angle", 0.0, np.pi / 2.0, shape=J)
            pm.Potential(f"{name}_prior", pt.sum(3.0 * pt.log(pt.cos(angle)) + pt.log(pt.sin(angle))))
            return pt.sin(angle), pm.Deterministic(name, pt.sin(angle) ** 2)

        halo_on = np.asarray(d["halo_on"], dtype=float)
        root_h, halo = share("halo_target")
        root_h, h = root_h * halo_on, halo * halo_on
        root_call, halo_call = share("halo_call")
        family_on = np.asarray(d["family_on"], dtype=float)
        if family_on.any():
            family_sd = pm.HalfNormal("family_sd", sigma=0.4 * scale, shape=D)
            family_raw = pm.Normal("family_raw", 0.0, 1.0, shape=(J, N, D))
            family = (family_raw - family_raw.mean(axis=1, keepdims=True)) * family_sd * family_on[:, None, :]
            family = pm.Deterministic("family", family)
        else:
            family = None
        if d["n_ji"]:
            sd_ji = pm.HalfNormal("sd_judge_input", sigma=0.6 * wide, shape=(J, D))
            w = pm.Normal("z_ji", 0.0, 1.0, shape=int(d["n_ji"])) * sd_ji[d["ji_j"], d["ji_d"]]
        z_jt = pm.Normal("z_jt", 0.0, 1.0, shape=int(d["n_jt"])) if d["n_jt"] else None
        z_cell = pm.Normal("z_cell", 0.0, 1.0, shape=int(d["n_cell"])) if d["n_cell"] else None
        z_pc = pm.Normal("z_pc", 0.0, 1.0, shape=int(d["n_pc"])) if d["n_pc"] else None
        z_e = pm.Normal("z_e", 0.0, 1.0, shape=int(d["n_e"])) if d["n_e"] else None

        def view(r, dd, j, f, jt, ji=None):
            """The judge's view of a cell, without the part of one dimension."""
            m = q[r, dd]
            if ji is not None:   # a reading: bias and the judge-by-input term do not cancel
                m = m + bias[j, dd]
                if d["n_ji"]:
                    m = m + take(w, ji)
            if family is not None:
                m = m + family[j, f, dd]
            if z_jt is not None:
                m = m + tau[j, dd] * root_h[j] * take(z_jt, jt)
            return m

        def own_var(j, dd):
            return tau[j, dd] ** 2 * (1.0 - h[j])

        if len(d["s_r"]):
            j, dd, n = d["s_j"], d["s_d"], d["s_n"].astype(float)
            m = view(d["s_r"], dd, j, d["s_f"], d["s_jt"], d["s_ji"])
            s2 = sigma[j, dd] ** 2
            var = own_var(j, dd) + s2 / n
            pm.Potential("lik_cells", pt.sum(-0.5 * pt.log(var) - 0.5 * (d["s_ybar"] - m) ** 2 / var
                                             - 0.5 * (n - 1.0) * pt.log(s2) - 0.5 * d["s_ss"] / s2))
            gain = own_var(j, dd) / var   # the law of the part of one dimension, given the cell's readings
            given_mean, given_sd = gain * (d["s_ybar"] - m), pt.sqrt(own_var(j, dd) * (1.0 - gain))

        if z_cell is not None:
            part = pt.sqrt(own_var(d["cell_j"], d["cell_d"])) * z_cell
            linked = d["cell_suff"] >= 0
            if linked.any():
                at = np.maximum(d["cell_suff"], 0)
                part = pt.switch(linked, given_mean[at] + given_sd[at] * z_cell, part)

        if len(d["k_r"]):   # no layout builds these rows; a fixed grid of points is not accurate enough
            raise ValueError("engine_pymc: grouped cut readings are not supported; "
                             "build the design with loopmath.eval.model.build")

        if len(d["g_r"]):
            j, dd = d["g_j"], d["g_d"]
            m = view(d["g_r"], dd, j, d["g_f"], d["g_jt"], d["g_ji"])
            own = own_var(j, dd)
            explicit = (d["g_cell"] >= 0).astype(float)
            if z_cell is not None:
                m = m + take(part, d["g_cell"])
            s = sigma[j, dd]
            s2 = s ** 2
            if z_pc is not None:
                shared = (d["g_pc"] >= 0).astype(float)
                m = m + s * root_call[j] * take(z_pc, d["g_pc"])
                s2 = s2 * (1.0 - halo_call[j] * shared)
            var = s2 + own * (1.0 - explicit)
            sd = pt.sqrt(var)
            kind = d["g_kind"]
            rows = np.flatnonzero(kind == LINEAR)
            if len(rows):
                pm.Potential("lik_linear", pt.sum(-0.5 * pt.log(var[rows])
                                                  - 0.5 * (d["g_y"][rows] - m[rows]) ** 2 / var[rows]))
            rows = np.flatnonzero(kind == BINARY)
            if len(rows):
                x, y = m[rows] / sd[rows], d["g_y"][rows]
                pm.Potential("lik_binary", pt.sum(y * lcdf(x) + (1.0 - y) * lcdf(-x)))
            rows = np.flatnonzero(kind == LEVELS)
            if len(rows):
                pm.Potential("lik_levels", pt.sum(interval((d["g_lo"][rows] - m[rows]) / sd[rows],
                                                           (d["g_hi"][rows] - m[rows]) / sd[rows])))
            rows = np.flatnonzero((kind == ITEMS) & (d["g_e"] >= 0))
            if len(rows):
                x = m[rows] + sd[rows] * z_e[d["g_e"][rows]]
                k, n = d["g_k"][rows].astype(float), d["g_n"][rows].astype(float)
                pm.Potential("lik_items", pt.sum(k * lcdf(x) + (n - k) * lcdf(-x)))
            rows = np.flatnonzero((kind == ITEMS) & (d["g_e"] < 0))
            if len(rows):
                x = m[rows] / pt.sqrt(1.0 + var[rows])
                k, n = d["g_k"][rows].astype(float), d["g_n"][rows].astype(float)
                pm.Potential("lik_item", pt.sum(k * lcdf(x) + (n - k) * lcdf(-x)))

        has_pairs = bool(len(d["p_ra"]))
        if has_pairs:
            j, dd = d["p_j"], d["p_d"]
            eta_a = view(d["p_ra"], dd, j, d["p_fa"], d["p_jta"]) + part[d["p_ca"]]
            eta_b = view(d["p_rb"], dd, j, d["p_fb"], d["p_jtb"]) + part[d["p_cb"]]
            pair_free = pm.Gamma("sigma_pair_free", alpha=2.0, beta=2.0 / (1.6 * wide), shape=(J, D))
            pair_log = pm.Normal("sigma_pair_log", 0.0, 0.5, shape=(J, D))
            pair_rel = pt.exp(pair_log - pt.sum(pair_log * d["pair_w"], axis=0))
            only = np.broadcast_to(np.asarray(d["pairs_only"], dtype=bool), (J, D))
            sigma_pair = pm.Deterministic("sigma_pair", pt.switch(only, pair_rel, pair_free))
            pair_scale = np.asarray(d["pair_scale"], dtype=float)
            position = pm.Deterministic("position", pm.Normal("position_raw", 0.0, 1.0, shape=J) * pair_scale)
            tie_width = pm.HalfNormal("tie_width", sigma=pair_scale, shape=J)
            z = (eta_a - eta_b + position[j] * d["p_o"]) / sigma_pair[j, dd]
            width = tie_width[j] / sigma_pair[j, dd]
            verdict = d["p_v"]
            logp = pt.switch(np.equal(verdict, 0), lcdf(z - width),
                             pt.switch(np.equal(verdict, 2), lcdf(-z - width), interval(-width - z, width - z)))
            pm.Potential("lik_pairs", pt.sum(logp))

        keep = ["mu", "sd_input", "sd_run", "q_run", "bias", "tau", "sigma", "halo_target", "halo_call"]
        keep += ["corr_input", "corr_run"] if form != "diag" else []
        keep += ["family"] if family is not None else []
        keep += ["sd_judge_input"] if d["n_ji"] else []
        keep += ["sigma_pair", "position", "tie_width"] if has_pairs else []
        quiet = logging.getLogger("pymc")
        before = quiet.level
        quiet.setLevel(logging.ERROR)
        try:
            idata = pm.sample(draws=int(draws), tune=tune, chains=int(chains), cores=cores, random_seed=int(seed),
                              progressbar=False, target_accept=float(target_accept), var_names=keep,
                              init="adapt_diag", compute_convergence_checks=False)
        finally:
            quiet.setLevel(before)

    posterior = idata.posterior
    out = {name: np.asarray(posterior[name], dtype=float) for name in keep}
    lead = out["mu"].shape[:2]
    if form == "diag":
        for name in ("corr_input", "corr_run"):
            out[name] = np.broadcast_to(np.eye(D), lead + (D, D)).copy()
    if family is None:
        out["family"] = np.zeros(lead + (J, N, D))
    if not d["n_ji"]:
        out["sd_judge_input"] = np.zeros(lead + (J, D))
    if not has_pairs:
        out["sigma_pair"] = np.full(lead + (J, D), np.nan)
        out["position"] = np.full(lead + (J,), np.nan)
        out["tie_width"] = np.full(lead + (J,), np.nan)
    divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())
    return {"draws": out, "divergences": divergences, "run_seconds": float(time.time() - started),
            "version": str(pm.__version__)}
