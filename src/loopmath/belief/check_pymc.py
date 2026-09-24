"""`fit --full`: a PyMC check of the cost and success heads (spec 04 section 4, `bayes` extra).

Each head is refit by NUTS on the fit's own design: the same rows, nodes, prior groups and
prior factors, with the scales integrated out under the hyperpriors that empirical Bayes
maximizes over (log-normal `phi` with sd `HYPER_SD` around the defaults; for the cost head a
log-normal `sigma` with sd `SIGMA_PRIOR_SD` around its start). Effects are non-centred
(`b = sd * z`). The comparison per node is the posterior mean and 80 percent interval of both;
the summary gives, per head, the share of nodes whose fitted mean lies inside the PyMC interval,
the largest mean gap in fitted standard deviations, the median ratio of interval widths, the
same share and ratio for the linear predictor of each distinct observed row (what predictions
use; nested nodes can trade off against each other, so node gaps alone overstate differences),
and the sampler's divergences and largest r-hat.

Sampling follows `fit_bayes.sample`: one process, pytensor's CVM backend, fixed seed. The table
goes to `fits/<fit>/pymc_check.json`, the summary to `meta.json` under `full`. A check, never
the default; importing this module without the extra raises ImportError.
"""

from __future__ import annotations

import time

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from .gaussian import HYPER_SD, SIGMA_PRIOR_SD, GaussianHead

HEADS = ("cost", "success")
SEED = 20260923


def _model(fitted: dict):
    X, y, w, prior, factors = fitted["design"]
    Xd = pytensor.shared(np.asarray(X.todense()), name="X")  # shared, not a constant: keeps compiling fast
    grouped = prior.group_idx >= 0
    gidx = np.where(grouped, prior.group_idx, 0)
    fixed_sd = np.sqrt(np.where(grouped, 1.0, prior.fixed_var))
    with pm.Model() as model:
        z = pm.Normal("z", 0.0, 1.0, shape=X.shape[1])
        if len(prior.groups):
            log_phi = pm.Normal("log_phi", np.log(prior.default_phi), HYPER_SD, shape=len(prior.groups))
            sd = pt.where(grouped, pt.exp(log_phi)[gidx], fixed_sd)
        else:
            sd = pt.as_tensor_variable(fixed_sd)
        b = pm.Deterministic("b", sd * z)
        eta = pt.dot(Xd, b)
        if fitted["engine"] == "gaussian":
            sigma0 = GaussianHead(X, y, w, prior, factors).sigma0  # the start EB's sigma prior centres on
            log_sigma = pm.Normal("log_sigma", np.log(sigma0), SIGMA_PRIOR_SD)
            pm.Normal("y", eta, pt.exp(log_sigma) / np.sqrt(w), observed=y)
        else:
            s = pm.math.sigmoid(eta)
            pm.Bernoulli("y", p=pt.clip(w * s + (1.0 - w) * (1.0 - s), 1e-9, 1 - 1e-9), observed=y)
        if factors is not None:
            pm.Normal("factors", pt.dot(pytensor.shared(np.asarray(factors.F.todense())), b), np.sqrt(factors.v), observed=factors.f)
    return model


def compare_head(fitted: dict, *, draws: int = 400, tune: int = 400, chains: int = 2, seed: int = SEED) -> dict:
    started = time.monotonic()
    model = _model(fitted)
    with pytensor.config.change_flags(mode="CVM"), model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed, progressbar=False,
                          target_accept=0.9, compute_convergence_checks=False)
    b = idata.posterior["b"].values.reshape(-1, len(fitted["node_ids"]))
    res = fitted["fit"]
    sd = np.sqrt(np.maximum(np.diag(res.cov()), 1e-300))
    lo, hi = res.mean - 1.2816 * sd, res.mean + 1.2816 * sd
    p_mean = b.mean(axis=0)
    p_lo, p_hi = np.percentile(b, [10, 90], axis=0)
    inside = (res.mean >= p_lo) & (res.mean <= p_hi)
    gap = np.abs(res.mean - p_mean) / sd
    ratio = (p_hi - p_lo) / np.maximum(hi - lo, 1e-300)
    rhat = _rhat(idata.posterior["b"].values) if chains > 1 else None
    rows = np.unique(np.asarray(fitted["design"][0].todense()), axis=0)  # the distinct observed rows
    r_mean = rows @ res.mean
    r_sd = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", rows, res.cov(), rows), 1e-300))
    r_lo, r_hi = np.percentile(b @ rows.T, [10, 90], axis=0)
    summary = {"nodes": len(p_mean), "inside": round(float(inside.mean()), 3), "max_gap_sd": round(float(gap.max()), 3),
               "median_width_ratio": round(float(np.median(ratio)), 3), "rows": len(rows),
               "rows_inside": round(float(((r_mean >= r_lo) & (r_mean <= r_hi)).mean()), 3),
               "rows_median_width_ratio": round(float(np.median((r_hi - r_lo) / (2 * 1.2816 * r_sd))), 3),
               "divergences": int(idata.sample_stats["diverging"].values.sum()),
               "max_rhat": round(float(rhat.max()), 3) if chains > 1 else None, "draws": draws, "chains": chains,
               "seconds": round(time.monotonic() - started, 1)}
    table = [{"node": n, "fit": [float(res.mean[j]), float(lo[j]), float(hi[j])],
              "pymc": [float(p_mean[j]), float(p_lo[j]), float(p_hi[j])]}
             for j, n in enumerate(fitted["node_ids"])]
    return {"summary": summary, "table": table}


def _rhat(x: np.ndarray) -> np.ndarray:
    """Gelman-Rubin r-hat per parameter from (chains, draws, p), without splitting chains; needs 2 chains."""
    n = x.shape[1]
    within = x.var(axis=1, ddof=1).mean(axis=0)
    between = n * x.mean(axis=1).var(axis=0, ddof=1)
    var = (n - 1) / n * within + between / n
    return np.sqrt(var / np.maximum(within, 1e-300))


def compare(fitted: dict, **sample_kw) -> dict:
    """The check over the cost and success heads that the fit has; `table` is popped by the fit writer."""
    out: dict = {"ran": True, "heads": {}, "table": {}}
    for name in HEADS:
        if name in fitted:
            r = compare_head(fitted[name], **sample_kw)
            out["heads"][name] = r["summary"]
            out["table"][name] = r["table"]
    return out
