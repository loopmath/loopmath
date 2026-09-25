"""Optional Bayesian model construction and sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fit_pricing import EFFORT_ORDER

__all__ = ["BAYES_HINT", "SEED", "require_bayes", "build_model", "sample"]


BAYES_HINT = (
    "loopmath research fit and research transfer-test need the optional "
    "modelling extra (PyMC); the default engine behind loopmath fit does not. "
    "Install it with: pip install 'loopmath[bayes]'"
)

from .research_defaults import SEED  # noqa: F401  (re-exported)


def require_bayes() -> None:
    """Import pymc; raise SystemExit(BAYES_HINT) with a clear message when absent."""
    try:
        import pymc  # noqa: F401
    except ImportError as exc:
        raise SystemExit(BAYES_HINT) from exc


def build_model(df: pd.DataFrame):
    """log10(usd_run_total) ~ StudentT(nu, alpha_m + f(effort) + task_effect, sigma).

    The modelled response is `usd_run_total`: the whole session set's priced
    cost for the run, planner and reviewer attempts included, not just the
    developer attempt (see the module docstring's CORRECTION paragraph for
    why this replaced the developer-only `usd` figure). `df["usd_run_total"]`
    must be populated and strictly positive for every row passed in; callers
    building `df` from `assemble_table` already guarantee this on this
    corpus (0 of 277 rows null it), but a caller passing a hand-built frame
    with a null or non-positive `usd_run_total` will hit `log10` of a bad
    value, on purpose -- this never guesses a run total.

    Returns (model, coords_and_index_maps).

    - `alpha_m`: per-model intercept, `Normal(0, 1.5)`.
    - `f(e)`: a shared effort curve over the fixed order `EFFORT_ORDER`,
      built as a cumulative sum of non-negative (`HalfNormal`) increments,
      with the first level pinned at 0 for identifiability. SPEC calls this
      "monotone-ish"; this implementation enforces it as *strictly*
      monotone (non-decreasing) by construction. That is the conservative
      reading: it never lets a higher-effort setting look cheaper in the
      shared curve, and any real non-monotonicity for a specific model shows
      up in that model's own `beta_m` term instead of contaminating the
      curve every model shares.
    - `beta_m`: a per-model multiplier on the shared curve, `Normal(0, tau)`
      with `tau ~ HalfNormal(0.2)`. 0.2 is a small scale on the log10(usd)
      axis (a one-sigma draw scales the shared curve's rise by roughly
      +-20%), which is the "tight shrinkage" SPEC asks for: a model needs
      real evidence in its own data to pull away from the shared curve.
    - `task_effect`: per-task offset, `Normal(0, tau_task)` with
      `tau_task ~ HalfNormal(0.5)` (its own, looser scale -- tasks are
      expected to differ more than models' effort-response shapes do).
    - `sigma ~ HalfNormal(1.0)`; `nu`, the StudentT degrees of freedom, is
      `1 + Gamma(2, 0.1)` (SPEC's suggested "Gamma(2, 0.1) shifted"): shifted
      by 1 so nu never gets close enough to 0 to make the tails pathological,
      while the Gamma(2, 0.1) part still lets the data pull nu down toward
      heavy-tailed territory when the cost data calls for it.
    - Bernoulli acceptance head: `accepted ~ Bernoulli(logit = gamma_m + g(effort)
      + task_gamma)`, sharing the same model/effort/task coords as the cost
      head, built the same way (own shared curve `g`, own per-task offset).
    - No harness anchor: SPEC section 7 forbids a claude-vs-openai family
      term, so `family` never enters either head, by design -- not an
      oversight.
    """
    require_bayes()
    import pymc as pm
    import pytensor.tensor as pt

    models_sorted = sorted(df["model"].unique())
    tasks_sorted = sorted(df["task"].unique())
    effort_levels = list(EFFORT_ORDER)

    model_idx_map = {m: i for i, m in enumerate(models_sorted)}
    effort_idx_map = {e: i for i, e in enumerate(effort_levels)}
    task_idx_map = {t: i for i, t in enumerate(tasks_sorted)}

    m_idx = df["model"].map(model_idx_map).to_numpy()
    e_idx = df["effort"].map(effort_idx_map).to_numpy()
    t_idx = df["task"].map(task_idx_map).to_numpy()

    y = np.log10(df["usd_run_total"].to_numpy(dtype=float))
    accepted = df["accepted"].to_numpy(dtype=float)

    coords = {
        "model": models_sorted,
        "effort": effort_levels,
        "effort_step": effort_levels[1:],
        "task": tasks_sorted,
    }

    with pm.Model(coords=coords) as model:
        # --- cost head ---
        alpha_m = pm.Normal("alpha_m", mu=0.0, sigma=1.5, dims="model")

        f_increments = pm.HalfNormal("f_increments", sigma=0.5, dims="effort_step")
        f_curve = pm.Deterministic(
            "f_curve", pt.concatenate([pt.zeros(1), pt.cumsum(f_increments)]), dims="effort"
        )

        tau_beta = pm.HalfNormal("tau_beta", sigma=0.2)
        beta_m = pm.Normal("beta_m", mu=0.0, sigma=tau_beta, dims="model")

        tau_task = pm.HalfNormal("tau_task", sigma=0.5)
        task_effect = pm.Normal("task_effect", mu=0.0, sigma=tau_task, dims="task")

        sigma = pm.HalfNormal("sigma", sigma=1.0)
        nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.1)
        nu = pm.Deterministic("nu", nu_raw + 1.0)

        mu = (
            alpha_m[m_idx]
            + f_curve[e_idx] * (1.0 + beta_m[m_idx])
            + task_effect[t_idx]
        )
        pm.StudentT("log10_usd_run_total_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        # --- acceptance head ---
        gamma_m = pm.Normal("gamma_m", mu=0.0, sigma=1.5, dims="model")
        g_increments = pm.HalfNormal("g_increments", sigma=0.5, dims="effort_step")
        g_curve = pm.Deterministic(
            "g_curve", pt.concatenate([pt.zeros(1), pt.cumsum(g_increments)]), dims="effort"
        )
        tau_task_g = pm.HalfNormal("tau_task_g", sigma=0.5)
        task_gamma = pm.Normal("task_gamma", mu=0.0, sigma=tau_task_g, dims="task")

        logit_p = gamma_m[m_idx] + g_curve[e_idx] + task_gamma[t_idx]
        pm.Bernoulli("accepted_obs", logit_p=logit_p, observed=accepted)

    ctx = {
        "models": models_sorted,
        "efforts": effort_levels,
        "tasks": tasks_sorted,
        "model_idx": model_idx_map,
        "effort_idx": effort_idx_map,
        "task_idx": task_idx_map,
    }
    return model, ctx


def sample(df: pd.DataFrame, draws: int = 200, tune: int = 200, chains: int = 2,
           seed: int = SEED, progressbar: bool = False):
    """Build the model and sample it. Returns (idata, model_ctx).

    `cores=1` is fixed regardless of `chains`: this keeps sampling inside a
    single process, which is what makes tiny dry runs (`draws=15, tune=15,
    chains=1`) reliable in a sandboxed test environment. Correctness of the
    plumbing is the bar here, not wall-clock parallelism.

    Compilation is pinned to pytensor's classic "CVM" backend for the
    duration of this call (restored after). Recent pytensor defaults to a
    numba-jitted backend when numba is importable, and numba is an extra,
    fast-moving native dependency; pinning CVM keeps `fit`'s sampling path
    working across numba/numpy version combinations that do not line up
    (this includes the sandbox this module was built and tested in) instead
    of depending on numba compiling cleanly at all.
    """
    require_bayes()
    import pymc as pm
    import pytensor

    model, ctx = build_model(df)
    with pytensor.config.change_flags(mode="CVM"):
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                random_seed=seed,
                progressbar=progressbar,
                target_accept=0.9,
            )
            pm.compute_log_likelihood(idata, model=model, progressbar=False)
    return idata, ctx
