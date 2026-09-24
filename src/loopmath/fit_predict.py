"""Posterior prediction for held-out fit rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fit_bayes import SEED

__all__ = ["predict"]


def predict(idata, df: pd.DataFrame, model_ctx: dict, seed: int = SEED) -> dict:
    """Per-row predictive draws for log10(usd) and acceptance probability.

    Draws are built directly from the posterior's per-model/effort/task
    terms rather than via PyMC's own posterior-predictive machinery, because
    held-out rows can name a model or task the fit never saw at all (that is
    the point of a transfer test).

    FLAG 5: a model or task not present in `model_ctx` does NOT fall back to
    0. `beta_m`, `task_effect`, and `task_gamma` each have a fitted
    hyperprior scale (`tau_beta`, `tau_task`, `tau_task_g` respectively); an
    unseen level draws its effect from `Normal(0, that fitted scale)`, per
    predictive sample, so the predictive spread carries the uncertainty of a
    level the fit never saw. `alpha_m` and `gamma_m` have no fitted scale of
    their own (their prior scale, 1.5, is a fixed hyperparameter in
    `build_model`, not estimated); an unseen model draws those two from that
    same fixed scale -- still the assumed population distribution for new
    models, just not itself learned from data. One shared draw per unique
    unseen name (not per row), so a level recurring across several held-out
    rows is treated consistently, the way it would be if it had actually
    been observed. `effort` is not expected to be unseen: the shared curve's
    coords are always the full fixed `EFFORT_ORDER`, not just what was
    observed, so every valid effort has a position; kept defensive (falls
    back to the curve's pinned-zero anchor) rather than raising.
    """
    import arviz as az

    post = az.extract(idata, group="posterior")
    n_samples = int(post.sizes["sample"])

    alpha_m = post["alpha_m"].values
    beta_m = post["beta_m"].values
    f_curve = post["f_curve"].values
    task_effect = post["task_effect"].values
    sigma = post["sigma"].values
    nu = post["nu"].values
    tau_beta = post["tau_beta"].values
    tau_task = post["tau_task"].values

    gamma_m = post["gamma_m"].values
    g_curve = post["g_curve"].values
    task_gamma = post["task_gamma"].values
    tau_task_g = post["tau_task_g"].values

    model_pos = {m: i for i, m in enumerate(model_ctx["models"])}
    effort_pos = {e: i for i, e in enumerate(model_ctx["efforts"])}
    task_pos = {t: i for i, t in enumerate(model_ctx["tasks"])}

    rng = np.random.default_rng(seed)

    _ALPHA_SIGMA = 1.5  # matches alpha_m's prior scale in build_model
    _GAMMA_SIGMA = 1.5  # matches gamma_m's prior scale in build_model

    unseen_models = sorted({m for m in df["model"] if m not in model_pos})
    unseen_tasks = sorted({t for t in df["task"] if t not in task_pos})

    unseen_alpha = {m: rng.standard_normal(n_samples) * _ALPHA_SIGMA for m in unseen_models}
    unseen_beta = {m: rng.standard_normal(n_samples) * tau_beta for m in unseen_models}
    unseen_gamma = {m: rng.standard_normal(n_samples) * _GAMMA_SIGMA for m in unseen_models}
    unseen_task_effect = {t: rng.standard_normal(n_samples) * tau_task for t in unseen_tasks}
    unseen_task_gamma = {t: rng.standard_normal(n_samples) * tau_task_g for t in unseen_tasks}

    n_rows = len(df)
    mu_rows = np.zeros((n_rows, n_samples))
    logit_rows = np.zeros((n_rows, n_samples))
    unseen_level_used = np.zeros(n_rows, dtype=bool)

    zeros = np.zeros(n_samples)
    for i, (row_model, row_effort, row_task) in enumerate(
        zip(df["model"], df["effort"], df["task"])
    ):
        mi = model_pos.get(row_model)
        ei = effort_pos.get(row_effort)
        ti = task_pos.get(row_task)

        if mi is not None:
            a, b, g = alpha_m[mi], beta_m[mi], gamma_m[mi]
        else:
            a, b, g = unseen_alpha[row_model], unseen_beta[row_model], unseen_gamma[row_model]
            unseen_level_used[i] = True

        fe = f_curve[ei] if ei is not None else zeros
        gc = g_curve[ei] if ei is not None else zeros

        if ti is not None:
            te, tg = task_effect[ti], task_gamma[ti]
        else:
            te, tg = unseen_task_effect[row_task], unseen_task_gamma[row_task]
            unseen_level_used[i] = True

        mu_rows[i] = a + fe * (1.0 + b) + te
        logit_rows[i] = g + gc + tg

    nu_safe = np.clip(nu, 1.01, None)
    t_noise = rng.standard_t(nu_safe, size=(n_rows, n_samples))
    log10_usd_draws = mu_rows + t_noise * sigma[None, :]
    accept_prob_draws = 1.0 / (1.0 + np.exp(-logit_rows))

    return {
        "log10_usd_draws": log10_usd_draws,
        "accept_prob_draws": accept_prob_draws,
        "row_index": list(df.index),
        "n_unseen_level_draws": int(unseen_level_used.sum()),
        "unseen_models": unseen_models,
        "unseen_tasks": unseen_tasks,
    }
