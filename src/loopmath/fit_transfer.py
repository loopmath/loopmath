"""Transfer scoring and CLI-level fit orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import research_paths
from .fit_assembly import assemble_table
from .fit_bayes import SEED, sample
from .fit_masks import E1A_OBSERVE, apply_mask, parse_mask
from .fit_predict import predict
from .fit_pricing import DEFAULT_SWEEP_DIR

__all__ = ["TRANSFER_TEST_NOTE", "transfer_test", "run_fit", "run_transfer_test"]


# SPEC section 7 / this build's hard rule: the transfer test is DESCRIPTIVE
# ONLY, and that means literally -- no pass, no fail, no threshold, anywhere
# in code or output, not even inside a sentence whose point is to deny
# having one. An earlier version of this constant read "descriptive
# scoring, no pass or fail thresholds", which said the right thing but said
# it using the two forbidden words; this wording carries the same meaning
# without them.
TRANSFER_TEST_NOTE = "descriptive scoring only, with no verdict and no threshold"


def transfer_test(idata, model_ctx: dict, observed: pd.DataFrame, heldout: pd.DataFrame,
                   ci: float = 0.80, seed: int = SEED) -> dict:
    """Score held-out cells. DESCRIPTIVE ONLY: no pass/fail, no thresholds.

    The scored outcome is `usd_run_total` -- the whole session set's priced
    cost for the run, planner and reviewer attempts included, matching what
    `build_model` now fits (see the module docstring's CORRECTION
    paragraph). It is never the developer-only `usd` diagnostic column.

    Space convention (this module's choice, since `loopmath.scoring` only fixes
    function signatures, not units): interval coverage and MLPD -- the
    model's own MLPD, the pooled-mean baseline B1', and the nearest-neighbor
    baseline B2' alike -- are all computed in log10(usd_run_total), the
    space the model's likelihood is actually defined on. A log density is
    not invariant under a change of units, so scoring two of those three on
    log10(usd_run_total) and the third on raw usd_run_total (an earlier,
    incorrect version of this function did exactly that for B2') would make
    the reported MLPD differences mathematically meaningless; B2' is
    therefore built here from log10-transformed copies of `observed` and
    `heldout`, not the raw dollar columns, and `test_fit.py` pins an
    invariance test against this. Kendall tau, within-factor, and decision
    regret are computed in raw usd_run_total, since those are read as dollar
    decisions rather than modelling diagnostics (within-factor is a ratio
    test on those raw dollars, so it is called with `log10_space=False`; the
    flag name refers to the space of the *inputs* `scoring.within_factor`
    receives, not to log10(usd_run_total) being the model's native scale).
    `loopmath.scoring` is imported here, not at module load time, so `loopmath.fit`
    still imports cleanly before that module exists.

    Decision regret picks a workflow configuration (model and effort
    together), not an individual held-out row: `scoring.decision_regret` is
    called with `config_cols=["model", "effort"]` so replicate runs of the
    same configuration are aggregated (mean cost) before either side of the
    comparison is chosen, per task group. See `scoring.decision_regret`'s
    docstring for why a row-level choice made a reported 0.00 regret
    meaningless.

    FLAG 5: also counts and prints how many held-out rows needed an
    unseen-level draw (see `predict`) -- printed directly for the same
    reason `run_fit` prints its cost/planner notes (module docstring).
    """
    try:
        from loopmath import scoring
    except ImportError as exc:
        raise ImportError(
            "loopmath.fit.transfer_test needs loopmath.scoring for its metrics, and "
            "that module is not importable yet. It ships as part of this "
            "build's scoring work; nothing to do here until it lands."
        ) from exc

    preds = predict(idata, heldout, model_ctx, seed=seed)
    draws = preds["log10_usd_draws"]
    y_log = np.log10(heldout["usd_run_total"].to_numpy(dtype=float))

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    lo = np.quantile(draws, lo_q, axis=1)
    hi = np.quantile(draws, hi_q, axis=1)
    y_pred_log = np.median(draws, axis=1)
    y_pred_usd = 10.0 ** y_pred_log
    y_true_usd = heldout["usd_run_total"].to_numpy(dtype=float)

    coverage = scoring.interval_coverage(y_log, lo, hi)
    # scoring.mlpd wants draws shaped (n_draws, n_points); predict() returns
    # (n_points, n_draws), so transpose here at the call site.
    mlpd_value = scoring.mlpd(y_log, draws.T)

    train_y_log = (
        np.log10(observed["usd_run_total"].to_numpy(dtype=float)) if len(observed) else np.array([])
    )
    baseline_pooled = scoring.baseline_pooled_mean(train_y_log, y_log)

    # B2' must be scored on the same log10(usd_run_total) scale as the model
    # MLPD and B1' above -- see this function's docstring. `usd_run_total` is
    # guaranteed strictly positive on every row this table produces (never
    # None, never <= 0; see `assemble_table`/`_price_run_total`), so log10 is
    # always defined here.
    observed_log = observed.assign(
        log10_usd_run_total=np.log10(observed["usd_run_total"].to_numpy(dtype=float))
    )
    heldout_log = heldout.assign(
        log10_usd_run_total=np.log10(heldout["usd_run_total"].to_numpy(dtype=float))
    )
    baseline_nn = scoring.baseline_nearest_neighbor(
        observed_log, heldout_log, "log10_usd_run_total", ["task", "effort"]
    )

    tau = scoring.kendall_tau(y_true_usd, y_pred_usd)
    w2 = scoring.within_factor(y_true_usd, y_pred_usd, factor=2.0, log10_space=False)

    regret = scoring.decision_regret(
        heldout, y_pred_usd, "task_group", "usd_run_total", config_cols=["model", "effort"]
    )

    accept_probs = preds["accept_prob_draws"].mean(axis=1)
    y_true_binary = heldout["accepted"].to_numpy(dtype=int)
    accept_cal = scoring.acceptance_calibration(y_true_binary, accept_probs, n_bins=5)

    n_unseen = preds["n_unseen_level_draws"]
    n_heldout = int(len(heldout))
    if n_unseen:
        parts = []
        if preds["unseen_models"]:
            parts.append(f"models: {', '.join(preds['unseen_models'])}")
        if preds["unseen_tasks"]:
            parts.append(f"tasks: {', '.join(preds['unseen_tasks'])}")
        detail = f" ({'; '.join(parts)})" if parts else ""

        # Not every unseen level is drawn the same way (FLAG 5 / see
        # `predict`'s docstring): a task's effect has a fitted group scale of
        # its own to draw from; a model's baseline cost/acceptance level does
        # not (it has only a fixed scale set in `build_model`, not learned
        # from this fit), while a model's effort-curve multiplier does. This
        # line must say that plainly, not describe every unseen level as
        # "drawn from a fitted group scale".
        scale_notes = []
        if preds["unseen_tasks"]:
            scale_notes.append(
                "an unseen task's effect on cost and on acceptance is drawn from a "
                "fitted group scale (learned from how much tasks varied in this fit), "
                "not fixed at zero"
            )
        if preds["unseen_models"]:
            scale_notes.append(
                "an unseen model's effort-curve multiplier is drawn from a fitted "
                "group scale the same way, but its baseline cost and acceptance level "
                "are drawn from a fixed scale set before fitting rather than one "
                "learned from this data (still not fixed at zero)"
            )
        print(
            f"{n_unseen} of {n_heldout} held-out rows used a model or task never seen "
            f"in the observed data{detail}: " + "; ".join(scale_notes) + "."
        )
    else:
        print(f"0 of {n_heldout} held-out rows needed an unseen-level draw.")

    return {
        "interval_coverage": coverage,
        "mlpd": mlpd_value,
        "baseline_pooled_mean": baseline_pooled,
        "baseline_nearest_neighbor": baseline_nn,
        "kendall_tau": tau,
        "within_factor": w2,
        "decision_regret": regret,
        "acceptance_calibration": accept_cal,
        "n_observed": int(len(observed)),
        "n_heldout": n_heldout,
        "ci": ci,
        "seed": seed,
        "note": TRANSFER_TEST_NOTE,
        "n_unseen_level_draws": n_unseen,
        "unseen_models": preds["unseen_models"],
        "unseen_tasks": preds["unseen_tasks"],
    }


def run_fit(sweep_dir=None, observe: str = E1A_OBSERVE, holdout: str = "rest",
            reveal: str | None = None, draws: int = 200, tune: int = 200,
            chains: int = 2, seed: int = SEED, out=None) -> dict:
    """Assemble, mask, sample. Returns a dict with the table shapes, the mask
    spec, and the idata (and writes idata to `out` as NetCDF when `out` is
    given).

    Prints a small number of mandatory, loud, data-honesty lines directly
    (the DATA DECISION A cost note, the DATA DECISION B planner note, the
    SPEC amendment's GPT-5.6 long-context caveat, an explicit statement of
    which column is the modelled response, the PATCH 4 acceptance-exclusion
    bias note, and the FLAG 1 median run-total share) -- see the module
    docstring for why this function prints at all when the rest of the
    CLI's output is `cli.py`'s job. Every other line remains the CLI's job.
    """
    root = research_paths.sweep_dir(sweep_dir if sweep_dir is not None else DEFAULT_SWEEP_DIR)
    df = assemble_table(sweep_dir=root)
    assembly = df.attrs.get("assembly", {})

    print(assembly.get("cost_price_note", ""))
    print(assembly.get("planner_note", ""))
    print(assembly.get("long_context_note", ""))
    print(
        "the modelled cost is usd_run_total: the whole session set for the run, "
        "planner and reviewer included, not the developer attempt alone. The "
        "developer-only figure (usd) is kept as a diagnostic column, not fit."
    )
    print(assembly.get("acceptance_bias_note", ""))

    usable_total = (
        df[df["usd_run_total"].notna() & (df["usd_run_total"] > 0)] if not df.empty else df
    )
    if not usable_total.empty:
        share = float((usable_total["usd"] / usable_total["usd_run_total"]).median())
        nulled = assembly.get("usd_run_total_nulled", {}).get("count", 0)
        extra = (
            f"; {nulled} runs could not get a run total and are excluded from that share"
            if nulled
            else ""
        )
        print(
            f"the developer attempt is a median {share * 100:.0f}% of the run's total "
            f"priced cost (planner + developer + reviewer) across {len(usable_total)} "
            f"runs{extra}."
        )
    else:
        print("median developer share of run total: not available (no run had a priced total).")

    mask = parse_mask(observe, holdout=holdout, reveal=reveal)
    observed, heldout = apply_mask(df, mask)

    if observed.empty:
        raise ValueError(
            f"the mask selected 0 of {len(df)} rows to observe "
            f"(observe={observe!r}, holdout={holdout!r}, reveal={reveal!r}). "
            "Check the mask against the models and efforts actually present "
            "in this table (see assemble_table's df.attrs['assembly'])."
        )

    idata, model_ctx = sample(observed, draws=draws, tune=tune, chains=chains, seed=seed)

    if out is not None:
        # arviz writes NetCDF through a backend it does not depend on, so a
        # working [bayes] install can still fail here after sampling has
        # already succeeded. Name the missing package instead of letting a raw
        # backend ValueError stand in for an actionable message; the sampling
        # itself is not lost, only the write.
        try:
            idata.to_netcdf(str(out))
        except ValueError as exc:
            if "backend" not in str(exc).lower():
                raise
            raise RuntimeError(
                f"the fit sampled fine but could not be written to {out}: "
                "arviz needs a NetCDF backend and none is installed. "
                "Install one with: pip install h5netcdf "
                "(it is declared in the [bayes] extra). "
                "Re-run without --out to see the fit without writing it."
            ) from exc

    return {
        "df": df,
        "observed": observed,
        "heldout": heldout,
        "mask": mask,
        "idata": idata,
        "model_ctx": model_ctx,
        "shapes": {
            "total_rows": len(df),
            "observed_rows": len(observed),
            "heldout_rows": len(heldout),
        },
        "seed": seed,
        "out": str(out) if out is not None else None,
        "assembly": assembly,
        # Flat keys read directly by cli.py's fit_verb -- see the module
        # docstring's note on the cli.py / fit.py contract gap.
        "n_rows": len(df),
        "sweep_dir": str(root),
        "n_observed": len(observed),
        "n_heldout": len(heldout),
        "mask_spec": mask.spec,
    }


def run_transfer_test(fit_result: dict, ci: float = 0.80) -> dict:
    """Score the held-out cells from a `run_fit` result. Descriptive only."""
    return transfer_test(
        fit_result["idata"],
        fit_result["model_ctx"],
        fit_result["observed"],
        fit_result["heldout"],
        ci=ci,
        seed=fit_result.get("seed", SEED),
    )
