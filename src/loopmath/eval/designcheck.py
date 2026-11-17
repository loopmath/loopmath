"""The design-based check used by `eval value --check design`. Not a command: `value.py` calls it.

For each arm and dimension with human readings that a `labels next` plan asked for, it estimates the
mean human score over the arm's targets in the study from the sampling design alone, and shows it
beside what the model expects a human to say. A gap larger than their joint uncertainty is flagged: the
model and the labels disagree about that arm.

The estimate
------------
Every score is put on 0 to 1 first: `linear`, `fraction` and `levels` by the fitting scale, `binary` as
0 or 1, `items` as the share passed. For target `t` of the arm:

- `f_t` is the prediction: the mean of the readings by judges that are not persons. A target with no
  such reading takes the arm's mean prediction. With no prediction at all `f_t` is 0.
- `y_t` is the mean of the human readings on `t` that name an assignment of the plans given.
- `pi_t` is that assignment's recorded inclusion probability.

The prediction-powered estimate is the mean prediction plus the weighted mean residual of the labelled
targets,

    theta = mean_t(f_t) + sum_labelled((y_t - f_t) / pi_t) / sum_labelled(1 / pi_t)

Its variance is the Hajek estimator for a fixed-size unequal-probability sample, on the centred
residuals. With 50 or more labelled targets the interval is normal. Below 50 it is a bootstrap
percentile interval over the labelled targets, and the cell says so. The variance has one centre: for
a sample stratified on the judge score, where the judge is off by much more in one band than another,
the interval is wider than its level says, never narrower.

The model's side is, per posterior draw, the mean over the arm's runs of the reading the reviewer is
expected to give for the run's hidden quality: `q + bias` for `linear` and `fraction`, the expected
level position for `levels`, `Phi((q + bias) / sqrt(tau^2 + sigma^2))` for `binary`, and
`Phi((q + bias) / sqrt(1 + tau^2 + sigma^2))` for `items`.

The gap between the two is flagged when it is larger than 1.645 times their joint sd. In the joint sd
the design side is taken without the finite population correction: the model's side is an expectation,
and to it the readings of the targets nobody labelled are still random.

Limits, stated in the output where they apply: the plans list only the assignments that were drawn, so
with several plans a target's inclusion probability is taken from the assignment that asked for its
reading, which is exact for one plan only. Pair assignments are not checked.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr, ndtri

from . import contracts as C
from . import fitfile as F
from . import quality as Q

LEVEL = 0.90
BOOTSTRAP_BELOW = 50
BOOTSTRAP_DRAWS = 2000
FIRST_KINDS = ("pilot", "active", "uniform")  # assignment kinds drawn with their own probability
METHOD = ("prediction-powered: the mean judge reading over the arm's targets plus the inclusion-weighted mean "
          "residual of the human readings")


def observed(dim: dict, value: Any, of: int | None = None) -> float:
    """A raw reading on 0 to 1: the fitting scale for `linear`, `fraction` and `levels`, 0 or 1 for
    `binary`, the share passed for `items` (`value` passed out of `of`)."""
    if dim["scale"]["kind"] == "items":
        passed, total = Q.fitting_count(dim, int(value), int(of))
        return passed / total
    return float(Q.fitting_value(dim, value))


def expected_reading(q: np.ndarray, kind: str, bias: np.ndarray, noise: np.ndarray,
                     cuts: np.ndarray | None = None) -> np.ndarray:
    """The reading on 0 to 1 a judge is expected to give for hidden quality `q[S, R]`.

    `bias[S]` and `noise[S]` (`sqrt(tau^2 + sigma^2)`) are the judge's; `cuts[S, L - 1]` are the cut
    points of a `levels` dimension."""
    eta = q + bias[:, None]
    spread = noise[:, None]
    safe = np.where(spread > 0, spread, 1.0)
    if kind in ("linear", "fraction"):
        return eta
    if kind == "binary":
        return np.where(spread > 0, ndtr(eta / safe), (eta >= 0).astype(float))
    if kind == "items":
        return ndtr(eta / np.sqrt(1.0 + spread * spread))
    above = eta[:, :, None] - cuts[:, None, :]
    share = np.where(spread[:, :, None] > 0, ndtr(above / safe[:, :, None]), (above >= 0).astype(float))
    return share.mean(axis=2)


def estimate(f_all: np.ndarray, y: np.ndarray, f: np.ndarray, pi: np.ndarray) -> float:
    """The prediction-powered estimate: `f_all` over every target, `y`, `f`, `pi` over the labelled ones."""
    w = 1.0 / pi
    return float(f_all.mean() + (w * (y - f)).sum() / w.sum())


def variance(y: np.ndarray, f: np.ndarray, pi: np.ndarray, finite: bool = True) -> float:
    """The Hajek variance estimate of `estimate`, on the centred residuals. 0 for a census.

    `finite=False` leaves out the finite population correction: the variance around the mean a model
    expects, for which the readings of the targets nobody labelled are still random."""
    n = len(y)
    w = 1.0 / pi
    e = y - f
    u = (e - (w * e).sum() / w.sum()) * w
    c = 1.0 - pi if finite else np.ones(n)
    if n < 2 or c.sum() <= 0:
        return 0.0
    centre = (c * u).sum() / c.sum()
    return float(n / (n - 1) * (c * (u - centre) ** 2).sum() / w.sum() ** 2)


def interval(f_all: np.ndarray, y: np.ndarray, f: np.ndarray, pi: np.ndarray, rng: np.random.Generator) -> dict:
    """The design-based estimate with its sd and 90% interval: normal at 50 labels or more, bootstrap below."""
    mean = estimate(f_all, y, f, pi)
    n = len(y)
    if n >= BOOTSTRAP_BELOW:
        sd = variance(y, f, pi) ** 0.5
        z = float(ndtri(0.5 + LEVEL / 2))
        return {"mean": mean, "sd": sd, "lo": mean - z * sd, "hi": mean + z * sd, "interval": "normal"}
    picks = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    w = 1.0 / pi[picks]
    boot = f_all.mean() + (w * (y - f)[picks]).sum(axis=1) / w.sum(axis=1)
    lo, hi = np.percentile(boot, [50 * (1 - LEVEL), 100 - 50 * (1 - LEVEL)])
    return {"mean": mean, "sd": float(boot.std()), "lo": float(lo), "hi": float(hi), "interval": "bootstrap"}


def _assignments(plans: Sequence[dict], names: Sequence[str]) -> tuple[dict[str, dict], int]:
    """Score assignments by id over every plan, and the number of pair assignments left out."""
    out: dict[str, dict] = {}
    pairs = 0
    for plan, name in zip(plans, names):
        for n, row in enumerate(plan["assignments"]):
            if row["id"] in out:
                raise C.ContractError(name, f"assignments[{n}].id", f'"{row["id"]}" is also in another plan given',
                                      "give every assignment id once across the plans", "duplicate")
            if row["ask"] == "pair":
                pairs += 1
                continue
            out[row["id"]] = row
    return out, pairs


def check(fit: F.Fit, scores: Sequence[dict], plans: Sequence[dict], *, seed: int = 0,
          plan_names: Sequence[str] | None = None) -> dict:
    """The design-based check over normalised score rows and labels plans. See the module docstring."""
    names = list(plan_names or [f"plan {n}" for n, _ in enumerate(plans, 1)])
    assignments, n_pairs = _assignments(plans, names)
    dims = fit.dimensions
    dim_at = {dim["id"]: d for d, dim in enumerate(dims)}
    run_at = {run: r for r, run in enumerate(fit.index["runs"])}
    person = {row["id"]: row["kind"] == "person" for row in fit["judges"]}
    read, replaced = C.live(list(scores))
    not_used = [dict(row) for row in replaced]

    # One observation per call: (run, dimension, judge, call) to a value on 0 to 1 and the assignment named.
    calls: dict[tuple, dict] = {}
    for row in read:
        d, r = dim_at.get(C.dimension_id(row["step"], row["component"])), run_at.get(row["run"])
        if d is None or r is None or row["judge"] not in person:
            what = "dimension" if d is None else "run" if r is None else "judge"
            not_used.append({"id": row["id"], "reason": f"its {what} is not in the fit"})
            continue
        key = (r, d, row["judge"], row["call"])
        try:
            if "item" in row:
                one = calls.setdefault(key, {"passed": 0, "of": 0, "ids": [], "assignment": row.get("assignment")})
                one["passed"] += int(row["value"])
                one["of"] += 1
                one["ids"].append(row["id"])
                continue
            value = observed(dims[d], row["value"], row.get("of"))
        except (ValueError, TypeError, KeyError, ZeroDivisionError):
            not_used.append({"id": row["id"], "reason": "its value is not on the scale of the dimension"})
            continue
        calls[key] = {"value": value, "ids": [row["id"]], "assignment": row.get("assignment")}

    predictions: dict[tuple[int, int], list[float]] = {}
    labels: dict[tuple[int, int], list[dict]] = {}
    for (r, d, judge, _), one in calls.items():
        if "value" not in one:
            one["value"] = observed(dims[d], one["passed"], one["of"])
        if not person[judge]:
            predictions.setdefault((r, d), []).append(one["value"])
            continue
        plan_row = assignments.get(one["assignment"]) if one["assignment"] is not None else None
        reason = None
        if one["assignment"] is None:
            reason = "a person's reading that names no assignment"
        elif plan_row is None:
            reason = f'its assignment "{one["assignment"]}" is not a score assignment of the plans given'
        elif (plan_row["target"]["run"] != fit.index["runs"][r] or plan_row["target"]["step"] != dims[d]["step"]
              or dims[d]["component"] not in plan_row["components"]):
            reason = f'it does not match its assignment "{one["assignment"]}"'
        if reason is not None:
            not_used += [{"id": rid, "reason": reason} for rid in one["ids"]]
            continue
        labels.setdefault((r, d), []).append({"value": one["value"], "judge": judge, "kind": plan_row["kind"],
                                              "pi": float(plan_row["inclusion_probability"])})

    rng = np.random.default_rng(seed)
    draws = fit.draws
    z = float(ndtri(0.5 + LEVEL / 2))
    run_arm = fit.run_arm
    cells = []
    for a, arm in enumerate(fit.arms):
        runs = np.flatnonzero(run_arm == a)
        for d, dim in enumerate(dims):
            labelled = [r for r in runs if (int(r), d) in labels]
            if not labelled:
                continue
            kind = dim["scale"]["kind"]
            known = {int(r): float(np.mean(predictions[(int(r), d)])) for r in runs if (int(r), d) in predictions}
            fallback = float(np.mean(list(known.values()))) if known else 0.0
            f_all = np.array([known.get(int(r), fallback) for r in runs])
            y, f, pi, reviewers = [], [], [], {}
            for r in labelled:
                rows = labels[(int(r), d)]
                first = next((row for row in rows if row["kind"] in FIRST_KINDS), rows[0])
                y.append(float(np.mean([row["value"] for row in rows])))
                f.append(known.get(int(r), fallback))
                pi.append(first["pi"])
                for row in rows:
                    reviewers[row["judge"]] = reviewers.get(row["judge"], 0) + 1
            y, f, pi = np.array(y), np.array(f), np.array(pi)
            cell: dict = {"arm": arm, "dimension": dim["id"], "step": dim["step"], "component": dim["component"],
                          "n_targets": int(len(runs)), "n_labels": int(len(labelled)), "n_predicted": len(known),
                          "reviewers": sorted(reviewers), "design": None, "model": None, "gap": None,
                          "joint_sd": None, "flagged": False, "own": None, "status": "ok"}
            cells.append(cell)
            q = np.asarray(draws["q_run"])[:, runs, d]
            if len(labelled) < 2:
                cell["status"] = "one labelled target: no interval, not checked"
                continue
            if np.isnan(q).any():
                cell["status"] = "the fit holds no quality for this dimension: not checked"
                continue
            cuts = None
            if kind == "levels":
                fixed = np.asarray(fit.index["dimensions"][d]["cuts"] or Q.fixed_cuts(len(dim["scale"]["levels"])))
                cuts = np.broadcast_to(fixed, (q.shape[0], len(fixed)))
                if dim.get("cuts") == "free" and "cuts" in draws:
                    free = np.asarray(draws["cuts"])[:, d, :len(fixed)]
                    cuts = free if not np.isnan(free).any() else cuts
            model = np.zeros(q.shape[0])
            for judge, count in reviewers.items():
                j = fit.judge(judge)
                bias = np.nan_to_num(np.asarray(draws["bias"])[:, j, d])
                noise = np.hypot(np.nan_to_num(np.asarray(draws["tau"])[:, j, d]),
                                 np.nan_to_num(np.asarray(draws["sigma"])[:, j, d]))
                model += count / sum(reviewers.values()) * expected_reading(q, kind, bias, noise, cuts).mean(axis=1)
            design = interval(f_all, y, f, pi, rng)
            # against the model the labelled sample is read without the finite population correction
            against = design["sd"] if design["interval"] == "bootstrap" else variance(y, f, pi, finite=False) ** 0.5
            joint = float(np.hypot(against, model.std()))
            gap = design["mean"] - float(model.mean())
            cell.update(design=design, model=F.stat(model), gap=gap, joint_sd=joint,
                        flagged=bool(abs(gap) > z * joint))
            own = Q.own_scale(np.array([design["mean"], model.mean()]), dim)
            if own is not None:
                cell["own"] = {"design": float(own[0]), "model": float(own[1])}
    notes = ["scores are on 0 to 1: the fitting scale for linear, fraction and levels, the share passed for "
             "binary and items"]
    if len(plans) > 1:
        notes.append("several plans: a target's inclusion probability is taken from the assignment that asked for "
                     "its reading, which is exact for one plan only")
    if n_pairs:
        notes.append(f"{n_pairs} pair assignments are not checked: the design-based check reads scores")
    if any(cell["design"] and cell["design"]["interval"] == "bootstrap" for cell in cells):
        notes.append(f"below {BOOTSTRAP_BELOW} labelled targets the interval is a bootstrap percentile interval")
    return {"method": METHOD, "level": LEVEL, "bootstrap_below": BOOTSTRAP_BELOW, "plans": names, "cells": cells,
            "flagged": sum(cell["flagged"] for cell in cells), "not_used": not_used, "notes": notes}


def check_files(fit: F.Fit, study: str | Path, plans: Sequence[str | Path], *, seed: int = 0) -> dict:
    """`check` over the `scores.jsonl` of a study folder and labels plan files."""
    scores = C.read_lines(Path(study) / C.SCORES_FILE, C.SCHEMA_SCORES)
    docs = [C.read_document(plan, C.SCHEMA_LABELS) for plan in plans]
    return check(fit, scores, docs, seed=seed, plan_names=[Path(plan).name for plan in plans])


def run(args: argparse.Namespace) -> int:
    """Not a command of its own: runs `eval value` with `--check design`."""
    from . import value

    args.check = "design"
    return value.run(args)
