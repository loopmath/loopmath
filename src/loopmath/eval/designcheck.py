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
residuals. The variance has one centre: for a sample stratified on the judge score, where the judge is
off by much more in one band than another, the interval is wider than its level says, never narrower.

The interval, and the cell says which one it is:

- `census`: every target of the arm has a human score, each labelled target drawn with probability 1.
  The estimate is the mean itself and the interval has no width. Labelling every target that was still
  open is not a census of the arm unless the scores of the others are given too.
- `bound`: the labelled readings show no spread (every human score equal, or every residual equal), so
  the sample says nothing about how often a target differs. The interval then holds every mean the
  targets nobody labelled could give without the sample being unlikely: at most `K` unseen targets sit
  at the far end of the scale, `K` the largest number that a sample with that inclusion probability
  misses altogether with a chance above 5%.
- `normal`: 50 or more labelled targets, the estimate plus and minus 1.645 sd.
- `t`: below 50, the estimate plus and minus the Student's t quantile times the sd. The degrees of
  freedom are the effective number of sampled targets, `(sum w)^2 / sum w^2` over the targets with
  `pi < 1`, less one, and less one more when the probabilities are unequal. Where a percentile
  interval from resampling the labelled targets' parts of the Hajek variance reaches further on one
  side, as it does for a skewed sample, the interval reaches that far.
- `withheld`: no interval, with a `reason`; `sd`, `lo` and `hi` are null and the cell is not checked.
  This is the result with fewer than 2 labelled targets, with less than one degree of freedom, and
  when labelled targets recorded as certain do not make up the population (then the mean is null too:
  the recorded probabilities do not describe a sample of it).

A person's reading that the plans given did not ask for is the known score of its target. Such targets
are set apart: the labels are read as a sample of the other targets, which is what a plan drawn after
those readings is, and the known scores enter the mean as they are. The design side counts them in
`known`.

The same allowance is made group by group. Labelled targets are grouped by inclusion probability,
which is how a stratum shows in a plan. A group without spread widens a `normal` or `t` interval
by its own `K` unseen targets, and `unseen` in the output is their total. Every score lies in 0 to 1,
so the interval is cut there.

The model's side is, per posterior draw, the mean over the arm's runs of the reading the reviewer is
expected to give for the run's hidden quality. With `s^2 = tau^2 + sigma^2 + sd_judge_input^2`, every
noise term of a reading by that reviewer (judge by output, call noise, judge by input): `q + bias` for
`linear` and `fraction`, the expected level position for `levels`, `Phi((q + bias) / s)` for `binary`,
and `Phi((q + bias) / sqrt(1 + s^2))` for `items`. A person has no family effect. On `levels`, `binary`
and `items` the expected reading depends on `s`: when the fit says one of the three terms is not
measured by the design (`prior_only` or `confounded`), the cell shows the design side and is not
checked. The model's side, its gap and the flag are computed on the draws in the unit the engine fitted
them in (`units.in_fitted_unit`), so a fit file of either schema gives the same check; a score on 0 to 1
is read against the latent quality of a relative dimension there.

The gap between the two is flagged when it is larger than 1.645 times their joint sd. In the joint sd
the design side is taken without the finite population correction: the model's side is an expectation,
and to it the readings of the targets nobody labelled are still random. Labelled targets on one input
share the reviewer's judge-by-input effect, so there the residuals are summed per input first. For a
`bound` interval the design side is the half-width of the same bound for a sample with replacement.

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
from scipy.stats import t as student

from . import contracts as C
from . import fitfile as F
from . import quality as Q
from .units import in_fitted_unit

LEVEL = 0.90
BOOTSTRAP_BELOW = 50
BOOTSTRAP_DRAWS = 2000
MIN_LABELS = 2
FIRST_KINDS = ("pilot", "active", "uniform")  # assignment kinds drawn with their own probability
SIDE = (1.0 - LEVEL) / 2                      # the chance left on each side of an interval
NOISE_TERMS = ("tau", "sigma", "judge_input")  # the terms behind the noise of one reading
NOT_MEASURED = ("prior_only", "confounded")
_FLAT = 1e-12
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


def variance(y: np.ndarray, f: np.ndarray, pi: np.ndarray, finite: bool = True,
             clusters: Sequence[Any] | None = None) -> float:
    """The Hajek variance estimate of `estimate`, on the centred residuals. 0 for a census.

    `finite=False` leaves out the finite population correction: the variance around the mean a model
    expects, for which the readings of the targets nobody labelled are still random. There `clusters`
    (one label per target, for example its input) sums the residuals of one cluster first, for targets
    whose readings share an effect."""
    n = len(y)
    w = 1.0 / pi
    e = y - f
    u = (e - (w * e).sum() / w.sum()) * w
    if not finite and clusters is not None:
        _, group = np.unique(np.asarray(clusters), return_inverse=True)
        u = np.bincount(group, weights=u)
        n = len(u)
    c = 1.0 - pi if finite else np.ones(n)
    if n < 2 or c.sum() <= 0:
        return 0.0
    centre = (c * u).sum() / c.sum()
    return float(n / (n - 1) * (c * (u - centre) ** 2).sum() / w.sum() ** 2)


def unseen(pi: float) -> int:
    """The largest number of targets, each drawn with probability `pi`, that a sample misses altogether
    with a chance above `SIDE`: the largest `K` with `(1 - pi)^K > SIDE`."""
    if pi >= 1.0:
        return 0
    return max(0, int(np.ceil(np.log(SIDE) / np.log1p(-pi) - 1e-9)) - 1)


def flat_groups(y: np.ndarray, f: np.ndarray, pi: np.ndarray) -> list[np.ndarray]:
    """The groups of labelled targets that show no spread, as index arrays.

    Targets are grouped by inclusion probability, which is how a stratum shows in a plan; probabilities
    held by one target only form one group together. A group is flat when every human score in it is
    equal, or every residual is: it cannot say how often a target like its own differs."""
    values, inverse, counts = np.unique(np.round(pi, 9), return_inverse=True, return_counts=True)
    label = np.where(counts[inverse] >= 2, inverse, -1)
    out = []
    for g in np.unique(label):
        at = np.flatnonzero(label == g)
        if np.ptp(y[at]) <= _FLAT or np.ptp((y - f)[at]) <= _FLAT:
            out.append(at)
    return out


def _room(f_all: np.ndarray, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """How far down and up each target's score could sit from what the estimate takes it to be, largest
    first."""
    implied = np.clip(f_all + shift, 0.0, 1.0)
    return np.sort(implied)[::-1], np.sort(1.0 - implied)[::-1]


def withheld(mean: float | None, reason: str) -> dict:
    """An estimate without an interval: `sd`, `lo` and `hi` are None, and `reason` says why."""
    return {"mean": mean, "sd": None, "lo": None, "hi": None, "interval": "withheld", "unseen": 0, "reason": reason}


def interval(f_all: np.ndarray, y: np.ndarray, f: np.ndarray, pi: np.ndarray,
             rng: np.random.Generator | None = None) -> dict:
    """The design-based estimate with its sd and 90% interval, which interval it is (`census`, `bound`,
    `normal` at 50 labels or more, `t` below, or `withheld` with a `reason` and no `lo` and `hi`), and
    `unseen`: the number of unlabelled targets the interval allows at the far end of the scale.

    `f_all` holds every target of the population the labels were drawn from: a census is only claimed
    when each of them is labelled. `rng` draws the resamples of the `t` interval; without one a fixed
    seed is used. See the module docstring."""
    n, total = len(y), len(f_all)
    if n == 0:
        return withheld(None, "no labelled target")
    mean = estimate(f_all, y, f, pi)
    w = 1.0 / pi
    e = y - f
    sampled = pi < 1.0
    if not sampled.any():
        if n >= total:
            return {"mean": mean, "sd": 0.0, "lo": mean, "hi": mean, "interval": "census", "unseen": 0}
        return withheld(None, f"{n} of {total} targets are labelled, each recorded as drawn with certainty: these "
                              "probabilities do not describe a sample of all the targets")
    if n < MIN_LABELS:
        return withheld(mean, f"not enough labels: {n} labelled, {MIN_LABELS} needed for an interval")
    sd = variance(y, f, pi) ** 0.5
    if n >= BOOTSTRAP_BELOW:
        z = float(ndtri(1.0 - SIDE))
        out = {"mean": mean, "sd": sd, "lo": mean - z * sd, "hi": mean + z * sd, "interval": "normal"}
    else:
        # few labels: Student's t, with the degrees of freedom the unequal weights leave
        freedom = degrees(pi)
        if freedom < 1.0:
            return withheld(mean, "not enough labels: one target carries nearly all the weight of the sample")
        t = float(student.ppf(1.0 - SIDE, freedom))
        # a skewed sample (a few scores far from the rest) pulls the interval to one side: resampling
        # each target's part of the Hajek variance shows by how much, and the interval reaches that far
        c = 1.0 - pi
        u = (e - (w * e).sum() / w.sum()) * w
        part = np.sqrt(n / (n - 1) * c) * (u - (c * u).sum() / c.sum()) / w.sum()
        rng = np.random.default_rng(0) if rng is None else rng
        boot = mean + part[rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))].sum(axis=1) - part.sum()
        lo, hi = np.percentile(boot, [100 * SIDE, 100 * (1 - SIDE)])
        out = {"mean": mean, "sd": sd, "lo": min(mean - t * sd, float(lo)), "hi": max(mean + t * sd, float(hi)),
               "interval": "t"}
    # groups without spread: allow for the targets like them that nobody labelled
    k, covered = 0, 0
    for at in flat_groups(y, f, pi):
        low = float(pi[at].min())
        if low < 1.0:
            k += min(unseen(low), int(np.ceil((w[at] - 1.0).sum() - 1e-9)))
            covered += len(at)
    k = min(k, total - n)
    down, up = _room(f_all, float((w * e).sum() / w.sum()))
    out["lo"] -= float(down[:k].sum()) / total
    out["hi"] += float(up[:k].sum()) / total
    out["unseen"] = k
    if k and covered == int(sampled.sum()):
        out["interval"] = "bound"
    # every score lies in 0 to 1, and so does their mean
    out["lo"], out["hi"] = min(mean, max(out["lo"], 0.0)), max(mean, min(out["hi"], 1.0))
    return out


def degrees(pi: np.ndarray) -> float:
    """Degrees of freedom for the t interval: the effective number of sampled targets less one, and less
    one more when they were drawn with unequal probabilities. Targets drawn with certainty add no
    variance and none."""
    w = 1.0 / pi[pi < 1.0]
    if not len(w):
        return 0.0
    effective = float(w.sum() ** 2 / (w * w).sum())
    return effective - 1.0 - (1.0 if np.ptp(w) > 1e-9 * w.max() else 0.0)


def with_known(design: dict, n_rest: int, scores: np.ndarray) -> dict:
    """`design`, an `interval` over `n_rest` targets, carried to those and the targets whose human scores
    `scores` are known apart from the sample: a known score is counted as it is, with no sampling error."""
    if not len(scores):
        return design
    total = n_rest + len(scores)
    out = dict(design, known=len(scores))
    for key in ("mean", "lo", "hi"):
        if out[key] is not None:
            out[key] = float((scores.sum() + n_rest * out[key]) / total)
    if out["sd"] is not None:
        out["sd"] = float(out["sd"] * n_rest / total)
    return out


def against_model(f_all: np.ndarray, y: np.ndarray, f: np.ndarray, pi: np.ndarray,
                  clusters: Sequence[Any] | None = None) -> float:
    """The sd of the design-based estimate around the mean a model expects: no finite population
    correction, residuals summed per cluster. A group without spread adds the half-width of its bound
    for a sample with replacement, as an sd: a share `1 - SIDE^(1 / n)` of its targets at the far end."""
    sd = variance(y, f, pi, finite=False, clusters=clusters) ** 0.5
    w = 1.0 / pi
    share = 0.0
    for at in flat_groups(y, f, pi):
        n = len(set(np.asarray(clusters)[at].tolist())) if clusters is not None else len(at)
        share += float(w[at].sum() / w.sum()) * (1.0 - SIDE ** (1.0 / n))
    if not share:
        return sd
    down, up = _room(f_all, float((w * (y - f)).sum() / w.sum()))
    return sd + share * float(max(down.mean(), up.mean())) / float(ndtri(1.0 - SIDE))


def term_state(fit: F.Fit, name: str, judge: str, dimension: str) -> str | None:
    """The state of the most specific entry of a term in the fit's identification report, or None."""
    for key in ((judge, dimension), (judge, None), (None, dimension), (None, None)):
        state = fit.term(name, *key)
        if state is not None:
            return state
    return None


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
    apart: dict[tuple[int, int], list[float]] = {}      # a person's readings that are not labels of the plans
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
            apart.setdefault((r, d), []).append(one["value"])
            continue
        labels.setdefault((r, d), []).append({"value": one["value"], "judge": judge, "kind": plan_row["kind"],
                                              "pi": float(plan_row["inclusion_probability"])})

    rng = np.random.default_rng(seed)
    draws = in_fitted_unit(fit.draws)                     # the model's side decides in the fitted unit
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
            # targets with a human score from outside the plans: known as they are, not part of the sample
            outside = [int(r) for r in runs if (int(r), d) in apart and (int(r), d) not in labels]
            scored = np.array([float(np.mean(apart[(r, d)])) for r in outside])
            f_all = np.array([known.get(int(r), fallback) for r in runs if int(r) not in outside])
            y, f, pi, reviewers, inputs = [], [], [], {}, []
            for r in labelled:
                rows = labels[(int(r), d)]
                first = next((row for row in rows if row["kind"] in FIRST_KINDS), rows[0])
                y.append(float(np.mean([row["value"] for row in rows])))
                f.append(known.get(int(r), fallback))
                pi.append(first["pi"])
                inputs.append(int(fit.run_input[r]))
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
            design = with_known(interval(f_all, y, f, pi, rng), len(f_all), scored)
            cell["design"] = design
            if design["interval"] == "withheld":
                cell["status"] = f"not checked: {design['reason']}"
                continue
            prior = sorted({f"{name} of {judge}" for judge in reviewers for name in NOISE_TERMS
                            if term_state(fit, name, judge, dim["id"]) in NOT_MEASURED})
            if prior and kind not in ("linear", "fraction"):
                cell["status"] = ("not checked: the expected reading on this scale depends on a noise term the "
                                  f"design does not measure ({', '.join(prior)})")
                continue
            model = np.zeros(q.shape[0])
            for judge, count in reviewers.items():
                j = fit.judge(judge)
                bias = np.nan_to_num(np.asarray(draws["bias"])[:, j, d])
                # every noise term of one reading: judge by output, call noise, judge by input
                noise = np.sqrt(sum(np.nan_to_num(np.asarray(draws[name])[:, j, d]) ** 2
                                    for name in ("tau", "sigma", "sd_judge_input")))
                model += count / sum(reviewers.values()) * expected_reading(q, kind, bias, noise, cuts).mean(axis=1)
            # against the model: no finite population correction, and one input's readings share an effect
            beside = against_model(f_all, y, f, pi, inputs) * len(f_all) / len(runs)
            if len(scored):
                # known scores are readings too: to the model's expectation they are as random as the rest
                spread = float(scored.var(ddof=1)) if len(scored) > 1 else 0.25
                beside = float(np.hypot(beside, np.sqrt(spread * len(scored)) / len(runs)))
            joint = float(np.hypot(beside, model.std()))
            gap = design["mean"] - float(model.mean())
            cell.update(model=F.stat(model), gap=gap, joint_sd=joint, flagged=bool(abs(gap) > z * joint))
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
    kinds = {cell["design"]["interval"] for cell in cells if cell["design"]}
    if "t" in kinds:
        notes.append(f"below {BOOTSTRAP_BELOW} labelled targets the interval uses Student's t")
    if "withheld" in kinds:
        notes.append("withheld: the labels do not support an interval, so none is shown and the cell is not checked")
    if any(cell["design"].get("known") for cell in cells if cell["design"]):
        notes.append("known: a person's reading that the plans given did not ask for counts as the known score of "
                     "its target, and the plans' labels are read as a sample of the other targets")
    if "bound" in kinds:
        notes.append("bound: the labelled readings show no spread, so the interval is what the targets nobody "
                     "labelled could still hold, not a spread seen in the sample")
    if "census" in kinds:
        notes.append("census: every target has a human score, so the design-based estimate is exact")
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
