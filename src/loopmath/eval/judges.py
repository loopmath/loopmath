"""`eval judges`: which judge, and how many repeats, from a fit.

Reads the fit file, and from the study folder only the number of items each call read on an `items`
component (the readings in scores.jsonl) and, for the stale-input warning, the digests of the study files. Every figure is computed per posterior draw and
reported as a stat: `{"mean", "sd", "lo", "hi"}` with the 90% interval.

Per judge and dimension, on the fitting scale and on the component's own scale where one exists:

- `bias`, `tau` (judge-by-output disagreement), `sigma` (run-to-run noise of one call).
- `mse` of the mean of k calls, k = 1 to 5: `systematic + tau^2 + sigma^2 / k`. The systematic part is
  the mean of `(bias + family effect)^2` over the targets of the study, each target with the family
  that wrote it. `mse.without_family` is the conditional figure `bias^2 + tau^2 + sigma^2 / k`. When the
  judge-by-input term is measured its variance is added: like `tau`, it does not shrink with repeats.
- `r2`: the squared correlation, over the targets of the study and the judge's noise, between reported
  quality `m(q)` and the mean of k calls as they are recorded: `eta + e` for `linear` and `fraction`,
  the position of the level read for `levels`, 0 or 1 for `binary`, the share of items passed for
  `items`. That share needs the number of items each call of the judge read: `of` on a count line, or the
  number of item lines in the call. The list of items written for an input is not used: it does not say
  that a call read every item. Without the judge's readings, or when its calls on one target read
  different numbers of items, `r2` of the dimension is not measured.
- `reliability`: the same for hidden quality on the fitting scale against the judge's view before it
  is cut into a verdict, `var_q / (var_q + tau^2 + sigma^2 / k)`, with `var_q` the variance between arm
  means (weighted by runs) plus the input and run variances. It is not what a recorded reading explains.
- `effective_repeats`: how many independent readings k repeats are worth,
  `(tau^2 + sigma^2) / (tau^2 + sigma^2 / k)`. Its limit, `1 + sigma^2 / tau^2`, is the most repeats can buy.
- `cost`: per call, per reading (a `panel` call is shared by the components it scores), and per unit of
  error removed, where the error removed by k calls is the variance of reported quality they explain,
  `var(m) x r2(k)`.

For a `binary` dimension, over the targets of the study (every run's hidden quality in the draws):
`false_pass`, `false_fail`, `flip` (two calls on one output disagree) and `agreement`. A target's hidden
status is `q >= 0`; one call passes with probability `Phi((q + bias + family) / sqrt(tau^2 + sigma^2))`;
two calls share the judge's view of the output and differ in call noise. For an `items` dimension the same
four figures are per item under the k of n model: an item's hidden status passes with probability
`Phi(q)`, a call passes it with probability `Phi(eta + e)`, independently given the two.

For pairs: `position`, `tie_width`, `sigma_pair`, and over every two runs of one input from different
arms, with both orders equally likely, the tie rate and the chance a call picks the output with the
higher hidden quality. `position` and `tie_width` are one number per judge, in the engine's unit; the
pair figures of a dimension divide them by its `unit` draws (`units.py`), so they are on the scale of
the dimension's other draws.

Units. On a `binary` dimension the latent figures of each judge (`bias`, `tau`, `sigma`, `mse` and the
sds) are in the anchors' total noise of one reading, on a pairs-only dimension in the anchors' total noise
of one pair verdict (`fitfile.py`, Units). The probabilities (`false_pass`, `false_fail`, `flip`,
`agreement`, tie rate, picks), `r2`, `reliability` and `effective_repeats` do not depend on the unit. The
choices are decisions, so they are made in the unit the engine fitted in (`dimensions[].unit.fitted_in`:
the anchors' call noise on a `binary` dimension, their pair noise on a pairs-only one), the same for a fit
file of either schema: the error of each (judge, repeats) in the table, the second judge and its weight,
and `--target-error`, which is read in that unit. A posterior mean of an error in the stated unit would
weigh the draws by a unit that differs from draw to draw, and could pick another judge. Each block of
choices names its unit in `unit`.

Family effects per model judge: `own_family` and `reference_family` contrasts (the effect on that family
minus the mean effect on the other families), flagged when the 90% interval excludes 0. `effect` is in
the dimension's stated unit; the flag reads the interval of the same draws in the unit the engine fitted
in, named in `decided_in` (Units), since a percentile interpolates between two draws whose units can
differ.

Across judges: a table of (judge, repeats) choices per dimension with error and cost, the cheapest choice
for `--target-error` and the best within `--budget-usd`, whether a second judge beats one more repeat,
the batching loss (two judges that differ only in `mode`) and the reference effect (two judges that
differ only in `reference.shown`).

A quantity whose term the design does not identify is null, and is listed under `not_measured` with the
design that would measure it. It is never reported as a finding. The same holds for every figure derived
from it: a family effect or a judge-by-input term that is fitted but not identified is neither used nor
replaced by zero, so `mse`, `r2`, cost, pass and fail figures and choices of that judge are not reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr, owens_t

from ..output import EXIT_OK, emit_json
from . import contracts as C
from . import fitfile as F
from . import quality as Q
from .contracts import ContractError

SCHEMA = "loopmath.eval-judges-report/1"
REPEATS = (1, 2, 3, 4, 5)
MAX_PAIRS = 4000        # pairs of targets read for the pair figures; more are thinned evenly
MIN_DEFINED = 0.1       # a ratio that is undefined in more than 90% of the draws is not reported
STUDY_ROLES = {"runs": C.RUNS_FILE, "scores": C.SCORES_FILE, "pairs": C.PAIRS_FILE, "items": C.ITEMS_FILE}


# ---------------------------------------------------------------------------------------------------
# Small helpers

def stat(x: Any) -> dict | None:
    """Mean, sd and the 90% interval over the draws where `x` is finite. None when too few are."""
    x = np.asarray(x, dtype=float).ravel()
    ok = np.isfinite(x)
    if x.size == 0 or ok.mean() < MIN_DEFINED:
        return None
    x = x[ok]
    lo, hi = np.percentile(x, [5.0, 95.0])
    return {"mean": float(x.mean()), "sd": float(x.std()), "lo": float(lo), "hi": float(hi)}


def _mean(x: Any) -> float | None:
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else None


def _nansum(x: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(x), 0.0, x).sum(axis=1)


def _nanmean(x: np.ndarray) -> np.ndarray:
    """The mean over the targets where `x` is a number, per draw. NaN when there is none."""
    n = (~np.isnan(x)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _nansum(x) / n


def resolve(fit: F.Fit, name: str, judge: str | None = None, dimension: str | None = None) -> tuple:
    """`(state, judge, dimension)` of the most specific entry of a term in the identification report.

    A fit may state a term per judge and dimension, per judge, per dimension or once. With no entry at
    all the state is None."""
    for key in dict.fromkeys(((judge, dimension), (judge, None), (None, dimension), (None, None))):
        state = fit.term(name, *key)
        if state is not None:
            return state, key[0], key[1]
    return None, judge, dimension


def term_state(fit: F.Fit, name: str, judge: str | None = None, dimension: str | None = None) -> str | None:
    return resolve(fit, name, judge, dimension)[0]


def is_measured(fit: F.Fit, name: str, judge: str | None = None, dimension: str | None = None) -> bool:
    """True only when the design identifies the term (`identified` or `weak`). No entry is not measured."""
    return term_state(fit, name, judge, dimension) in F.MEASURED_STATES


def own_factor(dim: dict) -> float | None:
    """Fitting-scale units to the component's own scale: the range of `linear`, L - 1 levels, 1 for `fraction`."""
    scale = dim["scale"]
    if scale["kind"] == "linear":
        return float(scale["range"][1] - scale["range"][0])
    if scale["kind"] == "levels":
        return float(len(scale["levels"]) - 1)
    if scale["kind"] == "fraction":
        return 1.0
    return None


def _scaled(x: np.ndarray, factor: float | None, power: int = 1, sign: float = 1.0) -> dict | None:
    return None if factor is None else stat(sign * x * factor ** power)


def _sign(dim: dict) -> float:
    """A bias on the own scale of a lower-is-better component has the other sign."""
    return -1.0 if dim.get("better", "higher") == "lower" else 1.0


def target_families(fit: F.Fit, step: str) -> np.ndarray:
    """Per run, the position in `index.families` of the family that wrote its output at `step` (-1: unknown).

    Read from the arm's step settings, so a run of a `mixed_arm` is given its arm's first setting."""
    families = fit.index["families"]
    by_arm = []
    for arm in fit["arms"]:
        family = F.target_family({"steps": arm["steps"]}, step) if arm.get("steps") else None
        by_arm.append(families.index(family) if family in families else -1)
    return np.asarray(by_arm, dtype=int)[fit.run_arm] if by_arm else np.zeros(0, dtype=int)


def quality_variance(fit: F.Fit) -> np.ndarray:
    """`[S, D]`: the variance of hidden quality over the study's targets, per draw."""
    draws = fit.draws
    mu = draws["mu"]
    counts = np.bincount(fit.run_arm, minlength=mu.shape[1]).astype(float)
    weights = counts / counts.sum() if counts.sum() else np.full(mu.shape[1], 1.0 / max(1, mu.shape[1]))
    centre = (mu * weights[None, :, None]).sum(axis=1, keepdims=True)
    between = ((mu - centre) ** 2 * weights[None, :, None]).sum(axis=1)
    return between + draws["sd_input"] ** 2 + draws["sd_run"] ** 2


def _scores_component(judge: dict, dim: dict) -> bool:
    return judge.get("components") is None or dim["component"] in judge["components"]


def components_per_call(judge: dict, dim: dict, dims: list[dict]) -> int:
    """How many components one call of this judge scores: 1, or the step's components in a `panel` call."""
    if judge.get("mode") != "panel":
        return 1
    return max(1, sum(1 for other in dims if other["step"] == dim["step"] and _scores_component(judge, other)))


# ---------------------------------------------------------------------------------------------------
# Figures from the draws

def status_figures(q: np.ndarray, shift: np.ndarray, shared_sd: np.ndarray, call_sd: np.ndarray,
                   kind: str) -> dict[str, np.ndarray]:
    """Pass and fail figures of one judge on one `binary` or `items` dimension, per draw.

    `q[S, R]` is the hidden quality of the targets, `shift[S, R]` the judge's bias plus its family effect
    on each target, `shared_sd[S]` the sd of what two calls on one output share, `call_sd[S]` the sd of
    what they do not share. Two calls disagree with probability `4 T(h, a)`, Owen's T, with
    `h = (q + shift) / total sd` and `a = sqrt(call^2 / (2 shared^2 + call^2))`.

    A ratio is NaN in a draw where its denominator is empty (no target with a hidden fail, say)."""
    hidden = (q >= 0).astype(float) if kind == "binary" else ndtr(q)
    hidden = np.where(np.isnan(q), np.nan, hidden)
    total = np.sqrt(shared_sd ** 2 + call_sd ** 2)
    some = total > 0
    m = q + shift
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(some[:, None], m / np.where(some, total, 1.0)[:, None], np.where(m >= 0, 40.0, -40.0))
        h = np.where(np.isnan(m), np.nan, np.clip(h, -40.0, 40.0))
        passes = ndtr(h)
        slope = np.sqrt(call_sd ** 2 / np.where(some, 2 * shared_sd ** 2 + call_sd ** 2, 1.0))
        flip = 4.0 * owens_t(h, np.broadcast_to(slope[:, None], h.shape))
        n_pass, n_fail = _nansum(hidden), _nansum(1.0 - hidden)
        out = {"false_pass": _nansum((1.0 - hidden) * passes) / n_fail,
               "false_fail": _nansum(hidden * (1.0 - passes)) / n_pass,
               "flip": _nanmean(flip),
               "agreement": _nanmean(hidden * passes + (1.0 - hidden) * (1.0 - passes)),
               "n_pass": n_pass, "n_fail": n_fail}
        if kind == "binary":
            rate, share = _nanmean(passes), _nanmean(hidden)
            cov = _nanmean(hidden * passes) - rate * share
            out["r2_status"] = cov ** 2 / (rate * (1.0 - rate) * share * (1.0 - share))
    return out


def bvn_cdf(h: Any, k: Any, rho: Any) -> np.ndarray:
    """`P(X <= h, Y <= k)` for standard normal X and Y with correlation `rho`, by Owen's T function."""
    h, k, rho = np.broadcast_arrays(np.asarray(h, dtype=float), np.asarray(k, dtype=float),
                                    np.asarray(rho, dtype=float))
    h = np.where(h == 0, 1e-150, np.clip(h, -40.0, 40.0))
    k = np.where(k == 0, 1e-150, np.clip(k, -40.0, 40.0))
    rho = np.clip(rho, -1.0 + 1e-12, 1.0 - 1e-12)
    root = np.sqrt(1.0 - rho * rho)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = (0.5 * (ndtr(h) + ndtr(k)) - owens_t(h, (k - rho * h) / (h * root))
               - owens_t(k, (h - rho * k) / (k * root)) - np.where(h * k > 0, 0.0, 0.5))
    return np.clip(out, 0.0, 1.0)


def reading_moments(mean: np.ndarray, shared_sd: np.ndarray, call_sd: np.ndarray, kind: str,
                    cuts: np.ndarray | None = None, n_items: np.ndarray | None = None) -> tuple:
    """Moments of the judge's reading of each target, per draw: `(expected, shared, per_call)`.

    `mean[S, R]` is the judge's expected view of each target (hidden quality, bias, family effect),
    `shared_sd[S]` the sd of what every call on one target shares, `call_sd[S]` the sd of one call's
    noise. A reading is what section 7.4 of the model says it is: `eta + e` for `linear` and `fraction`,
    the position of the level whose interval holds `eta + e` for `levels` (`cuts[1 or S, L - 1]`), 0 or 1
    for `binary`, and the share `k / n` with `k ~ Binomial(n, Phi(eta + e))` for `items` (`n_items[R]`).

    `expected[S, R]` is the mean reading of a target. The mean of k calls on it has the variance
    `shared + per_call / k`."""
    shared, call = shared_sd[:, None] ** 2, call_sd[:, None] ** 2
    if kind in ("linear", "fraction"):
        return mean, np.broadcast_to(shared, mean.shape), np.broadcast_to(call, mean.shape)
    if kind == "items":
        total = np.sqrt(1.0 + shared + call)
        h = mean / total
        expected = ndtr(h)
        same = bvn_cdf(h, h, (shared + call) / total ** 2)       # two items of one call both pass
        cross = bvn_cdf(h, h, shared / total ** 2)               # an item in each of two calls both pass
        second = same + (expected - same) / n_items[None, :]
        return expected, cross - expected ** 2, second - cross
    cuts = np.zeros((1, 1)) if kind == "binary" else np.atleast_2d(np.asarray(cuts, dtype=float))
    total = np.sqrt(shared + call)
    some = total > 0
    safe = np.where(some, total, 1.0)
    rho = np.where(some, shared / safe ** 2, 0.0)
    hs = []
    for c in range(cuts.shape[1]):
        gap = mean - cuts[:, c][:, None]
        hs.append(np.where(np.isnan(gap), np.nan, np.where(some, gap / safe, np.where(gap >= 0, 40.0, -40.0))))
    steps = float(len(hs))
    expected = sum(ndtr(h) for h in hs) / steps
    second = sum(ndtr(np.minimum(a, b)) for a in hs for b in hs) / steps ** 2
    cross = sum(bvn_cdf(a, b, rho) for a in hs for b in hs) / steps ** 2
    return expected, cross - expected ** 2, second - cross


def reading_r2(m: np.ndarray, expected: np.ndarray, shared: np.ndarray, per_call: np.ndarray,
               k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per draw: the squared correlation between reported quality and the mean of k calls, and `var(m)`.

    The correlation is over the targets of the study and the judge's noise: `m[S, R]` is the reported
    quality of each target, the other arrays come from `reading_moments`. NaN where either side has no
    variance."""
    ok = ~(np.isnan(m) | np.isnan(expected) | np.isnan(shared + per_call))
    m, expected = np.where(ok, m, np.nan), np.where(ok, expected, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm, ce = m - _nanmean(m)[:, None], expected - _nanmean(expected)[:, None]
        cov, var_m = _nanmean(cm * ce), _nanmean(cm * cm)
        var_y = _nanmean(ce * ce) + _nanmean(np.where(ok, shared + per_call / k, np.nan))
        return np.where((var_m > 0) & (var_y > 0), cov ** 2 / (var_m * var_y), np.nan), var_m


NO_READINGS = "needs the number of items each call read: the judge's readings in scores.jsonl in the study folder"


def item_counts(fit: F.Fit, scores: list[dict]) -> dict[tuple[str, str], tuple]:
    """Per (judge, `items` dimension) with readings: `(n, reason)`, the number of items a call of the
    judge reads on each target, `n[R]`, or None with the reason it cannot be told.

    `scores` are the rows of scores.jsonl. A replaced reading is not read. The number of items of a call
    is `of` on its count line, or the number of its item lines. When every call of the judge read the
    same number, that number holds for every target. Otherwise each target of the fit needs calls by the
    judge that all read one number."""
    kinds = {dim["id"]: dim["scale"]["kind"] for dim in fit.dimensions}
    position = {name: r for r, name in enumerate(fit.index["runs"])}
    calls: dict[tuple, dict] = {}
    for row in C.live(scores)[0] if scores else []:
        did = C.dimension_id(row.get("step") or C.FINAL, row["component"])
        if kinds.get(did) != "items" or row["run"] not in position or ("item" not in row and "of" not in row):
            continue
        entry = calls.setdefault((row["judge"], did, row["run"], row["call"]), {"lines": 0, "of": []})
        if "item" in row:
            entry["lines"] += 1
        else:
            entry["of"].append(int(row["of"]))
    seen: dict[tuple[str, str], dict[str, set]] = {}
    for (judge, did, run, _), entry in calls.items():
        mixed = bool(entry["lines"] and entry["of"]) or len(entry["of"]) > 1
        size = None if mixed else (entry["of"][0] if entry["of"] else entry["lines"])
        seen.setdefault((judge, did), {}).setdefault(run, set()).add(size)
    out: dict[tuple[str, str], tuple] = {}
    for key, by_run in seen.items():
        sizes = set().union(*by_run.values())
        if None in sizes:
            out[key] = (None, "a call of the judge has both item lines and a count line, or two count lines")
        elif len(sizes) == 1:
            out[key] = (np.full(len(position), float(next(iter(sizes)))), None)
        elif len(by_run) == len(position) and all(len(found) == 1 for found in by_run.values()):
            n = np.empty(len(position))
            for run, found in by_run.items():
                n[position[run]] = float(next(iter(found)))
            out[key] = (n, None)
        else:
            out[key] = (None, f"the calls of the judge read different numbers of items ({min(sizes)} to "
                              f"{max(sizes)}) and not one number per target")
    return out


def study_pairs(fit: F.Fit) -> tuple[np.ndarray, np.ndarray, int]:
    """Every two runs of one input from different arms: positions `(a, b)` and how many there are in all."""
    by_input: dict[int, list[int]] = {}
    for r, i in enumerate(fit.index["run_input"]):
        by_input.setdefault(i, []).append(r)
    arm = fit.index["run_arm"]
    first, second = [], []
    for runs in by_input.values():
        for n, a in enumerate(runs):
            for b in runs[n + 1:]:
                if arm[a] != arm[b]:
                    first.append(a)
                    second.append(b)
    total = len(first)
    a, b = np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    if total > MAX_PAIRS:
        keep = np.linspace(0, total - 1, MAX_PAIRS).round().astype(int)
        a, b = a[keep], b[keep]
    return a, b, total


def pair_figures(q_a: np.ndarray, q_b: np.ndarray, shift: np.ndarray, noise_sd: np.ndarray,
                 position: np.ndarray, width: np.ndarray) -> dict[str, np.ndarray]:
    """Tie rate and the chance a pairwise call picks the output with the higher hidden quality, per draw.

    `z = q_a - q_b + shift + position x o + noise`, verdict `a` above `width`, `b` below `-width`, else a
    tie. Both orders (`o` = +1, -1) are averaged. `noise_sd[S]` is `sqrt(2 tau^2 + sigma_pair^2)`."""
    diff = q_a - q_b
    sd = np.where(noise_sd > 0, noise_sd, 1e-12)[:, None]
    tie, right = 0.0, 0.0
    for order in (1.0, -1.0):
        mean = diff + shift + order * position[:, None]
        upper, lower = ndtr((width[:, None] - mean) / sd), ndtr((-width[:, None] - mean) / sd)
        tie = tie + 0.5 * (upper - lower)
        right = right + 0.5 * np.where(diff > 0, 1.0 - upper, lower)
    tie_rate, correct = _nanmean(tie), _nanmean(right)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"tie_rate": tie_rate, "correct": correct, "correct_when_decided": correct / (1.0 - tie_rate)}


# ---------------------------------------------------------------------------------------------------
# The report

class _Report:
    """Builds the report of one fit. `raw[(judge, dimension)]` keeps the per-draw arrays that the
    comparisons across judges read; only measured quantities are kept there."""

    def __init__(self, fit: F.Fit, items: dict[tuple[str, str], tuple] | None = None) -> None:
        self.fit = fit
        self.items = items or {}
        self.draws = fit.draws
        self.dims: list[dict] = fit.dimensions
        self.declared: list[dict] = fit["judges"]
        self.var_q = quality_variance(fit)
        self.not_measured: list[dict] = []
        self.raw: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self.cost: dict[tuple[str, str], float | None] = {}
        # the arrays the choices read, in the unit the engine fitted in (Units): mse(k), sigma and the shift
        self.fitted: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self.unit = np.asarray(self.draws["unit"], dtype=float)
        self._pairs: tuple | None = None
        self.anchor_state = {row["dimension"]: row["state"] for row in fit["anchor"]["dimensions"]}

    def _flag(self, quantity: str, names: list[str], judge: str, dimension: str | None) -> bool:
        """True when the design measures every term. Each term it does not is listed once."""
        ok = True
        for name in names:
            state, who, where = resolve(self.fit, name, judge, dimension)
            if state in F.MEASURED_STATES:
                continue
            ok = False
            entry = {"quantity": quantity, "judge": who, "dimension": where, "term": name,
                     "state": state or "no_entry", "needs": F.TERM_NEEDS[name]}
            if entry not in self.not_measured:
                self.not_measured.append(entry)
        return ok

    def _use(self, name: str, x: np.ndarray, judge: str, dimension: str) -> str:
        """How an optional term enters the figures derived from it: `measured` (its draws are used),
        `none` (the model has no such term here) or `unmeasured` (fitted but not identified: nothing
        that depends on it is reported, and it is not replaced by zero)."""
        if x.size == 0 or np.isnan(x).all():
            return "none"
        state = term_state(self.fit, name, judge, dimension)
        if state in F.MEASURED_STATES:
            return "measured"
        if state in ("off", "absent") or (state is None and not np.nan_to_num(x).any()):
            return "none"
        return "unmeasured"

    def _family_effect(self, j: int, d: int) -> np.ndarray:
        """`[S, R]`: the judge's family effect on each run's output at the dimension's step. 0 where none."""
        family = np.nan_to_num(self.draws["family"][:, j, :, d])
        where = target_families(self.fit, self.dims[d]["step"])
        if family.shape[1] == 0:
            return np.zeros((family.shape[0], len(where)))
        return np.where(where[None, :] >= 0, family[:, np.clip(where, 0, None)], 0.0)

    # -- one judge on one dimension

    def dimension(self, j: int, judge: dict, d: int, dim: dict) -> dict | None:
        fit, draws, jid, did = self.fit, self.draws, judge["id"], dim["id"]
        bias, tau, sigma = draws["bias"][:, j, d], draws["tau"][:, j, d], draws["sigma"][:, j, d]
        pair_sd = draws["sigma_pair"][:, j, d]
        present = {name: not np.isnan(x).any() for name, x in
                   (("bias", bias), ("tau", tau), ("sigma", sigma), ("sigma_pair", pair_sd))}
        if not _scores_component(judge, dim) or not any(present.values()):
            return None
        kind, factor = dim["scale"]["kind"], own_factor(dim)
        counted = next((row for row in judge.get("dims", []) if row["dimension"] == did), {})
        out: dict = {"dimension": did, "step": dim["step"], "component": dim["component"], "kind": kind,
                     "unit": (dim.get("unit") or {}).get("unit"), "n_targets": counted.get("n_targets", 0),
                     "n_calls": counted.get("n_calls", 0), "n_pairs": counted.get("n_pairs", 0)}
        raw: dict[str, np.ndarray] = {}

        anchored = fit["anchor"]["kind"] == "person" and self.anchor_state.get(did) in ("anchored", "weak")
        out["bias"] = None
        if present["bias"]:
            out["bias"] = {"fitting": stat(bias), "own": _scaled(bias, factor, sign=_sign(dim)),
                           "relative_to": "anchor" if anchored else "judges_mean",
                           "anchor_state": self.anchor_state.get(did)}
            raw["bias"] = bias
        ok = {}
        for name, x in (("tau", tau), ("sigma", sigma)):
            ok[name] = present[name] and self._flag(name, [name], jid, did)
            out[name] = {"fitting": stat(x) if ok[name] else None, "own": _scaled(x, factor) if ok[name] else None,
                         "state": (term_state(fit, name, jid, did) or "no_entry") if present[name] else "absent",
                         "measured": bool(ok[name])}
            if ok[name]:
                raw[name] = x

        # Optional terms: used when measured, absent when the model has none, and never set to zero when
        # they are fitted without being identified.
        input_sd = draws["sd_judge_input"][:, j, d]
        input_use = self._use("judge_input", input_sd, jid, did)
        family_use = self._use("family", draws["family"][:, j, :, d], jid, did)
        by_input = np.nan_to_num(input_sd) if input_use == "measured" else np.zeros_like(tau)
        if input_use == "measured":
            out["sd_judge_input"] = {"fitting": stat(by_input), "own": _scaled(by_input, factor),
                                     "state": term_state(fit, "judge_input", jid, did), "measured": True}
        stable = by_input ** 2 + tau ** 2          # what repeats do not shrink, besides the systematic part

        noise_ok = bool(ok["tau"] and ok["sigma"])
        reason = None if noise_ok else "tau or sigma is not measured"
        if noise_ok and input_use == "unmeasured":
            self._flag("sd_judge_input", ["judge_input"], jid, did)
            noise_ok, reason = False, "the judge-by-input term is fitted but not identified"
        total_ok = noise_ok and present["bias"]
        total_reason = reason
        if total_ok and family_use == "unmeasured":
            self._flag("mse, r2, cost, pass and fail figures", ["family"], jid, did)
            total_ok, total_reason = False, "the family effect is fitted but not identified"

        q = draws["q_run"][:, :, d]
        effect = self._family_effect(j, d) if family_use == "measured" and q.shape[1] else 0.0
        shift = np.nan_to_num(bias)[:, None] + effect                       # S, R (or S, 1): bias plus family
        systematic = (shift ** 2).mean(axis=1)
        var_q = self.var_q[:, d]
        out["mse"] = {"measured": bool(total_ok), "population": "the targets of the study",
                      "family": {"measured": "included", "none": "none", "unmeasured": "not_measured"}[family_use],
                      "systematic": None, "by_repeats": [], "without_family": []}
        out["r2"] = {"measured": False, "of": "reported quality", "by": "the mean of k calls as recorded",
                     "one_call": None, "by_repeats": []}
        out["reliability"] = {"measured": noise_ok, "of": "hidden quality on the fitting scale",
                              "by": "the judge's view before it is cut into a verdict", "one_call": None,
                              "by_repeats": []}
        out["effective_repeats"] = {"measured": noise_ok, "by_repeats": [], "limit": None}
        per_call = judge.get("cost_usd_per_call")
        per_reading = None if per_call is None else per_call / components_per_call(judge, dim, self.dims)
        out["cost"] = {"usd_per_call": per_call, "usd_per_reading": per_reading, "by_repeats": []}
        self.cost[(jid, did)] = per_reading
        if noise_ok:
            with np.errstate(invalid="ignore", divide="ignore"):
                for k in REPEATS:
                    noise = stable + sigma ** 2 / k
                    out["reliability"]["by_repeats"].append(stat(var_q / (var_q + noise)))
                    out["effective_repeats"]["by_repeats"].append(stat((stable + sigma ** 2) / noise))
                    raw[f"noise{k}"] = noise
                out["reliability"]["one_call"] = out["reliability"]["by_repeats"][0]
                out["effective_repeats"]["limit"] = stat((stable + sigma ** 2) / stable)
        if total_ok:
            raw["shift"] = shift
            out["mse"]["systematic"] = {"fitting": stat(systematic), "own": _scaled(systematic, factor, power=2)}
            for k in REPEATS:
                mse = systematic + stable + sigma ** 2 / k
                raw[f"mse{k}"] = mse
                out["mse"]["by_repeats"].append({"repeats": k, "fitting": stat(mse),
                                                 "own": _scaled(mse, factor, power=2)})
                if family_use == "measured":
                    plain = bias ** 2 + stable + sigma ** 2 / k
                    out["mse"]["without_family"].append({"repeats": k, "fitting": stat(plain),
                                                         "own": _scaled(plain, factor, power=2)})
        if total_reason and not total_ok:
            out["mse"]["reason"] = total_reason

        status_ok = False
        if kind in ("binary", "items") and present["bias"]:
            names = ["tau", "sigma"] + (["binary_status"] if kind == "binary" else [])
            status_ok = self._flag("false_pass, false_fail, flip, agreement", names, jid, did) and total_ok
            out["status"] = {"measured": bool(status_ok), "model": "binary" if kind == "binary" else "k_of_n"}
            if status_ok:
                call_sd = sigma if kind == "binary" else np.sqrt(1.0 + sigma ** 2)
                figures = status_figures(q, np.broadcast_to(shift, q.shape), np.sqrt(stable), call_sd, kind)
                for name, x in figures.items():
                    out["status"][name] = _mean(x) if name in ("n_pass", "n_fail") else stat(x)
                    raw[name] = x

        # R-squared of the reading as recorded against reported quality, over the targets of the study.
        r2_reason = total_reason if not total_ok else None
        if total_ok and kind == "binary" and not status_ok:
            r2_reason = "the hidden status of the binary dimension is not identified"
        n_items, why = self.items.get((jid, did), (None, NO_READINGS))
        if total_ok and kind == "items" and n_items is None:
            r2_reason = why
        if total_ok and not q.shape[1]:
            r2_reason = "the fit holds no target"
        if total_ok and r2_reason is None:
            cuts = None
            if kind == "levels":
                size = len(dim["scale"]["levels"]) - 1
                cuts = Q.fixed_cuts(size + 1)[None, :]
                free = draws.get("cuts")
                if dim.get("cuts") == "free" and free is not None and np.isfinite(free[:, d, :size]).all():
                    cuts = free[:, d, :size]
            expected, shared, call = reading_moments(q + shift, np.sqrt(stable), sigma, kind, cuts, n_items)
            m = Q.reported(q, kind)
            out["r2"]["measured"] = True
            for k in REPEATS:
                r2, var_m = reading_r2(m, expected, shared, call, k)
                out["r2"]["by_repeats"].append(stat(r2))
                if per_reading is not None:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        removed = var_m * r2
                        out["cost"]["by_repeats"].append({
                            "repeats": k, "usd": k * per_reading, "variance_removed": _mean(removed),
                            "usd_per_variance_removed": _mean(np.where(removed > 0, k * per_reading / removed,
                                                                       np.nan))})
            out["r2"]["one_call"] = out["r2"]["by_repeats"][0]
        elif r2_reason:
            out["r2"]["reason"] = r2_reason

        if present["sigma_pair"]:
            out["pairs"] = self.pairs(j, jid, d, did, tau, pair_sd, bool(ok["tau"]), factor, family_use)
        self.raw[(jid, did)] = raw
        u = self.unit[:, d]
        fitted = {name: x * u ** 2 for name, x in raw.items() if name.startswith("mse")}
        if "sigma" in raw:
            fitted["sigma"] = raw["sigma"] * u
        if "shift" in raw:
            fitted["shift"] = raw["shift"] * u[:, None]
        self.fitted[(jid, did)] = fitted
        return out

    def pairs(self, j: int, jid: str, d: int, did: str, tau: np.ndarray, pair_sd: np.ndarray, tau_ok: bool,
              factor: float | None, family_use: str = "none") -> dict:
        fit, draws = self.fit, self.draws
        pair_ok = self._flag("sigma_pair", ["sigma_pair"], jid, did)
        out: dict = {"sigma_pair": {"fitting": stat(pair_sd) if pair_ok else None,
                                    "own": _scaled(pair_sd, factor) if pair_ok else None,
                                    "state": term_state(fit, "sigma_pair", jid, did) or "no_entry",
                                    "measured": bool(pair_ok)},
                     "measured": False, "tie_rate": None, "correct": None, "correct_when_decided": None,
                     "pairs": 0, "pairs_read": 0}
        rest = self._flag("tie rate and the chance to pick the better output", ["tie_width", "position"], jid, did)
        if self._pairs is None:
            self._pairs = study_pairs(fit)
        a, b, total = self._pairs
        out["pairs"], out["pairs_read"] = total, int(len(a))
        if family_use == "unmeasured":      # the family effect stays in a pair of outputs from two families
            rest = self._flag("tie rate and the chance to pick the better output", ["family"], jid, did) and rest
        if not (pair_ok and rest and tau_ok and len(a)):
            return out
        effect = self._family_effect(j, d) if family_use == "measured" else np.zeros(draws["q_run"].shape[:2])
        q = draws["q_run"][:, :, d]
        figures = pair_figures(q[:, a], q[:, b], effect[:, a] - effect[:, b], np.sqrt(2 * tau ** 2 + pair_sd ** 2),
                               np.nan_to_num(draws["position"][:, j]) / draws["unit"][:, d],
                               np.nan_to_num(draws["tie_width"][:, j]) / draws["unit"][:, d])
        out["measured"] = True
        for name, x in figures.items():
            out[name] = stat(x)
        return out

    # -- one judge

    def family(self, j: int, judge: dict) -> dict | None:
        """Own-family and reference-family contrasts of a model judge, per dimension."""
        if judge["kind"] != "model":
            return None
        fit, jid, families = self.fit, judge["id"], self.fit.index["families"]
        own = judge["model"]["family"]
        reference = judge.get("reference") or {"shown": False}
        author = ((reference.get("author") or {}).get("model") or {}).get("family") if reference.get("shown") else None
        out: dict = {"confounded": bool(author is not None and author == own)}

        def contrast(family: str, name: str) -> dict:
            block: dict = {"family": family, "state": term_state(fit, name, jid) or "no_entry", "measured": False,
                           "dims": []}
            if family not in families:
                block["state"], block["reason"] = "absent", f"no output in the study was written by family {family}"
                return block
            if len(families) < 2:
                block["state"], block["reason"] = "absent", "every output in the study comes from one family"
                return block
            f = families.index(family)
            others = [n for n in range(len(families)) if n != f]
            effects = self.draws["family"][:, j]                                  # S, F, D
            for d, dim in enumerate(self.dims):
                x = effects[:, f, d] - effects[:, others, d].mean(axis=1)
                if np.isnan(x).any() or not _scores_component(judge, dim):
                    continue
                if not self._flag(name if name == "reference_family" else "own_family", [name], jid, dim["id"]):
                    continue
                # the flag reads the interval in the unit the engine fitted in (Units): a percentile
                # interpolates between two draws, and a unit that differs between them can move it past 0
                decided = stat(x * self.unit[:, d])
                block["dims"].append({"dimension": dim["id"], "effect": stat(x),
                                      "own": _scaled(x, own_factor(dim), sign=_sign(dim)),
                                      "excludes_zero": bool(decided["lo"] > 0 or decided["hi"] < 0),
                                      "decided_in": (dim.get("unit") or {}).get("fitted_in")})
            block["measured"] = bool(block["dims"])
            return block

        out["own_family"] = contrast(own, "family")
        out["reference_family"] = None
        if author is not None:
            if out["confounded"]:
                out["reference_family"] = {
                    "family": author, "state": "confounded", "measured": False, "dims": [],
                    "reason": "the judge and the reference's author share a family: the two contrasts are one number"}
            else:
                out["reference_family"] = contrast(author, "reference_family")
        return out

    def judge(self, j: int, judge: dict) -> dict:
        jid = judge["id"]
        out = {name: judge.get(name) for name in ("id", "kind", "model", "effort", "prompt", "mode", "reference",
                                                  "blinded", "cost_usd_per_call", "anchor")}
        halo_ok = self._flag("halo_target, halo_call", ["halo_split"], jid, None)
        # the shared part of a call exists only where one call reads several components: a `panel` judge
        in_panel = judge.get("mode") == "panel"
        out["halo"] = {"target": stat(self.draws["halo_target"][:, j]) if halo_ok else None,
                       "call": stat(self.draws["halo_call"][:, j]) if halo_ok and in_panel else None,
                       "state": term_state(self.fit, "halo_split", jid) or "no_entry", "measured": bool(halo_ok)}
        out["dims"] = [row for d, dim in enumerate(self.dims)
                       if (row := self.dimension(j, judge, d, dim)) is not None]
        out["pairs"] = None
        if any("pairs" in row for row in out["dims"]):
            out["pairs"] = {}
            for name in ("position", "tie_width"):
                x = self.draws[name][:, j]
                state = term_state(self.fit, name, jid)
                good = state in F.MEASURED_STATES and not np.isnan(x).any()
                out["pairs"][name] = {"fitting": stat(x) if good else None, "state": state or "no_entry",
                                      "measured": bool(good)}
        out["family"] = self.family(j, judge)
        return out

    # -- across judges

    def _rmse(self, jid: str, did: str, k: int) -> float | None:
        """The root of the posterior mean `mse(k)`, in the unit the engine fitted in (Units)."""
        mse = self.fitted.get((jid, did), {}).get(f"mse{k}")
        return None if mse is None else float(np.sqrt(mse.mean()))

    def next_step(self, jid: str, did: str, k: int) -> dict | None:
        """From (judge, k repeats): one more repeat against one call of the best second judge.

        The combined reading is `w x (mean of the k calls) + (1 - w) x (the second judge's call)`. Its
        error keeps both judges' systematic parts (bias and family effect, per target) with their cross
        term, `(w s1 + (1 - w) s2)^2`, plus `w^2 n1 + (1 - w)^2 n2` of the two independent noises. `w` is
        the weight in [0, 1] with the lowest such error at the posterior means, so a second judge is
        never worse than no second judge. No bias correction is assumed. A person is not offered as the
        second judge: reviewer time is planned by `eval labels next`. In the unit the engine fitted in
        (Units)."""
        base = self.fitted.get((jid, did), {})
        if f"mse{k}" not in base:
            return None
        cost = self.cost.get((jid, did))
        more = base[f"mse{k}"] - base["sigma"] ** 2 / k + base["sigma"] ** 2 / (k + 1)
        out: dict = {"judge": jid, "repeats": k, "rmse": self._rmse(jid, did, k),
                     "more_repeats": {"repeats": k + 1, "rmse": float(np.sqrt(more.mean())),
                                      "cost_usd": None if cost is None else (k + 1) * cost},
                     "second_judge": None, "second_judge_beats_repeat": None}
        best = None
        first = float(base[f"mse{k}"].mean())
        for declared in self.declared:
            other = declared["id"]
            second = self.fitted.get((other, did), {})
            if other == jid or "mse1" not in second or declared["kind"] == "person":
                continue
            other_mse = float(second["mse1"].mean())
            cross = float((base["shift"] * second["shift"]).mean())   # the two systematic errors, per target
            spread = first + other_mse - 2.0 * cross
            weight = min(1.0, max(0.0, (other_mse - cross) / spread)) if spread > 0 else 0.5
            combined = weight ** 2 * first + (1.0 - weight) ** 2 * other_mse + 2.0 * weight * (1.0 - weight) * cross
            rmse = float(np.sqrt(max(combined, 0.0)))
            if best is None or rmse < best["rmse"]:
                other_cost = self.cost.get((other, did))
                best = {"judge": other, "weight": weight, "rmse": rmse,
                        "cost_usd": None if cost is None or other_cost is None else k * cost + other_cost}
        if best is not None:
            out["second_judge"] = best
            out["second_judge_beats_repeat"] = bool(best["rmse"] < out["more_repeats"]["rmse"] - 1e-9)
        return out

    def choices(self, target_error: float | None, budget_usd: float | None) -> list[dict]:
        """Per dimension, the table of (judge, repeats) with error and cost, and the choices asked for.

        The error is the root of the posterior mean `mse(k)` on the fitting scale, in the unit the engine
        fitted in (Units), and `target_error` is read in it. A cost is for one output on this component:
        repeats times the cost of one reading."""
        out = []
        for dim in self.dims:
            did, factor = dim["id"], own_factor(dim)
            table = []
            for jid in self.fit.judges:
                cost = self.cost.get((jid, did))
                for k in REPEATS:
                    rmse = self._rmse(jid, did, k)
                    if rmse is not None:
                        table.append({"judge": jid, "repeats": k, "rmse": rmse,
                                      "rmse_own": None if factor is None else rmse * factor,
                                      "cost_usd": None if cost is None else k * cost})
            if not table:
                continue
            block: dict = {"dimension": did, "unit": (dim.get("unit") or {}).get("fitted_in"), "table": table,
                           "second_judge": [step for jid in self.fit.judges
                                            if (step := self.next_step(jid, did, 1)) is not None]}
            priced = [row for row in table if row["cost_usd"] is not None]
            if target_error is not None:
                reach = [row for row in priced if row["rmse"] <= target_error]
                pick = min(reach, key=lambda row: (row["cost_usd"], row["rmse"])) if reach else None
                block["target_error"] = {
                    "error": target_error, "reached": pick is not None, "choice": pick,
                    "next": None if pick is None else self.next_step(pick["judge"], did, pick["repeats"]),
                    "closest": None if pick is not None or not priced else min(priced, key=lambda row: row["rmse"]),
                    "reason": None if pick is not None else (
                        "no judge states cost_usd_per_call" if not priced else
                        f"no judge reaches it within {REPEATS[-1]} repeats: bias and tau do not shrink with repeats")}
            if budget_usd is not None:
                within = [row for row in priced if row["cost_usd"] <= budget_usd]
                pick = min(within, key=lambda row: (row["rmse"], row["cost_usd"])) if within else None
                block["budget"] = {
                    "usd": budget_usd, "choice": pick,
                    "next": None if pick is None else self.next_step(pick["judge"], did, pick["repeats"]),
                    "reason": None if pick is not None else (
                        "no judge states cost_usd_per_call" if not priced else "one call of every judge costs more")}
            out.append(block)
        return out

    def _difference(self, first: dict, second: dict) -> list[dict]:
        """Per dimension both judges have, the difference first minus second, per draw."""
        rows = []
        for dim in self.dims:
            a, b = self.raw.get((first["id"], dim["id"])), self.raw.get((second["id"], dim["id"]))
            if not a or not b:
                continue
            factor = own_factor(dim)
            row: dict = {"dimension": dim["id"]}
            for name, power in (("bias", 1), ("tau", 1), ("sigma", 1), ("mse1", 2), ("false_pass", 0),
                                ("false_fail", 0)):
                if not power and dim["scale"]["kind"] not in ("binary", "items"):
                    continue
                if name in a and name in b:
                    x = a[name] - b[name]
                    sign = _sign(dim) if name == "bias" else 1.0
                    row[name] = {"fitting": stat(x), "own": _scaled(x, factor, power=power, sign=sign) if power else None}
                else:
                    row[name] = None
            rows.append(row)
        return rows

    def comparisons(self) -> tuple[list[dict], list[dict]]:
        """Batching loss (the pair differs only in `mode`) and reference effect (only in `reference.shown`)."""
        def key(judge: dict, *names: str) -> str:
            return json.dumps([judge.get(name) for name in names], sort_keys=True)

        batching, reference = [], []
        for n, first in enumerate(self.declared):
            for second in self.declared[n + 1:]:
                if key(first, "kind", "model", "effort", "prompt") != key(second, "kind", "model", "effort", "prompt"):
                    continue
                shown = [bool((judge.get("reference") or {}).get("shown")) for judge in (first, second)]
                if key(first, "reference") == key(second, "reference") and first.get("mode") != second.get("mode"):
                    panel, single = (first, second) if first.get("mode") == "panel" else (second, first)
                    per_call = [components_per_call(panel, dim, self.dims) for dim in self.dims
                                if _scores_component(panel, dim)]
                    batching.append({"panel": panel["id"], "per_component": single["id"],
                                     "difference": "panel minus per_component",
                                     "calls_saved_per_output": max(per_call, default=1) - 1,
                                     "dims": self._difference(panel, single)})
                elif first.get("mode") == second.get("mode") and shown[0] != shown[1]:
                    with_ref, without = (first, second) if shown[0] else (second, first)
                    reference.append({"shown": with_ref["id"], "not_shown": without["id"],
                                      "difference": "shown minus not_shown",
                                      "author": with_ref["reference"].get("author"),
                                      "dims": self._difference(with_ref, without)})
        return batching, reference


def stale_inputs(fit: F.Fit, study: str | None, strict: bool) -> dict:
    """Compare the digests the fit recorded with the study folder. With `strict` a change is a refusal.

    The study files are looked for under their fixed names. The rubric and the judges file are checked
    only when a file of the recorded name sits in the study folder; otherwise they are listed as not checked."""
    recorded = fit["inputs"]
    if not recorded:
        return {"study": None, "changes": [], "not_checked": []}
    folder = Path(study) if study else fit.path.parent
    files, checked, skipped = {}, {}, []
    for role, entry in recorded.items():
        path = folder / STUDY_ROLES.get(role, entry["file"])
        if role in STUDY_ROLES or path.is_file():
            files[role], checked[role] = path, entry
        else:
            skipped.append(role)
    for role, name in STUDY_ROLES.items():
        if (folder / name).is_file():
            files[role] = folder / name
    changes = C.stale_inputs(checked, files)
    if changes and strict:
        raise ContractError(fit.path.name, "inputs", "; ".join(changes), "fit again, or drop --strict", "stale")
    return {"study": str(folder), "changes": changes, "not_checked": sorted(skipped)}


def report(fit: F.Fit, target_error: float | None = None, budget_usd: float | None = None,
           scores: list[dict] | None = None, items: list[dict] | None = None) -> dict:
    """The whole report of `eval judges` as one JSON-ready mapping. `scores` are the rows of scores.jsonl:
    only the number of items each call read on an `items` component is taken from them. `items`, the
    rows of items.jsonl, are accepted and not used: a list of items does not say what a call read."""
    for name, value in (("--target-error", target_error), ("--budget-usd", budget_usd)):
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ContractError(fit.path.name, name, f"{value:g} is not above 0", "give a number above 0",
                                "out_of_range")
    builder = _Report(fit, item_counts(fit, scores or []))
    judges = [builder.judge(j, judge) for j, judge in enumerate(builder.declared)]
    batching, reference = builder.comparisons()
    anchor = fit["anchor"]
    return {"fit": fit.path.name, "converged": fit.converged, "draws": fit.sizes["S"],
            "anchor": {"kind": anchor["kind"], "judges": anchor.get("judges", []), "dimensions": anchor["dimensions"]},
            "repeats": list(REPEATS), "judges": judges, "choices": builder.choices(target_error, budget_usd),
            "batching_loss": batching, "reference_effect": reference, "not_measured": builder.not_measured}


# ---------------------------------------------------------------------------------------------------
# Text

def _f(value: Any, digits: int = 3, signed: bool = False) -> str:
    if isinstance(value, dict):
        value = value.get("mean")
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _interval(value: dict | None, digits: int = 2) -> str:
    if value is None:
        return "not measured"
    return f"{value['mean']:.{digits}f} ({value['lo']:.{digits}f} to {value['hi']:.{digits}f})"


# the unit of an error on a choice line, where it is a judge's noise (Units)
UNIT_WORDS = {"call_noise": " (in the anchors' call noise)",
              "pair_noise": " (in the anchors' call noise of one comparison)"}


def render(payload: dict) -> str:
    """The report as text: one table per judge, then the choices, the comparisons and what is not measured."""
    lines = [f"eval judges (experimental): {payload['fit']}, {len(payload['judges'])} judges, "
             f"{payload['draws']} draws" + ("" if payload["converged"] else ", NOT CONVERGED"),
             "Figures are posterior means on the fitting scale; intervals are 90%. A dash is not measured."]
    for unit, what in (("total_noise", "reading"), ("total_pair_noise", "comparison")):
        dims = list(dict.fromkeys(row["dimension"] for judge in payload["judges"] for row in judge["dims"]
                                  if row.get("unit") == unit))
        if dims:
            lines.append(f"On {', '.join(dims)}: bias, tau, sigma and mse are in the anchors' total noise of one "
                         f"{what}. The choices use the scale the engine fits, named on each choice line.")
    for judge in payload["judges"]:
        what = [judge["kind"]]
        if judge.get("model"):
            what.append(judge["model"]["id"])
        if judge.get("effort"):
            what.append(f"effort {judge['effort']}")
        what.append(judge.get("mode") or "per_component")
        if (judge.get("reference") or {}).get("shown"):
            what.append("reference shown")
        if judge.get("cost_usd_per_call") is not None:
            what.append(f"{judge['cost_usd_per_call']:g} USD per call")
        if judge.get("anchor"):
            what.append("anchor")
        lines += ["", f"judge {judge['id']} ({', '.join(what)})",
                  f"  {'dimension':<28}{'bias':>8}{'tau':>8}{'sigma':>8}{'mse(1)':>9}{'mse(3)':>9}{'r2':>7}{'eff(3)':>8}"]
        for row in judge["dims"]:
            mse = {entry["repeats"]: entry["fitting"] for entry in row["mse"]["by_repeats"]}
            eff = row["effective_repeats"]["by_repeats"]
            lines.append(f"  {row['dimension']:<28}{_f(row['bias'] and row['bias']['fitting'], signed=True):>8}"
                         f"{_f(row['tau']['fitting']):>8}{_f(row['sigma']['fitting']):>8}{_f(mse.get(1), 4):>9}"
                         f"{_f(mse.get(3), 4):>9}{_f(row['r2']['one_call'], 2):>7}"
                         f"{_f(eff[2] if len(eff) > 2 else None, 2):>8}")
        for row in judge["dims"]:
            status = row.get("status")
            if status and status["measured"]:
                per = " per item" if status["model"] == "k_of_n" else ""
                lines.append(f"  {row['dimension']}{per}: false pass {_interval(status['false_pass'])}, false fail "
                             f"{_interval(status['false_fail'])}, flip {_interval(status['flip'])}, agreement "
                             f"{_interval(status['agreement'])}")
            pairs = row.get("pairs")
            if pairs and pairs["measured"]:
                lines.append(f"  {row['dimension']} pairs: tie rate {_interval(pairs['tie_rate'])}, picks the better "
                             f"output {_interval(pairs['correct'])}")
        if judge["halo"]["measured"]:
            call = (_interval(judge["halo"]["call"]) if judge.get("mode") == "panel"
                    else "does not apply (the judge reads one component per call)")
            lines.append(f"  halo: shared share of output disagreement {_interval(judge['halo']['target'])}, of call "
                         f"noise {call}")
        if judge.get("pairs"):
            lines.append(f"  pairs: position bias {_interval(judge['pairs']['position']['fitting'])}, tie width "
                         f"{_interval(judge['pairs']['tie_width']['fitting'])}")
        family = judge.get("family") or {}
        for name in ("own_family", "reference_family"):
            block = family.get(name)
            if not block:
                continue
            label = name.replace("_", " ")
            if block["state"] == "confounded":
                lines.append(f"  {label} ({block['family']}): confounded with the own-family effect")
            for row in block["dims"]:
                mark = ("  <- interval excludes 0" + UNIT_WORDS.get(row.get("decided_in"), "")
                        if row["excludes_zero"] else "")
                lines.append(f"  {label} ({block['family']}) on {row['dimension']}: {_interval(row['effect'])}{mark}")
    for block in payload["choices"]:
        for name, label in (("target_error", "cheapest choice for error"), ("budget", "best choice within USD")):
            asked = block.get(name)
            if not asked:
                continue
            amount = asked.get("error", asked.get("usd"))
            if asked["choice"] is None:
                lines.append(f"{block['dimension']}: {label} {amount:g}: none ({asked['reason']})")
                continue
            pick, step = asked["choice"], asked["next"]
            unit = UNIT_WORDS.get(block.get("unit"), "")
            line = (f"{block['dimension']}: {label} {amount:g}: {pick['judge']} x {pick['repeats']}, error "
                    f"{pick['rmse']:.3f}{unit}, {pick['cost_usd']:g} USD")
            if name == "target_error" and step and step["second_judge"]:
                better = "beats" if step["second_judge_beats_repeat"] else "does not beat"
                line += f"; adding {step['second_judge']['judge']} {better} one more repeat"
            lines.append(line)
    for name, label, first, second in (("batching_loss", "batching loss", "panel", "per_component"),
                                       ("reference_effect", "reference effect", "shown", "not_shown")):
        for pair in payload[name]:
            lines += ["", f"{label}: {pair[first]} minus {pair[second]}" + (
                f" ({pair['calls_saved_per_output']} calls saved per output)" if name == "batching_loss" else "")]
            for row in pair["dims"]:
                parts = [f"{key} {_f(row[key] and row[key]['fitting'], signed=True)}" for key in
                         ("bias", "tau", "sigma", "mse1", "false_pass", "false_fail") if key in row]
                lines.append(f"  {row['dimension']}: " + ", ".join(parts))
    if payload["not_measured"]:
        lines += ["", "not measured by this design:"]
        for entry in payload["not_measured"]:
            who = " of ".join(x for x in (entry["quantity"], entry["judge"]) if x)
            where = f" on {entry['dimension']}" if entry["dimension"] else ""
            lines.append(f"  {who}{where} ({entry['state']}): needs {entry['needs']}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    fit = F.read_fit(args.fit, allow_unconverged=getattr(args, "allow_unconverged", False))
    stale = stale_inputs(fit, getattr(args, "study", None), getattr(args, "strict", False))
    folder = Path(args.study) if getattr(args, "study", None) else fit.path.parent
    files = {"fit": fit.path, "draws": F.draws_path(fit.path)}
    scores: list[dict] = []
    if any(dim["scale"]["kind"] == "items" for dim in fit.dimensions) and (folder / C.SCORES_FILE).is_file():
        scores = C.read_lines(folder / C.SCORES_FILE, C.SCHEMA_SCORES)
        files["scores"] = folder / C.SCORES_FILE
    payload = report(fit, getattr(args, "target_error", None), getattr(args, "budget_usd", None), scores)
    payload["stale"] = stale
    payload["inputs"] = C.input_digests(files)
    payload["options"] = {"target_error": getattr(args, "target_error", None),
                          "budget_usd": getattr(args, "budget_usd", None),
                          "allow_unconverged": bool(getattr(args, "allow_unconverged", False)),
                          "strict": bool(getattr(args, "strict", False)),
                          "study": str(args.study) if getattr(args, "study", None) else None,
                          "repeats": list(REPEATS), "max_pairs": MAX_PAIRS}
    for change in stale["changes"]:
        print(f"warning: {fit.path.name}: {change}", file=sys.stderr)
    if getattr(args, "json", False):
        emit_json(SCHEMA, payload)
    else:
        sys.stdout.write(render(payload))
    return EXIT_OK
