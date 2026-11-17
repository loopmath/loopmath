"""`eval judges`: which judge, and how many repeats, from a fit.

Reads only the fit file (and, for the stale-input warning, the digests of the study files). Every figure
is computed per posterior draw and reported as a stat: `{"mean", "sd", "lo", "hi"}` with the 90% interval.

Per judge and dimension, on the fitting scale and on the component's own scale where one exists:

- `bias`, `tau` (judge-by-output disagreement), `sigma` (run-to-run noise of one call).
- `mse` of the mean of k calls, k = 1 to 5: `bias^2 + tau^2 + sigma^2 / k`. When the judge-by-input term
  is measured its variance is added: like `tau`, it does not shrink with repeats.
- `r2`: the share of the hidden quality's variance over the study's targets that a reading explains,
  `var_q / (var_q + tau^2 + sigma^2 / k)`, with `var_q` the variance between arm means (weighted by
  runs) plus the input and run variances. It is on the fitting scale: for `binary`, `levels` and `items`
  it describes the judge's view before it is cut into a verdict.
- `effective_repeats`: how many independent readings k repeats are worth,
  `(tau^2 + sigma^2) / (tau^2 + sigma^2 / k)`. Its limit, `1 + sigma^2 / tau^2`, is the most repeats can buy.
- `cost`: per call, per reading (a `panel` call is shared by the components it scores), and per unit of
  error removed, where the error removed by k calls is the variance of hidden quality they explain,
  `var_q x r2(k)`.

For a `binary` dimension, over the targets of the study (every run's hidden quality in the draws):
`false_pass`, `false_fail`, `flip` (two calls on one output disagree) and `agreement`. A target's hidden
status is `q >= 0`; one call passes with probability `Phi((q + bias + family) / sqrt(tau^2 + sigma^2))`;
two calls share the judge's view of the output and differ in call noise. For an `items` dimension the same
four figures are per item under the k of n model: an item's hidden status passes with probability
`Phi(q)`, a call passes it with probability `Phi(eta + e)`, independently given the two.

For pairs: `position`, `tie_width`, `sigma_pair`, and over every two runs of one input from different
arms, with both orders equally likely, the tie rate and the chance a call picks the output with the
higher hidden quality.

Family effects per model judge: `own_family` and `reference_family` contrasts (the effect on that family
minus the mean effect on the other families), flagged when the 90% interval excludes 0.

Across judges: a table of (judge, repeats) choices per dimension with error and cost, the cheapest choice
for `--target-error` and the best within `--budget-usd`, whether a second judge beats one more repeat,
the batching loss (two judges that differ only in `mode`) and the reference effect (two judges that
differ only in `reference.shown`).

A quantity whose term the design does not identify is null, and is listed under `not_measured` with the
design that would measure it. It is never reported as a finding.
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

    def __init__(self, fit: F.Fit) -> None:
        self.fit = fit
        self.draws = fit.draws
        self.dims: list[dict] = fit.dimensions
        self.declared: list[dict] = fit["judges"]
        self.var_q = quality_variance(fit)
        self.not_measured: list[dict] = []
        self.raw: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        self.cost: dict[tuple[str, str], float | None] = {}
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
                     "n_targets": counted.get("n_targets", 0), "n_calls": counted.get("n_calls", 0),
                     "n_pairs": counted.get("n_pairs", 0)}
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

        # The judge-by-input term counts only when the design measures it.
        by_input = np.nan_to_num(draws["sd_judge_input"][:, j, d])
        input_state = term_state(fit, "judge_input", jid, did)
        if input_state in F.MEASURED_STATES:
            out["sd_judge_input"] = {"fitting": stat(by_input), "own": _scaled(by_input, factor),
                                     "state": input_state, "measured": True}
        else:
            if input_state == "prior_only":
                self._flag("sd_judge_input", ["judge_input"], jid, did)
            by_input = np.zeros_like(by_input)
        stable = by_input ** 2 + tau ** 2          # what repeats do not shrink, besides the bias

        noise_ok = bool(ok["tau"] and ok["sigma"])
        var_q = self.var_q[:, d]
        out["mse"] = {"measured": noise_ok and present["bias"], "by_repeats": []}
        out["r2"] = {"measured": noise_ok, "one_call": None, "by_repeats": []}
        out["effective_repeats"] = {"measured": noise_ok, "by_repeats": [], "limit": None}
        per_call = judge.get("cost_usd_per_call")
        per_reading = None if per_call is None else per_call / components_per_call(judge, dim, self.dims)
        out["cost"] = {"usd_per_call": per_call, "usd_per_reading": per_reading, "by_repeats": []}
        self.cost[(jid, did)] = per_reading
        if noise_ok:
            with np.errstate(invalid="ignore", divide="ignore"):
                for k in REPEATS:
                    noise = stable + sigma ** 2 / k
                    r2 = var_q / (var_q + noise)
                    out["r2"]["by_repeats"].append(stat(r2))
                    out["effective_repeats"]["by_repeats"].append(stat((stable + sigma ** 2) / noise))
                    raw[f"noise{k}"] = noise
                    if present["bias"]:
                        mse = bias ** 2 + noise
                        raw[f"mse{k}"] = mse
                        out["mse"]["by_repeats"].append({"repeats": k, "fitting": stat(mse),
                                                         "own": _scaled(mse, factor, power=2)})
                    if per_reading is not None:
                        removed = var_q * r2
                        out["cost"]["by_repeats"].append({
                            "repeats": k, "usd": k * per_reading, "variance_removed": _mean(removed),
                            "usd_per_variance_removed": _mean(k * per_reading / removed)})
                out["r2"]["one_call"] = out["r2"]["by_repeats"][0]
                out["effective_repeats"]["limit"] = stat((stable + sigma ** 2) / stable)

        if kind in ("binary", "items") and present["bias"]:
            names = ["tau", "sigma"] + (["binary_status"] if kind == "binary" else [])
            measured = noise_ok and self._flag("false_pass, false_fail, flip, agreement", names, jid, did)
            out["status"] = {"measured": bool(measured), "model": "binary" if kind == "binary" else "k_of_n"}
            if measured:
                shift = bias[:, None] + self._family_effect(j, d)
                call_sd = sigma if kind == "binary" else np.sqrt(1.0 + sigma ** 2)
                figures = status_figures(draws["q_run"][:, :, d], shift, np.sqrt(stable), call_sd, kind)
                for name, x in figures.items():
                    out["status"][name] = _mean(x) if name in ("n_pass", "n_fail") else stat(x)
                    raw[name] = x

        if present["sigma_pair"]:
            out["pairs"] = self.pairs(j, jid, d, did, tau, pair_sd, bool(ok["tau"]), factor)
        self.raw[(jid, did)] = raw
        return out

    def pairs(self, j: int, jid: str, d: int, did: str, tau: np.ndarray, pair_sd: np.ndarray, tau_ok: bool,
              factor: float | None) -> dict:
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
        if not (pair_ok and rest and tau_ok and len(a)):
            return out
        effect = self._family_effect(j, d)
        q = draws["q_run"][:, :, d]
        figures = pair_figures(q[:, a], q[:, b], effect[:, a] - effect[:, b], np.sqrt(2 * tau ** 2 + pair_sd ** 2),
                               np.nan_to_num(draws["position"][:, j]), np.nan_to_num(draws["tie_width"][:, j]))
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
                summary = stat(x)
                block["dims"].append({"dimension": dim["id"], "effect": summary,
                                      "own": _scaled(x, own_factor(dim), sign=_sign(dim)),
                                      "excludes_zero": bool(summary["lo"] > 0 or summary["hi"] < 0)})
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
        out["halo"] = {"target": stat(self.draws["halo_target"][:, j]) if halo_ok else None,
                       "call": stat(self.draws["halo_call"][:, j]) if halo_ok else None,
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
        mse = self.raw.get((jid, did), {}).get(f"mse{k}")
        return None if mse is None else float(np.sqrt(mse.mean()))

    def next_step(self, jid: str, did: str, k: int) -> dict | None:
        """From (judge, k repeats): one more repeat against one call of the best second judge.

        The two judges' errors are taken as independent and combined with inverse-MSE weights. A person
        is not offered as the second judge: reviewer time is planned by `eval labels next`."""
        base = self.raw.get((jid, did), {})
        if f"mse{k}" not in base:
            return None
        cost = self.cost.get((jid, did))
        more = base[f"mse{k}"] - base["sigma"] ** 2 / k + base["sigma"] ** 2 / (k + 1)
        out: dict = {"judge": jid, "repeats": k, "rmse": self._rmse(jid, did, k),
                     "more_repeats": {"repeats": k + 1, "rmse": float(np.sqrt(more.mean())),
                                      "cost_usd": None if cost is None else (k + 1) * cost},
                     "second_judge": None, "second_judge_beats_repeat": None}
        best = None
        for declared in self.declared:
            other = declared["id"]
            second = self.raw.get((other, did), {}).get("mse1")
            if other == jid or second is None or declared["kind"] == "person":
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                combined = np.nan_to_num(1.0 / (1.0 / base[f"mse{k}"] + 1.0 / second))
            rmse = float(np.sqrt(combined.mean()))
            if best is None or rmse < best["rmse"]:
                other_cost = self.cost.get((other, did))
                best = {"judge": other, "rmse": rmse,
                        "cost_usd": None if cost is None or other_cost is None else k * cost + other_cost}
        if best is not None:
            out["second_judge"] = best
            out["second_judge_beats_repeat"] = bool(best["rmse"] < out["more_repeats"]["rmse"])
        return out

    def choices(self, target_error: float | None, budget_usd: float | None) -> list[dict]:
        """Per dimension, the table of (judge, repeats) with error and cost, and the choices asked for.

        The error is the root of the posterior mean `mse(k)` on the fitting scale. A cost is for one
        output on this component: repeats times the cost of one reading."""
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
            block: dict = {"dimension": did, "table": table,
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


def report(fit: F.Fit, target_error: float | None = None, budget_usd: float | None = None) -> dict:
    """The whole report of `eval judges` as one JSON-ready mapping."""
    for name, value in (("--target-error", target_error), ("--budget-usd", budget_usd)):
        if value is not None and not (np.isfinite(value) and value > 0):
            raise ContractError(fit.path.name, name, f"{value:g} is not above 0", "give a number above 0",
                                "out_of_range")
    builder = _Report(fit)
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


def render(payload: dict) -> str:
    """The report as text: one table per judge, then the choices, the comparisons and what is not measured."""
    lines = [f"eval judges (experimental): {payload['fit']}, {len(payload['judges'])} judges, "
             f"{payload['draws']} draws" + ("" if payload["converged"] else ", NOT CONVERGED"),
             "Figures are posterior means on the fitting scale; intervals are 90%. A dash is not measured."]
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
            lines.append(f"  halo: shared share of output disagreement {_interval(judge['halo']['target'])}, of call "
                         f"noise {_interval(judge['halo']['call'])}")
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
                mark = "  <- interval excludes 0" if row["excludes_zero"] else ""
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
            line = (f"{block['dimension']}: {label} {amount:g}: {pick['judge']} x {pick['repeats']}, error "
                    f"{pick['rmse']:.3f}, {pick['cost_usd']:g} USD")
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
    payload = report(fit, getattr(args, "target_error", None), getattr(args, "budget_usd", None))
    payload["stale"] = stale
    for change in stale["changes"]:
        print(f"warning: {fit.path.name}: {change}", file=sys.stderr)
    if getattr(args, "json", False):
        emit_json(SCHEMA, payload)
    else:
        sys.stdout.write(render(payload))
    return EXIT_OK
