"""`eval value`: value and gate pass probability per configuration, from a fit and a value file.

The value of a run with hidden final quality `q` and reported quality `m = m(q)` is

    pass(q) = 1 when m[d] >= t_d for every gate, else 0
    V(q)    = pass(q) x w'm + (1 - pass(q)) x gated_value

Per arm and per posterior draw this module integrates `V` over a new input and a new run,
`q = mu_arm + v + u` with `v ~ N(0, Sigma_input)` and `u ~ N(0, Sigma_run)`. Judge terms do not enter:
the value is of the hidden quality, not of a judge's reading. The spread of the per-draw figures over
the posterior draws gives every interval (the 5th to the 95th percentile, as in the fit file).

How each figure is computed
---------------------------
- `components`: the arm's reported quality per final dimension, in closed form (`quality.arm_reported`).
- one gate alone: closed form, `P(q[d] >= cut)` under `q[d] ~ N(mu, sd_input^2 + sd_run^2)`.
- no gates (the compensatory case): `E[V] = w'E[m]`, in closed form. Only means matter.
- gates: `P(all gates pass)` and `E[V]` by numerical integration over `NEW_RUNS` points per posterior
  draw. A new run on a new input has `q ~ N(mu_arm, Sigma_input + Sigma_run)`; the points are a Sobol
  sequence with a seeded shift, mapped to that normal. The same points serve every arm and every draw,
  so the integration error does not blur a difference between two arms or widen an interval. With one
  gate `p_pass` is the closed form.
- `p_pass_if_independent`: the product of the single-gate figures, what a model that ignores the
  correlation between gated components would report. It is shown beside `p_pass`, never used.

Comparisons are each arm against the reference: the difference in value and in `p_pass` per draw,
both arms read over the same population of new inputs (the paired comparison: an input's effect is
common to both arms), `p_better` (the share of draws where the arm's value is higher, ties counted
half) and the verdict of the objective (`adopt`, `reject` or `more`):

- `best_value`: adopt when `p_better >= 0.95`, reject when `p_better <= 0.05`.
- `cheapest_within_margin`: `p_within` is the share of draws where the arm's value is above the
  reference's minus `margin` (in value units). Adopt when `p_within >= 0.95` and the arm's mean run cost
  is lower; reject when `p_within <= 0.05` or the arm is not cheaper.
- `best_under_cost`: adopt when `p_better >= 0.95` and the mean run cost is at or under `cost_cap_usd`;
  reject when `p_better <= 0.05` or the cost is over the cap.
- `cost_per_pass`: `p_lower` is the share of draws where cost divided by `p_pass` is lower than the
  reference's. Adopt at 0.95, reject at 0.05.

A rule that needs a cost never adopts without one: the verdict is then `more`.

`arm_draws` is the shared piece: `eval steps` reads the same per-draw figures.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import qmc

from ..output import EXIT_OK, emit_json
from . import contracts as C
from . import fitfile as F
from . import quality as Q

REPORT_SCHEMA = "loopmath.eval-value-report/1"
NEW_RUNS = 4096    # integration points per posterior draw (a power of 2)
ADOPT_AT = 0.95    # a verdict adopts when its probability is at least this
REJECT_AT = 0.05   # and rejects when it is at most this
STUDY_ROLES = {"runs": C.RUNS_FILE, "scores": C.SCORES_FILE, "pairs": C.PAIRS_FILE, "items": C.ITEMS_FILE}
_CELLS = 2_000_000  # numbers held at once by the integral


@dataclass
class ArmDraws:
    """Per posterior draw and arm, the figures of a new run on a new input.

    `value[S, A]` is the expected value, `p_pass[S, A]` the chance every gate passes, `p_gate[S, A, G]`
    the chance of each gate alone, `reported[S, A, D]` the arm's reported quality per dimension (NaN
    where the fit has none). `exact` is true when every figure is a closed form; `new_runs` is the
    number of integration points otherwise."""

    value: np.ndarray
    p_pass: np.ndarray
    p_gate: np.ndarray
    reported: np.ndarray
    exact: bool
    new_runs: int


def roots(cov: np.ndarray) -> np.ndarray:
    """Matrix square roots of a stack of covariance matrices `[..., K, K]`.

    Cholesky where it exists: it is continuous in the matrix, so shared integration points give a smooth
    integral across posterior draws. A singular matrix (a factor structure, a zero sd) takes the
    eigenvalue root."""
    try:
        return np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(cov)
        return v * np.sqrt(np.clip(w, 0.0, None))[..., None, :]


def gate_pass(mu: Any, sd: Any, kind: str, threshold: float) -> np.ndarray:
    """`P(m(q) >= threshold)` for `q ~ N(mu, sd^2)`: the chance one gate passes, in closed form."""
    mu, sd = np.broadcast_arrays(np.asarray(mu, dtype=float), np.asarray(sd, dtype=float))
    if kind == "binary":
        cut = 0.0
    elif kind == "items":
        cut = float(ndtri(min(max(threshold, 0.0), 1.0)))
    else:
        cut = threshold if threshold > 0 else -np.inf
    if cut == -np.inf:
        return np.ones(mu.shape)
    safe = np.where(sd > 0, sd, 1.0)
    return np.where(sd > 0, ndtr((mu - cut) / safe), (mu >= cut).astype(float))


def needed_dims(spec: Q.ValueSpec) -> list[int]:
    """The positions of the dimensions a value file reads: weighted or gated."""
    return sorted(set(np.flatnonzero(spec.weights).tolist()) | {d for d, _ in spec.gates})


def check_dims(fit: F.Fit, dims: Sequence[int]) -> None:
    """Refuse a fit that holds no estimate for a dimension the value file reads."""
    draws = fit.draws
    for d in dims:
        if np.isnan(draws["mu"][:, :, d]).any() or np.isnan(draws["sd_input"][:, d]).any() or np.isnan(
                draws["sd_run"][:, d]).any():
            raise C.ContractError(fit.path.name, f"draws: {fit.dimensions[d]['id']}",
                                  "the fit holds no estimate for a dimension the value file weights or gates",
                                  "fit with this component, or drop it from the value file", "no_estimate")


def value_of(m: np.ndarray, spec: Q.ValueSpec, dims: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """`(V, pass)` for reported quality `m[..., K]` over the dimensions `dims` only."""
    at = {d: k for k, d in enumerate(dims)}
    ok = np.ones(m.shape[:-1], dtype=bool)
    for d, threshold in spec.gates:
        ok &= m[..., at[d]] >= threshold
    weighted = np.zeros(m.shape[:-1])
    for d in np.flatnonzero(spec.weights):
        weighted = weighted + spec.weights[d] * m[..., at[int(d)]]
    return np.where(ok, weighted, spec.gated_value), ok


def points(n: int, k: int, seed: int = 0) -> np.ndarray:
    """`[n', k]` standard normal integration points, `n'` the next power of 2 from `n`.

    A Sobol sequence with one seeded shift for all points, mapped by the normal quantile. The same `n`,
    `k` and `seed` give the same points."""
    m = max(1, int(np.ceil(np.log2(max(int(n), 2)))))
    base = qmc.Sobol(d=k, scramble=False).random_base2(m)
    shifted = (base + np.random.default_rng(seed).random(k)) % 1.0
    return ndtri(np.clip(shifted, 1e-12, 1.0 - 1e-12))


def sub_cov(draws: Any, level: str, dims: Sequence[int], sl: slice) -> np.ndarray:
    """The `input` or `run` covariance over the dimensions `dims`, for the draws `sl`: `[s, K, K]`."""
    corr = np.asarray(draws[f"corr_{level}"])[sl][:, dims][:, :, dims]
    return Q.covariance(np.asarray(draws[f"sd_{level}"])[sl][:, dims], corr)


def arm_draws(fit: F.Fit, spec: Q.ValueSpec, seed: int = 0, new_runs: int = NEW_RUNS) -> ArmDraws:
    """The per-draw figures of every arm (see the module docstring for the method)."""
    draws, kinds = fit.draws, fit.kinds
    mu = np.asarray(draws["mu"], dtype=float)
    S, A, D = mu.shape
    dims = needed_dims(spec)
    check_dims(fit, dims)
    total = np.sqrt(np.asarray(draws["sd_input"]) ** 2 + np.asarray(draws["sd_run"]) ** 2)
    reported = np.stack([Q.arm_reported(mu[:, :, d], total[:, None, d], kinds[d]) for d in range(D)], axis=-1)
    if spec.gates:
        p_gate = np.stack([gate_pass(mu[:, :, d], total[:, None, d], kinds[d], t) for d, t in spec.gates], axis=-1)
    else:
        p_gate = np.ones((S, A, 0))
    used = np.flatnonzero(spec.weights)
    if not spec.gates:
        return ArmDraws(reported[..., used] @ spec.weights[used], np.ones((S, A)), p_gate, reported, True, 0)

    K = len(dims)
    z = points(new_runs, K, seed)
    n = z.shape[0]
    sub_kinds = [kinds[d] for d in dims]
    value, p_pass = np.empty((S, A)), np.empty((S, A))
    step = max(1, _CELLS // (n * A * K))
    for start in range(0, S, step):
        sl = slice(start, start + step)
        root = roots(sub_cov(draws, "input", dims, sl) + sub_cov(draws, "run", dims, sl))
        spread = np.einsum("sde,ne->snd", root, z)
        q = mu[sl][:, None, :, :][..., dims] + spread[:, :, None, :]
        worth, ok = value_of(Q.reported_all(q, sub_kinds), spec, dims)
        value[sl], p_pass[sl] = worth.mean(axis=1), ok.mean(axis=1)
    if len(spec.gates) == 1:
        p_pass = p_gate[..., 0].copy()
    return ArmDraws(value, p_pass, p_gate, reported, False, n)


def p_higher(a: np.ndarray, b: np.ndarray) -> float:
    """The share of draws where `a > b`, ties counted half."""
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


def verdict(objective: dict, value: np.ndarray, ref_value: np.ndarray, p_pass: np.ndarray, ref_p_pass: np.ndarray,
            cost: float | None, ref_cost: float | None) -> dict:
    """The objective's verdict for one arm against the reference: `verdict`, `rule`, `reason` and the
    probability the rule read."""
    kind = objective["kind"]
    out: dict = {"rule": kind}

    def decide(p: float, name: str, adopt_block: str | None = None, reject_now: str | None = None) -> dict:
        out[name] = p
        if reject_now is not None:
            out.update(verdict="reject", reason=reject_now)
        elif p <= REJECT_AT:
            out.update(verdict="reject", reason=f"{name} {p:.3f} is at most {REJECT_AT:g}")
        elif p >= ADOPT_AT and adopt_block is None:
            out.update(verdict="adopt", reason=f"{name} {p:.3f} is at least {ADOPT_AT:g}")
        elif p >= ADOPT_AT:
            out.update(verdict="more", reason=adopt_block)
        else:
            out.update(verdict="more", reason=f"{name} {p:.3f} is between {REJECT_AT:g} and {ADOPT_AT:g}")
        return out

    no_cost = "cost unknown: state cost_usd on the runs of both arms"
    p_better = p_higher(value, ref_value)
    if kind == "best_value":
        return decide(p_better, "p_better")
    if kind == "cheapest_within_margin":
        p_within = p_higher(value, ref_value - float(objective["margin"]))
        if cost is None or ref_cost is None:
            return decide(p_within, "p_within", adopt_block=no_cost)
        dearer = None if cost < ref_cost else "the arm is not cheaper than the reference"
        return decide(p_within, "p_within", reject_now=dearer)
    if kind == "best_under_cost":
        cap = objective.get("cost_cap_usd")
        if cap is None:
            return decide(p_better, "p_better")
        if cost is None:
            return decide(p_better, "p_better", adopt_block=no_cost)
        over = None if cost <= cap else f"the mean run cost {cost:g} is over the cap {cap:g}"
        return decide(p_better, "p_better", reject_now=over)
    if cost is None or ref_cost is None:
        out.update(p_lower=None, verdict="more", reason=no_cost)
        return out
    per_pass = np.where(p_pass > 0, cost / np.where(p_pass > 0, p_pass, 1.0), np.inf)
    ref_per_pass = np.where(ref_p_pass > 0, ref_cost / np.where(ref_p_pass > 0, ref_p_pass, 1.0), np.inf)
    return decide(p_higher(ref_per_pass, per_pass), "p_lower")


def pick_reference(fit: F.Fit, option: str | None, objective: dict) -> tuple[int, str]:
    """The reference arm's position and where it came from: `option`, `objective` or `first_arm`."""
    if option is not None:
        return fit.arm(option), "option"
    if objective.get("reference") is not None:
        return fit.arm(objective["reference"]), "objective"
    return 0, "first_arm"


def stale_warnings(fit: F.Fit, study: Path, strict: bool, extra: dict[str, Path] | None = None) -> list[str]:
    """Stale inputs: what changed between the digests the fit recorded and the study folder now.

    The study files (`runs`, `scores`, `pairs`, `items`) are always checked. The rubric and the judges
    file are checked when a file of the recorded name sits in the study folder. `extra` names a file by
    role outright. With `strict` a change is a refusal."""
    recorded = fit["inputs"]
    files: dict[str, Path] = {}
    for role, name in STUDY_ROLES.items():
        if role in recorded or (study / name).is_file():
            files[role] = study / (recorded[role]["file"] if role in recorded else name)
    for role in recorded:
        if role not in STUDY_ROLES and (study / recorded[role]["file"]).is_file():
            files[role] = study / recorded[role]["file"]
    files.update(extra or {})
    known = {role: row for role, row in recorded.items() if role in files}
    changes = C.stale_inputs(known, files)
    if changes and strict:
        raise C.ContractError(fit.path.name, "inputs", "; ".join(changes), "fit again, or drop --strict", "stale")
    return changes


def read_inputs(args: argparse.Namespace) -> tuple[F.Fit, Q.ValueSpec, Path, list[str]]:
    """What `eval value` and `eval steps` both read: the fit, the value file compiled against it, the
    study folder and the first warnings."""
    fit = F.read_fit(args.fit, allow_unconverged=getattr(args, "allow_unconverged", False))
    value = C.read_document(args.value, C.SCHEMA_VALUE)
    spec = Q.compile_value(value, fit.dimensions, Path(args.value).name)
    study = Path(args.study) if getattr(args, "study", None) else fit.path.parent
    warnings = []
    if not fit.converged:
        warnings.append("the fit did not converge: read every figure with care")
    for d in np.flatnonzero(spec.weights):
        if fit.dimensions[int(d)].get("relative"):
            warnings.append(f"the weighted dimension {fit.dimensions[int(d)]['id']} is relative (pairs only): "
                            "only differences between arms are measured")
    return fit, spec, study, warnings


def report(fit: F.Fit, spec: Q.ValueSpec, reference: int, seed: int = 0, new_runs: int = NEW_RUNS) -> dict:
    """The `method`, `arms` and `comparisons` members of the value report."""
    arms = arm_draws(fit, spec, seed, new_runs)
    dims = fit.dimensions
    records = fit["arms"]
    rows = []
    for a, arm in enumerate(fit.arms):
        components = []
        for d in fit.final_dims():
            m = arms.reported[:, a, d]
            relative = bool(dims[d].get("relative"))
            own = None if relative or np.isnan(m).any() else Q.own_scale(m, dims[d])
            components.append({"dimension": dims[d]["id"], "component": dims[d]["component"],
                               "weight": float(spec.weights[d]), "relative": relative,
                               "reported": None if relative else F.stat(m),
                               "own": None if own is None else F.stat(own)})
        rows.append({"arm": arm, "mixed": bool(records[a].get("mixed")), "n_runs": records[a].get("n_runs"),
                     "components": components, "p_pass": F.stat(arms.p_pass[:, a]),
                     "gates": [{"dimension": label, "threshold": float(spec.gates[g][1]),
                                "p_pass": F.stat(arms.p_gate[:, a, g])} for g, label in enumerate(spec.gate_labels)],
                     "p_pass_if_independent": F.stat(arms.p_gate[:, a, :].prod(axis=-1)),
                     "value": F.stat(arms.value[:, a]), "cost_usd": records[a].get("cost_usd")})
    comparisons = []
    ref_cost = records[reference].get("cost_usd")
    for a, arm in enumerate(fit.arms):
        if a == reference:
            continue
        cost = records[a].get("cost_usd")
        row = {"arm": arm, "reference": fit.arms[reference],
               "value": F.stat(arms.value[:, a] - arms.value[:, reference]),
               "p_pass": F.stat(arms.p_pass[:, a] - arms.p_pass[:, reference]),
               "p_better": p_higher(arms.value[:, a], arms.value[:, reference]),
               "cost_usd": None if cost is None or ref_cost is None else float(cost - ref_cost)}
        row.update(verdict(spec.objective, arms.value[:, a], arms.value[:, reference], arms.p_pass[:, a],
                           arms.p_pass[:, reference], cost, ref_cost))
        comparisons.append(row)
    method = {"value": "closed form" if arms.exact else "integration", "new_runs": arms.new_runs,
              "paired_on_inputs": True, "interval": "5th to 95th percentile of the posterior draws",
              "draws": int(arms.value.shape[0]), "adopt_at": ADOPT_AT, "reject_at": REJECT_AT}
    return {"method": method, "arms": rows, "comparisons": comparisons}


def spec_record(fit: F.Fit, spec: Q.ValueSpec) -> dict:
    """The value file as compiled: weights and gates by dimension, thresholds on reported quality."""
    dims = fit.dimensions
    return {"weights": {dims[int(d)]["id"]: float(spec.weights[d]) for d in np.flatnonzero(spec.weights)},
            "gates": [{"dimension": label, "threshold": float(gate[1])}
                      for label, gate in zip(spec.gate_labels, spec.gates)],
            "gated_value": spec.gated_value, "objective": spec.objective}


def fmt(stat: dict | None, digits: int = 3) -> str:
    """A stat as `mean [lo, hi]`."""
    if stat is None:
        return "not measured"
    return f"{stat['mean']:.{digits}f} [{stat['lo']:.{digits}f}, {stat['hi']:.{digits}f}]"


def print_warnings(warnings: Sequence[str]) -> None:
    for line in warnings:
        print(f"warning: {line}", file=sys.stderr)


def _print(payload: dict) -> None:
    print(f"eval value (experimental): {len(payload['arms'])} arms, reference {payload['reference']}, "
          f"objective {payload['objective']['kind']}")
    gated = {gate["dimension"] for gate in payload["gates"]}
    for row in payload["arms"]:
        cost = "" if row["cost_usd"] is None else f"  cost ${row['cost_usd']:.4g}"
        print(f"  {row['arm']}: value {fmt(row['value'])}  p_pass {fmt(row['p_pass'])}{cost}")
        for gate in row["gates"]:
            print(f"      gate {gate['dimension']} alone: {fmt(gate['p_pass'])}")
        for comp in row["components"]:
            if comp["weight"] or comp["dimension"] in gated:
                print(f"      {comp['dimension']}: {fmt(comp['reported'])}")
    for row in payload["comparisons"]:
        print(f"  {row['arm']} against {row['reference']}: value {fmt(row['value'])}  p_pass {fmt(row['p_pass'])}  "
              f"p_better {row['p_better']:.3f}  {row['verdict']} ({row['reason']})")
    check = payload.get("design_check")
    if check is not None:
        print(f"  design-based check ({100 * check['level']:.0f}% intervals, scores on 0 to 1): "
              f"{check['flagged']} of {len(check['cells'])} cells flagged")
        for cell in check["cells"]:
            if cell["design"] is None or cell["design"]["interval"] == "withheld":
                print(f"      {cell['arm']} {cell['dimension']}: {cell['n_labels']} labels, {cell['status']}")
                continue
            flag = f"  FLAGGED: gap {cell['gap']:+.3f}, joint sd {cell['joint_sd']:.3f}" if cell["flagged"] else ""
            if cell["model"] is None:
                flag = f"  {cell['status']}"
            print(f"      {cell['arm']} {cell['dimension']}: design {fmt(cell['design'])} "
                  f"({cell['design']['interval']}, {cell['n_labels']} labels)  model {fmt(cell['model'])}{flag}")
    print_warnings(payload["warnings"])


def run(args: argparse.Namespace) -> int:
    fit, spec, study, warnings = read_inputs(args)
    reference, source = pick_reference(fit, getattr(args, "reference", None), spec.objective)
    check = getattr(args, "check", None)
    plans = list(getattr(args, "labels", None) or [])
    if check == "design" and not plans:
        raise C.ContractError(Path(args.fit).name, "--check design", "no labels plan was given",
                              "name the plans the human readings answer with --labels PLAN", "no_labels")
    warnings += stale_warnings(fit, study, getattr(args, "strict", False))
    seed = int(getattr(args, "seed", 0) or 0)
    files: dict[str, Any] = {"fit": fit.path, "draws": fit.path.with_name(fit["draws"]["file"]), "value": args.value}
    payload: dict = {"experimental": True,
                     "options": {"reference": fit.arms[reference], "seed": seed, "check": check},
                     "reference": fit.arms[reference], "reference_source": source}
    payload.update(spec_record(fit, spec))
    payload.update(report(fit, spec, reference, seed))
    if check == "design":
        from . import designcheck
        for n, plan in enumerate(plans, 1):
            files[f"labels-{n}"] = plan
        files["scores"] = Path(study) / C.SCORES_FILE
        payload["design_check"] = designcheck.check_files(fit, study, plans, seed=seed)
    payload["inputs"] = C.input_digests(files)
    payload["warnings"] = warnings
    if args.json:
        emit_json(REPORT_SCHEMA, payload)
        print_warnings(warnings)
    else:
        _print(payload)
    return EXIT_OK
