"""Build a fit file from known parameters, with no sampling, for the tests of every stage after the fit.

    from loopmath.eval import testing
    built = testing.build_fit(tmp_path / "fit.json")          # a default study: 3 arms, 5 dimensions, 3 judges
    fit, truth = built.fit, built.truth                        # a fitfile.Fit and the true arrays

`build_fit` takes a rubric, a judges file and run rows (the helpers below make small ones), and a `truth`
mapping with any of the arrays of `fitfile.DRAW_SHAPES` without the leading S axis (`mu` is A x D, `bias`
is J x D, and so on). What is not given gets a default. Every draw is the truth plus posterior noise of
sd `posterior_sd`; with `posterior_sd=0` every draw equals the truth, so a later stage's output can be
checked against a closed form. `unit` (D) is the exception: the factor of each dimension, written as it
is in every draw, with its stat as the dimension's `unit.factor` on `binary` and pairs-only dimensions.
It must be 1 on the dimensions the data fix (`fitfile.check_units`). Without it the draws carry no
conversion: the unit draws are 1 and every factor is null.

All data here is synthetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import contracts as C
from . import fitfile as F
from . import quality as Q
from . import units as U

ENGINE = {"name": "testing", "version": "1"}


def example_rubric() -> dict:
    """A rubric with every scale kind: 4 criteria and an `overall` component, all on `final`."""
    return C.validate_rubric({
        "schema": C.SCHEMA_RUBRIC, "id": "example", "version": "1",
        "components": [
            {"id": "numbers_right", "title": "The numbers are right", "scale": {"kind": "binary"}},
            {"id": "clarity", "title": "Clarity", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]}},
            {"id": "depth", "title": "Depth", "scale": {"kind": "linear", "range": [0, 10]}, "review_minutes": 3},
            {"id": "coverage", "title": "Coverage of case points", "scale": {"kind": "items"}},
            {"id": "overall", "title": "Overall", "role": "overall", "scale": {"kind": "linear", "range": [0, 10]},
             "review_minutes": 2},
        ]})


def example_judges() -> dict:
    """Three judges: a person who anchors the scale, a strong model judge and a cheap one from another family."""
    return C.validate_judges({
        "schema": C.SCHEMA_JUDGES, "anchor": ["reviewer"],
        "judges": [
            {"id": "reviewer", "kind": "person"},
            {"id": "strong", "kind": "model", "model": {"id": "model-large", "family": "alpha"}, "effort": "high",
             "prompt": "v1", "blinded": True, "cost_usd_per_call": 0.05},
            {"id": "cheap", "kind": "model", "model": {"id": "model-small", "family": "beta"}, "effort": "low",
             "prompt": "v1", "blinded": True, "cost_usd_per_call": 0.005},
        ]})


def example_runs(arms: Mapping[str, Sequence[dict]] | None = None, n_inputs: int = 12, runs_per_input: int = 1,
                 costs: Mapping[str, float] | None = None) -> list[dict]:
    """Run rows: every arm runs on every input. `arms` maps an arm id to its `steps[]` entries.

    The default is three arms over the steps `draft` and `final`: `baseline`, `strong-draft` (one step
    swapped) and `strong-all` (both swapped)."""
    small = {"model": {"id": "model-small", "family": "beta"}, "effort": "low"}
    large = {"model": {"id": "model-large", "family": "alpha"}, "effort": "high"}
    if arms is None:
        arms = {"baseline": [{"step": "draft", **small}, {"step": "final", **small}],
                "strong-draft": [{"step": "draft", **large}, {"step": "final", **small}],
                "strong-all": [{"step": "draft", **large}, {"step": "final", **large}]}
        costs = costs or {"baseline": 0.10, "strong-draft": 0.40, "strong-all": 0.90}
    rows = []
    for i in range(n_inputs):
        for arm, steps in arms.items():
            for k in range(runs_per_input):
                run = f"{arm}-i{i + 1:03d}" + (f"-r{k + 1}" if runs_per_input > 1 else "")
                row = {"schema": C.SCHEMA_RUNS, "run": run, "input": f"i{i + 1:03d}", "arm": arm,
                       "steps": [dict(step) for step in steps], "simulated": True,
                       "refs": {"input": f"inputs/i{i + 1:03d}", "output": f"outputs/{run}"}}
                if costs and arm in costs:
                    row["cost_usd"] = costs[arm]
                rows.append(C.validate_run(row))
    return rows


def default_truth(index: dict, dims: Sequence[dict], judges: dict) -> dict[str, np.ndarray]:
    """True parameters for an index: every array of `DRAW_SHAPES` but `cuts`, without the S axis.

    Arm means rise from the first arm to the last. Unit kinds (`linear`, `fraction`, `levels`) sit near
    0.6 with sds of 0.08; probit kinds (`binary`, `items`) sit near 0.5 with sds of 0.4. Anchor judges
    have no bias; model judges have a small bias and more noise. A person has no family effects (NaN).
    No dimension has pairs, so `sigma_pair`, `position` and `tie_width` are NaN."""
    A, D, J, N = len(index["arms"]), len(dims), len(index["judges"]), len(index["families"])
    probit = np.array([dim["scale"]["kind"] in Q.PROBIT_KINDS for dim in dims])
    step = np.linspace(0.0, 1.0, A)[:, None] if A > 1 else np.zeros((1, 1))
    truth = {"mu": np.where(probit, 0.2 + 0.8 * step, 0.5 + 0.2 * step) + 0.01 * np.arange(D),
             "sd_input": np.where(probit, 0.4, 0.08), "sd_run": np.where(probit, 0.4, 0.08),
             "corr_input": np.eye(D), "corr_run": np.eye(D)}
    bias, tau, sigma = np.zeros((J, D)), np.zeros((J, D)), np.zeros((J, D))
    family = np.zeros((J, N, D))
    for j, row in enumerate(judges["judges"]):
        person = row["kind"] == "person"
        scale = 0.5 if person else 1.0 + 0.5 * j
        bias[j] = 0.0 if row["id"] in judges["anchor"] else np.where(probit, 0.2, 0.04) * (1 if j % 2 else -1)
        tau[j] = np.where(probit, 0.3, 0.05) * scale
        sigma[j] = np.where(probit, 1.0, 0.05) * (1.0 if person else scale)
        if row["kind"] != "model":
            family[j] = np.nan
    truth.update({"bias": bias, "tau": tau, "sigma": sigma, "halo_target": np.full(J, 0.2),
                  "halo_call": np.full(J, 0.1), "family": family, "sd_judge_input": np.zeros((J, D)),
                  "sigma_pair": np.full((J, D), np.nan), "position": np.full(J, np.nan),
                  "tie_width": np.full(J, np.nan)})
    return truth


LOCATION = ("mu", "bias", "family", "position")              # truth + noise
POSITIVE = ("sd_input", "sd_run", "tau", "sigma", "sd_judge_input", "sigma_pair", "tie_width")  # truth x exp(noise)
SHARES = ("halo_target", "halo_call")                         # truth + noise, kept in [0, 1]


@dataclass
class Built:
    """What `build_fit` returns: the path, the fit as `fitfile.read_fit` gives it, and the true arrays.

    `truth` holds the arrays of `default_truth` plus `q_run` (R x D), the true hidden quality of each run."""

    path: Path
    fit: F.Fit
    truth: dict[str, np.ndarray]


def build_fit(path: str | os.PathLike, *, rubric: dict | None = None, judges: dict | None = None,
              runs: Sequence[dict] | None = None, truth: Mapping[str, Any] | None = None, n_draws: int = 200,
              posterior_sd: float = 0.02, seed: int = 0, converged: bool = True, calls: int = 2,
              person_share: float = 0.5, relative: Sequence[str] = (), terms: Sequence[dict] | None = None,
              inputs: dict | None = None, options: dict | None = None) -> Built:
    """Write `fit.json` and its draws from known parameters and return them.

    - `rubric`, `judges`, `runs`: normalised objects; the defaults are the `example_*` helpers.
    - `truth`: overrides for `default_truth`, by array name. `q_run` may be given too; else it is drawn
      from the model, `q = mu_arm + v_input + u_run`.
    - `n_draws`, `posterior_sd`, `seed`: S, the posterior noise and the generator's seed. The same
      arguments give the same two files, byte for byte.
    - `calls`: calls per model judge on every target. `person_share`: the share of targets each person
      scored. Both only fill the counts and the per-target records (`targets[].dims[].judges`,
      `unpooled`), which `labels next` reads.
    - `relative`: ids of dimensions to mark as pairs-only.
    - `terms`: the identification report. By default `sigma`, `tau`, the family effects of model judges,
      the anchor and the split between levels are `identified`; `halo_split` is `prior_only` (one call
      mode only); `judge_input` is `off`; family effects of a person, the reference contrast without a
      shown reference and the pair terms are `absent` (the default study has no pairs).
    - `converged=False` writes a fit that every later stage must refuse without `--allow-unconverged`."""
    rubric = rubric or example_rubric()
    judges = judges or example_judges()
    runs = list(runs) if runs is not None else example_runs()
    rng = np.random.default_rng(seed)
    index = F.build_index(rubric, judges, runs)
    dims = C.dimensions(rubric)
    A, D, J, R = len(index["arms"]), len(dims), len(index["judges"]), len(runs)
    true = default_truth(index, dims, judges)
    for name, value in (truth or {}).items():
        if name not in F.DRAW_SHAPES:
            raise KeyError(f"unknown truth array {name!r}; use the names of fitfile.DRAW_SHAPES")
        true[name] = np.asarray(value, dtype=float)
    for name in ("sd_input", "sd_run"):
        true[name] = np.broadcast_to(true[name], (D,)).astype(float)
    true["mu"] = np.broadcast_to(true["mu"], (A, D)).astype(float)
    if "q_run" not in true:
        v = rng.standard_normal((len(index["inputs"]), D)) @ Q._root(Q.covariance(true["sd_input"],
                                                                                    true["corr_input"])).T
        u = rng.standard_normal((R, D)) @ Q._root(Q.covariance(true["sd_run"], true["corr_run"])).T
        true["q_run"] = true["mu"][index["run_arm"]] + v[index["run_input"]] + u

    draws: dict[str, np.ndarray] = {}
    for name in sorted(true):
        if name == "unit":                                     # a factor, not an estimate: no posterior noise
            draws[name] = np.broadcast_to(true[name], (n_draws, D)).astype(float).copy()
            continue
        base = true[name]
        noise = posterior_sd * rng.standard_normal((n_draws,) + base.shape)
        if name in POSITIVE:
            draws[name] = base[None] * np.exp(noise)
        elif name in SHARES:
            draws[name] = np.clip(base[None] + noise, 0.0, 1.0)
        elif name in ("corr_input", "corr_run"):
            draws[name] = np.broadcast_to(base, (n_draws,) + base.shape).copy()
        else:
            draws[name] = base[None] + noise
    if len(judges["anchor"]) == 1:  # a single anchor defines the scale: its bias is 0 in every draw
        draws["bias"][:, index["judges"].index(judges["anchor"][0]), :] = 0.0

    # Which judge read which target, and what a plain mean of its readings would be.
    readings: dict[tuple[str, str], dict] = {}
    judge_counts: dict[tuple[str, str], dict] = {}
    n_readings = 0
    for r, run in enumerate(runs):
        for d, dim in enumerate(dims):
            seen, values = {}, []
            for j, row in enumerate(judges["judges"]):
                if row["components"] is not None and dim["component"] not in row["components"]:
                    continue
                n = calls if row["kind"] != "person" else int(rng.random() < person_share)
                if not n:
                    continue
                seen[row["id"]] = n
                n_readings += n
                tally = judge_counts.setdefault((row["id"], dim["id"]), {"n_targets": 0, "n_calls": 0, "n_pairs": 0})
                tally["n_targets"] += 1
                tally["n_calls"] += n
                values.append(true["q_run"][r, d] + true["bias"][j, d] + rng.standard_normal() * np.hypot(
                    true["tau"][j, d], true["sigma"][j, d] / np.sqrt(n)))
            entry: dict = {"judges": seen}
            if values:
                entry["unpooled"] = {"mean": float(np.mean(values)), "n": int(sum(seen.values())),
                                     "sd": float(np.std(values) / np.sqrt(len(values))) if len(values) > 1 else None}
            readings[(run["run"], dim["id"])] = entry

    if terms is None:
        terms = []
        for row in judges["judges"]:
            for dim in dims:
                for name in ("sigma", "tau"):
                    terms.append(F.term(name, "identified", judge=row["id"], dimension=dim["id"], count=calls))
            if row["kind"] == "model":
                terms.append(F.term("family", "identified", judge=row["id"], count=len(index["families"])))
            else:
                terms.append(F.term("family", "absent", judge=row["id"], reason="not a model judge"))
            for name in ("position", "sigma_pair", "tie_width"):
                terms.append(F.term(name, "absent", judge=row["id"], reason="the study has no pairs"))
            terms.append(F.term("halo_split", "prior_only", judge=row["id"]))
            terms.append(F.term("judge_input", "off", judge=row["id"]))
            if row["reference"]["shown"]:
                terms.append(F.term("reference_family", "identified", judge=row["id"]))
            else:
                terms.append(F.term("reference_family", "absent", judge=row["id"], reason="no reference is shown"))
        terms.append(F.term("levels_split", "identified", count=len(index["arms"])))
        for dim in dims:
            terms.append(F.term("anchor_bias", "identified", dimension=dim["id"], count=R))
            if dim["scale"]["kind"] == "binary":
                terms.append(F.term("binary_status", "identified", dimension=dim["id"], count=R))
    anchored = [row["id"] for row in judges["judges"] if row["id"] in judges["anchor"]]
    anchor = {"kind": "person" if anchored else "judges_mean", "judges": anchored,
              "min_anchor_targets": judges["min_anchor_targets"],
              "dimensions": [{"dimension": dim["id"], "state": "relative" if dim["id"] in relative else "anchored",
                              "n_targets": judge_counts.get((anchored[0], dim["id"]), {}).get("n_targets", 0)
                              if anchored else 0} for dim in dims]}
    diagnostics = {"converged": bool(converged), "rhat_max": 1.0 if converged else 1.2, "ess_min": float(n_draws),
                   "divergences": 0, "run_seconds": 0.0, "checks": []}
    meta = F.assemble(rubric=rubric, judges=judges, runs=runs, draws=draws, terms=terms, anchor=anchor,
                      diagnostics=diagnostics, inputs=inputs, engine=ENGINE, seed=seed,
                      options=options or {"n_draws": n_draws, "posterior_sd": posterior_sd},
                      counts={"readings": n_readings, "readings_used": n_readings}, judge_counts=judge_counts,
                      readings=readings, relative=relative,
                      units=U.records(dims, relative, [anchored or None] * D, draws["unit"]) if "unit" in draws
                      else None)
    path = F.write_fit(path, meta, draws)
    return Built(path, F.read_fit(path, allow_unconverged=True), true)


def write_study_files(folder: str | os.PathLike, rubric: dict | None = None, judges: dict | None = None,
                      runs: Sequence[dict] | None = None) -> dict[str, Path]:
    """Write `rubric.json`, `judges.json` and `runs.jsonl` of the example study into a folder: role to path."""
    folder = Path(folder)
    return {"rubric": C.write_document(folder / "rubric.json", rubric or example_rubric()),
            "judges": C.write_document(folder / "judges.json", judges or example_judges()),
            "runs": C.write_lines(folder / C.RUNS_FILE, C.SCHEMA_RUNS, runs if runs is not None else example_runs())}


def example_value(gated: bool = True) -> dict:
    """A value file for `example_rubric`: weights on three criteria, and with `gated` two gates."""
    gates = [{"component": "numbers_right"}, {"component": "clarity", "at_least": "ok"}] if gated else []
    return C.validate_value({"schema": C.SCHEMA_VALUE, "weights": {"clarity": 1.0, "depth": 2.0, "coverage": 1.0},
                             "gates": gates, "gated_value": 0.0, "objective": {"kind": "best_value"}})
