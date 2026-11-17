"""The fit file (`loopmath.eval-fit/1`): `fit.json` plus its draws, `fit.draws.npz`.

The fit file is the hub. `eval judges`, `rubric check --fit`, `value`, `steps` and `labels next` read
only it and the small user files, so it carries everything they need: the rubric's dimensions, the
judges as declared, the arms with their step settings and mean cost, and every target with its counts.

`fit.json` members
------------------
`schema`       `loopmath.eval-fit/1`.
`inputs`       per role (`runs`, `scores`, `pairs`, `items`, `rubric`, `judges`): `{"file", "sha256"}`.
`options`      the options the fit ran with.
`engine`       `{"name", "version"}`. `testing.py` writes the name `testing`.
`seed`         a whole number.
`index`        the ordered ids behind every array axis: `arms` (A), `dimensions` (D; each `{"id", "step",
               "component", "kind", "cuts"}`, where `cuts` is the list of fixed cut points of a `levels`
               dimension, else null), `judges` (J), `families` (F), `inputs` (I), `runs` (R), and per run
               `run_arm` and `run_input`, positions in `arms` and `inputs`.
`dimensions`   per dimension, the record of `contracts.dimensions` (`id`, `step`, `component`, `title`,
               `role`, `scale`, `better`, `review_minutes`, `compare_minutes`, `human_only`) plus
               `relative` (true for a pairs-only dimension: no gate can use it) and `cuts` (`fixed` or
               `free`, null when the scale is not `levels`).
`counts`       `runs`, `inputs`, `arms`, `targets`, `readings`, `readings_used`, `pairs`, `pairs_used`.
`terms`        the identification report, one entry per term, judge and dimension:
               `{"term", "judge", "dimension", "state", "count", "needs"}`; `judge` and `dimension` are
               null where the term has none. States are in `TERM_STATES`. A NaN in the draws is explained
               by an entry in state `absent` with a `reason`.
`diagnostics`  `converged` (true or false), `rhat_max`, `ess_min`, `divergences`, `run_seconds`, `checks`.
`anchor`       `kind` (`person` or `judges_mean`), `judges`, `min_anchor_targets`, and `dimensions`: per
               dimension `{"dimension", "state", "n_targets"}` with state `anchored`, `weak` or `relative`.
`arms`         per arm: `id`, `n_runs`, `n_inputs`, `cost_usd` (the mean run cost or null), `steps` (the
               step settings), `mixed` (true for a `mixed_arm`), and `dims`: per dimension `{"dimension",
               "reported", "own", "fitting"}`, each a stat (below); `own` is null where the scale has none.
`levels`       `input` and `run`: each `{"sd": [stat per dimension], "corr": {"mean", "lo", "hi"}}`, D x D.
`judges`       per judge: the declaration of the judges file (`id`, `kind`, `model`, `effort`, `prompt`,
               `mode`, `reference`, `blinded`, `cost_usd_per_call`, `components`), `anchor` (true or false),
               `halo_target`, `halo_call`, `position`, `tie_width` (stats), `dims`: per dimension
               `{"dimension", "bias", "tau", "sigma", "sd_judge_input", "sigma_pair", "n_targets",
               "n_calls", "n_pairs"}`, and `family`: per family and dimension `{"family", "dimension",
               "effect"}`.
`targets`      per target: `run`, `step`, `input`, `arm`, `refs`, and `dims`: per dimension of that step
               `{"dimension", "pooled": {"mean", "sd"}, "unpooled": {"mean", "sd", "n"} or null,
               "judges": {judge id: calls}}`.
`notes`        every reading and pair not used: `{"kind", "id", "reason"}`.
`draws`        `{"file", "sha256", "n", "arrays": {name: shape}}`.

A stat is `{"mean", "sd", "lo", "hi"}` with the 90% interval from the 5th to the 95th percentile of the
draws, or null when the draws hold NaN there.

The draws
---------
`DRAW_SHAPES` is the contract every later stage reads. S draws, A arms, D dimensions, R runs, J judges,
F families, C = Lmax - 1 cut points. Every array is float64. `cuts` is optional.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import contracts as C
from . import quality as Q
from .contracts import ContractError

DRAWS_SUFFIX = ".draws.npz"
DRAW_SHAPES = {
    "mu": "SAD", "sd_input": "SD", "sd_run": "SD", "corr_input": "SDD", "corr_run": "SDD", "q_run": "SRD",
    "bias": "SJD", "tau": "SJD", "sigma": "SJD", "halo_target": "SJ", "halo_call": "SJ", "family": "SJFD",
    "sd_judge_input": "SJD", "sigma_pair": "SJD", "position": "SJ", "tie_width": "SJ", "cuts": "SDC",
}
OPTIONAL_DRAWS = ("cuts",)
MEMBERS = ("schema", "inputs", "options", "engine", "seed", "index", "dimensions", "counts", "terms",
           "diagnostics", "anchor", "arms", "levels", "judges", "targets", "notes", "draws")
TERM_STATES = ("identified", "weak", "prior_only", "confounded", "off", "absent")
MEASURED_STATES = ("identified", "weak")
ANCHOR_STATES = ("anchored", "weak", "relative")
TERMS = ("sigma", "tau", "levels_split", "anchor_bias", "family", "reference_family", "halo_split",
         "judge_input", "binary_status", "position", "sigma_pair", "tie_width")
TERM_NEEDS = {
    "sigma": "2 or more calls by the judge on the same target and component",
    "tau": "repeats by the judge, and 2 or more judges on the same targets",
    "levels_split": "2 or more runs per input, of one arm or of paired arms",
    "anchor_bias": "enough targets per dimension scored by an anchor judge",
    "family": "judges from 2 or more families, each scoring outputs from 2 or more families",
    "reference_family": "a judge with a reference whose author's family differs from the judge's own",
    "halo_split": "panel and per_component calls by the same model, effort and prompt, or repeats of panel calls",
    "judge_input": "2 or more runs per input scored by the same judge, on 10 or more inputs",
    "binary_status": "repeats by the anchor, or 2 or more judges on the same targets",
    "position": "pairs judged in both orders, or shown_first balanced across pairs",
    "sigma_pair": "repeats of pairs by the judge",
    "tie_width": "repeats of pairs by the judge, and ties present",
}


def draws_path(fit_path: str | os.PathLike) -> Path:
    """`study/fit.json` keeps its draws in `study/fit.draws.npz`."""
    fit_path = Path(fit_path)
    return fit_path.with_name(fit_path.stem + DRAWS_SUFFIX)


# ---------------------------------------------------------------------------------------------------
# Building the members

def build_index(rubric: dict, judges: dict, runs: Sequence[dict], components: Sequence[str] | None = None) -> dict:
    """The `index` member, from a normalised rubric, judges file and run rows.

    Orders: dimensions as `contracts.dimensions`; judges as in the judges file; runs as given; arms and
    inputs by first appearance in `runs`; families sorted, over every step of every run."""
    arms, inputs, families = [], [], set()
    for run in runs:
        if run["arm"] not in arms:
            arms.append(run["arm"])
        if run["input"] not in inputs:
            inputs.append(run["input"])
        families.update(step["model"]["family"] for step in run["steps"])
    dims = []
    for dim in C.dimensions(rubric, components):
        scale = dim["scale"]
        cuts = [float(x) for x in Q.fixed_cuts(len(scale["levels"]))] if scale["kind"] == "levels" else None
        dims.append({"id": dim["id"], "step": dim["step"], "component": dim["component"], "kind": scale["kind"],
                     "cuts": cuts})
    return {"arms": arms, "dimensions": dims, "judges": [row["id"] for row in judges["judges"]],
            "families": sorted(families), "inputs": inputs, "runs": [run["run"] for run in runs],
            "run_arm": [arms.index(run["arm"]) for run in runs],
            "run_input": [inputs.index(run["input"]) for run in runs]}


def target_family(run: dict, step: str) -> str:
    """The model family that wrote a target: the family at that step; for `final`, of the last step."""
    for entry in run["steps"]:
        if entry["step"] == step:
            return entry["model"]["family"]
    return run["steps"][-1]["model"]["family"]


def setting(entry: dict) -> tuple:
    """A step's setting: the triple (model id, effort, variant)."""
    return (entry["model"]["id"], entry.get("effort"), entry.get("variant"))


def stat(x: Any) -> dict | None:
    """Mean, sd and the 90% interval of draws along axis 0 of a 1-d array. None when any draw is NaN."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.isnan(x).any():
        return None
    lo, hi = np.percentile(x, [5.0, 95.0])
    return {"mean": float(x.mean()), "sd": float(x.std()), "lo": float(lo), "hi": float(hi)}


def _corr_stat(corr: np.ndarray) -> dict:
    lo, hi = np.percentile(corr, [5.0, 95.0], axis=0)
    return {"mean": corr.mean(axis=0).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def arm_block(runs: Sequence[dict], index: dict, dims: Sequence[dict], draws: Mapping[str, np.ndarray]) -> list:
    """The `arms` member: step settings, counts, mean cost and the reported quality per dimension."""
    total_sd = np.sqrt(np.asarray(draws["sd_input"]) ** 2 + np.asarray(draws["sd_run"]) ** 2)
    out = []
    for a, arm in enumerate(index["arms"]):
        mine = [run for run in runs if run["arm"] == arm]
        costs = [run["cost_usd"] for run in mine if run.get("cost_usd") is not None]
        settings = {tuple((entry["step"],) + setting(entry) for entry in run["steps"]) for run in mine}
        rows = []
        for d, dim in enumerate(dims):
            mu = np.asarray(draws["mu"])[:, a, d]
            m = Q.arm_reported(mu, total_sd[:, d], dim["scale"]["kind"])
            own = Q.own_scale(m, dim)
            rows.append({"dimension": dim["id"], "reported": stat(m), "own": None if own is None else stat(own),
                         "fitting": stat(mu)})
        out.append({"id": arm, "n_runs": len(mine), "n_inputs": len({run["input"] for run in mine}),
                    "cost_usd": float(np.mean(costs)) if costs else None,
                    "steps": [dict(entry) for entry in mine[0]["steps"]], "mixed": len(settings) > 1, "dims": rows})
    return out


def level_block(draws: Mapping[str, np.ndarray]) -> dict:
    """The `levels` member: per level (input, run), the sd per dimension and the correlation matrix."""
    out = {}
    for level in ("input", "run"):
        sd = np.asarray(draws[f"sd_{level}"])
        out[level] = {"sd": [stat(sd[:, d]) for d in range(sd.shape[1])],
                      "corr": _corr_stat(np.asarray(draws[f"corr_{level}"]))}
    return out


def judge_block(judges: dict, index: dict, dims: Sequence[dict], draws: Mapping[str, np.ndarray],
                counts: Mapping[tuple[str, str], Mapping[str, int]] | None = None) -> list:
    """The `judges` member. `counts[(judge id, dimension id)]` gives `n_targets`, `n_calls`, `n_pairs`."""
    counts = counts or {}
    out = []
    for j, declared in enumerate(judges["judges"]):
        row = {k: v for k, v in declared.items() if k != "ext"}
        row["anchor"] = declared["id"] in judges["anchor"]
        for name in ("halo_target", "halo_call", "position", "tie_width"):
            row[name] = stat(np.asarray(draws[name])[:, j])
        row["dims"] = []
        for d, dim in enumerate(dims):
            n = counts.get((declared["id"], dim["id"]), {})
            one = {"dimension": dim["id"]}
            for name in ("bias", "tau", "sigma", "sd_judge_input", "sigma_pair"):
                one[name] = stat(np.asarray(draws[name])[:, j, d])
            one.update({"n_targets": int(n.get("n_targets", 0)), "n_calls": int(n.get("n_calls", 0)),
                        "n_pairs": int(n.get("n_pairs", 0))})
            row["dims"].append(one)
        row["family"] = [{"family": family, "dimension": dim["id"],
                          "effect": stat(np.asarray(draws["family"])[:, j, f, d])}
                         for f, family in enumerate(index["families"]) for d, dim in enumerate(dims)]
        out.append(row)
    return out


def target_block(runs: Sequence[dict], dims: Sequence[dict], draws: Mapping[str, np.ndarray],
                 readings: Mapping[tuple[str, str], dict] | None = None) -> list:
    """The `targets` member: one entry per (run, step) with a dimension, pooled quality from `q_run`.

    `readings[(run id, dimension id)]` may give `unpooled` (`mean`, `sd`, `n`) and `judges` (calls per judge)."""
    readings = readings or {}
    q = np.asarray(draws["q_run"])
    steps = []
    for dim in dims:
        if dim["step"] not in steps:
            steps.append(dim["step"])
    out = []
    for r, run in enumerate(runs):
        for step in steps:
            rows = []
            for d, dim in enumerate(dims):
                if dim["step"] != step:
                    continue
                seen = readings.get((run["run"], dim["id"]), {})
                x = q[:, r, d]
                pooled = None if np.isnan(x).any() else {"mean": float(x.mean()), "sd": float(x.std())}
                rows.append({"dimension": dim["id"], "pooled": pooled, "unpooled": seen.get("unpooled"),
                             "judges": dict(seen.get("judges", {}))})
            out.append({"run": run["run"], "step": step, "input": run["input"], "arm": run["arm"],
                        "refs": dict(run.get("refs", {})), "dims": rows})
    return out


def term(name: str, state: str, *, judge: str | None = None, dimension: str | None = None, count: int = 0,
         reason: str | None = None) -> dict:
    """One entry of `terms`."""
    out = {"term": name, "judge": judge, "dimension": dimension, "state": state, "count": int(count),
           "needs": TERM_NEEDS[name]}
    if reason is not None:
        out["reason"] = reason
    return out


def assemble(*, rubric: dict, judges: dict, runs: Sequence[dict], draws: Mapping[str, np.ndarray],
             terms: Sequence[dict], anchor: dict, diagnostics: dict, inputs: dict | None = None,
             options: dict | None = None, engine: dict | None = None, seed: int = 0,
             components: Sequence[str] | None = None, counts: dict | None = None,
             judge_counts: Mapping[tuple[str, str], Mapping[str, int]] | None = None,
             readings: Mapping[tuple[str, str], dict] | None = None, notes: Sequence[dict] = (),
             relative: Sequence[str] = (), free_cuts: Sequence[str] = ()) -> dict:
    """Build every member of `fit.json` but `draws` (which `write_fit` adds) from the pieces of a fit.

    `eval fit` and `testing.py` both go through here, so the two write the same shape. `relative` and
    `free_cuts` are lists of dimension ids."""
    index = build_index(rubric, judges, runs, components)
    dims = []
    for dim in C.dimensions(rubric, components):
        levels = dim["scale"]["kind"] == "levels"
        dims.append({**dim, "relative": dim["id"] in relative,
                     "cuts": ("free" if dim["id"] in free_cuts else "fixed") if levels else None})
    targets = target_block(runs, dims, draws, readings)
    base_counts = {"runs": len(runs), "inputs": len(index["inputs"]), "arms": len(index["arms"]),
                   "targets": len(targets), "readings": 0, "readings_used": 0, "pairs": 0, "pairs_used": 0}
    base_counts.update(counts or {})
    return {"schema": C.SCHEMA_FIT, "inputs": inputs or {}, "options": options or {},
            "engine": engine or {"name": "unknown", "version": ""}, "seed": int(seed), "index": index,
            "dimensions": dims, "counts": base_counts, "terms": list(terms), "diagnostics": dict(diagnostics),
            "anchor": dict(anchor), "arms": arm_block(runs, index, dims, draws), "levels": level_block(draws),
            "judges": judge_block(judges, index, dims, draws, judge_counts), "targets": targets,
            "notes": list(notes)}


# ---------------------------------------------------------------------------------------------------
# Validation

def _sizes(index: dict, draws: Mapping[str, np.ndarray]) -> dict[str, int]:
    sizes = {"A": len(index["arms"]), "D": len(index["dimensions"]), "J": len(index["judges"]),
             "F": len(index["families"]), "R": len(index["runs"])}
    if "mu" in draws:
        sizes["S"] = int(np.asarray(draws["mu"]).shape[0])
    if "cuts" in draws:
        sizes["C"] = int(np.asarray(draws["cuts"]).shape[-1])
    return sizes


def check_draws(index: dict, draws: Mapping[str, np.ndarray], file: str = "fit.draws.npz") -> dict[str, int]:
    """Refuse draws that break `DRAW_SHAPES` for this index. Returns the axis sizes."""
    for name in DRAW_SHAPES:
        if name not in draws and name not in OPTIONAL_DRAWS:
            raise ContractError(file, name, "the array is missing", "write every array of fitfile.DRAW_SHAPES",
                                "draws_missing")
    for name in draws:
        if name not in DRAW_SHAPES:
            raise ContractError(file, name, "unknown array", "write only the arrays of fitfile.DRAW_SHAPES",
                                "draws_unknown")
    sizes = _sizes(index, draws)
    if sizes["S"] < 1:
        raise ContractError(file, "mu", "no draws", "write at least one draw", "draws_shape")
    for name, array in draws.items():
        want = tuple(sizes[axis] for axis in DRAW_SHAPES[name])
        got = tuple(np.asarray(array).shape)
        if got != want:
            axes = ", ".join(DRAW_SHAPES[name])
            raise ContractError(file, name, f"shape {got} is not ({axes}) = {want}",
                                "match the index of fit.json", "draws_shape")
        if np.asarray(array).dtype != np.float64:
            raise ContractError(file, name, f"dtype {np.asarray(array).dtype} is not float64",
                                "write float64 arrays", "draws_dtype")
    return sizes


INPUT_ROLES = ("runs", "scores", "pairs", "items", "rubric", "judges")
COUNTS = ("runs", "inputs", "arms", "targets", "readings", "readings_used", "pairs", "pairs_used")
STAT = ("mean", "sd", "lo", "hi")
DIMENSION = ("id", "step", "component", "title", "role", "scale", "better", "review_minutes", "compare_minutes",
             "human_only", "relative", "cuts")
JUDGE_STATS = ("halo_target", "halo_call", "position", "tie_width")
JUDGE_DIM_STATS = ("bias", "tau", "sigma", "sd_judge_input", "sigma_pair")
JUDGE_DIM_COUNTS = ("n_targets", "n_calls", "n_pairs")
JUDGE = ("id", "kind", "model", "effort", "prompt", "mode", "reference", "blinded", "cost_usd_per_call",
         "components", "anchor", *JUDGE_STATS, "dims", "family")


def _record(c: C._Check, value: Any, where: str, members: Sequence[str], nullable: Sequence[str] = (),
            optional: Sequence[str] = ()) -> dict:
    """An object with only `members`. Each has to be there; a `nullable` one may be null; an `optional`
    one may be left out."""
    c.obj(value, where, members, [name for name in members if name not in nullable and name not in optional])
    for name in nullable:
        if name not in value:
            raise c.fail(where, f'missing member "{name}"', "add it", "missing_member")
    return value


def _number(c: C._Check, value: Any, where: str, **bounds: float) -> float:
    try:
        return c.number(value, where, **bounds)
    except OverflowError:  # a whole number too large for a float
        raise c.fail(where, "expected a finite number, got a number too large to read", "write a number here",
                     "type") from None


def _text(c: C._Check, value: Any, where: str) -> str:
    """A string that may be empty."""
    if not isinstance(value, str):
        raise c.fail(where, f"expected a string, got {C._kind(value)}", "write a string here", "type")
    return value


def _stat(c: C._Check, value: Any, where: str) -> None:
    """A stat: null, or the mean, sd and 90% interval."""
    if value is None:
        return
    _record(c, value, where, STAT)
    for name in STAT:
        _number(c, value[name], f"{where}.{name}", **({"low": 0} if name == "sd" else {}))


def _same(c: C._Check, got: Any, want: Any, where: str, what: str, fix: str) -> None:
    if got != want:
        shown = f'"{got}"' if isinstance(got, str) else C._num(got) if isinstance(got, (int, float)) else C._kind(got)
        wanted = f'"{want}"' if isinstance(want, str) else str(want)
        raise c.fail(where, f"{shown} is not {what} {wanted}", fix, "bad_index")


def _matrix(c: C._Check, value: Any, where: str, size: int) -> None:
    rows = c.array(value, where)
    if len(rows) != size or any(not isinstance(row, list) or len(row) != size for row in rows):
        raise c.fail(where, f"it is not a {size} by {size} matrix", "write one row and one column per dimension",
                     "bad_index")
    for n, row in enumerate(rows):
        for m, cell in enumerate(row):
            _number(c, cell, f"{where}[{n}][{m}]")


def _check_dimensions(c: C._Check, dims: list, index: dict) -> None:
    for n, (dim, known) in enumerate(zip(dims, index["dimensions"])):
        w = f"dimensions[{n}]"
        _record(c, dim, w, DIMENSION, nullable=("cuts",))
        for name in ("id", "step", "component", "title"):
            c.string(dim[name], f"{w}.{name}")
        c.choice(dim["role"], f"{w}.role", C.ROLES)
        scale = C._scale(c, dim["scale"], f"{w}.scale")
        c.choice(dim["better"], f"{w}.better", C.BETTER)
        for name in ("review_minutes", "compare_minutes"):
            _number(c, dim[name], f"{w}.{name}", above=0)
        for name in ("human_only", "relative"):
            c.boolean(dim[name], f"{w}.{name}")
        for name in ("step", "component"):
            _same(c, dim[name], known[name], f"{w}.{name}", f"the {name} of index.dimensions[{n}],",
                  "write the dimension as the index has it")
        _same(c, scale["kind"], known["kind"], f"{w}.scale.kind", f"the kind of index.dimensions[{n}],",
              "write the dimension as the index has it")
        if scale["kind"] == "levels":
            c.choice(dim["cuts"], f"{w}.cuts", ("fixed", "free"))
            cuts = known.get("cuts")
            if cuts is not None and len(cuts) != len(scale["levels"]) - 1:
                raise c.fail(f"index.dimensions[{n}].cuts", f"{len(cuts)} cuts for {len(scale['levels'])} levels",
                             "give one cut between each two levels", "bad_index")
        elif dim["cuts"] is not None:
            raise c.fail(f"{w}.cuts", f'a "{scale["kind"]}" scale has no cuts', "write null", "scale_member")


def _check_arms(c: C._Check, arms: list, index: dict) -> None:
    ids = [dim["id"] for dim in index["dimensions"]]
    for a, arm in enumerate(arms):
        w = f"arms[{a}]"
        _record(c, arm, w, ("id", "n_runs", "n_inputs", "cost_usd", "steps", "mixed", "dims"), nullable=("cost_usd",))
        mine = [r for r, position in enumerate(index["run_arm"]) if position == a]
        _same(c, c.integer(arm["n_runs"], f"{w}.n_runs", low=0), len(mine), f"{w}.n_runs",
              "the count of its runs in index.run_arm,", "count the runs of the arm")
        _same(c, c.integer(arm["n_inputs"], f"{w}.n_inputs", low=0), len({index["run_input"][r] for r in mine}),
              f"{w}.n_inputs", "the count of its inputs in index.run_input,", "count the inputs of the arm")
        if arm["cost_usd"] is not None:
            _number(c, arm["cost_usd"], f"{w}.cost_usd", low=0)
        for n, step in enumerate(c.array(arm["steps"], f"{w}.steps", non_empty=True)):
            v = f"{w}.steps[{n}]"
            c.obj(step, v, {"step", "model", "effort", "variant"}, {"step", "model"})
            c.ident(step["step"], f"{v}.step")
            C._model(c, step["model"], f"{v}.model")
            for name in ("effort", "variant"):
                if step.get(name) is not None:
                    c.string(step[name], f"{v}.{name}")
        c.boolean(arm["mixed"], f"{w}.mixed")
        rows = c.array(arm["dims"], f"{w}.dims")
        if len(rows) != len(ids):
            raise c.fail(f"{w}.dims", f"{len(rows)} entries for {len(ids)} dimensions",
                         "write one entry per dimension, in the order of index.dimensions", "bad_index")
        for d, row in enumerate(rows):
            v = f"{w}.dims[{d}]"
            _record(c, row, v, ("dimension", "reported", "own", "fitting"), nullable=("reported", "own", "fitting"))
            _same(c, row["dimension"], ids[d], f"{v}.dimension", "the dimension at this place,",
                  "write one entry per dimension, in the order of index.dimensions")
            for name in ("reported", "own", "fitting"):
                _stat(c, row[name], f"{v}.{name}")


def _check_levels(c: C._Check, levels: Any, index: dict) -> None:
    size = len(index["dimensions"])
    _record(c, levels, "levels", ("input", "run"))
    for level in ("input", "run"):
        w = f"levels.{level}"
        _record(c, levels[level], w, ("sd", "corr"))
        sds = c.array(levels[level]["sd"], f"{w}.sd")
        if len(sds) != size:
            raise c.fail(f"{w}.sd", f"{len(sds)} entries for {size} dimensions",
                         "write one entry per dimension, in the order of index.dimensions", "bad_index")
        for d, one in enumerate(sds):
            _stat(c, one, f"{w}.sd[{d}]")
        corr = _record(c, levels[level]["corr"], f"{w}.corr", ("mean", "lo", "hi"))
        for name in ("mean", "lo", "hi"):
            _matrix(c, corr[name], f"{w}.corr.{name}", size)


def _check_judges(c: C._Check, judges: list, index: dict) -> None:
    ids = [dim["id"] for dim in index["dimensions"]]
    cells = [(family, dim) for family in index["families"] for dim in ids]
    for j, judge in enumerate(judges):
        w = f"judges[{j}]"
        _record(c, judge, w, JUDGE, nullable=("components", *JUDGE_STATS),
                optional=("model", "effort", "prompt", "cost_usd_per_call"))
        kind = c.choice(judge["kind"], f"{w}.kind", C.JUDGE_KINDS)
        if kind == "model":
            if judge.get("model") is None:
                raise c.fail(w, 'a judge of kind "model" has no model', "add model with id and family",
                             "missing_member")
            C._model(c, judge["model"], f"{w}.model", provider=True)
        elif judge.get("model") is not None:
            raise c.fail(f"{w}.model", f'a judge of kind "{kind}" has a model', 'remove it, or set kind to "model"',
                         "judge_model")
        for name in ("effort", "prompt"):
            if judge.get(name) is not None:
                c.string(judge[name], f"{w}.{name}")
        c.choice(judge["mode"], f"{w}.mode", C.JUDGE_MODES)
        C._reference(c, judge["reference"], f"{w}.reference")
        for name in ("blinded", "anchor"):
            c.boolean(judge[name], f"{w}.{name}")
        if judge.get("cost_usd_per_call") is not None:
            _number(c, judge["cost_usd_per_call"], f"{w}.cost_usd_per_call", low=0)
        if judge["components"] is not None:
            for n, name in enumerate(c.array(judge["components"], f"{w}.components", non_empty=True)):
                c.ident(name, f"{w}.components[{n}]")
        for name in JUDGE_STATS:
            _stat(c, judge[name], f"{w}.{name}")
        rows = c.array(judge["dims"], f"{w}.dims")
        if len(rows) != len(ids):
            raise c.fail(f"{w}.dims", f"{len(rows)} entries for {len(ids)} dimensions",
                         "write one entry per dimension, in the order of index.dimensions", "bad_index")
        for d, row in enumerate(rows):
            v = f"{w}.dims[{d}]"
            _record(c, row, v, ("dimension", *JUDGE_DIM_STATS, *JUDGE_DIM_COUNTS), nullable=JUDGE_DIM_STATS)
            _same(c, row["dimension"], ids[d], f"{v}.dimension", "the dimension at this place,",
                  "write one entry per dimension, in the order of index.dimensions")
            for name in JUDGE_DIM_STATS:
                _stat(c, row[name], f"{v}.{name}")
            for name in JUDGE_DIM_COUNTS:
                c.integer(row[name], f"{v}.{name}", low=0)
        rows = c.array(judge["family"], f"{w}.family")
        if len(rows) != len(cells):
            raise c.fail(f"{w}.family", f"{len(rows)} entries for {len(index['families'])} families and "
                         f"{len(ids)} dimensions", "write one entry per family and dimension, families first",
                         "bad_index")
        for n, row in enumerate(rows):
            v = f"{w}.family[{n}]"
            _record(c, row, v, ("family", "dimension", "effect"), nullable=("effect",))
            _same(c, row["family"], cells[n][0], f"{v}.family", "the family at this place,",
                  "write one entry per family and dimension, families first")
            _same(c, row["dimension"], cells[n][1], f"{v}.dimension", "the dimension at this place,",
                  "write one entry per family and dimension, families first")
            _stat(c, row["effect"], f"{v}.effect")


def _check_targets(c: C._Check, targets: list, index: dict) -> None:
    runs = {run: r for r, run in enumerate(index["runs"])}
    by_step: dict[str, list[str]] = {}
    for dim in index["dimensions"]:
        by_step.setdefault(dim["step"], []).append(dim["id"])
    seen = set()
    for n, target in enumerate(targets):
        w = f"targets[{n}]"
        _record(c, target, w, ("run", "step", "input", "arm", "refs", "dims"))
        for name in ("run", "step", "input", "arm"):
            c.string(target[name], f"{w}.{name}")
        if target["run"] not in runs:
            raise c.fail(f"{w}.run", f'unknown run "{target["run"]}"', "name a run of index.runs", "bad_index")
        if target["step"] not in by_step:
            raise c.fail(f"{w}.step", f'no dimension of index.dimensions is on the step "{target["step"]}"',
                         "name a step that has a dimension", "bad_index")
        if (target["run"], target["step"]) in seen:
            raise c.fail(w, f'the run "{target["run"]}" at the step "{target["step"]}" appears twice',
                         "give every target once", "duplicate")
        seen.add((target["run"], target["step"]))
        r = runs[target["run"]]
        _same(c, target["input"], index["inputs"][index["run_input"][r]], f"{w}.input",
              "the input of the run in index.run_input,", "write the input of the run")
        _same(c, target["arm"], index["arms"][index["run_arm"][r]], f"{w}.arm", "the arm of the run in index.run_arm,",
              "write the arm of the run")
        refs = target["refs"]
        if not isinstance(refs, dict):
            raise c.fail(f"{w}.refs", f"expected an object, got {C._kind(refs)}",
                         "map a name (input, output) to a reference", "type")
        for name, ref in refs.items():
            c.string(ref, f"{w}.refs.{name}")
        rows = c.array(target["dims"], f"{w}.dims")
        ids = by_step[target["step"]]
        if len(rows) != len(ids):
            raise c.fail(f"{w}.dims", f'{len(rows)} entries for {len(ids)} dimensions of the step "{target["step"]}"',
                         "write one entry per dimension of the step, in the order of index.dimensions", "bad_index")
        for d, row in enumerate(rows):
            v = f"{w}.dims[{d}]"
            _record(c, row, v, ("dimension", "pooled", "unpooled", "judges"), nullable=("pooled", "unpooled"))
            _same(c, row["dimension"], ids[d], f"{v}.dimension", "the dimension at this place,",
                  "write one entry per dimension of the step, in the order of index.dimensions")
            if row["pooled"] is not None:
                _record(c, row["pooled"], f"{v}.pooled", ("mean", "sd"))
                _number(c, row["pooled"]["mean"], f"{v}.pooled.mean")
                _number(c, row["pooled"]["sd"], f"{v}.pooled.sd", low=0)
            if row["unpooled"] is not None:
                _record(c, row["unpooled"], f"{v}.unpooled", ("mean", "sd", "n"), nullable=("sd",))
                _number(c, row["unpooled"]["mean"], f"{v}.unpooled.mean")
                if row["unpooled"]["sd"] is not None:
                    _number(c, row["unpooled"]["sd"], f"{v}.unpooled.sd", low=0)
                c.integer(row["unpooled"]["n"], f"{v}.unpooled.n", low=0)
            calls = row["judges"]
            if not isinstance(calls, dict):
                raise c.fail(f"{v}.judges", f"expected an object, got {C._kind(calls)}",
                             "map a judge id to its count of calls", "type")
            for name, count in calls.items():
                if name not in index["judges"]:
                    raise c.fail(f"{v}.judges.{name}", f'unknown judge "{name}"', "name a judge of index.judges",
                                 "bad_index")
                c.integer(count, f"{v}.judges.{name}", low=0)


def _check_draws(c: C._Check, draws: dict, index: dict) -> None:
    name = c.string(draws["file"], "draws.file")
    if name in (".", "..") or "/" in name or "\\" in name:
        raise c.fail("draws.file", f'"{name}" is not a file name', "name the draws file that sits beside fit.json",
                     "bad_file")
    digest = c.string(draws["sha256"], "draws.sha256")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise c.fail("draws.sha256", "it is not a SHA-256 digest", "write 64 hex digits, lower case", "bad_digest")
    sizes = {**_sizes(index, {}), "S": c.integer(draws["n"], "draws.n", low=1)}
    arrays = draws["arrays"]
    if not isinstance(arrays, dict):
        raise c.fail("draws.arrays", f"expected an object, got {C._kind(arrays)}", "map an array name to its shape",
                     "type")
    for array in DRAW_SHAPES:
        if array not in arrays and array not in OPTIONAL_DRAWS:
            raise c.fail("draws.arrays", f'missing member "{array}"', "add it", "missing_member")
    for array, shape in arrays.items():
        w = f"draws.arrays.{array}"
        if array not in DRAW_SHAPES:
            raise c.fail(w, "unknown array", "write only the arrays of fitfile.DRAW_SHAPES", "draws_unknown")
        for n, size in enumerate(c.array(shape, w)):
            c.integer(size, f"{w}[{n}]", low=0)
        axes = DRAW_SHAPES[array]
        if len(shape) != len(axes) or any(axis in sizes and size != sizes[axis] for axis, size in zip(axes, shape)):
            want = ", ".join(str(sizes.get(axis, axis)) for axis in axes)
            raise c.fail(w, f"shape {tuple(shape)} is not ({', '.join(axes)}) = ({want})",
                         "match the index of fit.json", "draws_shape")


def validate_fit(obj: Any, file: str = "fit.json") -> dict:
    """Check every member of `fit.json` against the contract at the top of this module: the members that
    must be there, their types, no unknown member (but under `ext`), and that every block agrees with
    `index`. A later stage can then index what it reads without a check of its own. Returns the object
    as given."""
    try:
        return _validate(obj, file)
    except OverflowError:  # a whole number too large for a float, in a place a shared check reads
        raise ContractError(file, "", "a number is too large to read", "write finite numbers", "type") from None


def _validate(obj: Any, file: str) -> dict:
    c = C._Check(file)
    C._schema(c, obj, "", C.SCHEMA_FIT)
    c.obj(obj, "", MEMBERS, [name for name in MEMBERS if name != "schema"])
    index = c.obj(obj["index"], "index", {"arms", "dimensions", "judges", "families", "inputs", "runs",
                                           "run_arm", "run_input"},
                  {"arms", "dimensions", "judges", "families", "inputs", "runs", "run_arm", "run_input"})
    for name in ("arms", "judges", "families", "inputs", "runs"):
        for n, entry in enumerate(c.array(index[name], f"index.{name}")):
            c.string(entry, f"index.{name}[{n}]")
        c.unique(index[name], f"index.{name}", "id")
    for n, dim in enumerate(c.array(index["dimensions"], "index.dimensions", non_empty=True)):
        w = f"index.dimensions[{n}]"
        c.obj(dim, w, {"id", "step", "component", "kind", "cuts"}, {"id", "step", "component", "kind"})
        for name in ("id", "step", "component"):
            c.string(dim[name], f"{w}.{name}")
        _same(c, dim["id"], C.dimension_id(dim["step"], dim["component"]), f"{w}.id", "the step and the component,",
              "write the id as step/component")
        c.choice(dim["kind"], f"{w}.kind", C.SCALE_KINDS)
        if dim.get("cuts") is not None:
            if dim["kind"] != "levels":
                raise c.fail(f"{w}.cuts", f'a "{dim["kind"]}" scale has no cuts', "write null", "scale_member")
            cuts = [_number(c, cut, f"{w}.cuts[{m}]") for m, cut in enumerate(c.array(dim["cuts"], f"{w}.cuts",
                                                                                  non_empty=True))]
            if any(low >= high for low, high in zip(cuts, cuts[1:])):
                raise c.fail(f"{w}.cuts", "the cuts do not rise", "list the cuts lowest first", "bad_cuts")
    c.unique([dim["id"] for dim in index["dimensions"]], "index.dimensions", "dimension id")
    for name, size in (("run_arm", len(index["arms"])), ("run_input", len(index["inputs"]))):
        rows = c.array(index[name], f"index.{name}")
        if len(rows) != len(index["runs"]) or any(
                isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < size for v in rows):
            raise c.fail(f"index.{name}", "it does not give one valid position per run",
                         "write one position per entry of index.runs", "bad_index")
    ids = [dim["id"] for dim in index["dimensions"]]

    inputs = obj["inputs"]
    if not isinstance(inputs, dict):
        raise c.fail("inputs", f"expected an object, got {C._kind(inputs)}", "write a JSON object here", "type")
    for role, entry in inputs.items():
        if role not in INPUT_ROLES:
            raise c.fail(f"inputs.{role}", "unknown member", f"use one of {', '.join(INPUT_ROLES)}", "unknown_member")
        _record(c, entry, f"inputs.{role}", ("file", "sha256"))
        for name in ("file", "sha256"):
            c.string(entry[name], f"inputs.{role}.{name}")
    if not isinstance(obj["options"], dict):
        raise c.fail("options", f"expected an object, got {C._kind(obj['options'])}", "write a JSON object here",
                     "type")
    engine = _record(c, obj["engine"], "engine", ("name", "version"))
    c.string(engine["name"], "engine.name")
    _text(c, engine["version"], "engine.version")
    c.integer(obj["seed"], "seed")

    dims = c.array(obj["dimensions"], "dimensions")
    if [dim.get("id") if isinstance(dim, dict) else None for dim in dims] != ids:
        raise c.fail("dimensions", "the ids do not match index.dimensions", "list the same dimensions in order",
                     "bad_index")
    _check_dimensions(c, dims, index)
    for name, known in (("arms", index["arms"]), ("judges", index["judges"])):
        rows = c.array(obj[name], name)
        if [row.get("id") if isinstance(row, dict) else None for row in rows] != known:
            raise c.fail(name, f"the ids do not match index.{name}", f"list the same {name} in order", "bad_index")
    _check_arms(c, obj["arms"], index)
    _check_judges(c, obj["judges"], index)
    _check_levels(c, obj["levels"], index)
    targets = c.array(obj["targets"], "targets")
    _check_targets(c, targets, index)

    counts = _record(c, obj["counts"], "counts", COUNTS)
    for name in COUNTS:
        c.integer(counts[name], f"counts.{name}", low=0)
    for name, size, what in (("runs", len(index["runs"]), "index.runs"), ("inputs", len(index["inputs"]),
                                                                           "index.inputs"),
                             ("arms", len(index["arms"]), "index.arms"), ("targets", len(targets), "targets")):
        _same(c, counts[name], size, f"counts.{name}", f"the length of {what},", f"count the entries of {what}")

    for n, row in enumerate(c.array(obj["terms"], "terms")):
        w = f"terms[{n}]"
        c.obj(row, w, {"term", "judge", "dimension", "state", "count", "needs", "reason"}, {"term", "state"})
        c.choice(row["term"], f"{w}.term", TERMS)
        c.choice(row["state"], f"{w}.state", TERM_STATES)
        for name, known in (("judge", index["judges"]), ("dimension", ids)):
            if row.get(name) is not None and row[name] not in known:
                shown = f'"{row[name]}"' if isinstance(row[name], str) else C._kind(row[name])
                raise c.fail(f"{w}.{name}", f"unknown {name} {shown}", f"name a {name} of the index, or write null",
                             "bad_index")
        if row.get("count") is not None:
            c.integer(row["count"], f"{w}.count", low=0)
        for name in ("needs", "reason"):
            if row.get(name) is not None:
                _text(c, row[name], f"{w}.{name}")
    diagnostics = c.obj(obj["diagnostics"], "diagnostics",
                        {"converged", "rhat_max", "ess_min", "divergences", "run_seconds", "checks"}, {"converged"})
    c.boolean(diagnostics["converged"], "diagnostics.converged")
    for name in ("rhat_max", "ess_min", "run_seconds"):
        if diagnostics.get(name) is not None:
            _number(c, diagnostics[name], f"diagnostics.{name}", low=0)
    if diagnostics.get("divergences") is not None:
        c.integer(diagnostics["divergences"], "diagnostics.divergences", low=0)
    if diagnostics.get("checks") is not None:
        for n, row in enumerate(c.array(diagnostics["checks"], "diagnostics.checks")):
            if not isinstance(row, dict):
                raise c.fail(f"diagnostics.checks[{n}]", f"expected an object, got {C._kind(row)}",
                             "write a JSON object here", "type")
    anchor = c.obj(obj["anchor"], "anchor", {"kind", "judges", "min_anchor_targets", "dimensions"},
                   {"kind", "dimensions"})
    c.choice(anchor["kind"], "anchor.kind", ("person", "judges_mean"))
    if anchor.get("judges") is not None:
        for n, name in enumerate(c.array(anchor["judges"], "anchor.judges")):
            if name not in index["judges"]:
                shown = f'"{name}"' if isinstance(name, str) else C._kind(name)
                raise c.fail(f"anchor.judges[{n}]", f"unknown judge {shown}", "name a judge of index.judges",
                             "bad_index")
    if anchor.get("min_anchor_targets") is not None:
        c.integer(anchor["min_anchor_targets"], "anchor.min_anchor_targets", low=1)
    for n, row in enumerate(c.array(anchor["dimensions"], "anchor.dimensions")):
        w = f"anchor.dimensions[{n}]"
        c.obj(row, w, {"dimension", "state", "n_targets"}, {"dimension", "state"})
        c.choice(row["state"], f"{w}.state", ANCHOR_STATES)
        if row["dimension"] not in ids:
            shown = f'"{row["dimension"]}"' if isinstance(row["dimension"], str) else C._kind(row["dimension"])
            raise c.fail(f"{w}.dimension", f"unknown dimension {shown}", "name a dimension of index.dimensions",
                         "bad_index")
        if row.get("n_targets") is not None:
            c.integer(row["n_targets"], f"{w}.n_targets", low=0)
    c.unique([row["dimension"] for row in anchor["dimensions"]], "anchor.dimensions", "dimension")
    for n, row in enumerate(c.array(obj["notes"], "notes")):
        _record(c, row, f"notes[{n}]", ("kind", "id", "reason"))
        for name in ("kind", "id", "reason"):
            c.string(row[name], f"notes[{n}].{name}")
    _check_draws(c, c.obj(obj["draws"], "draws", {"file", "sha256", "n", "arrays"},
                          {"file", "sha256", "n", "arrays"}), index)
    return obj


# ---------------------------------------------------------------------------------------------------
# Writing and reading

def _npz_bytes(draws: Mapping[str, np.ndarray]) -> bytes:
    """An .npz with fixed entry order and times, so the same draws give the same bytes."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name in sorted(draws):
            entry = io.BytesIO()
            np.lib.format.write_array(entry, np.ascontiguousarray(draws[name], dtype=np.float64),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            archive.writestr(info, entry.getvalue())
    return buffer.getvalue()


def write_fit(path: str | os.PathLike, fit: dict, draws: Mapping[str, np.ndarray]) -> Path:
    """Write `fit.json` and its draws file. `fit` is every member but `draws` (see `assemble`).

    The draws are checked against the index first, so a refusal writes nothing. The draws file is written
    before `fit.json`, which records its digest."""
    path = Path(path)
    draws = {name: np.asarray(array, dtype=np.float64) for name, array in draws.items()}
    sizes = check_draws(fit["index"], draws, draws_path(path).name)
    data = _npz_bytes(draws)
    full = {name: fit[name] for name in MEMBERS if name in fit and name != "draws"}
    full["draws"] = {"file": draws_path(path).name, "sha256": hashlib.sha256(data).hexdigest(), "n": sizes["S"],
                     "arrays": {name: list(draws[name].shape) for name in sorted(draws)}}
    validate_fit(full, path.name)
    text = C.dumps(full)
    C.write_text_atomic(draws_path(path), data)
    return C.write_text_atomic(path, text)


class Fit:
    """A fit file, read. `meta` is `fit.json`; `draws` loads the arrays on first use."""

    def __init__(self, path: Path, meta: dict) -> None:
        self.path, self.meta = Path(path), meta
        self.index = meta["index"]
        self.dimensions: list[dict] = meta["dimensions"]
        self._draws: dict[str, np.ndarray] | None = None

    def __getitem__(self, name: str) -> Any:
        return self.meta[name]

    @property
    def draws(self) -> dict[str, np.ndarray]:
        """The arrays of `DRAW_SHAPES`. The file's digest and shapes are checked on first use."""
        if self._draws is None:
            file = self.meta["draws"]["file"]
            source = self.path.with_name(file)
            if not source.is_file():
                raise ContractError(self.path.name, "draws.file", f"the draws file {file} is missing",
                                    "keep it beside fit.json, or fit again", "draws_missing")
            if C.sha256_file(source) != self.meta["draws"]["sha256"]:
                raise ContractError(self.path.name, "draws.sha256", f"the draws file {file} does not match its digest",
                                    "fit again: fit.json and its draws come from different fits", "draws_digest")
            with np.load(source, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            check_draws(self.index, arrays, file)
            self._draws = arrays
        return self._draws

    @property
    def arms(self) -> list[str]:
        return self.index["arms"]

    @property
    def judges(self) -> list[str]:
        return self.index["judges"]

    @property
    def kinds(self) -> list[str]:
        """The scale kind of each dimension, as `quality.reported_all` takes them."""
        return [dim["kind"] for dim in self.index["dimensions"]]

    @property
    def sizes(self) -> dict[str, int]:
        """Axis sizes: S, A, D, J, F, I, R."""
        index = self.index
        return {"S": int(self.meta["draws"]["n"]), "A": len(index["arms"]), "D": len(index["dimensions"]),
                "J": len(index["judges"]), "F": len(index["families"]), "I": len(index["inputs"]),
                "R": len(index["runs"])}

    @property
    def run_arm(self) -> np.ndarray:
        return np.asarray(self.index["run_arm"], dtype=int)

    @property
    def run_input(self) -> np.ndarray:
        return np.asarray(self.index["run_input"], dtype=int)

    @property
    def converged(self) -> bool:
        return bool(self.meta["diagnostics"]["converged"])

    def _position(self, name: str, ids: list, value: str) -> int:
        try:
            return ids.index(value)
        except ValueError:
            known = ", ".join(str(x) for x in ids)
            raise ContractError(self.path.name, f"index.{name}", f'unknown {name[:-1]} "{value}"',
                                f"use one of {known}", f"unknown_{name[:-1]}") from None

    def arm(self, arm_id: str) -> int:
        return self._position("arms", self.index["arms"], arm_id)

    def judge(self, judge_id: str) -> int:
        return self._position("judges", self.index["judges"], judge_id)

    def dim(self, component: str, step: str = C.FINAL) -> int:
        """The position of the dimension `step/component`. A full id (`final/accuracy`) is taken too."""
        key = component if "/" in component else C.dimension_id(step, component)
        return self._position("dimensions", [dim["id"] for dim in self.index["dimensions"]], key)

    def final_dims(self) -> list[int]:
        return [n for n, dim in enumerate(self.index["dimensions"]) if dim["step"] == C.FINAL]

    def term(self, name: str, judge: str | None = None, dimension: str | None = None) -> str | None:
        """The identification state of a term, or None when the fit has no entry for it."""
        for row in self.meta["terms"]:
            if row["term"] == name and row.get("judge") == judge and row.get("dimension") == dimension:
                return row["state"]
        return None

    def measured(self, name: str, judge: str | None = None, dimension: str | None = None) -> bool:
        """True when the design identifies the term (`identified` or `weak`): only then is it a finding."""
        return self.term(name, judge, dimension) in MEASURED_STATES

    def stale(self, files: dict[str, str | os.PathLike]) -> list[str]:
        """What changed between the inputs the fit recorded and `files` (role to path)."""
        return C.stale_inputs(self.meta["inputs"], files)


def read_fit(path: str | os.PathLike, *, allow_unconverged: bool = False) -> Fit:
    """Read `fit.json`. Refuses an unknown schema, and a fit that did not converge unless allowed."""
    path = Path(path)
    meta = validate_fit(C.load_json(path), path.name)
    if not meta["diagnostics"]["converged"] and not allow_unconverged:
        raise ContractError(path.name, "diagnostics.converged", "the fit did not converge",
                            "fit again with more draws, or pass --allow-unconverged", "unconverged")
    return Fit(path, meta)


def check_stale(fit: Fit, files: dict[str, str | os.PathLike], strict: bool = False) -> list[str]:
    """Stale inputs: the changes as warning lines. With `strict`, a change is a refusal."""
    changes = fit.stale(files)
    if changes and strict:
        raise ContractError(fit.path.name, "inputs", "; ".join(changes), "fit again, or drop --strict", "stale")
    return changes
