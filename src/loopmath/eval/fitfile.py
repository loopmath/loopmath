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


def validate_fit(obj: Any, file: str = "fit.json") -> dict:
    """Check the members of `fit.json` that later stages index by. Returns the object as given."""
    c = C._Check(file)
    C._schema(c, obj, "", C.SCHEMA_FIT)
    c.obj(obj, "", MEMBERS, [name for name in MEMBERS if name != "schema"])
    index = c.obj(obj["index"], "index", {"arms", "dimensions", "judges", "families", "inputs", "runs",
                                           "run_arm", "run_input"},
                  {"arms", "dimensions", "judges", "families", "inputs", "runs", "run_arm", "run_input"})
    for name in ("arms", "judges", "families", "inputs", "runs"):
        c.array(index[name], f"index.{name}")
        c.unique(index[name], f"index.{name}", "id")
    for n, dim in enumerate(c.array(index["dimensions"], "index.dimensions", non_empty=True)):
        c.obj(dim, f"index.dimensions[{n}]", {"id", "step", "component", "kind", "cuts"},
              {"id", "step", "component", "kind"})
        c.choice(dim["kind"], f"index.dimensions[{n}].kind", C.SCALE_KINDS)
    c.unique([dim["id"] for dim in index["dimensions"]], "index.dimensions", "dimension id")
    for name, size in (("run_arm", len(index["arms"])), ("run_input", len(index["inputs"]))):
        rows = c.array(index[name], f"index.{name}")
        if len(rows) != len(index["runs"]) or any(
                isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < size for v in rows):
            raise c.fail(f"index.{name}", "it does not give one valid position per run",
                         "write one position per entry of index.runs", "bad_index")
    dims = c.array(obj["dimensions"], "dimensions")
    if [dim.get("id") if isinstance(dim, dict) else None for dim in dims] != [d["id"] for d in index["dimensions"]]:
        raise c.fail("dimensions", "the ids do not match index.dimensions", "list the same dimensions in order",
                     "bad_index")
    for name, ids in (("arms", index["arms"]), ("judges", index["judges"])):
        rows = c.array(obj[name], name)
        if [row.get("id") if isinstance(row, dict) else None for row in rows] != ids:
            raise c.fail(name, f"the ids do not match index.{name}", f"list the same {name} in order", "bad_index")
    for n, row in enumerate(c.array(obj["terms"], "terms")):
        c.obj(row, f"terms[{n}]", {"term", "judge", "dimension", "state", "count", "needs", "reason"},
              {"term", "state"})
        c.choice(row["state"], f"terms[{n}].state", TERM_STATES)
    diagnostics = c.obj(obj["diagnostics"], "diagnostics",
                        {"converged", "rhat_max", "ess_min", "divergences", "run_seconds", "checks"}, {"converged"})
    c.boolean(diagnostics["converged"], "diagnostics.converged")
    anchor = c.obj(obj["anchor"], "anchor", {"kind", "judges", "min_anchor_targets", "dimensions"},
                   {"kind", "dimensions"})
    c.choice(anchor["kind"], "anchor.kind", ("person", "judges_mean"))
    for n, row in enumerate(c.array(anchor["dimensions"], "anchor.dimensions")):
        c.obj(row, f"anchor.dimensions[{n}]", {"dimension", "state", "n_targets"}, {"dimension", "state"})
        c.choice(row["state"], f"anchor.dimensions[{n}].state", ANCHOR_STATES)
    c.obj(obj["draws"], "draws", {"file", "sha256", "n", "arrays"}, {"file", "sha256", "n", "arrays"})
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
