"""`eval simulate`: a synthetic study with known truth.

    loopmath eval simulate SPEC --out DIR [--seed N] [--json]
    loopmath eval simulate --list

SPEC is a spec file (`loopmath.eval-sim/1`) or a catalog name (the catalog lives in `specs/`). The command
writes a study folder: `runs.jsonl`, `scores.jsonl`, `pairs.jsonl` and `items.jsonl` as needed, the matching
`rubric.json` and `judges.json`, `value.json` when the spec carries a value file, and `truth.json`. Every
run is marked `simulated`. The same spec and `--seed` give the same files. A folder that already holds one
of these files is refused, and a refusal writes nothing.

The spec (`validate_spec` returns it normalised, every per-dimension map filled by dimension id):

| Member | Meaning |
|---|---|
| `name`, `title` | An id and one line of description. |
| `inputs`, `runs_per_input` | How many inputs, and runs of each arm per input (default 1). |
| `arms[]` | `id`, `steps[]` (as in a run row), `mean` (per dimension, on the fitting scale), `cost_usd`. |
| `rubric` | A rubric as in `rubric.json`. `schema` may be left out. |
| `quality` | `sd_input`, `sd_run` (per dimension); `corr_input`, `corr_run`; `cuts` (per `levels` dimension). |
| `judges` | A judges file whose judges each carry a `truth` member (below). |
| `design` | `scores[]`, `human[]`, `pairs[]`, `items[]` (below). |
| `value` | Optional value file. It is written as `value.json` and `truth.json` gains the true value per arm. |

A per-dimension value is one number for every dimension, or a map from a component id (every step it
applies to) or a `step/component` id to a number. A correlation is a number (every off-diagonal entry),
a D x D matrix, or `{"default": r, "pairs": [[dim, dim, r], ...]}`.

A judge's `truth`: `bias` (default 0), `tau`, `sigma`, `sd_judge_input` (default 0) and `sigma_pair` per
dimension; `halo_target`, `halo_call`, `position` and `tie_width` as numbers (default 0); `family`, a map
from a model family to a per-dimension effect (model judges only).

The design:

- `scores[]`: `judge`, `repeats` (calls per target, default 1), `share` (of the targets, default 1),
  `components`, `steps`, `arms` (default all). A `panel` judge scores every component of a target in one call.
- `human[]`: `judge` (a person), `rate`, `selection` (`uniform`, or `judge_scores`: targets the judges
  scored low are drawn more often, with weight `exp(-strength x z)`), `repeat_share` (the share of scored
  targets the person scores a second time), `components`, `steps`, `arms`. The inclusion probability of
  every candidate target is written to `truth.json`.
- `pairs[]`: `judge`, `component` (default the step's overall component), `step`, `arms` (pairs of arm
  ids, default every pair), `share` (of the inputs), `orders` (`both`, `random`, `a_first`, `b_first` or
  `unknown`), `repeats`.
- `items[]`: `component`, `per_input` (a count or `[low, high]`), `missing_share` (inputs with no item),
  `as` (`items`: one line per item, or `count`: one line with `of`), `item_sd` (spread of item
  difficulty; 0 gives the k of n model exactly).

Readings follow the model one to one. For judge j, run r and dimension d the judge's view is
`q_r[d] + bias + family + w_input + d_target`, a call adds noise, and the scale turns the sum into a value:
a number, the level whose interval holds it, 1 when it is at least 0, or k of n items passed. A pair
compares two views, plus position bias and pair noise, against the tie width.

Units. The spec states the truth on the scale the fit reports, so a spec that is off that scale is refused:
on a `binary` dimension the anchors' mean log `sigma` is 0; on a dimension with pairs only the anchors'
mean log `sigma_pair` is 0 and the arm means sum to 0; the anchors' biases sum to 0. Where no anchor reads
the dimension, the rule is over every judge that reads it.

`truth.json` (`loopmath.eval-truth/1`; `read_truth` gives the arrays as numpy): `spec`, `seed`, `index`
(the same orders as the fit file's index), the arrays `mu` (A x D), `sd_input`, `sd_run` (D), `corr_input`,
`corr_run` (D x D), `q_run` (R x D), `v_input` (I x D), `bias`, `tau`, `sigma`, `sd_judge_input`,
`sigma_pair` (J x D), `halo_target`, `halo_call`, `position`, `tie_width` (J), `family` (J x F x D) and
`cuts` (D x Lmax-1, when a `levels` dimension exists); `reported` (true reported quality per arm and per
run); `relative` (pairs-only dimensions) and `units`; `binary_rates` and `pair_rates` (per judge, as
simulated and, for binary, as expected); `overall` (the share of the overall quality the criteria
explain); `value`; `human`; `items`; `counts`. A parameter that does not exist for a judge or dimension
is null.

All data is synthetic.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr

from ..output import EXIT_OK, emit_json
from . import contracts as C
from . import quality as Q

TRUTH_FILE = "truth.json"
ORDERS = ("both", "random", "a_first", "b_first", "unknown")
SELECTIONS = ("uniform", "judge_scores")
ITEM_FORMS = ("items", "count")
ARRAYS = ("mu", "sd_input", "sd_run", "corr_input", "corr_run", "q_run", "v_input", "bias", "tau", "sigma",
          "sd_judge_input", "sigma_pair", "halo_target", "halo_call", "position", "tie_width", "family", "cuts")
VALUE_DRAWS = 200_000   # new (input, run) draws behind the true value and pass probability of an arm
VALUE_SEED = 20260101   # their generator's seed: the population truth does not depend on --seed
ROUND = 6               # decimals of a `linear` or `fraction` reading
TOLERANCE = 1e-9
START = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)


# ---------------------------------------------------------------------------------------------------
# The spec

def _sub(file: str, where: str) -> str:
    return f"{file}: {where}"


def _nested(c: C._Check, value: Any, where: str, schema: str) -> dict:
    if not isinstance(value, dict):
        raise c.fail(where, f"expected an object, got {C._kind(value)}", "write a JSON object here", "type")
    return value if "schema" in value else {"schema": schema, **value}


def _per_dim(c: C._Check, value: Any, where: str, dims: list[dict], *, default: float | None = None,
             required: bool = False, low: float | None = None, above: float | None = None) -> dict[str, float]:
    """A per-dimension value as a map from dimension id to a number, in dimension order."""
    ids = [dim["id"] for dim in dims]
    of_component: dict[str, list[str]] = {}
    for dim in dims:
        of_component.setdefault(dim["component"], []).append(dim["id"])
    given: dict[str, float] = {}
    if isinstance(value, dict):
        for key in value:
            if key not in of_component and key not in ids:
                raise c.fail(f"{where}.{key}", "unknown component or dimension",
                             f"use a component id or one of {', '.join(ids)}", "unknown_dimension")
        for exact in (False, True):  # a step/component entry wins over a component entry
            for key, x in value.items():
                if (key in ids) == exact:
                    x = float(c.number(x, f"{where}.{key}", low=low, above=above))
                    for dim_id in ([key] if exact else of_component[key]):
                        given[dim_id] = x
    elif value is not None:
        x = float(c.number(value, where, low=low, above=above))
        given = {dim_id: x for dim_id in ids}
    out = {}
    for dim_id in ids:
        if dim_id in given:
            out[dim_id] = given[dim_id]
        elif default is not None:
            out[dim_id] = default
        elif required:
            raise c.fail(where, f'no value for the dimension "{dim_id}"',
                         "give a number for every dimension, by component id or step/component",
                         "missing_dimension")
    return out


def _one_dim(c: C._Check, key: Any, where: str, dims: list[dict]) -> int:
    """The position of the one dimension a component id or step/component id names."""
    found = [n for n, dim in enumerate(dims) if key in (dim["id"], dim["component"])]
    if len(found) != 1:
        shown = f'"{key}"' if isinstance(key, str) else C._kind(key)
        problem = f"{shown} names {len(found)} dimensions" if found else f"unknown component or dimension {shown}"
        raise c.fail(where, problem, f"use one of {', '.join(dim['id'] for dim in dims)}", "unknown_dimension")
    return found[0]


def _corr(c: C._Check, value: Any, where: str, dims: list[dict]) -> list[list[float]]:
    D = len(dims)
    m = np.eye(D)
    if isinstance(value, dict):
        c.obj(value, where, {"default", "pairs"})
        if value.get("default") is not None:
            m[~np.eye(D, dtype=bool)] = c.number(value["default"], f"{where}.default", low=-1, high=1)
        for n, entry in enumerate(c.array([] if value.get("pairs") is None else value["pairs"], f"{where}.pairs")):
            w = f"{where}.pairs[{n}]"
            if not isinstance(entry, list) or len(entry) != 3:
                raise c.fail(w, "expected [dimension, dimension, correlation]", "give two dimensions and a number",
                             "bad_pair")
            a, b = _one_dim(c, entry[0], w, dims), _one_dim(c, entry[1], w, dims)
            if a == b:
                raise c.fail(w, "the two dimensions are the same", "name two dimensions", "bad_pair")
            m[a, b] = m[b, a] = c.number(entry[2], w, low=-1, high=1)
    elif isinstance(value, list):
        if len(value) != D or any(not isinstance(row, list) or len(row) != D for row in value):
            raise c.fail(where, f"expected a {D} x {D} matrix", "give one row per dimension", "bad_matrix")
        m = np.array([[c.number(x, f"{where}[{a}][{b}]", low=-1, high=1) for b, x in enumerate(row)]
                      for a, row in enumerate(value)], dtype=float)
        if not np.allclose(m, m.T, atol=TOLERANCE) or not np.allclose(np.diag(m), 1.0, atol=TOLERANCE):
            raise c.fail(where, "the matrix is not symmetric with a diagonal of 1", "give a correlation matrix",
                         "bad_matrix")
    elif value is not None:
        m[~np.eye(D, dtype=bool)] = c.number(value, where, low=-1, high=1)
    smallest = float(np.linalg.eigvalsh(m).min())
    if smallest < -1e-8:
        raise c.fail(where, f"not a correlation matrix (smallest eigenvalue {smallest:.3g})",
                     "lower the correlations until no eigenvalue is negative", "bad_matrix")
    return [[float(x) for x in row] for row in m]


def _listed(c: C._Check, value: Any, where: str, allowed: list[str], what: str) -> list[str]:
    """A subset of `allowed`, in the order of `allowed`. None means all."""
    if value is None:
        return list(allowed)
    for n, name in enumerate(c.array(value, where, non_empty=True)):
        if not isinstance(name, str) or name not in allowed:
            shown = f'"{name}"' if isinstance(name, str) else C._kind(name)
            raise c.fail(f"{where}[{n}]", f"unknown {what} {shown}", f"use one of {', '.join(allowed)}",
                         f"unknown_{what}")
    c.unique(value, where, what)
    return [name for name in allowed if name in value]


def _share(c: C._Check, value: Any, where: str) -> float:
    return float(c.number(c.number(value, where, above=0), where, high=1))


def _scope(c: C._Check, row: dict, w: str, rubric: dict, judge: dict, arm_ids: list[str]) -> dict:
    """The components, steps and arms a design row covers."""
    scored = [comp["id"] for comp in rubric["components"]
              if judge["components"] is None or comp["id"] in judge["components"]]
    if isinstance(row.get("components"), list):
        for n, name in enumerate(row["components"]):
            if isinstance(name, str) and C.component(rubric, name) is not None and name not in scored:
                raise c.fail(f"{w}.components[{n}]", f'the judge "{judge["id"]}" does not score "{name}"',
                             "add it to the judge's components, or drop it here", "judge_component")
    out = {"components": _listed(c, row.get("components"), f"{w}.components", scored, "component"),
           "steps": _listed(c, row.get("steps"), f"{w}.steps", rubric["steps"], "step"),
           "arms": _listed(c, row.get("arms"), f"{w}.arms", arm_ids, "arm")}
    if not _covered(out, C.dimensions(rubric)):
        raise c.fail(w, "the row covers no dimension", "name components that apply to its steps", "empty_row")
    return out


def _covered(row: dict, dims: list[dict]) -> list[int]:
    return [n for n, dim in enumerate(dims) if dim["step"] in row["steps"] and dim["component"] in row["components"]]


def validate_spec(obj: Any, file: str = "spec.json") -> dict:
    """A simulate spec, normalised. Refusals are `ContractError`s that name the place and the fix."""
    c = C._Check(file)
    C._schema(c, obj, "", C.SCHEMA_SIM)
    c.obj(obj, "", {"schema", "name", "title", "inputs", "runs_per_input", "arms", "rubric", "quality", "judges",
                    "design", "value"}, {"name", "inputs", "arms", "rubric", "quality", "judges", "design"})
    out: dict = {"schema": C.SCHEMA_SIM, "name": c.ident(obj["name"], "name")}
    if obj.get("title") is not None:
        out["title"] = c.string(obj["title"], "title")
    out["inputs"] = c.integer(obj["inputs"], "inputs", low=1)
    out["runs_per_input"] = c.integer(obj.get("runs_per_input", 1), "runs_per_input", low=1)
    rubric = C.validate_rubric(_nested(c, obj["rubric"], "rubric", C.SCHEMA_RUBRIC), _sub(file, "rubric"))
    dims = C.dimensions(rubric)
    dim_ids = [dim["id"] for dim in dims]

    arms = []
    for n, arm in enumerate(c.array(obj["arms"], "arms", non_empty=True)):
        w = f"arms[{n}]"
        c.obj(arm, w, {"id", "steps", "mean", "cost_usd"}, {"id", "steps", "mean"})
        row: dict = {"id": c.ident(arm["id"], f"{w}.id")}
        row["steps"] = C.validate_run({"schema": C.SCHEMA_RUNS, "run": "r", "input": "i", "arm": row["id"],
                                       "steps": arm["steps"]}, _sub(file, w))["steps"]
        named = [step["step"] for step in row["steps"]]
        for step in rubric["steps"]:
            if step != C.FINAL and step not in named:
                raise c.fail(f"{w}.steps", f'the rubric step "{step}" is missing', "give the arm a setting for it",
                             "missing_step")
        row["mean"] = _per_dim(c, arm["mean"], f"{w}.mean", dims, required=True)
        if arm.get("cost_usd") is not None:
            row["cost_usd"] = c.number(arm["cost_usd"], f"{w}.cost_usd", low=0)
        arms.append(row)
    arm_ids = [row["id"] for row in arms]
    c.unique(arm_ids, "arms", "arm id")
    out["arms"] = arms
    out["rubric"] = rubric

    quality = c.obj(obj["quality"], "quality", {"sd_input", "sd_run", "corr_input", "corr_run", "cuts"},
                    {"sd_input", "sd_run"})
    out["quality"] = {"sd_input": _per_dim(c, quality["sd_input"], "quality.sd_input", dims, required=True, low=0),
                      "sd_run": _per_dim(c, quality["sd_run"], "quality.sd_run", dims, required=True, low=0),
                      "corr_input": _corr(c, quality.get("corr_input"), "quality.corr_input", dims),
                      "corr_run": _corr(c, quality.get("corr_run"), "quality.corr_run", dims)}
    cuts = {}
    raw_cuts = quality.get("cuts")
    if raw_cuts is not None:
        if not isinstance(raw_cuts, dict):
            raise c.fail("quality.cuts", f"expected an object, got {C._kind(raw_cuts)}",
                         "map a levels dimension to its cut points", "type")
        for key, points in raw_cuts.items():
            w = f"quality.cuts.{key}"
            dim = dims[_one_dim(c, key, w, dims)]
            if dim["scale"]["kind"] != "levels":
                raise c.fail(w, f'the dimension has a "{dim["scale"]["kind"]}" scale', "give cut points for levels only",
                             "cuts_scale")
            want = len(dim["scale"]["levels"]) - 1
            points = [float(c.number(x, f"{w}[{n}]")) for n, x in enumerate(c.array(points, w))]
            if len(points) != want or any(b <= a for a, b in zip(points, points[1:])):
                raise c.fail(w, f"expected {want} rising cut points", "give one cut point between every two levels",
                             "bad_cuts")
            cuts[dim["id"]] = points
    if cuts:
        out["quality"]["cuts"] = {dim_id: cuts[dim_id] for dim_id in dim_ids if dim_id in cuts}

    # Judges: the declaration goes through the judges contract, the truth is checked here.
    given = c.obj(obj["judges"], "judges", {"schema", "judges", "anchor", "min_anchor_targets"}, {"judges"})
    plain, truths = [], []
    for n, judge in enumerate(c.array(given["judges"], "judges.judges", non_empty=True)):
        if not isinstance(judge, dict):
            raise c.fail(f"judges.judges[{n}]", f"expected an object, got {C._kind(judge)}",
                         "write a JSON object here", "type")
        plain.append({key: value for key, value in judge.items() if key != "truth"})
        truths.append(judge.get("truth"))
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "judges": plain,
                                **{key: given[key] for key in ("anchor", "min_anchor_targets") if key in given}},
                               _sub(file, "judges"))
    rows = []
    for n, (judge, truth) in enumerate(zip(judges["judges"], truths)):
        w = f"judges.judges[{n}]"
        for m, name in enumerate(judge["components"] or []):
            if C.component(rubric, name) is None:
                raise c.fail(f"{w}.components[{m}]", f'unknown component "{name}"',
                             "name a component of the rubric", "unknown_component")
        w = f"{w}.truth"
        truth = c.obj({} if truth is None else truth, w,
                      {"bias", "tau", "sigma", "sd_judge_input", "sigma_pair", "halo_target", "halo_call", "position",
                       "tie_width", "family"})
        t: dict = {"bias": _per_dim(c, truth.get("bias"), f"{w}.bias", dims, default=0.0),
                   "tau": _per_dim(c, truth.get("tau"), f"{w}.tau", dims, low=0),
                   "sigma": _per_dim(c, truth.get("sigma"), f"{w}.sigma", dims, low=0),
                   "sd_judge_input": _per_dim(c, truth.get("sd_judge_input"), f"{w}.sd_judge_input", dims,
                                              default=0.0, low=0),
                   "sigma_pair": _per_dim(c, truth.get("sigma_pair"), f"{w}.sigma_pair", dims, above=0)}
        for name in ("halo_target", "halo_call"):
            t[name] = float(c.number(truth.get(name, 0.0), f"{w}.{name}", low=0, high=1))
        t["position"] = float(c.number(truth.get("position", 0.0), f"{w}.position"))
        t["tie_width"] = float(c.number(truth.get("tie_width", 0.0), f"{w}.tie_width", low=0))
        family = truth.get("family")
        if family is not None:
            if judge["kind"] != "model":
                raise c.fail(f"{w}.family", f'a judge of kind "{judge["kind"]}" has family effects',
                             "give family effects to model judges only", "family_judge")
            if not isinstance(family, dict):
                raise c.fail(f"{w}.family", f"expected an object, got {C._kind(family)}",
                             "map a model family to its effect", "type")
            t["family"] = {c.ident(name, f"{w}.family.{name}"): _per_dim(c, effect, f"{w}.family.{name}", dims,
                                                                          default=0.0)
                           for name, effect in family.items()}
        elif judge["kind"] == "model":
            t["family"] = {}
        rows.append({**judge, "truth": t})
    out["judges"] = {"anchor": judges["anchor"], "min_anchor_targets": judges["min_anchor_targets"], "judges": rows}
    by_judge = {row["id"]: row for row in rows}

    # The design.
    design = c.obj(obj["design"], "design", {"scores", "human", "pairs", "items"})
    plan: dict = {"scores": [], "human": [], "pairs": [], "items": []}

    def judge_of(row: dict, w: str) -> dict:
        name = row["judge"]
        if not isinstance(name, str) or name not in by_judge:
            shown = f'"{name}"' if isinstance(name, str) else C._kind(name)
            raise c.fail(f"{w}.judge", f"unknown judge {shown}", "name a judge id from judges", "unknown_judge")
        return by_judge[name]

    for n, row in enumerate(c.array([] if design.get("scores") is None else design["scores"], "design.scores")):
        w = f"design.scores[{n}]"
        c.obj(row, w, {"judge", "repeats", "share", "components", "steps", "arms"}, {"judge"})
        judge = judge_of(row, w)
        plan["scores"].append({"judge": judge["id"], "repeats": c.integer(row.get("repeats", 1), f"{w}.repeats", low=1),
                               "share": _share(c, row.get("share", 1.0), f"{w}.share"),
                               **_scope(c, row, w, rubric, judge, arm_ids)})
    for n, row in enumerate(c.array([] if design.get("human") is None else design["human"], "design.human")):
        w = f"design.human[{n}]"
        c.obj(row, w, {"judge", "rate", "selection", "strength", "repeat_share", "components", "steps", "arms"},
              {"judge", "rate"})
        judge = judge_of(row, w)
        if judge["kind"] != "person":
            raise c.fail(f"{w}.judge", f'the judge "{judge["id"]}" is of kind "{judge["kind"]}"',
                         "name a person here, and put other judges under design.scores", "human_judge")
        plan["human"].append({"judge": judge["id"], "rate": _share(c, row["rate"], f"{w}.rate"),
                              "selection": c.choice(row.get("selection", "uniform"), f"{w}.selection", SELECTIONS),
                              "strength": float(c.number(row.get("strength", 1.0), f"{w}.strength", low=0)),
                              "repeat_share": float(c.number(row.get("repeat_share", 0.0), f"{w}.repeat_share",
                                                             low=0, high=1)),
                              **_scope(c, row, w, rubric, judge, arm_ids)})
    for n, row in enumerate(c.array([] if design.get("pairs") is None else design["pairs"], "design.pairs")):
        w = f"design.pairs[{n}]"
        c.obj(row, w, {"judge", "component", "step", "arms", "share", "orders", "repeats"}, {"judge"})
        judge = judge_of(row, w)
        step = row.get("step") or C.FINAL
        if step not in rubric["steps"]:
            raise c.fail(f"{w}.step", f'unknown step "{step}"', f"use one of {', '.join(rubric['steps'])}",
                         "unknown_step")
        comp = row.get("component")
        if comp is None:
            comp = C.overall_component(rubric, step)
            if comp is None:
                raise c.fail(w, f'the step "{step}" has no overall component', "name the component compared",
                             "no_overall")
        if C.dimension_id(step, comp) not in dim_ids:
            raise c.fail(f"{w}.component", f'no dimension "{C.dimension_id(step, comp)}"',
                         f"use one of {', '.join(dim_ids)}", "unknown_dimension")
        if judge["components"] is not None and comp not in judge["components"]:
            raise c.fail(f"{w}.component", f'the judge "{judge["id"]}" does not score "{comp}"',
                         "add it to the judge's components, or drop the row", "judge_component")
        if len(arm_ids) < 2:
            raise c.fail(w, "pairs need two arms", "add an arm, or drop design.pairs", "one_arm")
        if row.get("arms") is None:
            duos = [[a, b] for i, a in enumerate(arm_ids) for b in arm_ids[i + 1:]]
        else:
            duos = []
            for m, duo in enumerate(c.array(row["arms"], f"{w}.arms", non_empty=True)):
                if not isinstance(duo, list) or len(duo) != 2 or duo[0] == duo[1] or any(a not in arm_ids for a in duo):
                    raise c.fail(f"{w}.arms[{m}]", "expected two different arm ids",
                                 f"use two of {', '.join(arm_ids)}", "bad_pair")
                duos.append(list(duo))
        plan["pairs"].append({"judge": judge["id"], "component": comp, "step": step, "arms": duos,
                              "share": _share(c, row.get("share", 1.0), f"{w}.share"),
                              "orders": c.choice(row.get("orders", "both"), f"{w}.orders", ORDERS),
                              "repeats": c.integer(row.get("repeats", 1), f"{w}.repeats", low=1)})
    for n, row in enumerate(c.array([] if design.get("items") is None else design["items"], "design.items")):
        w = f"design.items[{n}]"
        c.obj(row, w, {"component", "per_input", "missing_share", "as", "item_sd"}, {"component", "per_input"})
        comp = C.component(rubric, row["component"]) if isinstance(row["component"], str) else None
        if comp is None or comp["scale"]["kind"] != "items":
            raise c.fail(f"{w}.component", "not a component with the scale kind items",
                         "name an items component of the rubric", "not_items")
        per = row["per_input"]
        per = per if isinstance(per, list) else [per, per]
        if len(per) != 2:
            raise c.fail(f"{w}.per_input", "expected a count or [low, high]", "give one or two whole numbers",
                         "bad_range")
        low, high = c.integer(per[0], f"{w}.per_input", low=1), c.integer(per[1], f"{w}.per_input", low=1)
        if high < low:
            raise c.fail(f"{w}.per_input", f"low {low} is above high {high}", "give [low, high]", "bad_range")
        plan["items"].append({"component": comp["id"], "per_input": [low, high],
                              "missing_share": float(c.number(row.get("missing_share", 0.0), f"{w}.missing_share",
                                                              low=0, high=0.9)),
                              "as": c.choice(row.get("as", "items"), f"{w}.as", ITEM_FORMS),
                              "item_sd": float(c.number(row.get("item_sd", 0.0), f"{w}.item_sd", low=0))})
    c.unique([row["component"] for row in plan["items"]], "design.items", "component")
    if not plan["scores"] and not plan["human"] and not plan["pairs"]:
        raise c.fail("design", "no scores, no human and no pairs", "add a row to one of them", "empty_design")
    out["design"] = plan

    # What each judge reads, and the parameters that reading needs.
    scored, paired = _coverage(plan, dims)
    with_items = {row["component"] for row in plan["items"]}
    for n, row in enumerate(rows):
        w = f"judges.judges[{n}].truth"
        for d, dim in enumerate(dims):
            reads, pairs = (row["id"], d) in scored, (row["id"], d) in paired
            needs = [name for name, used in (("tau", reads or pairs), ("sigma", reads), ("sigma_pair", pairs))
                     if used and dim["id"] not in row["truth"][name]]
            if needs:
                raise c.fail(f"{w}.{needs[0]}", f'no value for the dimension "{dim["id"]}"',
                             f'the judge "{row["id"]}" reads it in the design: give a number', "missing_dimension")
            if reads and dim["scale"]["kind"] == "items" and dim["component"] not in with_items:
                raise c.fail("design.items", f'no row for the items component "{dim["component"]}"',
                             "say how many items an input has", "items_design")
    anchors = set(judges["anchor"])
    for d, dim in enumerate(dims):
        readers = [row for row in rows if (row["id"], d) in scored]
        pairers = [row for row in rows if (row["id"], d) in paired]
        anchored = [row for row in readers if row["id"] in anchors]
        total = sum(row["truth"]["bias"][dim["id"]] for row in anchored)
        if abs(total) > TOLERANCE:
            raise c.fail("judges", f'the anchors\' biases on "{dim["id"]}" sum to {total:.6g}',
                         "an anchor has no bias: make them sum to 0", "anchor_bias")
        name, group = ("sigma", anchored or readers) if readers else (
            "sigma_pair", [row for row in pairers if row["id"] in anchors] or pairers)
        if group and (dim["scale"]["kind"] == "binary" or not readers):
            values = [row["truth"][name][dim["id"]] for row in group]
            unit = math.exp(sum(math.log(x) for x in values) / len(values)) if all(x > 0 for x in values) else 0.0
            if abs(unit - 1.0) > 1e-6:
                what = "binary dimension" if readers else "dimension with pairs only"
                raise c.fail("judges", f'on the {what} "{dim["id"]}" the {name} of '
                                       f'{", ".join(row["id"] for row in group)} has the geometric mean {unit:.6g}',
                             f"that {name} is the unit: scale the dimension so it is 1", "unit")
        if pairers and not readers:
            total = sum(arm["mean"][dim["id"]] for arm in arms)
            if abs(total) > 1e-6:
                raise c.fail("arms", f'the arm means on the pairs-only dimension "{dim["id"]}" sum to {total:.6g}',
                             "only differences are measured there: make them sum to 0", "relative_mean")

    if obj.get("value") is not None:
        value = C.validate_value(_nested(c, obj["value"], "value", C.SCHEMA_VALUE), _sub(file, "value"))
        relative = _relative(plan, dims)
        Q.compile_value(value, [{**dim, "relative": dim["id"] in relative} for dim in dims], _sub(file, "value"))
        out["value"] = value
    return out


def _coverage(plan: dict, dims: list[dict]) -> tuple[set, set]:
    """`(judge id, d)` for every dimension a judge scores, and for every dimension it compares."""
    scored = {(row["judge"], d) for name in ("scores", "human") for row in plan[name] for d in _covered(row, dims)}
    paired = {(row["judge"], d) for row in plan["pairs"] for d, dim in enumerate(dims)
              if dim["id"] == C.dimension_id(row["step"], row["component"])}
    return scored, paired


def _relative(plan: dict, dims: list[dict]) -> list[str]:
    """The dimensions with pairs and no scores: only differences between arms are measured there."""
    scored, paired = _coverage(plan, dims)
    return [dim["id"] for d, dim in enumerate(dims)
            if any(p == d for _, p in paired) and not any(s == d for _, s in scored)]


# ---------------------------------------------------------------------------------------------------
# The generator

@dataclass
class Simulated:
    """A simulated study in memory: the files' contents, before `write`."""

    spec: dict
    seed: int
    rubric: dict
    judges: dict
    value: dict | None
    runs: list[dict]
    scores: list[dict]
    pairs: list[dict]
    items: list[dict]
    truth: dict


def _root(cov: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(cov)
    return v * np.sqrt(np.clip(w, 0.0, None))


def _clean(value: Any) -> Any:
    """Plain JSON values: arrays as nested lists, NaN as null."""
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, dict):
        return {key: _clean(x) for key, x in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(x) for x in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return value


def _family_of(arm: dict, step: str) -> str:
    """The model family that wrote a target: the family at that step; for `final`, of the last step."""
    for entry in arm["steps"]:
        if entry["step"] == step:
            return entry["model"]["family"]
    return arm["steps"][-1]["model"]["family"]


def generate(spec: dict, seed: int = 0, file: str = "spec.json") -> Simulated:
    """Simulate a study from a spec. The same spec and seed give the same study."""
    spec = validate_spec(spec, file)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise C.ContractError("--seed", "", f"{seed!r} is not a whole number from 0 up", "give a seed like 0", "bad_seed")
    rubric = spec["rubric"]
    judges = C.validate_judges({"schema": C.SCHEMA_JUDGES, "anchor": spec["judges"]["anchor"],
                                "min_anchor_targets": spec["judges"]["min_anchor_targets"],
                                "judges": [{k: v for k, v in row.items() if k != "truth"}
                                           for row in spec["judges"]["judges"]]})
    truths = [row["truth"] for row in spec["judges"]["judges"]]
    judge_ids = [row["id"] for row in judges["judges"]]
    dims = C.dimensions(rubric)
    dim_ids = [dim["id"] for dim in dims]
    kinds = [dim["scale"]["kind"] for dim in dims]
    steps = rubric["steps"]
    step_of = [steps.index(dim["step"]) for dim in dims]
    arms, plan, ref = spec["arms"], spec["design"], C.rubric_ref(rubric)
    arm_ids = [arm["id"] for arm in arms]
    A, D, J, I, K = len(arms), len(dims), len(judge_ids), spec["inputs"], spec["runs_per_input"]
    scored, paired = _coverage(plan, dims)
    relative = _relative(plan, dims)

    # Runs: every arm runs on every input.
    width = max(3, len(str(I)))
    inputs = [f"i{n + 1:0{width}d}" for n in range(I)]
    runs, run_arm, run_input, run_at = [], [], [], {}
    for i, name in enumerate(inputs):
        for a, arm in enumerate(arms):
            for k in range(K):
                run = f"{arm['id']}-{name}" + (f"-r{k + 1}" if K > 1 else "")
                row = {"run": run, "input": name, "arm": arm["id"], "steps": [dict(step) for step in arm["steps"]],
                       "simulated": True, "refs": {"input": f"inputs/{name}", "output": f"outputs/{run}"}}
                if "cost_usd" in arm:
                    row["cost_usd"] = arm["cost_usd"]
                run_at[(arm["id"], i, k)] = len(runs)
                runs.append(C.validate_run({"schema": C.SCHEMA_RUNS, **row}))
                run_arm.append(a)
                run_input.append(i)
    R = len(runs)
    run_arm, run_input = np.array(run_arm), np.array(run_input)
    families = sorted({step["model"]["family"] for arm in arms for step in arm["steps"]})

    # Hidden quality: q = mu_arm + v_input + u_run.
    mu = np.array([[arm["mean"][dim_id] for dim_id in dim_ids] for arm in arms])
    sd_input = np.array([spec["quality"]["sd_input"][dim_id] for dim_id in dim_ids])
    sd_run = np.array([spec["quality"]["sd_run"][dim_id] for dim_id in dim_ids])
    corr_input, corr_run = np.array(spec["quality"]["corr_input"]), np.array(spec["quality"]["corr_run"])
    cov_input, cov_run = Q.covariance(sd_input, corr_input), Q.covariance(sd_run, corr_run)
    rng = np.random.default_rng([seed, 1])
    v = rng.standard_normal((I, D)) @ _root(cov_input).T
    u = rng.standard_normal((R, D)) @ _root(cov_run).T
    q = mu[run_arm] + v[run_input] + u

    # Judge parameters as arrays. NaN marks a parameter the design gives no reading for.
    def per_dim(name: str, used: set) -> np.ndarray:
        out = np.full((J, D), np.nan)
        for j, t in enumerate(truths):
            for d, dim_id in enumerate(dim_ids):
                if dim_id in t[name] and (judge_ids[j], d) in used:
                    out[j, d] = t[name][dim_id]
        return out

    bias, sigma = per_dim("bias", scored), per_dim("sigma", scored)
    tau, sd_judge_input = per_dim("tau", scored | paired), per_dim("sd_judge_input", scored)
    sigma_pair = per_dim("sigma_pair", paired)
    scores_any = [any(jd[0] == name for jd in scored) for name in judge_ids]
    pairs_any = [any(jd[0] == name for jd in paired) for name in judge_ids]
    halo_target = np.array([t["halo_target"] if scores_any[j] or pairs_any[j] else np.nan
                            for j, t in enumerate(truths)])
    halo_call = np.array([t["halo_call"] if scores_any[j] and judges["judges"][j]["mode"] == "panel" else np.nan
                          for j, t in enumerate(truths)])
    position = np.array([t["position"] if pairs_any[j] else np.nan for j, t in enumerate(truths)])
    tie_width = np.array([t["tie_width"] if pairs_any[j] else np.nan for j, t in enumerate(truths)])
    family = np.full((J, len(families), D), np.nan)
    for j, t in enumerate(truths):
        if judges["judges"][j]["kind"] != "model":
            continue
        for d, dim_id in enumerate(dim_ids):
            if (judge_ids[j], d) in scored | paired:
                family[j, :, d] = [t["family"].get(name, {}).get(dim_id, 0.0) for name in families]
    arm_family = np.array([[families.index(_family_of(arm, dim["step"])) for dim in dims] for arm in arms])

    # Each judge's view of every target: q + bias + family (the centre), + judge-by-input + judge-by-output.
    rng = np.random.default_rng([seed, 2])
    centre, view = np.empty((J, R, D)), np.empty((J, R, D))
    for j, t in enumerate(truths):
        t_tau = np.array([t["tau"].get(dim_id, 0.0) for dim_id in dim_ids])
        t_bias = np.array([t["bias"].get(dim_id, 0.0) for dim_id in dim_ids])
        t_wsd = np.array([t["sd_judge_input"].get(dim_id, 0.0) for dim_id in dim_ids])
        effect = np.array([[t.get("family", {}).get(name, {}).get(dim_id, 0.0) for dim_id in dim_ids]
                           for name in families])                      # F x D
        g = effect[arm_family, np.arange(D)][run_arm]                    # R x D
        w = (rng.standard_normal((I, D)) * t_wsd)[run_input]
        shared, own = rng.standard_normal((R, len(steps))), rng.standard_normal((R, D))
        h = t["halo_target"]
        centre[j] = q + t_bias + g
        view[j] = centre[j] + w + t_tau * (math.sqrt(h) * shared[:, step_of] + math.sqrt(1.0 - h) * own)

    # Check items per input.
    items, items_of, item_truth = [], {}, []
    for n, row in enumerate(plan["items"]):
        rng = np.random.default_rng([seed, 6, n])
        low, high = row["per_input"]
        for name in inputs:
            if rng.random() < row["missing_share"]:
                continue
            count = int(rng.integers(low, high + 1))
            hard = rng.standard_normal(count) * row["item_sd"]
            ids = [f"{row['component']}-{name}-{m + 1:02d}" for m in range(count)]
            items_of[(name, row["component"])] = (ids, hard, row["as"])
            items += [C.validate_item({"schema": C.SCHEMA_ITEMS, "id": item, "input": name,
                                       "component": row["component"], "label": f"point {m + 1}",
                                       "source": {"kind": "expert"}}) for m, item in enumerate(ids)]
            if row["item_sd"] > 0:
                item_truth += [{"id": item, "difficulty": float(x)} for item, x in zip(ids, hard)]

    scores: list[dict] = []
    tally = {"calls": 0, "clipped": 0, "clock": 0}
    seen: dict[tuple[int, str], list[float]] = {}        # per target, what the judges said, on [0, 1]
    binary_log: dict[tuple[int, int], list[tuple[int, int]]] = {}
    cuts = np.full((D, max([len(dim["scale"]["levels"]) - 1 for dim in dims if dim["scale"]["kind"] == "levels"],
                           default=0)), np.nan)
    for d, dim in enumerate(dims):
        if kinds[d] == "levels":
            points = spec["quality"].get("cuts", {}).get(dim["id"])
            points = Q.fixed_cuts(len(dim["scale"]["levels"])) if points is None else points
            cuts[d, :len(points)] = points

    def stamp() -> str:
        tally["clock"] += 1
        return (START + _dt.timedelta(seconds=tally["clock"])).isoformat()

    def new_call(prefix: str = "c") -> str:
        tally["calls"] += 1
        return f"{prefix}{tally['calls']:06d}"

    def emit(rng: np.random.Generator, r: int, d: int, j: int, call: str, x: float, cost: float | None,
             person: bool) -> None:
        """Turn the latent reading `x` of dimension d into rows on the component's own scale."""
        dim, kind = dims[d], kinds[d]
        scale, lower = dim["scale"], dim["better"] == "lower"
        base = {"run": runs[r]["run"], "step": dim["step"], "component": dim["component"], "judge": judge_ids[j],
                "call": call, "rubric": ref}
        values: list[dict] = []
        if kind in ("linear", "fraction"):
            unit = min(1.0, max(0.0, x))
            tally["clipped"] += int(unit != x)
            shown, raw = unit, 1.0 - unit if lower else unit
            if kind == "linear":
                raw = scale["range"][0] + raw * (scale["range"][1] - scale["range"][0])
            values.append({"value": round(float(raw), ROUND)})
        elif kind == "levels":
            L = len(scale["levels"])
            level = int(np.searchsorted(cuts[d, :L - 1], x, side="right"))
            shown = level / (L - 1)
            values.append({"value": scale["levels"][L - 1 - level if lower else level]})
        elif kind == "binary":
            passed = int(x >= 0)
            shown = float(passed)
            binary_log.setdefault((j, d), []).append((r, passed))
            values.append({"value": 1 - passed if lower else passed})
        else:
            ids, hard, form = items_of[(runs[r]["input"], dim["component"])]
            passed = (rng.random(len(ids)) < ndtr(x - hard)).astype(int)
            shown = float(passed.mean())
            if lower:
                passed = 1 - passed
            if form == "count":
                values.append({"value": int(passed.sum()), "of": len(ids)})
            else:
                values += [{"value": int(p), "item": item} for item, p in zip(ids, passed)]
        if not person:
            seen.setdefault((r, dim["step"]), []).append(shown)
        for n, value in enumerate(values):
            row = {"id": f"s{len(scores) + 1:06d}", **base, **value, "observed_at": stamp()}
            if cost is not None and n == 0:
                row["cost_usd"] = cost
            scores.append(row)

    # Scores: the judges' rows first, then the humans', whose selection may read the judges' scores.
    human_truth = []
    for part, stream in (("scores", 3), ("human", 4)):
        for n, row in enumerate(plan[part]):
            rng = np.random.default_rng([seed, stream, n])
            j = judge_ids.index(row["judge"])
            judge, t = judges["judges"][j], truths[j]
            panel, person = judge["mode"] == "panel", judge["kind"] == "person"
            at_step = {step: [d for d in _covered(row, dims) if dims[d]["step"] == step] for step in row["steps"]}
            cand = [(r, step) for r in range(R) if arm_ids[run_arm[r]] in row["arms"]
                    for step in row["steps"] if at_step[step]]
            share = row["share"] if part == "scores" else row["rate"]
            if part == "human" and row["selection"] == "judge_scores":
                level = np.array([np.mean(seen[key]) if key in seen else np.nan for key in cand])
                known = np.isfinite(level)
                spread = float(level[known].std()) if known.any() else 0.0
                z = np.where(known, (level - level[known].mean()) / spread, 0.0) if spread > 0 else np.zeros(len(cand))
                weight = np.exp(-row["strength"] * z)
                prob = np.clip(share * weight / weight.mean(), share / 5.0, 1.0)
                chosen = np.flatnonzero(rng.random(len(cand)) < prob)
            else:
                size = min(len(cand), max(1, int(round(share * len(cand)))))
                prob = np.full(len(cand), size / len(cand))
                chosen = np.arange(len(cand)) if size == len(cand) else np.sort(
                    rng.choice(len(cand), size=size, replace=False))
            again: set[int] = set()
            if part == "human":
                extra = int(round(row["repeat_share"] * len(chosen)))
                again = set(rng.choice(chosen, size=extra, replace=False).tolist()) if extra else set()
                picked = set(chosen.tolist())
                human_truth.append({"judge": row["judge"], "selection": row["selection"], "rate": row["rate"],
                                    "targets": [{"run": runs[r]["run"], "step": step,
                                                 "inclusion_probability": float(prob[m]), "selected": m in picked,
                                                 "calls": int(m in picked) + int(m in again)}
                                                for m, (r, step) in enumerate(cand)]})
            cost = judge.get("cost_usd_per_call")
            for m in chosen.tolist():
                r, step = cand[m]
                here = [d for d in at_step[step]
                        if kinds[d] != "items" or (runs[r]["input"], dims[d]["component"]) in items_of]
                if not here:
                    continue
                for _ in range(row.get("repeats", 1) + int(m in again)):
                    call, common = (new_call(), rng.standard_normal()) if panel else ("", 0.0)
                    for n_d, d in enumerate(here):
                        s = t["sigma"][dim_ids[d]]
                        if panel:
                            hc = t["halo_call"]
                            e = s * (math.sqrt(hc) * common + math.sqrt(1.0 - hc) * rng.standard_normal())
                        else:
                            call, e = new_call(), s * rng.standard_normal()
                        emit(rng, r, d, j, call, float(view[j, r, d] + e), cost if not panel or n_d == 0 else None,
                             person)

    # Pairs: two outputs of one input, compared by one judge on one dimension.
    pairs: list[dict] = []
    pair_log: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for n, row in enumerate(plan["pairs"]):
        rng = np.random.default_rng([seed, 5, n])
        j = judge_ids.index(row["judge"])
        t = truths[j]
        d = dim_ids.index(C.dimension_id(row["step"], row["component"]))
        size = min(I, max(1, int(round(row["share"] * I))))
        chosen = list(range(I)) if size == I else np.sort(rng.choice(I, size=size, replace=False)).tolist()
        cost = judges["judges"][j].get("cost_usd_per_call")
        for i in chosen:
            for arm_a, arm_b in row["arms"]:
                for k in range(K):
                    ra, rb = run_at[(arm_a, i, k)], run_at[(arm_b, i, k)]
                    for _ in range(row["repeats"]):
                        orders = {"both": ("a", "b"), "a_first": ("a",), "b_first": ("b",), "unknown": (None,),
                                  "random": ("a" if rng.random() < 0.5 else "b",)}[row["orders"]]
                        for first in orders:
                            o = {"a": 1.0, "b": -1.0, None: 0.0}[first]
                            z = (view[j, ra, d] - view[j, rb, d] + t["position"] * o
                                 + t["sigma_pair"][dim_ids[d]] * rng.standard_normal())
                            verdict = "a" if z > t["tie_width"] else "b" if z < -t["tie_width"] else "tie"
                            pair_log.setdefault((j, d), []).append((verdict, float(q[ra, d] - q[rb, d])))
                            line = {"id": f"p{len(pairs) + 1:06d}", "a": runs[ra]["run"], "b": runs[rb]["run"],
                                    "step": row["step"], "component": row["component"], "judge": row["judge"],
                                    "call": new_call("pc"), "shown_first": first, "verdict": verdict, "rubric": ref,
                                    "observed_at": stamp()}
                            if cost is not None:
                                line["cost_usd"] = cost
                            pairs.append(line)

    # Every row goes through its contract here, so `write` cannot refuse half way.
    components = {comp["id"]: comp for comp in rubric["components"]}
    checked = []
    for n, row in enumerate(scores, 1):
        row = C.validate_score({"schema": C.SCHEMA_SCORES, **row}, C.SCORES_FILE, f"line {n}: ")
        C.check_reading(row, components[row["component"]], C.SCORES_FILE, f"line {n}: ")
        checked.append(row)
    scores = checked
    pairs = [C.validate_pair({"schema": C.SCHEMA_PAIRS, **row}, C.PAIRS_FILE, f"line {n}: ")
             for n, row in enumerate(pairs, 1)]
    value = spec.get("value")

    index = {"arms": arm_ids,
             "dimensions": [{"id": dim["id"], "step": dim["step"], "component": dim["component"], "kind": kinds[d],
                             "cuts": [float(x) for x in Q.fixed_cuts(len(dim["scale"]["levels"]))]
                             if kinds[d] == "levels" else None} for d, dim in enumerate(dims)],
             "judges": judge_ids, "families": families, "inputs": inputs, "runs": [run["run"] for run in runs],
             "run_arm": run_arm.tolist(), "run_input": run_input.tolist()}
    total_sd = np.sqrt(sd_input ** 2 + sd_run ** 2)
    truth: dict = {"schema": C.SCHEMA_TRUTH, "spec": spec, "seed": seed, "index": index,
                   "mu": mu, "sd_input": sd_input, "sd_run": sd_run, "corr_input": corr_input, "corr_run": corr_run,
                   "q_run": q, "v_input": v, "bias": bias, "tau": tau, "sigma": sigma,
                   "sd_judge_input": sd_judge_input, "sigma_pair": sigma_pair, "halo_target": halo_target,
                   "halo_call": halo_call, "position": position, "tie_width": tie_width, "family": family}
    if cuts.shape[1]:
        truth["cuts"] = cuts
    truth["reported"] = {"arms": np.stack([Q.arm_reported(mu[:, d], total_sd[d], kinds[d]) for d in range(D)], axis=1),
                         "runs": Q.reported_all(q, kinds)}
    truth["relative"] = relative
    truth["units"] = _units(spec, dims, scored, paired)
    truth["binary_rates"] = _binary_rates(binary_log, q, centre, truths, judge_ids, dim_ids)
    truth["pair_rates"] = [{"judge": judge_ids[j], "dimension": dim_ids[d], "n": len(log),
                            "tie_rate": sum(verdict == "tie" for verdict, _ in log) / len(log),
                            "picks_higher": sum(verdict == ("a" if gap > 0 else "b") for verdict, gap in log) / len(log)}
                           for (j, d), log in sorted(pair_log.items())]
    truth["overall"] = _overall_truth(rubric, dims, cov_input, cov_run, mu, value)
    truth["value"] = None if value is None else _value_truth(value, dims, relative, mu, cov_input, cov_run, q,
                                                             run_arm, arm_ids)
    truth["human"] = human_truth
    truth["items"] = item_truth
    truth["counts"] = {"inputs": I, "arms": A, "runs": R, "dimensions": D, "judges": J, "readings": len(scores),
                       "calls": tally["calls"], "pairs": len(pairs), "items": len(items),
                       "clipped": int(tally["clipped"])}
    return Simulated(spec, seed, rubric, judges, value, runs, scores, pairs, items, _clean(truth))


def _units(spec: dict, dims: list[dict], scored: set, paired: set) -> list[dict]:
    """Per dimension whose unit the data does not fix: the rule, and the judges whose noise is the unit."""
    anchors, out = set(spec["judges"]["anchor"]), []
    ids = [row["id"] for row in spec["judges"]["judges"]]
    for d, dim in enumerate(dims):
        readers = [name for name in ids if (name, d) in scored]
        pairers = [name for name in ids if (name, d) in paired]
        if readers and dim["scale"]["kind"] == "binary":
            out.append({"dimension": dim["id"], "rule": "binary", "parameter": "sigma",
                        "judges": [name for name in readers if name in anchors] or readers})
        elif pairers and not readers:
            out.append({"dimension": dim["id"], "rule": "pairs_only", "parameter": "sigma_pair",
                        "judges": [name for name in pairers if name in anchors] or pairers})
    return out


def _binary_rates(log: dict, q: np.ndarray, centre: np.ndarray, truths: list[dict], judge_ids: list[str],
                  dim_ids: list[str]) -> list[dict]:
    """Per judge and binary dimension, over the targets it read: the rates as simulated and as expected.

    `false_pass` is the share of calls that pass an output whose hidden status is fail, `false_fail` the
    reverse, `agreement` the share of calls that match the hidden status, `flip` the share of pairs of
    calls on one output that disagree. `simulated` counts the calls of this study. `expected` integrates
    the judge-by-input and judge-by-output terms and the call noise for each target the judge read,
    given the target's true quality."""
    nodes, weights = np.polynomial.hermite.hermgauss(48)
    weights = weights / math.sqrt(math.pi)
    out = []
    for (j, d), calls in sorted(log.items()):
        said = np.array([p for _, p in calls])
        status = q[np.array([r for r, _ in calls]), d] >= 0
        by_run: dict[int, list[int]] = {}
        for r, p in calls:
            by_run.setdefault(r, []).append(p)
        same = both = 0
        for values in by_run.values():
            n, k = len(values), sum(values)
            both += n * (n - 1) // 2
            same += k * (k - 1) // 2 + (n - k) * (n - k - 1) // 2
        t, dim_id = truths[j], dim_ids[d]
        targets = np.array(sorted(by_run))
        hidden = q[targets, d] >= 0
        spread = math.hypot(t["tau"][dim_id], t["sd_judge_input"][dim_id])
        x = centre[j, targets, d][:, None] + spread * math.sqrt(2.0) * nodes[None, :]
        s = t["sigma"][dim_id]
        p = ndtr(x / s) if s > 0 else (x >= 0).astype(float)
        p_pass, p_flip = p @ weights, (2.0 * p * (1.0 - p)) @ weights

        def rate(hits: np.ndarray) -> float | None:
            return float(np.mean(hits)) if np.size(hits) else None

        out.append({"judge": judge_ids[j], "dimension": dim_id, "n_calls": len(calls), "n_targets": len(by_run),
                    "n_fail": int((~hidden).sum()), "n_pass": int(hidden.sum()),
                    "simulated": {"false_pass": rate(said[~status] == 1), "false_fail": rate(said[status] == 0),
                                  "agreement": rate(said == status), "flip": (both - same) / both if both else None},
                    "expected": {"false_pass": rate(p_pass[~hidden]), "false_fail": rate(1.0 - p_pass[hidden]),
                                 "agreement": rate(np.where(hidden, p_pass, 1.0 - p_pass)), "flip": rate(p_flip)}})
    return out


def _overall_truth(rubric: dict, dims: list[dict], cov_input: np.ndarray, cov_run: np.ndarray, mu: np.ndarray,
                   value: dict | None) -> list[dict]:
    """Per step with an overall component and criteria: how much of the overall quality the criteria explain.

    On the fitting scale. `explained_share` is the R squared of the best linear predictor of the overall
    quality from the criteria, at the run level, the input level and both together; `best_weights` are that
    predictor's weights at both levels together. With weights in the value file (final step), `weights_share`
    is the squared correlation between the weighted criteria and the overall quality, at the same levels and,
    with 3 or more arms, across the arm means."""
    out = []
    ids = [dim["id"] for dim in dims]
    for step in rubric["steps"]:
        overall = C.overall_component(rubric, step)
        crit = [d for d, dim in enumerate(dims) if dim["step"] == step and dim["role"] == "criterion"]
        if overall is None or not crit:
            continue
        o = ids.index(C.dimension_id(step, overall))
        levels = {"run": cov_run, "input": cov_input, "total": cov_input + cov_run}

        def share(cov: np.ndarray, weights: np.ndarray | None = None) -> float | None:
            if cov[o, o] <= 0:
                return None
            cc, co = cov[np.ix_(crit, crit)], cov[crit, o]
            if weights is None:
                return float(co @ np.linalg.pinv(cc) @ co / cov[o, o])
            spread = float(weights @ cc @ weights)
            return float((weights @ co) ** 2 / (spread * cov[o, o])) if spread > 0 else None

        total = levels["total"]
        best = np.linalg.pinv(total[np.ix_(crit, crit)]) @ total[crit, o]
        entry = {"step": step, "overall": ids[o], "criteria": [ids[d] for d in crit],
                 "explained_share": {name: share(cov) for name, cov in levels.items()},
                 "best_weights": {ids[d]: float(x) for d, x in zip(crit, best)}}
        if value is not None and step == C.FINAL:
            given = np.array([float(value["weights"].get(dims[d]["component"], 0.0)) for d in crit])
            if given.any():
                entry["weights_share"] = {name: share(cov, given) for name, cov in levels.items()}
                if len(mu) >= 3 and np.std(mu[:, crit] @ given) > 0 and np.std(mu[:, o]) > 0:
                    entry["weights_share"]["arms"] = float(np.corrcoef(mu[:, crit] @ given, mu[:, o])[0, 1] ** 2)
        out.append(entry)
    return out


def _value_truth(value: dict, dims: list[dict], relative: list[str], mu: np.ndarray, cov_input: np.ndarray,
                 cov_run: np.ndarray, q: np.ndarray, run_arm: np.ndarray, arm_ids: list[str]) -> dict:
    """The true value and gate pass probability of each arm, for a new run on a new input.

    `population` integrates over `VALUE_DRAWS` new inputs (shared by the arms) and runs with a fixed
    generator, so it depends on the spec only. `p_pass_independent` multiplies the gates' own pass
    probabilities: what a model that ignores correlation would say. `value_minus_first` is the paired
    difference to the first arm. `study` is the same over the runs of this study, from their true hidden
    quality."""
    spec = Q.compile_value(value, [{**dim, "relative": dim["id"] in relative} for dim in dims])
    kinds = [dim["scale"]["kind"] for dim in dims]
    rng = np.random.default_rng(VALUE_SEED)
    D = len(dims)
    v = rng.standard_normal((VALUE_DRAWS, D)) @ _root(cov_input).T
    root_run = _root(cov_run)
    population, study, first = [], [], None
    for a, arm in enumerate(arm_ids):
        m = Q.reported_all(mu[a] + v + rng.standard_normal((VALUE_DRAWS, D)) @ root_run.T, kinds)
        each = spec.passes_each(m)
        worth = spec.value(m)
        first = worth if first is None else first
        population.append({"arm": arm, "p_pass": float(each.all(axis=-1).mean()),
                           "p_gate": [float(x) for x in each.mean(axis=0)],
                           "p_pass_independent": float(np.prod(each.mean(axis=0))),
                           "value": float(worth.mean()), "value_sd": float(worth.std()),
                           "weighted": float(spec.weighted(m).mean()),
                           "value_minus_first": float((worth - first).mean())})
        mine = Q.reported_all(q[run_arm == a], kinds)
        study.append({"arm": arm, "n_runs": int(len(mine)), "pass_rate": float(spec.passes(mine).mean()),
                      "value": float(spec.value(mine).mean())})
    return {"gates": list(spec.gate_labels), "draws": VALUE_DRAWS, "population": population, "study": study}


def example_spec() -> dict:
    """A small spec that uses every member: two steps, every scale kind, pairs, items, a value file."""
    small = {"model": {"id": "model-small", "family": "beta"}, "effort": "low"}
    large = {"model": {"id": "model-large", "family": "alpha"}, "effort": "high", "variant": "prompt-2"}
    scored = ["numbers_right", "clarity", "depth", "coverage"]
    return {
        "schema": C.SCHEMA_SIM, "name": "example", "title": "a small study with every kind of reading",
        "inputs": 6, "runs_per_input": 1,
        "arms": [
            {"id": "baseline", "steps": [{"step": "draft", **small}, {"step": "final", **small}], "cost_usd": 0.1,
             "mean": {"numbers_right": 0.8, "clarity": 0.4, "final/clarity": 0.45, "depth": 0.45, "coverage": 0.2,
                      "overall": -0.4}},
            {"id": "strong", "steps": [{"step": "draft", **large}, {"step": "final", **large}], "cost_usd": 0.8,
             "mean": {"numbers_right": 2.0, "clarity": 0.6, "depth": 0.55, "coverage": 0.8, "overall": 0.4}}],
        "rubric": {"id": "brief", "version": "1", "steps": ["draft", "final"], "components": [
            {"id": "numbers_right", "title": "The numbers are right", "scale": {"kind": "binary"}},
            {"id": "clarity", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]},
             "applies_to": ["draft", "final"]},
            {"id": "depth", "scale": {"kind": "linear", "range": [0, 10]}},
            {"id": "coverage", "scale": {"kind": "items"}},
            {"id": "overall", "role": "overall", "scale": {"kind": "linear", "range": [0, 10]}}]},
        "quality": {
            "sd_input": {"numbers_right": 1.2, "clarity": 0.18, "depth": 0.07, "coverage": 0.4, "overall": 0.6},
            "sd_run": {"numbers_right": 1.0, "clarity": 0.15, "depth": 0.05, "coverage": 0.3, "overall": 0.8},
            "corr_input": 0.3, "corr_run": {"default": 0.2, "pairs": [["depth", "overall", 0.5]]},
            "cuts": {"draft/clarity": [0.3, 0.7]}},
        "judges": {"anchor": ["reviewer"], "min_anchor_targets": 6, "judges": [
            {"id": "reviewer", "kind": "person", "blinded": True,
             "truth": {"tau": {"numbers_right": 0.5, "clarity": 0.06, "depth": 0.03, "coverage": 0.15, "overall": 0.4},
                       "sigma": {"numbers_right": 1.0, "clarity": 0.08, "depth": 0.04, "coverage": 0.2},
                       "sigma_pair": {"overall": 1.0}, "position": 0.2, "tie_width": 0.5}},
            {"id": "strong", "kind": "model", "model": {"id": "model-large", "family": "alpha"}, "effort": "high",
             "mode": "panel", "cost_usd_per_call": 0.04, "components": scored,
             "truth": {"bias": {"numbers_right": 0.2, "clarity": 0.04, "depth": 0.03, "coverage": 0.1},
                       "tau": {"numbers_right": 0.8, "clarity": 0.08, "depth": 0.05, "coverage": 0.2},
                       "sigma": {"numbers_right": 0.8, "clarity": 0.08, "depth": 0.04, "coverage": 0.2},
                       "halo_target": 0.2, "halo_call": 0.4, "family": {"alpha": {"depth": 0.05}}}},
            {"id": "cheap", "kind": "model", "model": {"id": "model-small", "family": "beta"}, "effort": "low",
             "cost_usd_per_call": 0.004,
             "truth": {"bias": {"depth": 0.08}, "tau": {"depth": 0.08, "overall": 0.7}, "sigma": {"depth": 0.07},
                       "sd_judge_input": {"depth": 0.03}, "sigma_pair": {"overall": 1.3}, "position": 0.5,
                       "tie_width": 0.3}}]},
        "design": {
            "scores": [{"judge": "strong", "repeats": 2},
                       {"judge": "cheap", "repeats": 2, "share": 0.5, "components": ["depth"], "steps": ["final"]}],
            "human": [{"judge": "reviewer", "rate": 0.5, "selection": "judge_scores", "strength": 1.0,
                       "repeat_share": 0.5, "components": scored}],
            "pairs": [{"judge": "reviewer", "orders": "both"},
                      {"judge": "cheap", "component": "overall", "step": "final", "arms": [["baseline", "strong"]],
                       "share": 0.5, "orders": "random", "repeats": 2}],
            "items": [{"component": "coverage", "per_input": [2, 4], "missing_share": 0.1, "as": "items"}]},
        "value": {"weights": {"clarity": 1, "depth": 2}, "gates": [{"component": "numbers_right"}],
                  "gated_value": 0, "objective": {"kind": "best_value"}},
    }


# ---------------------------------------------------------------------------------------------------
# Files

def read_truth(path: str | os.PathLike) -> dict:
    """Read `truth.json`. The arrays named in `ARRAYS` come back as numpy arrays, null as NaN."""
    path = Path(path)
    obj = C.load_json(path)
    found = obj.get("schema") if isinstance(obj, dict) else None
    if found != C.SCHEMA_TRUTH:
        raise C.ContractError(path.name, "schema", "missing" if found is None else f'unknown schema "{found}"',
                              f'this stage reads "{C.SCHEMA_TRUTH}"',
                              "missing_schema" if found is None else "unknown_schema")
    for name in ARRAYS:
        if name in obj:
            obj[name] = np.array(obj[name], dtype=float)
    obj["reported"] = {name: np.array(rows, dtype=float) for name, rows in obj["reported"].items()}
    return obj


def study_files(sim: Simulated) -> dict[str, str]:
    """The text of every file `write` creates, by file name, in the order they are written."""
    def lines(rows: list[dict]) -> str:
        return "".join(json.dumps({k: v for k, v in row.items() if v is not None}, ensure_ascii=False,
                                  allow_nan=False) + "\n" for row in rows)

    out = {"rubric.json": C.dumps(sim.rubric), "judges.json": C.dumps(sim.judges)}
    if sim.value is not None:
        out["value.json"] = C.dumps(sim.value)
    out[C.RUNS_FILE] = lines(sim.runs)
    for name, rows in ((C.SCORES_FILE, sim.scores), (C.PAIRS_FILE, sim.pairs), (C.ITEMS_FILE, sim.items)):
        if rows:
            out[name] = lines(rows)
    out[TRUTH_FILE] = C.dumps(sim.truth)
    return out


def write(sim: Simulated, folder: str | os.PathLike) -> dict[str, Path]:
    """Write a simulated study into a folder: file name to path. A folder that holds one of the files is refused."""
    folder = Path(folder)
    texts = study_files(sim)
    if folder.exists() and not folder.is_dir():
        raise C.ContractError(str(folder), "", "not a folder", "name a folder for the study", "not_a_folder")
    for name in texts:
        if (folder / name).exists():
            raise C.ContractError(str(folder), "", f"the folder already holds {name}",
                                  "give a new or empty folder: a simulated study is never mixed into another",
                                  "exists")
    return {name: C.write_text_atomic(folder / name, text) for name, text in texts.items()}


def load_spec(name: str) -> tuple[dict, str]:
    """A spec by file path or catalog name, normalised, and where it came from (`file` or `catalog`)."""
    from . import specs

    if Path(name).is_file():
        return validate_spec(C.load_json(name), Path(name).name), "file"
    if name in specs.names():
        return validate_spec(specs.get(name), f"catalog spec {name}"), "catalog"
    raise C.ContractError(name, "", "no such spec file or catalog spec",
                          f"give a spec file, or one of {', '.join(specs.names())}", "no_spec")


def run(args: argparse.Namespace) -> int:
    from . import specs

    if getattr(args, "list", False):
        catalog = [{"name": name, "title": specs.title(name)} for name in specs.names()]
        if args.json:
            emit_json("loopmath.eval-simulate/1", {"catalog": catalog})
        else:
            width = max(len(row["name"]) for row in catalog)
            for row in catalog:
                print(f"{row['name']:<{width}}  {row['title']}")
        return EXIT_OK
    if args.spec is None:
        raise C.ContractError("eval simulate", "", "no spec given",
                              "name a spec file or a catalog spec (--list shows the catalog)", "no_spec")
    if args.out is None:
        raise C.ContractError("eval simulate", "", "no --out given", "name the study folder to write", "no_out")
    spec, source = load_spec(args.spec)
    sim = generate(spec, args.seed, Path(args.spec).name if source == "file" else f"catalog spec {args.spec}")
    paths = write(sim, args.out)
    counts = sim.truth["counts"]
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if args.json:
        emit_json("loopmath.eval-simulate/1", {
            "spec": {"name": spec["name"], "source": source, "sha256": digest}, "seed": args.seed,
            "out": str(args.out), "simulated": True, "counts": counts,
            "files": {name: {"file": name, "sha256": C.sha256_file(path)} for name, path in paths.items()}})
    else:
        print(f"simulated study {spec['name']} (seed {args.seed}) in {args.out}")
        print(f"  {counts['runs']} runs: {counts['arms']} arms on {counts['inputs']} inputs, "
              f"{counts['dimensions']} dimensions")
        print(f"  {counts['readings']} readings, {counts['pairs']} pairs, {counts['items']} items, "
              f"{counts['judges']} judges")
        print(f"  files: {', '.join(paths)}")
        print(f"  the truth is in {Path(args.out) / TRUTH_FILE}; every run is marked simulated")
    return EXIT_OK
