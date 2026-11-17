"""Data contracts of `loopmath eval`: schemas, validation, provenance, reading and writing files.

Every file of the group has a `schema` member, `loopmath.eval-<name>/1`. In a JSON Lines file every line
carries it. A stage refuses an unknown schema by name, refuses unknown members by path (except inside
`ext`), and writes nothing on a refusal.

A refusal is a `ContractError`. Its text is `<file>: <place>: <problem>; <fix>`, and it has a stable
`code` that other packages can test without pinning the text. `commands.py` turns it into the usual
`error:` line and exit code 1, so a module can simply let it rise.

The validators return a normalised copy with every default filled in (`step` is `final`, `role` is
`criterion`, and so on). Validating a normalised object again returns it unchanged.

This module imports the standard library only.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA_RUBRIC = "loopmath.eval-rubric/1"
SCHEMA_ITEMS = "loopmath.eval-items/1"
SCHEMA_JUDGES = "loopmath.eval-judges/1"
SCHEMA_RUNS = "loopmath.eval-runs/1"
SCHEMA_SCORES = "loopmath.eval-scores/1"
SCHEMA_PAIRS = "loopmath.eval-pairs/1"
SCHEMA_VALUE = "loopmath.eval-value/1"
SCHEMA_FIT = "loopmath.eval-fit/1"
SCHEMA_LABELS = "loopmath.eval-labels/1"
SCHEMA_SIM = "loopmath.eval-sim/1"      # the simulate spec; its members belong to simulate.py
SCHEMA_TRUTH = "loopmath.eval-truth/1"  # truth.json of a simulated study; its members belong to simulate.py

# The files of a study folder.
RUNS_FILE = "runs.jsonl"
SCORES_FILE = "scores.jsonl"
PAIRS_FILE = "pairs.jsonl"
ITEMS_FILE = "items.jsonl"

FINAL = "final"
SCALE_KINDS = ("binary", "levels", "linear", "fraction", "items")
ROLES = ("criterion", "overall")
BETTER = ("higher", "lower")
JUDGE_KINDS = ("model", "person", "command")
JUDGE_MODES = ("per_component", "panel")
AUTHOR_KINDS = ("model", "person", "verified")
ITEM_SOURCE_KINDS = ("expert", "model", "model_verified")
VERDICTS = ("a", "b", "tie")
OBJECTIVE_KINDS = ("best_value", "cheapest_within_margin", "best_under_cost", "cost_per_pass")
ASKS = ("score", "pair")
ASSIGNMENT_KINDS = ("pilot", "active", "uniform", "repeat", "overlap")
LABEL_MODES = ("pilot", "active")
LABEL_SCORES = ("judge_uncertainty", "decision_weight", "component_gain", "per_minute")
DEFAULT_MIN_ANCHOR_TARGETS = 6
WEAK_ANCHOR_BELOW = 30


class ContractError(Exception):
    """A refusal: the file, the place in it, the problem and the fix."""

    def __init__(self, file: str, where: str, problem: str, fix: str, code: str) -> None:
        self.file, self.where, self.problem, self.fix, self.code = file, where, problem, fix, code
        place = f"{file}: {where}" if where else file
        super().__init__(f"{place}: {problem}; {fix}")


# ---------------------------------------------------------------------------------------------------
# The small checker every validator uses. `where` is a path such as `components[1].scale.kind`; for a
# line of a JSON Lines file it starts with `line 3: `.

def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def _join(base: str, name: str) -> str:
    if not base:
        return name
    return f"{base}{name}" if base.endswith(": ") else f"{base}.{name}"


class _Check:
    def __init__(self, file: str) -> None:
        self.file = file

    def fail(self, where: str, problem: str, fix: str, code: str) -> ContractError:
        return ContractError(self.file, where.removesuffix(": "), problem, fix, code)

    def obj(self, value: Any, where: str, allowed: Iterable[str], required: Iterable[str] = ()) -> dict:
        if not isinstance(value, dict):
            raise self.fail(where, f"expected an object, got {_kind(value)}", "write a JSON object here", "type")
        allowed = set(allowed) | {"ext"}
        for name in value:
            if name not in allowed:
                raise self.fail(_join(where, str(name)), "unknown member",
                                'remove it, or move it under "ext"', "unknown_member")
        for name in required:
            if name not in value or value[name] is None:
                raise self.fail(where, f'missing member "{name}"', "add it", "missing_member")
        if "ext" in value and not isinstance(value["ext"], dict):
            raise self.fail(_join(where, "ext"), f"expected an object, got {_kind(value['ext'])}",
                            "write a JSON object here", "type")
        return value

    def string(self, value: Any, where: str) -> str:
        if not isinstance(value, str) or not value:
            raise self.fail(where, f"expected a non-empty string, got {_kind(value)}", "write a string here", "type")
        return value

    def ident(self, value: Any, where: str) -> str:
        """An id: a non-empty string with no space, `/` or `@`, so `step/component` and `id@version` stay readable."""
        self.string(value, where)
        if any(ch.isspace() or ch in "/@" for ch in value):
            raise self.fail(where, f'the id "{value}" holds a space, "/" or "@"',
                            'use letters, digits, "-", "_" and "."', "bad_id")
        return value

    def number(self, value: Any, where: str, *, low: float | None = None, high: float | None = None,
               above: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise self.fail(where, f"expected a finite number, got {_kind(value)}", "write a number here", "type")
        if low is not None and value < low or high is not None and value > high or above is not None and value <= above:
            bound = (f"above {_num(above)}" if above is not None else
                     f"from {_num(low)} to {_num(high)}" if low is not None and high is not None else
                     f"at least {_num(low)}" if low is not None else f"at most {_num(high)}")
            raise self.fail(where, f"{_num(value)} is out of range", f"give a number {bound}", "out_of_range")
        return value

    def integer(self, value: Any, where: str, *, low: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.fail(where, f"expected a whole number, got {_kind(value)}", "write a whole number here", "type")
        if low is not None and value < low:
            raise self.fail(where, f"{value} is out of range", f"give a whole number at least {low}", "out_of_range")
        return value

    def boolean(self, value: Any, where: str) -> bool:
        if not isinstance(value, bool):
            raise self.fail(where, f"expected true or false, got {_kind(value)}", "write true or false", "type")
        return value

    def choice(self, value: Any, where: str, choices: Iterable[str]) -> str:
        choices = tuple(choices)
        if not isinstance(value, str) or value not in choices:
            shown = f'"{value}"' if isinstance(value, str) else _kind(value)
            raise self.fail(where, f"{shown} is not one of {', '.join(choices)}", "use one of them", "bad_choice")
        return value

    def array(self, value: Any, where: str, *, non_empty: bool = False) -> list:
        if not isinstance(value, list):
            raise self.fail(where, f"expected a list, got {_kind(value)}", "write a JSON list here", "type")
        if non_empty and not value:
            raise self.fail(where, "the list is empty", "give at least one entry", "empty")
        return value

    def unique(self, values: list, where: str, what: str) -> None:
        seen: set = set()
        for value in values:
            if value in seen:
                raise self.fail(where, f'{what} "{value}" appears twice', f"give every {what} once", "duplicate")
            seen.add(value)

    def timestamp(self, value: Any, where: str) -> str:
        self.string(value, where)
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = _dt.datetime.fromisoformat(text)
        except ValueError:
            parsed = None
        if parsed is None or parsed.tzinfo is None:
            raise self.fail(where, f'"{value}" is not an ISO 8601 time with an offset',
                            "write it like 2026-01-15T09:30:00-08:00", "bad_time")
        return value


def _num(value: Any) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def _schema(check: _Check, obj: Any, where: str, expected: str) -> None:
    if not isinstance(obj, dict):
        raise check.fail(where, f"expected an object, got {_kind(obj)}", "write a JSON object here", "type")
    found = obj.get("schema")
    if found is None:
        raise check.fail(_join(where, "schema"), "missing", f'add "schema": "{expected}"', "missing_schema")
    if found != expected:
        raise check.fail(_join(where, "schema"), f'unknown schema "{found}"', f'this stage reads "{expected}"',
                         "unknown_schema")


def _ext(obj: dict, out: dict) -> dict:
    if "ext" in obj:
        out["ext"] = obj["ext"]
    return out


# ---------------------------------------------------------------------------------------------------
# Rubric

def validate_rubric(obj: Any, file: str = "rubric.json") -> dict:
    """A rubric, normalised. Members: `id`, `version` (a string), `components[]`, `steps[]` (default `["final"]`)."""
    c = _Check(file)
    _schema(c, obj, "", SCHEMA_RUBRIC)
    c.obj(obj, "", {"schema", "id", "version", "components", "steps"}, {"id", "version", "components"})
    version = obj["version"]
    if isinstance(version, int) and not isinstance(version, bool):
        version = str(version)
    out: dict = {"schema": SCHEMA_RUBRIC, "id": c.ident(obj["id"], "id"), "version": c.ident(version, "version")}
    steps = obj.get("steps")
    if steps is None:
        steps = [FINAL]
    c.array(steps, "steps", non_empty=True)
    for n, step in enumerate(steps):
        c.ident(step, f"steps[{n}]")
    c.unique(steps, "steps", "step")
    if steps[-1] != FINAL:
        raise c.fail("steps", f'the last step is "{steps[-1]}"', 'end the list with "final"', "steps_end")
    out["steps"] = list(steps)
    components = []
    overall_at: dict[str, str] = {}
    for n, comp in enumerate(c.array(obj["components"], "components", non_empty=True)):
        w = f"components[{n}]"
        c.obj(comp, w, {"id", "title", "role", "scale", "better", "applies_to", "anchors", "review_minutes",
                        "compare_minutes", "human_only"}, {"id", "scale"})
        row: dict = {"id": c.ident(comp["id"], f"{w}.id")}
        row["title"] = c.string(comp["title"], f"{w}.title") if comp.get("title") is not None else row["id"]
        row["role"] = c.choice(comp.get("role", "criterion"), f"{w}.role", ROLES)
        row["scale"] = _scale(c, comp["scale"], f"{w}.scale")
        row["better"] = c.choice(comp.get("better", "higher"), f"{w}.better", BETTER)
        applies = comp.get("applies_to")
        if applies is None:
            applies = [FINAL]
        c.array(applies, f"{w}.applies_to", non_empty=True)
        for m, step in enumerate(applies):
            if step not in steps:
                shown = f'"{step}"' if isinstance(step, str) else _kind(step)
                raise c.fail(f"{w}.applies_to[{m}]", f"unknown step {shown}",
                             f"use one of {', '.join(steps)}, or add the step to steps", "unknown_step")
        c.unique(applies, f"{w}.applies_to", "step")
        row["applies_to"] = list(applies)
        if comp.get("anchors") is not None:
            if not isinstance(comp["anchors"], (str, dict)):
                raise c.fail(f"{w}.anchors", f"expected a string or an object, got {_kind(comp['anchors'])}",
                             "give a reference to the anchor examples", "type")
            row["anchors"] = comp["anchors"]
        row["review_minutes"] = c.number(comp.get("review_minutes", 1.0), f"{w}.review_minutes", above=0)
        row["compare_minutes"] = c.number(comp.get("compare_minutes", row["review_minutes"] / 2),
                                          f"{w}.compare_minutes", above=0)
        row["human_only"] = c.boolean(comp.get("human_only", False), f"{w}.human_only")
        if row["role"] == "overall":
            for step in row["applies_to"]:
                if step in overall_at:
                    raise c.fail(f"{w}.role", f'step "{step}" already has the overall component "{overall_at[step]}"',
                                 "keep one overall component per step", "two_overall")
                overall_at[step] = row["id"]
        components.append(_ext(comp, row))
    c.unique([row["id"] for row in components], "components", "component id")
    out["components"] = components
    return _ext(obj, out)


def _scale(c: _Check, scale: Any, w: str) -> dict:
    c.obj(scale, w, {"kind", "levels", "range"}, {"kind"})
    kind = c.choice(scale["kind"], f"{w}.kind", SCALE_KINDS)
    out: dict = {"kind": kind}
    for name in ("levels", "range"):
        if name in scale and name != {"levels": "levels", "linear": "range"}.get(kind):
            raise c.fail(f"{w}.{name}", f'a "{kind}" scale has no {name}', "remove it", "scale_member")
    if kind == "levels":
        levels = c.array(scale.get("levels"), f"{w}.levels")
        if len(levels) == 2:
            raise c.fail(f"{w}.levels", "2 levels given", 'use scale.kind "binary" for a yes or no check',
                         "two_levels")
        if len(levels) < 3:
            raise c.fail(f"{w}.levels", f"{len(levels)} levels given", "list 3 or more levels, lowest first",
                         "few_levels")
        for n, level in enumerate(levels):
            if isinstance(level, bool) or not isinstance(level, (str, int)) or level == "":
                raise c.fail(f"{w}.levels[{n}]", f"expected a string or a whole number, got {_kind(level)}",
                             "name the level", "type")
        c.unique(levels, f"{w}.levels", "level")
        out["levels"] = list(levels)
    if kind == "linear":
        rng = c.array(scale.get("range"), f"{w}.range")
        if len(rng) != 2:
            raise c.fail(f"{w}.range", f"{len(rng)} numbers given", "give [low, high]", "bad_range")
        low, high = c.number(rng[0], f"{w}.range[0]"), c.number(rng[1], f"{w}.range[1]")
        if not low < high:
            raise c.fail(f"{w}.range", f"low {_num(low)} is not below high {_num(high)}", "give [low, high]",
                         "bad_range")
        out["range"] = [low, high]
    return _ext(scale, out)


def rubric_ref(rubric: dict) -> str:
    """`id@version`, as a score row names its rubric."""
    return f"{rubric['id']}@{rubric['version']}"


def component(rubric: dict, component_id: str) -> dict | None:
    for row in rubric["components"]:
        if row["id"] == component_id:
            return row
    return None


def dimension_id(step: str, component_id: str) -> str:
    return f"{step}/{component_id}"


def dimensions(rubric: dict, components: Iterable[str] | None = None) -> list[dict]:
    """The dimensions of a normalised rubric, in model order: steps in run order, components in rubric order.

    Each is `{"id": "step/component", "step", "component", "title", "role", "scale", "better",
    "review_minutes", "compare_minutes", "human_only"}`. `components` keeps a subset (`eval fit --components`)."""
    keep = None if components is None else set(components)
    out = []
    for step in rubric["steps"]:
        for comp in rubric["components"]:
            if step in comp["applies_to"] and (keep is None or comp["id"] in keep):
                out.append({"id": dimension_id(step, comp["id"]), "step": step, "component": comp["id"],
                            "title": comp["title"], "role": comp["role"], "scale": dict(comp["scale"]),
                            "better": comp["better"], "review_minutes": comp["review_minutes"],
                            "compare_minutes": comp["compare_minutes"], "human_only": comp["human_only"]})
    return out


def overall_component(rubric: dict, step: str = FINAL) -> str | None:
    """The id of the step's `overall` component: the default component of a pair."""
    for comp in rubric["components"]:
        if comp["role"] == "overall" and step in comp["applies_to"]:
            return comp["id"]
    return None


# ---------------------------------------------------------------------------------------------------
# Judges

def validate_judges(obj: Any, file: str = "judges.json") -> dict:
    """A judges file, normalised. `components` of a judge is None when it scores every component."""
    c = _Check(file)
    _schema(c, obj, "", SCHEMA_JUDGES)
    c.obj(obj, "", {"schema", "judges", "anchor", "min_anchor_targets"}, {"judges"})
    judges = []
    for n, judge in enumerate(c.array(obj["judges"], "judges", non_empty=True)):
        w = f"judges[{n}]"
        c.obj(judge, w, {"id", "kind", "model", "effort", "prompt", "mode", "reference", "blinded",
                         "cost_usd_per_call", "components"}, {"id", "kind"})
        row: dict = {"id": c.ident(judge["id"], f"{w}.id"), "kind": c.choice(judge["kind"], f"{w}.kind", JUDGE_KINDS)}
        if row["kind"] == "model":
            if judge.get("model") is None:
                raise c.fail(w, 'a judge of kind "model" has no model', "add model with id and family",
                             "missing_member")
            row["model"] = _model(c, judge["model"], f"{w}.model", provider=True)
        elif judge.get("model") is not None:
            raise c.fail(f"{w}.model", f'a judge of kind "{row["kind"]}" has a model',
                         'remove it, or set kind to "model"', "judge_model")
        for name in ("effort", "prompt"):
            if judge.get(name) is not None:
                row[name] = c.string(judge[name], f"{w}.{name}")
        row["mode"] = c.choice(judge.get("mode", "per_component"), f"{w}.mode", JUDGE_MODES)
        row["reference"] = _reference(c, judge.get("reference"), f"{w}.reference")
        row["blinded"] = c.boolean(judge.get("blinded", False), f"{w}.blinded")
        if judge.get("cost_usd_per_call") is not None:
            row["cost_usd_per_call"] = c.number(judge["cost_usd_per_call"], f"{w}.cost_usd_per_call", low=0)
        row["components"] = None
        if judge.get("components") is not None:
            listed = c.array(judge["components"], f"{w}.components", non_empty=True)
            for m, name in enumerate(listed):
                c.ident(name, f"{w}.components[{m}]")
            c.unique(listed, f"{w}.components", "component")
            row["components"] = list(listed)
        judges.append(_ext(judge, row))
    ids = [row["id"] for row in judges]
    c.unique(ids, "judges", "judge id")
    anchor = obj.get("anchor")
    if anchor is None:
        anchor = []
    c.array(anchor, "anchor")
    for n, name in enumerate(anchor):
        if name not in ids:
            shown = f'"{name}"' if isinstance(name, str) else _kind(name)
            raise c.fail(f"anchor[{n}]", f"unknown judge {shown}", "name a judge id from judges", "unknown_judge")
    c.unique(anchor, "anchor", "judge")
    out = {"schema": SCHEMA_JUDGES, "judges": judges, "anchor": list(anchor),
           "min_anchor_targets": c.integer(obj.get("min_anchor_targets", DEFAULT_MIN_ANCHOR_TARGETS),
                                           "min_anchor_targets", low=1)}
    return _ext(obj, out)


def _model(c: _Check, model: Any, w: str, *, provider: bool = False) -> dict:
    c.obj(model, w, {"id", "family"} | ({"provider"} if provider else set()), {"id", "family"})
    out = {"id": c.string(model["id"], f"{w}.id"), "family": c.ident(model["family"], f"{w}.family")}
    if provider and model.get("provider") is not None:
        out["provider"] = c.string(model["provider"], f"{w}.provider")
    return _ext(model, out)


def _reference(c: _Check, ref: Any, w: str) -> dict:
    if ref is None:
        return {"shown": False}
    c.obj(ref, w, {"shown", "author", "version"})
    out: dict = {"shown": c.boolean(ref.get("shown", False), f"{w}.shown")}
    if out["shown"] and ref.get("author") is None:
        raise c.fail(w, "a shown reference has no author", "add author with kind (model, person or verified)",
                     "missing_member")
    if ref.get("author") is not None:
        author = c.obj(ref["author"], f"{w}.author", {"kind", "model"}, {"kind"})
        kind = c.choice(author["kind"], f"{w}.author.kind", AUTHOR_KINDS)
        out["author"] = {"kind": kind}
        if kind in ("model", "verified") and author.get("model") is None:
            raise c.fail(f"{w}.author", f'an author of kind "{kind}" has no model', "add model with id and family",
                         "missing_member")
        if author.get("model") is not None:
            out["author"]["model"] = _model(c, author["model"], f"{w}.author.model")
        _ext(author, out["author"])
    if ref.get("version") is not None:
        out["version"] = c.string(ref["version"], f"{w}.version")
    return _ext(ref, out)


def judge(judges: dict, judge_id: str) -> dict | None:
    for row in judges["judges"]:
        if row["id"] == judge_id:
            return row
    return None


# ---------------------------------------------------------------------------------------------------
# Value

def validate_value(obj: Any, file: str = "value.json") -> dict:
    """A value file, normalised. Checks against a rubric or a fit are `quality.compile_value`."""
    c = _Check(file)
    _schema(c, obj, "", SCHEMA_VALUE)
    c.obj(obj, "", {"schema", "weights", "gates", "gated_value", "objective"})
    weights = obj.get("weights")
    if weights is None:
        weights = {}
    if not isinstance(weights, dict):
        raise c.fail("weights", f"expected an object, got {_kind(weights)}", "map component ids to weights", "type")
    for name, weight in weights.items():
        c.ident(name, f"weights.{name}")
        c.number(weight, f"weights.{name}", low=0)
    gates = []
    raw_gates = obj.get("gates")
    for n, gate in enumerate(c.array([] if raw_gates is None else raw_gates, "gates")):
        w = f"gates[{n}]"
        c.obj(gate, w, {"component", "at_least", "step"}, {"component"})
        row: dict = {"component": c.ident(gate["component"], f"{w}.component"),
                     "step": c.ident(gate.get("step", FINAL), f"{w}.step")}
        at_least = gate.get("at_least")
        if at_least is not None:
            if isinstance(at_least, bool) or not isinstance(at_least, (int, float, str)) or (
                    isinstance(at_least, float) and not math.isfinite(at_least)):
                raise c.fail(f"{w}.at_least", f"expected a number or a level, got {_kind(at_least)}",
                             "give the threshold on the component's own scale", "type")
            row["at_least"] = at_least
        gates.append(_ext(gate, row))
    if not weights and not gates:
        raise c.fail("", "no weight and no gate", "add weights, gates or both", "empty_value")
    objective = obj.get("objective")
    if objective is None:
        objective = {"kind": "best_value"}
    c.obj(objective, "objective", {"kind", "margin", "reference", "cost_cap_usd"}, {"kind"})
    goal: dict = {"kind": c.choice(objective["kind"], "objective.kind", OBJECTIVE_KINDS)}
    if objective.get("margin") is not None:
        goal["margin"] = c.number(objective["margin"], "objective.margin", low=0)
    if objective.get("reference") is not None:
        goal["reference"] = c.string(objective["reference"], "objective.reference")
    if objective.get("cost_cap_usd") is not None:
        goal["cost_cap_usd"] = c.number(objective["cost_cap_usd"], "objective.cost_cap_usd", above=0)
    for kind, name in (("cheapest_within_margin", "margin"), ("best_under_cost", "cost_cap_usd")):
        if goal["kind"] == kind and name not in goal:
            raise c.fail("objective", f'the objective "{kind}" has no {name}', "add it", "missing_member")
    out = {"schema": SCHEMA_VALUE, "weights": dict(weights), "gates": gates,
           "gated_value": c.number(obj.get("gated_value", 0), "gated_value"), "objective": _ext(objective, goal)}
    return _ext(obj, out)


# ---------------------------------------------------------------------------------------------------
# Lines: items, runs, scores, pairs. Each validator takes one parsed line and the place
# to name in a refusal (`line 3: `).

def validate_item(obj: Any, file: str = ITEMS_FILE, where: str = "") -> dict:
    c = _Check(file)
    _schema(c, obj, where, SCHEMA_ITEMS)
    c.obj(obj, where, {"schema", "id", "input", "component", "label", "source"}, {"id", "input", "component"})
    out: dict = {"schema": SCHEMA_ITEMS, "id": c.ident(obj["id"], _join(where, "id")),
                 "input": c.string(obj["input"], _join(where, "input")),
                 "component": c.ident(obj["component"], _join(where, "component"))}
    if obj.get("label") is not None:
        out["label"] = c.string(obj["label"], _join(where, "label"))
    if obj.get("source") is not None:
        w = _join(where, "source")
        source = c.obj(obj["source"], w, {"kind", "model"}, {"kind"})
        out["source"] = {"kind": c.choice(source["kind"], f"{w}.kind", ITEM_SOURCE_KINDS)}
        if out["source"]["kind"] != "expert" and source.get("model") is None:
            raise c.fail(w, f'a source of kind "{out["source"]["kind"]}" has no model',
                         "add the id of the model that wrote the item", "missing_member")
        if source.get("model") is not None:
            out["source"]["model"] = c.string(source["model"], f"{w}.model")
        _ext(source, out["source"])
    return _ext(obj, out)


def validate_run(obj: Any, file: str = RUNS_FILE, where: str = "") -> dict:
    c = _Check(file)
    _schema(c, obj, where, SCHEMA_RUNS)
    c.obj(obj, where, {"schema", "run", "input", "arm", "split", "steps", "cost_usd", "simulated", "refs"},
          {"run", "input", "arm", "steps"})
    out: dict = {"schema": SCHEMA_RUNS}
    for name in ("run", "input", "arm"):
        out[name] = c.string(obj[name], _join(where, name))
    if obj.get("split") is not None:
        out["split"] = c.string(obj["split"], _join(where, "split"))
    steps = []
    for n, step in enumerate(c.array(obj["steps"], _join(where, "steps"), non_empty=True)):
        w = _join(where, f"steps[{n}]")
        c.obj(step, w, {"step", "model", "effort", "variant"}, {"step", "model"})
        row: dict = {"step": c.ident(step["step"], f"{w}.step"), "model": _model(c, step["model"], f"{w}.model")}
        for name in ("effort", "variant"):
            if step.get(name) is not None:
                row[name] = c.string(step[name], f"{w}.{name}")
        steps.append(_ext(step, row))
    c.unique([row["step"] for row in steps], _join(where, "steps"), "step")
    out["steps"] = steps
    if obj.get("cost_usd") is not None:
        out["cost_usd"] = c.number(obj["cost_usd"], _join(where, "cost_usd"), low=0)
    out["simulated"] = c.boolean(obj.get("simulated", False), _join(where, "simulated"))
    if obj.get("refs") is not None:
        refs = obj["refs"]
        if not isinstance(refs, dict):
            raise c.fail(_join(where, "refs"), f"expected an object, got {_kind(refs)}",
                         "map a name (input, output) to a reference", "type")
        for name, ref in refs.items():
            c.string(ref, _join(where, f"refs.{name}"))
        out["refs"] = dict(refs)
    return _ext(obj, out)


def _reading_common(c: _Check, obj: dict, out: dict, where: str) -> None:
    out["judge"] = c.ident(obj["judge"], _join(where, "judge"))
    out["call"] = c.string(obj["call"], _join(where, "call"))
    for name, value in (("rubric", obj["rubric"]), ("observed_at", obj["observed_at"])):
        w = _join(where, name)
        if name == "observed_at":
            out[name] = c.timestamp(value, w)
        else:
            c.string(value, w)
            if value.count("@") != 1 or value.startswith("@") or value.endswith("@"):
                raise c.fail(w, f'"{value}" is not id@version', "name the rubric as id@version", "bad_rubric_ref")
            out[name] = value
    for name in ("replaces", "assignment"):
        if obj.get(name) is not None:
            out[name] = c.string(obj[name], _join(where, name))
    if out.get("replaces") == out["id"]:
        raise c.fail(_join(where, "replaces"), "a reading replaces itself", "name the earlier reading it corrects",
                     "replaces_self")
    if obj.get("cost_usd") is not None:
        out["cost_usd"] = c.number(obj["cost_usd"], _join(where, "cost_usd"), low=0)


def validate_score(obj: Any, file: str = SCORES_FILE, where: str = "") -> dict:
    """One reading. The value is checked against its scale by `check_reading`, which needs the rubric."""
    c = _Check(file)
    _schema(c, obj, where, SCHEMA_SCORES)
    c.obj(obj, where, {"schema", "id", "run", "step", "component", "judge", "call", "value", "item", "of", "rubric",
                       "observed_at", "replaces", "assignment", "cost_usd"},
          {"id", "run", "component", "judge", "call", "value", "rubric", "observed_at"})
    out: dict = {"schema": SCHEMA_SCORES, "id": c.string(obj["id"], _join(where, "id")),
                 "run": c.string(obj["run"], _join(where, "run")),
                 "step": c.ident(obj.get("step") or FINAL, _join(where, "step")),
                 "component": c.ident(obj["component"], _join(where, "component"))}
    _reading_common(c, obj, out, where)
    value = obj["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float, str)) or (
            isinstance(value, float) and not math.isfinite(value)) or value == "":
        raise c.fail(_join(where, "value"), f"expected a number or a level, got {_kind(value)}",
                     "give the raw score on the component's scale", "type")
    out["value"] = value
    item, of = obj.get("item"), obj.get("of")
    if item is not None and of is not None:
        raise c.fail(where, "both item and of are given", "write item lines or one count line, never both",
                     "item_and_of")
    if item is not None:
        out["item"] = c.string(item, _join(where, "item"))
        if isinstance(value, str) or value not in (0, 1):
            raise c.fail(_join(where, "value"), f"an item reading has the value {_show(value)}",
                         "give 0 or 1", "bad_value")
    if of is not None:
        out["of"] = c.integer(of, _join(where, "of"), low=1)
        if isinstance(value, str) or value != int(value) or not 0 <= value <= of:
            raise c.fail(_join(where, "value"), f"a count reading has the value {_show(value)} of {of}",
                         f"give a whole number from 0 to {of}", "bad_value")
    return _ext(obj, out)


def _show(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else _num(value)


def check_reading(row: dict, comp: dict, file: str = SCORES_FILE, where: str = "") -> None:
    """Refuse a normalised reading whose value is outside the scale of its rubric component `comp`."""
    c = _Check(file)
    scale, value = comp["scale"], row["value"]
    kind = scale["kind"]
    w = _join(where, "value")
    if kind != "items":
        for name in ("item", "of"):
            if name in row:
                raise c.fail(_join(where, name), f'the component "{comp["id"]}" has a "{kind}" scale',
                             f'drop {name}, or give the component the scale kind "items"', "not_items")
    if kind == "items":
        if "item" not in row and "of" not in row:
            raise c.fail(where, f'a reading of the items component "{comp["id"]}" has no item and no of',
                         "give item with value 0 or 1, or a count with of", "items_shape")
    elif kind == "binary":
        if isinstance(value, str) or value not in (0, 1):
            raise c.fail(w, f"{_show(value)} is not 0 or 1", f'the component "{comp["id"]}" is binary: give 0 or 1',
                         "bad_value")
    elif kind == "levels":
        if not any(type(value) is type(level) and value == level for level in scale["levels"]):
            raise c.fail(w, f"{_show(value)} is not a level of \"{comp['id']}\"",
                         f"use one of {', '.join(str(level) for level in scale['levels'])}", "bad_value")
    else:
        low, high = scale["range"] if kind == "linear" else (0, 1)
        if isinstance(value, str) or not low <= value <= high:
            raise c.fail(w, f"{_show(value)} is outside the scale of \"{comp['id']}\"",
                         f"give a number from {_num(low)} to {_num(high)}", "bad_value")


def validate_pair(obj: Any, file: str = PAIRS_FILE, where: str = "") -> dict:
    """One pairwise judgement. `component` is None when the step's `overall` component is meant."""
    c = _Check(file)
    _schema(c, obj, where, SCHEMA_PAIRS)
    c.obj(obj, where, {"schema", "id", "a", "b", "step", "component", "judge", "call", "shown_first", "verdict",
                       "rubric", "observed_at", "replaces", "assignment", "cost_usd"},
          {"id", "a", "b", "judge", "call", "verdict", "rubric", "observed_at"})
    out: dict = {"schema": SCHEMA_PAIRS, "id": c.string(obj["id"], _join(where, "id")),
                 "a": c.string(obj["a"], _join(where, "a")), "b": c.string(obj["b"], _join(where, "b")),
                 "step": c.ident(obj.get("step") or FINAL, _join(where, "step")),
                 "component": None if obj.get("component") is None else c.ident(obj["component"],
                                                                                 _join(where, "component"))}
    if out["a"] == out["b"]:
        raise c.fail(where, f'a and b are the same run "{out["a"]}"', "compare two runs of one input", "same_run")
    _reading_common(c, obj, out, where)
    out["shown_first"] = None if obj.get("shown_first") is None else c.choice(obj["shown_first"],
                                                                              _join(where, "shown_first"), ("a", "b"))
    out["verdict"] = c.choice(obj["verdict"], _join(where, "verdict"), VERDICTS)
    return _ext(obj, out)


# ---------------------------------------------------------------------------------------------------
# Labels plan

def validate_labels(obj: Any, file: str = "labels.json") -> dict:
    """A labels plan. It is blind by construction: an arm, a model or a judge score is an unknown member.

    Members: `inputs`, `options`, `seed`, `mode`, `reviewer`, `assignments[]`. A score assignment has
    `target` (`run`, `step`); a pair assignment has `a`, `b` (each `run`, `step`) and `shown_first`."""
    c = _Check(file)
    _schema(c, obj, "", SCHEMA_LABELS)
    c.obj(obj, "", {"schema", "inputs", "options", "seed", "mode", "reviewer", "assignments"},
          {"mode", "reviewer", "assignments"})
    out: dict = {"schema": SCHEMA_LABELS, "inputs": _free(c, obj.get("inputs"), "inputs"),
                 "options": _free(c, obj.get("options"), "options"),
                 "seed": c.integer(obj.get("seed", 0), "seed"),
                 "mode": c.choice(obj["mode"], "mode", LABEL_MODES), "reviewer": c.ident(obj["reviewer"], "reviewer")}
    rows = []
    for n, row in enumerate(c.array(obj["assignments"], "assignments")):
        w = f"assignments[{n}]"
        c.obj(row, w, {"id", "ask", "target", "a", "b", "shown_first", "components", "reviewer", "kind",
                       "inclusion_probability", "scores", "minutes", "refs"},
              {"id", "ask", "components", "kind", "inclusion_probability", "minutes"})
        one: dict = {"id": c.string(row["id"], f"{w}.id"), "ask": c.choice(row["ask"], f"{w}.ask", ASKS)}
        used = ("target",) if one["ask"] == "score" else ("a", "b")
        for name in ("target", "a", "b"):
            if name in used:
                if row.get(name) is None:
                    raise c.fail(w, f'a {one["ask"]} assignment has no {name}', "add it with run and step",
                                 "missing_member")
                target = c.obj(row[name], f"{w}.{name}", {"run", "step"}, {"run"})
                one[name] = {"run": c.string(target["run"], f"{w}.{name}.run"),
                             "step": c.ident(target.get("step") or FINAL, f"{w}.{name}.step")}
            elif row.get(name) is not None:
                raise c.fail(f"{w}.{name}", f'a {one["ask"]} assignment has {name}', "remove it", "assignment_shape")
        if one["ask"] == "pair":
            one["shown_first"] = c.choice(row.get("shown_first"), f"{w}.shown_first", ("a", "b"))
        elif row.get("shown_first") is not None:
            raise c.fail(f"{w}.shown_first", "a score assignment has shown_first", "remove it", "assignment_shape")
        listed = c.array(row["components"], f"{w}.components", non_empty=True)
        for m, name in enumerate(listed):
            c.ident(name, f"{w}.components[{m}]")
        one["components"] = list(listed)
        one["reviewer"] = c.ident(row.get("reviewer") or out["reviewer"], f"{w}.reviewer")
        one["kind"] = c.choice(row["kind"], f"{w}.kind", ASSIGNMENT_KINDS)
        p = c.number(row["inclusion_probability"], f"{w}.inclusion_probability", above=0)
        one["inclusion_probability"] = c.number(p, f"{w}.inclusion_probability", high=1)
        scores = row.get("scores")
        if scores is None:
            scores = {}
        c.obj(scores, f"{w}.scores", LABEL_SCORES)
        one["scores"] = {name: None if scores.get(name) is None else c.number(scores[name], f"{w}.scores.{name}")
                         for name in LABEL_SCORES}
        one["minutes"] = c.number(row["minutes"], f"{w}.minutes", above=0)
        one["refs"] = _free(c, row.get("refs"), f"{w}.refs")
        rows.append(_ext(row, one))
    c.unique([row["id"] for row in rows], "assignments", "assignment id")
    out["assignments"] = rows
    return _ext(obj, out)


def _free(c: _Check, value: Any, where: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise c.fail(where, f"expected an object, got {_kind(value)}", "write a JSON object here", "type")
    return value


# ---------------------------------------------------------------------------------------------------
# Reading and writing files

DOCUMENTS: dict[str, Callable[..., dict]] = {
    SCHEMA_RUBRIC: validate_rubric, SCHEMA_JUDGES: validate_judges, SCHEMA_VALUE: validate_value,
    SCHEMA_LABELS: validate_labels,
}
LINES: dict[str, Callable[..., dict]] = {
    SCHEMA_ITEMS: validate_item, SCHEMA_RUNS: validate_run, SCHEMA_SCORES: validate_score,
    SCHEMA_PAIRS: validate_pair,
}
LINE_KEY = {SCHEMA_ITEMS: "id", SCHEMA_RUNS: "run", SCHEMA_SCORES: "id", SCHEMA_PAIRS: "id"}


def _no_constant(name: str) -> float:
    raise ValueError(f"{name} is not JSON")


def load_json(path: str | os.PathLike, file: str | None = None) -> Any:
    """Parse a JSON file. A missing file or a syntax error is a refusal that names the file."""
    path = Path(path)
    file = file or path.name
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ContractError(file, "", "no such file", f"check the path {path}", "no_file") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(file, "", f"cannot be read ({type(exc).__name__})", "give a UTF-8 JSON file",
                            "unreadable") from None
    try:
        return json.loads(text, parse_constant=_no_constant)
    except ValueError as exc:
        raise ContractError(file, "", f"not valid JSON ({exc})", "fix the syntax", "bad_json") from None


def read_document(path: str | os.PathLike, schema: str) -> dict:
    """Read and validate a rubric, judges, value or labels file. Returns it normalised."""
    path = Path(path)
    return DOCUMENTS[schema](load_json(path), path.name)


def read_lines(path: str | os.PathLike, schema: str, *, missing_ok: bool = False) -> list[dict]:
    """Read and validate a JSON Lines file of runs, scores, pairs or items. Blank lines are skipped.

    A line that does not parse, breaks its schema, or repeats an earlier id is a refusal by line number."""
    path = Path(path)
    file, validate, key = path.name, LINES[schema], LINE_KEY[schema]
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if missing_ok:
            return []
        raise ContractError(file, "", "no such file", f"check the path {path}", "no_file") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ContractError(file, "", f"cannot be read ({type(exc).__name__})", "give a UTF-8 JSON Lines file",
                            "unreadable") from None
    rows, seen = [], {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line, parse_constant=_no_constant)
        except ValueError as exc:
            raise ContractError(file, f"line {n}", f"not valid JSON ({exc})", "write one JSON object per line",
                                "bad_json") from None
        row = validate(obj, file, f"line {n}: ")
        if row[key] in seen:
            raise ContractError(file, f"line {n}: {key}", f'"{row[key]}" is already on line {seen[row[key]]}',
                                f"give every line its own {key}", "duplicate")
        seen[row[key]] = n
        rows.append(row)
    return rows


def dumps(obj: Any) -> str:
    """The text of a JSON document as every stage writes it: indent 1, a final newline, no NaN."""
    return json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def write_text_atomic(path: str | os.PathLike, text: str | bytes) -> Path:
    """Write a file in one step, with the mode a plain `open` would give."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8") if isinstance(text, str) else text
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_document(path: str | os.PathLike, obj: dict) -> Path:
    """Validate (when the schema is a document of this module) and write a JSON document."""
    path = Path(path)
    validate = DOCUMENTS.get(obj.get("schema", ""))
    if validate is not None:
        obj = validate(obj, path.name)
    return write_text_atomic(path, dumps(obj))


def write_lines(path: str | os.PathLike, schema: str, rows: Iterable[dict]) -> Path:
    """Validate and write a whole JSON Lines file, one compact object per line, `schema` first."""
    path = Path(path)
    validate, key, out, seen = LINES[schema], LINE_KEY[schema], [], {}
    for n, row in enumerate(rows, 1):
        row = validate({"schema": schema, **{k: v for k, v in row.items() if k != "schema"}}, path.name,
                       f"line {n}: ")
        if row[key] in seen:
            raise ContractError(path.name, f"line {n}: {key}", f'"{row[key]}" is already on line {seen[row[key]]}',
                                f"give every line its own {key}", "duplicate")
        seen[row[key]] = n
        out.append(json.dumps({k: v for k, v in row.items() if v is not None}, ensure_ascii=False, allow_nan=False))
    return write_text_atomic(path, "".join(line + "\n" for line in out))


@dataclass
class Study:
    """A study folder, read and validated line by line. Cross checks against a rubric are `eval scores check`."""

    folder: Path
    runs: list[dict] = field(default_factory=list)
    scores: list[dict] = field(default_factory=list)
    pairs: list[dict] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)

    def files(self) -> dict[str, Path]:
        """The study files that exist, by role: `runs`, `scores`, `pairs`, `items`."""
        names = {"runs": RUNS_FILE, "scores": SCORES_FILE, "pairs": PAIRS_FILE, "items": ITEMS_FILE}
        return {role: self.folder / name for role, name in names.items() if (self.folder / name).is_file()}


def read_study(folder: str | os.PathLike) -> Study:
    """Read a study folder. `runs.jsonl` must exist; `scores.jsonl`, `pairs.jsonl` and `items.jsonl` may be absent."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ContractError(str(folder), "", "no such study folder", "name a folder that holds runs.jsonl", "no_file")
    return Study(folder, read_lines(folder / RUNS_FILE, SCHEMA_RUNS),
                 read_lines(folder / SCORES_FILE, SCHEMA_SCORES, missing_ok=True),
                 read_lines(folder / PAIRS_FILE, SCHEMA_PAIRS, missing_ok=True),
                 read_lines(folder / ITEMS_FILE, SCHEMA_ITEMS, missing_ok=True))


def live(rows: list[dict], file: str = SCORES_FILE) -> tuple[list[dict], list[dict]]:
    """Apply `replaces`: `(read, not_read)`. A replaced reading or pair is not read.

    Each entry of `not_read` is `{"id", "reason": "replaced", "by"}`. A `replaces` that names nothing or
    forms a loop is a refusal."""
    by_id = {row["id"]: row for row in rows}
    replaced_by: dict[str, str] = {}
    for row in rows:
        target = row.get("replaces")
        if target is None:
            continue
        if target not in by_id:
            raise ContractError(file, f'id "{row["id"]}": replaces', f'"{target}" names no reading',
                                "name the id of the reading it corrects", "replaces_unknown")
        replaced_by[target] = row["id"]
    for row in rows:
        seen, at = {row["id"]}, row.get("replaces")
        while at is not None:
            if at in seen:
                raise ContractError(file, f'id "{row["id"]}": replaces', "the corrections form a loop",
                                    "a correction names an earlier reading only", "replaces_loop")
            seen.add(at)
            at = by_id[at].get("replaces")
    read = [row for row in rows if row["id"] not in replaced_by]
    not_read = [{"id": row["id"], "reason": "replaced", "by": replaced_by[row["id"]]}
                for row in rows if row["id"] in replaced_by]
    return read, not_read


# ---------------------------------------------------------------------------------------------------
# Provenance

def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def input_digests(files: dict[str, str | os.PathLike]) -> dict[str, dict]:
    """The `inputs` member of an output: per role, `{"file": <name>, "sha256": <hex>}`, roles sorted."""
    return {role: {"file": Path(path).name, "sha256": sha256_file(path)} for role, path in sorted(files.items())}


def stale_inputs(recorded: dict[str, dict], files: dict[str, str | os.PathLike]) -> list[str]:
    """What changed between the digests an output recorded and the files given now, one line per role."""
    changes = []
    for role in sorted(set(recorded) | set(files)):
        if role not in files or not Path(files[role]).is_file():
            changes.append(f"{role}: {recorded[role]['file']} was read by the earlier stage and is missing now")
        elif role not in recorded:
            changes.append(f"{role}: {Path(files[role]).name} was not read by the earlier stage")
        elif sha256_file(files[role]) != recorded[role]["sha256"]:
            changes.append(f"{role}: {Path(files[role]).name} changed since the earlier stage read it")
    return changes


# ---------------------------------------------------------------------------------------------------
# Examples: per schema, objects that pass and objects that are refused with the exact refusal text.
# `tests/eval/test_eval_contracts.py` pins every message below. All data is synthetic.

def _edit(base: dict, change: Callable[[dict], Any]) -> dict:
    out = json.loads(json.dumps(base))
    change(out)
    return out


_RUBRIC = {
    "schema": SCHEMA_RUBRIC, "id": "brief", "version": "2", "steps": ["draft", "final"],
    "components": [
        {"id": "numbers_right", "title": "The numbers are right", "scale": {"kind": "binary"}, "human_only": True},
        {"id": "clarity", "scale": {"kind": "levels", "levels": ["poor", "ok", "good"]},
         "applies_to": ["draft", "final"], "anchors": "anchors/clarity.md"},
        {"id": "depth", "scale": {"kind": "linear", "range": [0, 10]}, "review_minutes": 3, "compare_minutes": 1},
        {"id": "errors", "scale": {"kind": "fraction"}, "better": "lower"},
        {"id": "coverage", "scale": {"kind": "items"}},
        {"id": "overall", "role": "overall", "scale": {"kind": "linear", "range": [1, 5]}, "ext": {"note": "any"}},
    ]}
_JUDGES = {
    "schema": SCHEMA_JUDGES, "anchor": ["reviewer"], "min_anchor_targets": 6,
    "judges": [
        {"id": "reviewer", "kind": "person", "blinded": True},
        {"id": "strong", "kind": "model", "model": {"id": "model-large", "family": "alpha", "provider": "p"},
         "effort": "high", "prompt": "v1", "mode": "panel", "cost_usd_per_call": 0.05, "components": ["clarity"],
         "reference": {"shown": True, "author": {"kind": "verified", "model": {"id": "model-large",
                                                                                "family": "alpha"}},
                       "version": "r1"}},
        {"id": "script", "kind": "command"},
    ]}
_VALUE = {"schema": SCHEMA_VALUE, "weights": {"clarity": 1, "depth": 2},
          "gates": [{"component": "numbers_right"}, {"component": "clarity", "at_least": "ok", "step": "draft"}],
          "gated_value": 0, "objective": {"kind": "cheapest_within_margin", "margin": 0.05, "reference": "baseline"}}
_ITEM = {"schema": SCHEMA_ITEMS, "id": "it-1", "input": "case-01", "component": "coverage",
         "label": "covers the fee change", "source": {"kind": "model_verified", "model": "model-large"}}
_RUN = {"schema": SCHEMA_RUNS, "run": "baseline-01", "input": "case-01", "arm": "baseline", "split": "dev",
        "steps": [{"step": "draft", "model": {"id": "model-small", "family": "beta"}, "effort": "low"},
                  {"step": "final", "model": {"id": "model-small", "family": "beta"}, "effort": "low",
                   "variant": "prompt-2"}],
        "cost_usd": 0.12, "simulated": False, "refs": {"input": "cases/01.md", "output": "out/baseline-01.md"}}
_SCORE = {"schema": SCHEMA_SCORES, "id": "s1", "run": "baseline-01", "step": "final", "component": "depth",
          "judge": "strong", "call": "c1", "value": 6.5, "rubric": "brief@2",
          "observed_at": "2026-01-15T09:30:00-08:00", "cost_usd": 0.05}
_PAIR = {"schema": SCHEMA_PAIRS, "id": "p1", "a": "baseline-01", "b": "strong-01", "judge": "reviewer",
         "call": "pc1", "shown_first": "b", "verdict": "tie", "rubric": "brief@2",
         "observed_at": "2026-01-15T09:30:00-08:00"}
_LABELS = {
    "schema": SCHEMA_LABELS, "mode": "active", "reviewer": "reviewer", "seed": 1,
    "assignments": [
        {"id": "a1", "ask": "score", "target": {"run": "baseline-01", "step": "final"},
         "components": ["clarity", "depth"], "kind": "active", "inclusion_probability": 0.4,
         "scores": {"judge_uncertainty": 0.02, "decision_weight": 0.1, "component_gain": 0.01, "per_minute": 0.03},
         "minutes": 4, "refs": {"output": "out/baseline-01.md"}},
        {"id": "a2", "ask": "pair", "a": {"run": "baseline-01"}, "b": {"run": "strong-01"}, "shown_first": "b",
         "components": ["overall"], "kind": "uniform", "inclusion_probability": 0.05, "minutes": 1},
    ]}

EXAMPLES: dict[str, dict[str, list]] = {
    SCHEMA_RUBRIC: {
        "valid": [_RUBRIC, {"schema": SCHEMA_RUBRIC, "id": "one", "version": 1,
                            "components": [{"id": "quality", "scale": {"kind": "fraction"}}]}],
        "invalid": [
            (_edit(_RUBRIC, lambda o: o.pop("schema")),
             'rubric.json: schema: missing; add "schema": "loopmath.eval-rubric/1"'),
            (_edit(_RUBRIC, lambda o: o.update(schema="loopmath.eval-rubric/2")),
             'rubric.json: schema: unknown schema "loopmath.eval-rubric/2"; this stage reads "loopmath.eval-rubric/1"'),
            (_edit(_RUBRIC, lambda o: o.pop("components")), 'rubric.json: missing member "components"; add it'),
            (_edit(_RUBRIC, lambda o: o["components"][0].update(weight=2)),
             'rubric.json: components[0].weight: unknown member; remove it, or move it under "ext"'),
            (_edit(_RUBRIC, lambda o: o["components"][0].update(id="a/b")),
             'rubric.json: components[0].id: the id "a/b" holds a space, "/" or "@"; '
             'use letters, digits, "-", "_" and "."'),
            (_edit(_RUBRIC, lambda o: o["components"][0]["scale"].update(kind="stars")),
             'rubric.json: components[0].scale.kind: "stars" is not one of binary, levels, linear, fraction, items; '
             'use one of them'),
            (_edit(_RUBRIC, lambda o: o["components"][0]["scale"].update(range=[0, 1])),
             'rubric.json: components[0].scale.range: a "binary" scale has no range; remove it'),
            (_edit(_RUBRIC, lambda o: o["components"][1]["scale"].update(levels=["no", "yes"])),
             'rubric.json: components[1].scale.levels: 2 levels given; use scale.kind "binary" for a yes or no check'),
            (_edit(_RUBRIC, lambda o: o["components"][1]["scale"].update(levels=["ok", "ok", "good"])),
             'rubric.json: components[1].scale.levels: level "ok" appears twice; give every level once'),
            (_edit(_RUBRIC, lambda o: o["components"][2]["scale"].update(range=[10, 0])),
             'rubric.json: components[2].scale.range: low 10 is not below high 0; give [low, high]'),
            (_edit(_RUBRIC, lambda o: o["components"][2].update(review_minutes=0)),
             'rubric.json: components[2].review_minutes: 0 is out of range; give a number above 0'),
            (_edit(_RUBRIC, lambda o: o["components"][3].update(better="smaller")),
             'rubric.json: components[3].better: "smaller" is not one of higher, lower; use one of them'),
            (_edit(_RUBRIC, lambda o: o["components"][1].update(applies_to=["review"])),
             'rubric.json: components[1].applies_to[0]: unknown step "review"; '
             'use one of draft, final, or add the step to steps'),
            (_edit(_RUBRIC, lambda o: o.update(steps=["final", "draft"])),
             'rubric.json: steps: the last step is "draft"; end the list with "final"'),
            (_edit(_RUBRIC, lambda o: o["components"][2].update(role="overall")),
             'rubric.json: components[5].role: step "final" already has the overall component "depth"; '
             'keep one overall component per step'),
            (_edit(_RUBRIC, lambda o: o["components"][2].update(id="clarity")),
             'rubric.json: components: component id "clarity" appears twice; give every component id once'),
            (_edit(_RUBRIC, lambda o: o.update(components=[])),
             'rubric.json: components: the list is empty; give at least one entry'),
            ([], 'rubric.json: expected an object, got a list; write a JSON object here'),
        ]},
    SCHEMA_JUDGES: {
        "valid": [_JUDGES, {"schema": SCHEMA_JUDGES, "judges": [{"id": "j", "kind": "person"}]}],
        "invalid": [
            (_edit(_JUDGES, lambda o: o.update(schema="loopmath.judges/1")),
             'judges.json: schema: unknown schema "loopmath.judges/1"; this stage reads "loopmath.eval-judges/1"'),
            (_edit(_JUDGES, lambda o: o["judges"][1].pop("model")),
             'judges.json: judges[1]: a judge of kind "model" has no model; add model with id and family'),
            (_edit(_JUDGES, lambda o: o["judges"][0].update(model={"id": "m", "family": "alpha"})),
             'judges.json: judges[0].model: a judge of kind "person" has a model; remove it, or set kind to "model"'),
            (_edit(_JUDGES, lambda o: o["judges"][1]["model"].pop("family")),
             'judges.json: judges[1].model: missing member "family"; add it'),
            (_edit(_JUDGES, lambda o: o["judges"][1].update(kind="robot")),
             'judges.json: judges[1].kind: "robot" is not one of model, person, command; use one of them'),
            (_edit(_JUDGES, lambda o: o["judges"][1].update(mode="batch")),
             'judges.json: judges[1].mode: "batch" is not one of per_component, panel; use one of them'),
            (_edit(_JUDGES, lambda o: o["judges"][1]["reference"].pop("author")),
             'judges.json: judges[1].reference: a shown reference has no author; '
             'add author with kind (model, person or verified)'),
            (_edit(_JUDGES, lambda o: o["judges"][1]["reference"]["author"].pop("model")),
             'judges.json: judges[1].reference.author: an author of kind "verified" has no model; '
             'add model with id and family'),
            (_edit(_JUDGES, lambda o: o["judges"][1].update(cost_usd_per_call=-1)),
             'judges.json: judges[1].cost_usd_per_call: -1 is out of range; give a number at least 0'),
            (_edit(_JUDGES, lambda o: o["judges"][1].update(blinded="yes")),
             'judges.json: judges[1].blinded: expected true or false, got a string; write true or false'),
            (_edit(_JUDGES, lambda o: o["judges"][2].update(id="strong")),
             'judges.json: judges: judge id "strong" appears twice; give every judge id once'),
            (_edit(_JUDGES, lambda o: o.update(anchor=["nobody"])),
             'judges.json: anchor[0]: unknown judge "nobody"; name a judge id from judges'),
            (_edit(_JUDGES, lambda o: o.update(min_anchor_targets=0)),
             'judges.json: min_anchor_targets: 0 is out of range; give a whole number at least 1'),
            (_edit(_JUDGES, lambda o: o.update(judges=[])),
             'judges.json: judges: the list is empty; give at least one entry'),
        ]},
    SCHEMA_VALUE: {
        "valid": [_VALUE, {"schema": SCHEMA_VALUE, "weights": {"overall": 1}},
                  {"schema": SCHEMA_VALUE, "gates": [{"component": "depth", "at_least": 6}]}],
        "invalid": [
            (_edit(_VALUE, lambda o: o.pop("schema")),
             'value.json: schema: missing; add "schema": "loopmath.eval-value/1"'),
            ({"schema": SCHEMA_VALUE, "weights": {}, "gates": []},
             'value.json: no weight and no gate; add weights, gates or both'),
            (_edit(_VALUE, lambda o: o["weights"].update(clarity=-1)),
             'value.json: weights.clarity: -1 is out of range; give a number at least 0'),
            (_edit(_VALUE, lambda o: o["gates"][0].pop("component")),
             'value.json: gates[0]: missing member "component"; add it'),
            (_edit(_VALUE, lambda o: o["gates"][1].update(at_most="ok")),
             'value.json: gates[1].at_most: unknown member; remove it, or move it under "ext"'),
            (_edit(_VALUE, lambda o: o["gates"][1].update(at_least=[1])),
             "value.json: gates[1].at_least: expected a number or a level, got a list; "
             "give the threshold on the component's own scale"),
            (_edit(_VALUE, lambda o: o["objective"].update(kind="maximise")),
             'value.json: objective.kind: "maximise" is not one of best_value, cheapest_within_margin, '
             'best_under_cost, cost_per_pass; use one of them'),
            (_edit(_VALUE, lambda o: o["objective"].pop("margin")),
             'value.json: objective: the objective "cheapest_within_margin" has no margin; add it'),
            (_edit(_VALUE, lambda o: o.update(objective={"kind": "best_under_cost"})),
             'value.json: objective: the objective "best_under_cost" has no cost_cap_usd; add it'),
            (_edit(_VALUE, lambda o: o.update(gated_value="zero")),
             'value.json: gated_value: expected a finite number, got a string; write a number here'),
        ]},
    SCHEMA_ITEMS: {
        "valid": [_ITEM, {"schema": SCHEMA_ITEMS, "id": "it-2", "input": "case-01", "component": "coverage"}],
        "invalid": [
            (_edit(_ITEM, lambda o: o.pop("input")), 'items.jsonl: missing member "input"; add it'),
            (_edit(_ITEM, lambda o: o["source"].pop("model")),
             'items.jsonl: source: a source of kind "model_verified" has no model; '
             'add the id of the model that wrote the item'),
            (_edit(_ITEM, lambda o: o["source"].update(kind="intern")),
             'items.jsonl: source.kind: "intern" is not one of expert, model, model_verified; use one of them'),
            (_edit(_ITEM, lambda o: o.update(text="the full item text")),
             'items.jsonl: text: unknown member; remove it, or move it under "ext"'),
        ]},
    SCHEMA_RUNS: {
        "valid": [_RUN, {"schema": SCHEMA_RUNS, "run": "r", "input": "i", "arm": "a",
                         "steps": [{"step": "final", "model": {"id": "m", "family": "alpha"}}]}],
        "invalid": [
            (_edit(_RUN, lambda o: o.update(schema="loopmath.eval-scores/1")),
             'runs.jsonl: schema: unknown schema "loopmath.eval-scores/1"; this stage reads "loopmath.eval-runs/1"'),
            (_edit(_RUN, lambda o: o.pop("arm")), 'runs.jsonl: missing member "arm"; add it'),
            (_edit(_RUN, lambda o: o.update(steps=[])), 'runs.jsonl: steps: the list is empty; give at least one entry'),
            (_edit(_RUN, lambda o: o["steps"][0].pop("model")), 'runs.jsonl: steps[0]: missing member "model"; add it'),
            (_edit(_RUN, lambda o: o["steps"][1].update(step="draft")),
             'runs.jsonl: steps: step "draft" appears twice; give every step once'),
            (_edit(_RUN, lambda o: o.update(cost_usd=-0.5)),
             'runs.jsonl: cost_usd: -0.5 is out of range; give a number at least 0'),
            (_edit(_RUN, lambda o: o.update(simulated="no")),
             'runs.jsonl: simulated: expected true or false, got a string; write true or false'),
            (_edit(_RUN, lambda o: o.update(output="the full text")),
             'runs.jsonl: output: unknown member; remove it, or move it under "ext"'),
        ]},
    SCHEMA_SCORES: {
        "valid": [_SCORE,
                  _edit(_SCORE, lambda o: o.update(id="s2", component="coverage", item="it-1", value=1)),
                  _edit(_SCORE, lambda o: o.update(id="s3", component="coverage", of=5, value=3)),
                  _edit(_SCORE, lambda o: o.update(id="s4", component="clarity", value="ok", replaces="s1",
                                                   assignment="a1", observed_at="2026-01-15T17:30:00Z")),
                  _edit(_SCORE, lambda o: o.pop("step"))],
        "invalid": [
            (_edit(_SCORE, lambda o: o.pop("schema")),
             'scores.jsonl: schema: missing; add "schema": "loopmath.eval-scores/1"'),
            (_edit(_SCORE, lambda o: o.pop("call")), 'scores.jsonl: missing member "call"; add it'),
            (_edit(_SCORE, lambda o: o.pop("rubric")), 'scores.jsonl: missing member "rubric"; add it'),
            (_edit(_SCORE, lambda o: o.update(rubric="brief")),
             'scores.jsonl: rubric: "brief" is not id@version; name the rubric as id@version'),
            (_edit(_SCORE, lambda o: o.update(observed_at="2026-01-15T09:30:00")),
             'scores.jsonl: observed_at: "2026-01-15T09:30:00" is not an ISO 8601 time with an offset; '
             'write it like 2026-01-15T09:30:00-08:00'),
            (_edit(_SCORE, lambda o: o.update(value=True)),
             "scores.jsonl: value: expected a number or a level, got true or false; "
             "give the raw score on the component's scale"),
            (_edit(_SCORE, lambda o: o.update(item="it-1", of=5)),
             'scores.jsonl: both item and of are given; write item lines or one count line, never both'),
            (_edit(_SCORE, lambda o: o.update(item="it-1", value=2)),
             'scores.jsonl: value: an item reading has the value 2; give 0 or 1'),
            (_edit(_SCORE, lambda o: o.update(of=5, value=6)),
             'scores.jsonl: value: a count reading has the value 6 of 5; give a whole number from 0 to 5'),
            (_edit(_SCORE, lambda o: o.update(of=0, value=0)),
             'scores.jsonl: of: 0 is out of range; give a whole number at least 1'),
            (_edit(_SCORE, lambda o: o.update(replaces="s1")),
             'scores.jsonl: replaces: a reading replaces itself; name the earlier reading it corrects'),
            (_edit(_SCORE, lambda o: o.update(arm="baseline")),
             'scores.jsonl: arm: unknown member; remove it, or move it under "ext"'),
        ]},
    SCHEMA_PAIRS: {
        "valid": [_PAIR, _edit(_PAIR, lambda o: o.update(id="p2", component="depth", step="final", shown_first=None,
                                                         verdict="a", replaces="p1"))],
        "invalid": [
            (_edit(_PAIR, lambda o: o.update(b="baseline-01")),
             'pairs.jsonl: a and b are the same run "baseline-01"; compare two runs of one input'),
            (_edit(_PAIR, lambda o: o.update(verdict="left")),
             'pairs.jsonl: verdict: "left" is not one of a, b, tie; use one of them'),
            (_edit(_PAIR, lambda o: o.update(shown_first="left")),
             'pairs.jsonl: shown_first: "left" is not one of a, b; use one of them'),
            (_edit(_PAIR, lambda o: o.pop("verdict")), 'pairs.jsonl: missing member "verdict"; add it'),
            (_edit(_PAIR, lambda o: o.update(winner="a")),
             'pairs.jsonl: winner: unknown member; remove it, or move it under "ext"'),
        ]},
    SCHEMA_LABELS: {
        "valid": [_LABELS, {"schema": SCHEMA_LABELS, "mode": "pilot", "reviewer": "reviewer", "assignments": []}],
        "invalid": [
            (_edit(_LABELS, lambda o: o["assignments"][0].update(arm="baseline")),
             'labels.json: assignments[0].arm: unknown member; remove it, or move it under "ext"'),
            (_edit(_LABELS, lambda o: o["assignments"][0]["scores"].update(judge_score=0.4)),
             'labels.json: assignments[0].scores.judge_score: unknown member; remove it, or move it under "ext"'),
            (_edit(_LABELS, lambda o: o["assignments"][0].update(inclusion_probability=0)),
             'labels.json: assignments[0].inclusion_probability: 0 is out of range; give a number above 0'),
            (_edit(_LABELS, lambda o: o["assignments"][0].update(inclusion_probability=1.5)),
             'labels.json: assignments[0].inclusion_probability: 1.5 is out of range; give a number at most 1'),
            (_edit(_LABELS, lambda o: o["assignments"][1].pop("b")),
             'labels.json: assignments[1]: a pair assignment has no b; add it with run and step'),
            (_edit(_LABELS, lambda o: o["assignments"][0].update(shown_first="a")),
             'labels.json: assignments[0].shown_first: a score assignment has shown_first; remove it'),
            (_edit(_LABELS, lambda o: o["assignments"][1].update(kind="random")),
             'labels.json: assignments[1].kind: "random" is not one of pilot, active, uniform, repeat, overlap; '
             'use one of them'),
            (_edit(_LABELS, lambda o: o["assignments"][1].update(id="a1")),
             'labels.json: assignments: assignment id "a1" appears twice; give every assignment id once'),
            (_edit(_LABELS, lambda o: o.update(mode="greedy")),
             'labels.json: mode: "greedy" is not one of pilot, active; use one of them'),
        ]},
}
VALIDATORS: dict[str, Callable[..., dict]] = {**DOCUMENTS, **LINES}
