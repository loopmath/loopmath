"""The questions `loopmath flow init` asks, and the flow document the answers make.

The answers are one JSON object with exactly the eight keys of `KEYS`, one per question of `QUESTIONS`, in any
order. A key whose answer may be empty takes null; the key itself is still required. A missing key and an
unknown key are refused by name at any depth, with a path such as `steps[2].modle`:

- the answer records (purpose, a step, a `writes` entry, checks, a gate, judging, objective, hypothesis, an
  edit and inputs) take their own members only;
- the judges, the acceptance rule, the check tables and `allow` are held to the OCP 0.4 schema: its members
  only, the required ones present, each of its type and within its enum, minimum and lengths. Piece ids,
  option names and `ext` members are open. A null inside them has no TOML form and is refused.

`tables` builds the TOML view's tables from the answers, and `build` reads them with `flow_from_tables`. So the
defaults, the configuration id, the digest and the producer are those `flow import` and `load` use. The
hypothesis edits must be single edits that `flow.allow` permits (`edits.candidates`), and the document must
validate with no error. `ask` puts the questions on a terminal; `in_schema_order` gives a document's members in
the order of the OCP 0.4 schema, for the one writer of `flow.ocp.json`.

With a prefill (`given`, a `prefill.Prefill`), the answers give only what the runs cannot say (`keys`), and a
value the runs give is refused at its path, never taken again. Declared runs give the configuration whole, so the
answers have no `steps` or `checks`, and the flow's configuration is the runs' one as given. Inferred runs give
each step's role and setting, so each step is answered `{id, reads, writes}`, one per step in the runs' order, and
is completed from the runs before the checks of the eight-key answers, which take the runs' option values as
given. What the tables cannot hold of the runs' values (a null, an integer past 64 bits) goes to
`[[document.keep]]` entries. Either way `flow.ext` gains the
provenance (`EXT_PREFILL`) after the hypothesis edits. With no prefill, nothing changes.
"""

from __future__ import annotations

import copy
import functools
import json
import math
import re
import unicodedata
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, TextIO

from ..ocp import conformance
from .edits import candidates
from .records import EXT_HYPOTHESIS, EXT_PREFILL, Edit, Flow
from .tomlview import _diff, _first_difference, _one_line, _prune, _same, flow_from_tables

QUESTIONS = (
    ("purpose", "What is the flow for? Its id, a title, and one example input."),
    ("steps", "What are the steps, in run order? Each step's model, and what it reads and writes."),
    ("checks", "Which checks run after each step, and what does a failure rerun?"),
    ("judging", "How is a run judged? The judges, the rubric, the person rate and the acceptance rule."),
    ("objective", "What is the objective? The score it reads, and the margin (0.10 when absent)."),
    ("hypothesis", "What is the hypothesis, and which allowed edits does it name? null for none."),
    ("allow", "What may change at each step? null for nothing."),
    ("inputs", "What are the inputs, and which of them are the holdout?"),
)
KEYS = tuple(key for key, _ in QUESTIONS)
DECLARED_KEYS = ("steps", "checks")  # what declared runs give
STEPS_GIVEN = "What does each step read and write? The runs give each step's role and setting."
GIVEN_STEP = ("role", "model", "harness", "effort", "width", "context_policy", "options")  # inferred runs give these
_IDS = [f"q{i:02d}" for i in range(1, 21)]
EXAMPLE: dict[str, Any] = {  # the shipped answers, examples/flows/lit-brief/answers.json; the walk shows them
    "purpose": {"id": "lit-brief", "title": "Question to a one-page brief", "example_input": "Stub question 1."},
    "steps": [
        {"id": "scout", "role": "worker", "model": "claude-sonnet-5-5", "harness": "api", "effort": "default",
         "reads": ["question"], "writes": ["sources"]},
        {"id": "extract", "role": "worker", "model": "claude-opus-5-5", "harness": "api", "effort": "high",
         "reads": ["sources"], "writes": ["claims"]},
        {"id": "synthesize", "role": "worker", "model": "gpt-6-astra", "harness": "api", "effort": "high",
         "reads": ["claims", "question"], "writes": ["brief"]},
        {"id": "critic", "role": "reviewer", "model": "claude-haiku-5-5", "harness": "api", "effort": "default",
         "reads": ["brief", "claims"], "writes": [{"id": "verdict", "kind": "verdict"}]},
    ],
    "checks": {"gates": [{"after": "extract", "rule": "command:claims-check", "on_fail": "extract"},
                         {"after": "critic", "on_fail": "synthesize"}],
               "budget_rounds": 2, "rescue": "redo_usual"},
    "judging": {
        "judges": [{"id": "j-claude", "kind": "model", "model": {"id": "claude-sonnet-5-5"}, "rubric": "rubric.md#v1"},
                   {"id": "j-astra", "kind": "model", "model": {"id": "gpt-6-astra"}, "rubric": "rubric.md#v1"},
                   {"id": "person", "kind": "person", "rate": 0.2}],
        "accept": {"name": "quality>=0.7+claims-check",
                   "definition": "quality at least 0.7 and the claims check passes", "requires": ["claims-check"],
                   "score": {"name": "quality", "target": 0.7, "better": "higher", "scale": "fraction"}},
    },
    "objective": {"score": "quality"},
    "hypothesis": {"text": "extract can run on Sonnet without hurting the brief; the critic adds little",
                   "edits": [{"kind": "model", "piece": "extract", "value": {"id": "claude-sonnet-5-5"}},
                             {"kind": "drop", "piece": "critic"}]},
    "allow": {
        "pieces": {"extract": {"models": [{"id": "claude-sonnet-5-5"}, {"id": "claude-haiku-5-5"}]},
                   "synthesize": {"efforts": ["medium", "high"]},
                   "critic": {"drop": True},
                   "scout": {"widths": [1, 2], "select": {"harness": "api", "model": {"id": "claude-haiku-5-5"}}}},
        "budgets": [1, 2, 3],
        "merges": [{"id": "extract+synthesize", "replace": ["extract", "synthesize"],
                    "into": {"id": "extract_synthesize", "role": "worker"},
                    "setting": {"harness": "api", "model": {"id": "claude-opus-5-5"}, "effort": "high",
                                "options": {"prompt": "merged-v1"}}}],
    },
    "inputs": {"ref": "questions.jsonl", "type": "research", "ids": _IDS, "holdout": _IDS[12:]},
}
OBJECTIVE_DEFAULTS = {"kind": "cheapest_within_margin", "margin": 0.10, "reference": "base"}
NO_RESCUE = "none"  # what a null checks.rescue gives
_EOF = object()

# An answer record: its required members, then its optional ones.
_RECORDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "purpose": (("id", "title", "example_input"), ()),
    "step": (("id", "role", "model", "harness", "effort", "reads", "writes"),
             ("width", "context_policy", "options")),
    "write": (("id",), ("kind",)),
    "checks": (("budget_rounds", "rescue"), ("gates", "checks")),
    "gate": (("after",), ("id", "rule", "on_fail")),
    "judging": (("judges", "accept"), ()),
    "objective": (("score",), ("kind", "margin", "reference", "cost_cap_usd")),
    "hypothesis": (("text", "edits"), ()),
    "edit": (("kind",), ("piece", "key", "value")),
    "inputs": (("ref", "type", "ids", "holdout"), ("n",)),
}
_INPUTS_ORDER = ("ref", "type", "n", "ids", "holdout", "ext")  # the spec example's order: n after type
_GIVEN_ORDER = {("producer", "capabilities")}  # flags, kept in their given order as the spec example writes them


class AnswersError(ValueError):
    """Answers refused: `problems` holds one `path: reason` each."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


class Incomplete(Exception):
    """The terminal closed before every question had an answer: `missing` holds the keys still to give."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        super().__init__(", ".join(missing))
        self.missing = tuple(missing)


# ------------------------------------------------------------------ the OCP 0.4 schema, as a guide

@functools.cache
def _schema() -> dict[str, Any]:
    return json.loads(conformance.SCHEMA_PATHS["0.4"].read_text(encoding="utf-8"))


def _node(node: Any) -> tuple[dict[str, Any] | None, Any, Any, str | None]:
    """A schema node's member schemas, its schema for other members, its item schema and its definition name.
    A `$ref` is followed; the object branches of an `anyOf` are joined."""
    name = None
    while isinstance(node, dict) and "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        node = _schema()["$defs"][name]
    if not isinstance(node, dict):
        return None, None, None, name
    props = dict(node["properties"]) if isinstance(node.get("properties"), dict) else None
    other = node.get("additionalProperties") if isinstance(node.get("additionalProperties"), dict) else None
    items = node.get("items") if isinstance(node.get("items"), dict) else None
    for branch in node.get("anyOf") or ():
        b_props, b_other, b_items, _ = _node(branch)
        if b_props is not None:
            props = {**(props or {}), **{k: v for k, v in b_props.items() if k not in (props or {})}}
        other, items = other or b_other, items or b_items
    return props, other, items, name


def in_schema_order(value: Any, node: Any = None) -> Any:
    """A copy of `value` whose objects list the members the OCP 0.4 schema defines in its order, then any others
    in their given order. As in the spec's example, a flow's inputs have `n` after `type`, and the producer's
    capabilities keep their given order. Open maps keep their order. Nothing is dropped or changed."""
    props, other, items, name = _node(_schema() if node is None else node)
    if isinstance(value, dict):
        order = list(_INPUTS_ORDER if name == "flowInputs" else (props or {}))
        keys = [k for k in order if k in value] + [k for k in value if k not in order]
        return {k: copy.deepcopy(value[k]) if (name, k) in _GIVEN_ORDER
                else in_schema_order(value[k], (props or {}).get(k, other)) for k in keys}
    if isinstance(value, list):
        return [in_schema_order(v, items) for v in value]
    return value


_TYPES = {"object": "an object", "array": "a list", "string": "a string", "number": "a number",
          "integer": "an integer", "boolean": "true or false", "null": "null"}


def _resolved(node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        node = _schema()["$defs"][node["$ref"].rsplit("/", 1)[-1]]
    return node


def _is_type(value: Any, name: str) -> bool:
    """JSON Schema's test of a type: a bool is no number, and a float with no fraction is an integer."""
    if name in ("number", "integer"):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return name == "number" or isinstance(value, int) or value.is_integer()
    return isinstance(value, {"object": dict, "array": list, "string": str, "boolean": bool,
                              "null": type(None)}[name])


def _type_names(node: Any) -> list[str]:
    names = node.get("type") if isinstance(node, dict) else None
    return [names] if isinstance(names, str) else list(names or ())


def _by_schema(value: Any, node: Any, where: str, problems: list[str]) -> None:
    """`value` held to its OCP 0.4 schema node, by path, as the document's schema check will hold it: a member
    the node does not define, a required member missing, a value of another type, outside the node's enum or
    minimum or of the wrong length, and a null, which has no TOML form. An open map (`additionalProperties`)
    is held member by member; a node that defines no members takes any. With `anyOf`, one branch must take
    the value; else the problems are those of the first branch of the value's type."""
    node = _resolved(node)
    if value is None:
        problems.append(f"{where}: null has no form in the flow's tables; leave the member out")
        return
    if not isinstance(node, dict):
        return
    branches = [_resolved(b) for b in node.get("anyOf") or ()]
    tries = []
    for branch in branches:
        tries.append([])
        _by_schema(value, branch, where, tries[-1])
    if tries and all(tries):
        fitting = [t for b, t in zip(branches, tries) if any(_is_type(value, n) for n in _type_names(b))]
        problems += (fitting or tries)[0]
        return
    types = _type_names(node)
    if types and not any(_is_type(value, n) for n in types):
        words = [_TYPES[n] for n in types if n != "null"] or ["null"]
        problems.append(f"{where}: expected {' or '.join(words)}, got {_shown(value)}")
        return
    if "enum" in node and not any(value == e and isinstance(value, bool) == isinstance(e, bool)
                                  for e in node["enum"]):
        problems.append(f"{where}: expected one of {', '.join(_shown(e) for e in node['enum'])}, "
                        f"got {_shown(value)}")
    if "minimum" in node and _is_type(value, "number") and value < node["minimum"]:
        problems.append(f"{where}: expected at least {node['minimum']}, got {_shown(value)}")
    if isinstance(value, str) and len(value) < node.get("minLength", 0):
        least = node["minLength"]
        problems.append(f"{where}: expected a string of at least {least} character{'s' * (least != 1)}, "
                        f"got {_shown(value)}")
    if isinstance(value, str) and len(value) > node.get("maxLength", len(value)):
        problems.append(f"{where}: expected at most {node['maxLength']} characters, got {len(value)}")
    if isinstance(value, dict):
        props = node.get("properties") if isinstance(node.get("properties"), dict) else None
        other = node.get("additionalProperties") if isinstance(node.get("additionalProperties"), dict) else None
        problems += [f"{where}.{k}: missing" for k in node.get("required") or () if k not in value]
        for k, v in value.items():
            if props is not None and k not in props:
                problems.append(f"{where}.{k}: unknown key; {where} takes {', '.join(props)}")
            elif props is not None or other is not None:
                _by_schema(v, props[k] if props is not None else other, f"{where}.{k}", problems)
            elif v is None:
                problems.append(f"{where}.{k}: null has no form in the flow's tables; leave the member out")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _by_schema(v, node.get("items"), f"{where}[{i}]", problems)


def _def(name: str) -> dict[str, str]:
    return {"$ref": f"#/$defs/{name}"}


# ------------------------------------------------------------------ checking the answers

def _shown(value: Any) -> str:
    """A value in a refusal: its JSON, with what JSON leaves as it is and still ends a line escaped too (U+007F to
    U+009F, U+2028 and U+2029), so the message stays one line wherever it is read: `--json` reads stderr line by
    line."""
    text = _one_line(json.dumps(value, ensure_ascii=False))
    return text if len(text) <= 100 else text[:97] + "..."


def _at(where: str, key: str) -> str:
    return f"{where}.{key}" if where else key


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_number(value: Any) -> bool:
    """A number a float can hold: the planner reads the objective's numbers as floats, so 10**400 is refused."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _is_count(value: Any, least: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _record(value: Any, name: str, where: str, problems: list[str]) -> dict[str, Any] | None:
    """The record when it is an object; its missing and unknown members are refused by name."""
    required, optional = _RECORDS[name]
    if not isinstance(value, dict):
        problems.append(f"{where}: expected an object with {', '.join(required + optional)}, got {_shown(value)}")
        return None
    problems += [f"{_at(where, k)}: missing" for k in required if k not in value]
    problems += [f"{_at(where, k)}: unknown key; {where} takes {', '.join(required + optional)}"
                 for k in value if k not in required + optional]
    return value


def _want(record: dict[str, Any], key: str, where: str, problems: list[str], ok: Any, what: str) -> None:
    if key in record and not ok(record[key]):
        problems.append(f"{_at(where, key)}: expected {what}, got {_shown(record[key])}")


def _purpose(value: Any, where: str, problems: list[str]) -> None:
    v = _record(value, "purpose", where, problems)
    if v is not None:
        _want(v, "id", where, problems, _is_text, "the flow id, a string")
        _want(v, "title", where, problems, lambda x: x is None or isinstance(x, str), "a string or null")
        _want(v, "example_input", where, problems, lambda x: isinstance(x, str), "a string")


def _steps(value: Any, where: str, problems: list[str], run_options: bool = False) -> None:
    if not isinstance(value, list) or not value:
        problems.append(f"{where}: expected a list of steps in run order, at least one, got {_shown(value)}")
        return
    ids: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for i, step in enumerate(value):
        at = f"{where}[{i}]"
        s = _record(step, "step", at, problems)
        if s is None:
            continue
        for key in ("id", "role", "model", "harness", "effort", "context_policy"):
            _want(s, key, at, problems, _is_text, "a string")
        _want(s, "width", at, problems, lambda x: _is_count(x, 1), "an integer, 1 or more")
        if _is_text(s.get("id")):
            if s["id"] in ids:
                problems.append(f"{at}.id: {_shown(s['id'])} is also {where}[{ids[s['id']]}]'s id")
            ids.setdefault(s["id"], i)
        _reads_writes(s, at, problems, kinds)
        options = s.get("options", {})
        if not isinstance(options, dict):
            problems.append(f"{at}.options: expected an object of option names to string values")
            continue
        for name, v in options.items():
            if not isinstance(v, str) and not run_options:
                problems.append(f"{at}.options.{name}: option values are strings, got {_shown(v)}")


def _reads_writes(s: dict[str, Any], at: str, problems: list[str], kinds: dict[str, str]) -> None:
    """A step's `reads` and `writes`: lists of artifact ids, a write also `{id, kind}`, one kind per artifact over
    the steps (`kinds`)."""
    for key in ("reads", "writes"):
        if key in s and not isinstance(s[key], list):
            problems.append(f"{at}.{key}: expected a list of artifact ids, got {_shown(s[key])}")
            continue
        for j, item in enumerate(s.get(key) or ()):
            item_at = f"{at}.{key}[{j}]"
            if key == "writes" and isinstance(item, dict):
                w = _record(item, "write", item_at, problems)
                if w is None:
                    continue
                _want(w, "id", item_at, problems, _is_text, "an artifact id")
                _want(w, "kind", item_at, problems, _is_text, "an artifact kind")
                if _is_text(w.get("id")) and _is_text(w.get("kind")):
                    if kinds.setdefault(w["id"], w["kind"]) != w["kind"]:
                        problems.append(f"{item_at}.kind: {w['id']} is already written as kind "
                                        f"{_shown(kinds[w['id']])}")
            elif not _is_text(item):
                problems.append(f"{item_at}: expected an artifact id" + (" or {id, kind}" if key == "writes"
                                                                           else "") + f", got {_shown(item)}")


def _given_steps(value: Any, where: str, problems: list[str], given: Any) -> None:
    """Steps answered after inferred runs: `{id, reads, writes}` each, one per step of the runs in their order. A
    member the runs give is refused at its path, as is a wrong, missing or extra step."""
    ids = [step["id"] for step in given.steps]
    order = f"{', '.join(ids)}, in that order"
    if not isinstance(value, list) or not value:
        problems.append(f"{where}: expected a list of {{id, reads, writes}}, one per step of the runs ({order}), "
                        f"got {_shown(value)}")
        return
    kinds: dict[str, str] = {}
    for i, step in enumerate(value):
        at = f"{where}[{i}]"
        if not isinstance(step, dict):
            problems.append(f"{at}: expected an object with id, reads, writes, got {_shown(step)}")
            continue
        problems += [f"{at}.{k}: missing" for k in ("id", "reads", "writes") if k not in step]
        for k in step:
            if k in GIVEN_STEP:
                problems.append(f"{at}.{_one_line(k)}: the runs give it (how inferred); {at} takes id, reads, writes")
            elif k not in ("id", "reads", "writes"):
                problems.append(f"{at}.{_one_line(k)}: unknown key; {at} takes id, reads, writes")
        if i >= len(ids):
            problems.append(f"{at}: the runs give {len(ids)} steps ({order}); leave this one out")
        elif "id" in step and step["id"] != ids[i]:
            problems.append(f"{at}.id: expected {_shown(ids[i])}, the runs' step {i + 1}, got {_shown(step['id'])}")
        _reads_writes(step, at, problems, kinds)
    if len(value) < len(ids):
        problems.append(f"{where}: {', '.join(ids[len(value):])} missing; give one entry per step of the runs "
                        f"({order})")


def _checks(value: Any, where: str, problems: list[str]) -> None:
    v = _record(value, "checks", where, problems)
    if v is None:
        return
    if "gates" in v and "checks" in v:  # an empty list states its form too
        problems.append(f"{where}: give gates or checks, not both")
    elif "gates" not in v and "checks" not in v:
        problems.append(f"{where}.gates: missing; give gates (gate tables) or checks (check tables)")
    first: dict[str, str] = {}
    for key in ("gates", "checks"):
        if key in v and not isinstance(v[key], list):
            problems.append(f"{where}.{key}: expected a list of tables, got {_shown(v[key])}")
            continue
        for i, table in enumerate(v.get(key) or ()):
            at = f"{where}.{key}[{i}]"
            if key == "checks":
                if not isinstance(table, dict):
                    problems.append(f"{at}: expected a check table, got {_shown(table)}")
                else:
                    _by_schema(table, _def("check"), at, problems)
                continue
            g = _record(table, "gate", at, problems)
            if g is None:
                continue
            for name in ("after", "id", "rule"):
                _want(g, name, at, problems, _is_text, "a string")
            _want(g, "on_fail", at, problems, lambda x: x is None or _is_text(x), "a piece id, or null to stop")
            after = g.get("after")
            if not _is_text(after):
                continue
            if after in first and "id" not in g:
                problems.append(f"{at}: a second gate after {after} needs an id ({first[after]} is the first one's)")
            first.setdefault(after, g.get("id") if _is_text(g.get("id")) else f"g_{after}")
    _want(v, "budget_rounds", where, problems, lambda x: _is_count(x, 1), "an integer, 1 or more")
    _want(v, "rescue", where, problems, lambda x: x is None or _is_text(x), "a named rescue or null")


def _local(findings: list[Any], where: str, members: dict[str, str], problems: list[str]) -> None:
    """The errors the flow's own rules find in one answer, at the answer's paths: `$['flow']['judges'][3]['rate']`
    is `judging.judges[3].rate`. So a terminal answer is asked again at once; `build` checks the whole document."""
    for f in findings:
        if f.level != "error" or not f.path.startswith(conformance.FLOW + "['"):
            continue
        head, _, tail = f.path[len(conformance.FLOW) + 2:].partition("']")
        path = where + (f".{members[head]}" if members.get(head) else "") + re.sub(r"\['([^']*)'\]", r".\1", tail)
        problems.append(f"{path}: {f.message} ({f.code})")


def _judging(value: Any, where: str, problems: list[str]) -> None:
    start = len(problems)
    v = _record(value, "judging", where, problems)
    if v is None:
        return
    if "judges" in v:
        if not isinstance(v["judges"], list):
            problems.append(f"{where}.judges: expected a list of judges, got {_shown(v['judges'])}")
        else:
            for i, judge in enumerate(v["judges"]):
                if not isinstance(judge, dict):
                    problems.append(f"{where}.judges[{i}]: expected a judge table, got {_shown(judge)}")
                else:
                    _by_schema(judge, _def("judge"), f"{where}.judges[{i}]", problems)
    if "accept" in v:
        if not isinstance(v["accept"], dict):
            problems.append(f"{where}.accept: expected the acceptance rule, got {_shown(v['accept'])}")
        else:
            _by_schema(v["accept"], _def("acceptanceRule"), f"{where}.accept", problems)
    if len(problems) == start:  # E214, and E215's judge scores against the verdicts the rule requires
        flow = {"judges": v["judges"], "acceptance_rule": v["accept"]}
        _local(conformance._judge_findings(flow) + conformance._objective_findings(flow), where,
               {"judges": "judges", "acceptance_rule": "accept"}, problems)


def _objective(value: Any, where: str, problems: list[str], judges: list[Any] | None = None) -> None:
    start = len(problems)
    v = _record(value, "objective", where, problems)
    if v is not None:
        for key in ("score", "kind", "reference"):
            _want(v, key, where, problems, _is_text, "a string")
        for key in ("margin", "cost_cap_usd"):
            _want(v, key, where, problems, _is_number, "a number a float can hold")
    if v is not None and len(problems) == start:  # E215 on the objective with its defaults
        _local(conformance._objective_findings({"objective": {**OBJECTIVE_DEFAULTS, **v}, "judges": judges}), where,
               {"objective": ""}, problems)


def _edit(value: Any) -> dict[str, Any]:
    """An edit as answered, with its null members left out."""
    return {k: v for k, v in value.items() if v is not None}


def _hypothesis(value: Any, where: str, problems: list[str]) -> None:
    if value is None:
        return
    v = _record(value, "hypothesis", where, problems)
    if v is None:
        return
    _want(v, "text", where, problems, lambda x: isinstance(x, str), "a string")
    if "edits" in v and not isinstance(v["edits"], list):
        problems.append(f"{where}.edits: expected a list of edits, got {_shown(v['edits'])}")
        return
    for i, edit in enumerate(v.get("edits") or ()):
        at = f"{where}.edits[{i}]"
        e = _record(edit, "edit", at, problems)
        if e is not None:
            _want(e, "kind", at, problems, _is_text, "an edit kind")
            for key in ("piece", "key"):
                _want(e, key, at, problems, lambda x: x is None or _is_text(x), "a string or null")


def _allow(value: Any, where: str, problems: list[str]) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        problems.append(f"{where}: expected flow.allow as an object, or null, got {_shown(value)}")
        return
    _by_schema(value, _def("flowAllow"), where, problems)


def _ref_problem(ref: str) -> str | None:
    if any(unicodedata.category(ch) == "Cc" for ch in ref):  # U+0000 to U+001F and U+007F to U+009F
        return "a path with a control character; give a plain path"
    if PurePosixPath(ref).is_absolute() or PureWindowsPath(ref).is_absolute() or ref.startswith("~"):
        return "an absolute path; give a path inside the flow folder"
    if ".." in PurePosixPath(ref.replace("\\", "/")).parts:
        return "a path with '..' leaves the flow folder; give a path inside it"
    return None


def _inputs(value: Any, where: str, problems: list[str]) -> None:
    start = len(problems)
    v = _record(value, "inputs", where, problems)
    if v is None:
        return
    _want(v, "ref", where, problems, _is_text, "a path inside the flow folder")
    if _is_text(v.get("ref")) and _ref_problem(v["ref"]):
        problems.append(f"{where}.ref: {_shown(v['ref'])} is {_ref_problem(v['ref'])}")
    _want(v, "type", where, problems, lambda x: x is None or _is_text(x), "a task type or null")
    for key in ("ids", "holdout"):
        _want(v, key, where, problems, lambda x: isinstance(x, list) and all(_is_text(i) for i in x),
              "a list of input ids")
    _want(v, "n", where, problems, lambda x: _is_count(x, 0), "an integer, 0 or more")
    if _is_count(v.get("n"), 0) and isinstance(v.get("ids"), list) and v["n"] != len(v["ids"]):
        problems.append(f"{where}.n: {v['n']} differs from the {len(v['ids'])} ids; leave it out, it is counted")
    if len(problems) == start:  # E216: no id twice, and the holdout among the ids
        _local(conformance._inputs_findings({"inputs": v}), where, {"inputs": ""}, problems)


_CHECKS = {"purpose": _purpose, "steps": _steps, "checks": _checks, "judging": _judging, "objective": _objective,
           "hypothesis": _hypothesis, "allow": _allow, "inputs": _inputs}


def check_answer(key: str, value: Any, before: dict[str, Any] | None = None, given: Any = None, *,
                 run_options: bool = False) -> list[str]:
    """The problems of one answer, by path; none when it is accepted. `before` holds the answers given with it or
    before it: the objective's margin is checked at the judges of its judging answer (1 judge without one). After
    inferred runs (`given`), the steps are answered `{id, reads, writes}`. With `run_options` the steps' option
    values are the runs' (`completed`), taken as given."""
    problems: list[str] = []
    if key == "objective":
        _objective(value, key, problems, _judges(before))
    elif key == "steps" and given is not None and given.how != "declared":
        _given_steps(value, key, problems, given)
    elif key == "steps" and run_options:
        _steps(value, key, problems, run_options=True)
    else:
        _CHECKS[key](value, key, problems)
    return [_one_line(p) for p in problems]  # a key of the answer is in the path as given


def _judges(answers: dict[str, Any] | None) -> list[Any]:
    """The judges of the judging answer, as given; none when there is no such list."""
    judging = answers.get("judging") if isinstance(answers, dict) else None
    judges = judging.get("judges") if isinstance(judging, dict) else None
    return judges if isinstance(judges, list) else []


def keys(given: Any = None) -> tuple[str, ...]:
    """The keys the answers take, in question order: all eight, or after declared runs all but steps and checks."""
    if given is not None and given.how == "declared":
        return tuple(k for k in KEYS if k not in DECLARED_KEYS)
    return KEYS


def missing(answers: dict[str, Any], given: Any = None) -> list[str]:
    """The keys not given, in question order."""
    return [k for k in keys(given) if k not in answers]


def _unknown(key: str, taken: tuple[str, ...], given: Any) -> str:
    if given is not None and given.how == "declared" and key in DECLARED_KEYS:
        return f"{key}: the runs declare it ({given.configuration.get('id')}, how declared); leave it out"
    return f"{_one_line(key)}: unknown key; the answers take {', '.join(taken)}"


def check(answers: Any, given: Any = None, *, run_options: bool = False) -> list[str]:
    """The problems of the answers, by path: missing and unknown keys, then each answer's own. After a prefill
    (`given`), the answers take `keys(given)`. With `run_options`, as `check_answer`."""
    taken = keys(given)
    if not isinstance(answers, dict):
        return [f"the answers: expected a JSON object with the keys {', '.join(taken)}, got {_shown(answers)}"]
    problems = [f"{k}: missing (question {KEYS.index(k) + 1})" for k in missing(answers, given)]
    problems += [_unknown(k, taken, given) for k in answers if k not in taken]
    for key in taken:
        if key in answers:
            problems += check_answer(key, answers[key], answers, given, run_options=run_options)
    return problems


# ------------------------------------------------------------------ the tables and the document

def _workflow_tables(purpose: dict[str, Any], steps: list[dict[str, Any]], checks: dict[str, Any]) -> dict[str, Any]:
    """The workflow part of the tables: the workflow file format's members and the settings."""
    kinds: dict[str, str | None] = {}  # artifact ids, in first-appearance order: each step's reads, then writes
    edges: list[list[str]] = []
    for step in steps:
        for aid in step["reads"]:
            kinds.setdefault(aid, None)
            edges.append([aid, step["id"]])
        for item in step["writes"]:
            aid, kind = (item["id"], item.get("kind")) if isinstance(item, dict) else (item, None)
            if kinds.get(aid) is None:
                kinds[aid] = kind
            edges.append([step["id"], aid])
    control: dict[str, Any] = {key: copy.deepcopy([{k: v for k, v in t.items() if v is not None}
                                                   if key == "gates" else t for t in checks[key]])
                               for key in ("gates", "checks") if key in checks}
    control["budget_rounds"] = checks["budget_rounds"]
    control["rescue"] = NO_RESCUE if checks["rescue"] is None else checks["rescue"]
    title = purpose["title"]
    data: dict[str, Any] = {"id": purpose["id"], "version": 1, **({"title": title} if title is not None else {})}
    data["pieces"] = [{"id": s["id"], "role": s["role"], "width": s.get("width", 1)} for s in steps]
    data["artifacts"] = [{"id": aid, "kind": kind or "other"} for aid, kind in kinds.items()]
    data["edges"] = edges
    data["control"] = control
    data["settings"] = {}
    for s in steps:
        setting = {"harness": s["harness"], "model": s["model"], "effort": s["effort"],
                   "context_policy": s.get("context_policy", "fresh")}
        options = _prune(s.get("options") or {})  # the rest goes to keep entries (`tables`)
        if options:
            setting["options"] = options
        data["settings"][s["id"]] = setting
    return data


def tables(answers: dict[str, Any], given: Any = None) -> dict[str, Any]:
    """The TOML view's tables the answers give, as `tomllib.loads` would read them; `check` passed. After
    declared runs (`given`) there is no workflow part: `[flow.configuration]` holds the runs' configuration less
    its id, which is computed. Inferred runs' steps are completed first (`completed`). What the tables cannot hold
    of the runs' values is `[[document.keep]]` entries (`document.keep`), as `flow_to_toml` writes them."""
    purpose, judging = answers["purpose"], answers["judging"]
    hypothesis, inputs = answers["hypothesis"], answers["inputs"]
    declared = given is not None and given.how == "declared"
    data = {} if declared else _workflow_tables(purpose, answers["steps"], answers["checks"])
    title = purpose["title"]
    flow: dict[str, Any] = {"id": purpose["id"], "version": 1, **({"title": title} if title is not None else {})}
    if hypothesis is not None:
        flow["hypothesis"] = hypothesis["text"]
    keep: list[dict[str, Any]] = []
    if declared:
        configuration = {k: copy.deepcopy(v) for k, v in given.configuration.items() if k != "id"}
        flow["configuration"] = _prune(configuration)
        _diff(configuration, flow["configuration"], ["flow", "configuration"], keep)
    else:
        flow["configuration"] = {"source": "usual"}
        for step in answers["steps"]:
            options = step.get("options") or {}
            _diff(options, _prune(options), ["flow", "configuration", "settings", step["id"], "options"], keep)
    flow["inputs"] = {"ref": inputs["ref"], **({"type": inputs["type"]} if inputs["type"] is not None else {}),
                      "n": len(inputs["ids"]), "ids": list(inputs["ids"]), "holdout": list(inputs["holdout"])}
    flow["accept"] = copy.deepcopy(judging["accept"])
    objective = {**OBJECTIVE_DEFAULTS, **answers["objective"]}
    flow["objective"] = {k: objective[k] for k in ("kind", "score", "margin", "reference", "cost_cap_usd")
                         if k in objective}
    flow["judges"] = copy.deepcopy(judging["judges"])
    if answers["allow"] is not None:
        flow["allow"] = copy.deepcopy(answers["allow"])
    edits = [Edit.from_dict(_edit(e)).to_dict() for e in (hypothesis or {}).get("edits") or ()]
    ext: dict[str, Any] = {EXT_HYPOTHESIS: edits} if edits else {}
    if given is not None:
        ext[EXT_PREFILL] = copy.deepcopy(given.ext)
    if ext:
        flow["ext"] = ext
    data["flow"] = flow
    if keep:
        data["document"] = {"keep": keep}
    return data


def completed(answers: dict[str, Any], given: Any = None) -> dict[str, Any]:
    """The answers with each step completed from inferred runs (`given`): the runs' role and setting, its option
    values as the runs give them, then the answer's id, reads and writes. Otherwise the answers as given."""
    if given is None or given.how == "declared":
        return answers
    steps = [{"id": answer["id"], "role": run["role"], "model": run["model"], "harness": run["harness"],
              "effort": run["effort"], "reads": answer["reads"], "writes": answer["writes"], "width": run["width"],
              "context_policy": run["context_policy"], "options": run["options"]}
             for run, answer in zip(given.steps, answers["steps"])]
    return {**answers, "steps": steps}


def _key(edit: Edit) -> str:
    return json.dumps(edit.to_dict(), sort_keys=True, ensure_ascii=False)


def declared_problem(doc: dict[str, Any], given: Any) -> str | None:
    """Why the flow's configuration is not the declared one exactly, as JSON with its id, or None when it is."""
    have, want = doc.get("flow", {}).get("configuration"), given.configuration
    if _same(have, want):
        return None
    at = ".".join(str(k) for k in _first_difference(want, have, [])) or "the whole object"
    return (f"the runs' configuration {_shown(want.get('id'))} does not come back exactly in the flow (first at "
            f"configuration.{at}); nothing is taken from it")


def build(answers: Any, given: Any = None) -> dict[str, Any]:
    """The flow document the answers make, after a prefill when `given`. Refused answers raise `AnswersError`
    with every problem by path."""
    problems = check(answers, given)
    if problems:
        raise AnswersError(problems)
    full = completed(answers, given)
    if full is not answers:  # the eight-key checks on the completed steps, the runs' options as given
        problems = check(full, run_options=True)
        if problems:
            raise AnswersError(problems)
    try:
        doc = flow_from_tables(tables(full, given))
    except ValueError as exc:
        raise AnswersError([f"the flow the answers make cannot be read: {exc}"]) from None
    if given is not None and given.how == "declared":
        problem = declared_problem(doc, given)
        if problem:
            raise AnswersError([problem])
    errors = [f for f in conformance.validate_doc(doc) if f.level == "error"]
    if errors:
        raise AnswersError([f"{f.code} at {f.path}: {f.message}" for f in errors])
    named = (answers["hypothesis"] or {}).get("edits") or []
    if named:
        try:
            arms, excluded = candidates(Flow.from_document(doc))
        except ValueError as exc:
            raise AnswersError([f"hypothesis.edits: the edits allow permits cannot be listed ({exc})"]) from None
        permitted = {_key(a.edits[0]) for a in arms} | {_key(x.edit) for x in excluded}
        problems = [f"hypothesis.edits[{i}]: {_shown(_edit(e))} is not one of the single edits allow permits "
                    "that change the base configuration" for i, e in enumerate(named)
                    if _key(Edit.from_dict(_edit(e))) not in permitted]
        if problems:
            raise AnswersError(problems)
    return doc


# ------------------------------------------------------------------ the walk on a terminal

def _block(stdin: TextIO) -> str | None:
    """Lines up to a blank line, leading blank lines skipped; None at the end of input with nothing read."""
    lines: list[str] = []
    while True:
        line = stdin.readline()
        if not line:
            return "\n".join(lines) if lines else None
        if line.strip():
            lines.append(line.rstrip("\n"))
        elif lines:
            return "\n".join(lines)


def _line(stdin: TextIO, stderr: TextIO, prompt: str) -> str | None:
    print(prompt, file=stderr)
    line = stdin.readline()
    return None if not line else line.rstrip("\n").strip()


def _fill(key: str, value: Any, stdin: TextIO, stderr: TextIO) -> Any:
    """Text members missing from an object answer, asked as bare text; for the judges, the rubric of the model
    judges and the person rate, asked once and filled in where absent. `_EOF` at the end of input."""
    if not isinstance(value, dict):
        return value
    texts = {"purpose": (("title", "a title (a blank line for none)"),
                         ("example_input", "one example input")),
             "hypothesis": (("text", "the hypothesis, in words"),)}.get(key, ())
    for member, what in texts:
        if member not in value:
            text = _line(stdin, stderr, f"{key}.{member}: {what}, as text:")
            if text is None:
                return _EOF
            value[member] = None if member == "title" and not text else text
    judges = value.get("judges") if key == "judging" else None
    if not isinstance(judges, list):
        return value
    judges = [j for j in judges if isinstance(j, dict)]
    if any(j.get("kind") == "model" and "rubric" not in j for j in judges):
        text = _line(stdin, stderr, "the rubric of the model judges, for example rubric.md#v1 (a blank line for "
                                    "none):")
        if text is None:
            return _EOF
        for j in judges:
            if text and j.get("kind") == "model" and "rubric" not in j:
                j["rubric"] = text
    while any(j.get("kind") == "person" and "rate" not in j for j in judges):
        text = _line(stdin, stderr, "the person rate: the share of runs a person scores, for example 0.2 (a blank "
                                    "line for all of them):")
        if text is None:
            return _EOF
        if not text:
            break
        try:
            rate = float(text)
        except ValueError:
            print(f"refused: {text!r} is not a number; asked again", file=stderr)
            continue
        for j in judges:
            if j.get("kind") == "person" and "rate" not in j:
                j["rate"] = rate
    return value


def _given_examples(examples: dict[str, Any], given: Any) -> dict[str, Any]:
    """The examples after inferred runs: the runs' step ids with empty reads and writes, and the checks with the
    proposed budget."""
    if given.how == "declared":
        return examples
    checks = examples.get("checks")
    return {**examples, "steps": [{"id": step["id"], "reads": [], "writes": []} for step in given.steps],
            **({"checks": {**checks, "budget_rounds": given.rounds}} if isinstance(checks, dict) else {})}


def ask(stdin: TextIO, stderr: TextIO, examples: dict[str, Any], given: Any = None) -> dict[str, Any]:
    """The answers, asked on a terminal, one question at a time with an example value. An answer is JSON
    ended by a blank line; a refused one is asked again with the reason. The end of input raises `Incomplete`
    with the keys still missing. After a prefill (`given`), its summary comes first and only `keys(given)` are
    asked, each under its question's number."""
    answers: dict[str, Any] = {}
    taken = keys(given)
    if given is not None:
        for line in given.summary():
            print(line, file=stderr)
        examples = _given_examples(examples, given)
    for number, (key, question) in enumerate(QUESTIONS, 1):
        if key not in taken:
            continue
        if key == "steps" and given is not None:
            question = STEPS_GIVEN
        while True:
            print(f"\n{number}. {question}", file=stderr)
            print(f"For example: {json.dumps(examples.get(key), ensure_ascii=False)}", file=stderr)
            print("Give the answer as JSON, then a blank line.", file=stderr)
            text = _block(stdin)
            if text is None:
                raise Incomplete(taken[taken.index(key):])
            try:
                value = json.loads(text)
            except ValueError as exc:  # also an integer of 4300+ digits
                print(f"refused: not JSON ({exc}); asked again", file=stderr)
                continue
            value = _fill(key, value, stdin, stderr)
            if value is _EOF:
                raise Incomplete(taken[taken.index(key):])
            problems = check_answer(key, value, answers, given)
            if not problems:
                answers[key] = value
                break
            for problem in problems:
                print(f"refused: {problem}", file=stderr)
            print("asked again", file=stderr)
    return answers
