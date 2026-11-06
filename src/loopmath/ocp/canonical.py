"""Canonical JSON of a configuration and its `cfg_` id (spec 01 section 2.2).

This is the one definition of the configuration id. It works on the OCP v0.3
form of a workflow (section 2.3) and of the settings (section 2.4), so a
producer can compute the id from the spec alone; `loopmath.workflows.ids`
converts `types` values through `loopmath.ocp.emit` and calls `config_id`.

The canonical form keeps only the fields that define the shape and the
settings, so ids, titles, versions, `ext` and unknown fields never change it:

- workflow: `pieces` (`id`, `role`, `width` default 1, nested `workflow`),
  `artifacts` (`id`, `kind`), `edges`, `control` (`gates`, `repair`,
  `budget` read as max(1, budget) with 1 when absent since it counts round 1,
  `rescue` `kind` and `ref`, and the one extension key
  `dev.loopmath.gate_rules` when nonempty, since a non-default gate rule
  changes behaviour); pieces and artifacts sorted by id, edges and
  gates sorted; workflow `id`, `version`, `title` and every other `ext` key
  excluded;
- `control.checks` (OCP 0.4): while no piece has two or more checks, the
  checks become that same gates form (`gates` the `after` values, `repair`
  each non-null `on_fail`, `gate_rules` each rule other than the role
  default), so a workflow written either way has one id; otherwise control is
  `budget`, `rescue` and the checks as `{after, on_fail, rule}` sorted by
  `after`, `rule`, then `on_fail` with null first. Check ids and `ext` are
  excluded, and `gates`, `repair` and `gate_rules` are ignored next to checks;
- settings: piece id to `harness`, `model` (the model's `id`, else its `raw`),
  `effort` default "default", `context_policy` default "fresh",
  `options` default {};
- JSON with sorted keys, no whitespace, UTF-8.

The id is "cfg_" plus the first 12 hex digits of SHA-256 over that JSON.

The checks rule applies whatever the document version, since `config_id`
takes none and ids are also computed for workflows outside documents. OCP 0.3
does not define `control.checks` and no 0.3 producer writes it, so no 0.3 id
moves.

The flow digest of an OCP 0.4 flow document is "flw_" plus the first 12 hex
digits of SHA-256 over the same JSON encoding of `acceptance_rule`, `allow`,
`configuration` (its id), `hypothesis`, `inputs`, `judges` and `objective`,
each as written and null when absent. The flow's `id`, `version`, `title`,
`digest` and `ext` are left out.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

GATE_RULES_KEY = "dev.loopmath.gate_rules"
# A gate's rule by the role of its piece when gate_rules names none; any other role: review_approve
DEFAULT_GATE_RULE = {"reviewer": "review_approve", "referee": "referee_pick", "tester": "tests_pass"}

Resolver = Callable[[str, "int | None"], "dict[str, Any] | None"]


class UnresolvedWorkflow(ValueError):
    """A workflow given by reference that the caller could not resolve."""


def model_key(model: Any) -> str | None:
    """The model part of a setting: modelRef.id, else modelRef.raw (a bare string passes through)."""
    if isinstance(model, dict):
        for key in ("id", "raw"):
            value = model.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    return model if isinstance(model, str) else None


def canonical_setting(setting: Any) -> dict[str, Any]:
    s = setting if isinstance(setting, dict) else {}
    options = s.get("options")
    return {
        "harness": s.get("harness"),
        "model": model_key(s.get("model")),
        "effort": s.get("effort", "default"),
        "context_policy": s.get("context_policy", "fresh"),
        "options": options if isinstance(options, dict) else {},
    }


def is_workflow_ref(workflow: Any) -> bool:
    return isinstance(workflow, dict) and "ref" in workflow and "pieces" not in workflow


def canonical_workflow(workflow: Any, *, resolve: Resolver | None = None) -> dict[str, Any]:
    """The shape part of the canonical form. Raises UnresolvedWorkflow for a reference it cannot resolve."""
    if is_workflow_ref(workflow):
        resolved = resolve(workflow.get("ref"), workflow.get("version")) if resolve else None
        if resolved is None:
            raise UnresolvedWorkflow(f"workflow reference {workflow.get('ref')!r} is not resolved")
        workflow = resolved
    if not isinstance(workflow, dict):
        raise ValueError("workflow is not an object")

    pieces = []
    for piece in _dicts(workflow.get("pieces")):
        out: dict[str, Any] = {"id": piece.get("id"), "width": piece.get("width", 1)}
        if "role" in piece:
            out["role"] = piece["role"]
        if "workflow" in piece:
            out["workflow"] = canonical_workflow(piece["workflow"], resolve=resolve)
        pieces.append(out)
    artifacts = []
    for artifact in _dicts(workflow.get("artifacts")):
        out = {"id": artifact.get("id")}
        if "kind" in artifact:
            out["kind"] = artifact["kind"]
        artifacts.append(out)
    edges = [list(edge) for edge in workflow.get("edges") or [] if isinstance(edge, (list, tuple))]

    control = workflow.get("control") if isinstance(workflow.get("control"), dict) else {}
    budget = control.get("budget")
    budget = max(1, budget) if isinstance(budget, int) and not isinstance(budget, bool) else 1
    if isinstance(control.get("checks"), list):
        roles = {p["id"]: p.get("role") for p in _dicts(workflow.get("pieces")) if isinstance(p.get("id"), str)}
        canonical_control = _checks_control(control["checks"], roles, budget)
    else:
        canonical_control = {
            "gates": sorted(g for g in control.get("gates") or [] if isinstance(g, str)),
            "repair": dict(control.get("repair") or {}),
            "budget": budget,
        }
        ext = control.get("ext")
        if isinstance(ext, dict) and ext.get(GATE_RULES_KEY):  # empty gate rules: every gate uses its default
            canonical_control["gate_rules"] = ext[GATE_RULES_KEY]
    rescue = control.get("rescue")
    if isinstance(rescue, dict):
        canonical_control["rescue"] = {k: rescue[k] for k in ("kind", "ref") if k in rescue}

    return {
        "pieces": sorted(pieces, key=_sort_key),
        "artifacts": sorted(artifacts, key=_sort_key),
        "edges": sorted(edges, key=lambda e: [str(v) for v in e]),
        "control": canonical_control,
    }


def _checks_control(checks: list[Any], roles: dict[str, Any], budget: int) -> dict[str, Any]:
    """The canonical control of the checks form: the gates form while no piece has two or more checks."""
    read = [(c["after"], c.get("rule"), c.get("on_fail"))
            for c in _dicts(checks) if isinstance(c.get("after"), str)]
    afters = [after for after, _, _ in read]
    if len(set(afters)) < len(afters):
        ordered = sorted(read, key=lambda c: (c[0], str(c[1]), c[2] is not None, str(c[2])))
        return {"budget": budget, "checks": [{"after": a, "on_fail": f, "rule": r} for a, r, f in ordered]}
    out: dict[str, Any] = {
        "gates": sorted(afters),
        "repair": {after: on_fail for after, _, on_fail in read if on_fail is not None},
        "budget": budget,
    }
    rules = {}
    for after, rule, _ in read:
        role = roles.get(after)
        if rule != DEFAULT_GATE_RULE.get(role if isinstance(role, str) else "", "review_approve"):
            rules[after] = rule
    if rules:
        out["gate_rules"] = rules
    return out


def canonical_json(workflow: Any, settings: Any, *, resolve: Resolver | None = None) -> str:
    settings = settings if isinstance(settings, dict) else {}
    body = {
        "workflow": canonical_workflow(workflow, resolve=resolve),
        "settings": {str(k): canonical_setting(v) for k, v in settings.items()},
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_id(workflow: Any, settings: Any, *, resolve: Resolver | None = None) -> str:
    """'cfg_' + the first 12 hex digits of SHA-256 over the canonical JSON."""
    text = canonical_json(workflow, settings, resolve=resolve)
    return "cfg_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# The members of a flow that its digest covers; configuration enters as its id
FLOW_DIGEST_MEMBERS = ("acceptance_rule", "allow", "configuration", "hypothesis", "inputs", "judges", "objective")


def flow_digest(flow: dict[str, Any]) -> str:
    """'flw_' + the first 12 hex digits of SHA-256 over the canonical JSON of the flow's digest members."""
    body = {name: flow.get(name) for name in FLOW_DIGEST_MEMBERS}
    configuration = body["configuration"]
    body["configuration"] = configuration.get("id") if isinstance(configuration, dict) else None
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "flw_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _sort_key(record: dict[str, Any]) -> str:
    return str(record.get("id"))
