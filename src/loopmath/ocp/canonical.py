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
- settings: piece id to `harness`, `model` (the model's `id`, else its `raw`),
  `effort` default "default", `context_policy` default "fresh",
  `options` default {};
- JSON with sorted keys, no whitespace, UTF-8.

The id is "cfg_" plus the first 12 hex digits of SHA-256 over that JSON.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

GATE_RULES_KEY = "dev.loopmath.gate_rules"

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
    canonical_control: dict[str, Any] = {
        "gates": sorted(g for g in control.get("gates") or [] if isinstance(g, str)),
        "repair": dict(control.get("repair") or {}),
        "budget": max(1, budget) if isinstance(budget, int) and not isinstance(budget, bool) else 1,
    }
    rescue = control.get("rescue")
    if isinstance(rescue, dict):
        canonical_control["rescue"] = {k: rescue[k] for k in ("kind", "ref") if k in rescue}
    ext = control.get("ext")
    if isinstance(ext, dict) and ext.get(GATE_RULES_KEY):  # empty gate rules: every gate uses its default
        canonical_control["gate_rules"] = ext[GATE_RULES_KEY]

    return {
        "pieces": sorted(pieces, key=_sort_key),
        "artifacts": sorted(artifacts, key=_sort_key),
        "edges": sorted(edges, key=lambda e: [str(v) for v in e]),
        "control": canonical_control,
    }


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


def _dicts(value: Any) -> list[dict[str, Any]]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def _sort_key(record: dict[str, Any]) -> str:
    return str(record.get("id"))
