"""`types.Workflow` and `Configuration` to and from OCP 2.2 to 2.4.

The forward direction (`workflow_to_ocp`, `settings_to_ocp`) is lane 01's
`loopmath.ocp.emit`; these functions call it. The inverse is this lane's.

Mapping:
- pieces `{id, role, width}`; roles verbatim, never rewritten;
- artifacts `{id, kind}`, kind from `Workflow.extra["artifact_kinds"]`, else the
  id when it is a recommended kind, else `other`;
- `control.gates` lists the piece after which each gate runs, `control.repair`
  is `{gate.after: gate.on_fail}`, `control.budget` is `budget_rounds` (K_max
  counting the first round; missing or 0 reads as 1), rescue `redo_usual` is
  `{kind: configuration, ref: usual}` (any other OCP rescue object is kept
  whole in `Control.rescue`); a gate rule that is not its role's default goes
  to `control.ext["dev.loopmath.gate_rules"]`;
- settings `{harness, model: {raw, id}, effort, context_policy, options}`,
  `context_policy` always written.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..ocp import emit as _emit
from ..types import Configuration, Control, Gate, Piece, Setting, Workflow
from .shapes import GATE_RULE_DEFAULTS

RECOMMENDED_KINDS = ("issue", "repo", "plan", "spec", "diff", "review", "verdict", "test_record", "report", "other")
GATE_RULES_KEY = "dev.loopmath.gate_rules"
CONFIG_SOURCES = ("usual", "alternative", "exploration", "user_edit", "habit", "designed")

_RESCUE_TO_OCP = {
    "redo_usual": {"kind": "configuration", "ref": "usual"},
    "person": {"kind": "person"},
    "none": {"kind": "none"},
}

Resolver = Callable[[str, int | None], Workflow | None]


class CycleError(ValueError):
    """The workflow graph has a cycle; repair loops belong in control."""


def default_gate_rule(role: str | None) -> str:
    return GATE_RULE_DEFAULTS.get(str(role or ""), "review_approve")


def artifact_kind(workflow: Workflow, artifact_id: str) -> str:
    kinds = (workflow.extra or {}).get("artifact_kinds") or {}
    kind = kinds.get(artifact_id)
    if kind:
        return str(kind)
    return artifact_id if artifact_id in RECOMMENDED_KINDS else "other"


def find_cycle(nodes: list[str], edges: list[tuple[str, str]]) -> list[str] | None:
    """One cycle as a node list, or None when the graph is acyclic."""
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    for a, b in edges:
        succ.setdefault(a, []).append(b)
        succ.setdefault(b, [])
    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(n: str) -> list[str] | None:
        state[n] = 1
        stack.append(n)
        for m in succ[n]:
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in state:
                found = visit(m)
                if found:
                    return found
        stack.pop()
        state[n] = 2
        return None

    for n in list(succ):
        if n not in state:
            found = visit(n)
            if found:
                return found
    return None


# ------------------------------------------------------------------ forward

def workflow_to_ocp(workflow: Workflow) -> dict[str, Any]:
    """The OCP 2.3 workflow, from lane 01's emitter; a cycle raises `CycleError`."""
    try:
        return _emit.workflow_to_ocp(workflow)
    except _emit.WorkflowCycleError as exc:
        raise CycleError(str(exc)) from exc


def settings_to_ocp(settings: Mapping[str, Setting]) -> dict[str, Any]:
    """OCP 2.2 settings, from lane 01's emitter (a kept model object goes back into `model`)."""
    return _emit.settings_to_ocp(dict(settings))


def rescue_to_ocp(rescue: Any) -> dict[str, Any]:
    """`redo_usual`, `person`, `none` to OCP 2.3 rescue objects; an OCP object passes through."""
    if isinstance(rescue, Mapping):
        return dict(rescue)
    return dict(_RESCUE_TO_OCP.get(str(rescue), {"kind": str(rescue)}))


def configuration_to_ocp(config: Configuration, *, source: str | None = None, rec: str | None = None) -> dict[str, Any]:
    """`run.configuration` (OCP 2.2) with the workflow inline, so E190 can check the id."""
    out: dict[str, Any] = {"id": config.id, "workflow": workflow_to_ocp(config.workflow),
                           "settings": settings_to_ocp(config.settings)}
    if source is not None:
        if source not in CONFIG_SOURCES:
            raise ValueError(f"configuration source must be one of {', '.join(CONFIG_SOURCES)}")
        out["source"] = source
    if rec is not None:
        out["rec"] = rec
    return out


# ------------------------------------------------------------------ inverse

def _model_id(model: Any) -> str:
    if isinstance(model, dict):
        return str(model.get("id") or model.get("raw") or "")
    return "" if model is None else str(model)


def setting_from_ocp(obj: Mapping[str, Any]) -> Setting:
    model = obj.get("model")
    extra = {k: v for k, v in obj.items() if k not in ("harness", "model", "effort", "context_policy", "options")}
    if isinstance(model, dict):
        extra["model_ref"] = dict(model)
    return Setting(
        harness=str(obj.get("harness") or ""),
        model=_model_id(model),
        effort=str(obj.get("effort") or "default"),
        context_policy=str(obj.get("context_policy") or "fresh"),
        options={str(k): str(v) for k, v in (obj.get("options") or {}).items()},
        extra=extra,
    )


def settings_from_ocp(obj: Mapping[str, Any] | None) -> dict[str, Setting]:
    return {str(pid): setting_from_ocp(s) for pid, s in (obj or {}).items() if isinstance(s, dict)}


def _rescue_from_ocp(obj: Any) -> str | dict[str, Any]:
    """The types rescue: one of the three words when the object is exactly its mapping, else the object."""
    if not isinstance(obj, dict):
        return "redo_usual"
    for word, mapped in _RESCUE_TO_OCP.items():
        if obj == mapped:
            return word
    return dict(obj)


def workflow_from_ocp(obj: Mapping[str, Any], resolver: Resolver | None = None) -> Workflow:
    """Read an OCP workflow (inline, or a `{ref, version}` resolved through `resolver`, default the catalog)."""
    if "ref" in obj and "pieces" not in obj:
        ref, version = str(obj["ref"]), obj.get("version")
        found = (resolver or catalog_resolver)(ref, int(version) if version is not None else None)
        if found is None:
            raise LookupError(f"workflow ref {ref!r} version {version} is not in the catalog")
        return found
    pieces = []
    for p in obj.get("pieces") or []:
        extra = {k: v for k, v in p.items() if k not in ("id", "role", "width")}
        pieces.append(Piece(id=str(p["id"]), role=str(p.get("role") or ""), width=int(p.get("width") or 1), extra=extra))
    roles = {p.id: p.role for p in pieces}
    artifacts, kinds = [], {}
    for a in obj.get("artifacts") or []:
        if isinstance(a, dict):
            artifacts.append(str(a["id"]))
            kinds[str(a["id"])] = str(a.get("kind") or "other")
        else:
            artifacts.append(str(a))
    control = obj.get("control") or {}
    repair = control.get("repair") or {}
    ext = control.get("ext") or {}
    rules = ext.get(GATE_RULES_KEY) or {}
    gates = tuple(
        Gate(id=f"g_{g}", after=str(g), rule=str(rules.get(g) or default_gate_rule(roles.get(g))),
             on_fail=repair.get(g))
        for g in control.get("gates") or []
    )
    rescue = _rescue_from_ocp(control.get("rescue"))
    control_extra = {k: v for k, v in control.items() if k not in ("gates", "repair", "budget", "rescue", "ext")}
    other_ext = {k: v for k, v in ext.items() if k != GATE_RULES_KEY}
    if other_ext:
        control_extra["ext"] = other_ext
    extra = {k: v for k, v in obj.items()
             if k not in ("id", "version", "title", "pieces", "artifacts", "edges", "control")}
    extra["artifact_kinds"] = kinds
    return Workflow(
        id=str(obj.get("id") or "workflow"),
        version=int(obj.get("version") or 1),
        title=str(obj.get("title") or ""),
        pieces=tuple(pieces),
        artifacts=tuple(artifacts),
        edges=tuple((str(a), str(b)) for a, b in obj.get("edges") or []),
        control=Control(gates=gates, budget_rounds=max(1, int(control.get("budget") or 0)), rescue=rescue,
                        extra=control_extra),
        extra=extra,
    )


def configuration_from_ocp(obj: Mapping[str, Any], resolver: Resolver | None = None) -> Configuration:
    """Read `run.configuration`. The id is recomputed; a different declared id is kept in `extra["declared_id"]`."""
    from .ids import config_id

    workflow = workflow_from_ocp(obj.get("workflow") or {}, resolver)
    settings = settings_from_ocp(obj.get("settings"))
    cid = config_id(workflow, settings)
    extra = {k: v for k, v in obj.items() if k not in ("id", "workflow", "settings")}
    declared = obj.get("id")
    if declared and declared != cid:
        extra["declared_id"] = declared
    return Configuration(id=cid, workflow=workflow, settings=settings, extra=extra)


def catalog_resolver(ref: str, version: int | None) -> Workflow | None:
    from .format import catalog

    wf = catalog().get(ref)
    if wf is None or (version is not None and wf.version != version):
        return None
    return wf


def is_types_workflow(wf: Any) -> bool:
    """True for a `types.Workflow.to_dict()` object, False for an OCP workflow (inline or `{ref, version}`).

    The two differ in `control` (`budget_rounds` and gate objects against
    `budget` and piece ids) and in `artifacts` (ids against `{id, kind}`).
    """
    if not isinstance(wf, Mapping) or "pieces" not in wf:
        return False
    control = wf.get("control") or {}
    if "budget_rounds" in control or any(isinstance(g, Mapping) for g in control.get("gates") or []):
        return True
    arts = wf.get("artifacts") or []
    return bool(arts) and all(isinstance(a, str) for a in arts)


def workflow_from_any(obj: Any, resolver: Resolver | None = None) -> Workflow:
    if isinstance(obj, Workflow):
        return obj
    if is_types_workflow(obj):
        return Workflow.from_dict(dict(obj))
    return workflow_from_ocp(obj, resolver)


def configuration_from_any(obj: Any, resolver: Resolver | None = None) -> Configuration:
    """A Configuration from itself, its `to_dict()` form, or an OCP `run.configuration` object."""
    if isinstance(obj, Configuration):
        return obj
    if not isinstance(obj, Mapping):
        raise TypeError(f"not a configuration: {type(obj).__name__}")
    if is_types_workflow(obj.get("workflow") or {}):
        from .ids import config_id

        cfg = Configuration.from_dict(dict(obj))
        cid = config_id(cfg.workflow, cfg.settings)
        return Configuration(id=cid, workflow=cfg.workflow, settings=cfg.settings, extra=dict(cfg.extra))
    return configuration_from_ocp(obj, resolver)
