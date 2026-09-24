"""Build OCP v0.3 documents and records for the store (spec 01, spec 03 section 4).

Owner: lane 01. Every helper accepts a `loopmath.types` dataclass or its
`to_dict()` form and returns plain JSON-ready dicts. Unknown keys a `types`
record carries in `extra` are kept, so a newer writer's fields survive.

The `types` to OCP mapping follows decisions D2, D3 and D29 to D32:

- pieces `{id, role, width}`, roles written verbatim (planner, implementer, ...);
- artifacts `{id, kind}`, kind from `Workflow.extra["artifact_kinds"]`, else the
  id when it is a recommended kind, else `other`;
- `control.gates` are the gated pieces (`Gate.after`), `control.repair` maps
  them to `Gate.on_fail`, a rule other than the role's default goes to
  `control.ext["dev.loopmath.gate_rules"]`; `budget` is `budget_rounds`
  (rounds including the first); `rescue` `redo_usual` is
  `{kind: configuration, ref: usual}`;
- settings `{harness, model: {raw, id}, effort, context_policy, options}`, with
  `context_policy` always written;
- a cyclic workflow raises `WorkflowCycleError` (repair loops live in control).
"""

from __future__ import annotations

import copy
from datetime import datetime
from typing import Any

from .. import __version__
from .canonical import GATE_RULES_KEY, config_id
from .version import version_at_least

OCP_VERSION = "0.3"
PRODUCER_NAME = "loopmath"
ARTIFACT_KINDS = ("issue", "repo", "plan", "spec", "diff", "review", "verdict", "test_record", "report", "other")
DEFAULT_GATE_RULE = {"reviewer": "review_approve", "referee": "referee_pick", "tester": "tests_pass"}
RESCUE_TO_OCP = {
    "redo_usual": {"kind": "configuration", "ref": "usual"},
    "person": {"kind": "person"},
    "none": {"kind": "none"},
}
TASK_SOURCE_KIND = {"orchestrator": "live", "onboard": "history", "benchmark": "benchmark", "repo_history": "backlog"}
NODE_KIND_BY_ROLE = {
    "planner": "plan", "plan": "plan",
    "implementer": "impl", "implement": "impl", "worker": "impl",
    "reviewer": "review", "review": "review",
    "tester": "test", "test": "test",
    "referee": "gate", "select": "gate",
    "docs": "docs",
}
CAPABILITIES = {
    "task": True, "configuration": True, "signals": True, "slate": True, "receipt": True,
    "edges_dep": True, "artifacts": True, "cost_usd": True, "cost_tokens": True, "outcome_evidence": True,
    "groups": False,
}


class WorkflowCycleError(ValueError):
    """A workflow whose edges form a cycle; the repair loop belongs in control.repair (D3)."""


def now() -> str:
    """The current time with the local offset, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _as_dict(value: Any) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, dict):
        return copy.deepcopy(value)
    raise TypeError(f"expected a loopmath.types record or a dict, got {type(value).__name__}")


def _drop_none(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if v is not None}


# Workflows, settings, configurations ---------------------------------------------------------

def _is_ocp_workflow(w: dict[str, Any]) -> bool:
    control = w.get("control") if isinstance(w.get("control"), dict) else {}
    return (
        any(isinstance(a, dict) for a in w.get("artifacts") or [])
        or "budget" in control or "repair" in control
        or any(isinstance(g, str) for g in control.get("gates") or [])
    )


def _check_acyclic(w: dict[str, Any]) -> None:
    graph: dict[str, list[str]] = {}
    for edge in w.get("edges") or []:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            graph.setdefault(str(edge[0]), []).append(str(edge[1]))
    state: dict[str, int] = {}  # 1 visiting, 2 done

    def visit(vertex: str, trail: list[str]) -> None:
        state[vertex] = 1
        for nxt in graph.get(vertex, []):
            if state.get(nxt) == 1:
                cycle = trail[trail.index(nxt):] + [nxt] if nxt in trail else [vertex, nxt]
                raise WorkflowCycleError(
                    f"workflow {w.get('id')!r} has a cycle: {' -> '.join(cycle)}; "
                    "the repair loop belongs in control.repair, not in an edge"
                )
            if state.get(nxt) is None:
                visit(nxt, trail + [nxt])
        state[vertex] = 2

    for vertex in sorted(graph):
        if state.get(vertex) is None:
            visit(vertex, [vertex])


def workflow_to_ocp(workflow: Any) -> dict[str, Any]:
    """OCP 2.3 workflow from a `types.Workflow` (or its dict). An OCP-form dict is copied as is."""
    w = _as_dict(workflow)
    if _is_ocp_workflow(w):
        _check_acyclic(w)
        return w
    known = {"id", "version", "title", "pieces", "artifacts", "edges", "control", "artifact_kinds"}
    roles: dict[str, str] = {}
    pieces = []
    for piece in w.get("pieces") or []:
        p = _as_dict(piece)
        out: dict[str, Any] = {"id": p["id"]}
        if p.get("role") is not None:
            out["role"] = p["role"]
            roles[p["id"]] = p["role"]
        out["width"] = p.get("width", 1)
        out.update({k: v for k, v in p.items() if k not in ("id", "role", "width", "setting")})
        pieces.append(out)

    kinds = w.get("artifact_kinds") if isinstance(w.get("artifact_kinds"), dict) else {}
    artifacts = []
    for artifact in w.get("artifacts") or []:
        if isinstance(artifact, dict):
            artifacts.append(copy.deepcopy(artifact))
            continue
        kind = kinds.get(artifact) or (artifact if artifact in ARTIFACT_KINDS else "other")
        artifacts.append({"id": artifact, "kind": kind})

    control_in = w.get("control") if isinstance(w.get("control"), dict) else {}
    gates, repair, rules = [], {}, {}
    for gate in control_in.get("gates") or []:
        g = _as_dict(gate)
        piece = g.get("after")
        if piece is None:
            continue
        gates.append(piece)
        if g.get("on_fail"):
            repair[piece] = g["on_fail"]
        rule = g.get("rule")
        if rule and rule != DEFAULT_GATE_RULE.get(roles.get(piece, ""), "review_approve"):
            rules[piece] = rule
    control: dict[str, Any] = {"gates": gates, "repair": repair, "budget": control_in.get("budget_rounds", 1)}
    rescue = control_in.get("rescue", "redo_usual")
    control["rescue"] = copy.deepcopy(RESCUE_TO_OCP.get(rescue, {"kind": rescue})) if isinstance(rescue, str) \
        else copy.deepcopy(rescue)
    extras = {k: v for k, v in control_in.items() if k not in ("gates", "budget_rounds", "rescue")}
    # D41: merge the control's own ext with the gate rules; the gates are the truth for the rules
    ext = dict(extras.pop("ext", None) or {})
    ext.pop(GATE_RULES_KEY, None)
    if rules:
        ext[GATE_RULES_KEY] = rules
    control.update(extras)
    if ext:
        control["ext"] = ext

    out = {
        "id": w.get("id"),
        "version": w.get("version"),
        "title": w.get("title"),
        "pieces": pieces,
        "artifacts": artifacts,
        "edges": [list(e) for e in w.get("edges") or []],
        "control": control,
    }
    out = _drop_none(out)
    out.update({k: v for k, v in w.items() if k not in known})
    _check_acyclic(out)
    return out


def setting_to_ocp(setting: Any) -> dict[str, Any]:
    s = _as_dict(setting)
    # D52: a model object kept by setting_from_ocp (extra["model_ref"]) goes back into `model`
    ref = s.pop("model_ref", None)
    model = s.get("model")
    if isinstance(model, str):
        model = {"raw": model, **ref, "id": model} if isinstance(ref, dict) else {"raw": model, "id": model}
    out = {
        "harness": s.get("harness"),
        "model": model,
        "effort": s.get("effort", "default"),
        "context_policy": s.get("context_policy", "fresh"),
        "options": s.get("options") or {},
    }
    out = _drop_none(out)
    out.update({k: v for k, v in s.items() if k not in out and k != "model"})
    return out


def settings_to_ocp(settings: dict[str, Any]) -> dict[str, Any]:
    return {str(piece): setting_to_ocp(s) for piece, s in (settings or {}).items()}


def configuration_to_ocp(config: Any, *, source: str | None = None, rec: str | None = None) -> dict[str, Any]:
    """OCP 2.2 configuration. The id is always the canonical hash of what is written (E190)."""
    c = _as_dict(config)
    workflow = workflow_to_ocp(c["workflow"])
    settings = settings_to_ocp(c.get("settings") or {})
    out: dict[str, Any] = {"id": config_id(workflow, settings), "workflow": workflow, "settings": settings}
    for key, value in (("source", source), ("rec", rec)):
        if value is not None:
            out[key] = value
    out.update({k: v for k, v in c.items() if k not in ("id", "workflow", "settings") and k not in out})
    return out


# Task, rule, signals, preferences --------------------------------------------------------------

def task_to_ocp(task: Any) -> dict[str, Any]:
    """OCP 2.1 task from a `types.Task` (or its dict). Empty optional fields are left out."""
    t = _as_dict(task)
    out: dict[str, Any] = {"id": t["id"]}
    for key in ("title", "type", "subtype", "repo", "org", "base_commit"):
        if t.get(key):
            out[key] = t[key]
    if t.get("features"):
        out["features"] = dict(t["features"])
    source = t.get("source")
    if isinstance(source, str) and source:
        out["source"] = {"kind": TASK_SOURCE_KIND.get(source, source)}
    elif isinstance(source, dict):
        out["source"] = source
    labeled_by = t.get("labeled_by")
    if isinstance(labeled_by, str) and labeled_by:
        if labeled_by.startswith("model:"):
            out["labeled_by"] = {"how": "labeler", "model": labeled_by[len("model:"):]}
        else:
            out["labeled_by"] = {"how": labeled_by}
    elif isinstance(labeled_by, dict):
        out["labeled_by"] = labeled_by
    known = {"id", "title", "type", "subtype", "repo", "org", "base_commit", "features", "source", "labeled_by"}
    out.update({k: v for k, v in t.items() if k not in known})
    return out


def rule_to_ocp(rule: Any) -> dict[str, Any]:
    """OCP 2.6 acceptance rule from a `types.AcceptanceRule` (or its dict)."""
    r = _as_dict(rule)
    out: dict[str, Any] = {
        "name": r["name"],
        "definition": r.get("definition") or r["name"],
        "requires": list(r.get("requires", ("tests",))),
        "score": _drop_none(dict(r["score"])) if isinstance(r.get("score"), dict) else None,
        "excludes_events": list(r.get("excludes_events", ("revert", "incident"))),
        "window_days": r.get("window_days", 14),
    }
    out.update({k: v for k, v in r.items() if k not in out})
    return out


def signal_to_ocp(signal: Any) -> dict[str, Any]:
    """OCP 2.6 signal from a `types.Signal` (or its dict). `run` is implied by the file and dropped."""
    s = _as_dict(signal)
    s.pop("run", None)
    if s.get("kind") != "score":
        for key in ("unit", "better", "target", "scale"):
            s.pop(key, None)
    for key in ("unit", "better", "target", "at_attempt"):
        if s.get(key) is None:
            s.pop(key, None)
    if not s.get("source"):
        s["source"] = {"kind": "orchestrator"}
    if not s.get("observed_at"):
        s["observed_at"] = now()
    s.setdefault("tier", "reported")
    return s


def preference_record(*, pref_id: str, slate: str, winner: str, members: list[str], judge: str = "referee",
                      model: str | None = None, blinded: bool = False, observed_at: str | None = None,
                      tier: str = "reported") -> dict[str, Any]:
    """OCP 2.7 preference. `winner` is a member run id or 'tie'."""
    judge_record: dict[str, Any] = {"kind": judge, "blinded": blinded}
    if model:
        judge_record["model"] = model
    return {"id": pref_id, "slate": slate, "winner": winner, "members": list(members), "judge": judge_record,
            "observed_at": observed_at or now(), "tier": tier}


def attempt_record(*, attempt_id: str, node: str, harness: str, model: str | dict[str, Any], effort: str,
                   vertex: str | None = None, n: int | None = None, round: int | None = None,
                   cwd: str | None = None, session: str | None = None, cause: str | dict[str, Any] = "initial",
                   started_at: str | None = None, status: str = "working") -> dict[str, Any]:
    """OCP attempt with the v0.3 fields (vertex, round, cwd). Cost and outcome are added at finish."""
    out: dict[str, Any] = {
        "id": attempt_id,
        "node": node,
        "vertex": vertex or node,
        "n": n,
        "round": round,
        "harness": harness,
        "model": model if isinstance(model, dict) else {"raw": model, "id": model},
        "effort": effort,
        "cause": cause if isinstance(cause, dict) else {"type": cause},
        "status": status,
        "started_at": started_at or now(),
        "cwd": cwd,
        "session": session,
    }
    return _drop_none(out)


def add_event(doc: dict[str, Any], event_type: str, *, at: str | None = None, **fields: Any) -> dict[str, Any]:
    """Append an event (for example signal_observed, receipt_written, run_finished)."""
    event = {"at": at or now(), "type": event_type}
    event.update({k: v for k, v in fields.items() if v is not None})
    doc.setdefault("events", []).append(event)
    return event


def add_signal(doc: dict[str, Any], signal: Any) -> dict[str, Any]:
    """Append a signal to run.signals, with its signal_observed event."""
    record = signal_to_ocp(signal)
    doc["run"].setdefault("signals", []).append(record)
    add_event(doc, "signal_observed", at=record.get("observed_at"),
              detail=f"{record.get('kind')} {record.get('name')}")
    return record


# Whole documents ---------------------------------------------------------------------------------

def _provenance(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"kind": value}
    return _as_dict(value)


def _slate(value: Any, run_id: str, base_commit: str | None) -> dict[str, Any]:
    if isinstance(value, str):
        out: dict[str, Any] = {"id": value, "members": [run_id]}
        if base_commit:
            out["base_commit"] = base_commit
        return out
    return _as_dict(value)


def new_run_doc(*, run_id: str, task: Any, configuration: Any, acceptance_rule: Any, provenance: Any,
                slate: Any = None, source: str | None = None, rec: str | None = None,
                started_at: str | None = None, profile: str = "metadata_only") -> dict[str, Any]:
    """The OCP v0.3 skeleton `loopmath run start` writes (spec 02).

    producer loopmath, privacy `metadata_only`, run with task, configuration,
    provenance, acceptance rule and slate, one node per piece vertex (node id =
    vertex id), and a dep edge between pieces joined through an artifact.
    `slate` is a slate id (this run becomes its first member) or a slate object.
    `source` and `rec` go to configuration.source and configuration.rec.
    """
    at = started_at or now()
    task_ocp = task_to_ocp(task)
    config_ocp = configuration_to_ocp(configuration, source=source, rec=rec)
    run: dict[str, Any] = {
        "id": run_id,
        "started_at": at,
        "task": task_ocp,
        "configuration": config_ocp,
        "provenance": _provenance(provenance),
        "acceptance_rule": rule_to_ocp(acceptance_rule),
        "signals": [],
    }
    if slate is not None:
        run["slate"] = _slate(slate, run_id, task_ocp.get("base_commit"))

    workflow = config_ocp["workflow"]
    piece_ids = [p["id"] for p in workflow.get("pieces", [])]
    gates = (workflow.get("control") or {}).get("gates", [])
    nodes = []
    for piece in workflow.get("pieces", []):
        kind = NODE_KIND_BY_ROLE.get(piece.get("role", ""), "task")
        if piece["id"] in gates and kind not in ("review", "test"):
            kind = "gate"
        nodes.append({"id": piece["id"], "kind": kind, "vertex": piece["id"], "state": "queued"})
    producers: dict[str, list[str]] = {}
    consumers: dict[str, list[str]] = {}
    for frm, to in workflow.get("edges", []):
        if frm in piece_ids:
            producers.setdefault(to, []).append(frm)
        elif to in piece_ids:
            consumers.setdefault(frm, []).append(to)
    edges, seen = [], set()
    for artifact, makers in producers.items():
        for maker in makers:
            for user in consumers.get(artifact, []):
                if maker != user and (maker, user) not in seen:
                    seen.add((maker, user))
                    edges.append({"from": maker, "to": user, "kind": "dep", "tier": "reported"})

    return {
        "ocp": OCP_VERSION,
        "producer": {"name": PRODUCER_NAME, "version": __version__, "emitted_at": at,
                     "capabilities": dict(CAPABILITIES)},
        "privacy": {"profile": profile},
        "run": run,
        "nodes": nodes,
        "edges": edges,
        "attempts": [],
        "artifacts": [],
        "events": [{"at": at, "type": "note", "detail": "run started"}],
    }


def validate_strict(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every finding as a dict {level, code, path, message}; the caller fails on any 'error'.

    Strict means the document must declare OCP 0.3 (E005, a loopmath store
    rule, not an OCP rule) on top of every conformance rule.
    """
    from .conformance import validate_doc

    findings = [f._asdict() for f in validate_doc(doc)]
    if not isinstance(doc, dict) or doc.get("ocp") != OCP_VERSION:
        version = doc.get("ocp") if isinstance(doc, dict) else None
        hint = " (run 'loopmath ocp migrate')" if version_at_least(version, "0.1") else ""
        findings.insert(0, {"level": "error", "code": "E005", "path": "$['ocp']",
                            "message": f"the store takes OCP {OCP_VERSION} documents, got {version!r}{hint}"})
    return findings
