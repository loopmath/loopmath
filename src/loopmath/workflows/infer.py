"""Map an extracted session graph (or a run file) to the nearest shape with its settings.

`infer(source) -> (Configuration, confidence)`. The configuration's workflow is
the exact composed shape (`plan_team`, `implement_review`, ...); `extra` holds

- `node_vertex`: every node id of the input to a piece id of the returned
  workflow. Nodes that do not decide the shape (sessions outside the
  build phase, external launchers, harness gates) go to the nearest piece by
  role. For a declared configuration the targets are its own pieces, found by
  `node.vertex`, then attempt vertices, then role, and `node_vertex_basis`
  says which (see `declared_node_vertex`);
- `shape`, `nearest_catalog`, `confidence`, `reasons` (one line each)
  and `inferred_from` (`graph`, `ocp`, `sweep`, `rq1`, `configuration`).

Sources, one agent list for all of them:

- `graph.schema.Graph` or its dict (extractor payload);
- an OCP document of any version: a declared `run.configuration` is taken as
  is; a v0.2 document, or one carrying the graph writer's extension
  (`dev.loopmath.graph`, or `dev.dagr.graph` from before 0.3) at any version,
  goes through `ingest.ocp.from_ocp`; anything else is read from its
  nodes and attempts;
- a contract v3 sweep run file (`ext.experiment` declares arm, planner, reviewer);
- an RQ1 run `config.json` (`topology`, `agents`).

Shape rules (D33 front, middle, back): planners, and a lead that hands work to
workers, give `plan`; reviewers give `review`; workers that overlap in time
give `team` (`best_of_n` with a referee or a declared best-of), one worker at a
time gives `implement`; with no workers the main session implements. Each
piece's setting is the most common (harness, model, effort) of its sessions,
ties to the one that cost most. Confidence starts at 0.98 for declared
metadata, 0.95 for a declared configuration and 0.9 for extracted graphs, and
drops for heuristic or missing role labels, mixed settings in a piece, missing
timing and short model names.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..types import Configuration, Setting, Workflow
from .ids import make_config
from .models import family_of, harness_for, normalize_role
from .shapes import (DEFAULT_REVIEW_ROUNDS, MAX_ROUNDS, MAX_WIDTH, ShapeParams, build_shape, nearest_catalog,
                     normalize, shape_name, work_piece)

BUILD_PHASES = (None, "", "build")
WORKER_ROLES = ("worker", "implementer", "cli")
MAIN_ROLES = ("solo", "lead")
REVIEW_ROLES = ("reviewer", "tester")
SCORER = Setting(harness="command", model="score", effort="default")  # a best-of pick made by a scoring command
_KIND_ROLE = {"plan": "planner", "impl": "worker", "review": "reviewer", "test": "tester", "gate": "gate",
              "ops": "lead", "docs": "worker"}


class InferError(ValueError):
    """Nothing in the input says what ran (no sessions, or no model anywhere)."""


@dataclass
class Agent:
    """One session or attempt, whatever the source."""

    id: str
    role: str | None  # normalized: planner, worker, reviewer, tester, referee, lead, solo, gate, external, None
    harness: str | None = None
    model: str | None = None
    effort: str | None = None
    start: float | None = None
    end: float | None = None
    tier: str | None = None  # of the role label: verified, reported, heuristic
    phase: str | None = None
    source: str | None = None  # top, subagent, codex, external
    cost: float = 0.0
    short_model: bool = False


@dataclass
class Inference:
    configuration: Configuration
    confidence: float
    params: ShapeParams
    node_vertex: dict[str, str]
    reasons: list[str] = field(default_factory=list)
    inferred_from: str = "graph"

    @property
    def shape(self) -> str:
        return shape_name(self.params)

    @property
    def nearest_catalog(self) -> str:
        return nearest_catalog(self.params)


# ------------------------------------------------------------------ helpers

def _ts(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _canonical(model: str | None) -> str | None:
    if not model:
        return None
    try:
        from ..ingest.base import canonical_model
    except ImportError:  # pragma: no cover - the ingest package ships with loopmath
        return str(model)
    return canonical_model(str(model)) or str(model)


# Graph labeller words that mean something to the classifier; other roles pass through normalize_role.
_GRAPH_ROLES = {"lead": "lead", "solo": "solo", "cli": "cli", "external": "external", "dev": "worker",
                "developer": "worker", "harness": "gate", "gate": "gate"}


def _role(value: Any) -> str | None:
    if value is None or value == "":
        return None
    key = str(value).strip().lower()
    return _GRAPH_ROLES.get(key) or normalize_role(key)


def _overlap(agents: list[Agent]) -> int | None:
    """The most agents running at once, or None when timing is missing for any of them."""
    spans = [(a.start, a.end) for a in agents]
    if any(s is None or e is None for s, e in spans):
        return None
    events = sorted([(s, 1) for s, _ in spans] + [(e, -1) for _, e in spans], key=lambda x: (x[0], x[1]))
    best = cur = 0
    for _, step in events:
        cur += step
        best = max(best, cur)
    return best


# ------------------------------------------------------------------ adapters

def agents_from_graph(graph: Any) -> list[Agent]:
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else getattr(graph, "nodes", None)
    out = []
    for n in nodes or []:
        get = n.get if isinstance(n, Mapping) else (lambda k, _n=n: getattr(_n, k, None))
        start = _ts(get("ts"))
        wall = get("wall_s")
        out.append(Agent(
            id=str(get("id")), role=_role(get("role")), harness=get("harness"), model=_canonical(get("model")),
            effort=get("effort"), start=start, end=start + float(wall) if start is not None and wall is not None else None,
            tier=get("role_tier"), phase=get("phase"), source=get("source"), cost=float(get("usd") or 0.0)))
    return out


def _attempt_model(model: Any) -> tuple[str | None, bool]:
    if isinstance(model, Mapping):
        raw = model.get("id") or model.get("raw")
    else:
        raw = model
    if not raw:
        return None, False
    raw = str(raw)
    short = "·" in raw
    return _canonical(raw.split("·")[0]), short or ("-" not in raw and not raw.startswith("gpt"))


def agents_from_ocp(doc: Mapping[str, Any]) -> list[Agent]:
    """Nodes and their last attempt, for OCP documents `ingest.ocp.from_ocp` does not read."""
    last: dict[str, Mapping[str, Any]] = {}
    for a in doc.get("attempts") or []:
        if isinstance(a, Mapping) and isinstance(a.get("node"), str):
            if a["node"] not in last or (a.get("n") or 0) >= (last[a["node"]].get("n") or 0):
                last[a["node"]] = a
    out = []
    for node in doc.get("nodes") or []:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str):
            continue
        a = last.get(node["id"], {})
        role_obj = a.get("role")
        role = role_obj.get("value") if isinstance(role_obj, Mapping) else role_obj
        tier = role_obj.get("tier") if isinstance(role_obj, Mapping) else None
        if not role:
            role, tier = _KIND_ROLE.get(str(node.get("kind")), a.get("actor")), "reported"
        model, short = _attempt_model(a.get("model"))
        cost = a.get("cost") if isinstance(a.get("cost"), Mapping) else {}
        role = _role(role)
        out.append(Agent(
            id=node["id"], role=role, harness=a.get("harness"), model=model, effort=a.get("effort"),
            start=_ts(a.get("started_at")), end=_ts(a.get("ended_at")), tier=tier,
            phase="external" if role == "external" else "build", source=None,
            cost=float(cost.get("usd") or 0.0) if isinstance(cost.get("usd"), (int, float)) else 0.0,
            short_model=short))
    # As the extractor does: a session that starts after the lead ended is after the build.
    lead_end = max((a.end for a in out if a.role == "lead" and a.end is not None), default=None)
    if lead_end is not None:
        for a in out:
            if a.phase == "build" and a.start is not None and a.start > lead_end:
                a.phase = "post"
    return out


def _experiment_setting(obj: Mapping[str, Any] | None) -> Setting | None:
    if not isinstance(obj, Mapping) or not obj.get("model"):
        return None
    model = _canonical(obj["model"])
    family = obj.get("family")
    harness = {"claude": "claude-code", "codex": "codex"}.get(str(family)) or harness_for(model) or "unknown"
    return Setting(harness=harness, model=model, effort=str(obj.get("effort") or "default"))


def _resolve_short(raw: str | None, known: Iterable[Setting]) -> Setting | None:
    """`sol·xhigh` against the experiment's own models (same family), else the canonical short name."""
    if not raw:
        return None
    name, _, effort = str(raw).partition("·")
    fam = family_of(name.rstrip("0123456789").rstrip("-")) if name else name
    for s in known:
        if family_of(s.model) == fam or _canonical(name) == s.model:
            return Setting(s.harness, s.model, effort or s.effort)
    model = _canonical(name)
    return Setting(harness_for(model) or "unknown", model, effort or "default")


def infer_sweep(doc: Mapping[str, Any]) -> Inference:
    """A contract v3 sweep run file: PLAN, DEV, REV, GATE with settings declared in `ext.experiment`."""
    exp = doc["ext"]["experiment"]
    arm = _experiment_setting(exp.get("arm"))
    reviewer = _experiment_setting(exp.get("reviewer"))
    planner = _experiment_setting(exp.get("planner"))
    if arm is None:
        raise InferError("sweep run file without ext.experiment.arm")
    reasons = ["declared by the sweep (ext.experiment)"]
    tasks = {t.get("id"): t for t in doc.get("tasks") or [] if isinstance(t, Mapping)}
    conf = 0.98
    if planner is None and "PLAN" in tasks:
        attempts = tasks["PLAN"].get("attempts") or [{}]
        planner = _resolve_short(attempts[-1].get("model"), [s for s in (arm, reviewer) if s])
        reasons.append(f"planner not declared; the shared plan's setting {planner.model}/{planner.effort} "
                       "was read from the PLAN attempt")
        conf -= 0.03
    if exp.get("planner_shared"):
        reasons.append("the plan was generated once per task and shared by every arm")
    rounds = max(DEFAULT_REVIEW_ROUNDS, int(exp.get("dev_attempts") or 1))
    params = normalize(ShapeParams(plan=planner is not None, middle="implement", review=reviewer is not None,
                                   budget_rounds=min(MAX_ROUNDS, rounds)))
    settings = {"implement": arm}
    if planner is not None:
        settings["plan"] = planner
    if reviewer is not None:
        settings["review"] = reviewer
    if "GATE" in tasks:
        reasons.append("the harness gate (GATE) before review is not a piece; it maps to the review piece")
    node_vertex = {}
    for tid, t in tasks.items():
        kind = _role(_KIND_ROLE.get(str(t.get("kind")), t.get("owner")))
        node_vertex[str(tid)] = _nearest_piece(kind, params)
    return _finish(params, settings, node_vertex, conf, reasons, "sweep")


def infer_rq1(config: Mapping[str, Any]) -> Inference:
    """An RQ1 run `config.json`: topology solo, reviewer, team, planner or best; agents [name, role, model, effort]."""
    topology = str(config.get("topology") or "")
    agents = [a for a in config.get("agents") or [] if isinstance(a, (list, tuple)) and len(a) >= 3]
    if not agents:
        raise InferError("RQ1 config without agents")
    reasons = [f"declared by the RQ1 run config (arm {config.get('arm')}, topology {topology})"]
    by_role: dict[str, list[Setting]] = {}
    names: dict[str, str] = {}
    for a in agents:
        name, role, model = str(a[0]), _role(a[1]), _canonical(a[2])
        effort = str(a[3]) if len(a) > 3 and a[3] else "default"
        by_role.setdefault(role, []).append(Setting(harness_for(model) or "unknown", model, effort))
        names[name] = role
    workers = by_role.get("worker", []) + by_role.get("solo", [])
    middle = {"best": "best_of_n", "team": "team"}.get(topology, "team" if len(workers) > 1 else "implement")
    params = normalize(ShapeParams(plan="planner" in by_role, middle=middle, width=len(workers),
                                   review="reviewer" in by_role,
                                   budget_rounds=DEFAULT_REVIEW_ROUNDS if "reviewer" in by_role else 1))
    conf = 0.98
    settings: dict[str, Setting] = {}
    work, share = _mode(workers)
    settings[work_piece(params)] = work
    if share < 1:
        reasons.append(f"workers ran mixed settings; the piece takes the most common, {work.model}/{work.effort}")
        conf -= 0.15 * (1 - share)
    if params.plan:
        settings["plan"] = _mode(by_role["planner"])[0]
    if params.review:
        settings["review"] = _mode(by_role["reviewer"])[0]
    if params.middle == "best_of_n":
        settings["select"] = SCORER
        reasons.append("best of n: the best-scoring submission is kept by a command (setting harness command)")
    node_vertex = {name: _nearest_piece(role, params) for name, role in names.items()}
    return _finish(params, settings, node_vertex, conf, reasons, "rq1")


def _mode(settings: list[Setting], weights: list[float] | None = None) -> tuple[Setting, float]:
    """The most common setting (ties to the larger weight), and the share of sessions that ran it."""
    counts = Counter((s.harness, s.model, s.effort) for s in settings)
    cost: Counter = Counter()
    for s, w in zip(settings, weights or [0.0] * len(settings)):
        cost[(s.harness, s.model, s.effort)] += w
    key = max(counts, key=lambda k: (counts[k], cost[k]))
    return Setting(*key), counts[key] / len(settings)


# ------------------------------------------------------------------ the classifier

def _nearest_piece(role: str | None, params: ShapeParams) -> str:
    """The piece a session with this role belongs to (unmatched nodes go to the nearest piece by role)."""
    work = work_piece(params)
    prefs = {
        "planner": ["plan"], "lead": ["plan"], "external": ["plan"],
        "reviewer": ["review", "select"], "tester": ["review", "select"], "gate": ["review", "select"],
        "referee": ["select", "review"],
    }.get(role or "", [])
    present = {p.id for p in build_shape(params).pieces}
    for piece in prefs:
        if piece in present:
            return piece
    return work


def infer_agents(agents: list[Agent], *, base: float = 0.9, source: str = "graph") -> Inference:
    if not agents:
        raise InferError("no sessions to infer a workflow from")
    reasons: list[str] = []
    core = [a for a in agents if a.phase in BUILD_PHASES and a.role not in ("external", "gate")]
    left_out = len(agents) - len(core)
    if not core:
        core = [a for a in agents if a.role not in ("external", "gate")] or list(agents)
        reasons.append("no session is marked as the build; all of them were used")
    elif left_out:
        reasons.append(f"{left_out} session(s) outside the build (external or after it) do not shape the workflow")

    def role_of(a: Agent) -> str | None:
        if a.role is not None:
            return a.role
        return "lead" if a.source in ("top", None) else "worker"

    roles = {a.id: role_of(a) for a in core}
    planners = [a for a in core if roles[a.id] == "planner"]
    workers = [a for a in core if roles[a.id] in WORKER_ROLES]
    mains = [a for a in core if roles[a.id] in MAIN_ROLES or roles[a.id] not in
             WORKER_ROLES + REVIEW_ROLES + ("planner", "referee")]
    reviewers = [a for a in core if roles[a.id] in REVIEW_ROLES]
    referees = [a for a in core if roles[a.id] == "referee"]
    conf = base

    if workers:
        overlap = _overlap(workers)
        if overlap is None:
            overlap = len(workers) if len(workers) > 1 else 1
            if len(workers) > 1:
                reasons.append("worker timing is missing; parallel work was assumed")
                conf -= 0.1
        if referees:
            middle = "best_of_n"
        elif overlap >= 2:
            middle = "team"
        else:
            middle = "implement"
        width = min(MAX_WIDTH, overlap) if middle != "implement" else 1
        if middle != "implement" and overlap > MAX_WIDTH:
            reasons.append(f"up to {overlap} workers ran at once; width is capped at {MAX_WIDTH}")
            conf -= 0.05
        plan = bool(planners or mains)
        if mains and not planners:
            reasons.append("the lead session that handed out the work is read as the planner")
    else:
        middle, width = ("best_of_n", 2) if referees else ("implement", 1)
        plan = bool(planners)
        if not mains:
            mains = planners[:1]
            plan = False
            reasons.append("only planning sessions; the planner is read as the implementer")
    review = bool(reviewers)
    rounds = 1
    if review:
        rounds = DEFAULT_REVIEW_ROUNDS
        if middle == "implement" and len(reviewers) > rounds:
            rounds = min(MAX_ROUNDS, len(reviewers))
    params = normalize(ShapeParams(plan=plan, middle=middle, width=width, review=review, budget_rounds=rounds))
    work = work_piece(params)

    core_ids, main_ids = {a.id for a in core}, {a.id for a in mains}
    node_vertex: dict[str, str] = {}
    for a in agents:
        role = roles.get(a.id) or role_of(a)
        if a.id in core_ids and role in WORKER_ROLES:
            node_vertex[a.id] = work
        elif a.id in main_ids:
            node_vertex[a.id] = "plan" if params.plan and workers else work
        else:
            node_vertex[a.id] = _nearest_piece(role, params)

    settings: dict[str, Setting] = {}
    shares = []
    fallback = None
    for piece in build_shape(params).pieces:
        members = [a for a in core if node_vertex[a.id] == piece.id and a.model]
        if piece.id == "plan" and any(roles[a.id] == "planner" for a in members):
            members = [a for a in members if roles[a.id] == "planner"]  # the lead only hands out the plan
        if not members:
            continue
        chosen, share = _mode([Setting(a.harness or harness_for(a.model) or "unknown", a.model, a.effort or "default")
                               for a in members], [a.cost for a in members])
        settings[piece.id] = chosen
        shares.append(share)
        if share < 1:
            reasons.append(f"{piece.role} sessions ran mixed settings; the piece takes the most common, "
                           f"{chosen.model}/{chosen.effort}")
        if piece.id == work:
            fallback = chosen
    if not settings:
        raise InferError("no session records a model")
    fallback = fallback or next(iter(settings.values()))
    for piece in build_shape(params).pieces:
        if piece.id not in settings:
            settings[piece.id] = fallback
            reasons.append(f"the {piece.role} piece has no session with a model; it takes {fallback.model}/{fallback.effort}")
            conf -= 0.1
    if shares:
        conf -= 0.15 * (1 - min(shares))

    labelled = [a for a in core if a.role is not None]
    unlabelled = len(core) - len(labelled)
    if unlabelled:
        reasons.append(f"{unlabelled} session(s) have no role label")
        conf -= 0.2 * unlabelled / len(core)
    heuristic = sum(1 for a in labelled if a.tier == "heuristic")
    if heuristic:
        conf -= 0.1 * heuristic / len(core)
        reasons.append(f"{heuristic} of {len(core)} role labels are heuristic")
    if any(a.short_model for a in core):
        reasons.append("some model names are short forms")
        conf -= 0.1
    return _finish(params, settings, node_vertex, conf, reasons, source)


def _finish(params: ShapeParams, settings: Mapping[str, Setting], node_vertex: dict[str, str], conf: float,
            reasons: list[str], source: str) -> Inference:
    wf = build_shape(params)
    conf = round(min(0.99, max(0.05, conf)), 3)
    extra = {"node_vertex": dict(node_vertex), "shape": shape_name(params), "nearest_catalog": nearest_catalog(params),
             "confidence": conf, "reasons": list(reasons), "inferred_from": source}
    config = make_config(wf, {p.id: settings[p.id] for p in wf.pieces}, extra=extra)
    return Inference(config, conf, params, dict(node_vertex), list(reasons), source)


# ------------------------------------------------------------------ entry points

def _is_graph_dict(obj: Mapping[str, Any]) -> bool:
    nodes = obj.get("nodes")
    return isinstance(nodes, list) and "ocp" not in obj and all(
        isinstance(n, Mapping) and ("harness" in n or "source" in n) for n in nodes[:5])


# A session's role to the piece roles it may belong to, most likely first (declared workflows).
_ROLE_TARGETS = {
    "planner": ("planner",), "lead": ("planner",), "external": ("planner",),
    "implementer": ("implementer", "worker"), "worker": ("worker", "implementer"),
    "reviewer": ("reviewer", "tester", "referee"), "tester": ("tester", "reviewer", "referee"),
    "gate": ("reviewer", "tester", "referee"), "referee": ("referee", "reviewer"),
}


def _work_piece(workflow: Workflow) -> str:
    """The declared workflow's main working piece: the first implementer or worker, else the widest piece."""
    for p in workflow.pieces:
        if normalize_role(p.role) in ("implementer", "worker"):
            return p.id
    return max(workflow.pieces, key=lambda p: p.width).id


def declared_node_vertex(doc: Mapping[str, Any], workflow: Workflow) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """(node to piece, node to basis, reasons) for a run that declares `workflow`; every target is one of its pieces.

    Basis, strongest first: `node` (a valid `node.vertex`), `attempt` (the
    node's attempts name a valid `vertex`), `role` (one piece of the declared
    workflow has the session's role), `ambiguous` (several pieces or attempt
    vertices fit; the first is taken), `default` (nothing fits; the main
    working piece).
    """
    pieces = [p.id for p in workflow.pieces]
    piece_roles = [(p.id, normalize_role(p.role)) for p in workflow.pieces]
    attempt_vertices: dict[str, list[str]] = {}
    for a in doc.get("attempts") or []:
        if isinstance(a, Mapping) and isinstance(a.get("node"), str) and isinstance(a.get("vertex"), str):
            attempt_vertices.setdefault(a["node"], []).append(a["vertex"])
    agents = {a.id: a for a in agents_from_ocp(doc)}
    node_vertex: dict[str, str] = {}
    basis: dict[str, str] = {}
    reasons: list[str] = []
    for node in doc.get("nodes") or []:
        if not isinstance(node, Mapping) or not isinstance(node.get("id"), str):
            continue
        nid, vertex = node["id"], node.get("vertex")
        if vertex in pieces:
            node_vertex[nid], basis[nid] = vertex, "node"
            continue
        if vertex is not None:
            reasons.append(f"node {nid} names vertex {vertex!r}, which is not a piece of {workflow.id}")
        named = [v for v in attempt_vertices.get(nid, []) if v in pieces]
        if named:
            counts = Counter(named)
            best = max(counts, key=lambda v: (counts[v], -named[::-1].index(v)))  # ties go to the latest attempt
            node_vertex[nid], basis[nid] = best, "attempt" if len(counts) == 1 else "ambiguous"
            if len(counts) > 1:
                reasons.append(f"node {nid}'s attempts name several pieces ({', '.join(sorted(counts))}); took {best}")
            continue
        role = getattr(agents.get(nid), "role", None)
        target = None
        for want in _ROLE_TARGETS.get(role or "", ()):
            fits = [pid for pid, r in piece_roles if r == want]
            if fits:
                target = fits[0]
                basis[nid] = "role" if len(fits) == 1 else "ambiguous"
                if len(fits) > 1:
                    reasons.append(f"node {nid} ({role}) fits pieces {', '.join(fits)}; took {target}")
                break
        if target is None:
            target, basis[nid] = _work_piece(workflow), "default"
            reasons.append(f"node {nid} ({role or 'no role'}) matches no piece by vertex or role; "
                           f"put on {target}")
        node_vertex[nid] = target
    return node_vertex, basis, reasons


def _from_declared(doc: Mapping[str, Any]) -> Inference | None:
    """An OCP document that states its configuration inline: taken as is, nodes mapped onto its own pieces."""
    run = doc.get("run") if isinstance(doc.get("run"), Mapping) else {}
    cfg = run.get("configuration")
    if not isinstance(cfg, Mapping) or not isinstance(cfg.get("workflow"), Mapping) or not cfg.get("settings"):
        return None
    from .ocp import configuration_from_ocp
    from .shapes import guess_params, shape_params

    try:
        config = configuration_from_ocp(cfg)
    except (LookupError, ValueError, KeyError, TypeError):
        return None
    if not config.workflow.pieces:
        return None
    params = shape_params(config.workflow) or guess_params(config.workflow) or ShapeParams()
    node_vertex, basis, notes = declared_node_vertex(doc, config.workflow)
    reasons = ["the run declares its configuration", *notes]
    weak = sum(1 for b in basis.values() if b in ("ambiguous", "default"))
    conf = round(0.95 - (0.1 * weak / len(basis) if basis else 0.0), 3)
    extra = {**config.extra, "node_vertex": node_vertex, "node_vertex_basis": basis, "shape": config.workflow.id,
             "nearest_catalog": nearest_catalog(params), "confidence": conf,
             "reasons": reasons, "inferred_from": "configuration"}
    config = Configuration(config.id, config.workflow, config.settings, extra)
    return Inference(config, conf, params, node_vertex, reasons, "configuration")


# The graph writer's extension (`graph --format ocp`): its v0.3 key, and the key it wrote before 0.3.
GRAPH_EXT_KEYS = ("dev.loopmath.graph", "dev.dagr.graph")


def is_graph_document(doc: Mapping[str, Any]) -> bool:
    """True when an OCP document carries the graph extension on itself, a node or an attempt."""
    def carries(record: Any) -> bool:
        ext = record.get("ext") if isinstance(record, Mapping) else None
        return isinstance(ext, Mapping) and any(isinstance(ext.get(k), Mapping) for k in GRAPH_EXT_KEYS)

    return carries(doc) or any(carries(r) for key in ("nodes", "attempts") for r in doc.get(key) or [])


def infer_detail(source: Any) -> Inference:
    """Everything `infer` knows: configuration, confidence, parameters, node mapping and reasons."""
    if isinstance(source, (str, Path)):
        return infer_detail(load_source(source))
    if not isinstance(source, Mapping):
        return infer_agents(agents_from_graph(source))
    if "topology" in source and "agents" in source:
        return infer_rq1(source)
    if "dagr" in source and "tasks" in source:
        exp = (source.get("ext") or {}).get("experiment")
        if isinstance(exp, Mapping) and exp.get("arm"):
            return infer_sweep(source)
        from ..ocp.contractv3 import convert

        return infer_detail(convert(source))
    if "ocp" in source:
        declared = _from_declared(source)
        if declared is not None:
            return declared
        if str(source.get("ocp")) == "0.2" or is_graph_document(source):
            try:
                from ..ingest.ocp import from_ocp

                return infer_agents(agents_from_graph(from_ocp(source)), base=0.9, source="ocp")
            except Exception:  # noqa: BLE001 - fall back to reading nodes and attempts directly
                pass
        return infer_agents(agents_from_ocp(source), base=0.85, source="ocp")
    if _is_graph_dict(source):
        return infer_agents(agents_from_graph(source))
    raise InferError("not a graph, OCP document, sweep run file or RQ1 config")


def infer(graph: Any) -> tuple[Configuration, float]:
    """(configuration, confidence) for a graph or run file; `extra["node_vertex"]` maps every node."""
    result = infer_detail(graph)
    return result.configuration, result.confidence


infer_workflow = infer  # the name spec 01 uses for `ocp migrate`


def load_source(path: str | Path) -> Any:
    """A file or RQ1 run folder as `infer_detail` input (gzip+base64 graph envelopes are unpacked)."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, Mapping) and obj.get("encoding") == "gzip+base64" and "data" in obj:
        import base64
        import gzip

        obj = json.loads(gzip.decompress(base64.b64decode(obj["data"])))
    return obj
