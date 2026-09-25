"""Sparse design rows for every head, from run documents (fitting) and configurations (prediction).

Spec 04 section 2. The same term builders make the rows for fitting (`rows_for_run`, from an
OCP v0.3 run document) and for prediction (`rows_for_config`, from a task and a
configuration), so a fitted node and a predicted node always mean the same thing.

A term is `(node_id, parent_id, value)`. Rows:
- cost and tokens heads: one row per attempt (piece, round);
- gate head: one row per gate evaluation (gate, round);
- success head and score heads: one row per run.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

from ..taskmodel import BUILTIN_FEATURES, HORIZON_KEY, FeatureSet, parse_horizon
from ..types import (
    DEFAULT_RULE, AcceptanceRule, Configuration, Control, Evidence, Gate, Piece, ScoreTarget, Setting, Task,
    Workflow,
)
from .forest import canonical_model_id, canonical_role, model_path

Term = tuple[str, "str | None", float]

# Node names of this design (spec 04 section 4). 2 (loopmath 0.2): shape keys, psrc nodes and role groups.
# 3 (0.2): `horizon:log2h` is relative to the fit's reference horizon. A fit of another version is not read.
DESIGN_VERSION = 3
HEURISTIC_COST_WEIGHT = 0.7  # spec 04 section 2 and D27: allocated (heuristic) cost rows
USER_SOURCES = ("live", "backlog", "history", "user", "orchestrator", "onboard", "habit", "")
HORIZON_NODES = ("horizon:timebox", "horizon:log2h")  # fixed terms of a timeboxed run (spec 04 section 1)
RQ1_EXT = "dev.loopmath.rq1"  # loopmath-exp's ext key; its horizon_s is the fallback for older RQ1 runs


def other_design(meta: dict) -> str | None:
    """Why a fit cannot be read: it was written by another design version (fits before 0.2 have none,
    version 1). None when it can. Light on purpose: `status` and `doctor` call it without the engine."""
    version = meta.get("design_version", 1)
    if version == DESIGN_VERSION:
        return None
    by = "an older" if isinstance(version, int) and version < DESIGN_VERSION else "another"
    return (f"fit {meta.get('fit') or '?'} is from design version {version} (made by {by} loopmath) and this "
            f"loopmath reads version {DESIGN_VERSION}; run `loopmath fit` for a new fit")


# ---------------------------------------------------------------- terms

def round_terms(k: int) -> list[Term]:
    if k <= 1:
        return []
    return [("round:2", None, 1.0)] if k == 2 else [("round:3+", None, 1.0)]


@lru_cache(maxsize=65536)
def _task_terms_cached(org: str | None, ttype: str, repo: str, subtype: str | None, task_id: str,
                       features: tuple[tuple[str, str], ...], source: str) -> tuple[Term, ...]:
    terms: list[Term] = [("fixed:intercept", None, 1.0)]
    if org:
        terms.append((f"org:{org}", None, 1.0))
    type_id = f"type:{ttype}"
    terms.append((type_id, None, 1.0))
    repo_key = f"{ttype}/{repo}"
    terms.append((f"repo:{repo_key}", type_id, 1.0))
    parent = f"repo:{repo_key}"
    if subtype:
        sub_key = f"{repo_key}/{subtype}"
        terms.append((f"subtype:{sub_key}", parent, 1.0))
        parent = f"subtype:{sub_key}"
    terms.append((f"task:{task_id}", parent, 1.0))
    for key, value in features:
        terms.append((f"feature:{key}={value}", None, 1.0))
    terms.append((f"source:{source}", None, 1.0))
    return tuple(terms)


def horizon_terms(horizon: float | None, coding: dict | None) -> list[Term]:
    """A run's horizon terms under a fit's coding (`meta.json` `horizons`, spec 04 section 1).

    `horizon:log2h` is log2(horizon / reference_s), so a run at the fit's reference horizon adds 0;
    `horizon:timebox` only when the fit could tell timeboxed from open-ended runs (`timebox`).
    None for an open-ended run, or under a fit without a reference (no timeboxed runs)."""
    ref = (coding or {}).get("reference_s")
    if not horizon or not ref:
        return []
    terms = [(HORIZON_NODES[1], None, math.log2(horizon / float(ref)))]
    return [(HORIZON_NODES[0], None, 1.0)] + terms if coding.get("timebox") else terms


def task_features(task: Task, features: FeatureSet | None = None) -> tuple[tuple[str, str], ...]:
    """Declared feature keys with a known value, sorted. Unknown values carry no information;
    `horizon_s` is not a feature level (see `task_horizon`)."""
    raw = {k: _feature_str(v) for k, v in (task.features or {}).items()}
    return (features or BUILTIN_FEATURES).model_features(raw)


def task_horizon(task: Task) -> float | None:
    """`task.features.horizon_s` in seconds; None when open-ended or unreadable."""
    try:
        return parse_horizon((task.features or {}).get(HORIZON_KEY))
    except ValueError:
        return None


def _feature_str(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def task_terms(task: Task, source: str, features: FeatureSet | None = None,
               horizons: dict | None = None) -> list[Term]:
    """The task's terms; the horizon terms only under a fit's coding (`horizons`, see `horizon_terms`)."""
    terms = list(_task_terms_cached(task.org or None, task.type or "unknown", task.repo or "unknown",
                                    task.subtype or None, task.id, task_features(task, features), source))
    return terms + horizon_terms(task_horizon(task), horizons) if horizons else terms


@lru_cache(maxsize=65536)
def setting_terms(harness: str, model: str, effort: str, weight: float = 1.0,
                  with_harness: bool = True) -> tuple[Term, ...]:
    prov, fam, ver = model_path(model)
    eff = (effort or "default").lower()
    terms = [(f"provider:{prov}", None, weight), (f"family:{fam}", f"provider:{prov}", weight),
             (f"model:{ver}", f"family:{fam}", weight), (f"effort:{eff}", None, weight),
             (f"family_effort:{fam}|{eff}", f"effort:{eff}", weight)]
    if with_harness:
        terms.append((f"harness:{harness or 'unknown'}", None, weight))
    return tuple(terms)


@lru_cache(maxsize=65536)
def role_terms(role: str, model: str, weight: float = 1.0) -> tuple[Term, ...]:
    fam = model_path(model)[1]
    return ((f"role:{role}", None, weight), (f"role_family:{role}|{fam}", f"role:{role}", weight))


def _s(setting: Setting) -> tuple[str, str, str]:
    return (setting.harness or "unknown", canonical_model_id(setting.model), (setting.effort or "default").lower())


# ---------------------------------------------------------------- workflow structure

@dataclass
class GateInfo:
    id: str
    after: str
    rule: str
    on_fail: str | None
    judged: str  # piece whose output the gate judges: the repair target, else the piece it follows


@dataclass
class Structure:
    """What the composition needs from a configuration (spec 04 section 3)."""

    workflow_id: str
    pieces: list[str]  # topological order
    roles: dict[str, str]
    settings: dict[str, Setting]
    widths: dict[str, int]
    gates: list[GateInfo]
    loops: list[tuple[tuple[int, ...], list[str]]]  # (gate indices in piece order, pieces in the repair loop)
    k_max: int
    piece_loop: dict[str, int | None] = field(default_factory=dict)  # piece -> index into loops
    gate_loop: dict[int, int | None] = field(default_factory=dict)  # gate index -> index into loops
    shape: str = ""  # topology and position key (`shape_key`); empty means the workflow id
    copies: dict[str, int] = field(default_factory=dict)  # piece -> copies in its group (`copy_groups`)


def topological_pieces(workflow: Workflow) -> list[str]:
    declared = [p.id for p in workflow.pieces]
    nodes = list(declared) + [a for a in workflow.artifacts if a not in declared]
    order_key = {n: i for i, n in enumerate(nodes)}
    succ: dict[str, list[str]] = {n: [] for n in nodes}
    indeg = {n: 0 for n in nodes}
    for a, b in workflow.edges:
        if a in succ and b in indeg:
            succ[a].append(b)
            indeg[b] += 1
    ready = sorted([n for n in nodes if indeg[n] == 0], key=order_key.get)
    out: list[str] = []
    while ready:
        n = ready.pop(0)
        out.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                ready.append(m)
                ready.sort(key=order_key.get)
    placed = set(out)
    out += [n for n in nodes if n not in placed]  # cyclic leftovers keep their declared order
    return [n for n in out if n in set(declared)]


def _piece_reach(workflow: Workflow) -> dict[str, set[str]]:
    """Pieces reachable from each piece through artifacts (piece -> artifact -> piece)."""
    succ: dict[str, set[str]] = {}
    for a, b in workflow.edges:
        succ.setdefault(a, set()).add(b)
    pieces = {p.id for p in workflow.pieces}
    reach: dict[str, set[str]] = {}
    for p in pieces:
        seen: set[str] = set()
        stack = [p]
        while stack:
            n = stack.pop()
            for m in succ.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        reach[p] = {m for m in seen if m in pieces}
    return reach


@lru_cache(maxsize=256)
def _catalog_roles(workflow_id: str) -> frozenset[str] | None:
    wf = _catalog_workflow(workflow_id)
    return frozenset(canonical_role(p.role) for p in wf.pieces) if wf is not None else None


def shape_key(workflow_id: str, roles: Iterable[str]) -> str:
    """The key of a workflow's topology and position nodes (spec 04 section 1): its id, or
    `<id>~<sorted roles joined by +>` when its set of roles differs from the catalog workflow of that
    id (RQ1's `plan_implement` with workers is `plan_implement~planner+worker`). A set, so parallel
    copies of a role (one width-3 piece or three width-1 pieces) are one shape."""
    have = frozenset(canonical_role(r) for r in roles)
    catalog_roles = _catalog_roles(workflow_id)
    if catalog_roles is None or catalog_roles == have:
        return workflow_id
    return f"{workflow_id}~{'+'.join(sorted(have))}"


def copy_groups(order: list[str], roles: dict[str, str], reach: dict[str, set[str]]) -> dict[str, int]:
    """piece -> the number of copies in its group. A group is the pieces of one role with no path
    between them either way, such as `implement-1`, `implement-2`, `implement-3`; a lone piece is 1."""
    groups: list[list[str]] = []
    for p in order:
        for g in groups:
            if roles[g[0]] == roles[p] and all(p not in reach.get(q, ()) and q not in reach.get(p, ()) for q in g):
                g.append(p)
                break
        else:
            groups.append([p])
    return {p: len(g) for g in groups for p in g}


def default_gate_rule(role: str) -> str:
    return {"reviewer": "review_approve", "tester": "tests_pass", "referee": "referee_pick"}.get(role, "gate")


def structure(config: Configuration) -> Structure:
    wf = config.workflow
    order = topological_pieces(wf)
    roles = {p.id: canonical_role(p.role) for p in wf.pieces}
    widths = {p.id: max(1, int(p.width or 1)) for p in wf.pieces}
    settings = {}
    for p in wf.pieces:
        s = config.settings.get(p.id) or p.setting
        settings[p.id] = s if s is not None else Setting("unknown", "unknown")
    k_max = max(1, int(wf.control.budget_rounds or 1))  # Counts the first round
    gates: list[GateInfo] = []
    for i, g in enumerate(wf.control.gates):
        judged = g.on_fail if g.on_fail in roles else g.after
        gates.append(GateInfo(g.id or f"g_{g.after}", g.after, g.rule or default_gate_rule(roles.get(g.after, "")),
                              g.on_fail if g.on_fail in roles else None, judged))
    reach = _piece_reach(wf)
    loops: list[tuple[tuple[int, ...], list[str]]] = []
    piece_loop: dict[str, int | None] = {p: None for p in order}
    gate_loop: dict[int, int | None] = {gi: None for gi in range(len(gates))}
    # Loops exist at K_max = 1 too, so gates still decide which pieces a round reaches
    merged: list[tuple[set[int], set[str]]] = []
    for gi, g in enumerate(gates):
        if not g.on_fail or g.after not in roles:
            continue
        start, end = g.on_fail, g.after
        members = {p for p in order if p in (start, end) or (p in reach.get(start, ()) and end in reach.get(p, ()))}
        if start != end and end not in reach.get(start, ()):
            i0, i1 = order.index(start), order.index(end)
            members = set(order[min(i0, i1):max(i0, i1) + 1])
        gs = {gi}
        # repair loops that share a piece share their rounds: merge them (the sweep's tests gate on
        # implement and review gate back to implement are one loop with two gates in order)
        for other in [m for m in merged if m[1] & members]:
            merged.remove(other)
            gs |= other[0]
            members |= other[1]
        merged.append((gs, members))
    for gs, ms in sorted(merged, key=lambda m: min(order.index(p) for p in m[1])):
        members = [p for p in order if p in ms]
        gidx = tuple(sorted(gs, key=lambda i: (order.index(gates[i].after), i)))
        loops.append((gidx, members))
        for p in members:
            piece_loop[p] = len(loops) - 1
        for i in gidx:
            gate_loop[i] = len(loops) - 1
    return Structure(wf.id, order, roles, settings, widths, gates, loops, k_max, piece_loop, gate_loop,
                     shape_key(wf.id, roles.values()), copy_groups(order, roles, reach))


# ---------------------------------------------------------------- rows

def cost_rest(st: Structure, piece: str, k: int, setting: Setting | None = None, *,
              source: str | None = None) -> tuple[Term, ...]:
    """Cost and tokens rows without the task part: setting, role, topology, position, round, control,
    and with a `source` the position x source node `psrc:<shape>#<pos>|<source>` and the family x source
    node `fsrc:<family>|<source>` (spec 04 section 1)."""
    s = setting or st.settings[piece]
    h, m, e = _s(s)
    shape = st.shape or st.workflow_id
    position = f"{shape}#{st.pieces.index(piece)}"
    terms = (setting_terms(h, m, e) + role_terms(st.roles[piece], m)
             + ((f"topology:{shape}", None, 1.0), (f"position:{position}", f"topology:{shape}", 1.0)))
    if source is not None:
        fam = model_path(m)[1]
        terms += ((f"psrc:{position}|{source}", f"position:{position}", 1.0),
                  (f"fsrc:{fam}|{source}", f"family:{fam}", 1.0))
    return terms + tuple(round_terms(k)) + tuple(_control_terms(st.k_max, st.widths[piece]))


def gate_rest(st: Structure, gate: GateInfo, k: int) -> tuple[Term, ...]:
    """Gate rows without the task part: gate rule, round, and the judged piece's setting and role."""
    h, m, e = _s(st.settings[gate.judged])
    return (((f"gate:{gate.rule}", None, 1.0),) + tuple(round_terms(k)) + setting_terms(h, m, e, 1.0, False)
            + role_terms(st.roles[gate.judged], m))


def run_rest(st: Structure) -> tuple[Term, ...]:
    """Success and score rows without the task part: topology, control, and per piece its model chain,
    effort and harness (weight 1/n) plus its role and role x family (weight 1/g, g the copies in its
    group, so parallel copies carry the role effect once, as one wide piece does)."""
    terms: list[Term] = [(f"topology:{st.shape or st.workflow_id}", None, 1.0)]
    terms += _control_terms(st.k_max, max(st.widths.values() or [1]))
    n = max(1, len(st.pieces))
    for piece in st.pieces:
        h, m, e = _s(st.settings[piece])
        terms += setting_terms(h, m, e, 1.0 / n)
        terms += role_terms(st.roles[piece], m, 1.0 / (st.copies.get(piece) or 1))
    return tuple(terms)


def cost_row(task: Task, source: str, st: Structure, piece: str, k: int,
             setting: Setting | None = None, features: FeatureSet | None = None) -> list[Term]:
    return task_terms(task, source, features) + list(cost_rest(st, piece, k, setting, source=source))


def gate_row(task: Task, source: str, st: Structure, gate: GateInfo, k: int,
             features: FeatureSet | None = None) -> list[Term]:
    return task_terms(task, source, features) + list(gate_rest(st, gate, k))


def run_row(task: Task, source: str, st: Structure, features: FeatureSet | None = None) -> list[Term]:
    return task_terms(task, source, features) + list(run_rest(st))


def _control_terms(k_max: int, width: int) -> list[Term]:
    """Budget and width on a log2 scale, so a few runs with a huge budget (RQ1's 200 rounds) do
    not set the coefficient that the usual 2 to 5 rounds use."""
    out = []
    if k_max > 1:
        out.append(("control:budget", None, math.log2(k_max)))
    if width > 1:
        out.append(("control:width", None, math.log2(width)))
    return out


def rows_for_config(task: Task, config: Configuration, source: str = "user",
                    features: FeatureSet | None = None) -> dict:
    """Prediction rows: per (piece, round), per (gate, round), one run row, and the structure."""
    st = structure(config)
    cost = {}
    for piece in st.pieces:
        rounds = st.k_max if st.piece_loop.get(piece) is not None else 1
        for k in range(1, rounds + 1):
            cost[(piece, k)] = cost_row(task, source, st, piece, k, features=features)
    gate = {}
    for gi, g in enumerate(st.gates):
        rounds = st.k_max if st.gate_loop.get(gi) is not None else 1
        for k in range(1, rounds + 1):
            gate[(gi, k)] = gate_row(task, source, st, g, k, features)
    return {"structure": st, "cost": cost, "gate": gate, "run": run_row(task, source, st, features)}


# ---------------------------------------------------------------- reading run documents

@dataclass
class AttemptObs:
    piece: str
    round: int
    usd: float | None
    tokens: float | None
    weight: float
    setting: Setting
    repriced: bool = True  # False: the recorded dollars, not the current tariff


@dataclass
class ParsedRun:
    run_id: str
    task: Task
    config: Configuration
    source: str
    attempts: list[AttemptObs]
    gates: list[tuple[int, int, bool]]  # (gate index, round, passed)
    evidence: Evidence | None
    score_meta: dict[str, dict]  # score name -> {unit, better, scale}
    rule: AcceptanceRule
    dropped: dict[str, int] = field(default_factory=dict)
    checks: dict[str, int] = field(default_factory=dict)  # check name -> attempts that ran no model


class Unusable(ValueError):
    """A run document the fit cannot use; the message is the reason recorded in meta.json."""


def data_source(doc: dict, origin: str | None = None) -> str:
    """spec 04 section 1 `source` level: user, sweep, e0, rq1, repo_history, benchmark, shared:<org>.

    The source is where the fit found the run: `origin` is `user` for a run in the
    store, whatever its label says, the manifest source for a shipped run, `shared:<org>` for
    an import. A document given without an origin falls back to its own label.
    """
    return origin or source_label(doc)


def source_label(doc: dict) -> str:
    """The source the document names itself: a user kind is `user`, a share export `shared:<org>`."""
    run = doc.get("run") or {}
    ext = run.get("ext") or {}
    share = ext.get("dev.loopmath.share")
    if isinstance(share, dict):
        org = share.get("org") or share.get("org_hash") or (run.get("task") or {}).get("org") or "shared"
        return f"shared:{org}"
    src = (run.get("task") or {}).get("source")
    kind = src.get("kind") if isinstance(src, dict) else src
    kind = str(kind or "").strip().lower()
    if kind in USER_SOURCES:
        return "user"
    return kind


def task_from_doc(doc: dict, origin: str | None = None) -> Task:
    """The task of a run document; `origin` is where the fit found it (see `data_source`).

    When the origin overrides the document's own label, the label stays readable in
    `task.extra["source_label"]`; it is not a model node (spec 04 section 1).
    """
    run = doc.get("run") or {}
    t = run.get("task") or {}
    labels = run.get("labels") or {}
    features = {str(k): _feature_str(v) for k, v in (t.get("features") or {}).items()}
    if HORIZON_KEY not in features:  # older RQ1 runs keep the horizon in loopmath-exp's own ext key
        rq1 = (run.get("ext") or {}).get(RQ1_EXT)
        if isinstance(rq1, dict) and rq1.get(HORIZON_KEY) is not None:
            features[HORIZON_KEY] = _feature_str(rq1[HORIZON_KEY])
    labeled = t.get("labeled_by")
    source = data_source(doc, origin)
    label = source_label(doc) if origin else source
    return Task(
        id=str(t.get("id") or labels.get("task") or run.get("id") or "unknown"),
        type=str(t.get("type") or labels.get("type") or "unknown"),
        repo=str(t.get("repo") or labels.get("repo") or run.get("workspace") or "unknown"),
        title="",
        subtype=t.get("subtype") or None,
        org=t.get("org") or None,
        features=features,
        base_commit=t.get("base_commit"),
        source=source,
        labeled_by=(labeled.get("how") if isinstance(labeled, dict) else labeled),
        extra={"source_label": label} if label != source else {},
    )


def _model_str(model: Any) -> str:
    if isinstance(model, dict):
        return str(model.get("id") or model.get("raw") or "unknown")
    return str(model or "unknown")


def _setting_from(d: Any) -> Setting:
    if isinstance(d, Setting):
        return d
    d = d or {}
    return Setting(harness=str(d.get("harness") or "unknown"), model=canonical_model_id(_model_str(d.get("model"))),
                   effort=str(d.get("effort") or "default"), context_policy=str(d.get("context_policy") or "fresh"),
                   options={str(k): str(v) for k, v in (d.get("options") or {}).items()})


def _catalog_workflow(ref: str) -> Workflow | None:
    from ..workflows.format import catalog

    return catalog().get(ref)


def workflow_from_doc(wf: dict) -> Workflow:
    """An OCP 2.3 workflow, a `types.Workflow.to_dict()`, or a catalog `{ref, version}`."""
    if "ref" in wf and "pieces" not in wf:
        found = _catalog_workflow(str(wf["ref"]))
        if found is None:
            raise Unusable("workflow ref not resolved")
        return found
    pieces = []
    for p in wf.get("pieces") or []:
        role = p.get("role") or ("worker" if p.get("workflow") else "worker")
        setting = _setting_from(p["setting"]) if p.get("setting") else None
        pieces.append(Piece(id=str(p["id"]), role=canonical_role(role), setting=setting, width=int(p.get("width") or 1)))
    if not pieces:
        raise Unusable("workflow has no pieces")
    artifacts = tuple(a["id"] if isinstance(a, dict) else str(a) for a in wf.get("artifacts") or [])
    edges = tuple((str(e[0]), str(e[1])) if isinstance(e, (list, tuple)) else (str(e["from"]), str(e["to"]))
                  for e in wf.get("edges") or [])
    ctl = wf.get("control") or {}
    raw_gates = ctl.get("gates") or []
    roles = {p.id: p.role for p in pieces}
    gates: list[Gate] = []
    if raw_gates and isinstance(raw_gates[0], dict):  # types form
        for g in raw_gates:
            gates.append(Gate(id=str(g.get("id") or f"g_{g.get('after')}"), after=str(g.get("after")),
                              rule=str(g.get("rule") or default_gate_rule(roles.get(str(g.get("after")), ""))),
                              on_fail=g.get("on_fail")))
        budget = ctl.get("budget_rounds", ctl.get("budget"))
    else:  # OCP form
        repair = ctl.get("repair") or {}
        rules = ((ctl.get("ext") or {}).get("dev.loopmath.gate_rules") or {})
        for after in raw_gates:
            after = str(after)
            gates.append(Gate(id=f"g_{after}", after=after,
                              rule=str(rules.get(after) or default_gate_rule(roles.get(after, ""))),
                              on_fail=repair.get(after)))
        budget = ctl.get("budget", ctl.get("budget_rounds"))
    rescue = ctl.get("rescue")
    rescue_kind = rescue.get("kind") if isinstance(rescue, dict) else (rescue or "redo_usual")
    control = Control(gates=tuple(gates), budget_rounds=max(1, int(budget or 1)), rescue=str(rescue_kind))
    return Workflow(id=str(wf.get("id") or wf.get("ref") or "custom"), version=int(wf.get("version") or 1),
                    title=str(wf.get("title") or ""), pieces=tuple(pieces), artifacts=artifacts, edges=edges,
                    control=control)


def config_from_doc(cfg: dict) -> Configuration:
    wf = workflow_from_doc(cfg.get("workflow") or {})
    raw = cfg.get("settings") or {}
    settings = {}
    for p in wf.pieces:
        if p.id in raw:
            settings[p.id] = _setting_from(raw[p.id])
        elif p.setting is not None:
            settings[p.id] = p.setting
    missing = [p.id for p in wf.pieces if p.id not in settings]
    if missing:
        raise Unusable("piece without a setting")
    cid = cfg.get("id")
    if not cid:
        from ..workflows.ids import config_id

        cid = config_id(wf, settings)
    return Configuration(id=str(cid), workflow=wf, settings=settings)


def rule_from_doc(doc: dict, default: AcceptanceRule = DEFAULT_RULE) -> AcceptanceRule:
    r = (doc.get("run") or {}).get("acceptance_rule")
    if not isinstance(r, dict) or not r.get("name"):
        return default
    score = r.get("score")
    st = None
    if isinstance(score, dict) and score.get("name") and score.get("target") is not None:
        st = ScoreTarget(name=str(score["name"]), target=float(score["target"]),
                         better=str(score.get("better") or "higher"), scale=str(score.get("scale") or "linear"))
    return AcceptanceRule(name=str(r["name"]), definition=str(r.get("definition") or r["name"]),
                          requires=tuple(r.get("requires") if r.get("requires") is not None else ("tests",)),
                          score=st, excludes_events=tuple(r.get("excludes_events") or ("revert", "incident")),
                          window_days=int(r.get("window_days") or 14))


_PRICES = None


def _price_table():
    global _PRICES
    if _PRICES is None:
        try:
            from ..price import load_prices

            _PRICES = load_prices()
        except Exception:  # noqa: BLE001 - a missing or broken table falls back to recorded dollars
            _PRICES = False
    return _PRICES or None


MODEL_TOKENS = "dev.loopmath.model_tokens"  # {model_id: {OCP cost token fields}}
_OCP_TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")


def _ocp_fields(cost: dict) -> dict[str, float]:
    """The four OCP token fields of a cost record; cache writes fall back to their 5m and 1h halves."""
    fields = {name: float(cost.get(name) or 0) for name in _OCP_TOKEN_FIELDS}
    if not cost.get("cache_creation_tokens"):
        fields["cache_creation_tokens"] = float(cost.get("cache_creation_5m_tokens") or 0) + \
            float(cost.get("cache_creation_1h_tokens") or 0)
    return fields


def _price_split(split: dict, table) -> float | None:
    """Dollars for `{model: OCP token fields}`, each model at its own current rate, summed; None when any
    part is unpriced or unlabelled (lane 02's `price.price_model_tokens`)."""
    from ..price import price_model_tokens

    usd = price_model_tokens(split, table).get("usd")
    return float(usd) if usd is not None else None


def attempt_cost(cost: dict, model: str) -> tuple[float | None, float | None, bool]:
    """(usd, total tokens, repriced) for one attempt's cost record (spec 04 section 2).

    Dollars are repriced under the current tariff, each model's tokens at that model's rate:
    - a split in `ext["dev.loopmath.model_tokens"]`, of any number of models, prices each model's
      part; an unpriced part or one under `"unknown"` leaves no dollars, so the attempt gives no
      cost observation;
    - `{}` there means several models and no split: the recorded dollars stay, not repriced, and are
      never collapsed onto one model;
    - no split but logmatch `parts`: the split is summed from the parts and priced the same way;
    - neither: every token ran on `model`, and without a rate for it the recorded dollars stay.
    Without a price table every attempt keeps its recorded dollars.
    """
    fields = _ocp_fields(cost)
    tokens = sum(fields.values()) or None
    recorded = cost.get("usd")
    recorded = float(recorded) if recorded is not None else None
    table = _price_table()
    if table is None or not tokens:
        return recorded, tokens, False
    ext = cost.get("ext") or {}
    split = ext.get(MODEL_TOKENS)
    if split is not None and not (isinstance(split, dict) and split):
        return recorded, tokens, False
    if split is None:
        from ..logmatch.costs import model_tokens_from_parts

        parts = (ext.get("dev.loopmath.logmatch") or {}).get("parts")
        split = model_tokens_from_parts(parts) if isinstance(parts, list) else {}
    if split:
        usd = _price_split(split, table)
        return usd, tokens, usd is not None
    usd = _price_split({model: fields}, table)
    return (usd, tokens, True) if usd is not None else (recorded, tokens, False)


def _cost_weight(cost: dict) -> float:
    """`basis: measured` is verified (1.0), `allocated` is heuristic (0.7); asserted rows are excluded (0)."""
    basis = str(cost.get("basis") or "").lower()
    match = ((cost.get("ext") or {}).get("dev.loopmath.logmatch") or {})
    tier = str(match.get("tier") or cost.get("evidence") or cost.get("tier") or "").lower()
    if tier == "asserted" or basis == "asserted":
        return 0.0
    if basis == "allocated" or tier == "heuristic":
        return HEURISTIC_COST_WEIGHT
    return 1.0


def _verdict_pass(value: Any) -> bool | None:
    v = str(value or "").lower()
    if v in ("accept", "pass", "approve", "approved", "done"):
        return True
    if v in ("reject", "fail", "rejected", "failed"):
        return False
    return None


def _check_name(a: dict, node: dict) -> str | None:
    """The check a non-model attempt ran (a gate's command, such as the test suite), else None.

    A check has no model, no usage and no workflow piece: the harness ran it, so it is not
    evidence about any setting and it is not a dropped observation.
    """
    setting = a.get("setting") if isinstance(a.get("setting"), dict) else {}
    cost = a.get("cost") if isinstance(a.get("cost"), dict) else {}
    if a.get("model") or setting.get("model"):
        return None
    if any(float(cost.get(k) or 0) > 0 for k in ("usd", *_OCP_TOKEN_FIELDS)):
        return None
    gate = node.get("gate") if isinstance(node.get("gate"), dict) else {}
    return str(gate.get("rule") or node.get("id") or a.get("node") or a.get("harness") or "check")


def parse_run(doc: dict, *, now: datetime | None = None, rule: AcceptanceRule | None = None,
              origin: str | None = None) -> ParsedRun:
    """Everything the fit needs from one finished OCP v0.3 run document. Raises Unusable.

    `origin` is where the fit found the document (`data_source`); None reads its label.
    """
    from .outcome import outcome_evidence

    run = doc.get("run") or {}
    cfg = run.get("configuration")
    if not isinstance(cfg, dict):
        raise Unusable("no configuration")
    config = config_from_doc(cfg)
    task = task_from_doc(doc, origin)
    source = task.source
    st = structure(config)
    dropped: dict[str, int] = {}
    checks: dict[str, int] = {}

    nodes = {n.get("id"): n for n in doc.get("nodes") or [] if isinstance(n, dict)}
    node_vertex = {i: n.get("vertex") for i, n in nodes.items()}
    attempts: list[AttemptObs] = []
    by_piece_round: dict[tuple[str, int], list[dict]] = {}
    for a in doc.get("attempts") or []:
        piece = a.get("vertex") or node_vertex.get(a.get("node"))
        if piece not in st.settings:
            name = _check_name(a, nodes.get(a.get("node")) or {})
            if name is not None:
                checks[name] = checks.get(name, 0) + 1
            else:
                dropped["attempt outside the workflow"] = dropped.get("attempt outside the workflow", 0) + 1
            continue
        k = a.get("round")
        if k is None:
            k = a.get("n") if st.widths.get(piece, 1) == 1 else 1
        k = max(1, int(k or 1))
        by_piece_round.setdefault((piece, k), []).append(a)
        base = st.settings[piece]
        override = a.get("setting") or {}
        model = override.get("model") or a.get("model")
        setting = Setting(
            harness=str(override.get("harness") or a.get("harness") or base.harness),
            model=canonical_model_id(_model_str(model)) if model else base.model,
            effort=str(override.get("effort") or a.get("effort") or base.effort),
        )
        cost = a.get("cost") or {}
        weight = _cost_weight(cost) if cost else 0.0
        usd, tokens, repriced = attempt_cost(cost, setting.model) if cost else (None, None, False)
        if weight <= 0 or ((usd is None or usd <= 0) and not tokens):
            dropped["attempt without usable cost"] = dropped.get("attempt without usable cost", 0) + 1
            continue
        attempts.append(AttemptObs(piece, k, usd if usd and usd > 0 else None, tokens, weight, setting, repriced))

    gates = _gate_observations(doc, st, by_piece_round)
    use_rule = rule or rule_from_doc(doc)
    evidence = outcome_evidence(doc, use_rule, now=now or datetime.now().astimezone())
    score_meta: dict[str, dict] = {}
    for s in run.get("signals") or []:
        if s.get("kind") == "score" and isinstance(s.get("value"), (int, float)) and not isinstance(s.get("value"), bool):
            score_meta.setdefault(str(s["name"]), {"unit": s.get("unit"), "better": s.get("better") or "higher",
                                                   "scale": s.get("scale") or "linear"})
    share = (run.get("ext") or {}).get("dev.loopmath.share")
    if isinstance(share, dict):
        for name, meta in (share.get("score_meta") or {}).items():
            score_meta.setdefault(str(name), dict(meta))
        for name in (evidence.scores or {}):
            score_meta.setdefault(str(name), {"unit": None, "better": "higher", "scale": "linear"})
    return ParsedRun(str(run.get("id") or task.id), task, config, source, attempts, gates, evidence,
                     score_meta, use_rule, dropped, checks)


def _gate_observations(doc: dict, st: Structure, by_piece_round: dict) -> list[tuple[int, int, bool]]:
    """(gate index, round, passed) per gate evaluation, read from which pieces ran in which round.

    Within a round the pieces of a loop run in order and each gate either lets the round go on
    or sends it back. So a gate followed by more pieces of its loop (or, outside loops, of the
    workflow) passed in round `k` exactly when one of them ran in round `k`. The loop's last
    gate failed in every round but the last; its last result comes from a verdict signal on the
    gate piece's attempt, else from that attempt's outcome (done passes; rejected or failed fails).

    Infrastructure failures judge nothing: a run flagged `infra_error` gives no gate observations,
    and neither does a round whose gate-piece attempt is flagged `dev.loopmath.infra_error` or
    carries an `error` verdict. Their costs still count.
    """
    run = doc.get("run") or {}
    ext = run.get("ext") or {}
    if (ext.get("dev.loopmath.prior") or {}).get("infra_error") or ext.get("dev.loopmath.infra_error"):
        return []
    signals_by_attempt: dict[str, list[dict]] = {}
    for s in run.get("signals") or []:
        if s.get("kind") == "verdict" and s.get("at_attempt"):
            signals_by_attempt.setdefault(str(s["at_attempt"]), []).append(s)

    def infra(a: dict) -> bool:
        if (a.get("ext") or {}).get("dev.loopmath.infra_error"):
            return True
        return any(str(s.get("value") or "").lower() == "error" for s in signals_by_attempt.get(str(a.get("id")), []))

    ran = set(by_piece_round)
    out: list[tuple[int, int, bool]] = []
    for gi, g in enumerate(st.gates):
        li = st.gate_loop.get(gi)
        members = st.loops[li][1] if li is not None else st.pieces
        if g.after not in members:
            continue
        later = members[members.index(g.after) + 1:]
        rounds = sorted({k for (p, k) in ran if p in members}) if li is not None else [1]
        last = rounds[-1] if rounds else 1
        for k in sorted({k for (p, k) in ran if p == g.after and (li is not None or k == 1)}):
            if any(infra(a) for a in by_piece_round.get((g.after, k), [])):
                continue
            if later:
                out.append((gi, k, any((p, k) in ran for p in later)))
                continue
            if k < last:
                out.append((gi, k, False))
                continue
            passed = None
            for a in by_piece_round.get((g.after, k), []):
                for s in signals_by_attempt.get(str(a.get("id")), []):
                    v = _verdict_pass(s.get("value"))
                    passed = v if v is not None else passed
            if passed is None:
                for a in by_piece_round.get((g.after, k), []):
                    res = ((a.get("outcome") or {}).get("result") or a.get("status") or "").lower()
                    if res in ("rejected", "failed"):
                        passed = False
                    elif res == "done" and passed is None:
                        passed = True
            if passed is not None:
                out.append((gi, k, passed))
    return out


def rows_for_run(run_doc: dict) -> dict:
    """Fitting rows for one run document: {cost, tokens, gate, success, scores} plus its structure."""
    pr = parse_run(run_doc)
    st = structure(pr.config)
    out: dict[str, Any] = {"run": pr.run_id, "source": pr.source, "structure": st,
                           "cost": [], "tokens": [], "gate": [], "success": None, "scores": {}}
    for a in pr.attempts:
        terms = cost_row(pr.task, pr.source, st, a.piece, a.round, a.setting)
        if a.usd:
            out["cost"].append((terms, math.log(a.usd), a.weight))
        if a.tokens:
            out["tokens"].append((terms, math.log(a.tokens), a.weight))
    for gi, k, passed in pr.gates:
        out["gate"].append((gate_row(pr.task, pr.source, st, st.gates[gi], k), 1.0 if passed else 0.0))
    row = run_row(pr.task, pr.source, st)
    if pr.evidence is not None and pr.evidence.z is not None:
        out["success"] = (row, float(pr.evidence.z), float(pr.evidence.q))
    for name, value in (pr.evidence.scores if pr.evidence else {}).items():
        out["scores"][name] = (row, float(value))
    return out
