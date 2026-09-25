"""The posterior view and the `loopmath posterior` handler (spec 06 section 3, spec 03 section 8).

`build_view` turns the latest fit into one `loopmath.view.posterior/1` object;
`--json` prints it and `--html` embeds it in a self-contained page that renders
from that object and nothing else. `enrich` adds the derived numbers (each
piece's share of the run's cost, the repair loops, the estimates behind each
piece's setting) so the terminal, the JSON and the page show the same values.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..output import EXIT_NO_FIT, EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail, home as store_home
from ..types import TASK_TYPE_IDS, Configuration, Task
from .common import TAIL_NOTE, embed_json, extract_data, html_target, write_page  # noqa: F401 (extract_data, embed_json: the page's own)

SCHEMA = "loopmath.view.posterior/1"
SECTIONS = ("model", "effort", "role", "topology", "type", "repo", "feature")
HEADS = ("cost", "success", "gate")
TERMINAL_LINES = 25

# Fine level name (NodeSummary.level) -> view section. Levels not listed get their own section.
_LEVEL_SECTION = {
    "provider": "model", "family": "model", "version": "model", "model": "model",
    "effort": "effort",
    "role": "role",
    "topology": "topology", "position": "topology", "workflow": "topology",
    "type": "type",
    "repo": "repo", "subtype": "repo",
    "psrc": "topology", "fsrc": "model",  # lane 2B (0.2): position x source and family x source, cost and tokens only
}
# Lane 5's interaction levels (belief.forest), keys joined with `|`, and how people read level names.
_INTERACTIONS = {"family_effort": ["family", "effort"], "role_family": ["role", "family"]}
_LEVEL_WORDS = {"family_effort": "family x effort", "role_family": "role x family", "model": "version",
                "psrc": "position x source", "fsrc": "family x source"}
UNTYPED = "untyped (no task type recorded)"  # the prior's `unknown` task type (E0 runs), view text only
_MODEL_LEVELS = ("provider", "family", "version", "model")
_TREE_RANK = {"provider": 0, "family": 1, "version": 2, "model": 2}


class ViewError(Exception):
    def __init__(self, message: str, code: int = EXIT_USER):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- levels
def interaction_parts(level: str) -> list[str] | None:
    """`family_effort` or `family x effort` -> ["family", "effort"]; None for a plain level."""
    if level in _INTERACTIONS:
        return list(_INTERACTIONS[level])
    for sep in (" x ", "_x_", " × ", "×"):
        if sep in level:
            return [part.strip() for part in level.split(sep)]
    return None


def parent_ref(parent: Any) -> tuple[str | None, str] | None:
    """A node's parent as (level, key): lane 5 gives node ids (`family:opus`, `repo:feature/acme/app`),
    the fixtures give bare keys (`opus`), which match on the key alone."""
    if parent is None:
        return None
    text = str(parent)
    level, sep, rest = text.partition(":")
    if not sep or not level.replace("_", "").isalpha():
        return (None, text)
    if level in ("repo", "subtype") and "/" in rest:
        rest = rest.split("/", 1)[1]
    return (level, rest)


def split_key(key: str, n: int) -> list[str]:
    """An interaction node's key, `opus|xhigh`, split into its `n` parts."""
    for sep in ("|", " x ", "×", ",", ":", "/"):
        parts = [p.strip() for p in str(key).split(sep)]
        if len(parts) == n:
            return parts
    return [str(key)]


def section_of(level: str) -> str:
    if level.startswith("feature:") or level == "feature":
        return "feature"
    parts = interaction_parts(level)
    if parts:
        for name in ("effort", "role"):
            if name in parts:
                return name
        if any(p in _MODEL_LEVELS for p in parts):
            return "model"
        return level
    return _LEVEL_SECTION.get(level, level)


def head_rank(head: str) -> tuple[int, str]:
    if head in HEADS:
        return (HEADS.index(head), head)
    return (len(HEADS) + (0 if head.startswith("score:") else 1), head)


def _node_dict(node: Any) -> dict[str, Any]:
    return node.to_dict() if hasattr(node, "to_dict") else dict(node)


def _order_section(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Heads in order; within a head, parents before children (the model tree), bigger support first."""
    out: list[dict[str, Any]] = []
    for head in sorted({n["head"] for n in nodes}, key=head_rank):
        group = [n for n in nodes if n["head"] == head]
        plain = [n for n in group if not interaction_parts(n["level"])]
        inter = [n for n in group if interaction_parts(n["level"])]
        by_full = {(n["level"], n["key"]): n for n in plain}
        by_key = {n["key"]: n for n in plain}
        children: dict[int, list[dict[str, Any]]] = {}
        roots = []
        for n in plain:
            ref = parent_ref(n.get("parent"))
            up = (by_full.get(ref) if ref and ref[0] else by_key.get(ref[1])) if ref else None
            if up is not None and up is not n:
                children.setdefault(id(up), []).append(n)
            else:
                roots.append(n)

        def rank(n: dict[str, Any]) -> tuple:
            return (_TREE_RANK.get(n["level"], 0), n["level"], -int(n.get("support") or 0), str(n["key"]))

        seen: set[int] = set()

        def walk(n: dict[str, Any]) -> None:
            if id(n) in seen:
                return
            seen.add(id(n))
            out.append(n)
            for child in sorted(children.get(id(n), []), key=rank):
                walk(child)

        for root in sorted(roots, key=rank):
            walk(root)
        for n in plain:  # nodes on a parent cycle
            walk(n)
        out.extend(sorted(inter, key=lambda n: (n["level"], str(n["key"]))))
    return out


def group_levels(nodes: Iterable[Any]) -> dict[str, list[dict[str, Any]]]:
    """Node summaries grouped into view sections: the seven of spec 06 always, then any other level."""
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in SECTIONS}
    for node in nodes:
        d = _node_dict(node)
        grouped.setdefault(section_of(str(d.get("level", ""))), []).append(d)
    extra = sorted(k for k in grouped if k not in SECTIONS)
    return {k: _order_section(grouped[k]) for k in (*SECTIONS, *extra)}


def heads_in(levels: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted({n["head"] for nodes in levels.values() for n in nodes}, key=head_rank)


def filter_levels(levels: dict[str, list[dict[str, Any]]], *, level: str = "all",
                  head: str | None = None) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for name, nodes in levels.items():
        if level not in ("all", None) and name != level:
            continue
        out[name] = [n for n in nodes if head is None or n["head"] == head]
    return out


def _all_nodes(levels: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [n for nodes in levels.values() for n in nodes]


# ---------------------------------------------------------------- workflow graph
def view_label(config: Configuration) -> str:
    """`Configuration.label()` with each piece's width (I12): 'best_of_n: 3 x gpt-5.6-sol/xhigh'."""
    parts = []
    for piece in config.workflow.pieces:
        s = config.settings.get(piece.id)
        if s is not None:
            parts.append((f"{piece.width} x " if piece.width > 1 else "") + f"{s.model}/{s.effort}")
    return f"{config.workflow.id}: " + ", ".join(parts)


def workflow_graph(config: Configuration, prediction: Any) -> dict[str, Any]:
    """The workflow graph dict the fixtures use: pieces with setting and prediction, artifacts, edges, gates."""
    per_piece = getattr(prediction, "per_piece", {}) or {}
    nodes = []
    for piece in config.workflow.pieces:
        setting = config.settings.get(piece.id) or piece.setting
        node: dict[str, Any] = {"id": piece.id, "kind": "piece", "role": piece.role,
                                "setting": setting.to_dict() if setting else None}
        if piece.width != 1:
            node["width"] = piece.width
        if piece.id in per_piece:
            node["prediction"] = _node_dict(per_piece[piece.id])
        nodes.append(node)
    nodes += [{"id": a, "kind": "artifact"} for a in config.workflow.artifacts]
    return {"config": config.id, "label": view_label(config), "nodes": nodes,
            "edges": [{"from": a, "to": b} for a, b in config.workflow.edges],
            "gates": [g.to_dict() for g in config.workflow.control.gates],
            "budget_rounds": max(1, config.workflow.control.budget_rounds or 1)}


def _reach(start: str, adj: dict[str, list[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        for nxt in adj.get(stack.pop(), []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _pieces_of(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in graph.get("nodes", []) if n.get("kind", "piece") == "piece"]


def _per_piece(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_piece = dict(workflow.get("per_piece") or {})
    for node in _pieces_of(workflow.get("graph") or {}):
        if node["id"] not in per_piece and node.get("prediction"):
            per_piece[node["id"]] = node["prediction"]
    return per_piece


def _mean(d: Any, default: float = 0.0) -> float:
    value = d.get("mean") if isinstance(d, dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def piece_totals(per_piece: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Expected spend per piece over the run (means).

    `PiecePrediction.cost` is already the piece's full-run contribution and sums to the run's
    cost, so it is never multiplied by the expected rounds here.
    """
    out = {}
    for pid, pred in per_piece.items():
        cost = pred.get("cost") or {}
        out[pid] = {"usd": _mean(cost.get("usd")), "tokens": _mean(cost.get("tokens"))}
    return out


def cost_shares(per_piece: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Each piece's share of the run's expected dollars."""
    totals = piece_totals(per_piece)
    whole = sum(t["usd"] for t in totals.values())
    if whole <= 0:
        return {pid: 0.0 for pid in totals}
    return {pid: round(t["usd"] / whole, 4) for pid, t in totals.items()}


def repair_loops(graph: dict[str, Any], per_piece: dict[str, dict[str, Any]],
                 shares: dict[str, float]) -> list[dict[str, Any]]:
    """One loop per gate with `on_fail`: the pieces on the path from `on_fail` to the judged piece.

    Workflows are acyclic, so loops come from control, never from back edges.
    """
    adj: dict[str, list[str]] = {}
    radj: dict[str, list[str]] = {}
    for e in graph.get("edges", []):
        a, b = (e["from"], e["to"]) if isinstance(e, dict) else (e[0], e[1])
        adj.setdefault(a, []).append(b)
        radj.setdefault(b, []).append(a)
    order = [n["id"] for n in _pieces_of(graph)]
    loops = []
    for gate in graph.get("gates", []):
        start, end = gate.get("on_fail"), gate.get("after")
        if not start or not end:
            continue
        inside = (_reach(start, adj) | {start}) & (_reach(end, radj) | {end})
        pieces = [pid for pid in order if pid in inside] or [p for p in (start, end) if p in order]
        judged = per_piece.get(end) or {}
        loops.append({"gate": gate.get("id"), "rule": gate.get("rule"), "from": start, "to": end,
                      "pieces": pieces, "rounds": judged.get("rounds"),
                      "share": round(sum(shares.get(p, 0.0) for p in pieces), 4)})
    return loops


def gate_rows(graph: dict[str, Any], per_piece: dict[str, dict[str, Any]],
              given: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Every gate with its pass chance per round and expected rounds."""
    by_id = {g.get("id"): dict(g) for g in (given or [])}
    rows = []
    for gate in graph.get("gates", []):
        row = {**gate, **by_id.pop(gate.get("id"), {})}
        judged = per_piece.get(gate.get("after")) or {}
        if row.get("pass") is None:
            row["pass"] = judged.get("gate_pass")
        if row.get("rounds") is None:
            row["rounds"] = judged.get("rounds")
        rows.append(row)
    rows.extend(by_id.values())
    return rows


def _model_ancestors(model: str | None, nodes: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """(family, provider) of a model id, from the fit's own tree, else from the belief forest."""
    if not model:
        return None, None
    parent_of = {(n["level"], n["key"]): (parent_ref(n.get("parent")) or (None, None))[1] for n in nodes}
    family = parent_of.get(("version", model)) or parent_of.get(("model", model))
    provider = parent_of.get(("family", family)) if family else None
    if family is None:
        from ..belief.forest import model_path

        provider, family, _version = model_path(model)
    return family, provider


def piece_effects(node: dict[str, Any], nodes: list[dict[str, Any]], position: str | None = None) -> list[dict[str, Any]]:
    """The estimates behind one piece's setting: version, family, provider, effort, role, harness,
    the piece's position in its workflow (`<workflow>#<index>`, lane 5's key) and the interactions, plus
    lane 2B's position x source and family x source nodes for the source predictions use (0.2)."""
    try:
        from ..belief.state import PREDICT_SOURCE as source
    except ImportError:
        source = "user"
    setting = node.get("setting") or {}
    model = setting.get("model")
    family, provider = _model_ancestors(model, nodes)
    values = {"version": model, "model": model, "family": family, "provider": provider,
              "effort": setting.get("effort"), "role": node.get("role"), "harness": setting.get("harness"),
              "position": position, "psrc": f"{position}|{source}" if position else None,
              "fsrc": f"{family}|{source}" if family else None}
    out = []
    for n in nodes:
        level = n["level"]
        parts = interaction_parts(level)
        if parts:
            wanted = [values.get(p) for p in parts]
            if None not in wanted and split_key(n["key"], len(parts)) == [str(w) for w in wanted]:
                out.append(n)
        elif level in values and values[level] is not None and n["key"] == values[level]:
            out.append(n)
    return out


def node_ref(node: dict[str, Any]) -> str:
    """`<level>:<key>`: how `effects` points into `levels` (one node per head shares the ref)."""
    return f"{node['level']}:{node['key']}"


def _refs(nodes: Iterable[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(node_ref(n) for n in nodes))


def resolve_refs(refs: Iterable[str], levels: dict[str, list[dict[str, Any]]],
                 head: str | None = None) -> list[dict[str, Any]]:
    """The nodes `effects` refers to, in ref order, for one head or all."""
    index: dict[str, list[dict[str, Any]]] = {}
    for n in _all_nodes(levels):
        if head is None or n["head"] == head:
            index.setdefault(node_ref(n), []).append(n)
    return [n for ref in refs for n in index.get(ref, [])]


def loop_text(loop: dict[str, Any]) -> str:
    """`review back to implement`, or `implement retries itself` when the gate sends a piece back to itself."""
    return f"{loop['to']} retries itself" if loop["to"] == loop["from"] else f"{loop['to']} back to {loop['from']}"


def _workflow_id(workflow: dict[str, Any]) -> str:
    label = workflow.get("label") or (workflow.get("graph") or {}).get("label") or ""
    return label.split(":", 1)[0].strip()


def shape_of(wid: str, graph: dict[str, Any], nodes: list[dict[str, Any]]) -> str:
    """The key belief gives this graph's topology and position nodes (lane 2B, 0.2): the workflow id for a catalog
    shape, `<id>~<sorted roles joined by +>` when the role set differs from the catalog workflow of that id. Read
    from `belief.design.shape_key` when it is there; otherwise the `~` key when the fit has such a node, else the id."""
    roles = tuple(str(n.get("role") or n.get("id")) for n in _pieces_of(graph))
    try:
        from ..belief.design import shape_key
    except ImportError:
        shape_key = None
    if shape_key is not None:
        try:
            return str(shape_key(wid, roles))
        except (TypeError, ValueError, KeyError):
            pass
    keys = {str(n.get("key")) for n in nodes if n.get("level") in ("topology", "position")}
    for tilde in (f"{wid}~{'+'.join(sorted(roles))}", f"{wid}~{'+'.join(sorted(set(roles)))}"):
        if tilde in keys or any(k.startswith(tilde + "#") for k in keys):
            return tilde
    return wid


def enrich_workflow(workflow: dict[str, Any], nodes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Add per-piece predictions, shares, loops, gate rows and setting effects; values already present are kept."""
    w = dict(workflow)
    graph = w.get("graph") or {}
    per_piece = _per_piece(w)
    w["per_piece"] = per_piece
    shares = w.setdefault("shares", cost_shares(per_piece))
    w.setdefault("loops", repair_loops(graph, per_piece, shares))
    w["gates"] = gate_rows(graph, per_piece, w.get("gates"))
    if nodes is not None and "effects" not in w:
        shape = shape_of(_workflow_id(w), graph, nodes)
        effects = {n["id"]: _refs(piece_effects(n, nodes, f"{shape}#{i}")) for i, n in enumerate(_pieces_of(graph))}
        topo = _refs(n for n in nodes if n["level"] == "topology" and n["key"] == shape)
        w["effects"] = {"pieces": effects, "workflow": topo}
    return w


def _finite(value: Any) -> Any:
    """NaN and infinities become null, so the object is valid JSON everywhere."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def enrich(data: dict[str, Any]) -> dict[str, Any]:
    """The object the page embeds and `--json` prints: `data` plus derived numbers. Idempotent."""
    out = _finite(copy.deepcopy(data))
    levels = out.get("levels") or {}
    nodes = _all_nodes(levels)
    heads = out.get("heads") if isinstance(out.get("heads"), dict) else {}
    for head in heads_in(levels):
        meta = heads.setdefault(head, {})
        meta.setdefault("kind", "multiplier" if head in ("cost", "tokens") else
                        "pp" if head in ("success", "gate") else "shift")
    out["heads"] = heads
    if out.get("workflow"):
        out["workflow"] = enrich_workflow(out["workflow"], nodes)
    if out.get("workflows"):
        out["workflows"] = [enrich_workflow(w, nodes) for w in out["workflows"]]
    return out


# ---------------------------------------------------------------- store readers
def _index_rows(home: Path) -> list[dict[str, Any]]:
    """One runs/index.jsonl row per run, the last one (lane 7's `Store.index_rows`: a run appends a row when it
    starts and again when it finishes). Readers never take the lock (spec 03 section 4)."""
    from ..store import Store

    try:
        return list(Store(home).index_rows().values())
    except OSError:
        return []


def _json_files(folder: Path) -> list[Path]:
    try:
        return sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []


def _find_config_dict(obj: Any, cfg: str) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if obj.get("id") == cfg and "workflow" in obj and "settings" in obj:
            return obj
        values: Iterable[Any] = obj.values()
    elif isinstance(obj, list):
        values = obj
    else:
        return None
    for value in values:
        found = _find_config_dict(value, cfg)
        if found is not None:
            return found
    return None


_ROLE_RULE = {"reviewer": "review_approve", "tester": "tests_pass", "referee": "referee_pick"}
_RESCUE_WORD = {"person": "person", "none": "none"}


def _control_from_dict(control: dict[str, Any], pieces: list[Any]) -> dict[str, Any]:
    """The types form of `control`, also from OCP v0.3.

    OCP lists the pieces after which a gate runs, maps each to the piece that reruns on
    reject in `repair`, counts K_max in `budget` (missing or 0 reads as 1) and keeps
    non-default gate rules in `ext["dev.loopmath.gate_rules"]`.
    """
    out = dict(control)
    roles = {p.get("id"): p.get("role") for p in pieces if isinstance(p, dict)}
    repair = control.get("repair") if isinstance(control.get("repair"), dict) else {}
    rules = (control.get("ext") or {}).get("dev.loopmath.gate_rules") or {}
    gates = []
    for g in control.get("gates") or []:
        if isinstance(g, dict):
            gates.append(g)
        elif isinstance(g, str):
            rule = rules.get(g) if isinstance(rules, dict) else None
            gates.append({"id": f"g_{g}", "after": g, "rule": rule or _ROLE_RULE.get(roles.get(g), "review_approve"),
                          "on_fail": repair.get(g)})
    out["gates"] = gates
    out.pop("repair", None)
    if "budget" in out:
        out["budget_rounds"] = max(1, int(out.pop("budget") or 0))
    rescue = out.get("rescue")
    if isinstance(rescue, dict):
        kind = rescue.get("kind")
        if kind == "configuration" and rescue.get("ref") == "usual":
            out["rescue"] = "redo_usual"
        elif kind in _RESCUE_WORD:
            out["rescue"] = _RESCUE_WORD[kind]
    return out


def _config_from_dict(d: dict[str, Any]) -> Configuration:
    """A configuration from the types form or the OCP v0.3 form (artifacts and edges as objects, model refs)."""
    wf = dict(d["workflow"])
    wf["artifacts"] = [a["id"] if isinstance(a, dict) else a for a in wf.get("artifacts", [])]
    wf["edges"] = [[e["from"], e["to"]] if isinstance(e, dict) else e for e in wf.get("edges", [])]
    wf.setdefault("version", 1)
    wf.setdefault("title", wf.get("id", ""))
    control = wf.get("control")
    if isinstance(control, dict):
        wf["control"] = _control_from_dict(control, wf.get("pieces") or [])
    settings = {}
    for pid, s in (d.get("settings") or {}).items():
        s = dict(s)
        if isinstance(s.get("model"), dict):
            s["model"] = s["model"].get("id") or s["model"].get("raw")
        settings[pid] = s
    return Configuration.from_dict({"id": d["id"], "workflow": wf, "settings": settings})


def resolve_config(home: Path, cfg: str) -> Configuration | None:
    """A configuration id seen in the store: recommendations first (newest first), then run documents."""
    runs = sorted((home / "runs").glob("*.ocp.json")) if (home / "runs").is_dir() else []
    for path in [*_json_files(home / "recs"), *runs]:
        try:
            found = _find_config_dict(json.loads(path.read_text(encoding="utf-8")), cfg)
        except (OSError, json.JSONDecodeError):
            continue
        if found is not None:
            config = _read_config(found)
            if config is not None:
                return config
    return None


def _config_not_found(workflow: str, home: Path) -> str:
    """Where configuration ids come from, and what to pass for a catalog shape, which has no models."""
    from ..workflows.format import catalog

    text = f"configuration {workflow} not found in recommendations or runs under {home}"
    if workflow in catalog():
        return (text + f"; {workflow} is a catalog workflow with no models set: pass a configuration id (cfg_...)"
                " from `loopmath recommend --json` or your runs, or a workflow file with settings")
    return text + "; configuration ids (cfg_...) come from `loopmath recommend --json` and your runs"


def _read_config(d: dict[str, Any]) -> Configuration | None:
    from ..workflows.ocp import configuration_from_any  # also reads an OCP workflow given as a catalog `{ref, version}`

    try:
        return _config_from_dict(d)
    except (KeyError, TypeError, ValueError, AttributeError):
        pass
    try:
        return configuration_from_any(d)
    except (LookupError, KeyError, TypeError, ValueError, AttributeError):
        return None


def _usual_config_id(home: Path, task: Task) -> str | None:
    """The usual configuration from config.toml, read by lane 7's `Config.usual`: `usual.<type>."<repo>"`,
    then `usual.<type>."*"`, or `usual.<type>` itself when `loopmath config set usual.<type> CFG` wrote a plain id."""
    from ..store import Store
    from ..store.config import ConfigError

    try:
        return Store(home).config().usual(task.type, task.repo)
    except (OSError, ConfigError):
        return None


def _load_toml_config(path: Path) -> Configuration:
    """A configuration from a workflow file: settings come from its `[settings.<piece>]` tables."""
    from ..workflows.format import load_workflow_file

    try:
        found = load_workflow_file(path)
    except ValueError as exc:  # WorkflowFormatError names every problem
        raise ViewError(str(exc)) from None
    config = found.configuration()
    if config is None:
        missing = [p.id for p in found.workflow.pieces if p.id not in found.settings]
        raise ViewError(f"{path}: pieces without a setting: {', '.join(missing)}; "
                        "add a [settings.<piece>] table with harness, model and effort for each")
    return config


# ---------------------------------------------------------------- fit metadata
def _read_meta(home: Path, fit_id: str) -> dict[str, Any]:
    for folder in (home / "fits" / fit_id, home / "fits" / "latest"):
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(meta, dict):
            return meta
    return {}


def _dropped(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [{"reason": str(k), "n": v} for k, v in value.items()]
    if not isinstance(value, list):
        return []
    counts: Counter[str] = Counter()
    out = []
    for item in value:
        if isinstance(item, dict) and "n" in item:
            out.append(item)
        elif isinstance(item, dict):
            counts[str(item.get("reason", "unknown"))] += 1
        else:
            counts[str(item)] += 1
    return out + [{"reason": r, "n": n} for r, n in counts.items()]


def data_block(meta: dict[str, Any]) -> dict[str, Any]:
    """The data tab: rows per source and head, dropped rows, fit time, scales, sensitivity."""
    def first(*keys: str) -> Any:
        return next((meta[k] for k in keys if meta.get(k) is not None), None)

    heads = meta.get("heads") if isinstance(meta.get("heads"), dict) else {}

    def per_head(field: str) -> dict[str, Any]:  # lane 5 keeps rows and phi per head
        return {h: dict(m[field]) for h, m in heads.items() if isinstance(m, dict) and isinstance(m.get(field), dict)}

    options = meta.get("options") if isinstance(meta.get("options"), dict) else {}
    return {"rows": first("rows", "row_counts") or per_head("rows_by_source") or {},
            "runs_by_source": first("runs_by_source") or {},
            "dropped": _dropped(first("dropped", "dropped_rows")),
            "fit_time_s": first("fit_time_s", "elapsed_s", "seconds"),
            "code_version": first("code_version", "loopmath_version"),
            "scales": first("scales") or per_head("phi") or {},
            "without": list(options.get("without") or []),
            "sensitivity": first("sensitivity")}


def _n_runs(meta: dict[str, Any]) -> Any:
    value = meta.get("n_runs")
    if value is not None:
        return value
    runs = meta.get("runs_by_source")
    if isinstance(runs, dict):
        user = int(runs.get("user", 0))
        return {"prior": sum(int(v) for v in runs.values()) - user, "user": user}
    return None


# ---------------------------------------------------------------- building the view
def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def choose_task(home: Path, task_type: str | None, repo: str | None, subtype: str | None = None,
                features: dict[str, str] | None = None) -> tuple[Task, str]:
    """The task the graphs are computed for: the arguments, else the most common (type, repo) in the store.

    `subtype` and `features` (`--subtype`, `--feature`, as `recommend`) go into the prediction either way.
    """
    if task_type is not None and task_type not in TASK_TYPE_IDS:
        raise ViewError(f"unknown task type {task_type!r}; see loopmath task-types")
    more = {"subtype": subtype or None, "features": dict(features or {})}
    if task_type and repo:
        return Task(id="tsk_posterior_view", type=task_type, repo=repo, **more), "arguments"
    counts = Counter((r.get("task_type"), r.get("repo")) for r in _index_rows(home)
                     if r.get("task_type") in TASK_TYPE_IDS and r.get("repo")
                     and task_type in (None, r.get("task_type")) and repo in (None, r.get("repo")))
    if counts:
        (t, r), _ = counts.most_common(1)[0]
        return Task(id="tsk_posterior_view", type=t, repo=r, **more), "store"
    return Task(id="tsk_posterior_view", type=task_type or "feature", repo=repo or "unknown", **more), "default"


def task_features(items: Iterable[str], features: Any = None,
                  horizon: str | None = None) -> tuple[dict[str, str], list[str]]:
    """`--feature K=V` items and `--horizon` as `recommend` reads them (with the fit's declared
    features), and the keys the model does not know."""
    from ..taskmodel import HORIZON_KEY, normalize_features, parse_horizon

    raw: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ViewError(f"--feature takes K=V; got {item!r}")
        k, v = item.split("=", 1)
        raw[k.strip()] = v.strip()
    feats = normalize_features(raw, features)
    if horizon is not None:
        try:
            secs = parse_horizon(horizon)
        except ValueError as exc:
            raise ViewError(f"--horizon: {exc}") from None
        feats[HORIZON_KEY] = (str(int(secs)) if float(secs).is_integer() else str(secs)) if secs else "none"
    return feats, sorted(k[len("extra:"):] for k in feats if k.startswith("extra:"))


def subtype_note(state: Any, task: dict[str, Any]) -> str | None:
    """A note when `--subtype` names a subtype the fit has no runs of, with the ones it has."""
    sub = task.get("subtype")
    if not sub:
        return None
    ts = (getattr(state, "design", None) or {}).get("task_support") or {}
    prefix = f"subtype:{task.get('type')}/{task.get('repo')}/"
    if ts.get(prefix + sub):
        return None
    known = sorted(k[len(prefix):] for k in ts if k.startswith(prefix))
    return (f"note: fit {state.fit_id} has no runs of subtype {sub} for {task.get('type')} in {task.get('repo')}"
            + (f" (it has {', '.join(known)})" if known else "") + "; the estimates are for a new subtype")


def _entry(state: Any, task: Task, config: Configuration, origin: str) -> dict[str, Any]:
    prediction = state.predict(task, config)
    return {"config": config.id, "label": view_label(config), "origin": origin, "group": config.workflow.id,
            "graph": workflow_graph(config, prediction),
            "per_piece": {k: _node_dict(v) for k, v in (prediction.per_piece or {}).items()},
            "gates": [], "prediction": _node_dict(prediction)}


def choose_target(home: Path, task: Task, head: str | None, present: list[str]) -> dict[str, Any] | None:
    """The head that orders the workflow list (I11): `--head` when it is success or a score, else the score the
    newest recommendation for this task type targets, else success."""
    if head == "success" or (head or "").startswith("score:"):
        return {"head": head, "from": "argument"}
    for path in _json_files(home / "recs"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict) or (rec.get("task") or {}).get("type") not in (None, task.type):
            continue
        score = (rec.get("rule") or {}).get("score") or {}
        if score.get("name") and f"score:{score['name']}" in present:
            return {"head": f"score:{score['name']}", "better": score.get("better") or "higher",
                    "target": score.get("target"), "from": "recommendation"}
        break  # the newest recommendation for the type decides
    return {"head": "success", "from": "default"} if "success" in present else None


def target_value(prediction: dict[str, Any], target: dict[str, Any] | None) -> float | None:
    """The prediction's mean on the target head as a sort key, bigger is better; None when not predicted."""
    if not target:
        return None
    if target["head"] == "success":
        value, better = (prediction.get("p_success") or {}).get("mean"), "higher"
    else:
        score = (prediction.get("scores") or {}).get(target["head"].split(":", 1)[1]) or {}
        value, better = (score.get("value") or {}).get("mean"), score.get("better") or target.get("better")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return -float(value) if better == "lower" else float(value)


def order_workflows(entries: list[dict[str, Any]], target: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Grouped by workflow graph (I11): the usual's group first, then groups by their best configuration on the
    target; inside a group the usual, then by runs, ties broken by the target, then by label."""
    def key(e: dict[str, Any]) -> float:
        v = target_value(e.get("prediction") or {}, target)
        return -math.inf if v is None else v

    groups: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        groups.setdefault(str(e.get("group") or ""), []).append(e)
    for members in groups.values():
        members.sort(key=lambda e: (e.get("origin") != "usual", -int(e.get("runs") or 0), -key(e), str(e.get("label") or "")))
    order = sorted(groups, key=lambda g: (not any(e.get("origin") == "usual" for e in groups[g]),
                                          -max(key(e) for e in groups[g]), g))
    return [e for g in order for e in groups[g]]


def _run_config(home: Path, run: Any, cfg: str) -> Configuration | None:
    """A configuration read from the one run document the index names for it."""
    if not run:
        return None
    try:
        found = _find_config_dict(json.loads((home / "runs" / f"{run}.ocp.json").read_text(encoding="utf-8")), cfg)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return _read_config(found) if found is not None else None


def _workflow_list(state: Any, home: Path, task: Task, target: dict[str, Any] | None = None,
                   configs: dict[str, Configuration] | None = None) -> list[dict[str, Any]]:
    """Every configuration recorded for the task's (type, repo) and the usual one (I11), grouped by graph.
    `configs`, when given, collects each listed configuration by id."""
    usual = _usual_config_id(home, task)
    rows = [r for r in _index_rows(home)
            if r.get("task_type") == task.type and r.get("repo") == task.repo and r.get("config")]
    counts = Counter(r["config"] for r in rows)
    first_run: dict[str, Any] = {}
    for r in rows:
        first_run.setdefault(r["config"], r.get("run"))
    ids = ([usual] if usual else []) + [cfg for cfg, _ in counts.most_common() if cfg != usual]
    out = []
    for cfg in ids:
        config = _run_config(home, first_run.get(cfg), cfg) or resolve_config(home, cfg)
        if config is not None:
            entry = _entry(state, task, config, "usual" if cfg == usual else "recorded")
            entry["runs"] = counts.get(cfg, 0)
            out.append(entry)
            if configs is not None:
                configs[config.id] = config
    return order_workflows(out, target)


# ---------------------------------------------------------------- results page (0.2, D119)
def _iv3(d: Any) -> dict[str, Any] | None:
    """An interval as {mean, lo, hi}, rounded like the recommend JSON."""
    d = _node_dict(d) if d is not None else None
    if not isinstance(d, dict) or not isinstance(d.get("mean"), (int, float)):
        return None
    return {k: (round(float(d[k]), 6) if isinstance(d.get(k), (int, float)) else None) for k in ("mean", "lo", "hi")}


def _rec_rule(rec: dict[str, Any]) -> Any:
    """A stored recommendation's rule when it has a score target, else None."""
    from ..recommend.commands import UserError, parse_target
    from ..types import AcceptanceRule

    rule = rec.get("rule") if isinstance(rec.get("rule"), dict) else {}
    score = rule.get("score") if isinstance(rule.get("score"), dict) else None
    if not score or not score.get("name") or not isinstance(score.get("target"), (int, float)):
        return None
    try:
        return AcceptanceRule.from_dict(rule)
    except (TypeError, ValueError, KeyError):
        try:
            return parse_target(f"{score['name']}{'<=' if score.get('better') == 'lower' else '>='}{float(score['target']):g}")
        except UserError:
            return None


def _same_target(a: Any, b: Any) -> bool:
    sa, sb = a.score, b.score
    return (sa is not None and sb is not None and sa.name == sb.name and float(sa.target) == float(sb.target)
            and (sa.better or "higher") == (sb.better or "higher"))


def results_target(home: Path, task: Task, text: str | None) -> tuple[Any, dict[str, Any] | None, dict[str, Any] | None]:
    """The rule behind the chance-to-reach and cost-per-accepted columns, where it came from, and the rescue that
    prices a miss. The rule is `--target` when given, else the newest stored recommendation's for the task's type
    and repo. The rescue is the one the newest recommendation for the type and repo with that same target priced;
    with none there is no rescue, and the page leaves out the cost per accepted result."""
    from ..recommend.commands import UserError, parse_target

    recs = []
    for path in _json_files(home / "recs"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_of = rec.get("task") if isinstance(rec, dict) and isinstance(rec.get("task"), dict) else None
        if not task_of or task_of.get("type") != task.type or task_of.get("repo") != task.repo:
            continue
        rule = _rec_rule(rec)
        if rule is not None:
            recs.append((rec, rule))
    if text:
        try:
            rule = parse_target(text)
        except UserError as exc:
            raise ViewError(str(exc)) from None
        source: dict[str, Any] = {"from": "argument", "rec": None}
    elif recs:
        rule, source = recs[0][1], {"from": "recommendation", "rec": recs[0][0].get("rec")}
    else:
        return None, None, None
    rescue = None
    for rec, other in recs:
        if _same_target(other, rule):
            found = rec.get("rescue") if isinstance(rec.get("rescue"), dict) else {}
            if isinstance(found.get("usd"), (int, float)) and math.isfinite(found["usd"]):
                rescue = {"kind": found.get("kind"), "usd": round(float(found["usd"]), 6), "basis": found.get("basis"),
                          "of": found.get("of"), "rec": rec.get("rec")}
            break
    return rule, source, rescue


def _chance_to_reach(pred: Any, name: str) -> dict[str, Any] | None:
    """The chance to reach the target, as recommend reads it (I13, P3a): the success head's draws when the fit
    switched `g` to the score head, else the score head's mean chance without a range."""
    if pred.success_from == "score_head":
        return _iv3(pred.p_success)
    s = (pred.scores or {}).get(name)
    return {"mean": round(float(s.p_reach), 6), "lo": None, "hi": None} if s is not None and s.p_reach is not None else None


def results_block(state: Any, task: Task, configs: list[Configuration], rule: Any, source: dict[str, Any] | None,
                  rescue: dict[str, Any] | None) -> dict[str, Any]:
    """Per listed workflow: the chance to reach the target, the target score's interval and, when a rescue is
    priced, the expected rescue and the cost per accepted result (run cost + chance of a miss x rescue).

    `p_accepted` is the chance that arithmetic uses, the prediction's `p_success`: the chance to reach the target
    when `reach_from` is `score_head`, else the success head's chance of an accepted result, which the fit uses
    when it has too few of these scores for the task type (recommend's `score_backed`). `reach` stays the score
    head's chance to reach, so a page can show both and keep the totals the model's. `chance_from` is the one
    `reach_from` of every row (it depends on the task type), or None when the rows differ or there are none."""
    if rule is None or rule.score is None:
        return {"target": None, "rescue": None, "workflows": {}, "chance_from": None}
    name = rule.score.name
    preds = state.predict_many(task, configs, rule, rescue_usd=rescue["usd"] if rescue else None) if configs else []
    rows = {}
    for cfg, p in zip(configs, preds):
        s = (p.scores or {}).get(name)
        rows[cfg.id] = {"reach": _chance_to_reach(p, name), "reach_from": p.success_from,
                        "p_accepted": _iv3(p.p_success),
                        "score": _iv3(s.value) if s is not None else None,
                        "cost_usd": _iv3(p.cost.usd),
                        "expected_rescue_usd": round(max(0.0, p.ell.usd.mean - p.cost.usd.mean), 6) if rescue else None,
                        "cost_per_accepted_usd": _iv3(p.ell.usd) if rescue else None,
                        "support": p.support}
    target = {"rule": rule.name, "definition": rule.definition, "score": name, "target": float(rule.score.target),
              "better": rule.score.better or "higher", **(source or {})}
    froms = {r["reach_from"] for r in rows.values()}
    return {"target": target, "rescue": rescue, "workflows": rows, "chance_from": froms.pop() if len(froms) == 1 else None}


def _allowed(home: Path) -> dict[str, Any]:
    """Recommend's allowed settings from config.toml, so the units are the settings recommend searches."""
    from ..recommend.commands import allowed_settings
    from ..recommend.storeread import Conf

    try:
        return allowed_settings(Conf.load(home), None)
    except (OSError, ValueError):
        return {}


def units_block(state: Any, home: Path, task: Task, configs: list[Configuration], counts: dict[str, int],
                rule: Any, score_name: str | None) -> list[dict[str, Any]]:
    """One agent working alone per allowed harness, model and effort (the models in config `models.allowed`,
    else the models in these runs, at each offered effort), predicted for the task. `user_model` marks a model
    these runs used; `runs` counts the recorded solo runs at that exact setting and `configs` names them."""
    from ..types import Setting
    from ..workflows.candidates import allowed_from, settings_for_shape
    from ..workflows.ids import make_config
    from ..workflows.models import provider_of
    from ..workflows.shapes import catalog_shapes

    seen = [s for c in configs for s in c.settings.values()]
    try:
        al = allowed_from(_allowed(home), seen)
    except (KeyError, ValueError):
        return []
    solo = catalog_shapes()["solo"]
    used = {s.model for s in seen}
    mine: dict[tuple[str, str], list[str]] = {}
    for c in configs:
        pieces = c.workflow.pieces
        if len(pieces) == 1 and pieces[0].width == 1:
            s = c.settings.get(pieces[0].id)
            if s is not None:
                mine.setdefault((s.model, s.effort), []).append(c.id)
    keys, units = [], []
    for model in al.models:
        for effort in al.efforts_of(model):
            setting = Setting(al.harness(model), model, effort)
            keys.append(setting)
            units.append(make_config(solo, settings_for_shape(solo, setting)))
    if not units:
        return []
    preds = state.predict_many(task, units, rule)
    out = []
    for setting, cfg, p in zip(keys, units, preds):
        s = (p.scores or {}).get(score_name) if score_name else None
        ids = mine.get((setting.model, setting.effort), [])
        out.append({"provider": provider_of(setting.model), "harness": setting.harness, "model": setting.model,
                    "effort": setting.effort, "config": cfg.id, "cost": _iv3(p.cost.usd),
                    "perf": _iv3(s.value) if s is not None else None,
                    "reach": _chance_to_reach(p, score_name) if rule is not None and score_name else None,
                    "success": _iv3(p.p_success), "support": p.support, "user_model": setting.model in used,
                    "runs": sum(counts.get(i, 0) for i in ids), "configs": ids})
    return out


def run_points(home: Path, task: Task, score_name: str | None, rule: Any) -> list[dict[str, Any]]:
    """The recorded runs for the task's type and repo: configuration, cost, the named score and whether it
    reached the target, read the way the runs page reads them."""
    from ..store import Store
    from . import common as vcommon
    from . import runs as vruns

    out = []
    try:
        docs = vruns.load_docs(Store(home), [])
    except OSError:
        return out
    target = rule.score if rule is not None and rule.score is not None and rule.score.name == score_name else None
    for doc in docs:
        run = doc.get("run") or {}
        t = run.get("task") if isinstance(run.get("task"), dict) else {}
        if t.get("type") != task.type or t.get("repo") != task.repo:
            continue
        cfg = run.get("configuration") if isinstance(run.get("configuration"), dict) else {}
        sig = vruns._latest(vcommon.run_signals(doc), "score").get(score_name) if score_name else None
        value = sig.get("value") if isinstance(sig, dict) else None
        score = float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
        reached = None
        if score is not None and target is not None:
            reached = score <= target.target if target.better == "lower" else score >= target.target
        cost = vruns.run_cost(doc).get("usd")
        out.append({"run": run.get("id"), "config": cfg.get("id"), "subtype": t.get("subtype"),
                    "started_at": vruns._started(doc), "state": vruns._state(doc),
                    "cost_usd": cost if isinstance(cost, (int, float)) else None, "score": score, "reached": reached})
    out.sort(key=lambda r: str(r.get("started_at") or ""))
    return out


def build_view(state: Any, *, home: Path, level: str = "all", head: str | None = None,
               workflow: str | None = None, task_type: str | None = None, repo: str | None = None,
               now: str | None = None, subtype: str | None = None,
               features: dict[str, str] | None = None, target_rule: str | None = None) -> dict[str, Any]:
    """The `loopmath.view.posterior/1` object for a belief state (spec 03 section 8). Additive keys for the 0.2
    results page: `results` (the target, the rescue and each workflow's chance and cost per accepted result),
    `units` (one agent alone per allowed setting) and `runs` (the recorded runs as points)."""
    levels = group_levels(state.node_summary())
    present = heads_in(levels)
    if head is not None and head not in present:
        raise ViewError(f"no estimates for head {head!r} in fit {state.fit_id}; heads: {', '.join(present) or 'none'}")
    nodes = [n for n in _all_nodes(levels) if head is None or n["head"] == head]
    task, task_from = choose_task(home, task_type, repo, subtype, features)
    target = choose_target(home, task, head, present)
    rule, rule_from, rescue = results_target(home, task, target_rule)
    if target_rule and head is None and f"score:{rule.score.name}" in present:  # `--target` orders the list too
        target = {"head": f"score:{rule.score.name}", "better": rule.score.better, "target": rule.score.target,
                  "from": "argument"}
    configs: dict[str, Configuration] = {}
    if workflow:
        path = Path(workflow).expanduser()
        if workflow.endswith(".toml") or path.is_file():
            if not path.is_file():
                raise ViewError(f"workflow file not found: {workflow}", EXIT_NOT_FOUND)
            config = _load_toml_config(path)
            entries = [_entry(state, task, config, "file")]
        else:
            config = resolve_config(home, workflow)
            if config is None:
                raise ViewError(_config_not_found(workflow, home), EXIT_NOT_FOUND)
            entries = [_entry(state, task, config, "argument")]
        configs[config.id] = config
    else:
        entries = _workflow_list(state, home, task, target, configs)
    score_name = (rule.score.name if rule is not None
                  else next((h.split(":", 1)[1] for h in [(target or {}).get("head") or "", *present] if h.startswith("score:")), None))
    listed = [configs[e["config"]] for e in entries if e["config"] in configs]
    counts = {e["config"]: int(e.get("runs") or 0) for e in entries}
    meta = _read_meta(home, state.fit_id)
    workflows = [enrich_workflow(e, nodes) for e in entries]
    data = {
        "schema": SCHEMA,
        "generated_at": now or _now(),
        "fit": {"id": state.fit_id, "at": state.created_at, "n_runs": _n_runs(meta)},
        "task": {"type": task.type, "repo": task.repo, "subtype": task.subtype, "features": dict(task.features),
                 "from": task_from},
        "head": head,
        "target": target,
        "heads": {h: head_info(h, meta, state) for h in (present if head is None else [head])},
        "levels": filter_levels(levels, level=level, head=head),
        "workflow": workflows[0] if workflows else None,
        "workflows": workflows,
        "data": data_block(meta),
        "results": results_block(state, task, listed, rule, rule_from, rescue),
        "units": units_block(state, home, task, listed, counts, rule, score_name),
        "runs": run_points(home, task, score_name, rule),
        "score_name": score_name,
    }
    return enrich(data)


# ---------------------------------------------------------------- terminal
def _num(x: float, digits: int = 1) -> str:
    text = f"{x:.{digits}f}"
    text = text.rstrip("0").rstrip(".") if "." in text else text
    return "0" if text == "-0" else text


def _signed(x: float, digits: int = 1) -> str:
    text = _num(x, digits)
    return text if text.startswith("-") or text == "0" else "+" + text


def head_info(head: str, meta: dict[str, Any], state: Any = None) -> dict[str, Any]:
    """How the page shows a head: `multiplier` (cost, tokens, log-scale scores), `pp` (success, gate,
    fraction scores) or `shift` in the score's unit, plus the score's `better` and `unit`."""
    if not head.startswith("score:"):
        return {"kind": "multiplier" if head in ("cost", "tokens") else "pp" if head in HEADS else "shift"}
    name = head[len("score:"):]
    info = (meta.get("scores") or {}).get(name) if isinstance(meta.get("scores"), dict) else None
    if not info and state is not None:
        info = state.score_info(name)
    info = info if isinstance(info, dict) else {}
    scale = info.get("scale") or "linear"
    out = {"kind": {"log": "multiplier", "fraction": "pp"}.get(scale, "shift"), "scale": scale}
    out.update({k: info[k] for k in ("better", "unit") if info.get(k)})
    return out


def head_kind(head: str, heads: dict[str, Any] | None = None) -> str:
    meta = (heads or {}).get(head) or {}
    return meta.get("kind") or ("multiplier" if head in ("cost", "tokens") else "pp" if head in HEADS else "shift")


def _tail(mean: Any, hi: Any) -> str:
    """A mean above its interval's upper end keeps the mean and gets the note, inside the brackets."""
    try:
        return f"; {TAIL_NOTE}" if float(mean) > float(hi) else ""
    except (TypeError, ValueError):
        return ""


def _mean_tail(d: Any) -> str:
    """A mean shown without its bounds gets the note too, in brackets after the number."""
    return f" ({TAIL_NOTE})" if isinstance(d, dict) and _tail(d.get("mean"), d.get("hi")) else ""


def fmt_display(node: dict[str, Any], heads: dict[str, Any] | None = None) -> str:
    """The node's effect for people: `x1.28 (1.05 to 1.57)`, `+5 pp (0 to +10)`, `+60 perf (-40 to +160)`."""
    d = node.get("display") or {}
    m, lo, hi = (float(d.get(k) or 0.0) for k in ("mean", "lo", "hi"))
    head = node.get("head", "")
    kind = head_kind(head, heads)
    if kind == "multiplier":
        return f"x{m:.2f} ({lo:.2f} to {hi:.2f}{_tail(m, hi)})"
    if kind == "pp":
        return f"{_signed(m)} pp ({_signed(lo)} to {_signed(hi)}{_tail(m, hi)})"
    unit = ((heads or {}).get(head) or {}).get("unit")
    digits = 0 if max(abs(m), abs(lo), abs(hi)) >= 100 else 2
    return (f"{_signed(m, digits)}{' ' + unit if unit else ''} ({_signed(lo, digits)} to {_signed(hi, digits)}"
            f"{_tail(m, hi)})")


def _usd(x: float) -> str:
    return f"${x:,.2f}" if abs(x) >= 0.01 or x == 0 else f"${x:.4f}"


def _tokens(x: float) -> str:
    if x >= 1e6:
        return f"{_num(x / 1e6)}M tokens"
    if x >= 1e3:
        return f"{_num(x / 1e3, 0)}k tokens"
    return f"{int(round(x))} tokens"


def _pct(v: float) -> str:
    return f"{_num(v * 100, 0)}%"


def _iv(d: dict[str, Any] | None, fmt) -> str:
    if not d or d.get("mean") is None:
        return "n/a"
    return f"{fmt(d['mean'])} ({fmt(d['lo'])} to {fmt(d['hi'])}{_tail(d['mean'], d.get('hi'))})"


def _fit_line(data: dict[str, Any]) -> str:
    fit = data.get("fit") or {}
    n = fit.get("n_runs")
    if isinstance(n, dict):
        runs = f"{sum(int(v) for v in n.values()):,} runs ({int(n.get('user', 0)):,} yours)"
    elif n is not None:
        runs = f"{int(n):,} runs"
    else:
        runs = "run count not recorded"
    return f"Current estimates (the posterior) from fit {fit.get('id')} at {fit.get('at')}, {runs}. Ranges are 80%."


def _workflow_lines(data: dict[str, Any]) -> list[str]:
    w = data.get("workflow")
    if not w:
        return []
    task = data.get("task") or {}
    about = [x for x in [task.get("subtype")] + [f"{k}={v}" for k, v in (task.get("features") or {}).items()] if x]
    lines = [f"Graph: {w.get('label')}" + (f" for a {task['type']} task in {task['repo']}" if task else "")
             + (f" ({'; '.join(about)})" if task and about else "")]
    shares = w.get("shares") or {}
    for pid, pred in (w.get("per_piece") or {}).items():
        cost, per_round = pred.get("cost") or {}, pred.get("cost_per_round") or {}  # Run total, one execution
        text = f"  {pid}: " + (f"{_iv(per_round.get('usd'), _usd)} per round, " if per_round.get("usd") else "")
        lines.append(text + f"{_iv(cost.get('usd'), _usd)} per run, {_tokens(_mean(cost.get('tokens')))}"
                     f"{_mean_tail(cost.get('tokens'))} per run, expected rounds {_num(_mean(pred.get('rounds'), 1.0), 2)}"
                     f"{_mean_tail(pred.get('rounds'))}, {_pct(shares.get(pid, 0.0))} of cost")
    for gate in w.get("gates") or []:
        text = f"  gate {gate.get('id')}" + (f" after {gate['after']}" if gate.get("after") else "")
        if gate.get("pass"):
            text += f": pass {_iv(gate['pass'], _pct)} per round"
        if gate.get("rounds"):
            text += f", expected rounds {_num(_mean(gate['rounds'], 1.0), 2)}{_mean_tail(gate['rounds'])}"
        lines.append(text)
    for loop in w.get("loops") or []:
        text = f"  repair loop, {loop_text(loop)}"
        if loop.get("rounds"):
            text += f": expected rounds {_num(_mean(loop['rounds'], 1.0), 2)}{_mean_tail(loop['rounds'])}"
        pieces = ", ".join(loop.get("pieces") or [])
        lines.append(text + f"; the pieces it reruns ({pieces}) are {_pct(loop['share'])} of cost")
    return lines


def key_text(n: dict[str, Any]) -> str:
    """A node's key as the page shows it: `piece 1 of solo` for a position, `s` under `feature:size`,
    and the prior's `unknown` task type as `untyped (no task type recorded)`."""
    level, key = str(n.get("level", "")), str(n.get("key", ""))
    if level == "type" and key == "unknown":
        return UNTYPED
    shape, _, index = key.rpartition("#")
    if level == "position" and shape and index.isdigit():
        return f"piece {int(index) + 1} of {shape}"
    prefix = level[len("feature:"):] + "=" if level.startswith("feature:") else ""
    return key[len(prefix):] if prefix and key.startswith(prefix) else key


def _support_text(n: dict[str, Any]) -> str:
    """Runs behind a node, with the page's notes: no runs here, or shared data only."""
    support = int(n.get("support") or 0)
    if support <= 0:
        ref = parent_ref(n.get("parent"))
        if ref is None:
            return "no runs here: the prior for this level, widened"
        level, key = ref
        key = key_text({"level": level or "", "key": key})
        return f"no runs here: the parent's estimate ({_LEVEL_WORDS.get(level, level) + ' ' if level else ''}{key}), widened"
    shared = "" if (n.get("source_mix") or {}).get("user") else ", shared data"
    return f"{support} run{'' if support == 1 else 's'}{shared}"


def summary_lines(data: dict[str, Any], limit: int = TERMINAL_LINES) -> list[str]:
    """At most `limit` plain lines: the fit, the workflow graph when there is one, then the levels."""
    heads = data.get("heads") or {}
    lines = [_fit_line(data)] + _workflow_lines(data)
    levels = data.get("levels") or {}
    many = len(heads_in(levels)) > 1
    sections: list[tuple[str, list[str]]] = []
    for name, nodes in levels.items():
        body = []
        for n in nodes:
            tag = f"[{n['head']}] " if many else ""
            body.append(f"  {tag}{_LEVEL_WORDS.get(n['level'], n['level'])} {key_text(n)}: {fmt_display(n, heads)}, "
                        f"{_support_text(n)}")
        sections.append((name, body or ["  no estimates at this level yet"]))
    room = limit - len(lines) - 1
    total = sum(len(body) for _, body in sections)
    hidden = 0
    for name, body in sections:
        if room < 2:
            hidden += len(body)
            continue
        take = min(len(body), room - 1)
        lines.append(name)
        lines.extend(body[:take])
        hidden += len(body) - take
        room -= 1 + take
    if hidden:
        lines.append(f"{hidden} of {total} rows not shown: use --level, --head, --json or --html")
    return lines[:limit]


# ---------------------------------------------------------------- page
def render(data: dict[str, Any]) -> str:
    """The results page (D119 Z3, direction A): the questions in one scroll, drawn by `results.js` on the shared
    `viz.js`, with the estimates by level, one workflow in full and the data behind the fit drawn by
    `estimates.js` inside those questions. Self-contained: the page renders from the embedded object alone."""
    from . import common

    obj = enrich(data)
    fit = obj.get("fit") or {}
    body = ('<noscript><p>This page draws with JavaScript. The same numbers are in the data block and in '
            '<code>loopmath posterior --json</code>.</p></noscript>')
    return common.page(title=f"loopmath results, fit {fit.get('id', '')}", data=obj, body=body,
                       scripts=[common.asset(n) for n in ("viz.js", "estimates.js", "results.js")],
                       styles=[common.asset(n) for n in ("viz.css", "estimates.css", "results.css")])


# ---------------------------------------------------------------- handler
class _OtherDesign(Exception):
    """fits/latest is from another design version; the message says so."""


def _load_state(home: Path, fit_id: str | None = None) -> Any:
    if fit_id:  # `--fit ID`: a kept fit instead of fits/latest
        from ..belief.fit import load_fit

        return load_fit(home, fit_id)
    from ..belief.state import latest_problem, load_latest

    why = latest_problem(home)
    if why is not None:
        raise _OtherDesign(why)
    return load_latest(home)


def command(args: argparse.Namespace) -> int:
    import sys

    from ..belief.fit import UnknownFit

    home = store_home(getattr(args, "home", None))
    fit_id = getattr(args, "fit", None)
    try:
        state = _load_state(home, fit_id) if fit_id else _load_state(home)
    except UnknownFit as exc:
        return fail(str(exc), EXIT_NOT_FOUND)
    except _OtherDesign as exc:  # one line that says so, not "no fit yet"
        return fail(str(exc), EXIT_NO_FIT)
    if state is None:
        return fail(f"no fit yet under {home}: run `loopmath fit` (or `loopmath onboard`) first", EXIT_NO_FIT)
    try:
        features, unknown = task_features(getattr(args, "feature", None) or [], getattr(state, "features", None),
                                          getattr(args, "horizon", None))
        if unknown:
            print(f"note: loopmath does not model the feature {', '.join(unknown)} (see loopmath task-types); "
                  "it is shown but does not change the estimates", file=sys.stderr)
        data = build_view(state, home=home, level=args.level or "all", head=args.head, workflow=args.workflow,
                          task_type=args.task_type, repo=args.repo, subtype=getattr(args, "subtype", None),
                          features=features, target_rule=getattr(args, "target", None))
    except ViewError as err:
        return fail(str(err), err.code)
    note = subtype_note(state, data["task"])
    if note:
        print(note, file=sys.stderr)
    target = html_target(getattr(args, "html", None), "posterior", getattr(args, "home", None))
    if target is not None:  # with --json the path goes to stderr, so stdout holds only the JSON object
        write_page(target, render(data), json_mode=bool(args.json))
    if args.json:
        emit_json(SCHEMA, data)
    elif target is None:
        print("\n".join(summary_lines(data)))
    return EXIT_OK
