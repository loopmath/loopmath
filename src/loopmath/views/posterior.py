"""The posterior view and the `loopmath posterior` handler (spec 06 section 3, spec 03 section 8).

`build_view` turns the latest fit into one `loopmath.view.posterior/1` object;
`--json` prints it and `--html` embeds it in a self-contained page that renders
from that object and nothing else. `enrich` adds the derived numbers (each
piece's share of the run's cost, the repair loops, the estimates behind each
piece's setting) so the terminal, the JSON and the page show the same values.

Owner: lane 13.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ..output import EXIT_NO_FIT, EXIT_NOT_FOUND, EXIT_OK, EXIT_USER, emit_json, fail, home as store_home
from ..types import TASK_TYPE_IDS, Configuration, Task
from .common import TAIL_NOTE, embed_json, extract_data, html_target, write_page  # noqa: F401 (extract_data: the page's inverse)

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
}
# Lane 5's interaction levels (belief.forest), keys joined with `|`, and how people read level names.
_INTERACTIONS = {"family_effort": ["family", "effort"], "role_family": ["role", "family"]}
_LEVEL_WORDS = {"family_effort": "family x effort", "role_family": "role x family", "model": "version"}
UNTYPED = "untyped (no task type recorded)"  # the prior's `unknown` task type (E0 runs), view text only (D92)
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
    cost (decision D60), so it is never multiplied by the expected rounds here.
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

    Workflows are acyclic (decision D3), so loops come from control, never from back edges.
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
    the piece's position in its workflow (`<workflow>#<index>`, lane 5's key) and the interactions."""
    setting = node.get("setting") or {}
    model = setting.get("model")
    family, provider = _model_ancestors(model, nodes)
    values = {"version": model, "model": model, "family": family, "provider": provider,
              "effort": setting.get("effort"), "role": node.get("role"), "harness": setting.get("harness"),
              "position": position}
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
        wid = _workflow_id(w)
        effects = {n["id"]: _refs(piece_effects(n, nodes, f"{wid}#{i}")) for i, n in enumerate(_pieces_of(graph))}
        topo = _refs(n for n in nodes if n["level"] == "topology" and n["key"] == wid)
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
    """The types form of `control`, also from OCP v0.3 (D29, D30, D42).

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
    """A configuration id seen in the store: recommendations first (newest first), then run documents (D13)."""
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
    """The usual configuration from config.toml, read by lane 7's `Config.usual` (decision D5): `usual.<type>."<repo>"`,
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


def task_features(items: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """`--feature K=V` items as `recommend` reads them, and the keys the model does not know."""
    from ..taskmodel import normalize_features

    raw: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ViewError(f"--feature takes K=V; got {item!r}")
        k, v = item.split("=", 1)
        raw[k.strip()] = v.strip()
    feats = normalize_features(raw)
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


def _workflow_list(state: Any, home: Path, task: Task, target: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every configuration recorded for the task's (type, repo) and the usual one (D20, I11), grouped by graph."""
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
    return order_workflows(out, target)


def build_view(state: Any, *, home: Path, level: str = "all", head: str | None = None,
               workflow: str | None = None, task_type: str | None = None, repo: str | None = None,
               now: str | None = None, subtype: str | None = None,
               features: dict[str, str] | None = None) -> dict[str, Any]:
    """The `loopmath.view.posterior/1` object for a belief state (spec 03 section 8, D20)."""
    levels = group_levels(state.node_summary())
    present = heads_in(levels)
    if head is not None and head not in present:
        raise ViewError(f"no estimates for head {head!r} in fit {state.fit_id}; heads: {', '.join(present) or 'none'}")
    nodes = [n for n in _all_nodes(levels) if head is None or n["head"] == head]
    task, task_from = choose_task(home, task_type, repo, subtype, features)
    target = choose_target(home, task, head, present)
    if workflow:
        path = Path(workflow).expanduser()
        if workflow.endswith(".toml") or path.is_file():
            if not path.is_file():
                raise ViewError(f"workflow file not found: {workflow}", EXIT_NOT_FOUND)
            entries = [_entry(state, task, _load_toml_config(path), "file")]
        else:
            config = resolve_config(home, workflow)
            if config is None:
                raise ViewError(_config_not_found(workflow, home), EXIT_NOT_FOUND)
            entries = [_entry(state, task, config, "argument")]
    else:
        entries = _workflow_list(state, home, task, target)
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
    """D107: a mean above its interval's upper end keeps the mean and gets the note, inside the brackets."""
    try:
        return f"; {TAIL_NOTE}" if float(mean) > float(hi) else ""
    except (TypeError, ValueError):
        return ""


def _mean_tail(d: Any) -> str:
    """D107 note 2: a mean shown without its bounds gets the note too, in brackets after the number."""
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
        cost, per_round = pred.get("cost") or {}, pred.get("cost_per_round") or {}  # D60: run total, one execution
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
    and the prior's `unknown` task type as `untyped (no task type recorded)` (D92)."""
    level, key = str(n.get("level", "")), str(n.get("key", ""))
    if level == "type" and key == "unknown":
        return UNTYPED
    shape, _, index = key.rpartition("#")
    if level == "position" and shape and index.isdigit():
        return f"piece {int(index) + 1} of {shape}"
    prefix = level[len("feature:"):] + "=" if level.startswith("feature:") else ""
    return key[len(prefix):] if prefix and key.startswith(prefix) else key


def _support_text(n: dict[str, Any]) -> str:
    """Runs behind a node, with the page's D21 notes: no runs here, or shared data only."""
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
    """The self-contained posterior page for one view object."""
    obj = enrich(data)
    fit = obj.get("fit") or {}
    title = html.escape(f"loopmath: current estimates, fit {fit.get('id', '')}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<div class="app">
<header class="top">
  <h1>Current estimates</h1>
  <p class="lede" id="lede"></p>
  <p class="task" id="task"></p>
</header>
<nav class="tabs" role="tablist">
  <button type="button" role="tab" data-tab="levels">By level</button>
  <button type="button" role="tab" data-tab="graph">By workflow graph</button>
  <button type="button" role="tab" data-tab="data">Data behind the fit</button>
</nav>
<main>
  <section class="tab" id="tab-levels" data-panel="levels"></section>
  <section class="tab" id="tab-graph" data-panel="graph" hidden></section>
  <section class="tab" id="tab-data" data-panel="data" hidden></section>
</main>
<noscript><p>This page draws with JavaScript. The same numbers are in the data block below and in <code>loopmath posterior --json</code>.</p></noscript>
</div>
<div class="tip" id="tip" hidden></div>
<script type="application/json" id="data">{embed_json(obj)}</script>
<script>{PAGE_JS}</script>
</body>
</html>
"""


PAGE_CSS = r"""
:root{--surface:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--rule:#e4e3df;
--band:#9ec5f4;--dot:#1c5cab;--neutral:#a3a29d;--good:#2a78d6;--bad:#e34948;--mid:#f0efec;--chip:#f3f2ef;
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);font-size:14px;line-height:1.45}
.app{max-width:1180px;margin:0 auto;padding:20px 20px 60px}
h1{font-size:22px;margin:0 0 6px}
h2{font-size:17px;margin:26px 0 8px}
h3{font-size:14px;margin:16px 0 6px;color:var(--ink2)}
.lede{margin:0 0 6px;max-width:860px}
.task{margin:0;color:var(--ink2)}
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--rule);margin:16px 0 0;flex-wrap:wrap}
.tabs button,.heads button{font:inherit;border:1px solid var(--rule);background:var(--panel);color:var(--ink2);
padding:6px 12px;border-radius:6px 6px 0 0;cursor:pointer;margin-bottom:-1px}
.tabs button[aria-selected=true]{color:var(--ink);border-bottom-color:var(--surface);background:var(--surface);font-weight:600}
.heads{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:14px 0 4px}
.heads button{border-radius:14px;margin:0;padding:4px 11px}
.heads button[aria-pressed=true]{background:var(--ink);color:#fff;border-color:var(--ink)}
.note{color:var(--ink2);margin:4px 0 10px;max-width:860px}
.empty{color:var(--muted);font-style:italic;margin:6px 0}
table{border-collapse:collapse;width:100%;background:var(--panel)}
table.nodes{table-layout:fixed;min-width:760px}
col.c-node{width:34%}col.c-eff{width:21%}col.c-bar{width:200px}col.c-sup{width:64px}
table.nodes td{overflow:hidden;text-overflow:ellipsis}
table.heat{width:auto;min-width:320px}
table.compact{width:auto;min-width:420px}table.compact th,table.compact td{padding-right:24px}
.heat td{min-width:84px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--rule);vertical-align:middle}
th{font-weight:600;color:var(--ink2);font-size:12px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.eff{font-variant-numeric:tabular-nums;white-space:nowrap}
.scroll{overflow-x:auto}
.node{white-space:nowrap}
.lvl{color:var(--muted);font-size:12px}
.tag{display:inline-block;font-size:11px;padding:0 6px;border-radius:9px;background:var(--chip);color:var(--ink2);margin-left:6px}
.tag.head{background:#e8f0fb;color:#184f95}
.why{display:block;color:var(--ink2);font-size:12px;white-space:normal}
.sup{font-variant-numeric:tabular-nums}
.sup.low{color:var(--muted)}
.mix{color:var(--ink2);font-size:12px}
svg.bar{display:block}
.heat td{text-align:center;font-variant-numeric:tabular-nums;border:2px solid var(--panel);font-size:12px}
.heat th{text-align:center}
.heat th.rowh{text-align:left}
.legend{color:var(--ink2);font-size:12px;margin:4px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin:0 4px}
.graphwrap{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--panel);margin:10px 0}
.graphwrap svg text{font-family:inherit}.graphwrap svg.graph{display:block;max-width:100%;height:auto}
select{font:inherit;padding:4px 6px;max-width:100%}
.kv td:first-child{color:var(--ink2);width:220px}
.tip{position:fixed;z-index:10;max-width:360px;background:#0b0b0b;color:#fff;padding:7px 9px;border-radius:6px;
font-size:12px;white-space:pre-line;pointer-events:none}
code{background:var(--chip);padding:0 4px;border-radius:3px}
details.more{margin:4px 0 8px}details.more summary{cursor:pointer;color:var(--ink2);font-size:12px;padding:4px 0}
details.wfgroup{margin:0 0 4px;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
details.wfgroup summary{cursor:pointer;padding:5px 10px}details.wfgroup table{margin:0 0 4px}
table.wflist tr.on td{background:#e8f0fb}table.wflist a{color:var(--ink);text-decoration:none}table.wflist a:hover{text-decoration:underline}
@media (max-width:700px){.app{padding:12px}td.bar,th.bar,col.bar{display:none}table.nodes{min-width:0}}
"""

PAGE_JS = r"""
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('data').textContent);
  var HEADS = D.heads || {};
  var TITLES = {model: 'Model', effort: 'Effort', role: 'Role', topology: 'Workflow shape (topology)',
    type: 'Task type', repo: 'Repo', feature: 'Task features', harness: 'Harness', source: 'Data source',
    gate: 'Gate rule', task: 'Single task', org: 'Organization'};
  var INTERACTIONS = {family_effort: ['family', 'effort'], role_family: ['role', 'family']};
  var EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];  // fit_pricing.EFFORT_ORDER
  var HEAD_ORDER = ['cost', 'tokens', 'success', 'gate'];
  var LEVEL_WORDS = {family_effort: 'family x effort', role_family: 'role x family', model: 'version'};
  var UNTYPED = 'untyped (no task type recorded)';
  var TAIL_NOTE = 'the average is pulled up by rare very large outcomes';  // D107: same words in every view
  var S = {head: null, wf: 0, open: {}};

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
    });
  }
  function fin(x) { return typeof x === 'number' && isFinite(x); }
  function num(x, d) {
    if (!fin(x)) return 'n/a';
    var t = Number(x).toFixed(d == null ? 1 : d);
    if (t.indexOf('.') >= 0) t = t.replace(/0+$/, '').replace(/\.$/, '');
    return t === '-0' ? '0' : t;
  }
  function signed(x, d) { var t = num(x, d); return (t.charAt(0) === '-' || t === '0' || t === 'n/a') ? t : '+' + t; }
  function usd(x) {
    if (!fin(x)) return 'n/a';
    return Math.abs(x) >= 0.01 || x === 0 ? '$' + x.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})
      : '$' + num(x, 4);
  }
  function tokn(x) {
    if (!fin(x)) return 'n/a';
    if (x >= 1e6) return num(x / 1e6) + 'M';
    if (x >= 1e3) return num(x / 1e3, 0) + 'k';
    return String(Math.round(x));
  }
  function tok(x) { return fin(x) ? tokn(x) + ' tokens' : 'n/a'; }
  function rounds(x) { var t = num(x, 2); return t + (t === '1' ? ' round' : ' rounds'); }
  function pct(x) { return fin(x) ? num(x * 100, 0) + '%' : 'n/a'; }
  // D107: a mean above its interval's upper end keeps the mean and gets the note: inside the brackets in running
  // text and tooltips (`bare` leaves it out), on its own line under the value in table cells (`ivCell`, `tailBlock`).
  function above(d) { return !!d && fin(d.mean) && fin(d.hi) && d.mean > d.hi; }
  function tail(d) { return above(d) ? '; ' + TAIL_NOTE : ''; }
  function tailBlock(d) { return above(d) ? '<span class="why">' + esc(TAIL_NOTE) + '</span>' : ''; }
  function iv(d, f, bare) { return d && fin(d.mean) ? f(d.mean) + ' (' + f(d.lo) + ' to ' + f(d.hi) + (bare ? '' : tail(d)) + ')' : 'n/a'; }
  function ivCell(d, f) { return esc(iv(d, f, true)) + tailBlock(d); }
  function meanTail(d) { return above(d) ? ' (' + TAIL_NOTE + ')' : ''; }  // D107 note 2: a mean shown without its bounds
  function kindOf(head) {
    var m = HEADS[head] || {};
    return m.kind || (head === 'cost' || head === 'tokens' ? 'multiplier' : (head === 'success' || head === 'gate') ? 'pp' : 'shift');
  }
  function headLabel(head) {
    if (head === 'cost') return 'Cost';
    if (head === 'success') return 'Success';
    if (head === 'gate') return 'Gate pass';
    if (head === 'tokens') return 'Tokens';
    if (head.indexOf('score:') === 0) return 'Score: ' + head.slice(6);
    return head;
  }
  function headNote(head) {
    var k = kindOf(head), m = HEADS[head] || {};
    var arrow = m.better === 'lower' ? ' Lower is better (↓).' : m.better === 'higher' ? ' Higher is better (↑).' : '';
    if (head.indexOf('score:') === 0 && k === 'multiplier') return 'How much a node multiplies the score ' + head.slice(6) + ' on one run: x1.00 is no change.' + arrow;
    if (head.indexOf('score:') === 0 && k === 'pp') return 'How a node shifts the score ' + head.slice(6) + ' on one run, in percentage points.' + arrow;
    if (head === 'tokens') return 'How much a node multiplies tokens per round, relative to its parent: x1.00 is no change, below x1 is fewer.';
    if (k === 'multiplier') return 'How much a node multiplies dollars per round, relative to its parent: x1.00 is no change, below x1 is cheaper.';
    if (head === 'gate') return 'How a node shifts the chance that a gate passes on a round, in percentage points at the group\'s base rate.';
    if (k === 'pp') return 'How a node shifts the chance of an accepted result, in percentage points at the group\'s base rate.';
    return 'How a node shifts the score ' + head.slice(6) + (m.unit ? ' (in ' + m.unit + ')' : '') + ' on one run.' + arrow;
  }
  function fmtDisplay(n, bare) {
    var d = n.display || {}, k = kindOf(n.head), m = +d.mean || 0, lo = +d.lo || 0, hi = +d.hi || 0;
    var t = bare ? '' : tail({mean: m, hi: hi});
    if (k === 'multiplier') return 'x' + m.toFixed(2) + ' (' + lo.toFixed(2) + ' to ' + hi.toFixed(2) + t + ')';
    if (k === 'pp') return signed(m) + ' pp (' + signed(lo) + ' to ' + signed(hi) + t + ')';
    var unit = (HEADS[n.head] || {}).unit, dg = Math.max(Math.abs(m), Math.abs(lo), Math.abs(hi)) >= 100 ? 0 : 2;
    return signed(m, dg) + (unit ? ' ' + unit : '') + ' (' + signed(lo, dg) + ' to ' + signed(hi, dg) + t + ')';
  }
  function neutral(head) { return kindOf(head) === 'multiplier' ? 1 : 0; }
  function betterOf(head) {
    if (head === 'cost' || head === 'tokens') return 'lower';
    if (head === 'success' || head === 'gate') return 'higher';
    var b = (HEADS[head] || {}).better;
    return b === 'lower' || b === 'higher' ? b : null;
  }
  function goodness(n) {
    var k = kindOf(n.head), m = +(n.display || {}).mean || 0, b = betterOf(n.head);
    var up = k === 'multiplier' ? (m > 0 ? Math.log(m) : 0) : m;
    return b === 'lower' ? -up : b === 'higher' ? up : 0;
  }
  function hasUser(n) { return ((n.source_mix || {}).user || 0) > 0; }
  function mixText(n) {
    var mix = n.source_mix || {}, keys = Object.keys(mix);
    keys.sort(function (a, b) { return (mix[b] || 0) - (mix[a] || 0); });
    return keys.map(function (k) { return k + ' ' + mix[k]; }).join(', ');
  }
  function parts(level) {
    if (INTERACTIONS[level]) return INTERACTIONS[level].slice();
    var seps = [' x ', '_x_', ' × ', '×'];
    for (var i = 0; i < seps.length; i++) if (level.indexOf(seps[i]) >= 0) return level.split(seps[i]).map(function (s) { return s.trim(); });
    return null;
  }
  function splitKey(key, n) {
    var seps = ['|', ' x ', '×', ',', ':', '/'];
    for (var i = 0; i < seps.length; i++) {
      var p = String(key).split(seps[i]).map(function (s) { return s.trim(); });
      if (p.length === n) return p;
    }
    return [String(key)];
  }
  function tipText(n) {
    var lines = [levelWord(n.level) + ' ' + keyText(n) + ' (' + headLabel(n.head) + ')', fmtDisplay(n),
      'model scale: ' + iv(n.effect, function (v) { return num(v, 3); }),
      (n.support || 0) + ' runs' + (mixText(n) ? ': ' + mixText(n) : '')];
    if (n.parent) lines.push('parent: ' + parentText(n.parent));
    return lines.join('\n');
  }

  // ---------------------------------------------------------------- interval bars
  function axis(nodes, head) {
    var log = kindOf(head) === 'multiplier', z = neutral(head), lo = z, hi = z;
    nodes.forEach(function (n) {
      var d = n.display || {};
      [d.lo, d.hi, d.mean].forEach(function (v) {
        if (!fin(v) || (log && v <= 0)) return;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      });
    });
    var f = log ? Math.log : function (v) { return v; };
    var a = f(lo), b = f(hi);
    if (a === b) { a -= 1; b += 1; }
    var pad = (b - a) * 0.06;
    return {f: f, a: a - pad, b: b + pad, zero: f(z), log: log};
  }
  function bar(n, ax) {
    var W = 180, H = 18, d = n.display || {};
    function x(v) { var t = ax.f(ax.log ? Math.max(v, 1e-9) : v); return Math.max(2, Math.min(W - 2, (t - ax.a) / (ax.b - ax.a) * W)); }
    var s = '<svg class="bar" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + esc(fmtDisplay(n)) + '">';
    var z = (ax.zero - ax.a) / (ax.b - ax.a) * W;
    s += '<line x1="' + z.toFixed(1) + '" x2="' + z.toFixed(1) + '" y1="1" y2="' + (H - 1) + '" stroke="var(--neutral)" stroke-width="1" stroke-dasharray="2 2"/>';
    if (fin(d.lo) && fin(d.hi)) {
      var x0 = x(d.lo), x1 = x(d.hi);
      s += '<rect x="' + Math.min(x0, x1).toFixed(1) + '" y="6" width="' + Math.max(2, Math.abs(x1 - x0)).toFixed(1) + '" height="6" rx="3" fill="var(--band)"/>';
    }
    if (fin(d.mean)) s += '<circle cx="' + x(d.mean).toFixed(1) + '" cy="9" r="4" fill="var(--dot)" stroke="#fff" stroke-width="2"/>';
    return s + '</svg>';
  }

  // ---------------------------------------------------------------- levels tab
  function parentRef(parent) {
    if (parent == null) return null;
    var t = String(parent), i = t.indexOf(':');
    if (i < 0 || !/^[a-z_]+$/.test(t.slice(0, i))) return {level: null, key: t};
    var level = t.slice(0, i), key = t.slice(i + 1);
    if ((level === 'repo' || level === 'subtype') && key.indexOf('/') >= 0) key = key.slice(key.indexOf('/') + 1);
    return {level: level, key: key};
  }
  function parentText(parent) {
    var r = parentRef(parent);
    return r ? (r.level ? levelWord(r.level) + ' ' : '') + keyText({level: r.level || '', key: r.key}) : '';
  }
  function levelWord(level) { return LEVEL_WORDS[level] || level; }
  function keyText(n) {
    if (n.level === 'type' && String(n.key) === 'unknown') return UNTYPED;
    var m = n.level === 'position' ? /^(.*)#(\d+)$/.exec(String(n.key)) : null;
    if (m) return 'piece ' + (+m[2] + 1) + ' of ' + m[1];
    var f = n.level.indexOf('feature:') === 0 ? n.level.slice(8) + '=' : null;
    return f && String(n.key).indexOf(f) === 0 ? String(n.key).slice(f.length) : String(n.key);
  }
  function depthMap(nodes) {
    var byKey = {}, byFull = {}, depth = {};
    nodes.forEach(function (n) {
      if (parts(n.level)) return;
      byKey[n.head + '\u0000' + n.key] = n;
      byFull[n.head + '\u0000' + n.level + '\u0000' + n.key] = n;
    });
    function dep(n, guard) {
      var id = n.head + '\u0000' + n.level + '\u0000' + n.key;
      if (depth[id] != null) return depth[id];
      var r = parentRef(n.parent);
      var p = !r ? null : r.level ? byFull[n.head + '\u0000' + r.level + '\u0000' + r.key] : byKey[n.head + '\u0000' + r.key];
      depth[id] = (p && p !== n && guard < 8) ? dep(p, guard + 1) + 1 : 0;
      return depth[id];
    }
    return function (n) { return dep(n, 0); };
  }
  function pageAxes(nodes) {
    var by = {}, axes = {};
    nodes.forEach(function (n) { if (!parts(n.level)) (by[n.head] = by[n.head] || []).push(n); });
    Object.keys(by).forEach(function (h) { axes[h] = axis(by[h], h); });
    return axes;
  }
  function allNodes() {
    var out = [];
    Object.keys(D.levels || {}).forEach(function (k) { out = out.concat(D.levels[k] || []); });
    return out;
  }
  var SHOWN = 30;
  function nodeRows(nodes, opts) {
    opts = opts || {};
    if (opts.limit && nodes.length > opts.limit + 5) {
      var rest = {showHead: opts.showHead, axes: opts.axes || pageAxes(nodes)};
      return nodeRows(nodes.slice(0, opts.limit), rest) + '<details class="more"><summary>Show ' + (nodes.length - opts.limit) +
        ' more (fewer runs)</summary>' + nodeRows(nodes.slice(opts.limit), rest) + '</details>';
    }
    var heads = [];
    nodes.forEach(function (n) { if (heads.indexOf(n.head) < 0) heads.push(n.head); });
    var axes = opts.axes || pageAxes(nodes);
    var depth = depthMap(nodes), showHead = heads.length > 1 || opts.showHead;
    var s = '<div class="scroll"><table class="nodes"><colgroup><col class="c-node"><col class="c-eff"><col class="c-bar bar"><col class="c-sup">' +
      '<col class="c-mix"></colgroup><thead><tr><th>Estimate</th><th>Effect (80% range)</th><th class="bar">Range</th>' +
      '<th class="num">Runs</th><th>Sources</th></tr></thead><tbody>';
    nodes.forEach(function (n) {
      var pad = depth(n) * 18, notes = '';
      if (showHead) notes += '<span class="tag head">' + esc(headLabel(n.head)) + '</span>';
      if (!hasUser(n)) notes += '<span class="tag">from shared data</span>';
      var why = '';
      if (!(n.support > 0)) why = n.parent != null ? 'No runs here: the parent\'s estimate (' + esc(parentText(n.parent)) + '), widened.'
        : 'No runs here: the prior for this level, widened.';
      s += '<tr data-tip="' + esc(tipText(n)) + '"><td class="node" style="padding-left:' + (8 + pad) + 'px">' +
        '<span class="lvl">' + esc(levelWord(n.level)) + '</span> ' + esc(keyText(n)) + notes + (why ? '<span class="why">' + why + '</span>' : '') + '</td>' +
        '<td class="eff">' + esc(fmtDisplay(n, true)) + tailBlock({mean: +(n.display || {}).mean, hi: +(n.display || {}).hi}) + '</td><td class="bar">' + bar(n, axes[n.head]) + '</td>' +
        '<td class="num"><span class="sup' + ((n.support || 0) < 5 ? ' low' : '') + '">' + esc(n.support || 0) + '</span></td>' +
        '<td class="mix">' + esc(mixText(n)) + '</td></tr>';
    });
    return s + '</tbody></table></div>';
  }
  function heatScale(n, maxg) {
    var k = kindOf(n.head), g = goodness(n);
    if (k === 'multiplier') return g / Math.LN2;
    if (k === 'pp') return g / 15;
    return maxg > 0 ? g / maxg : 0;
  }
  function heatColor(t) {
    var mid = [240, 239, 236], pole = t >= 0 ? [42, 120, 214] : [227, 73, 72], a = Math.min(1, Math.abs(t));
    var c = mid.map(function (m, i) { return Math.round(m + (pole[i] - m) * a); });
    return {bg: 'rgb(' + c.join(',') + ')', fg: a > 0.55 ? '#fff' : 'var(--ink)'};
  }
  function heatNote(head) {
    var k = kindOf(head);
    if (!betterOf(head)) return ' (this score records no better direction, so cells stay uncolored)';
    return k === 'multiplier' ? ' (full color at x0.5 or x2)' : k === 'pp' ? ' (full color at 15 points)' : ' (scaled to the largest shift)';
  }
  function effortOrder(list) {  // low to max; efforts the list does not know keep their order at the end
    var rank = function (x) { var i = EFFORTS.indexOf(x); return i < 0 ? EFFORTS.length : i; };
    return list.map(function (x, i) { return [x, i]; })
      .sort(function (a, b) { return rank(a[0]) - rank(b[0]) || a[1] - b[1]; }).map(function (a) { return a[0]; });
  }
  function heatTable(level, nodes) {
    var p = parts(level), rows = [], cols = [], cell = {}, maxg = 0;
    nodes.forEach(function (n) {
      var k = splitKey(n.key, p.length), r = k[0], c = k.slice(1).join(' | ');
      if (rows.indexOf(r) < 0) rows.push(r);
      if (cols.indexOf(c) < 0) cols.push(c);
      cell[r + '\u0000' + c] = n;
      maxg = Math.max(maxg, Math.abs(goodness(n)));
    });
    if (p[0] === 'effort') rows = effortOrder(rows);
    if (p.length === 2 && p[1] === 'effort') cols = effortOrder(cols);
    var s = '<h3>' + esc(levelWord(level)) + ' <span class="tag head">' + esc(headLabel(nodes[0].head)) + '</span></h3>' +
      '<p class="legend">Each cell: the effect of this ' + esc(p[0]) + ' at this ' + esc(p.slice(1).join(' and ')) +
      '. <span class="sw" style="background:rgb(42,120,214)"></span>better <span class="sw" style="background:rgb(240,239,236)"></span>no change ' +
      '<span class="sw" style="background:rgb(227,73,72)"></span>worse' + heatNote(nodes[0].head) + '. Hover a cell for its range and runs.</p>' +
      '<div class="scroll"><table class="heat"><thead><tr><th class="rowh">' + esc(p[0]) + ' \\ ' + esc(p.slice(1).join(' | ')) + '</th>';
    cols.forEach(function (c) { s += '<th>' + esc(c) + '</th>'; });
    s += '</tr></thead><tbody>';
    rows.forEach(function (r) {
      s += '<tr><th class="rowh">' + esc(r) + '</th>';
      cols.forEach(function (c) {
        var n = cell[r + '\u0000' + c];
        if (!n) { s += '<td class="empty">none</td>'; return; }
        var col = heatColor(heatScale(n, maxg)), d = n.display || {};
        var txt = kindOf(n.head) === 'multiplier' ? 'x' + (+d.mean || 0).toFixed(2) : signed(d.mean) + (kindOf(n.head) === 'pp' ? ' pp' : '');
        s += '<td style="background:' + col.bg + ';color:' + col.fg + '" data-tip="' + esc(tipText(n)) + '">' + esc(txt) +
          ((n.support || 0) < 5 ? '*' : '') + '</td>';
      });
      s += '</tr>';
    });
    var few = nodes.some(function (n) { return (n.support || 0) < 5; });
    var tailed = nodes.filter(function (n) { return above(n.display); }).map(keyText);  // cells show the mean only (D107)
    return s + '</tbody></table></div>' + (few ? '<p class="legend">* fewer than 5 runs.</p>' : '') +
      (tailed.length ? '<p class="legend">' + esc(tailed.join(', ')) + ': ' + esc(TAIL_NOTE) + '.</p>' : '');
  }
  function sectionHtml(name, nodes) {
    var title = TITLES[name] || (name.indexOf('score:') === 0 ? 'Score ' + name.slice(6) : name);
    var shown = S.head === 'all' ? nodes : nodes.filter(function (n) { return n.head === S.head; });
    var s = '<section class="level" data-section="' + esc(name) + '"><h2>' + esc(title) + '</h2>';
    if (!shown.length) {
      s += '<p class="empty">' + (nodes.length ? 'No ' + esc(headLabel(S.head).toLowerCase()) + ' estimates at this level. Other heads have ' + nodes.length + '.'
        : 'No estimates at this level yet.') + '</p>';
      return s + '</section>';
    }
    var plain = shown.filter(function (n) { return !parts(n.level); });
    var inter = shown.filter(function (n) { return parts(n.level); });
    if (name === 'feature') {
      var groups = {}, order = [];
      plain.forEach(function (n) {
        var g = n.level.indexOf('feature:') === 0 ? n.level.slice(8) : n.level;
        if (!groups[g]) { groups[g] = []; order.push(g); }
        groups[g].push(n);
      });
      order.forEach(function (g) { s += '<h3>' + esc(g) + '</h3>' + nodeRows(groups[g], {showHead: S.head === 'all', axes: S.axes, limit: SHOWN}); });
    } else if (plain.length) {
      s += nodeRows(plain, {showHead: S.head === 'all', axes: S.axes, limit: SHOWN});
    }
    var byLevel = {}, lv = [];
    inter.forEach(function (n) {
      var k = n.level + '\u0000' + n.head;
      if (!byLevel[k]) { byLevel[k] = []; lv.push(k); }
      byLevel[k].push(n);
    });
    lv.forEach(function (k) { s += heatTable(k.split('\u0000')[0], byLevel[k]); });
    return s + '</section>';
  }
  function headList() {
    var hs = Object.keys(HEADS);
    Object.keys(D.levels || {}).forEach(function (k) {
      (D.levels[k] || []).forEach(function (n) { if (hs.indexOf(n.head) < 0) hs.push(n.head); });
    });
    var rank = function (h) { var i = HEAD_ORDER.indexOf(h); return i < 0 ? HEAD_ORDER.length : i; };
    return hs.map(function (h, i) { return [h, i]; })
      .sort(function (a, b) { return rank(a[0]) - rank(b[0]) || a[1] - b[1]; }).map(function (a) { return a[0]; });
  }
  function levelsHtml(head) {
    if (head) S.head = head;
    var hs = headList();
    var s = '<div class="heads" role="group" aria-label="Head">';
    hs.concat(hs.length > 1 ? ['all'] : []).forEach(function (h) {
      s += '<button type="button" data-head="' + esc(h) + '" aria-pressed="' + (S.head === h) + '">' + esc(h === 'all' ? 'All heads' : headLabel(h)) + '</button>';
    });
    S.axes = pageAxes(allNodes());
    s += '</div><p class="note">' + (S.head === 'all' ? 'Every head, each row tagged with its head. Bars of one head share an axis across the page.'
      : esc(headNote(S.head))) + ' Grey run counts are under 5. The dashed line on each bar is no change.</p>';
    var names = Object.keys(D.levels || {});
    if (!names.length) return s + '<p class="empty">This fit reports no estimates.</p>';
    names.forEach(function (name) { s += sectionHtml(name, D.levels[name] || []); });
    return s;
  }

  // ---------------------------------------------------------------- graph tab
  function layout(g) {
    var nodes = g.nodes || [], ids = nodes.map(function (n) { return n.id; }), out = {}, indeg = {};
    ids.forEach(function (id) { out[id] = []; indeg[id] = 0; });
    (g.edges || []).forEach(function (e) {
      var a = e.from != null ? e.from : e[0], b = e.to != null ? e.to : e[1];
      if (out[a] && out[b]) { out[a].push(b); indeg[b]++; }
    });
    var color = {}, fwd = [], back = [];
    function dfs(u) {
      color[u] = 1;
      out[u].forEach(function (v) {
        if (color[v] === 1) { back.push([u, v]); return; }
        fwd.push([u, v]);
        if (!color[v]) dfs(v);
      });
      color[u] = 2;
    }
    ids.filter(function (id) { return !indeg[id]; }).concat(ids).forEach(function (id) { if (!color[id]) dfs(id); });
    var layer = {}, changed = true, guard = 0;
    ids.forEach(function (id) { layer[id] = 0; });
    while (changed && guard++ < ids.length + 2) {
      changed = false;
      fwd.forEach(function (e) { if (layer[e[1]] < layer[e[0]] + 1) { layer[e[1]] = layer[e[0]] + 1; changed = true; } });
    }
    var cols = [];
    ids.forEach(function (id) { (cols[layer[id]] = cols[layer[id]] || []).push(id); });
    var maxRows = Math.max.apply(null, cols.map(function (c) { return c ? c.length : 0 }).concat([1]));
    var kind = {}, RH = 130, pos = {}, x = 30;
    nodes.forEach(function (n) { kind[n.id] = n.kind || 'piece'; });
    cols.forEach(function (c, i) {
      var cw = (c || []).some(function (id) { return kind[id] === 'piece'; }) ? 262 : 168;
      (c || []).forEach(function (id, j) { pos[id] = {x: x, y: 40 + (j + (maxRows - c.length) / 2) * RH}; });
      x += cw;
    });
    return {pos: pos, fwd: fwd, back: back, width: x + 30, height: 40 + (maxRows - 1) * RH + 86 + 30};
  }
  // I12: a piece of width n is drawn as n worker boxes (up to MAX_WORKERS; above that one box marked xn), each with
  // its setting and its part of the piece's cost. The cost model prices a piece as its width times one worker
  // (belief/compose.py), so one worker's cost and range are the piece's divided by n.
  var MAX_WORKERS = 6;
  function copy(o, extra) { var r = {}; Object.keys(o || {}).forEach(function (k) { r[k] = o[k]; }); Object.keys(extra || {}).forEach(function (k) { r[k] = extra[k]; }); return r; }
  function scaled(d, f) { if (!d) return d; var r = copy(d); ['mean', 'lo', 'hi', 'median'].forEach(function (k) { if (fin(d[k])) r[k] = d[k] * f; }); return r; }
  function perWorker(pr, n) {
    if (!pr) return pr;
    var r = copy(pr);
    ['cost', 'cost_per_round'].forEach(function (k) { if (pr[k]) r[k] = copy(pr[k], {usd: scaled(pr[k].usd, 1 / n), tokens: scaled(pr[k].tokens, 1 / n)}); });
    return r;
  }
  function workers(w) {
    var g = w.graph || {}, wide = {};
    (g.nodes || []).forEach(function (n) { if ((n.kind || 'piece') === 'piece' && n.width > 1 && n.width <= MAX_WORKERS) wide[n.id] = n.width; });
    if (!Object.keys(wide).length) return w;
    function ids(id) { var out = []; for (var k = 1; k <= (wide[id] || 0); k++) out.push(id + '#' + k); return out.length ? out : [id]; }
    function last(id) { var l = ids(id); return l[l.length - 1]; }
    var nodes = [], per = copy(w.per_piece), shares = copy(w.shares);
    (g.nodes || []).forEach(function (n) {
      if (!wide[n.id]) { nodes.push(n); return; }
      var pr = perWorker((w.per_piece || {})[n.id] || n.prediction, wide[n.id]);
      ids(n.id).forEach(function (id, k) {
        nodes.push(copy(n, {id: id, piece: n.id, name: n.id + ' ' + (k + 1) + ' of ' + wide[n.id], worker: k + 1, workers: wide[n.id], width: 1, prediction: pr}));
        per[id] = pr;
        shares[id] = fin((w.shares || {})[n.id]) ? w.shares[n.id] / wide[n.id] : null;
      });
    });
    var edges = [];
    (g.edges || []).forEach(function (e) {
      var a = e.from != null ? e.from : e[0], b = e.to != null ? e.to : e[1];
      ids(a).forEach(function (x) { ids(b).forEach(function (y) { edges.push({from: x, to: y}); }); });
    });
    return copy(w, {graph: copy(g, {nodes: nodes, edges: edges}), per_piece: per, shares: shares,
      gates: (w.gates || []).map(function (gt) { return copy(gt, {after: gt.after && last(gt.after)}); }),
      loops: (w.loops || []).map(function (lp) { return copy(lp, {from: ids(lp.from)[0], to: last(lp.to)}); })});
  }
  function graphSvg(w) {
    w = workers(w);
    var g = w.graph || {}, L = layout(g), byId = {}, shares = w.shares || {}, per = w.per_piece || {};
    (g.nodes || []).forEach(function (n) { byId[n.id] = n; });
    var PW = 214, PH = 86, AW = 120, AH = 28;
    function box(id) {
      var n = byId[id], p = L.pos[id], piece = (n.kind || 'piece') === 'piece';
      return {x: p.x, y: p.y, w: piece ? PW : AW, h: piece ? PH : AH, cy: p.y + (piece ? PH : AH) / 2};
    }
    var loops = w.loops || [], H = L.height + (L.back.length ? 30 : 0) + loops.length * 30, W = L.width;
    (w.gates || []).forEach(function (gt) {
      if (gt.after && L.pos[gt.after]) W = Math.max(W, box(gt.after).x + gateText(gt).length * 7 + 20);
    });
    var s = '<svg class="graph" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" style="min-width:' + Math.round(W * 0.75) + 'px"' +
      ' role="img" aria-label="workflow graph">' +
      '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">' +
      '<path d="M0,0 L8,4 L0,8 z" fill="#8a8984"/></marker></defs>';
    L.fwd.concat(L.back).forEach(function (e, i) {
      var a = box(e[0]), b = box(e[1]), isBack = i >= L.fwd.length;
      var x1 = a.x + a.w, y1 = a.cy, x2 = b.x, y2 = b.cy, mx = (x1 + x2) / 2;
      var d = isBack ? 'M' + (a.x + a.w / 2) + ',' + (a.y + a.h) + ' C' + (a.x + a.w / 2) + ',' + (a.y + a.h + 50) + ' ' + (b.x + b.w / 2) + ',' + (b.y + b.h + 50) + ' ' + (b.x + b.w / 2) + ',' + (b.y + b.h)
        : 'M' + x1 + ',' + y1 + ' C' + mx + ',' + y1 + ' ' + mx + ',' + y2 + ' ' + x2 + ',' + y2;
      s += '<path d="' + d + '" fill="none" stroke="#8a8984" stroke-width="1.5"' + (isBack ? ' stroke-dasharray="4 3"' : '') + ' marker-end="url(#arr)"/>';
    });
    loops.forEach(function (lp, i) {
      if (!L.pos[lp.from] || !L.pos[lp.to]) return;
      var a = box(lp.to), b = box(lp.from), y = L.height - 4 + (L.back.length ? 30 : 0) + i * 30;
      var self = lp.to === lp.from ? 8 : 0, ax = a.x + a.w / 2 + i * 14 + self, bx = b.x + b.w / 2 - i * 14 - self;
      s += '<path d="M' + ax + ',' + (a.y + a.h) + ' L' + ax + ',' + y + ' L' + bx + ',' + y + ' L' + bx + ',' + (b.y + b.h) + '" fill="none" stroke="#1c5cab" stroke-width="1.5" stroke-dasharray="5 3" marker-end="url(#arr)"/>';
      var label = 'repair, ' + loopText(lp) + ': ' + (lp.rounds ? rounds(lp.rounds.mean) + ', ' : '') + 'reruns pieces with ' + pct(lp.share) + ' of cost';
      s += '<text x="' + ((self ? Math.max(ax, bx) : Math.min(ax, bx)) + 6) + '" y="' + (y - 5) + '" font-size="12" fill="#184f95" data-tip="' + esc(loopTip(lp)) + '">' + esc(label) + '</text>';
    });
    (g.nodes || []).forEach(function (n) {
      var b = box(n.id);
      if ((n.kind || 'piece') !== 'piece') {
        s += '<g data-tip="' + esc('artifact ' + n.id) + '"><rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" rx="14" fill="#f3f2ef" stroke="#c9c8c3"/>' +
          '<text x="' + (b.x + b.w / 2) + '" y="' + (b.y + 18) + '" font-size="12" text-anchor="middle" fill="#52514e">' + esc(n.id) + '</text></g>';
        return;
      }
      var pr = per[n.id] || n.prediction || {}, cost = pr.cost || {}, st = n.setting || {};
      var lines = [
        [esc(n.name || n.id) + ' <tspan fill="#52514e" font-weight="400">(' + esc(n.role || '') + (n.width > 1 ? ' x' + n.width : '') + ')</tspan>', 13, 600],
        [esc((st.model || 'no setting') + (st.effort ? '/' + st.effort : '')), 12, 400],
        [esc(costLabel(pr)), 12, 400],
        [esc(tok((cost.tokens || {}).mean) + ' per run, ' + pct(shares[n.id]) + ' of cost'), 12, 400]];
      s += '<g data-tip="' + esc(pieceTip(w, n)) + '">';
      for (var k = Math.min(2, (n.width || 1) - 1); k > 0; k--) {
        s += '<rect x="' + (b.x + 4 * k) + '" y="' + (b.y - 4 * k) + '" width="' + b.w + '" height="' + b.h + '" rx="8" fill="#fff" stroke="#9ec5f4" stroke-width="1.5"/>';
      }
      s += '<rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" rx="8" fill="#fff" stroke="#2a78d6" stroke-width="1.5"/>';
      lines.forEach(function (l, i) {
        s += '<text x="' + (b.x + 10) + '" y="' + (b.y + 20 + i * 19) + '" font-size="' + l[1] + '" font-weight="' + l[2] + '" fill="#0b0b0b">' + l[0] + '</text>';
      });
      var sh = Math.max(0, Math.min(1, shares[n.id] || 0));
      s += '<rect x="' + b.x + '" y="' + (b.y + b.h - 4) + '" width="' + (b.w * sh).toFixed(1) + '" height="4" fill="#9ec5f4"/></g>';
    });
    (w.gates || []).forEach(function (gt) {
      if (!gt.after || !L.pos[gt.after]) return;
      var b = box(gt.after), text = gateText(gt);
      s += '<g data-tip="' + esc(gateTip(gt)) + '"><path d="M' + (b.x + b.w + 10) + ',' + (b.cy - 8) + ' l8,8 l-8,8 l-8,-8 z" fill="#1c5cab"/>' +
        '<text x="' + (b.x) + '" y="' + (b.y - 8) + '" font-size="12" fill="#184f95">' + esc(text) + '</text></g>';
    });
    return s + '</svg>';
  }
  function gateText(gt) {
    return 'gate ' + gt.id + ': pass ' + (gt.pass ? pct(gt.pass.mean) : 'n/a') + ', ' + rounds(gt.rounds ? gt.rounds.mean : 1);
  }
  function loopTip(lp) {
    return 'repair loop after gate ' + lp.gate + (lp.rule ? ' (' + lp.rule + ')' : '') + '\npieces: ' + (lp.pieces || []).join(', ') +
      '\nexpected rounds: ' + iv(lp.rounds, function (v) { return num(v, 2); }) + '\nthese pieces\' share of the run\'s cost: ' + pct(lp.share);
  }
  function gateTip(gt) {
    return 'gate ' + gt.id + (gt.rule ? ' (' + gt.rule + ')' : '') + ' after ' + gt.after + (gt.on_fail ? ', on fail back to ' + gt.on_fail : ', on fail stop') +
      '\npass chance per round: ' + iv(gt.pass, pct) + '\nexpected rounds: ' + iv(gt.rounds, function (v) { return num(v, 2); });
  }
  var REFS = null;
  function effNodes(refs) {
    if (!REFS) {
      REFS = {};
      allNodes().forEach(function (nd) { (REFS[nd.level + ':' + nd.key] = REFS[nd.level + ':' + nd.key] || []).push(nd); });
    }
    var out = [];
    (refs || []).forEach(function (r) { (REFS[r] || []).forEach(function (nd) { if (S.head === 'all' || nd.head === S.head) out.push(nd); }); });
    return out;
  }
  function loopText(lp) { return lp.to === lp.from ? lp.to + ' retries itself' : lp.to + ' back to ' + lp.from; }
  // D60: `cost` is the piece's whole-run contribution, `cost_per_round` one execution (absent before lane 5 fills it).
  function perRound(pr) { var c = pr.cost_per_round; return c && c.usd && fin(c.usd.mean) ? c : null; }
  function costLabel(pr) {
    var c = perRound(pr);
    return c ? iv(c.usd, usd, true) + ' per round' : iv((pr.cost || {}).usd, usd, true) + ' per run';
  }
  function costAbove(pr) { var c = perRound(pr); return above(c ? c.usd : (pr.cost || {}).usd); }
  function pieceTip(w, n) {
    var pr = (w.per_piece || {})[n.id] || n.prediction || {}, cost = pr.cost || {}, st = n.setting || {};
    var lines = [(n.name || n.id) + ' (' + (n.role || '') + (n.width > 1 ? ', ' + n.width + ' parallel copies' : '') + ')',
      (st.harness || '') + ' ' + (st.model || '') + (st.effort ? '/' + st.effort : ''),
      'cost per round: ' + (perRound(pr) ? iv(perRound(pr).usd, usd) : 'not given'), 'cost per run: ' + iv(cost.usd, usd),
      'tokens per run: ' + iv(cost.tokens, tokn),
      'expected rounds: ' + iv(pr.rounds, function (v) { return num(v, 2); }), 'share of the run\'s cost: ' + pct((w.shares || {})[n.id])];
    if (pr.gate_pass) lines.push('gate pass per round: ' + iv(pr.gate_pass, pct));
    if (n.workers) lines.push('one of ' + n.workers + ' parallel workers of piece ' + n.piece + '; the figures above are one worker\'s part');
    effNodes(((w.effects || {}).pieces || {})[n.piece || n.id]).forEach(function (e) {
      lines.push(levelWord(e.level) + ' ' + keyText(e) + ' [' + headLabel(e.head) + ']: ' + fmtDisplay(e));
    });
    return lines.join('\n');
  }
  function predictionLine(p) {
    if (!p) return '';
    var s = 'Predicted run: ' + iv((p.cost || {}).usd, usd) + ', ' + tok(((p.cost || {}).tokens || {}).mean) + meanTail((p.cost || {}).tokens) +
      '; chance of an accepted result ' + iv(p.p_success, pct) + '; repair rounds ' + iv(p.rounds, function (v) { return num(v, 2); });
    if (p.support === 0) s += '; no runs yet in any group close to this configuration';
    else if (p.support != null) s += '; ' + p.support + (p.support === 1 ? ' run' : ' runs') + ' in the tightest group with data';
    return '<p class="note">' + esc(s) + '.</p>';
  }
  // I11: every configuration, grouped by workflow graph in the order the view gives; a group opens when the
  // drawn configuration is in it or the reader opened it.
  function groupsOf(list) {
    var order = [], by = {};
    list.forEach(function (x, j) {
      var gname = x.group || ((x.label || '').split(':')[0]) || 'workflow';
      if (!by[gname]) { by[gname] = []; order.push(gname); }
      by[gname].push(j);
    });
    return order.map(function (gname) { return {name: gname, items: by[gname]}; });
  }
  function targetOf(p) {
    var t = D.target;
    if (!t || !p) return null;
    if (t.head === 'success') return {d: p.p_success, f: pct};
    var sc = (p.scores || {})[t.head.split(':').slice(1).join(':')];
    return sc && sc.value ? {d: sc.value, f: function (v) { return num(v, 0) + (sc.unit ? ' ' + sc.unit : ''); }} : null;
  }
  function targetWord() {
    var t = D.target;
    if (!t) return '';
    return t.head === 'success' ? 'chance of an accepted result' : 'expected ' + t.head.split(':').slice(1).join(':');
  }
  function pickerHtml(list) {
    var groups = groupsOf(list), tw = targetWord();
    var s = '<p><label>Configuration: <select id="wfpick">';
    groups.forEach(function (gr) {
      s += '<optgroup label="' + esc(gr.name + ' (' + gr.items.length + ')') + '">';
      gr.items.forEach(function (j) {
        var x = list[j];
        s += '<option value="' + j + '"' + (j === S.wf ? ' selected' : '') + '>' + esc(x.label || x.config) + (x.origin ? ' (' + esc(x.origin) + ')' : '') + '</option>';
      });
      s += '</optgroup>';
    });
    s += '</select></label></p>';
    s += '<p class="note">' + list.length + ' configurations in ' + groups.length + (groups.length === 1 ? ' workflow graph' : ' workflow graphs') +
      ', grouped by graph. In a group: your usual first, then by recorded runs' + (tw ? ', ties broken by ' + esc(tw) : '') + '. Click one to draw it.</p>';
    groups.forEach(function (gr) {
      var open = gr.items.indexOf(S.wf) >= 0 || S.open[gr.name];
      s += '<details class="wfgroup"' + (open ? ' open' : '') + '><summary data-group="' + esc(gr.name) + '"><b>' + esc(gr.name) + '</b> <span class="lvl">' +
        gr.items.length + (gr.items.length === 1 ? ' configuration' : ' configurations') + '</span></summary>' +
        '<div class="scroll"><table class="compact wflist"><thead><tr><th>Configuration</th><th class="num">Runs</th>' + (tw ? '<th class="num">' + esc(tw) + '</th>' : '') +
        '<th class="num">Cost per run</th></tr></thead><tbody>';
      gr.items.forEach(function (j) {
        var x = list[j], p = x.prediction || {}, t = targetOf(p), cost = (p.cost || {}).usd;
        s += '<tr data-wf="' + j + '"' + (j === S.wf ? ' class="on"' : '') + '><td><a href="#graph" data-wf="' + j + '">' + esc(x.label || x.config) + '</a>' +
          (x.origin === 'usual' ? ' <span class="tag">usual</span>' : '') + '</td><td class="num">' + (fin(x.runs) ? x.runs : '') + '</td>' +
          (tw ? '<td class="num">' + (t && t.d && fin(t.d.mean) ? esc(t.f(t.d.mean)) : 'n/a') + '</td>' : '') +
          '<td class="num">' + (cost && fin(cost.mean) ? esc(usd(cost.mean)) : 'n/a') + '</td></tr>';
      });
      s += '</tbody></table></div></details>';
    });
    return s;
  }
  function graphHtml(i) {
    var list = (D.workflows && D.workflows.length) ? D.workflows : (D.workflow ? [D.workflow] : []);
    if (!list.length) {
      return '<p class="empty">No configuration to draw. Pass <code>--workflow CFG</code> or <code>--workflow FILE.toml</code>, ' +
        'or record runs for this task type and repo, then run <code>loopmath posterior --html</code> again.</p>';
    }
    if (i != null) S.wf = i;
    if (S.wf >= list.length) S.wf = 0;
    var w = list[S.wf], s = '';
    if (list.length > 1) s += pickerHtml(list);
    s += '<h2>' + esc(w.label || w.config) + '</h2><p class="note">Configuration <code>' + esc(w.config) + '</code>. ' +
      'Each piece shows its predicted cost per round with an 80% range (per run when the fit gives no per-round figure), ' +
      'its tokens per run and its share of the run\'s cost. ' +
      'Gates show the chance of passing per round and the expected rounds; dashed arcs are repair loops. Hover for details.</p>';
    s += predictionLine(w.prediction);
    s += '<div class="graphwrap">' + graphSvg(w) + '</div>';
    var per = w.per_piece || {}, shares = w.shares || {}, pieces = ((w.graph || {}).nodes || []).filter(function (n) { return (n.kind || 'piece') === 'piece'; });
    var tailed = [];  // what the graph's boxes and labels show without the note (D107 and its note 2)
    pieces.forEach(function (n) {
      var pr = per[n.id] || n.prediction || {};
      if (costAbove(pr)) tailed.push('cost of ' + n.id);
      if (above((pr.cost || {}).tokens)) tailed.push('tokens of ' + n.id);
    });
    (w.gates || []).forEach(function (gt) {
      if (above(gt.pass)) tailed.push('pass chance of gate ' + gt.id);
      if (above(gt.rounds)) tailed.push('rounds of gate ' + gt.id);
    });
    (w.loops || []).forEach(function (lp) { if (above(lp.rounds)) tailed.push('rounds of the repair ' + loopText(lp)); });
    if (tailed.length) s += '<p class="note">In the graph, ' + esc(tailed.join(', ')) + ': ' + esc(TAIL_NOTE) + '.</p>';
    s += '<h3>Pieces</h3><div class="scroll"><table><thead><tr><th>Piece</th><th>Setting</th><th class="num">Cost per round</th>' +
      '<th class="num">Cost per run</th><th class="num">Tokens per run</th><th class="num">Expected rounds</th><th class="num">Share of cost</th><th class="num">Gate pass per round</th></tr></thead><tbody>';
    pieces.forEach(function (n) {
      var pr = per[n.id] || n.prediction || {}, cost = pr.cost || {}, st = n.setting || {};
      s += '<tr><td>' + esc(n.id) + ' <span class="lvl">' + esc(n.role || '') + (n.width > 1 ? ' x' + n.width : '') + '</span></td>' +
        '<td>' + esc((st.harness || '') + ' ' + (st.model || '') + (st.effort ? '/' + st.effort : '')) + '</td>' +
        '<td class="num">' + (perRound(pr) ? ivCell(perRound(pr).usd, usd) : 'n/a') + '</td>' +
        '<td class="num">' + ivCell(cost.usd, usd) + '</td><td class="num">' + ivCell(cost.tokens, tokn) + '</td>' +
        '<td class="num">' + ivCell(pr.rounds, function (v) { return num(v, 2); }) + '</td><td class="num">' + esc(pct(shares[n.id])) + '</td>' +
        '<td class="num">' + (pr.gate_pass ? ivCell(pr.gate_pass, pct) : '') + '</td></tr>';
    });
    s += '</tbody></table></div>';
    if ((w.gates || []).length) {
      s += '<h3>Gates</h3><div class="scroll"><table><thead><tr><th>Gate</th><th>After</th><th>On fail</th>' +
        '<th class="num">Pass chance per round</th><th class="num">Expected rounds</th></tr></thead><tbody>';
      w.gates.forEach(function (gt) {
        var byRound = (gt.pass_by_round || []).map(function (p, k) { return 'round ' + (k + 1) + ': ' + pct(p.mean) + meanTail(p); }).join(', ');
        s += '<tr><td>' + esc(gt.id) + (gt.rule ? ' <span class="lvl">' + esc(gt.rule) + '</span>' : '') + '</td><td>' + esc(gt.after || '') + '</td>' +
          '<td>' + esc(gt.on_fail || 'stop') + '</td><td class="num">' + ivCell(gt.pass, pct) + (byRound ? '<span class="why">' + esc(byRound) + '</span>' : '') + '</td>' +
          '<td class="num">' + ivCell(gt.rounds, function (v) { return num(v, 2); }) + '</td></tr>';
      });
      s += '</tbody></table></div>';
    }
    if ((w.loops || []).length) {
      s += '<h3>Repair loops</h3><div class="scroll"><table><thead><tr><th>Loop</th><th>Pieces</th><th class="num">Expected rounds</th>' +
        '<th class="num">Pieces\' share of cost</th></tr></thead><tbody>';
      w.loops.forEach(function (lp) {
        s += '<tr><td>' + esc(loopText(lp)) + ' <span class="lvl">gate ' + esc(lp.gate) + '</span></td><td>' + esc((lp.pieces || []).join(', ')) + '</td>' +
          '<td class="num">' + ivCell(lp.rounds, function (v) { return num(v, 2); }) + '</td><td class="num">' + esc(pct(lp.share)) + '</td></tr>';
      });
      s += '</tbody></table></div>';
    }
    var eff = w.effects || {}, any = false;
    var axes = pageAxes(effNodes([].concat.apply(eff.workflow || [], Object.keys(eff.pieces || {}).map(function (k) { return eff.pieces[k]; }))));
    var block = '<h3>Estimates behind each piece\'s setting</h3><p class="note">The level estimates that apply to each piece, for the head chosen on the first tab (' +
      esc(S.head === 'all' ? 'all heads' : headLabel(S.head)) + ').</p>';
    pieces.forEach(function (n) {
      var rows = effNodes((eff.pieces || {})[n.id]);
      if (!rows.length) return;
      any = true;
      block += '<h3>' + esc(n.id) + '</h3>' + nodeRows(rows, {showHead: S.head === 'all', axes: axes});
    });
    var topo = effNodes(eff.workflow);
    if (topo.length) { any = true; block += '<h3>Workflow shape</h3>' + nodeRows(topo, {showHead: S.head === 'all', axes: axes}); }
    if (any) s += block;
    return s;
  }

  // ---------------------------------------------------------------- data tab
  var COL = {reason: 'Reason', n: 'Rows', run: 'Run', level: 'Level', source: 'Source', head: 'Head', runs: 'Runs'};
  function colName(k) { return COL[k] || (HEADS[k] || /^(cost|success|gate|score:)/.test(k) ? headLabel(k) : k); }
  function matrix(obj, rowName, rowLabel) {
    var cols = Object.keys(obj || {}), rows = [];
    cols.forEach(function (c) { Object.keys(obj[c] || {}).forEach(function (r) { if (rows.indexOf(r) < 0) rows.push(r); }); });
    if (!cols.length) return '<p class="empty">Not recorded by this fit.</p>';
    var s = '<div class="scroll"><table class="compact"><thead><tr><th>' + esc(colName(rowName)) + '</th>';
    cols.forEach(function (c) { s += '<th class="num">' + esc(colName(c)) + '</th>'; });
    s += '</tr></thead><tbody>';
    rows.forEach(function (r) {
      s += '<tr><td>' + esc(rowLabel ? rowLabel(r) : r) + '</td>';
      cols.forEach(function (c) { var v = (obj[c] || {})[r]; s += '<td class="num">' + esc(v == null ? '' : (typeof v === 'number' ? num(v, 3) : generic(v, true))) + '</td>'; });
      s += '</tr>';
    });
    return s + '</tbody></table></div>';
  }
  function bySource(rows) {  // {head: {source: n}} to {source: {head: n}}: a few sources across, one head per row
    var out = {};
    Object.keys(rows || {}).forEach(function (h) {
      Object.keys(rows[h] || {}).forEach(function (src) { (out[src] = out[src] || {})[h] = rows[h][src]; });
    });
    return out;
  }
  function generic(v, flat) {
    if (v == null) return '';
    if (typeof v !== 'object') return typeof v === 'number' ? num(v, 3) : String(v);
    if (flat) return JSON.stringify(v);
    if (Array.isArray(v)) {
      if (!v.length) return '<p class="empty">None.</p>';
      var keys = [];
      v.forEach(function (x) { if (x && typeof x === 'object') Object.keys(x).forEach(function (k) { if (keys.indexOf(k) < 0) keys.push(k); }); });
      if (!keys.length) return '<p>' + esc(v.join(', ')) + '</p>';
      var numeric = {};
      keys.forEach(function (k) { numeric[k] = v.every(function (x) { var y = (x || {})[k]; return y == null || typeof y === 'number'; }); });
      var s = '<div class="scroll"><table class="compact"><thead><tr>';
      keys.forEach(function (k) { s += '<th' + (numeric[k] ? ' class="num"' : '') + '>' + esc(colName(k)) + '</th>'; });
      s += '</tr></thead><tbody>';
      v.forEach(function (x) {
        s += '<tr>';
        keys.forEach(function (k) { s += '<td' + (numeric[k] ? ' class="num"' : '') + '>' + esc(generic((x || {})[k], true)) + '</td>'; });
        s += '</tr>';
      });
      return s + '</tbody></table></div>';
    }
    var t = '<table class="kv"><tbody>';
    Object.keys(v).forEach(function (k) {
      var x = v[k];
      t += '<tr><td>' + esc(k) + '</td><td>' + (x && typeof x === 'object' ? generic(x) : esc(generic(x))) + '</td></tr>';
    });
    return t + '</tbody></table>';
  }
  function dataHtml() {
    var d = D.data || {}, fit = D.fit || {}, n = fit.n_runs, runs;
    if (n && typeof n === 'object') runs = Object.keys(n).map(function (k) { return k + ' ' + n[k]; }).join(', ');
    else runs = n == null ? 'not recorded' : String(n);
    var s = '<h2>The fit</h2><table class="kv"><tbody>' +
      '<tr><td>Fit</td><td><code>' + esc(fit.id) + '</code></td></tr><tr><td>Fitted at</td><td>' + esc(fit.at) + '</td></tr>' +
      '<tr><td>Runs</td><td>' + esc(runs) + '</td></tr>' +
      '<tr><td>Time to fit</td><td>' + esc(fin(d.fit_time_s) ? num(d.fit_time_s, 1) + ' s' : 'not recorded') + '</td></tr>' +
      (d.code_version ? '<tr><td>Code version</td><td>' + esc(d.code_version) + '</td></tr>' : '') +
      '<tr><td>Page generated</td><td>' + esc(D.generated_at) + '</td></tr></tbody></table>';
    s += '<h2>Rows per source and head</h2><p class="note">How many rows each data source gave each head. The prior is data too: ' +
      'our sweep, E0, RQ1 and benchmark rows enter as their own sources.</p>' + matrix(bySource(d.rows), 'head', headLabel);
    var rbs = d.runs_by_source || {};
    if (Object.keys(rbs).length) s += '<h3>Runs per source</h3>' + matrix({runs: rbs}, 'source');
    s += '<h2>Dropped rows</h2>' + (d.dropped && d.dropped.length ? generic(d.dropped) : '<p class="empty">No rows were dropped.</p>');
    s += '<h2>Scales per level (phi)</h2><p class="note">The empirical Bayes scale of each level: how far a child node may move from its parent. ' +
      'Larger means the data showed more spread at that level.</p>' + matrix(d.scales, 'level', levelWord);
    s += '<h2>Sensitivity to the benchmark prior</h2>' + ((d.without || []).length ? '<p class="note">This fit left out: ' +
      esc(d.without.join(', ')) + '. Compare it with a fit on everything to see what those sources move.</p>' : '') + (d.sensitivity ? generic(d.sensitivity)
      : '<p class="empty">No comparison yet. Run <code>loopmath fit --without benchmark</code> to see how much the benchmark priors move the estimates.</p>');
    return s;
  }

  // ---------------------------------------------------------------- page
  function lede() {
    var fit = D.fit || {}, n = fit.n_runs, total = null, user = null;
    if (n && typeof n === 'object') { total = 0; Object.keys(n).forEach(function (k) { total += +n[k] || 0; }); user = +n.user || 0; }
    else if (fin(n)) total = n;
    return 'What loopmath currently estimates about how each model, effort level, role, workflow shape, task type and repo changes cost, ' +
      'the chance of success and the chance a gate passes. Every estimate has an 80% range and the number of runs it rests on. ' +
      'These are the posterior estimates of fit ' + esc(fit.id) + ' (' + esc(fit.at) + ')' +
      (total != null ? ', from ' + total.toLocaleString('en-US') + ' runs' + (user != null ? ', ' + user.toLocaleString('en-US') + ' of them yours' : '') : '') + '.';
  }
  function taskLine() {
    var t = D.task;
    if (!t) return '';
    var why = {arguments: 'from the command line', store: 'the most common type and repo in your runs',
      'default': 'no task given and no runs yet: a task in a repo with no runs'}[t.from] || t.from;
    var feats = Object.keys(t.features || {}).map(function (k) { return k + '=' + t.features[k]; }).join(', ');
    return 'The graph tab is computed for a <b>' + esc(t.type) + '</b> task in <b>' + esc(t.repo) + '</b>' + (t.subtype ? ' (' + esc(t.subtype) + ')' : '') +
      (feats ? ' with ' + esc(feats) : '') + ' (' + esc(why) + '). For another task: <code>loopmath posterior --type T --repo R --html</code>.';
  }
  function show(tab) {
    if (['levels', 'graph', 'data'].indexOf(tab) < 0) tab = 'levels';
    Array.prototype.forEach.call(document.querySelectorAll('[data-panel]'), function (el) { el.hidden = el.getAttribute('data-panel') !== tab; });
    Array.prototype.forEach.call(document.querySelectorAll('[data-tab]'), function (el) { el.setAttribute('aria-selected', String(el.getAttribute('data-tab') === tab)); });
  }
  function draw() {
    document.getElementById('tab-levels').innerHTML = levelsHtml();
    document.getElementById('tab-graph').innerHTML = graphHtml();
    document.getElementById('tab-data').innerHTML = dataHtml();
  }
  function init() {
    var hs = headList();
    S.head = D.head && hs.indexOf(D.head) >= 0 ? D.head : (hs[0] || 'all');
    document.getElementById('lede').innerHTML = lede();
    document.getElementById('task').innerHTML = taskLine();
    draw();
    show((location.hash || '').slice(1));
    document.addEventListener('click', function (ev) {
      var sum = ev.target && ev.target.closest ? ev.target.closest('summary[data-group]') : null;
      if (sum && sum.getAttribute('data-group') != null) { S.open[sum.getAttribute('data-group')] = !(sum.parentNode && sum.parentNode.open); return; }  // before it toggles
      var t = ev.target && ev.target.closest ? ev.target.closest('[data-tab],[data-head],a[data-wf]') : null;
      if (!t) return;
      if (t.getAttribute('data-wf') != null) {
        ev.preventDefault();
        S.wf = +t.getAttribute('data-wf') || 0;
        document.getElementById('tab-graph').innerHTML = graphHtml();
        return;
      }
      if (t.getAttribute('data-tab')) { show(t.getAttribute('data-tab')); if (history.replaceState) history.replaceState(null, '', '#' + t.getAttribute('data-tab')); }
      else { S.head = t.getAttribute('data-head'); draw(); }
    });
    document.addEventListener('change', function (ev) {
      if (ev.target && ev.target.id === 'wfpick') { S.wf = +ev.target.value || 0; document.getElementById('tab-graph').innerHTML = graphHtml(); }
    });
    var tip = document.getElementById('tip');
    document.addEventListener('mousemove', function (ev) {
      var t = ev.target && ev.target.closest ? ev.target.closest('[data-tip]') : null;
      if (!t) { tip.hidden = true; return; }
      tip.textContent = t.getAttribute('data-tip');
      tip.hidden = false;
      var x = ev.clientX + 14, y = ev.clientY + 14;
      if (x + 370 > window.innerWidth) x = Math.max(4, ev.clientX - 370);
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
    });
  }
  window.LMPosterior = {levelsHtml: levelsHtml, graphHtml: graphHtml, dataHtml: dataHtml, layout: layout,
    fmtDisplay: fmtDisplay, heads: headList, state: S, init: init};
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
"""


# ---------------------------------------------------------------- handler
def _load_state(home: Path, fit_id: str | None = None) -> Any:
    if fit_id:  # `--fit ID`: a kept fit instead of fits/latest
        from ..belief.fit import load_fit

        return load_fit(home, fit_id)
    from ..belief.state import load_latest

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
    if state is None:
        return fail(f"no fit yet under {home}: run `loopmath fit` (or `loopmath onboard`) first", EXIT_NO_FIT)
    try:
        features, unknown = task_features(getattr(args, "feature", None) or [])
        if unknown:
            print(f"note: loopmath does not model the feature {', '.join(unknown)} (see loopmath task-types); "
                  "it is shown but does not change the estimates", file=sys.stderr)
        data = build_view(state, home=home, level=args.level or "all", head=args.head, workflow=args.workflow,
                          task_type=args.task_type, repo=args.repo, subtype=getattr(args, "subtype", None),
                          features=features)
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
