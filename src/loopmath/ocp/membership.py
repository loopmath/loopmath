"""Read saved import membership without guessing roles or repairing explicit references."""
from __future__ import annotations

from typing import Any

_ADAPTERS = {"pi", "opencode", "omp", "otel-genai", "paseo", "orca", "bb", "atrium"}
_MIGRATED_FROM = {"0.1", "0.2", "dagr/1", "dagr/2", "dagr/3"}
_COST_FIELDS = ("usd", "input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")


def imported_node_map(doc: dict) -> dict:
    """Saved node map only for recognized migration or adapter/graph provenance."""
    run = doc.get("run") or {}
    ext = run.get("ext") or {}
    marker = ext.get("dev.loopmath.migration") if isinstance(ext, dict) else None
    migrated = (isinstance(marker, dict) and isinstance(marker.get("from"), str)
                and marker["from"] in _MIGRATED_FROM)
    producer = doc.get("producer") or {}
    name = producer.get("name") if isinstance(producer, dict) else None
    adapter = isinstance(name, str) and name in {
        f"{prefix}/adapter-{a}" for prefix in ("loopmath", "dagr") for a in _ADAPTERS}
    top_ext = doc.get("ext") or {}
    graph = name in ("loopmath", "dagr") and isinstance(top_ext, dict) and any(
        isinstance(top_ext.get(key), dict) for key in ("dev.loopmath.graph", "dev.dagr.graph"))
    config = run.get("configuration") or {}
    mapping = config.get("node_vertex") if isinstance(config, dict) else None
    return mapping if (migrated or adapter or graph) and isinstance(mapping, dict) else {}


def model_work(attempt: dict) -> bool:
    """The membership counterpart of the fitter's costless, model-free check rule."""
    setting = attempt.get("setting") or {}
    if attempt.get("model") or (isinstance(setting, dict) and setting.get("model")):
        return True
    cost = attempt.get("cost") or {}
    return isinstance(cost, dict) and any(
        isinstance(cost.get(key), (int, float)) and cost[key] > 0 for key in _COST_FIELDS)


def materialize(doc: dict[str, Any]) -> None:
    """Fill absent vertices from a validated saved map, on the migration's private copy."""
    mapping = imported_node_map(doc)
    if not mapping:
        return
    config = (doc.get("run") or {}).get("configuration") or {}
    workflow = config.get("workflow") or {}
    settings = config.get("settings") or {}
    if not isinstance(workflow, dict) or not isinstance(settings, dict):
        return
    pieces = {p.get("id") for p in workflow.get("pieces") or [] if isinstance(p, dict) and isinstance(p.get("id"), str)}
    by_node: dict[str, list[dict]] = {}
    for attempt in doc.get("attempts") if isinstance(doc.get("attempts"), list) else []:
        if isinstance(attempt, dict) and isinstance(attempt.get("node"), str):
            by_node.setdefault(attempt["node"], []).append(attempt)
    for node in doc.get("nodes") if isinstance(doc.get("nodes"), list) else []:
        if not isinstance(node, dict) or "vertex" in node:
            continue
        nid = node.get("id")
        piece = mapping.get(nid) if isinstance(nid, str) else None
        if not isinstance(piece, str) or piece not in pieces or piece not in settings:
            continue
        attempts = by_node.get(nid, [])
        # A shared node must not give a model-free check (or an explicit empty
        # vertex) membership indirectly through node.vertex.
        eligible = [a for a in attempts if "vertex" not in a and model_work(a)]
        if eligible and len(eligible) == len(attempts):
            node["vertex"] = piece
        for attempt in eligible:
            attempt["vertex"] = piece
