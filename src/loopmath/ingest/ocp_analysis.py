"""Analysis coverage and graph aggregation helpers for OCP ingestion."""

from __future__ import annotations

import copy

from .ocp_common import OCPError
from ..graph.extract import summarize
from ..graph.schema import Artifact, Graph, GraphEdge, GraphNode


def coverage_from_records(records: list[dict]) -> dict:
    """Build grade coverage from native or OCP analysis rows."""
    tiers = {tier: 0 for tier in ("verified", "reported", "heuristic", "asserted", "censored", "ungraded")}
    accepted = rejected = unknown = 0
    for record in records:
        grade = record.get("grade") if isinstance(record.get("grade"), dict) else {}
        tier = grade.get("tier")
        tiers[tier if tier in tiers else "ungraded"] += 1
        value = grade.get("accepted")
        if value is True:
            accepted += 1
        elif value is False:
            rejected += 1
        else:
            unknown += 1
    total = len(records)
    graded = accepted + rejected
    return {
        "n_total": total,
        "n_graded": graded,
        "pct_structural": 100.0 * graded / total if total else 0.0,
        "tiers": tiers,
        "n_accepted": accepted,
        "n_rejected": rejected,
        "n_unknown": unknown,
        "denominator": graded,
        "unevaluable_heuristic": 0,
        "n_synthetic_excluded": 0,
        "n_total_parsed": total,
    }


def merge_graphs(graphs: list[Graph]) -> Graph:
    """Combine independent Graph sources, refusing ambiguous identity collisions."""
    if not graphs:
        return Graph(meta=summarize(Graph()))
    if len(graphs) == 1:
        return graphs[0]
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    artifacts: list[Artifact] = []
    ocp_node_ext: dict[str, dict] = {}
    node_ids: set[str] = set()
    artifact_ids: set[str] = set()
    for graph in graphs:
        for node in graph.nodes:
            if node.id in node_ids:
                raise OCPError(f"graph sources contain duplicate node id {node.id!r}")
            node_ids.add(node.id)
            nodes.append(node)
            if node.id in graph._ocp_node_ext:
                ocp_node_ext[node.id] = copy.deepcopy(graph._ocp_node_ext[node.id])
        for artifact in graph.artifacts:
            if artifact.id in artifact_ids:
                raise OCPError(f"graph sources contain duplicate artifact path {artifact.id!r}")
            artifact_ids.add(artifact.id)
            artifacts.append(artifact)
        edges.extend(graph.edges)
    graph = Graph(
        nodes=nodes,
        edges=edges,
        artifacts=artifacts,
        meta={"component_meta": [copy.deepcopy(graph.meta) for graph in graphs]},
        _ocp_node_ext=ocp_node_ext,
    )
    graph.meta.update(summarize(graph))
    return graph
