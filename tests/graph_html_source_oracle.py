#!/usr/bin/env python3
"""Write an independent attempt oracle from source graph nodes."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, fields
from pathlib import Path

from loopmath.graph.ocp_support import _COST_FIELDS, sanitize
from loopmath.graph.schema import Artifact, Graph, GraphNode
from loopmath.graph.token_completeness import TOKEN_STREAMS
from loopmath.ingest.ocp import load_ocp


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str):
        return sanitize(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def source_oracle(graph: Graph) -> dict:
    """Serialize source nodes without using the HTML reduction."""
    artifact_links = {
        node.id: {
            "written": [
                index
                for index, artifact in enumerate(graph.artifacts)
                if node.id in artifact.writers
            ],
            "read": [
                index
                for index, artifact in enumerate(graph.artifacts)
                if node.id in artifact.consumers
            ],
        }
        for node in graph.nodes
    }
    return {
        "schema": "GraphNode",
        "source_fields": [item.name for item in fields(GraphNode)],
        "artifact_fields": [item.name for item in fields(Artifact)],
        "token_streams": [key for _, key in TOKEN_STREAMS],
        "token_total_streams": [key for key, _ in _COST_FIELDS],
        "omissions": [],
        "nodes": [_json_safe(asdict(node)) for node in graph.nodes],
        "artifacts": [_json_safe(asdict(artifact)) for artifact in graph.artifacts],
        "edges": [_json_safe(asdict(edge)) for edge in graph.edges],
        "artifact_links": artifact_links,
    }


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: graph_html_source_oracle.py OCP_FILE OUTPUT_JSON", file=sys.stderr)
        return 2
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    payload = source_oracle(load_ocp(source))
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        f"wrote {output} from {len(payload['nodes'])} source GraphNode records "
        f"with {len(payload['source_fields'])} fields, {len(payload['artifacts'])} "
        f"artifacts, and {len(payload['omissions'])} omissions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
