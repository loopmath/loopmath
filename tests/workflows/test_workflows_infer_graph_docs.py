"""`infer` on `graph --format ocp` documents: routed by the graph extension, not the OCP version (D65; lane 04)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loopmath.graph import extract, to_ocp
from loopmath.graph.schema import Artifact, Graph, GraphEdge, GraphNode
from loopmath.workflows.infer import agents_from_ocp, infer_agents, infer_detail, is_graph_document, load_source
from tests.test_graph_extract import ALPHA, skeleton_records  # the graph writer's own test input (as lane 02's tests)

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "tests" / "fixtures" / "graph" / "roundtrip"
NEW, OLD = "dev.loopmath.", "dev.dagr."  # the graph writer's key prefix from 0.3 on, and before
KEYS = ("graph", "artifact")


def _native(name: str) -> Graph:
    payload = load_source(NATIVE / f"native-swarm-{name}.json")
    return Graph(nodes=[GraphNode(**n) for n in payload["nodes"]], edges=[GraphEdge(**e) for e in payload["edges"]],
                 artifacts=[Artifact(**a) for a in payload["artifacts"]], meta=payload["meta"])


def _twin(doc: dict, version: str, prefix: str) -> dict:
    """`doc` labelled `version`, with the graph writer's ext keys under `prefix` (whichever it wrote)."""
    text = json.dumps(doc)
    for key in KEYS:
        for src in (NEW, OLD):
            text = text.replace(json.dumps(src + key), json.dumps(prefix + key))
    out = json.loads(text)
    out["ocp"] = version
    return out


def _strip(value):
    """`value` without the graph writer's ext keys: an OCP document from some other producer."""
    if isinstance(value, dict):
        return {k: ({x: _strip(y) for x, y in v.items() if not x.startswith((NEW + "graph", OLD + "graph"))}
                    if k == "ext" and isinstance(v, dict) else _strip(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


@pytest.fixture(scope="module", params=["skeleton", "native-swarm-c", "native-swarm-b"])
def doc(request) -> dict:
    if request.param == "skeleton":
        return to_ocp(extract(skeleton_records(), workspaces=[ALPHA]))
    return to_ocp(_native(request.param.rsplit("-", 1)[1]))


def test_v03_graph_documents_infer_as_their_v02_twins(doc):
    base = infer_detail(_twin(doc, "0.2", OLD))
    assert base.inferred_from == "ocp"
    for prefix in (NEW, OLD):  # the 0.3 writer's keys, and the older namespace the readers still take
        twin = _twin(doc, "0.3", prefix)
        assert is_graph_document(twin)
        r = infer_detail(twin)
        assert (r.configuration.id, r.confidence) == (base.configuration.id, base.confidence), prefix
        assert r.node_vertex == base.node_vertex and r.reasons == base.reasons


def test_a_v03_document_without_the_graph_extension_keeps_its_route(doc):
    other = _strip(_twin(doc, "0.3", NEW))
    assert not is_graph_document(other) and "graph" not in json.dumps(other.get("ext") or {})
    r = infer_detail(other)
    before = infer_agents(agents_from_ocp(other), base=0.85, source="ocp")  # the route before D65
    assert (r.configuration.id, r.confidence, r.reasons) == (before.configuration.id, before.confidence, before.reasons)


def test_graph_extension_detection():
    assert is_graph_document({"ocp": "0.3", "ext": {"dev.loopmath.graph": {}}})
    assert is_graph_document({"ocp": "0.1", "ext": {"dev.dagr.graph": {"meta": {}}}})
    assert is_graph_document({"ocp": "0.3", "attempts": [{"id": "a", "ext": {"dev.loopmath.graph": {}}}]})
    assert not is_graph_document({"ocp": "0.3", "ext": {"dev.loopmath.share": {}, "dev.loopmath.artifact": {}}})
    assert not is_graph_document({"ocp": "0.3", "ext": {"dev.loopmath.graph": "not an object"}})
    assert not is_graph_document({"ocp": "0.3", "nodes": [{"id": "n"}], "attempts": []})
