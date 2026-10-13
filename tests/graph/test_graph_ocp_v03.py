"""`graph --format ocp` writes OCP v0.3 under `dev.loopmath.` keys and still reads
the graph documents written before 0.3 under `dev.dagr.` (spec/OCP.md section 8)."""

import json

import pytest

from loopmath.graph import extract, to_ocp
from loopmath.graph.runfile import to_runfile
from loopmath.ingest.ocp import from_ocp
from loopmath.ocp.emit import validate_strict
from tests.test_graph_extract import ALPHA, skeleton_records
from tests.test_ingest_ocp import _native_eval_graph

RENAMED = {"dev.loopmath.graph": "dev.dagr.graph", "dev.loopmath.artifact": "dev.dagr.artifact"}


def _ext_keys(value) -> set[str]:
    """Every key of every `ext` object in `value`."""
    found: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "ext" and isinstance(nested, dict):
                found.update(nested)
            found |= _ext_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            found |= _ext_keys(nested)
    return found


def _pre_03(doc: dict) -> dict:
    """`doc` as the graph writer wrote it before 0.3: version 0.2, `dev.dagr.` keys."""
    text = json.dumps(doc)
    for new, old in RENAMED.items():
        text = text.replace(json.dumps(new), json.dumps(old))
    out = json.loads(text)
    out["ocp"] = "0.2"
    return out


@pytest.fixture(scope="module", params=["skeleton", "native-swarm-c"])
def doc(request) -> dict:
    if request.param == "skeleton":
        return to_ocp(extract(skeleton_records(), workspaces=[ALPHA]))
    return to_ocp(_native_eval_graph("c"))


def test_the_graph_writes_v03_with_no_pre_rename_key(doc):
    assert doc["ocp"] == "0.3"
    keys = _ext_keys(doc)
    assert {"dev.loopmath.graph", "dev.loopmath.artifact"} <= keys
    assert not [k for k in keys if k.startswith("dev.dagr.")]
    strict = validate_strict(doc)
    assert [f for f in strict if f["level"] == "error"] == []
    assert [f for f in strict if f["code"] == "W182"] == []


def test_a_pre_03_graph_document_reads_as_the_same_graph(doc):
    old = _pre_03(doc)
    assert {"dev.dagr.graph", "dev.dagr.artifact"} <= _ext_keys(old)
    from_old, from_new = from_ocp(old), from_ocp(doc)
    # The same projection counters either way (the skeleton graph loses nine
    # members on any round trip, before 0.3 too), none for the rename, and
    # the same v0.3 document out.
    assert from_old.meta == from_new.meta
    assert to_ocp(from_old) == to_ocp(from_new)


def test_a_pre_03_native_swarm_round_trips_losslessly_to_v03():
    doc = to_ocp(_native_eval_graph("c"))
    graph = from_ocp(_pre_03(doc))
    assert not [k for k in graph.meta if k.startswith("ocp_projection_")]
    assert to_ocp(graph) == doc


def test_the_contract_run_file_keeps_its_own_key():
    # `graph --format run` writes a herdr-dagr contract file, not OCP.
    run = to_runfile(extract(skeleton_records(), workspaces=[ALPHA]))
    assert set(run["ext"]) == {"dev.dagr.graph"}
    assert all(set(t.get("ext", {})) <= {"dev.dagr.graph"} for t in run["tasks"])
