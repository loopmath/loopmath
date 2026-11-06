"""The example flow folder: the spec example plus the hypothesis edits in `flow.ext`."""

from __future__ import annotations

import json
from pathlib import Path

from loopmath.flows.records import EXT_HYPOTHESIS, Edit, Flow, hypothesis_edits
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id, flow_digest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief" / "flow.ocp.json"
SPEC_FLOW = ROOT / "spec" / "examples" / "v0.4" / "lit-brief.flow.ocp.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# 16


def test_example_validates_and_keeps_the_base_id():
    doc = _read(EXAMPLE)
    assert conformance.validate_doc(doc) == []
    c = doc["flow"]["configuration"]
    assert c["id"] == config_id(c["workflow"], c["settings"]) == "cfg_9bc4f6e876f9"


def test_example_is_the_spec_example_plus_ext():
    doc, spec = _read(EXAMPLE), _read(SPEC_FLOW)
    assert flow_digest(doc["flow"]) == flow_digest(spec["flow"]) == doc["flow"]["digest"]
    assert list(doc["flow"]["ext"]) == [EXT_HYPOTHESIS]
    del doc["flow"]["ext"]
    assert doc == spec


def test_example_has_the_spec_example_formatting():
    text = EXAMPLE.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), indent=2, ensure_ascii=False) + "\n"


# 17


def test_hypothesis_edits_parse_and_name_allowed_edits():
    flow = Flow.from_document(_read(EXAMPLE))
    edits = hypothesis_edits(flow)
    assert len(edits) == 2 and all(isinstance(e, Edit) for e in edits)
    model, drop = edits
    assert (model.kind, model.piece) == ("model", "extract")
    assert model.value in flow.allow.pieces["extract"].models
    assert (drop.kind, drop.piece, drop.value) == ("drop", "critic", None) and "value" not in drop.to_dict()
    assert flow.allow.pieces["critic"].drop is True
    assert [e.to_dict() for e in edits] == flow.ext[EXT_HYPOTHESIS]


def test_no_hypothesis_member_gives_no_edits():
    assert hypothesis_edits(Flow.from_document(_read(SPEC_FLOW))) == ()
