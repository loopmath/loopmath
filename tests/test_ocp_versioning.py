"""OCP version routing and additive-field compatibility."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "spec"
EXAMPLE = SPEC / "examples" / "swarm-v02.ocp.json"

_spec = importlib.util.spec_from_file_location(
    "ocp_conformance_versioning", SPEC / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def load_example():
    return json.loads(EXAMPLE.read_text())


def errors(doc):
    return [finding for finding in conf.validate_doc(doc) if finding.level == "error"]


def test_unknown_fields_are_tolerated_at_multiple_levels():
    doc = load_example()
    doc["future_top_level"] = {"value": 1}
    doc["attempts"][0]["future_attempt_field"] = "new"
    doc["artifacts"][0]["future_artifact_field"] = True
    doc["edges"][0]["future_edge_field"] = [1, 2]

    assert errors(doc) == []


def test_unknown_version_produces_e003():
    doc = load_example()
    doc["ocp"] = "9.9"

    assert [finding.code for finding in errors(doc)] == ["E003"]
