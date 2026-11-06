"""OCP v0.4 schema: a strict superset of v0.3, one copy in spec/ and the package."""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from loopmath.ocp import CURRENT_VERSION, conformance

from ._common import EXAMPLES, PACKAGE_SCHEMA, SPEC, V03_EXAMPLES, V03_GOLDEN, codes, example, load

V03 = json.loads((SPEC / "ocp-v0.3.schema.json").read_text())
V04 = json.loads((SPEC / "ocp-v0.4.schema.json").read_text())

# The only places where the v0.4 schema may differ from the v0.3 one. A change anywhere else fails
# test_v04_differs_from_v03_only_at_the_listed_paths; each new 0.4 field adds its path here.
V04_CHANGES = (
    ("$id",),
    ("title",),
    ("description",),
    ("properties", "ocp"),
    ("$defs", "producer", "properties", "capabilities", "properties", "flow"),
    ("$defs", "check"),
    ("$defs", "control", "properties", "checks"),
    ("$defs", "control", "properties", "gates", "description"),
    ("$defs", "control", "properties", "repair", "description"),
    ("$defs", "control", "description"),
    ("required",),
    ("allOf",),
    ("properties", "kind"),
    ("properties", "flow"),
    ("$defs", "flow"),
    ("$defs", "flowInputs"),
    ("$defs", "objective"),
    ("$defs", "judge"),
    ("$defs", "flowAllow"),
    ("$defs", "pieceAllow"),
    ("$defs", "merge"),
    ("$defs", "run", "properties", "flow"),
    ("$defs", "runFlow"),
    ("$defs", "cost", "properties", "basis", "x-recommended"),
    ("$defs", "cost", "properties", "basis", "description"),
    ("$defs", "signal", "properties", "value", "description"),
    ("$defs", "scoreTarget", "properties", "target", "description"),
    ("$defs", "acceptanceRule", "properties", "window_days", "description"),
)


def _without(schema, paths):
    out = copy.deepcopy(schema)
    for path in paths:
        parent = out
        for key in path[:-1]:
            parent = parent.get(key, {}) if isinstance(parent, dict) else {}
        if isinstance(parent, dict):
            parent.pop(path[-1], None)
    return out


def _v03_schema_valid_documents():
    """Every v0.3 document the repo ships that the v0.3 schema accepts."""
    paths = sorted(EXAMPLES.glob("*.ocp.json")) + sorted(V03_EXAMPLES.glob("*.ocp.json")) + sorted(
        V03_GOLDEN.glob("*.ocp.json"))
    validator = Draft202012Validator(V03)
    docs = []
    for path in paths:
        doc = load(path)
        if doc.get("ocp") == "0.3" and not list(validator.iter_errors(doc)):
            docs.append(pytest.param(doc, id=path.name))
    return docs


def test_spec_and_package_copies_are_identical():
    for name in ("ocp-v0.schema.json", "ocp-v0.2.schema.json", "ocp-v0.3.schema.json", "ocp-v0.4.schema.json"):
        assert (SPEC / name).read_bytes() == (PACKAGE_SCHEMA / name).read_bytes(), name


def test_schema_paths_include_v04_inside_the_package():
    assert set(conformance.SCHEMA_PATHS) == {"0.1", "0.2", "0.3", "0.4"}
    path = conformance.SCHEMA_PATHS["0.4"]
    assert path.is_file() and path.parent == PACKAGE_SCHEMA.resolve()
    assert conformance.CURRENT_VERSION == CURRENT_VERSION == "0.4"
    assert V04["properties"]["ocp"]["const"] == "0.4"
    Draft202012Validator.check_schema(V04)


def test_v04_differs_from_v03_only_at_the_listed_paths():
    assert _without(V04, V04_CHANGES) == _without(V03, V04_CHANGES)
    assert V04["$id"] == "urn:ocp:0.4" and V04["title"] == "OCP v0.4 document"


def test_v03_definitions_survive_unchanged_in_shape():
    """Structural superset: no v0.3 property is dropped and no new property became required."""
    for name, old in V03["$defs"].items():
        new = V04["$defs"][name]
        assert set(old.get("properties", {})) <= set(new.get("properties", {})), name
        assert set(new.get("required", [])) <= set(old.get("required", [])), name
    assert set(V03["properties"]) <= set(V04["properties"])
    assert set(V04.get("required", [])) <= set(V03.get("required", []))


@pytest.mark.parametrize("doc", _v03_schema_valid_documents())
def test_every_valid_v03_document_is_valid_v04_after_changing_ocp(doc):
    doc["ocp"] = "0.4"
    errors = [e.message for e in Draft202012Validator(V04).iter_errors(doc)]
    assert errors == []


def test_the_flow_capability_is_a_boolean():
    doc = example("solo")
    doc["ocp"] = "0.4"
    doc["producer"]["capabilities"]["flow"] = True
    assert conformance.validate_doc(doc) == []
    doc["producer"]["capabilities"]["flow"] = "yes"
    assert "E010" in codes(conformance.validate_doc(doc))
