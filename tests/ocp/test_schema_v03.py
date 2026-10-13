"""OCP v0.3 schema: a strict superset of v0.2, one copy in spec/ and the package, examples valid."""

from __future__ import annotations

import json

import pytest
from jsonschema import Draft202012Validator

from loopmath.ocp import conformance

from ._common import EXAMPLES, PACKAGE_SCHEMA, SHAPES, SPEC, V03_EXAMPLES, example, load

V02 = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
V03 = json.loads((SPEC / "ocp-v0.3.schema.json").read_text())


def _v02_documents():
    """Every v0.2 document the repo ships that the v0.2 schema accepts."""
    paths = sorted(EXAMPLES.glob("*.ocp.json")) + sorted((EXAMPLES / "golden").glob("*.ocp.json"))
    validator = Draft202012Validator(V02)
    docs = []
    for path in paths:
        doc = load(path)
        if doc.get("ocp") == "0.2" and not list(validator.iter_errors(doc)):
            docs.append(pytest.param(doc, id=path.name))
    return docs


def test_spec_and_package_copies_are_identical():
    for name in ("ocp-v0.schema.json", "ocp-v0.2.schema.json", "ocp-v0.3.schema.json"):
        assert (SPEC / name).read_bytes() == (PACKAGE_SCHEMA / name).read_bytes(), name


def test_schema_paths_cover_every_version_and_resolve_inside_the_package():
    assert set(conformance.SCHEMA_PATHS) == {"0.1", "0.2", "0.3"}
    for version, path in conformance.SCHEMA_PATHS.items():
        assert path.is_file(), version
        assert path.parent == PACKAGE_SCHEMA.resolve(), version
    assert conformance.CURRENT_VERSION == "0.3"
    assert V03["properties"]["ocp"] == {"const": "0.3"} or V03["properties"]["ocp"].get("const") == "0.3"
    Draft202012Validator.check_schema(V03)


@pytest.mark.parametrize("doc", _v02_documents())
def test_every_valid_v02_document_is_valid_v03_after_changing_ocp(doc):
    doc["ocp"] = "0.3"
    errors = [e.message for e in Draft202012Validator(V03).iter_errors(doc)]
    assert errors == []


def test_v02_definitions_survive_unchanged_in_shape():
    """Structural superset: no v0.2 property is dropped and no new property became required."""
    for name, old in V02["$defs"].items():
        new = V03["$defs"][name]
        assert set(old.get("properties", {})) <= set(new.get("properties", {})), name
        assert set(new.get("required", [])) <= set(old.get("required", [])), name
    assert set(V02["properties"]) <= set(V03["properties"])
    assert set(V03.get("required", [])) <= set(V02.get("required", []))


def test_there_is_one_example_per_catalog_shape_plus_full_fields():
    names = {p.name.removesuffix(".ocp.json") for p in V03_EXAMPLES.glob("*.ocp.json")}
    assert names == set(SHAPES) | {"full-fields"}
    for shape in SHAPES:
        assert example(shape)["run"]["configuration"]["workflow"]["id"] == shape


@pytest.mark.parametrize("name", [*SHAPES, "full-fields"])
def test_example_workflows_are_the_catalog_shapes(name):
    workflow = example(name)["run"]["configuration"]["workflow"]
    assert workflow == conformance.catalog_resolver(workflow["id"], workflow["version"])


@pytest.mark.parametrize("name", [*SHAPES, "full-fields"])
def test_examples_have_no_findings_at_all(name):
    findings = conformance.validate_file(V03_EXAMPLES / f"{name}.ocp.json")
    assert findings == []


def _new_properties():
    added = {}
    for name, definition in V03["$defs"].items():
        old = V02["$defs"].get(name, {})
        new = set(definition.get("properties", {})) - set(old.get("properties", {})) - {"ext"}
        if new:
            added[name] = new
    return added


def _keys(value, out):
    if isinstance(value, dict):
        out.update(value)
        for item in value.values():
            _keys(item, out)
    elif isinstance(value, list):
        for item in value:
            _keys(item, out)
    return out


def test_full_fields_example_uses_every_new_property():
    seen = _keys(example("full-fields"), set())
    missing = {name: sorted(props - seen) for name, props in _new_properties().items() if props - seen}
    assert missing == {}
