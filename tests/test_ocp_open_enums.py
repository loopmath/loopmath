"""Open v0.2 vocabularies warn without rejecting new values."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "spec"
EXAMPLES = SPEC / "examples"

_spec = importlib.util.spec_from_file_location("ocp_open_enum_conformance", SPEC / "ocp_conformance.py")
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def load_swarm():
    return json.loads((EXAMPLES / "swarm-v02.ocp.json").read_text())


def test_only_four_string_vocabularies_remain_closed():
    schema = json.loads((SPEC / "ocp-v0.2.schema.json").read_text())
    found = {}

    def visit(value, path=()):
        if isinstance(value, dict):
            if "enum" in value:
                found[path] = value["enum"]
            for key, child in value.items():
                visit(child, path + (key,))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + (index,))

    visit(schema)

    assert found == {
        ("$defs", "evidenceTier"): ["verified", "heuristic", "reported"],
        ("$defs", "privacy", "properties", "profile"): ["metadata_only", "full"],
        ("$defs", "outcome", "properties", "result"): [
            "done", "failed", "rejected", "canceled", "settled_unverified", "lost",
        ],
        ("$defs", "attempt", "properties", "status"): [
            "queued", "working", "done", "failed", "rejected", "canceled",
            "settled_unverified", "lost",
        ],
    }


def test_nonrecommended_cause_is_one_named_warning():
    doc = load_swarm()
    doc["attempts"][0]["cause"]["type"] = "rate_limited"

    findings = conf.validate_doc(doc)
    errors = [finding for finding in findings if finding.level == "error"]
    warnings = [finding for finding in findings if finding.level == "warning"]

    assert errors == []
    assert len(warnings) == 1
    assert warnings[0].code == "W200"
    assert "cause" in warnings[0].message
    assert "rate_limited" in warnings[0].message


def test_edge_tier_remains_closed():
    doc = load_swarm()
    doc["edges"][0]["tier"] = "pretty_sure"

    findings = conf.validate_doc(doc)

    assert any(finding.level == "error" for finding in findings)


@pytest.mark.parametrize(
    "path",
    sorted(EXAMPLES.glob("*.ocp.json")),
    ids=lambda path: path.name,
)
def test_existing_example_has_no_recommended_vocabulary_warnings(path):
    findings = conf.validate_file(path)

    assert [finding for finding in findings if finding.code == "W200"] == []
