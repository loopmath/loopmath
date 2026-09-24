"""Conformance checks for the compact OCP v0.2 fixture set."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "spec"
GOLDEN = SPEC / "examples" / "golden"

_spec = importlib.util.spec_from_file_location(
    "ocp_conformance_golden", SPEC / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)

GOOD_FILES = {
    "artifacts.ocp.json",
    "groups-events.ocp.json",
    "minimal-run.ocp.json",
}
BROKEN_FILE = "broken-tier.ocp.json"


def test_every_golden_fixture_has_the_expected_conformance_result():
    paths = sorted(GOLDEN.glob("*.ocp.json"))
    assert {path.name for path in paths} == GOOD_FILES | {BROKEN_FILE}

    for path in paths:
        findings = conf.validate_file(path)
        errors = [finding for finding in findings if finding.level == "error"]
        if path.name == BROKEN_FILE:
            assert errors, path
            assert conf.main([str(path)]) == 1
        else:
            assert errors == [], (path, findings)
            assert conf.main([str(path)]) == 0
