"""OCP 0.4 is a strict superset of 0.3: raising `ocp` to 0.4 keeps the findings and every configuration id."""

from __future__ import annotations

import json

import pytest

from loopmath.cli import main
from loopmath.ocp import conformance
from loopmath.ocp.canonical import config_id

from ._common import EXAMPLES, V03_EXAMPLES, V03_GOLDEN, example, load

V03_FILES = sorted(V03_EXAMPLES.glob("*.ocp.json")) + sorted(V03_GOLDEN.glob("*.ocp.json"))


def _with_configuration_id():
    paths = []
    for path in sorted(EXAMPLES.rglob("*.ocp.json")):
        configuration = (load(path).get("run") or {}).get("configuration")
        if isinstance(configuration, dict) and isinstance(configuration.get("id"), str):
            paths.append(path)
    return paths


ID_FILES = _with_configuration_id()


def raised(doc, version="0.4"):
    doc["ocp"] = version
    return doc


def _findings(doc):
    return sorted((f.level, f.code, f.path) for f in conformance.validate_doc(doc))


def test_the_parametrized_sets_are_not_empty():
    assert len(V03_FILES) == 23  # 7 examples, 16 goldens
    assert len(ID_FILES) >= 24  # the 0.3 examples and goldens, and review-loop-v03


@pytest.mark.parametrize("path", V03_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_raising_a_v03_file_keeps_its_findings(path):
    doc = load(path)
    assert doc["ocp"] == "0.3"
    assert _findings(raised(load(path))) == _findings(doc)


@pytest.mark.parametrize("path", ID_FILES, ids=lambda p: str(p.relative_to(EXAMPLES)))
def test_raising_keeps_every_configuration_id(path):
    original = load(path)
    doc = raised(load(path))
    configuration = doc["run"]["configuration"]
    e190 = "E190" in {f.code for f in conformance.validate_doc(original)}
    assert ("E190" in {f.code for f in conformance.validate_doc(doc)}) == e190
    if not e190:
        # a workflow given by reference resolves through the catalog, as E190 does
        assert config_id(configuration.get("workflow"), configuration.get("settings"),
                         resolve=conformance.catalog_resolver) == configuration["id"]


@pytest.mark.parametrize("name", ["solo", "full-fields"])
def test_validate_prints_pass_for_a_raised_file(capsys, tmp_path, name):
    path = tmp_path / f"{name}.ocp.json"
    path.write_text(json.dumps(raised(example(name)), indent=2))
    code = main(["ocp", "validate", str(path)])
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("PASS") and "(ocp 0.4; 0 errors," in out


def _flow_capability_doc():
    doc = example("solo")
    doc["producer"]["capabilities"]["flow"] = False
    doc["run"]["flow"] = {"id": "lit-brief", "version": 1}
    return doc


def test_the_flow_capability_is_checked_from_v04_only():
    path = "$['producer']['capabilities']['flow']"
    at_03 = [f for f in conformance.validate_doc(_flow_capability_doc()) if f.code == "E180"]
    at_04 = [f for f in conformance.validate_doc(raised(_flow_capability_doc())) if f.code == "E180"]
    assert at_03 == []
    assert [f.path for f in at_04] == [path]


def test_e003_accepts_v04_and_lists_it_for_an_unknown_version():
    assert "E003" not in {f.code for f in conformance.validate_doc(raised(example("solo")))}
    e003 = [f for f in conformance.validate_doc(raised(example("solo"), "0.5")) if f.code == "E003"]
    assert len(e003) == 1
    assert "known: 0.1, 0.2, 0.3, 0.4" in e003[0].message
