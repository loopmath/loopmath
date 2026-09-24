"""The review of 9bd45a3 as tests: each case of the reviewer's repro, with the outcome
the fix gives, plus the D48 boundaries."""

from __future__ import annotations

import copy
import json

import pytest

from loopmath.cli import main
from loopmath.ocp import conformance

from ._common import EXAMPLES, codes, example, load


def errors(doc):
    return [f for f in conformance.validate_doc(doc) if f.level == "error"]


def allocated(doc, *, tier="heuristic", producer="loopmath", logmatch=True):
    doc = copy.deepcopy(doc)
    doc["producer"]["name"] = producer
    cost = doc["attempts"][0]["cost"]
    cost["basis"] = "allocated"
    if logmatch:
        cost.setdefault("ext", {})["dev.loopmath.logmatch"] = {"tier": tier}
    return doc


# Finding 3 and D48 ----------------------------------------------------------------------------------

def test_repro_d27_allocated_cost_is_accepted_in_v03():
    base = example("solo")
    assert errors(base) == []
    assert errors(allocated(base)) == []


@pytest.mark.parametrize("record", [
    {"tier": "heuristic"},
    {"tier": "heuristic", "clip": {"from": "2026-09-23T10:00:00-07:00", "to": "2026-09-23T10:20:00-07:00"},
     "requests": 3},
], ids=["shared-tier-only", "full-record"])
def test_d56_the_logmatch_record_needs_only_its_tier(record):
    """D56: a shared record carries exactly cost.ext = {logmatch: {tier: heuristic}}; the store's has more."""
    doc = allocated(example("solo"), logmatch=False)
    doc["attempts"][0]["cost"]["ext"] = {"dev.loopmath.logmatch": record}
    assert errors(doc) == []


@pytest.mark.parametrize("record", [
    {"tier": "verified", "shared_session": {"session": "s-1", "attempts": ["a1", "a2"]}},
    {"tier": "verified", "shared_session": True},
], ids=["store", "shared"])
def test_d88_a_split_shared_session_may_be_allocated_at_any_tier(record):
    """D88: a session several attempts name is split among them; the store keeps an object, share keeps true."""
    doc = allocated(example("solo"), logmatch=False)
    doc["attempts"][0]["cost"]["ext"] = {"dev.loopmath.logmatch": record}
    assert errors(doc) == []


@pytest.mark.parametrize("shared", [False, None, {}, "s-1"], ids=["false", "null", "empty", "string"])
def test_d88_a_verified_allocation_needs_a_shared_session(shared):
    doc = allocated(example("solo"), tier="verified")
    doc["attempts"][0]["cost"]["ext"]["dev.loopmath.logmatch"]["shared_session"] = shared
    assert codes(errors(doc)) == ["E171"]


def test_d88_stands_only_for_loopmath_v03():
    record = {"tier": "verified", "shared_session": True}
    doc = allocated(example("solo"), producer="dagr", logmatch=False)
    doc["attempts"][0]["cost"]["ext"] = {"dev.loopmath.logmatch": record}
    assert codes(errors(doc)) == ["E171"]
    doc = load(EXAMPLES / "swarm-v02.ocp.json")
    doc["attempts"][0]["cost"].update(basis="allocated", ext={"dev.loopmath.logmatch": record})
    assert "E171" in codes(errors(doc))


@pytest.mark.parametrize("kwargs", [
    {"logmatch": False},
    {"tier": "reported"},
    {"tier": "verified"},
    {"producer": "dagr"},
    {"producer": "dagr-graph"},
], ids=["no-logmatch", "reported", "verified", "dagr", "dagr-suffixed"])
def test_e171_stands_without_the_d48_evidence(kwargs):
    assert codes(errors(allocated(example("solo"), **kwargs))) == ["E171"]


def test_e171_stands_for_v02():
    doc = load(EXAMPLES / "swarm-v02.ocp.json")
    assert doc["producer"]["name"] == "loopmath"
    cost = doc["attempts"][0]["cost"]
    cost["basis"] = "allocated"
    cost["ext"] = {"dev.loopmath.logmatch": {"tier": "heuristic"}}
    assert "E171" in codes(errors(doc))


def test_other_producers_may_allocate_freely():
    assert errors(allocated(example("solo"), producer="example-orchestrator", logmatch=False)) == []


# Finding 1: malformed values give findings, never an exception ---------------------------------------

def _edit(case):
    doc = example("solo")
    if case == "verdict_object":
        doc["run"]["signals"][0]["value"] = {"bad": True}
    elif case == "edge_endpoint_list":
        doc["run"]["configuration"]["workflow"]["edges"][0][0] = []
    elif case == "repair_target_object":
        doc["run"]["configuration"]["workflow"]["control"]["repair"] = {"implement": {}}
    else:
        doc["run"]["acceptance_rule"]["requires"] = [{}]
    return doc


@pytest.mark.parametrize("case", ["verdict_object", "edge_endpoint_list", "repair_target_object",
                                  "requires_object"])
def test_repro_malformed_values_are_reported(case, tmp_path, capsys):
    doc = _edit(case)
    found = errors(doc)
    assert "E010" in codes(found)  # the schema finding is kept
    path = tmp_path / "bad.ocp.json"
    path.write_text(json.dumps(doc))
    assert main(["ocp", "validate", "--json", str(path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False and payload["files"][0]["errors"] == len(found)


# Finding 2: migrate never writes or prints a nonconforming document ---------------------------------

def _invalid_source(tmp_path):
    doc = example("solo")
    doc["run"]["acceptance_rule"].pop("name")
    src = tmp_path / "bad.json"
    src.write_text(json.dumps(doc))
    return src


def test_repro_invalid_migration_writes_nothing(tmp_path, capsys):
    src = _invalid_source(tmp_path)
    out = tmp_path / "out"
    assert main(["ocp", "migrate", str(src), "--out", str(out), "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False and result["files"][0]["out"] is None
    assert "E010" in {f["code"] for f in result["files"][0]["findings"]}
    assert not (out / "bad.ocp.json").exists()


def test_invalid_migration_keeps_an_existing_output(tmp_path, capsys):
    src = _invalid_source(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "bad.ocp.json").write_text("previous\n")
    assert main(["ocp", "migrate", str(src), "--out", str(out)]) == 1
    printed = capsys.readouterr().out
    assert printed.startswith("FAIL") and "nothing written" in printed and "E010" in printed
    assert (out / "bad.ocp.json").read_text() == "previous\n"


def test_invalid_migration_prints_no_document(tmp_path, capsys):
    src = _invalid_source(tmp_path)
    assert main(["ocp", "migrate", str(src)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "nothing written" in captured.err and "E010" in captured.err
    assert main(["ocp", "migrate", "--json", str(src)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert "doc" not in result["files"][0]
