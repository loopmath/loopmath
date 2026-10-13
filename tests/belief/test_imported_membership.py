"""Recover only absent membership on recognized imported documents, without changing checks."""
import copy

import pytest

from loopmath.belief.design import parse_run
from loopmath.ocp.migrate import migrate_doc
from tests.ocp._common import example


def old_import(shape="solo"):
    doc = example(shape)
    cfg = doc["run"]["configuration"]
    cfg["node_vertex"] = {n["id"]: n.pop("vertex") for n in doc["nodes"] if "vertex" in n}
    for attempt in doc["attempts"]:
        attempt.pop("vertex", None)
        attempt.pop("role", None)
    doc["run"].setdefault("ext", {})["dev.loopmath.migration"] = {"from": "0.2"}
    return doc


@pytest.mark.parametrize("shape", ["solo", "best_of_n", "plan_implement_review", "swarm"])
def test_existing_import_uses_its_saved_map_without_mutating_it(shape):
    doc = old_import(shape)
    before = copy.deepcopy(doc)
    parsed = parse_run(doc)
    assert len(parsed.attempts) == len(example(shape)["attempts"])
    assert not parsed.dropped
    assert doc == before


@pytest.mark.parametrize("case", ["absent", "not-object", "not-string", "unknown-piece", "missing-node",
                                 "no-provenance", "bad-provenance", "attempt-invalid", "node-invalid",
                                 "attempt-empty", "node-empty", "attempt-null", "node-null"])
def test_bad_or_explicit_membership_is_never_repaired(case):
    doc = old_import()
    cfg, attempt, node = doc["run"]["configuration"], doc["attempts"][0], doc["nodes"][0]
    if case == "absent":
        cfg.pop("node_vertex")
    elif case == "not-object":
        cfg["node_vertex"] = ["implement"]
    elif case == "not-string":
        cfg["node_vertex"][node["id"]] = ["implement"]
    elif case == "unknown-piece":
        cfg["node_vertex"][node["id"]] = "missing"
    elif case == "missing-node":
        doc["nodes"] = []
    elif case in ("no-provenance", "bad-provenance"):
        doc["producer"]["name"] = "unknown-producer"
        doc["run"]["ext"] = {} if case == "no-provenance" else {"dev.loopmath.migration": {"from": "unknown"}}
        cfg["source"] = "habit"
    else:
        where, value = case.split("-")
        (attempt if where == "attempt" else node)["vertex"] = {"invalid": "missing", "empty": "", "null": None}[value]
    parsed = parse_run(doc)
    assert parsed.attempts == []
    assert parsed.dropped == {"attempt outside the workflow": 1}
    assert parse_run(migrate_doc(doc)).dropped == parsed.dropped


@pytest.mark.parametrize("where", ["attempt", "node"])
def test_explicit_valid_membership_wins_over_a_conflicting_map(where):
    doc = old_import()
    doc["run"]["configuration"]["node_vertex"]["implement"] = "missing"
    doc["attempts" if where == "attempt" else "nodes"][0]["vertex"] = "implement"
    assert len(parse_run(doc).attempts) == 1


@pytest.mark.parametrize("migrate", [False, True], ids=["old-store", "new-migration"])
def test_model_free_gate_keeps_check_accounting(migrate):
    doc = old_import()
    doc["nodes"].append({"id": "tests", "kind": "gate", "gate": {"rule": "tests"}})
    doc["attempts"].append({"id": "tests.a1", "node": "tests", "status": "done", "cost": {"usd": 0, "basis": "measured"}})
    doc["run"]["configuration"]["node_vertex"]["tests"] = "implement"
    if migrate:
        doc = migrate_doc(doc)
    before = copy.deepcopy(doc)
    parsed = parse_run(doc)
    assert parsed.checks == {"tests": 1}
    assert len(parsed.attempts) == 1
    assert parsed.dropped == {}
    assert doc == before


def test_custom_piece_round_model_override_and_token_only_attempt():
    doc = old_import()
    cfg = doc["run"]["configuration"]
    cfg["workflow"]["pieces"][0]["id"] = "build-widget"
    cfg["settings"]["build-widget"] = cfg["settings"].pop("implement")
    cfg["node_vertex"]["implement"] = "build-widget"
    a = doc["attempts"][0]
    a.update(round=3, model={"raw": "unpriced-synthetic-model"}, cost={"input_tokens": 100, "basis": "measured"})
    parsed = parse_run(doc)
    assert len(parsed.attempts) == 1 and parsed.dropped == {}
    observed = parsed.attempts[0]
    assert observed.piece == "build-widget" and observed.round == 3
    assert observed.usd is None and observed.tokens == 100
    assert observed.setting.model == "unpriced-synthetic-model"


def test_migration_materializes_map_before_derived_role_and_is_idempotent():
    doc = old_import()
    doc["ocp"] = "0.2"
    config = doc["run"].pop("configuration")
    doc["attempts"][0]["role"] = {"value": "implementer", "tier": "reported"}
    before = copy.deepcopy(doc)
    migrated = migrate_doc(doc, infer=lambda _: (config, 0.9))
    assert migrated["nodes"][0]["vertex"] == "implement"
    assert migrated["attempts"][0]["vertex"] == "implement"
    assert len(parse_run(migrated).attempts) == 1
    assert migrated["run"]["configuration"]["id"] == config["id"]
    assert migrate_doc(migrated) == migrated and doc == before


@pytest.mark.parametrize("producer", ["loopmath/adapter-pi", "loopmath/adapter-opencode", "dagr/adapter-pi"])
def test_recognized_adapter_provenance_and_legacy_extensions_are_preserved(producer):
    doc = old_import()
    doc["run"]["ext"] = {"dev.dagr.adapter.pi": {"unknown": "keep this"}}
    doc["producer"]["name"] = producer
    before = copy.deepcopy(doc)
    assert len(parse_run(doc).attempts) == 1
    migrated = migrate_doc(doc)
    assert migrated["run"]["ext"] == before["run"]["ext"] and doc == before
    assert "dev.loopmath.adapter.pi" not in migrated["run"]["ext"]
