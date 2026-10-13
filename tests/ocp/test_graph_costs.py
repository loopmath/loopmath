"""Graph contract conversion preserves measured data by task and attempt identity."""
import copy

import pytest

from loopmath.graph.runfile import to_runfile
from loopmath.graph.schema import Graph, GraphNode
from loopmath.ocp import conformance, contractv3
from loopmath.ocp.migrate import MigrationError, migrate_doc

LEGACY = "dev.dagr.graph"
CURRENT = "dev.loopmath.graph"


def graph_export():
    graph = Graph(nodes=[GraphNode(
        id=name, harness="claude-code", source="top", session_path="/synthetic/missing.jsonl",
        model="claude-sonnet-5", model_tier="reported", effort="high", workspace="example/widget",
        ts="2024-01-01T00:00:00Z", wall_s=60, role="implementer", role_tier="reported",
        tokens=tokens, usd=usd) for name, tokens, usd in (
            ("one", {"in": 100, "out": 30, "cache_read": 40, "cache_write": 12,
                     "cache_creation_5m_tokens": 5, "cache_creation_1h_tokens": 7}, 0.006),
            ("two", {"in": 200, "out": 60, "cache_read": 0, "cache_write": 0}, None),
            ("unknown", None, None))])
    return to_runfile(graph, finish={}, signals={})


@pytest.mark.parametrize("namespace", [LEGACY, CURRENT])
def test_graph_costs_and_full_settings_survive_by_identity(namespace):
    source = graph_export()
    if namespace == CURRENT:
        source["ext"][CURRENT] = source["ext"].pop(LEGACY)
        for task in source["tasks"]:
            task["ext"][CURRENT] = task["ext"].pop(LEGACY)
    rows = source["ext"][namespace]["costs"]
    rows[0]["cost"].update(requests=3, reasoning_tokens=10, ext={"example.cost": {"tier": "reported"}})
    rows.reverse()
    original = copy.deepcopy(source)
    converted = contractv3.convert(source)
    migrated = migrate_doc(source)
    for doc in (converted, migrated):
        attempts = {(a["node"], a["id"]): a for a in doc["attempts"]}
        for row in rows:
            assert attempts[row["task"], row["attempt"]]["cost"] == row["cost"]
        for task in source["tasks"]:
            a = attempts[task["id"], task["attempts"][0]["id"]]
            meta = task["ext"][namespace]
            assert a["model"] == meta["model"]
            assert a["harness"] == meta["harness"] and a["effort"] == meta["effort"]
        assert "cost" not in next(a for a in doc["attempts"] if a["node"] == "unknown")
    assert source == original
    findings = conformance.validate_doc(migrated)
    assert not [f for f in findings if f.level == "error" or f.code == "W180"], findings


def test_identical_rows_are_not_summed_and_zero_stays_zero():
    source = graph_export()
    rows = source["ext"][LEGACY]["costs"]
    rows[0]["cost"]["usd"] = 0
    rows.append(copy.deepcopy(rows[0]))
    converted = contractv3.convert(source)
    assert converted["attempts"][0]["cost"] == rows[0]["cost"]
    assert "usd" not in converted["attempts"][1]["cost"]


@pytest.mark.parametrize("case", ["conflict", "unknown-attempt", "wrong-task", "bad-cost", "negative",
                                 "boolean", "fractional-tokens", "nan", "namespaces", "ambiguous-setting"])
def test_ambiguous_or_malformed_graph_evidence_is_rejected(case):
    source = graph_export()
    rows = source["ext"][LEGACY]["costs"]
    if case == "conflict":
        rows.append({**rows[0], "cost": {"usd": 99}})
    elif case == "unknown-attempt":
        rows[0]["attempt"] = "unknown-attempt"
    elif case == "wrong-task":
        rows[0]["task"] = "two"
    elif case == "bad-cost":
        rows[0]["cost"] = "six cents"
    elif case in ("negative", "boolean", "fractional-tokens", "nan"):
        rows[0]["cost"]["input_tokens"] = {"negative": -1, "boolean": True,
                                            "fractional-tokens": 1.2, "nan": float("nan")}[case]
    elif case == "namespaces":
        source["ext"][CURRENT] = {**copy.deepcopy(source["ext"][LEGACY]), "costs": []}
    else:
        task = source["tasks"][0]
        task["attempts"].append({**copy.deepcopy(task["attempts"][0]), "id": "second-attempt"})
    with pytest.raises(MigrationError, match="graph"):
        migrate_doc(source)


def test_plain_contract_without_graph_extension_keeps_its_conversion():
    source = graph_export()
    source.pop("ext")
    for task in source["tasks"]:
        task.pop("ext")
    converted = contractv3.convert(source)
    assert all("cost" not in a and "harness" not in a for a in converted["attempts"])
    assert "capabilities" not in converted["producer"]
