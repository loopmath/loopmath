"""Focused acceptance tests for the synthetic Orca adapter fixture."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from loopmath.adapters import Selection, lookup
from loopmath.adapters.orca import OrcaAdapter
from loopmath.price import load_prices, validate_prices


ROOT = Path(__file__).resolve().parent.parent
FIXTURE_SQL = ROOT / "tests" / "fixtures" / "adapters" / "orca" / "orchestration.sql"
_spec = importlib.util.spec_from_file_location(
    "ocp_conformance_orca", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def _diagnostics(doc: dict) -> dict[str, int]:
    records = doc["ext"]["dev.dagr.adapter.orca"]["diagnostics"]
    return {record["reason"]: record["count"] for record in records}


def _warning_signatures(findings):
    return {
        (finding.code, finding.path, finding.message)
        for finding in findings
        if finding.level == "warning"
    }


def _w200(path: str, value: str) -> tuple[str, str, str]:
    return (
        "W200",
        path,
        f"event.type value {value!r} is outside the recommended vocabulary; "
        "readers must accept it",
    )


ORCA_WARNINGS = {
    _w200("$['events'][0]['type']", "escalation"),
    _w200("$['events'][1]['type']", "worker_done"),
    _w200("$['events'][2]['type']", "merge_ready"),
    _w200("$['events'][3]['type']", "decision_gate"),
}


@pytest.fixture
def orca_store(tmp_path: Path) -> Path:
    store = tmp_path / "orchestration.db"
    connection = sqlite3.connect(store)
    try:
        connection.executescript(FIXTURE_SQL.read_text(encoding="utf-8"))
    finally:
        connection.close()
    store.chmod(0o444)
    return store


def test_orca_fixture_emits_conforming_native_id_graph_with_exact_counts(
    orca_store, capsys
):
    doc = OrcaAdapter().emit(Selection(stores=(orca_store,)))
    expected_diagnostics = {
        "start_options_missing": 0,
        "start_options_invalid_type": 0,
        "start_options_invalid_json": 1,
        "start_options_json_not_object": 0,
        "message_payload_missing": 0,
        "message_payload_invalid_type": 0,
        "message_payload_invalid_json": 1,
        "message_payload_json_not_object": 0,
    }
    captured = capsys.readouterr()
    findings = conf.validate_doc(doc)

    assert captured.out == ""
    assert captured.err == "orca diagnostics: " + ", ".join(
        f"{reason}={count}" for reason, count in expected_diagnostics.items()
    ) + "\n"
    assert _diagnostics(doc) == expected_diagnostics
    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == ORCA_WARNINGS
    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (5, 5, 3)
    assert {node["id"] for node in doc["nodes"]} == {
        "task-alpha-plan",
        "task-alpha-build",
        "task-alpha-review",
        "task-alpha-gate",
        "task-beta-solo",
    }
    assert {attempt["id"] for attempt in doc["attempts"]} == {
        "dispatch-alpha-plan",
        "dispatch-alpha-build",
        "dispatch-alpha-review",
        "dispatch-alpha-gate",
        "dispatch-beta-solo",
    }
    assert all(
        edge["kind"] == "spawn" and edge["tier"] == "verified"
        for edge in doc["edges"]
    )
    assert {event["type"] for event in doc["events"]} == {
        "escalation",
        "worker_done",
        "merge_ready",
        "decision_gate",
    }
    assert all(
        "ext" not in entity
        for key in ("nodes", "edges", "attempts", "events")
        for entity in doc[key]
    )
    gate = next(event for event in doc["events"] if event["type"] == "decision_gate")
    assert "node" not in gate and "attempt" not in gate
    serialized = json.dumps(doc, sort_keys=True)
    assert "PRIVATE SYNTHETIC MALFORMED START OPTIONS" not in serialized
    assert "PRIVATE SYNTHETIC MALFORMED MESSAGE PAYLOAD" not in serialized


def test_orca_fixture_selection_materializes_sql_and_emits_exact_graph():
    adapter = OrcaAdapter()
    materialized = None

    with adapter.fixture_selection(FIXTURE_SQL.parent) as selection:
        assert len(selection.stores) == 1
        materialized = selection.stores[0]
        assert materialized.name == "orchestration.db"
        assert materialized.is_file()
        assert materialized.parent != FIXTURE_SQL.parent

        doc = adapter.emit(selection)

        assert [group["id"] for group in doc["groups"]] == [
            "run-synthetic-alpha",
            "run-synthetic-beta",
        ]
        assert (len(doc["nodes"]), len(doc["attempts"])) == (5, 5)
        assert {
            (edge["from"], edge["to"], edge["kind"], edge["tier"])
            for edge in doc["edges"]
        } == {
            ("task-alpha-plan", "task-alpha-build", "spawn", "verified"),
            ("task-alpha-build", "task-alpha-review", "spawn", "verified"),
            ("task-alpha-review", "task-alpha-gate", "spawn", "verified"),
        }
        assert [
            (event["type"], event.get("node"), event.get("attempt"))
            for event in doc["events"]
        ] == [
            ("escalation", "task-alpha-build", "dispatch-alpha-build"),
            ("worker_done", "task-alpha-build", "dispatch-alpha-build"),
            ("merge_ready", "task-alpha-review", "dispatch-alpha-review"),
            ("decision_gate", None, None),
        ]
        findings = conf.validate_doc(doc)
        assert [finding for finding in findings if finding.level == "error"] == []
        assert _warning_signatures(findings) == ORCA_WARNINGS

    assert materialized is not None
    assert not materialized.exists()


def test_orca_direct_selected_store_failures_are_actionable(tmp_path):
    adapter = OrcaAdapter()

    with pytest.raises(
        ValueError,
        match=r"Orca selected store .* is a directory;.*fixture_selection\(\)",
    ):
        adapter.emit(Selection(stores=(FIXTURE_SQL.parent,)))

    missing = tmp_path / "missing.db"
    with pytest.raises(
        ValueError,
        match=r"Orca selected store .* does not exist or is not a regular file",
    ):
        adapter.emit(Selection(stores=(missing,)))

    unreadable = tmp_path / "not-a-database.db"
    unreadable.write_text("synthetic non-SQLite fixture", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"Orca selected store .* is not a readable orchestration SQLite database",
    ):
        adapter.emit(Selection(stores=(unreadable,)))


def test_orca_fixture_maps_only_reported_effective_launch_configuration(orca_store):
    doc = OrcaAdapter().emit(
        Selection(stores=(orca_store,), session_ids=("run-synthetic-alpha",))
    )
    attempts = {attempt["id"]: attempt for attempt in doc["attempts"]}

    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (4, 4, 3)
    assert attempts["dispatch-alpha-build"]["harness"] == "claude"
    assert attempts["dispatch-alpha-build"]["model"] == {
        "raw": "claude-opus-5",
        "tier": "reported",
    }
    assert attempts["dispatch-alpha-build"]["effort"] == "max"
    assert "model" not in attempts["dispatch-alpha-gate"]
    assert "effort" not in attempts["dispatch-alpha-gate"]
    assert all(
        "session" not in attempt and "origin" not in attempt
        for attempt in attempts.values()
    )


def test_orca_sessions_filters_and_registry_are_deterministic(orca_store, monkeypatch):
    import loopmath.adapters.orca as orca_module

    monkeypatch.setattr(orca_module, "_DEFAULT_STORE", orca_store)
    adapter = OrcaAdapter()
    assert [session["id"] for session in adapter.sessions()] == [
        "run-synthetic-alpha",
        "run-synthetic-beta",
    ]
    assert lookup("orca") is OrcaAdapter

    selected = adapter.emit(
        Selection(
            stores=(orca_store,),
            workspaces=("worktree-synthetic-beta",),
            since="2026-08-11T00:00:00Z",
            until="2026-08-11T23:59:59Z",
            limit=1,
        )
    )
    assert selected["run"]["id"] == "run-synthetic-beta"
    assert [node["id"] for node in selected["nodes"]] == ["task-beta-solo"]


def test_orca_fixture_models_need_no_new_price_rows():
    table = load_prices()
    assert table.rate("gpt-5.6-sol") is not None
    assert table.rate("claude-opus-5") is not None
    result = validate_prices()
    assert result["errors"] == []
    assert result["ok"] is True
