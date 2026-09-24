"""Focused acceptance tests for the synthetic Atrium adapter fixture."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from loopmath.adapters import Selection, list_adapters, lookup
from loopmath.adapters.atrium import AtriumAdapter
from loopmath.price import validate_prices


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "atrium"
_spec = importlib.util.spec_from_file_location(
    "ocp_conformance_atrium", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def _diagnostics(doc: dict) -> dict[str, int]:
    records = doc["ext"]["dev.dagr.adapter.atrium"]["diagnostics"]
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


ATRIUM_WARNINGS = {
    _w200("$['events'][0]['type']", "task-run-segment"),
    _w200("$['events'][1]['type']", "session-end"),
    _w200("$['events'][2]['type']", "agent-message"),
    _w200("$['events'][3]['type']", "subagent-stop"),
}


@pytest.fixture
def atrium_store(tmp_path: Path) -> Path:
    for name in ("tasks", "timeline"):
        store = tmp_path / f"{name}.db"
        connection = sqlite3.connect(store)
        try:
            connection.executescript((FIXTURE / f"{name}.sql").read_text(encoding="utf-8"))
        finally:
            connection.close()
        store.chmod(0o444)
    return tmp_path


def test_atrium_fixture_emits_conforming_relation_events_with_exact_counts(
    atrium_store, capsys
):
    doc = AtriumAdapter().emit(Selection(stores=(atrium_store,)))
    expected_diagnostics = {
        "timeline_kind_unsupported": 1,
        "segment_adapter_session_id_missing": 1,
        "segment_adapter_session_id_invalid_type": 0,
        "filter_session": 0,
        "filter_workspace": 0,
        "filter_before_since": 0,
        "filter_after_until": 0,
        "filter_limit": 0,
    }
    captured = capsys.readouterr()
    findings = conf.validate_doc(doc)

    assert captured.out == ""
    assert captured.err == "atrium diagnostics: " + ", ".join(
        f"{reason}={count}" for reason, count in expected_diagnostics.items()
    ) + "\n"
    assert _diagnostics(doc) == expected_diagnostics
    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == ATRIUM_WARNINGS
    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (0, 0, 0)
    assert len(doc["events"]) == 4
    assert {event["type"] for event in doc["events"]} == {
        "agent-message",
        "session-end",
        "subagent-stop",
        "task-run-segment",
    }
    payloads = [event["ext"]["dev.dagr.adapter.atrium"] for event in doc["events"]]
    assert {
        payload["related_kind"] for payload in payloads if "related_kind" in payload
    } == {
        "session",
        "target-pane",
    }
    assert all(payload["tier"] == "verified" for payload in payloads)
    subagent = next(
        event["ext"]["dev.dagr.adapter.atrium"]
        for event in doc["events"]
        if event["type"] == "subagent-stop"
    )
    assert "related_kind" not in subagent
    assert "related_id" not in subagent


def test_atrium_fixture_selection_materializes_sql_and_emits_exact_events():
    adapter = AtriumAdapter()
    materialized: tuple[Path, ...] = ()

    with adapter.fixture_selection(FIXTURE) as selection:
        materialized = selection.stores
        assert [store.name for store in materialized] == [
            "tasks.db",
            "timeline.db",
        ]
        assert all(store.is_file() for store in materialized)
        assert all(store.parent != FIXTURE for store in materialized)

        doc = adapter.emit(selection)

        assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (
            0,
            0,
            0,
        )
        assert [event["type"] for event in doc["events"]] == [
            "task-run-segment",
            "session-end",
            "agent-message",
            "subagent-stop",
        ]
        findings = conf.validate_doc(doc)
        assert [finding for finding in findings if finding.level == "error"] == []
        assert _warning_signatures(findings) == ATRIUM_WARNINGS

    assert materialized
    assert all(not store.exists() for store in materialized)


def test_atrium_direct_selected_store_failures_are_actionable(tmp_path):
    adapter = AtriumAdapter()

    with pytest.raises(
        ValueError,
        match=r"Atrium selected store directory .*fixture_selection\(\)",
    ):
        adapter.emit(Selection(stores=(FIXTURE,)))

    missing = tmp_path / "missing.db"
    with pytest.raises(
        ValueError,
        match=r"Atrium selected store .* does not exist or is not a regular file",
    ):
        adapter.emit(Selection(stores=(missing,)))

    unreadable = tmp_path / "not-a-database.db"
    unreadable.write_text("synthetic non-SQLite fixture", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=r"Atrium selected store .* is not a readable SQLite database",
    ):
        adapter.emit(Selection(stores=(unreadable,)))


def test_atrium_fixture_omits_content_and_does_not_infer_acceptance(atrium_store):
    doc = AtriumAdapter().emit(Selection(stores=(atrium_store,)))
    content_document = {
        **doc,
        "producer": {
            key: value
            for key, value in doc["producer"].items()
            if key != "capabilities"
        },
    }
    serialized = json.dumps(content_document, sort_keys=True)

    assert "PRIVATE SYNTHETIC" not in serialized
    assert "invented-harness" not in serialized
    assert "invented-end" not in serialized
    assert "PRIVATE SYNTHETIC UNSUPPORTED KIND" not in serialized
    assert "PRIVATE SYNTHETIC OMITTED SEGMENT" not in serialized
    assert all("node" not in event and "attempt" not in event for event in doc["events"])
    assert all("outcome" not in attempt for attempt in doc["attempts"])
    assert "model" not in serialized
    assert all("cost" not in attempt for attempt in doc["attempts"])
    assert "token" not in serialized


def test_atrium_selection_and_emit_are_deterministic(atrium_store):
    tasks = atrium_store / "tasks.db"
    timeline = atrium_store / "timeline.db"
    first = AtriumAdapter().emit(Selection(stores=(timeline, tasks)))
    second = AtriumAdapter().emit(Selection(stores=(tasks, timeline)))
    assert first == second

    selected = AtriumAdapter().emit(
        Selection(
            stores=(atrium_store,),
            session_ids=("session-synthetic-alpha",),
            workspaces=("workspace-synthetic-alpha",),
            since="2026-08-14T10:01:00Z",
            until="2026-08-14T10:02:00Z",
            limit=1,
        )
    )
    assert [event["type"] for event in selected["events"]] == [
        "agent-message",
        "subagent-stop",
    ]
    assert _diagnostics(selected)["filter_before_since"] == 2
    absent = AtriumAdapter().emit(
        Selection(stores=(atrium_store,), workspaces=("workspace-synthetic-absent",))
    )
    assert absent["events"] == []
    assert _diagnostics(absent)["filter_workspace"] == 4


def test_atrium_discovery_sessions_and_registry_are_deterministic(
    atrium_store, monkeypatch
):
    import loopmath.adapters.atrium as atrium_module

    monkeypatch.setattr(atrium_module, "_DEFAULT_ROOT", atrium_store)
    adapter = AtriumAdapter()
    assert adapter.discover() == (
        atrium_store / "tasks.db",
        atrium_store / "timeline.db",
    )
    sessions = adapter.sessions()
    assert sessions == adapter.sessions()
    assert [session["id"] for session in sessions] == [
        "workspace-synthetic-alpha"
    ]
    assert lookup("atrium") is AtriumAdapter
    assert list_adapters() == tuple(sorted(list_adapters()))
    assert "atrium" in list_adapters()


def test_atrium_needs_no_price_rows_and_validator_stays_clean(atrium_store):
    doc = AtriumAdapter().emit(Selection(stores=(atrium_store,)))
    assert doc["attempts"] == []
    result = validate_prices()
    assert result["errors"] == []
    assert result["ok"] is True
