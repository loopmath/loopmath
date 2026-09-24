"""Focused acceptance tests for the synthetic Paseo adapter fixture."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from loopmath.adapters import Selection, list_adapters, lookup
from loopmath.adapters.paseo import PaseoAdapter
from loopmath.price import load_prices, validate_prices


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "paseo"
AGENTS = FIXTURE / "agents"
CLAUDE_AGENT = "11111111-2222-4333-8444-555555555555"
CODEX_AGENT = "22222222-3333-4444-8555-666666666666"
UNKNOWN_STATUS_AGENT = "33333333-4444-4555-8666-777777777777"
MISSING_STATUS_AGENT = "44444444-5555-4666-8777-888888888888"
_spec = importlib.util.spec_from_file_location(
    "ocp_conformance_paseo", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def _diagnostics(doc: dict) -> dict[str, int]:
    records = doc["ext"]["dev.dagr.adapter.paseo"]["diagnostics"]
    return {record["reason"]: record["count"] for record in records}


def _warning_signatures(findings):
    return {
        (finding.code, finding.path, finding.message)
        for finding in findings
        if finding.level == "warning"
    }


def test_paseo_fixture_emits_conforming_graph_with_exact_counts(capsys):
    doc = PaseoAdapter().emit(Selection(stores=(AGENTS,)))
    expected_diagnostics = {
        "record_unreadable": 0,
        "record_invalid_json": 0,
        "record_json_not_object": 0,
        "record_not_agent": 0,
        "record_last_status_missing": 1,
        "record_last_status_invalid_type": 0,
        "record_last_status_unknown": 1,
        "session_id_missing_or_invalid": 0,
        "session_corroborator_missing_or_invalid": 1,
        "session_corroborator_mismatch": 0,
        "timestamp_missing": 0,
        "timestamp_invalid_type": 0,
        "timestamp_invalid_value": 0,
        "timestamp_timezone_missing": 1,
        "filter_session": 0,
        "filter_workspace": 0,
        "filter_created_at_unknown": 0,
        "filter_before_since": 0,
        "filter_after_until": 0,
        "filter_limit": 0,
    }
    captured = capsys.readouterr()
    findings = conf.validate_doc(doc)

    assert captured.out == ""
    assert captured.err == "paseo diagnostics: " + ", ".join(
        f"{reason}={count}" for reason, count in expected_diagnostics.items()
    ) + "\n"
    assert _diagnostics(doc) == expected_diagnostics
    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == set()
    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (2, 2, 0)
    assert [node["id"] for node in doc["nodes"]] == [CLAUDE_AGENT, CODEX_AGENT]
    attempts = {attempt["node"]: attempt for attempt in doc["attempts"]}
    assert attempts[CLAUDE_AGENT]["session"] == (
        "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    )
    assert "session" not in attempts[CODEX_AGENT]
    assert all(attempt["status"] == "settled_unverified" for attempt in doc["attempts"])
    assert all(attempt["outcome"]["evidence"] == "reported" for attempt in doc["attempts"])
    assert all(
        "ext" not in entity
        for key in ("nodes", "edges", "attempts")
        for entity in doc[key]
    )
    serialized = str(doc)
    assert "Invented fixture" not in serialized
    assert "systemPrompt" not in serialized
    assert "PRIVATE SYNTHETIC UNKNOWN STATUS" not in serialized


def test_paseo_default_fixture_selection_counts_non_agent_json(capsys):
    adapter = PaseoAdapter()

    with adapter.fixture_selection(FIXTURE) as selection:
        assert selection == Selection(stores=(FIXTURE,))
        doc = adapter.emit(selection)

    captured = capsys.readouterr()
    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (2, 2, 0)
    assert [node["id"] for node in doc["nodes"]] == [CLAUDE_AGENT, CODEX_AGENT]
    assert _diagnostics(doc)["record_json_not_object"] == 1
    assert "record_json_not_object=1" in captured.err
    findings = conf.validate_doc(doc)
    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == set()


def test_paseo_counts_every_safely_skipped_record_category(tmp_path):
    source = next(AGENTS.rglob(f"{CODEX_AGENT}.json"))
    (tmp_path / "valid.json").write_text(source.read_text(encoding="utf-8"))
    (tmp_path / "unreadable.json").write_bytes(b"\xff")
    (tmp_path / "invalid.json").write_text(
        "PRIVATE SYNTHETIC INVALID JSON {", encoding="utf-8"
    )
    (tmp_path / "non-object.json").write_text("[]", encoding="utf-8")
    (tmp_path / "not-agent.json").write_text(
        json.dumps({"projectId": "PRIVATE SYNTHETIC NON-AGENT"}),
        encoding="utf-8",
    )

    doc = PaseoAdapter().emit(Selection(stores=(tmp_path,)))
    diagnostics = _diagnostics(doc)

    assert [node["id"] for node in doc["nodes"]] == [CODEX_AGENT]
    assert len(doc["attempts"]) == 1
    assert diagnostics["record_unreadable"] == 1
    assert diagnostics["record_invalid_json"] == 1
    assert diagnostics["record_json_not_object"] == 1
    assert diagnostics["record_not_agent"] == 1
    serialized = json.dumps(doc, sort_keys=True)
    assert "PRIVATE SYNTHETIC INVALID JSON" not in serialized
    assert "PRIVATE SYNTHETIC NON-AGENT" not in serialized
    findings = conf.validate_doc(doc)
    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == set()


def test_paseo_fixture_maps_reported_configuration_and_workspace():
    doc = PaseoAdapter().emit(Selection(stores=(AGENTS,)))
    attempts = {attempt["node"]: attempt for attempt in doc["attempts"]}

    assert attempts[CLAUDE_AGENT]["harness"] == "claude"
    assert attempts[CLAUDE_AGENT]["model"] == {
        "raw": "claude-sonnet-5",
        "tier": "reported",
    }
    assert attempts[CLAUDE_AGENT]["effort"] == "high"
    assert attempts[CODEX_AGENT]["harness"] == "codex"
    assert attempts[CODEX_AGENT]["model"] == {
        "raw": "gpt-5.6-sol",
        "tier": "reported",
    }
    assert attempts[CODEX_AGENT]["effort"] == "xhigh"
    assert doc["run"]["workspace"] == "/synthetic/workspace"
    assert all(
        attempt["origin"] == {
            "launched_by": None,
            "workspace": "/synthetic/workspace",
            "external": None,
            "how": None,
            "tier": "reported",
            "evidence": "Paseo cwd",
        }
        for attempt in attempts.values()
    )
    assert all("cost" not in attempt for attempt in attempts.values())


def test_paseo_selection_filters_are_inclusive_and_deterministic():
    selected = PaseoAdapter().emit(
        Selection(
            stores=(AGENTS,),
            session_ids=(CODEX_AGENT,),
            workspaces=("/synthetic/workspace",),
            since="2026-08-10T19:00:00Z",
            until="2026-08-10T19:00:00Z",
            limit=1,
        )
    )
    assert selected["run"]["id"] == CODEX_AGENT
    assert [node["id"] for node in selected["nodes"]] == [CODEX_AGENT]
    assert _diagnostics(selected)["filter_session"] == 3

    time_selected = PaseoAdapter().emit(
        Selection(stores=(AGENTS,), since="2026-08-10T17:00:00Z")
    )
    assert [node["id"] for node in time_selected["nodes"]] == [CODEX_AGENT]
    assert _diagnostics(time_selected)["filter_created_at_unknown"] == 1

    files = tuple(reversed(sorted(AGENTS.rglob("*.json"))))
    first = PaseoAdapter().emit(Selection(stores=files))
    second = PaseoAdapter().emit(Selection(stores=tuple(reversed(files))))
    assert first == second
    absent = PaseoAdapter().emit(
        Selection(stores=(AGENTS,), workspaces=("/synthetic/absent",))
    )
    assert absent["nodes"] == []
    assert _diagnostics(absent)["filter_workspace"] == 4


def test_paseo_session_requires_both_matching_corroborators(tmp_path):
    source = next(AGENTS.rglob(f"{CODEX_AGENT}.json"))
    record = json.loads(source.read_text(encoding="utf-8"))
    record["id"] = "55555555-6666-4777-8888-999999999999"
    record["runtimeInfo"] = {"sessionId": "different-synthetic-session"}
    path = tmp_path / "mismatched-session.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    doc = PaseoAdapter().emit(Selection(stores=(path,)))

    assert len(doc["attempts"]) == 1
    assert "session" not in doc["attempts"][0]
    assert _diagnostics(doc)["session_corroborator_mismatch"] == 1


def test_paseo_node_identity_does_not_deduplicate_a_shared_session(tmp_path):
    source = next(AGENTS.rglob(f"{CODEX_AGENT}.json"))
    base = json.loads(source.read_text(encoding="utf-8"))
    session_id = base["persistence"]["sessionId"]
    agent_ids = (
        "77777777-8888-4999-8aaa-bbbbbbbbbbbb",
        "88888888-9999-4aaa-8bbb-cccccccccccc",
    )
    for index, agent_id in enumerate(agent_ids):
        record = json.loads(json.dumps(base))
        record["id"] = agent_id
        record["createdAt"] = f"2026-08-10T2{index}:00:00Z"
        record["updatedAt"] = f"2026-08-10T2{index}:30:00Z"
        record["runtimeInfo"] = {"sessionId": session_id}
        (tmp_path / f"agent-{index}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    doc = PaseoAdapter().emit(Selection(stores=(tmp_path,)))

    assert [node["id"] for node in doc["nodes"]] == list(agent_ids)
    assert [attempt["session"] for attempt in doc["attempts"]] == [
        session_id,
        session_id,
    ]
    assert doc["edges"] == []


def test_paseo_invalid_timestamp_is_counted_when_time_filter_excludes_it(tmp_path):
    source = next(AGENTS.rglob(f"{CODEX_AGENT}.json"))
    record = json.loads(source.read_text(encoding="utf-8"))
    record["id"] = "66666666-7777-4888-8999-000000000000"
    record["createdAt"] = "PRIVATE SYNTHETIC INVALID TIMESTAMP"
    record["runtimeInfo"] = {"sessionId": record["persistence"]["sessionId"]}
    path = tmp_path / "invalid-timestamp.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    doc = PaseoAdapter().emit(
        Selection(stores=(path,), since="2026-08-10T00:00:00Z")
    )

    assert doc["nodes"] == []
    assert _diagnostics(doc)["timestamp_invalid_value"] == 1
    assert _diagnostics(doc)["filter_created_at_unknown"] == 1
    assert "PRIVATE SYNTHETIC INVALID TIMESTAMP" not in json.dumps(doc)


def test_paseo_discovery_sessions_and_registry_are_deterministic(monkeypatch):
    import loopmath.adapters.paseo as paseo_module

    monkeypatch.setattr(paseo_module, "_DEFAULT_STORE", AGENTS)
    adapter = PaseoAdapter()
    assert adapter.discover() == (AGENTS,)
    assert [session["id"] for session in adapter.sessions()] == [
        CLAUDE_AGENT,
        CODEX_AGENT,
        UNKNOWN_STATUS_AGENT,
        MISSING_STATUS_AGENT,
    ]
    assert lookup("paseo") is PaseoAdapter
    assert list_adapters() == tuple(sorted(list_adapters()))
    assert "paseo" in list_adapters()


def test_paseo_fixture_models_have_prices_and_validator_stays_clean():
    table = load_prices()
    assert table.rate("claude-sonnet-5") is not None
    assert table.rate("gpt-5.6-sol") is not None
    result = validate_prices()
    assert result["errors"] == []
    assert result["ok"] is True
