"""Pi adapter tests use only invented JSONL sessions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import loopmath.adapters.pi as pi_module
from loopmath.adapters import Selection, lookup
from loopmath.adapters.pi import PiAdapter
from loopmath.price import load_prices, validate_prices


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "pi"

_spec = importlib.util.spec_from_file_location(
    "pi_ocp_conformance", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def _emit(selection: Selection | None = None) -> dict:
    return PiAdapter().emit(selection or Selection(stores=(FIXTURE,)))


def _attempt(doc: dict, session_id: str) -> dict:
    return next(attempt for attempt in doc["attempts"] if attempt["session"] == session_id)


def _warning_signatures(findings):
    return {
        (finding.code, finding.path, finding.message)
        for finding in findings
        if finding.level == "warning"
    }


def test_pi_fixture_emits_conforming_exact_graph_counts_and_no_relations():
    doc = _emit()
    findings = conf.validate_doc(doc)

    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == set()
    assert len(doc["nodes"]) == 3
    assert len(doc["attempts"]) == 3
    assert len(doc["edges"]) == 0
    assert len(doc["groups"]) == 3
    assert len(doc["events"]) == 3
    assert doc["artifacts"] == []
    assert doc["run"]["ext"]["dev.dagr.adapter.pi"]["selection"] == {
        "files_seen": 3,
        "sessions_parsed": 3,
        "sessions_selected": 3,
        "retained_unknown_workspace": 0,
        "retained_unknown_timestamp": 0,
        "omitted": {
            "unreadable_or_missing_id": 0,
            "unsupported_or_missing_version": 0,
            "duplicate_session_id": 0,
            "id_filter": 0,
            "workspace_filter": 0,
            "time_filter": 0,
            "limit": 0,
        },
    }


def test_pi_without_terminal_signal_remains_working():
    doc = _emit()
    assert {node["state"] for node in doc["nodes"]} == {"working"}
    assert {attempt["status"] for attempt in doc["attempts"]} == {"working"}
    assert all("outcome" not in attempt for attempt in doc["attempts"])
    assert all("ended_at" not in attempt for attempt in doc["attempts"])
    assert "ended_at" not in doc["run"]


def test_pi_maps_per_message_usage_cost_model_and_effort():
    doc = _emit()
    alpha = _attempt(doc, "pi-synthetic-alpha")
    assert alpha["model"] == {
        "raw": "gpt-5.5",
        "id": "gpt-5.5",
        "provider": "openai-codex",
        "tier": "verified",
    }
    assert alpha["effort"] == "high"
    assert alpha["cost"] == {
        "input_tokens": 300,
        "cached_input_tokens": 120,
        "cache_creation_tokens": 0,
        "output_tokens": 90,
        "reasoning_tokens": 18,
        "requests": 2,
        "usd": pytest.approx(0.00213),
        "basis": "measured",
    }
    alpha_ext = alpha["ext"]["dev.dagr.adapter.pi"]
    assert alpha_ext["incomplete_usage_records"] == 0
    assert alpha_ext["malformed_records"] == 0
    assert [record["id"] for record in alpha_ext["usage_records"]] == [
        "alpha-assistant-1",
        "alpha-assistant-2",
    ]
    assert alpha_ext["usage_records"][0]["tokens"] == {
        "input": 100,
        "cache_read": 20,
        "cache_write": 0,
        "output": 50,
        "reasoning": 10,
        "total": 170,
    }
    assert alpha_ext["usage_records"][0]["usd"]["total"] == pytest.approx(0.001005)

    beta = _attempt(doc, "pi-synthetic-beta")
    assert beta["model"]["id"] == "claude-sonnet-5"
    assert beta["model"]["provider"] == "anthropic"
    assert beta["model"]["tier"] == "reported"
    assert beta["effort"] == "medium"
    assert beta["cost"] == {
        "input_tokens": 120,
        "cached_input_tokens": 30,
        "cache_creation_tokens": 60,
        "cache_creation_5m_tokens": 45,
        "cache_creation_1h_tokens": 15,
        "output_tokens": 40,
        "requests": 2,
        "usd": pytest.approx(0.001329),
        "basis": "measured",
    }
    beta_records = beta["ext"]["dev.dagr.adapter.pi"]["usage_records"]
    assert beta["ext"]["dev.dagr.adapter.pi"]["incomplete_usage_records"] == 1
    assert "reasoning_tokens" not in beta["cost"]
    assert "reasoning" not in beta_records[1]["tokens"]
    assert [record["kind"] for record in beta_records] == [
        "assistant_message",
        "compaction",
    ]
    assert beta_records[0]["model_tier"] == "verified"
    assert beta_records[1]["model_tier"] == "reported"
    assert all(record["effort_tier"] == "reported" for record in beta_records)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (Selection(session_ids=("pi-synthetic-alpha",)), {"pi-synthetic-alpha"}),
        (Selection(workspaces=("/synthetic/workspace/beta",)), {"pi-synthetic-beta"}),
        (
            Selection(since="2026-08-31T11:00:00.000Z"),
            {"pi-synthetic-beta", "pi-synthetic-gamma"},
        ),
        (
            Selection(until="2026-08-30T10:00:00.000Z"),
            {"pi-synthetic-alpha"},
        ),
        (Selection(limit=1), {"pi-synthetic-alpha"}),
    ],
)
def test_pi_respects_each_common_session_filter(selection: Selection, expected: set[str]):
    selection = Selection(
        stores=(FIXTURE,),
        session_ids=selection.session_ids,
        workspaces=selection.workspaces,
        since=selection.since,
        until=selection.until,
        limit=selection.limit,
    )
    doc = _emit(selection)
    assert {attempt["session"] for attempt in doc["attempts"]} == expected
    assert len(doc["nodes"]) == len(expected)
    assert len(doc["attempts"]) == len(expected)
    assert doc["edges"] == []


def test_pi_store_selection_accepts_one_jsonl_file_and_replaces_discovery():
    beta_file = FIXTURE / "project-beta" / "session-beta.jsonl"
    doc = _emit(Selection(stores=(beta_file,)))
    assert [attempt["session"] for attempt in doc["attempts"]] == [
        "pi-synthetic-beta"
    ]
    assert doc["run"]["ext"]["dev.dagr.adapter.pi"]["selection"]["files_seen"] == 1


def test_pi_discover_sessions_and_registry_are_deterministic(monkeypatch):
    monkeypatch.setattr(pi_module, "_DEFAULT_STORE", FIXTURE)
    adapter = PiAdapter()

    assert adapter.discover() == (FIXTURE,)
    assert [session["id"] for session in adapter.sessions()] == [
        "pi-synthetic-alpha",
        "pi-synthetic-beta",
        "pi-synthetic-gamma",
    ]
    assert lookup("pi") is PiAdapter


def test_pi_accepts_session_version_2_and_reports_fixture_accounting():
    doc = _emit()
    assert "pi-synthetic-gamma" in {
        attempt["session"] for attempt in doc["attempts"]
    }
    selection = doc["run"]["ext"]["dev.dagr.adapter.pi"]["selection"]
    assert selection["files_seen"] == 3
    assert selection["sessions_parsed"] == 3
    assert selection["sessions_selected"] == 3
    assert selection["omitted"]["unsupported_or_missing_version"] == 0


@pytest.mark.parametrize("version", [None, 0, 4, "2", True])
def test_pi_counts_unsupported_or_missing_session_versions(tmp_path, version):
    header = {
        "type": "session",
        "id": "pi-invalid-version",
        "timestamp": "2026-09-01T00:00:00.000Z",
        "cwd": "/synthetic/workspace/invalid",
    }
    if version is not None:
        header["version"] = version
    path = tmp_path / "invalid-version.jsonl"
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")

    doc = _emit(Selection(stores=(path,)))
    assert doc["nodes"] == []
    selection = doc["run"]["ext"]["dev.dagr.adapter.pi"]["selection"]
    assert selection["files_seen"] == 1
    assert selection["sessions_parsed"] == 0
    assert selection["sessions_selected"] == 0
    assert selection["omitted"]["unsupported_or_missing_version"] == 1


def test_pi_workspace_filter_retains_session_with_unknown_workspace(tmp_path):
    path = tmp_path / "unknown-workspace.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 2,
                "id": "pi-unknown-workspace",
                "timestamp": "2026-09-01T00:00:00.000Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    doc = _emit(
        Selection(stores=(path,), workspaces=("/synthetic/workspace/requested",))
    )
    assert [attempt["session"] for attempt in doc["attempts"]] == [
        "pi-unknown-workspace"
    ]
    selection = doc["run"]["ext"]["dev.dagr.adapter.pi"]["selection"]
    assert selection["retained_unknown_workspace"] == 1
    assert selection["omitted"]["workspace_filter"] == 0


def test_pi_time_filter_retains_session_with_unknown_start_timestamp(tmp_path):
    path = tmp_path / "unknown-timestamp.jsonl"
    path.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 2,
                "id": "pi-unknown-timestamp",
                "cwd": "/synthetic/workspace/unknown-time",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    doc = _emit(
        Selection(
            stores=(path,),
            since="2026-09-01T00:00:00.000Z",
            until="2026-09-02T00:00:00.000Z",
        )
    )
    assert [attempt["session"] for attempt in doc["attempts"]] == [
        "pi-unknown-timestamp"
    ]
    selection = doc["run"]["ext"]["dev.dagr.adapter.pi"]["selection"]
    assert selection["retained_unknown_timestamp"] == 1
    assert selection["omitted"]["time_filter"] == 0


def test_pi_new_selection_and_version_extension_paths_are_documented():
    documented = (ROOT / "docs" / "adapters.md").read_text()
    for path in (
        "selection.retained_unknown_workspace",
        "selection.retained_unknown_timestamp",
        "selection.omitted.unsupported_or_missing_version",
    ):
        assert f"| `{path}` |" in documented


def test_pi_rejects_invalid_or_reversed_time_bounds():
    with pytest.raises(ValueError, match="RFC 3339"):
        _emit(Selection(stores=(FIXTURE,), since="yesterday"))
    with pytest.raises(ValueError, match="must not be later"):
        _emit(
            Selection(
                stores=(FIXTURE,),
                since="2026-09-01T00:00:00Z",
                until="2026-08-01T00:00:00Z",
            )
        )

    with pytest.raises(ValueError, match="must be at least 1"):
        _emit(Selection(stores=(FIXTURE,), limit=0))


def test_pi_fixture_models_have_valid_packaged_price_rows():
    table = load_prices()
    assert table.rate("gpt-5.5") is not None
    assert table.rate("claude-sonnet-5") is not None
    validation = validate_prices()
    assert validation["ok"] is True
    assert validation["errors"] == []
