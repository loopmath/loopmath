"""OpenCode adapter coverage using invented export-shaped sessions."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import loopmath.adapters.opencode as opencode_module
from loopmath.adapters import Selection, lookup
from loopmath.adapters.opencode import OpenCodeAdapter
from loopmath.price import load_prices, price_all, price_run


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "opencode"
_spec = importlib.util.spec_from_file_location(
    "opencode_ocp_conformance", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


@pytest.fixture(scope="module")
def doc() -> dict:
    return OpenCodeAdapter().emit(Selection(stores=(FIXTURE,)))


def _attempt(doc: dict, session_id: str) -> dict:
    return next(item for item in doc["attempts"] if item["session"] == session_id)


def _ids(doc: dict) -> set[str]:
    return {item["session"] for item in doc["attempts"]}


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


def test_fixture_emits_conforming_exact_graph_counts(doc):
    findings = conf.validate_doc(doc)
    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == {
        _w200("$['events'][1]['type']", "session_updated"),
        _w200("$['events'][4]['type']", "session_updated"),
        _w200("$['events'][6]['type']", "session_updated"),
        _w200("$['events'][9]['type']", "session_updated"),
    }
    assert len(doc["nodes"]) == 4
    assert len(doc["attempts"]) == 4
    assert len(doc["edges"]) == 1
    assert len(doc["groups"]) == 2
    assert len(doc["events"]) == 11


def test_parent_id_is_the_only_edge_and_is_verified(doc):
    assert doc["edges"] == [
        {
            "from": "ses_fixture_root",
            "to": "ses_fixture_child",
            "kind": "spawn",
            "tier": "verified",
            "from_attempt": "attempt:ses_fixture_root",
            "to_attempt": "attempt:ses_fixture_child",
            "evidence": "OpenCode session.parentID matched the selected parent session id",
        }
    ]


def test_five_message_usage_streams_and_session_usd_are_measured(doc):
    root = _attempt(doc, "ses_fixture_root")
    assert root["cost"] == {
        "input_tokens": 150,
        "cached_input_tokens": 15,
        "cache_creation_tokens": 3,
        "output_tokens": 30,
        "reasoning_tokens": 7,
        "requests": 2,
        "usd": 0.0015,
        "basis": "measured",
    }

    child = _attempt(doc, "ses_fixture_child")
    assert child["model"] == {
        "raw": "gpt-5.5",
        "id": "gpt-5.5",
        "provider": "openai",
        "tier": "verified",
    }
    assert child["effort"] == "xhigh"


def test_mixed_models_are_not_misattributed_to_one_core_model(doc):
    root = _attempt(doc, "ses_fixture_root")
    assert "model" not in root
    usage = root["ext"]["dev.dagr.adapter.opencode"]["model_usage"]
    assert [(item["provider"], item["model"], item["requests"]) for item in usage] == [
        ("openai", "gpt-5.3-codex-spark", 1),
        ("openai", "gpt-5.4-mini", 1),
    ]
    assert sum(item["input_tokens"] for item in usage) == root["cost"]["input_tokens"]
    assert sum(item["usd"] for item in usage) == pytest.approx(root["cost"]["usd"])


def test_metadata_only_output_never_copies_fixture_transcript_text(doc):
    rendered = json.dumps(doc)
    assert "INVENTED_PRIVATE" not in rendered
    assert "Invented title" not in rendered
    assert doc["privacy"]["profile"] == "metadata_only"


def test_registry_and_sessions_are_deterministic(monkeypatch):
    assert lookup("opencode") is OpenCodeAdapter
    monkeypatch.setattr(OpenCodeAdapter, "discover", lambda self: (FIXTURE,))
    rows = OpenCodeAdapter().sessions()
    assert [row["id"] for row in rows] == [
        "ses_fixture_child",
        "ses_fixture_free_big_pickle",
        "ses_fixture_other",
        "ses_fixture_root",
    ]
    assert rows[0]["parent_id"] == "ses_fixture_root"
    assert rows[0]["workspace"] == "/synthetic/workspaces/alpha"


def test_session_and_workspace_filters_keep_only_exact_matches():
    adapter = OpenCodeAdapter()
    child = adapter.emit(
        Selection(stores=(FIXTURE,), session_ids=("ses_fixture_child",))
    )
    assert _ids(child) == {"ses_fixture_child"}
    assert len(child["nodes"]) == len(child["attempts"]) == 1
    assert child["edges"] == []
    assert child["run"]["ext"]["dev.dagr.adapter.opencode"] == {
        "parent_relations_omitted": 1,
        "selection_excluded_by_session_id": 3,
    }

    alpha = adapter.emit(
        Selection(stores=(FIXTURE,), workspaces=("/synthetic/workspaces/alpha",))
    )
    assert _ids(alpha) == {"ses_fixture_root", "ses_fixture_child"}
    assert len(alpha["edges"]) == 1
    assert alpha["run"]["ext"]["dev.dagr.adapter.opencode"] == {
        "selection_excluded_by_workspace": 2
    }


@pytest.mark.parametrize(
    ("since", "until", "expected", "excluded"),
    [
        (
            "2024-01-02T00:00:00Z",
            None,
            {"ses_fixture_child", "ses_fixture_other", "ses_fixture_free_big_pickle"},
            1,
        ),
        (
            None,
            "2024-01-02T00:00:00Z",
            {"ses_fixture_root", "ses_fixture_child"},
            2,
        ),
        (
            "2024-01-02T00:00:00Z",
            "2024-01-02T00:00:00Z",
            {"ses_fixture_child"},
            3,
        ),
    ],
)
def test_time_bounds_are_inclusive(since, until, expected, excluded):
    doc = OpenCodeAdapter().emit(
        Selection(stores=(FIXTURE,), since=since, until=until)
    )
    assert _ids(doc) == expected
    assert doc["run"]["ext"]["dev.dagr.adapter.opencode"][
        "selection_excluded_by_time"
    ] == excluded


def test_limit_selects_the_most_recent_session():
    doc = OpenCodeAdapter().emit(Selection(stores=(FIXTURE,), limit=1))
    assert _ids(doc) == {"ses_fixture_free_big_pickle"}
    assert len(doc["nodes"]) == len(doc["attempts"]) == 1
    assert doc["edges"] == []
    assert doc["run"]["ext"]["dev.dagr.adapter.opencode"] == {
        "selection_excluded_by_limit": 3
    }


def test_inventory_selection_counts_survive_single_materialization_pass(
    tmp_path, monkeypatch
):
    database = tmp_path / "opencode.db"

    def source(session_id, workspace, created):
        return opencode_module._SessionSource(
            {
                "id": session_id,
                "directory": workspace,
                "version": "synthetic",
                "cost": 0,
                "time": {"created": created},
            },
            None,
            database,
            "database",
        )

    inventoried = [
        source("excluded-id", "/keep", "2024-01-03T00:00:00Z"),
        source("excluded-workspace", "/other", "2024-01-03T00:00:00Z"),
        source("excluded-time", "/keep", "2024-01-01T00:00:00Z"),
        source("selected-old", "/keep", "2024-01-02T00:00:00Z"),
        source("selected-new", "/keep", "2024-01-03T00:00:00Z"),
    ]
    exported = []

    def export_session(session):
        exported.append(session.info["id"])
        return opencode_module._SessionSource(
            session.info, (), source_format="export"
        )

    monkeypatch.setattr(opencode_module, "_store_sessions", lambda path: inventoried)
    monkeypatch.setattr(opencode_module, "_default_database", lambda: database)
    monkeypatch.setattr(opencode_module, "_export_session", export_session)

    doc = OpenCodeAdapter().emit(
        Selection(
            stores=(database,),
            session_ids=(
                "excluded-workspace",
                "excluded-time",
                "selected-old",
                "selected-new",
            ),
            workspaces=("/keep",),
            since="2024-01-02T00:00:00Z",
            limit=1,
        )
    )

    assert _ids(doc) == {"selected-new"}
    assert exported == ["selected-new"]
    assert doc["run"]["ext"]["dev.dagr.adapter.opencode"] == {
        "selection_excluded_by_limit": 1,
        "selection_excluded_by_session_id": 1,
        "selection_excluded_by_time": 1,
        "selection_excluded_by_workspace": 1,
    }


def test_unknown_filter_fields_are_included_and_counted(tmp_path):
    raw = json.loads((FIXTURE / "other.json").read_text())
    raw["info"]["id"] = "ses_fixture_unknown_fields"
    raw["info"].pop("directory")
    raw["info"]["time"].pop("created")
    export = tmp_path / "unknown.json"
    export.write_text(json.dumps(raw), encoding="utf-8")

    doc = OpenCodeAdapter().emit(
        Selection(
            stores=(export,),
            workspaces=("/synthetic/workspaces/none",),
            since="2024-01-01T00:00:00Z",
        )
    )
    assert _ids(doc) == {"ses_fixture_unknown_fields"}
    assert doc["run"]["ext"]["dev.dagr.adapter.opencode"] == {
        "selection_unknown_created_at": 1,
        "selection_unknown_workspace": 1,
    }


def test_incomplete_usage_and_model_attribution_are_counted(tmp_path):
    raw = json.loads((FIXTURE / "child.json").read_text())
    assistant = raw["messages"][1]["info"]
    assistant.pop("modelID")
    assistant["tokens"]["cache"].pop("write")
    export = tmp_path / "incomplete.json"
    export.write_text(json.dumps(raw), encoding="utf-8")

    doc = OpenCodeAdapter().emit(Selection(stores=(export,)))
    attempt = _attempt(doc, "ses_fixture_child")
    assert "cache_creation_tokens" not in attempt["cost"]
    assert attempt["cost"]["input_tokens"] == 200
    assert attempt["model"]["tier"] == "reported"
    assert attempt["ext"]["dev.dagr.adapter.opencode"] == {
        "model_requests_unattributed": 1,
        "usage_incomplete_messages": {"cache_creation_tokens": 1},
    }


def test_every_possible_extension_member_is_documented():
    documented = (ROOT / "docs" / "adapters.md").read_text()
    paths = {
        "selection_excluded_by_session_id",
        "selection_excluded_by_workspace",
        "selection_excluded_by_time",
        "selection_excluded_by_limit",
        "selection_unknown_created_at",
        "selection_unknown_workspace",
        "parent_relations_omitted",
        "parent_relations_unresolved",
        "model_usage",
        "model_usage[].provider",
        "model_usage[].model",
        "model_usage[].variants",
        "model_usage[].requests",
        "model_usage[].input_tokens",
        "model_usage[].cached_input_tokens",
        "model_usage[].cache_creation_tokens",
        "model_usage[].output_tokens",
        "model_usage[].reasoning_tokens",
        "model_usage[].usd",
        "model_requests_unattributed",
        "usage_incomplete_messages",
        "usage_incomplete_messages.input_tokens",
        "usage_incomplete_messages.cached_input_tokens",
        "usage_incomplete_messages.cache_creation_tokens",
        "usage_incomplete_messages.output_tokens",
        "usage_incomplete_messages.reasoning_tokens",
    }
    for path in paths:
        assert f"| `{path}` |" in documented
    assert (
        "| OpenCode | `dev.dagr.adapter.opencode` | `(namespace root)` | run, attempt |"
        in documented
    )


def test_every_fixture_model_has_a_packaged_price_row():
    table = load_prices()
    assert {
        "gpt-5.3-codex-spark",
        "gpt-5.4-mini",
        "gpt-5.5",
        "big-pickle",
        "deepseek-v4-flash-free",
        "north-mini-code-free",
    } <= set(table.rates)
    assert table.rates["gpt-5.3-codex-spark"] == {
        "input": 1.75,
        "cache_read": 0.175,
        "cache_write": 0.0,
        "output": 14.0,
    }
    assert table.rates["gpt-5.4-mini"] == {
        "input": 0.75,
        "cache_read": 0.075,
        "cache_write": 0.0,
        "output": 4.5,
    }
    for free_model in (
        "big-pickle",
        "deepseek-v4-flash-free",
        "north-mini-code-free",
    ):
        assert set(table.rates[free_model].values()) == {0.0}
        assert free_model not in table.todo


def test_free_model_is_priced_at_zero_while_missing_model_is_unpriced(doc):
    attempt = _attempt(doc, "ses_fixture_free_big_pickle")
    cost = attempt["cost"]
    emitted_record = {
        "model": attempt["model"]["id"],
        "tokens": {
            "in": cost["input_tokens"],
            "cache_read": cost["cached_input_tokens"],
            "cache_write": cost["cache_creation_tokens"],
            "out": cost["output_tokens"],
        },
    }
    missing_record = {**emitted_record, "model": "fixture-model-with-no-price-row"}
    table = load_prices()

    free = price_run(emitted_record, table)
    missing = price_run(missing_record, table)
    assert free["priced"] is True
    assert free["usd"] == 0.0
    assert free["reason"] is None
    assert missing["priced"] is False
    assert missing["usd"] is None
    assert missing["reason"] == "no price entry for model 'fixture-model-with-no-price-row'"

    priced, warnings = price_all([emitted_record, missing_record], table)
    assert priced[0]["usd"] == 0.0
    assert priced[1]["usd"] is None
    assert warnings["n_unpriced_runs"] == 1
    assert warnings["unpriced_models"] == {"fixture-model-with-no-price-row": 1}
