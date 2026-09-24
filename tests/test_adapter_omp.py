"""Focused regression tests for the OMP 18.1.5 adapter."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from loopmath.adapters import (
    AdapterServices,
    PricingResult,
    Selection,
    Usage,
    list_adapters,
    lookup,
)
from loopmath.adapters import omp
from loopmath.adapters.omp import OMPAdapter
from loopmath.cli import main


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "omp"
PARENT_ID = "00000000000000a1"
CHILD_ID = "00000000000000b2"
_MISSING = object()

_spec = importlib.util.spec_from_file_location(
    "omp_ocp_conformance", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(conf)


def _errors(doc: dict[str, Any]) -> list[Any]:
    return [finding for finding in conf.validate_doc(doc) if finding.level == "error"]


def _warning_signatures(findings):
    return {
        (finding.code, finding.path, finding.message)
        for finding in findings
        if finding.level == "warning"
    }


def _ext(doc: dict[str, Any]) -> dict[str, Any]:
    return doc["run"]["ext"]["dev.dagr.adapter.omp"]


def _attempt_ext(doc: dict[str, Any], session_id: str) -> dict[str, Any]:
    attempt = next(item for item in doc["attempts"] if item["session"] == session_id)
    return attempt["ext"]["dev.dagr.adapter.omp"]


def _finding_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {row["reason"]: row["count"] for row in rows}


def _filter_counts(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in _ext(doc)["selection_filters"]}


def _usage(
    input_tokens: int = 1,
    output_tokens: int = 1,
    cache_read: int = 0,
    cache_write: int = 0,
) -> dict[str, int]:
    return {
        "input": input_tokens,
        "output": output_tokens,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "totalTokens": input_tokens + output_tokens + cache_read + cache_write,
    }


def _write_session(
    path: Path,
    session_id: object,
    *,
    timestamp: object = "2026-01-15T12:00:00.000Z",
    cwd: object = "/tmp/workspace",
    version: object = 3,
    records: tuple[object, ...] = (),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {
        "type": "session",
        "version": version,
        "id": session_id,
        "timestamp": timestamp,
    }
    if cwd is not _MISSING:
        header["cwd"] = cwd
    lines = [
        json.dumps({"type": "title", "v": 1, "title": "synthetic", "pad": ""}),
        json.dumps(header),
    ]
    lines.extend(item if isinstance(item, str) else json.dumps(item) for item in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _assistant(
    record_id: str,
    timestamp: str,
    *,
    provider: object = "provider-a",
    model: object = "model-a",
    usage: object = _MISSING,
    content: object = (),
    stop_reason: str = "stop",
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": list(content) if isinstance(content, tuple) else content,
        "provider": provider,
        "model": model,
        "stopReason": stop_reason,
    }
    if usage is not _MISSING:
        message["usage"] = usage
    return {
        "type": "message",
        "id": record_id,
        "parentId": None,
        "timestamp": timestamp,
        "message": message,
    }


def _copy_fixture(tmp_path: Path) -> Path:
    store = tmp_path / "omp"
    shutil.copytree(FIXTURE, store)
    return store


def _parent_file(store: Path) -> Path:
    return next(path for path in store.rglob("*.jsonl") if path.name != "general.jsonl")


def _child_file(store: Path) -> Path:
    return next(path for path in store.rglob("general.jsonl"))


def _rewrite_jsonl(path: Path, mutate: Callable[[list[dict[str, Any]]], None]) -> None:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(records)
    path.write_text(
        "\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n",
        encoding="utf-8",
    )


def test_omp_discover_sessions_and_real_nested_fixture_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(omp, "DEFAULT_STORE", tmp_path / "missing")
    assert OMPAdapter().discover() == ()
    assert OMPAdapter().sessions() == ()

    child_path = (
        FIXTURE
        / "-tmp-synthetic-omp-workspace"
        / "2026-01-15T12-00-00-000Z_00000000000000a1"
        / "general.jsonl"
    )
    assert child_path.is_file()
    assert len(tuple(FIXTURE.rglob("*.jsonl"))) == 2
    assert sum(len(path.read_text().splitlines()) for path in FIXTURE.rglob("*.jsonl")) == 13

    monkeypatch.setattr(omp, "DEFAULT_STORE", FIXTURE)
    adapter = OMPAdapter()
    assert adapter.discover() == (FIXTURE,)
    sessions = adapter.sessions()
    assert [session["id"] for session in sessions] == [PARENT_ID, CHILD_ID]
    assert [session["format_version"] for session in sessions] == [3, 3]
    assert [session["workspace"] for session in sessions] == [
        "/tmp/synthetic-omp-workspace",
        "/tmp/synthetic-omp-workspace",
    ]
    assert [session["is_subagent"] for session in sessions] == [False, True]
    assert sessions == adapter.sessions()


def test_omp_emit_fixture_has_exact_graph_counts_and_honest_evidence():
    doc = OMPAdapter().emit(Selection(stores=(FIXTURE,)))
    findings = conf.validate_doc(doc)

    assert (len(doc["nodes"]), len(doc["attempts"]), len(doc["edges"])) == (2, 2, 1)
    assert [finding for finding in findings if finding.level == "error"] == []
    assert _warning_signatures(findings) == set()
    assert _errors(doc) == []
    assert doc["edges"][0] == {
        "from": f"omp-session:{PARENT_ID}",
        "to": f"omp-session:{CHILD_ID}",
        "kind": "spawn",
        "tier": "heuristic",
        "evidence": (
            "One OMP task job and child session_init match exactly by task and "
            "agent within overlapping timestamps; OMP records no parent-session key"
        ),
    }

    attempts = {attempt["session"]: attempt for attempt in doc["attempts"]}
    parent = attempts[PARENT_ID]
    child = attempts[CHILD_ID]
    assert parent["cost"] == {
        "requests": 2,
        "basis": "measured",
        "input_tokens": 50,
        "cached_input_tokens": 0,
        "cache_creation_tokens": 0,
        "output_tokens": 12,
    }
    assert child["cost"]["input_tokens"] == 12
    assert child["cost"]["output_tokens"] == 1
    assert parent["role"]["tier"] == "heuristic"
    assert child["role"] == {
        "value": "subagent",
        "tier": "verified",
        "evidence": "OMP session_init record",
    }
    assert child["origin"]["launched_by"] is None
    assert child["origin"]["external"] is None
    assert child["origin"]["tier"] == "reported"
    assert all({"origin", "role", "phase"} <= set(attempt) for attempt in doc["attempts"])

    parent_ext = _attempt_ext(doc, PARENT_ID)
    child_ext = _attempt_ext(doc, CHILD_ID)
    assert parent_ext["message_role_counts"] == {
        "assistant": 2,
        "toolResult": 1,
        "user": 1,
    }
    assert parent_ext["record_parent_links"] == 4
    assert parent_ext["unresolved_parent_links"] == 0
    assert parent_ext["malformed_parent_links"] == 0
    assert parent_ext["task_calls"] == 1
    assert parent_ext["linked_task_results"] == 1
    assert parent_ext["linked_task_jobs"] == 1
    assert parent_ext["task_jobs"] == 1
    assert child_ext["record_parent_links"] == 3
    assert child_ext["linked_task_jobs"] == 0
    assert _ext(doc)["spawn_matching"] == {
        "child_sessions": 1,
        "matched_children": 1,
        "zero_candidate_children": 0,
        "ambiguous_children": 0,
        "multiple_job_children": 0,
        "missing_signature_children": 0,
    }

    serialized = json.dumps(doc, allow_nan=False)
    assert "ALPHA" not in serialized
    assert "Synthetic fixture context" not in serialized
    assert "Synthetic helper instructions" not in serialized


def test_parser_surfaces_all_file_record_id_and_parent_exclusions(tmp_path, monkeypatch):
    store = tmp_path / "store"
    valid_record = {
        "type": "model_change",
        "id": "r1",
        "parentId": None,
        "timestamp": "2026-01-15T12:00:00.100Z",
        "model": "provider/model",
    }
    _write_session(
        store / "00-main.jsonl",
        "diagnostic-session",
        records=(
            valid_record,
            "{",
            "[]",
            {"id": "missing-type", "parentId": None, "timestamp": "2026-01-15T12:00:01Z"},
            {"type": "message", "id": "", "parentId": None, "timestamp": "2026-01-15T12:00:01Z"},
            {"type": "model_change", "id": "r2", "parentId": 42, "timestamp": "invalid", "model": "m"},
            {"type": "model_change", "id": "r2", "parentId": None, "timestamp": "2026-01-15T12:00:02Z", "model": "m"},
            '{"type":"message","id":"nan","parentId":null,"timestamp":"2026-01-15T12:00:03Z","message":{"usage":{"input":NaN}}}',
        ),
    )
    _write_session(store / "zz-duplicate.jsonl", "diagnostic-session")
    (store / "bad-header.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    _write_session(store / "unsupported.jsonl", "unsupported", version=2)
    _write_session(store / "noninteger-version.jsonl", "float-version", version=3.0)
    _write_session(store / "invalid-id.jsonl", "")
    (store / "decode.jsonl").write_bytes(b"\xff\xfe")
    read_failure = _write_session(store / "read-failure.jsonl", "read-failure")

    original_read_text = Path.read_text

    def fail_one_read(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == read_failure:
            raise OSError("synthetic read failure")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_one_read)
    doc = OMPAdapter().emit(
        Selection(stores=(store, store, tmp_path / "missing-store"))
    )
    source = _ext(doc)
    assert source["source_files"] == {"candidates": 8, "loaded": 1}
    findings = _finding_counts(source["source_findings"])
    for reason in (
        "bad_header",
        "decode_failure",
        "duplicate_record_id",
        "duplicate_session_id",
        "duplicate_source_path",
        "invalid_record_id",
        "invalid_record_timestamp",
        "invalid_session_id",
        "malformed_json",
        "malformed_parent_id",
        "malformed_record_type",
        "non_object_record",
        "read_failure",
        "store_path_unavailable",
        "unsupported_version",
    ):
        assert findings[reason] >= 1
    assert findings["unsupported_version"] == 2
    attempt_ext = _attempt_ext(doc, "diagnostic-session")
    assert attempt_ext["unresolved_parent_links"] == 1
    assert attempt_ext["malformed_parent_links"] == 1
    json.dumps(doc, allow_nan=False)


def test_selection_retains_unknown_source_fields_and_reports_limit_omissions(tmp_path):
    store = tmp_path / "store"
    _write_session(
        store / "unknown.jsonl",
        "unknown-fields",
        timestamp="not-a-time",
        cwd=_MISSING,
    )
    _write_session(
        store / "match.jsonl",
        "known-match",
        timestamp="2026-01-15T12:00:00Z",
        cwd="/match",
    )
    _write_session(
        store / "excluded.jsonl",
        "known-excluded",
        timestamp="2026-01-15T10:00:00Z",
        cwd="/other",
    )
    selection = Selection(
        stores=(store,),
        workspaces=("/match",),
        since="2026-01-15T11:00:00Z",
        until="2026-01-15T13:00:00Z",
    )
    doc = OMPAdapter().emit(selection)
    assert [attempt["session"] for attempt in doc["attempts"]] == [
        "known-match",
        "unknown-fields",
    ]
    filters = _filter_counts(doc)
    assert filters["workspace"] == {
        "name": "workspace",
        "active": True,
        "unknown": 1,
        "excluded": 1,
    }
    assert filters["since"]["unknown"] == 1
    assert filters["until"]["unknown"] == 1
    assert _ext(doc)["limit_omitted"] == 0

    limited = OMPAdapter().emit(
        Selection(
            stores=(store,),
            workspaces=("/match",),
            since="2026-01-15T11:00:00Z",
            until="2026-01-15T13:00:00Z",
            limit=1,
        )
    )
    assert [attempt["session"] for attempt in limited["attempts"]] == ["known-match"]
    assert _ext(limited)["limit_omitted"] == 1


def test_run_start_requires_every_selected_start_and_keeps_diagnostic(tmp_path):
    store = tmp_path / "store"
    _write_session(
        store / "unknown.jsonl",
        "unknown-start",
        timestamp="not-a-time",
    )
    _write_session(
        store / "known.jsonl",
        "known-start",
        timestamp="2026-01-15T12:00:00Z",
    )

    doc = OMPAdapter().emit(Selection(stores=(store,)))

    findings = _finding_counts(_ext(doc)["source_findings"])
    assert findings["invalid_header_timestamp"] == 1
    assert "started_at" not in doc["run"]


def test_task_jobs_parse_both_arrays_deduplicate_and_do_not_overcount(tmp_path):
    store = _copy_fixture(tmp_path)

    def add_result_snapshot(records: list[dict[str, Any]]) -> None:
        tool_result = next(
            record
            for record in records
            if record.get("type") == "message"
            and record.get("message", {}).get("role") == "toolResult"
        )
        progress_job = tool_result["message"]["details"]["progress"][0]
        tool_result["message"]["details"]["results"] = [dict(progress_job)]

    _rewrite_jsonl(_parent_file(store), add_result_snapshot)
    doc = OMPAdapter().emit(Selection(stores=(store,)))
    parent = _attempt_ext(doc, PARENT_ID)
    assert parent["task_jobs"] == 1
    assert parent["linked_task_jobs"] == 1
    assert _finding_counts(parent["task_findings"])["duplicate_task_job"] == 1
    assert len(doc["edges"]) == 1


def test_task_diagnostics_surface_malformed_duplicate_and_unmatched_records(tmp_path):
    store = tmp_path / "store"
    calls = (
        {"type": "toolCall", "name": "task", "id": "call-a"},
        {"type": "toolCall", "name": "task", "id": "call-a"},
        {"type": "toolCall", "name": "task", "id": "call-b"},
        {"type": "toolCall", "name": "task", "id": None},
    )
    linked_details = {
        "results": [{"id": "job-result", "task": "task-a", "agent": "agent-a"}],
        "progress": [
            {"id": "job-progress", "task": "task-a", "agent": "agent-a"},
            {"id": "job-progress", "task": "task-a", "agent": "agent-a"},
            {"id": "bad-job", "task": "task-a", "agent": None},
        ],
        "async": {"jobId": "not-present"},
    }
    records = (
        _assistant(
            "r1",
            "2026-01-15T12:00:01Z",
            usage=_usage(),
            content=calls,
            stop_reason="toolUse",
        ),
        {
            "type": "message",
            "id": "r2",
            "parentId": "r1",
            "timestamp": "2026-01-15T12:00:02Z",
            "message": {
                "role": "toolResult",
                "toolName": "task",
                "toolCallId": "call-a",
                "details": linked_details,
            },
        },
        {
            "type": "message",
            "id": "r3",
            "parentId": "r2",
            "timestamp": "2026-01-15T12:00:03Z",
            "message": {
                "role": "toolResult",
                "toolName": "task",
                "toolCallId": "call-missing",
                "details": {
                    "results": [
                        {"id": "job-unmatched", "task": "task-a", "agent": "agent-a"}
                    ],
                    "progress": [],
                },
            },
        },
        {
            "type": "message",
            "id": "r4",
            "parentId": "r3",
            "timestamp": "2026-01-15T12:00:04Z",
            "message": {
                "role": "toolResult",
                "toolName": "task",
                "toolCallId": None,
                "details": None,
            },
        },
    )
    _write_session(store / "tasks.jsonl", "task-diagnostics", records=records)
    doc = OMPAdapter().emit(Selection(stores=(store,)))
    attempt = _attempt_ext(doc, "task-diagnostics")
    assert attempt["task_calls"] == 4
    assert attempt["linked_task_results"] == 1
    assert attempt["linked_task_jobs"] == 0
    assert attempt["task_jobs"] == 3
    findings = _finding_counts(attempt["task_findings"])
    for reason in (
        "duplicate_task_call_id",
        "duplicate_task_job",
        "malformed_task_call",
        "malformed_task_details",
        "malformed_task_job",
        "malformed_task_result",
        "unmatched_async_job_id",
        "unmatched_task_call",
        "unmatched_task_job",
        "unmatched_task_result",
    ):
        assert findings[reason] >= 1


def test_spawn_requires_one_exact_non_null_task_agent_job_across_all_parents(tmp_path):
    multiple_store = _copy_fixture(tmp_path / "multiple")

    def add_second_job(records: list[dict[str, Any]]) -> None:
        tool_result = next(
            record
            for record in records
            if record.get("type") == "message"
            and record.get("message", {}).get("role") == "toolResult"
        )
        job = dict(tool_result["message"]["details"]["progress"][0])
        job["id"] = "job-synthetic-02"
        tool_result["message"]["details"]["progress"].append(job)

    _rewrite_jsonl(_parent_file(multiple_store), add_second_job)
    multiple = OMPAdapter().emit(Selection(stores=(multiple_store,)))
    assert multiple["edges"] == []
    assert _ext(multiple)["spawn_matching"]["multiple_job_children"] == 1
    assert _ext(multiple)["spawn_matching"]["ambiguous_children"] == 0

    ambiguous_store = _copy_fixture(tmp_path / "ambiguous")
    parent_lines = _parent_file(ambiguous_store).read_text(encoding="utf-8").splitlines()
    duplicate_parent = [json.loads(line) for line in parent_lines]
    duplicate_parent[1]["id"] = "00000000000000c3"
    second_parent = ambiguous_store / "-tmp-synthetic-omp-workspace" / "second.jsonl"
    second_parent.write_text(
        "\n".join(json.dumps(record) for record in duplicate_parent) + "\n",
        encoding="utf-8",
    )
    ambiguous = OMPAdapter().emit(Selection(stores=(ambiguous_store,)))
    assert ambiguous["edges"] == []
    assert _ext(ambiguous)["spawn_matching"]["multiple_job_children"] == 1
    assert _ext(ambiguous)["spawn_matching"]["ambiguous_children"] == 1

    missing_store = _copy_fixture(tmp_path / "missing")

    def remove_agent(records: list[dict[str, Any]]) -> None:
        session_init = next(record for record in records if record.get("type") == "session_init")
        session_init["agent"] = None

    _rewrite_jsonl(_child_file(missing_store), remove_agent)
    missing = OMPAdapter().emit(Selection(stores=(missing_store,)))
    assert missing["edges"] == []
    assert _ext(missing)["spawn_matching"]["missing_signature_children"] == 1
    assert _ext(missing)["spawn_matching"]["zero_candidate_children"] == 1

    mismatch_store = _copy_fixture(tmp_path / "mismatch")

    def mismatch_agent(records: list[dict[str, Any]]) -> None:
        session_init = next(record for record in records if record.get("type") == "session_init")
        session_init["agent"] = "different-agent"

    _rewrite_jsonl(_child_file(mismatch_store), mismatch_agent)
    mismatch = OMPAdapter().emit(Selection(stores=(mismatch_store,)))
    assert mismatch["edges"] == []
    assert _ext(mismatch)["spawn_matching"]["zero_candidate_children"] == 1


def test_usage_enumerates_assistant_and_model_usage_without_partial_totals(tmp_path):
    store = tmp_path / "store"
    bad_usage = _usage()
    bad_usage["output"] = -1
    records = (
        _assistant("r1", "2026-01-15T12:00:01Z", usage=_usage(3, 2), stop_reason="toolUse"),
        {
            "type": "model_usage",
            "id": "r2",
            "parentId": "r1",
            "timestamp": "2026-01-15T12:00:02Z",
            "provider": None,
            "model": "model-b",
            "usage": _usage(5, 4, 1, 2),
        },
        _assistant("r3", "2026-01-15T12:00:03Z", usage=bad_usage, stop_reason="toolUse"),
        _assistant(
            "r4",
            "2026-01-15T12:00:04Z",
            provider="provider-c",
            model="model-c",
            usage=_MISSING,
        ),
    )
    _write_session(store / "usage.jsonl", "usage-session", records=records)
    price_calls: list[tuple[str | None, Usage | None]] = []

    def price(model: str | None, usage: Usage | None) -> PricingResult:
        price_calls.append((model, usage))
        return PricingResult(usage, None, False, "synthetic unpriced")

    adapter = OMPAdapter(AdapterServices(price_usage=price))
    doc = adapter.emit(Selection(stores=(store,)))
    attempt = doc["attempts"][0]
    ext = attempt["ext"]["dev.dagr.adapter.omp"]
    assert "cost" not in attempt
    assert ext["model_calls"] == 4
    assert ext["usage_complete_calls"] == 2
    assert ext["usage_unknown_calls"] == 2
    findings = _finding_counts(ext["usage_findings"])
    assert findings["provider_missing"] == 1
    assert findings["usage_output_invalid"] == 1
    assert findings["usage_missing"] == 1
    groups = {(row["provider"], row["model"]): row for row in ext["usage_by_model"]}
    assert groups[("provider-a", "model-a")]["calls"] == 2
    assert groups[("provider-a", "model-a")]["usage_complete_calls"] == 1
    assert groups[("provider-a", "model-a")]["known_input_tokens"] == 3
    assert groups[(None, "model-b")]["known_output_tokens"] == 4
    assert groups[("provider-c", "model-c")]["usage_unknown_calls"] == 1
    assert sorted((model, usage is None) for model, usage in price_calls) == [
        ("model-a", True),
        ("model-b", False),
        ("model-c", True),
    ]
    json.dumps(doc, allow_nan=False)


def test_pricing_hook_preserves_settled_provisional_and_unpriced_states():
    def emit_with(result_kind: str) -> dict[str, Any]:
        def price(model: str | None, usage: Usage | None) -> PricingResult:
            assert model == "synthetic-zero-cost"
            assert usage is not None
            if result_kind == "settled":
                return PricingResult(usage, 0.125, True, None)
            if result_kind == "provisional":
                return PricingResult(
                    usage,
                    None,
                    False,
                    "price-table entry is provisional",
                    provisional=True,
                    estimate_usd=0.25,
                )
            return PricingResult(usage, None, False, "model is absent from pricing")

        services = AdapterServices(price_usage=price, producer_version="test-version")
        return OMPAdapter(services).emit(
            Selection(stores=(FIXTURE,), session_ids=(PARENT_ID,))
        )

    settled = emit_with("settled")
    assert settled["producer"]["version"] == "test-version"
    assert settled["attempts"][0]["cost"]["usd"] == 0.125
    assert _attempt_ext(settled, PARENT_ID)["pricing_by_model"][0] == {
        "provider": "ollama",
        "model": "synthetic-zero-cost",
        "priced": True,
        "provisional": False,
        "reason": None,
        "usd": 0.125,
        "estimate_usd": None,
    }

    provisional = emit_with("provisional")
    assert "usd" not in provisional["attempts"][0]["cost"]
    assert "estimate_usd" not in provisional["attempts"][0]["cost"]
    assert _attempt_ext(provisional, PARENT_ID)["pricing_by_model"][0] == {
        "provider": "ollama",
        "model": "synthetic-zero-cost",
        "priced": False,
        "provisional": True,
        "reason": "price-table entry is provisional",
        "usd": None,
        "estimate_usd": 0.25,
    }

    unpriced = emit_with("unpriced")
    assert "usd" not in unpriced["attempts"][0]["cost"]
    assert _attempt_ext(unpriced, PARENT_ID)["pricing_by_model"][0] == {
        "provider": "ollama",
        "model": "synthetic-zero-cost",
        "priced": False,
        "provisional": False,
        "reason": "model is absent from pricing",
        "usd": None,
        "estimate_usd": None,
    }


def test_run_id_depends_only_on_selected_native_session_ids(tmp_path):
    first = _copy_fixture(tmp_path / "first")
    second = _copy_fixture(tmp_path / "second")
    first_doc = OMPAdapter().emit(Selection(stores=(first,)))
    second_doc = OMPAdapter().emit(Selection(stores=(second,)))
    assert first_doc["run"]["id"] == second_doc["run"]["id"]


def test_omp_filters_registry_and_cli_wiring(capsys):
    selection = Selection(
        stores=(FIXTURE,),
        session_ids=(CHILD_ID,),
        workspaces=("/tmp/synthetic-omp-workspace",),
        since="2026-01-15T12:00:02.500Z",
        until="2026-01-15T12:00:02.500Z",
        limit=1,
    )
    selected = OMPAdapter().emit(selection)
    assert [attempt["session"] for attempt in selected["attempts"]] == [CHILD_ID]
    assert (len(selected["nodes"]), len(selected["attempts"]), len(selected["edges"])) == (
        1,
        1,
        0,
    )
    assert _errors(selected) == []

    assert lookup("omp") is OMPAdapter
    assert "omp" in list_adapters()
    rc = main(
        [
            "adapt",
            "omp",
            "--store",
            str(FIXTURE),
            "--session",
            CHILD_ID,
        ]
    )
    captured = capsys.readouterr()
    emitted = json.loads(captured.out)
    assert rc == 0
    assert captured.err == ""
    assert emitted["ocp"] == "0.2"
    assert (len(emitted["nodes"]), len(emitted["attempts"]), len(emitted["edges"])) == (
        1,
        1,
        0,
    )
    assert _errors(emitted) == []
