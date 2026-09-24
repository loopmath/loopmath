"""Focused coverage for the OTLP/JSON GenAI adapter."""

from __future__ import annotations

import copy
import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from loopmath.adapter_host import default_adapter_services
from loopmath.adapters import AdapterServices, PricingResult, Selection, Usage, lookup
from loopmath.cli import main


ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "adapters" / "otel-genai" / "trace.otlp.jsonl"
TRACE_ID = "1" * 32
PREFIX = f"otel:{TRACE_ID}"
EXPECTED_NODES = {
    f"{PREFIX}:claude:session:s-main",
    f"{PREFIX}:claude:agent:a-parent",
    f"{PREFIX}:claude:agent:a-child",
    f"{PREFIX}:codex:session:c-review",
}

_spec = importlib.util.spec_from_file_location(
    "ocp_conformance_otel_genai", ROOT / "spec" / "ocp_conformance.py"
)
conf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conf)


def _emit(**selection) -> dict:
    return lookup("otel-genai")().emit(
        Selection(stores=(FIXTURE,), **selection)
    )


def _errors(doc: dict) -> list:
    return [finding for finding in conf.validate_doc(doc) if finding.level == "error"]


def _fixture_records() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def _spans(records: list[dict]) -> list[dict]:
    return [
        span
        for record in records
        for resource_span in record["resourceSpans"]
        for scope_span in resource_span["scopeSpans"]
        for span in scope_span["spans"]
    ]


def _span(records: list[dict], span_id: str) -> dict:
    return next(span for span in _spans(records) if span["spanId"] == span_id)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def _attribute(span: dict, key: str) -> dict:
    return next(item for item in span["attributes"] if item["key"] == key)


def _whole_object_with_second_trace(path: Path) -> Path:
    records = _fixture_records()
    original = [item for record in records for item in record["resourceSpans"]]
    later = copy.deepcopy(original)
    for resource_span in later:
        for scope_span in resource_span["scopeSpans"]:
            for span in scope_span["spans"]:
                span["traceId"] = "2" * 32
                span["startTimeUnixNano"] = str(int(span["startTimeUnixNano"]) + 20_000_000_000)
                span["endTimeUnixNano"] = str(int(span["endTimeUnixNano"]) + 20_000_000_000)
                for link in span.get("links", []):
                    link["traceId"] = "2" * 32
    path.write_text(json.dumps({"resourceSpans": original + later}), encoding="utf-8")
    return path


def test_registration_requires_explicit_store_and_has_no_canonical_discovery():
    adapter_type = lookup("otel-genai")
    assert adapter_type.name == "otel-genai"
    adapter = adapter_type()
    assert adapter.discover() == ()
    assert adapter.sessions() == ()
    with pytest.raises(ValueError, match="explicit --store"):
        adapter.emit(Selection())


def test_fixture_selection_targets_trace_file_and_explicit_directory_is_rejected(
    tmp_path,
):
    adapter = lookup("otel-genai")()
    with adapter.fixture_selection(tmp_path) as selection:
        assert selection == Selection(stores=(tmp_path / "trace.otlp.jsonl",))
    with pytest.raises(ValueError, match="not a regular file"):
        adapter.emit(Selection(stores=(tmp_path,)))


def test_adapter_module_depends_only_on_stdlib_and_portable_adapter_contract():
    source_path = ROOT / "src" / "loopmath" / "adapters" / "otel_genai.py"
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.partition(".")[0] in sys.stdlib_module_names for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                assert (node.level, node.module) in {(1, "base"), (1, "registry")}
            else:
                assert node.module.partition(".")[0] in sys.stdlib_module_names


def test_synthetic_jsonl_emits_exact_conforming_graph_and_private_fields_are_omitted():
    records = _fixture_records()
    assert len(records) == 2
    assert all("resourceSpans" in record for record in records)

    doc = _emit()
    assert _errors(doc) == []
    assert {
        name: len(doc[name])
        for name in ("groups", "nodes", "attempts", "edges", "artifacts", "events")
    } == {
        "groups": 1,
        "nodes": 4,
        "attempts": 4,
        "edges": 3,
        "artifacts": 0,
        "events": 8,
    }
    assert {node["id"] for node in doc["nodes"]} == EXPECTED_NODES
    assert [(edge["kind"], edge["from"], edge["to"], edge["tier"]) for edge in doc["edges"]] == [
        (
            "launch",
            f"{PREFIX}:claude:session:s-main",
            f"{PREFIX}:codex:session:c-review",
            "verified",
        ),
        (
            "spawn",
            f"{PREFIX}:claude:agent:a-parent",
            f"{PREFIX}:claude:agent:a-child",
            "verified",
        ),
        (
            "spawn",
            f"{PREFIX}:claude:session:s-main",
            f"{PREFIX}:claude:agent:a-parent",
            "verified",
        ),
    ]
    assert all(edge["from_attempt"].endswith(".a1") for edge in doc["edges"])
    assert all(edge["to_attempt"].endswith(".a1") for edge in doc["edges"])

    extension = doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]
    assert extension == {
        "input_span_count": 8,
        "mapped_span_count": 8,
        "omissions": [
            {"reason": "missing_reported_cost", "count": 1},
            {"reason": "unpriced_attempt", "count": 1},
        ],
    }
    serialized = json.dumps(doc)
    for private_marker in (
        "SYNTHETIC_PRIVATE_PROMPT_DO_NOT_COPY",
        "SYNTHETIC_PRIVATE_TOOL_INPUT_DO_NOT_COPY",
        "SYNTHETIC_PRIVATE_COMMAND_DO_NOT_COPY",
        "SYNTHETIC_PRIVATE_RESPONSE_DO_NOT_COPY",
    ):
        assert private_marker not in serialized


def test_beta_fields_token_costs_and_exact_launch_origin_are_mapped():
    doc = _emit()
    attempts = {attempt["node"]: attempt for attempt in doc["attempts"]}
    child = attempts[f"{PREFIX}:claude:agent:a-child"]
    parent = attempts[f"{PREFIX}:claude:agent:a-parent"]
    main = attempts[f"{PREFIX}:claude:session:s-main"]
    codex = attempts[f"{PREFIX}:codex:session:c-review"]

    assert child["session"] == parent["session"] == main["session"] == "s-main"
    assert codex["session"] == "c-review"
    assert child["effort"] == "high"
    assert codex["effort"] == "xhigh"
    assert child["model"] == {
        "raw": "claude-opus-5",
        "tier": "reported",
        "id": "claude-opus-5",
        "provider": "anthropic",
    }
    assert child["cost"] == {
        "requests": 1,
        "basis": "measured",
        "input_tokens": 20,
        "cached_input_tokens": 4,
        "cache_creation_tokens": 1,
        "output_tokens": 6,
        "usd": pytest.approx(0.000262),
    }
    assert child["ext"]["dev.dagr.adapter.otel-genai"]["reported_cost_usd"] == pytest.approx(
        0.000262
    )
    assert codex["cost"]["input_tokens"] == 150
    assert codex["cost"]["cached_input_tokens"] == 50
    assert main["cost"]["usd"] == pytest.approx(0.001235)
    assert parent["cost"]["usd"] == pytest.approx(0.0004725)
    assert "usd" not in codex["cost"]
    assert "ext" not in codex
    assert codex["origin"] == {
        "launched_by": main["id"],
        "workspace": "/synthetic/workspace",
        "external": False,
        "how": "W3C trace context",
        "tier": "verified",
        "evidence": "OTLP traceId/parentSpanId linkage",
    }


def test_complete_measured_usage_is_priced_by_the_injected_dagr_hook():
    adapter = lookup("otel-genai")(
        default_adapter_services(producer_version="synthetic-test")
    )
    doc = adapter.emit(Selection(stores=(FIXTURE,)))
    attempts = {attempt["node"]: attempt for attempt in doc["attempts"]}
    assert attempts[f"{PREFIX}:claude:session:s-main"]["cost"]["usd"] == pytest.approx(
        0.001235
    )
    assert attempts[f"{PREFIX}:claude:agent:a-parent"]["cost"]["usd"] == pytest.approx(
        0.0004725
    )
    assert attempts[f"{PREFIX}:claude:agent:a-child"]["cost"]["usd"] == pytest.approx(
        0.000262
    )
    assert attempts[f"{PREFIX}:codex:session:c-review"]["cost"]["usd"] == pytest.approx(
        0.00102
    )
    assert doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]["omissions"] == [
        {"reason": "missing_reported_cost", "count": 1}
    ]
    assert (
        attempts[f"{PREFIX}:claude:agent:a-child"]["ext"]
        ["dev.dagr.adapter.otel-genai"]["reported_cost_usd"]
        == pytest.approx(0.000262)
    )


def test_reported_cost_needs_no_pricing_when_one_request_model_is_missing(tmp_path):
    records = _fixture_records()
    main_request = _span(records, "0000000000000002")
    missing_model_request = copy.deepcopy(main_request)
    missing_model_request["spanId"] = "0000000000000090"
    missing_model_request["startTimeUnixNano"] = "1767225602100000000"
    missing_model_request["endTimeUnixNano"] = "1767225602200000000"
    missing_model_request["attributes"] = [
        item
        for item in missing_model_request["attributes"]
        if item["key"] not in {"gen_ai.request.model", "model"}
    ]
    records[0]["resourceSpans"][0]["scopeSpans"][0]["spans"].append(
        missing_model_request
    )
    _span(records, "0000000000000008")["attributes"].append(
        {"key": "cost_usd", "value": {"doubleValue": 0.00102}}
    )
    store = _write_jsonl(tmp_path / "mixed-request-model.jsonl", records)

    calls: list[tuple[str | None, Usage | None]] = []

    def price(model: str | None, usage: Usage | None) -> PricingResult:
        calls.append((model, usage))
        return PricingResult(usage=usage, usd=1.0, priced=True, reason=None)

    doc = lookup("otel-genai")(AdapterServices(price_usage=price)).emit(
        Selection(stores=(store,))
    )
    main = next(
        attempt
        for attempt in doc["attempts"]
        if attempt["node"].endswith(":claude:session:s-main")
    )
    assert main["cost"]["requests"] == 2
    assert main["cost"]["usd"] == pytest.approx(0.00247)
    assert calls == []
    assert doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]["omissions"] == [
        {"reason": "missing_request_model", "count": 1},
    ]


def test_partial_reported_cost_preserves_measured_component_and_prices_only_uncovered(
    tmp_path,
):
    records = _fixture_records()
    child_request = _span(records, "0000000000000005")
    missing_cost_request = copy.deepcopy(child_request)
    missing_cost_request["spanId"] = "0000000000000091"
    missing_cost_request["startTimeUnixNano"] = "1767225607100000000"
    missing_cost_request["endTimeUnixNano"] = "1767225607200000000"
    missing_cost_request["attributes"] = [
        item
        for item in missing_cost_request["attributes"]
        if item["key"] not in {"cost_usd", "cost_usd_micros"}
    ]
    records[0]["resourceSpans"][0]["scopeSpans"][0]["spans"].append(
        missing_cost_request
    )
    _span(records, "0000000000000008")["attributes"].append(
        {"key": "cost_usd", "value": {"doubleValue": 0.00102}}
    )
    store = _write_jsonl(tmp_path / "mixed-reported-cost.jsonl", records)

    calls: list[tuple[str | None, Usage | None]] = []

    def price(model: str | None, usage: Usage | None) -> PricingResult:
        calls.append((model, usage))
        return PricingResult(usage=usage, usd=2.0, priced=True, reason=None)

    doc = lookup("otel-genai")(AdapterServices(price_usage=price)).emit(
        Selection(stores=(store,))
    )
    child = next(
        attempt
        for attempt in doc["attempts"]
        if attempt["node"].endswith(":claude:agent:a-child")
    )
    assert calls == [
        (
            "claude-opus-5",
            Usage(
                input_tokens=20,
                cached_input_tokens=4,
                cache_creation_tokens=1,
                output_tokens=6,
            ),
        )
    ]
    assert child["cost"]["requests"] == 2
    assert child["cost"]["usd"] == pytest.approx(2.000262)
    assert child["ext"]["dev.dagr.adapter.otel-genai"][
        "reported_cost_usd"
    ] == pytest.approx(0.000262)
    assert doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]["omissions"] == [
        {"reason": "missing_reported_cost", "count": 1}
    ]


def test_mixed_request_models_refuse_pricing_with_explicit_reason(tmp_path):
    records = _fixture_records()
    main_request = _span(records, "0000000000000002")
    main_request["attributes"] = [
        item
        for item in main_request["attributes"]
        if item["key"] not in {"cost_usd", "cost_usd_micros"}
    ]
    other_model_request = copy.deepcopy(main_request)
    other_model_request["spanId"] = "0000000000000092"
    other_model_request["startTimeUnixNano"] = "1767225602100000000"
    other_model_request["endTimeUnixNano"] = "1767225602200000000"
    for key in ("gen_ai.request.model", "model"):
        _attribute(other_model_request, key)["value"] = {
            "stringValue": "claude-sonnet-5"
        }
    records[0]["resourceSpans"][0]["scopeSpans"][0]["spans"].append(
        other_model_request
    )
    _span(records, "0000000000000008")["attributes"].append(
        {"key": "cost_usd", "value": {"doubleValue": 0.00102}}
    )
    store = _write_jsonl(tmp_path / "mixed-request-models.jsonl", records)

    calls: list[tuple[str | None, Usage | None]] = []

    def price(model: str | None, usage: Usage | None) -> PricingResult:
        calls.append((model, usage))
        return PricingResult(usage=usage, usd=1.0, priced=True, reason=None)

    doc = lookup("otel-genai")(AdapterServices(price_usage=price)).emit(
        Selection(stores=(store,))
    )
    main = next(
        attempt
        for attempt in doc["attempts"]
        if attempt["node"].endswith(":claude:session:s-main")
    )
    assert calls == []
    assert "usd" not in main["cost"]
    assert main["ext"]["dev.dagr.adapter.otel-genai"][
        "pricing_refusal_reason"
    ] == "mixed request models"
    assert doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]["omissions"] == [
        {"reason": "missing_reported_cost", "count": 2},
        {"reason": "model_conflict", "count": 1},
        {"reason": "unpriced_attempt", "count": 1},
    ]


def test_unpriced_and_provisional_hook_results_keep_tokens_without_money():
    calls: list[tuple[str | None, Usage | None]] = []

    def price(model: str | None, usage: Usage | None) -> PricingResult:
        calls.append((model, usage))
        if model == "gpt-5.6-sol":
            return PricingResult(
                usage,
                None,
                False,
                "price-table entry is provisional",
                provisional=True,
                estimate_usd=9.5,
            )
        return PricingResult(usage, None, False, "synthetic model is unpriced")

    adapter = lookup("otel-genai")(AdapterServices(price_usage=price))
    doc = adapter.emit(Selection(stores=(FIXTURE,)))
    assert calls == [
        (
            "gpt-5.6-sol",
            Usage(
                input_tokens=150,
                cached_input_tokens=50,
                cache_creation_tokens=0,
                output_tokens=20,
            ),
        )
    ]
    attempts = {attempt["node"]: attempt for attempt in doc["attempts"]}
    assert attempts[f"{PREFIX}:claude:session:s-main"]["cost"]["usd"] == pytest.approx(
        0.001235
    )
    assert attempts[f"{PREFIX}:claude:agent:a-parent"]["cost"]["usd"] == pytest.approx(
        0.0004725
    )
    assert attempts[f"{PREFIX}:claude:agent:a-child"]["cost"]["usd"] == pytest.approx(
        0.000262
    )
    assert "usd" not in attempts[f"{PREFIX}:codex:session:c-review"]["cost"]
    assert all("estimate_usd" not in attempt["cost"] for attempt in doc["attempts"])
    assert doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]["omissions"] == [
        {"reason": "missing_reported_cost", "count": 1},
        {"reason": "unpriced_attempt", "count": 1}
    ]


@pytest.mark.parametrize("session_id", [TRACE_ID, "s-main", "c-review"])
def test_trace_or_session_selection_keeps_the_complete_trace(session_id):
    doc = _emit(session_ids=(session_id,))
    assert len(doc["nodes"]) == 4
    assert len(doc["attempts"]) == 4
    assert len(doc["edges"]) == 3


def test_workspace_and_inclusive_time_filters_apply_to_trace_start():
    for selection in (
        {"workspaces": ("/synthetic/workspace",)},
        {"since": "2026-01-01T00:00:00Z"},
        {"since": "2025-12-31T16:00:00-08:00"},
        {"until": "2026-01-01T00:00:00Z"},
    ):
        assert len(_emit(**selection)["nodes"]) == 4

    for selection in (
        {"workspaces": ("/different/workspace",)},
        {"since": "2026-01-01T00:00:00.000000001Z"},
        {"until": "2025-12-31T23:59:59.999999Z"},
    ):
        doc = _emit(**selection)
        assert len(doc["nodes"]) == len(doc["attempts"]) == len(doc["edges"]) == 0
        assert _errors(doc) == []

    with pytest.raises(ValueError, match="include a timezone"):
        _emit(since="2026-01-01T00:00:00")
    with pytest.raises(ValueError, match="finer than one nanosecond"):
        _emit(since="2026-01-01T00:00:00.0000000001Z")


def test_whole_otlp_object_limit_is_deterministic_and_applied_after_trace_selection(tmp_path):
    store = _whole_object_with_second_trace(tmp_path / "bundle.json")
    selection = Selection(stores=(store,), limit=1)
    first = lookup("otel-genai")().emit(selection)
    second = lookup("otel-genai")().emit(selection)
    assert first == second
    assert first["run"]["id"] == f"otel-genai:{TRACE_ID}"
    assert {node["id"] for node in first["nodes"]} == EXPECTED_NODES
    assert first["run"]["ext"]["dev.dagr.adapter.otel-genai"] == {
        "input_span_count": 16,
        "mapped_span_count": 8,
        "omissions": [
            {"reason": "filtered_trace", "count": 1},
            {"reason": "missing_reported_cost", "count": 1},
            {"reason": "unpriced_attempt", "count": 1},
        ],
    }


def test_launch_requires_cross_harness_root_to_match_tool_execution(tmp_path):
    records = _fixture_records()
    _span(records, "0000000000000008")["parentSpanId"] = "0000000000000002"
    store = _write_jsonl(tmp_path / "bad-launch.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    assert [edge["kind"] for edge in doc["edges"]] == ["spawn", "spawn"]
    codex = next(attempt for attempt in doc["attempts"] if attempt["harness"] == "codex")
    assert "origin" not in codex
    assert _errors(doc) == []


def test_provider_is_omitted_when_only_harness_or_model_prefix_implies_it(tmp_path):
    records = _fixture_records()
    child = _span(records, "0000000000000005")
    child["attributes"] = [
        item for item in child["attributes"] if item["key"] != "gen_ai.system"
    ]
    store = _write_jsonl(tmp_path / "provider-not-reported.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    attempt = next(
        item for item in doc["attempts"] if item["node"].endswith(":agent:a-child")
    )
    assert attempt["model"]["tier"] == "reported"
    assert "provider" not in attempt["model"]
    assert _errors(doc) == []


def test_cross_harness_match_on_nonroot_span_is_not_a_launch(tmp_path):
    records = _fixture_records()
    earlier_codex_root = {
        "traceId": TRACE_ID,
        "spanId": "0000000000000098",
        "name": "codex.session",
        "startTimeUnixNano": "1767225609000000000",
        "endTimeUnixNano": "1767225611600000000",
        "attributes": [
            {"key": "conversation.id", "value": {"stringValue": "c-review"}},
            {"key": "cwd", "value": {"stringValue": "/synthetic/workspace"}},
        ],
        "status": {"code": 1},
    }
    records[1]["resourceSpans"][0]["scopeSpans"][0]["spans"].append(
        earlier_codex_root
    )
    store = _write_jsonl(tmp_path / "nonroot-launch.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    assert [edge["kind"] for edge in doc["edges"]] == ["spawn", "spawn"]
    codex = next(attempt for attempt in doc["attempts"] if attempt["harness"] == "codex")
    assert "origin" not in codex
    assert _errors(doc) == []


@pytest.mark.parametrize("conflicting_alias", [False, True])
def test_invalid_explicit_parent_suppresses_agent_tool_ancestry_fallback(
    tmp_path, conflicting_alias
):
    records = _fixture_records()
    child = _span(records, "0000000000000005")
    _attribute(child, "parent_agent_id")["value"] = {"stringValue": "a-missing"}
    if conflicting_alias:
        child["attributes"].append(
            {
                "key": "gen_ai.agent.parent.id",
                "value": {"stringValue": "a-different"},
            }
        )
    store = _write_jsonl(tmp_path / "invalid-parent.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    spawn_edges = [edge for edge in doc["edges"] if edge["kind"] == "spawn"]
    assert len(spawn_edges) == 1
    assert spawn_edges[0]["to"].endswith(":agent:a-parent")
    assert _errors(doc) == []


def test_source_string_limits_hash_opaque_ids_and_omit_other_oversize_fields(tmp_path):
    records = _fixture_records()
    long_session = "session-" + "s" * 600
    long_workspace = "/synthetic/" + "w" * 600
    for span in _spans(records):
        for attribute in span["attributes"]:
            if attribute["key"] == "session.id":
                attribute["value"] = {"stringValue": long_session}
            elif attribute["key"] == "conversation.id":
                attribute["value"] = {"stringValue": long_session}
            elif attribute["key"] in {"workspace", "cwd"}:
                attribute["value"] = {"stringValue": long_workspace}
    for record in records:
        for resource_span in record["resourceSpans"]:
            _attribute(resource_span["resource"], "process.working_directory")["value"] = {
                "stringValue": long_workspace
            }

    child = _span(records, "0000000000000005")
    _attribute(child, "gen_ai.request.model")["value"] = {"stringValue": "m" * 201}
    _attribute(child, "effort")["value"] = {"stringValue": "e" * 101}
    parent = _span(records, "0000000000000004")
    _attribute(parent, "gen_ai.system")["value"] = {"stringValue": "p" * 101}

    store = _write_jsonl(tmp_path / "long-fields.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    assert _errors(doc) == []
    assert "workspace" not in doc["run"]
    assert all(len(attempt["session"]) <= 500 for attempt in doc["attempts"])
    assert any(attempt["session"].startswith("sha256:") for attempt in doc["attempts"])
    child_attempt = next(
        attempt for attempt in doc["attempts"] if attempt["node"].endswith(":agent:a-child")
    )
    parent_attempt = next(
        attempt for attempt in doc["attempts"] if attempt["node"].endswith(":agent:a-parent")
    )
    codex_attempt = next(attempt for attempt in doc["attempts"] if attempt["harness"] == "codex")
    assert "model" not in child_attempt
    assert "effort" not in child_attempt
    assert "provider" not in parent_attempt["model"]
    assert codex_attempt["origin"]["workspace"] is None
    reasons = {
        item["reason"]
        for item in doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]["omissions"]
    }
    assert "oversize_source_string" in reasons


def test_sessionless_disconnected_roots_stay_distinct_and_start_events_win_ties(tmp_path):
    record = copy.deepcopy(_fixture_records()[0])
    resource_span = record["resourceSpans"][0]
    root = copy.deepcopy(resource_span["scopeSpans"][0]["spans"][0])
    root["attributes"] = []
    root["startTimeUnixNano"] = root["endTimeUnixNano"] = "1767225600000000000"
    other = copy.deepcopy(root)
    other["spanId"] = "0000000000000099"
    resource_span["scopeSpans"][0]["spans"] = [root, other]
    store = _write_jsonl(tmp_path / "two-roots.jsonl", [record])
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    assert len(doc["nodes"]) == len(doc["attempts"]) == 2
    assert {attempt["session"] for attempt in doc["attempts"]} == {
        "0000000000000001",
        "0000000000000099",
    }
    assert [event["type"] for event in doc["events"]] == [
        "attempt_started",
        "attempt_started",
        "attempt_settled",
        "attempt_settled",
    ]
    assert _errors(doc) == []


def test_only_recognized_requests_are_costed_and_effort_uses_all_owned_spans(tmp_path):
    records = _fixture_records()
    child_request = _span(records, "0000000000000005")
    child_request["attributes"] = [
        item for item in child_request["attributes"] if item["key"] != "effort"
    ]
    extra = {
        "traceId": TRACE_ID,
        "spanId": "0000000000000009",
        "parentSpanId": "0000000000000005",
        "name": "claude_code.tool",
        "startTimeUnixNano": "1767225606000000000",
        "endTimeUnixNano": "1767225606000000000",
        "attributes": [
            {"key": "session.id", "value": {"stringValue": "s-main"}},
            {"key": "agent_id", "value": {"stringValue": "a-child"}},
            {"key": "effort", "value": {"stringValue": "high"}},
            {"key": "input_tokens", "value": {"intValue": "999999"}},
            {"key": "output_tokens", "value": {"intValue": "999999"}},
        ],
        "status": {"code": 1},
    }
    records[0]["resourceSpans"][0]["scopeSpans"][0]["spans"].append(extra)
    store = _write_jsonl(tmp_path / "non-request-tokens.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    attempt = next(
        item for item in doc["attempts"] if item["node"].endswith(":agent:a-child")
    )
    assert attempt["effort"] == "high"
    assert attempt["cost"]["requests"] == 1
    assert attempt["cost"]["input_tokens"] == 20
    assert attempt["cost"]["output_tokens"] == 6


def test_nonfinite_fractional_and_negative_cost_numbers_are_not_coerced(tmp_path):
    records = _fixture_records()
    child = _span(records, "0000000000000005")
    _attribute(child, "input_tokens")["value"] = {"doubleValue": 1.5}
    _attribute(child, "cost_usd")["value"] = {"stringValue": "NaN"}
    child["attributes"].append(
        {"key": "cost_usd_micros", "value": {"doubleValue": -1.25}}
    )
    store = _write_jsonl(tmp_path / "invalid-numbers.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    attempt = next(
        item for item in doc["attempts"] if item["node"].endswith(":agent:a-child")
    )
    assert "input_tokens" not in attempt["cost"]
    assert "ext" not in attempt
    omissions = doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]["omissions"]
    assert next(item["count"] for item in omissions if item["reason"] == "dropped_source_data") >= 3
    assert _errors(doc) == []


def test_non_list_span_attributes_are_malformed_and_not_counted_as_valid(tmp_path):
    records = _fixture_records()
    _span(records, "0000000000000002")["attributes"] = {"not": "an attribute list"}
    store = _write_jsonl(tmp_path / "non-list-attributes.jsonl", records)
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    extension = doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]
    assert extension["input_span_count"] == 7
    assert extension["mapped_span_count"] == 7
    assert extension["omissions"] == [
        {"reason": "malformed_record", "count": 1},
        {"reason": "missing_reported_cost", "count": 1},
        {"reason": "unpriced_attempt", "count": 1},
    ]
    assert len(doc["nodes"]) == 4
    assert len(doc["edges"]) == 3
    assert _errors(doc) == []


def test_unmapped_trace_does_not_change_run_identity_groups_or_timestamps(tmp_path):
    records = _fixture_records()
    unsupported = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "unknown-harness"}}
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "synthetic.unknown"},
                        "spans": [
                            {
                                "traceId": "3" * 32,
                                "spanId": "0000000000000033",
                                "name": "unknown.operation",
                                "startTimeUnixNano": "1767225500000000000",
                                "endTimeUnixNano": "1767225700000000000",
                                "attributes": [],
                                "status": {"code": 1},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    store = _write_jsonl(tmp_path / "unsupported-trace.jsonl", records + [unsupported])
    doc = lookup("otel-genai")().emit(Selection(stores=(store,)))
    assert doc["run"]["id"] == f"otel-genai:{TRACE_ID}"
    assert doc["run"]["started_at"] == "2026-01-01T00:00:00.000000Z"
    assert doc["run"]["ended_at"] == "2026-01-01T00:00:12.000000Z"
    assert doc["groups"] == [{"id": f"trace:{TRACE_ID}", "title": "OpenTelemetry trace"}]
    extension = doc["run"]["ext"]["dev.dagr.adapter.otel-genai"]
    assert extension["input_span_count"] == 9
    assert extension["mapped_span_count"] == 8
    assert {item["reason"] for item in extension["omissions"]} >= {"unsupported_signal"}


def test_cli_uses_registered_adapter_and_explicit_store(capsys):
    assert main(["adapt", "otel-genai", "--store", str(FIXTURE), "--session", "s-main"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["ocp"] == "0.2"
    assert len(doc["nodes"]) == 4
