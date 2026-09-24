"""OTLP/JSON GenAI spans to OCP v0.2.

An explicit ``parent_agent_id`` or exact Claude Agent/Task span ancestry is a
``verified`` spawn relation. An exact cross-harness ``traceId`` plus
``parentSpanId`` match is a ``verified`` launch relation. This adapter never
infers a relation from timing, text, commands, paths, or a shared workspace.

Claude Code 2.1.257 emitted ``agent_id`` in the lane's minimal capture but did
not emit ``parent_agent_id``, ``effort``, or ``cost_usd``. Those optional beta
attributes are supported when present. Source ``cost_usd`` remains an
extension cross-check because the OTel convention defines no cost attribute.
Settled core USD preserves valid source-reported request costs and uses the
portable base's injected pricing hook only for uncovered requests. Unpriced
and provisional attempts are counted rather than defaulted to zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys as _sys
from collections import Counter
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from importlib import import_module
from pathlib import Path
from types import ModuleType as _ModuleType
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from .base import Adapter, Selection, Usage
from .registry import register


# Load the split implementation without widening this compatibility module's
# statically allowed dependency surface (stdlib plus the portable contract).
_reexports = {
    "otel_genai_types": (
        "EXT",
        "_TRACE_ID",
        "_SPAN_ID",
        "_RFC3339",
        "_SESSION_KEYS",
        "_WORKSPACE_KEYS",
        "_AGENT_KEYS",
        "_PARENT_AGENT_KEYS",
        "_MODEL_KEYS",
        "_EFFORT_KEYS",
        "_Span",
        "_Trace",
        "_NodeKey",
        "_NodeSource",
        "_Omissions",
        "_string",
        "_nonnegative_integer",
        "_nonnegative_decimal",
        "_first_string",
        "_alias_string",
        "_has_alias",
        "_bounded_source_string",
        "_harness",
        "_timestamp",
        "_node_id",
    ),
    "otel_genai_parse": (
        "_decode_value",
        "_decode_attributes",
        "_file_records",
        "_valid_id",
        "_nanoseconds",
        "_read_spans",
    ),
    "otel_genai_graph": (
        "_traces",
        "_bound_ns",
        "_select_traces",
        "_main_identity",
        "_source_session",
        "_owners",
        "_agent_relations",
        "_launch_relations",
        "_paths",
    ),
    "otel_genai_usage": (
        "_alias_number",
        "_request_usage",
        "_is_request",
        "_reported_usd",
        "_model",
        "_attempt_cost_and_fields",
        "_provider",
    ),
}
for _module_name, _names in _reexports.items():
    _module = import_module(f"{__package__}.{_module_name}")
    for _name in _names:
        globals()[_name] = getattr(_module, _name)
_otel_types = import_module(f"{__package__}.otel_genai_types")
_otel_parse = import_module(f"{__package__}.otel_genai_parse")
_otel_usage = import_module(f"{__package__}.otel_genai_usage")
_otel_graph = import_module(f"{__package__}.otel_genai_graph")
del _module_name, _names, _module, _name, _reexports, import_module


@register
class OtelGenAIAdapter(Adapter):
    """Convert explicit OTLP/JSON trace files into one OCP document."""

    name = "otel-genai"

    def discover(self) -> Sequence[Path]:
        return ()

    def sessions(self) -> Sequence[Mapping[str, Any]]:
        return ()

    def fixture_selection(
        self, fixture_dir: Path, /
    ) -> AbstractContextManager[Selection]:
        return nullcontext(Selection(stores=(fixture_dir / "trace.otlp.jsonl",)))

    def _price_cost(
        self,
        cost: dict[str, Any] | None,
        model: str | None,
        omissions: _Omissions,
    ) -> dict[str, Any] | None:
        if cost is None:
            return None
        usage_fields = (
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_tokens",
            "output_tokens",
        )
        usage = (
            Usage(*(cost[field] for field in usage_fields), basis="measured")
            if all(field in cost for field in usage_fields)
            else None
        )
        pricing = self.price_usage(model, usage)
        normalized = pricing.to_ocp()
        if normalized is not None:
            normalized["requests"] = cost["requests"]
            if "reasoning_tokens" in cost:
                normalized["reasoning_tokens"] = cost["reasoning_tokens"]
            cost = normalized
        if not pricing.priced:
            # PricingResult deliberately keeps provisional estimates and all
            # other unpriced money out of OCP while preserving complete usage.
            omissions.add("unpriced_attempt")
        return cost

    def emit(self, selection: Selection) -> dict[str, Any]:
        omissions = _Omissions()
        paths = _paths(selection.stores)
        spans, input_span_count = _read_spans(paths, omissions)
        selected_traces = _select_traces(_traces(spans, omissions), selection, omissions)
        nodes, owners, all_spans = _owners(selected_traces, omissions)
        spawn_relations = _agent_relations(nodes, owners, all_spans, omissions)
        launch_relations = _launch_relations(owners, all_spans, omissions)

        node_ids = {key: _node_id(key, omissions) for key in nodes}
        attempt_ids = {key: f"{node_ids[key]}.a1" for key in nodes}
        launch_parent = {child: parent for parent, child in launch_relations}

        ocp_nodes: list[dict[str, Any]] = []
        attempts: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        node_workspaces: dict[_NodeKey, str | None] = {}
        for key, source in sorted(nodes.items(), key=lambda item: node_ids[item[0]]):
            node_id = node_ids[key]
            workspaces = sorted(
                {
                    workspace
                    for span in source.spans
                    if (
                        workspace := _alias_string(span, _WORKSPACE_KEYS, omissions)
                    )
                    is not None
                }
            )
            if len(workspaces) > 1:
                omissions.add("conflicting_attribute")
                workspace = None
            else:
                workspace = workspaces[0] if workspaces else None
            workspace = _bounded_source_string(workspace, 500, omissions)
            node_workspaces[key] = workspace
            ocp_nodes.append(
                {
                    "id": node_id,
                    "kind": "impl",
                    "title": f"{key.harness} agent",
                    "group": f"trace:{key.trace_id}",
                    "state": "settled_unverified",
                    "labels": {"harness": key.harness},
                }
            )

            (
                cost,
                reported_cost,
                models,
                efforts,
                pricing_model,
                pricing_refusal_reason,
                uncovered_cost,
            ) = _attempt_cost_and_fields(source, omissions)
            if cost is not None:
                if uncovered_cost is None:
                    if reported_cost is not None:
                        cost["usd"] = reported_cost
                elif pricing_model is None:
                    omissions.add("unpriced_attempt")
                else:
                    priced_uncovered = self._price_cost(
                        uncovered_cost, pricing_model, omissions
                    )
                    reported_component = (
                        0.0
                        if uncovered_cost["requests"] == cost["requests"]
                        else reported_cost
                    )
                    if (
                        priced_uncovered is not None
                        and "usd" in priced_uncovered
                        and reported_component is not None
                    ):
                        cost["usd"] = math.fsum(
                            (reported_component, priced_uncovered["usd"])
                        )
            started_at = _timestamp(min(span.start_ns for span in source.spans))
            ended_at = _timestamp(max(span.end_ns for span in source.spans))
            attempt: dict[str, Any] = {
                "id": attempt_ids[key],
                "node": node_id,
                "n": 1,
                "actor": "agent",
                "harness": key.harness,
                "cause": {"type": "initial"},
                "status": "settled_unverified",
                "started_at": started_at,
                "ended_at": ended_at,
                "outcome": {
                    "result": "settled_unverified",
                    "evidence": "heuristic",
                },
                "session": _source_session(source, omissions)
                or _bounded_source_string(
                    key.identity, 500, omissions, hash_opaque=True
                ),
            }
            if len(models) == 1:
                raw_model = _bounded_source_string(models[0], 200, omissions)
                if raw_model is not None:
                    model_ref: dict[str, Any] = {
                        "raw": raw_model,
                        "tier": "reported",
                    }
                    if raw_model.startswith(("claude-", "gpt-")):
                        model_ref["id"] = raw_model
                    provider = _provider(source, omissions)
                    if provider:
                        model_ref["provider"] = provider
                    attempt["model"] = model_ref
            if len(efforts) == 1:
                effort = _bounded_source_string(efforts[0], 100, omissions)
                if effort is not None:
                    attempt["effort"] = effort
            if cost is not None:
                attempt["cost"] = cost
            attempt_extension: dict[str, Any] = {}
            if reported_cost is not None:
                attempt_extension["reported_cost_usd"] = reported_cost
            if uncovered_cost is not None and pricing_refusal_reason is not None:
                attempt_extension["pricing_refusal_reason"] = pricing_refusal_reason
            if attempt_extension:
                attempt["ext"] = {EXT: attempt_extension}

            if key in launch_parent:
                parent = launch_parent[key]
                attempt["origin"] = {
                    "launched_by": attempt_ids[parent],
                    "workspace": workspace,
                    "external": False,
                    "how": "W3C trace context",
                    "tier": "verified",
                    "evidence": "OTLP traceId/parentSpanId linkage",
                }
            attempts.append(attempt)
            events.extend(
                [
                    {
                        "at": started_at,
                        "type": "attempt_started",
                        "node": node_id,
                        "attempt": attempt_ids[key],
                    },
                    {
                        "at": ended_at,
                        "type": "attempt_settled",
                        "node": node_id,
                        "attempt": attempt_ids[key],
                    },
                ]
            )

        edges: list[dict[str, Any]] = []
        for parent, child, evidence in spawn_relations:
            edges.append(
                {
                    "from": node_ids[parent],
                    "to": node_ids[child],
                    "kind": "spawn",
                    "tier": "verified",
                    "from_attempt": attempt_ids[parent],
                    "to_attempt": attempt_ids[child],
                    "evidence": evidence,
                }
            )
        for parent, child in launch_relations:
            edges.append(
                {
                    "from": node_ids[parent],
                    "to": node_ids[child],
                    "kind": "launch",
                    "tier": "verified",
                    "from_attempt": attempt_ids[parent],
                    "to_attempt": attempt_ids[child],
                    "evidence": "OTLP child parentSpanId matched launcher spanId in one trace",
                }
            )
        edges.sort(key=lambda edge: (edge["kind"], edge["from"], edge["to"]))
        event_rank = {"attempt_started": 0, "attempt_settled": 1}
        events.sort(
            key=lambda event: (
                event["at"],
                event_rank.get(event["type"], 2),
                event["node"],
                event.get("attempt", ""),
            )
        )

        trace_ids = sorted({key.trace_id for key in nodes})
        if not trace_ids:
            run_id = "otel-genai:empty"
        elif len(trace_ids) == 1:
            run_id = f"otel-genai:{trace_ids[0]}"
        else:
            digest = hashlib.sha256("\n".join(sorted(trace_ids)).encode()).hexdigest()
            run_id = f"otel-genai:bundle:{digest}"
        run: dict[str, Any] = {
            "id": run_id,
            "title": "OpenTelemetry GenAI trace export",
            "ext": {
                EXT: {
                    "input_span_count": input_span_count,
                    "mapped_span_count": sum(
                        len(source.spans) for source in nodes.values()
                    ),
                    "omissions": omissions.records(),
                }
            },
        }
        mapped_spans = [span for source in nodes.values() for span in source.spans]
        if mapped_spans:
            run["started_at"] = _timestamp(min(span.start_ns for span in mapped_spans))
            run["ended_at"] = _timestamp(max(span.end_ns for span in mapped_spans))
        mapped_workspaces = list(node_workspaces.values())
        if (
            mapped_workspaces
            and all(mapped_workspaces)
            and len(set(mapped_workspaces)) == 1
        ):
            run["workspace"] = mapped_workspaces[0]

        harnesses = sorted({key.harness for key in nodes})
        return {
            "ocp": "0.2",
            "producer": {
                "name": "dagr-adapter-otel-genai",
                "framework": "+".join(harnesses) if harnesses else "unknown",
                "source_contract": "OTLP/JSON file exporter",
                "capabilities": {
                    "groups": True, "events": True, "artifacts": False, "edges_dep": False,
                    "edges_spawn": True, "edges_launch": True, "edges_artifact": False,
                    "cost_usd": True, "cost_tokens": True, "outcome_evidence": True,
                },
            },
            "privacy": {
                "profile": "metadata_only",
                "note": "Prompt, response, command, tool input, and account identity attributes are omitted",
            },
            "run": run,
            "groups": [
                {"id": f"trace:{trace_id}", "title": "OpenTelemetry trace"}
                for trace_id in sorted(trace_ids)
            ],
            "nodes": sorted(ocp_nodes, key=lambda node: node["id"]),
            "edges": edges,
            "attempts": sorted(attempts, key=lambda attempt: attempt["id"]),
            "artifacts": [],
            "events": events,
        }


class _OtelGenAIModule(_ModuleType):
    def __setattr__(
        self,
        name: str,
        value: Any,
        _targets={
            "Any": (_otel_types, _otel_parse, _otel_usage),
            "Counter": (_otel_types,),
            "Decimal": (_otel_types, _otel_usage),
            "InvalidOperation": (_otel_types,),
            "Iterable": (_otel_types, _otel_usage),
            "Mapping": (_otel_types, _otel_parse, _otel_graph),
            "_NodeKey": (_otel_types, _otel_graph),
            "_Omissions": (_otel_types, _otel_parse, _otel_usage, _otel_graph),
            "_Span": (_otel_types, _otel_parse, _otel_usage, _otel_graph),
            "_first_string": (_otel_types,),
            "_string": (_otel_types, _otel_usage),
            "datetime": (_otel_types, _otel_graph),
            "hashlib": (_otel_types,),
            "math": (_otel_types, _otel_usage),
            "quote": (_otel_types,),
            "re": (_otel_types, _otel_parse, _otel_graph),
            "timezone": (_otel_types,),
            "Path": (_otel_parse, _otel_graph),
            "Sequence": (_otel_parse, _otel_graph),
            "_SPAN_ID": (_otel_parse,),
            "_TRACE_ID": (_otel_parse,),
            "_decode_attributes": (_otel_parse,),
            "_decode_value": (_otel_parse,),
            "_file_records": (_otel_parse,),
            "_nanoseconds": (_otel_parse,),
            "_timestamp": (_otel_parse,),
            "_valid_id": (_otel_parse,),
            "json": (_otel_parse,),
            "_EFFORT_KEYS": (_otel_usage,),
            "_MODEL_KEYS": (_otel_usage,),
            "_NodeSource": (_otel_usage, _otel_graph),
            "_alias_number": (_otel_usage,),
            "_alias_string": (_otel_usage, _otel_graph),
            "_bounded_source_string": (_otel_usage, _otel_graph),
            "_is_request": (_otel_usage,),
            "_model": (_otel_usage,),
            "_nonnegative_decimal": (_otel_usage,),
            "_nonnegative_integer": (_otel_usage,),
            "_reported_usd": (_otel_usage,),
            "_request_usage": (_otel_usage,),
            "Selection": (_otel_graph,),
            "_AGENT_KEYS": (_otel_graph,),
            "_PARENT_AGENT_KEYS": (_otel_graph,),
            "_RFC3339": (_otel_graph,),
            "_SESSION_KEYS": (_otel_graph,),
            "_Trace": (_otel_graph,),
            "_WORKSPACE_KEYS": (_otel_graph,),
            "_bound_ns": (_otel_graph,),
            "_harness": (_otel_graph,),
            "_has_alias": (_otel_graph,),
            "_main_identity": (_otel_graph,),
            "timedelta": (_otel_graph,),
        },
    ) -> None:
        for target in _targets.get(name, ()):
            setattr(target, name, value)
        super().__setattr__(name, value)


_otel_module = _sys.modules[__name__]
_otel_module.__class__ = _OtelGenAIModule
del _ModuleType, _OtelGenAIModule, _otel_graph, _otel_module, _otel_parse
del _otel_types, _otel_usage, _sys
