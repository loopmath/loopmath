"""Usage, model, and cost extraction for the OTLP/JSON GenAI adapter."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Iterable

from .otel_genai_types import (
    _EFFORT_KEYS,
    _MODEL_KEYS,
    _NodeSource,
    _Omissions,
    _Span,
    _alias_string,
    _bounded_source_string,
    _nonnegative_decimal,
    _nonnegative_integer,
    _string,
)


def _alias_number(
    span: _Span, keys: Iterable[str], omissions: _Omissions
) -> int | None:
    values: set[int] = set()
    for key in keys:
        if key not in span.attrs:
            continue
        value = _nonnegative_integer(span.attrs[key])
        if value is None:
            omissions.add("dropped_source_data")
            continue
        values.add(value)
    if len(values) > 1:
        omissions.add("dropped_source_data")
        return None
    return next(iter(values)) if values else None


def _request_usage(
    span: _Span, harness: str, omissions: _Omissions
) -> dict[str, int | None]:
    cached = _alias_number(
        span,
        (
            "cache_read_tokens",
            "cached_input_tokens",
            "codex.turn.token_usage.cached_input_tokens",
            "gen_ai.usage.cache_read.input_tokens",
        ),
        omissions,
    )
    if harness == "codex":
        input_tokens = _alias_number(
            span,
            ("codex.turn.token_usage.non_cached_input_tokens",),
            omissions,
        )
        if input_tokens is None:
            total = _alias_number(
                span,
                ("codex.turn.token_usage.input_tokens", "gen_ai.usage.input_tokens"),
                omissions,
            )
            if total is not None and cached is not None:
                if cached > total:
                    omissions.add("dropped_source_data")
                else:
                    input_tokens = total - cached
    else:
        input_tokens = _alias_number(
            span, ("input_tokens", "gen_ai.usage.input_tokens"), omissions
        )
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "cache_creation_tokens": _alias_number(
            span,
            (
                "cache_creation_tokens",
                "cache_write_input_tokens",
                "codex.turn.token_usage.cache_write_input_tokens",
                "gen_ai.usage.cache_write.input_tokens",
            ),
            omissions,
        ),
        "output_tokens": _alias_number(
            span,
            (
                "output_tokens",
                "codex.turn.token_usage.output_tokens",
                "gen_ai.usage.output_tokens",
            ),
            omissions,
        ),
        "reasoning_tokens": _alias_number(
            span,
            (
                "reasoning_tokens",
                "codex.turn.token_usage.reasoning_output_tokens",
                "codex.usage.reasoning_output_tokens",
            ),
            omissions,
        ),
    }


def _is_request(span: _Span) -> bool:
    if span.name in {
        "claude_code.llm_request",
        "codex.api_request",
        "gen_ai.client.inference",
    }:
        return True
    marker = _string(span.attrs.get("span.type"))
    if marker in {"llm_request", "gen_ai.request"}:
        return True
    operation = _string(span.attrs.get("gen_ai.operation.name"))
    return operation in {"chat", "text_completion", "generate_content", "responses"}


def _reported_usd(span: _Span, omissions: _Omissions) -> float | None:
    direct_present = "cost_usd" in span.attrs
    micros_present = "cost_usd_micros" in span.attrs
    direct = _nonnegative_decimal(span.attrs.get("cost_usd"))
    micros = _nonnegative_integer(span.attrs.get("cost_usd_micros"))
    if direct_present and direct is None:
        omissions.add("dropped_source_data")
    if micros_present and micros is None:
        omissions.add("dropped_source_data")
    converted = Decimal(micros) / Decimal(1_000_000) if micros is not None else None
    if (
        direct is not None
        and converted is not None
        and abs(direct - converted) > Decimal("0.0000005")
    ):
        omissions.add("dropped_source_data")
        return None
    if direct is not None:
        chosen = direct
    else:
        chosen = converted
    if chosen is None:
        return None
    result = float(chosen)
    if not math.isfinite(result):
        omissions.add("dropped_source_data")
        return None
    return result


def _model(span: _Span, omissions: _Omissions) -> str | None:
    values: set[str] = set()
    for key in _MODEL_KEYS:
        if key not in span.attrs and key not in span.resource:
            continue
        value = _string(span.attr((key,)))
        if value is None:
            omissions.add("dropped_source_data")
            continue
        values.add(value)
    if len(values) > 1:
        omissions.add("model_conflict")
        return None
    return next(iter(values)) if values else None


def _attempt_cost_and_fields(
    source: _NodeSource, omissions: _Omissions
) -> tuple[
    dict[str, Any] | None,
    float | None,
    list[str],
    list[str],
    str | None,
    str | None,
    dict[str, Any] | None,
]:
    harness = source.key.harness
    requests = [span for span in source.spans if _is_request(span)]
    request_models = [_model(span, omissions) for span in requests]
    models = sorted({model for model in request_models if model is not None})
    missing_request_models = sum(model is None for model in request_models)
    omissions.add("missing_request_model", missing_request_models)
    pricing_model = (
        models[0]
        if len(models) == 1 and missing_request_models == 0
        else None
    )
    if len(models) > 1:
        pricing_refusal_reason = "mixed request models"
    elif missing_request_models:
        pricing_refusal_reason = "missing request model"
    else:
        pricing_refusal_reason = None
    efforts = sorted(
        {
            effort
            for span in source.spans
            if (effort := _alias_string(span, _EFFORT_KEYS, omissions)) is not None
        }
    )
    if len(models) > 1:
        omissions.add("model_conflict")
    if len(efforts) > 1:
        omissions.add("effort_conflict")
    reported_values = [_reported_usd(span, omissions) for span in requests]
    missing_reported_costs = sum(value is None for value in reported_values)
    omissions.add("missing_reported_cost", missing_reported_costs)
    valid_reported_values = [
        value for value in reported_values if value is not None
    ]
    reported = math.fsum(valid_reported_values) if valid_reported_values else None
    if reported is not None and not math.isfinite(reported):
        omissions.add("dropped_source_data")
        reported = None
    if not requests:
        return (
            None,
            reported,
            models,
            efforts,
            pricing_model,
            pricing_refusal_reason,
            None,
        )

    usages = [_request_usage(span, harness, omissions) for span in requests]

    def aggregate(selected: list[dict[str, int | None]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requests": len(selected),
            "basis": "measured",
        }
        for field in (
            "input_tokens",
            "cached_input_tokens",
            "cache_creation_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            values = [usage[field] for usage in selected]
            if all(value is not None for value in values):
                result[field] = int(
                    sum(value for value in values if value is not None)
                )
        return result

    uncovered_usages = [
        usage
        for usage, reported_value in zip(usages, reported_values)
        if reported_value is None
    ]
    uncovered_cost = aggregate(uncovered_usages) if uncovered_usages else None
    return (
        aggregate(usages),
        reported,
        models,
        efforts,
        pricing_model,
        pricing_refusal_reason,
        uncovered_cost,
    )


def _provider(source: _NodeSource, omissions: _Omissions) -> str | None:
    reported = {
        provider
        for span in source.spans
        if (
            provider := _alias_string(
                span, ("gen_ai.provider.name", "gen_ai.system"), omissions
            )
        )
    }
    if len(reported) > 1:
        omissions.add("conflicting_attribute")
        return None
    if len(reported) == 1:
        return _bounded_source_string(next(iter(reported)), 100, omissions)
    return None
