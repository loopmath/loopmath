"""Task, message, model, and usage facts for the OMP adapter."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from .base import Usage
from .omp_parse import _message, _nonempty_string, _record_timestamp
from .omp_types import (
    _KNOWN_MESSAGE_ROLES,
    _TOKEN_FIELDS,
    _NativeSession,
    _TaskFacts,
    _TaskJob,
    _UsageFacts,
    _UsageGroup,
)


def _task_facts(session: _NativeSession) -> _TaskFacts:
    findings: Counter[str] = Counter()
    calls: dict[str, datetime | None] = {}
    call_records = 0
    for record in session.records:
        message = _message(record)
        if message is None or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (
                isinstance(item, dict)
                and item.get("type") == "toolCall"
                and item.get("name") == "task"
            ):
                continue
            call_records += 1
            call_id = _nonempty_string(item.get("id"))
            if call_id is None:
                findings["malformed_task_call"] += 1
            elif call_id in calls:
                findings["duplicate_task_call_id"] += 1
            else:
                calls[call_id] = _record_timestamp(record)

    linked_result_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    linked_async_result_ids: set[str] = set()
    jobs: dict[str, dict[str, Any]] = {}

    for record in session.records:
        message = _message(record)
        if not (
            message is not None
            and message.get("role") == "toolResult"
            and message.get("toolName") == "task"
        ):
            continue
        call_id = _nonempty_string(message.get("toolCallId"))
        linked = call_id is not None and call_id in calls
        if call_id is None:
            findings["malformed_task_result"] += 1
            findings["unmatched_task_result"] += 1
        else:
            if call_id in seen_result_ids:
                findings["duplicate_task_result"] += 1
            seen_result_ids.add(call_id)
            if linked:
                linked_result_ids.add(call_id)
            else:
                findings["unmatched_task_result"] += 1

        details = message.get("details")
        if not isinstance(details, dict):
            findings["malformed_task_details"] += 1
            continue

        progress_value = details.get("progress")
        results_value = details.get("results")
        malformed_collections = 0
        if progress_value is None:
            progress: list[object] = []
        elif isinstance(progress_value, list):
            progress = progress_value
        else:
            progress = []
            malformed_collections += 1
        if isinstance(results_value, list):
            results = results_value
        else:
            results = []
            malformed_collections += 1
        findings["malformed_task_details"] += malformed_collections

        progress_ids: set[str] = set()
        for collection_name, collection in (("progress", progress), ("results", results)):
            for item in collection:
                if not isinstance(item, dict):
                    findings["malformed_task_job"] += 1
                    continue
                job_id = _nonempty_string(item.get("id"))
                task = _nonempty_string(item.get("task"))
                agent = _nonempty_string(item.get("agent"))
                if job_id is None or task is None or agent is None:
                    findings["malformed_task_job"] += 1
                    continue
                if collection_name == "progress":
                    progress_ids.add(job_id)
                existing = jobs.get(job_id)
                if existing is None:
                    jobs[job_id] = {
                        "task": task,
                        "agent": agent,
                        "linked": linked,
                        "called_at": calls.get(call_id) if linked else None,
                    }
                else:
                    findings["duplicate_task_job"] += 1
                    if existing["task"] != task or existing["agent"] != agent:
                        findings["malformed_task_job"] += 1
                        continue
                    if linked:
                        existing["linked"] = True
                        called_at = calls.get(call_id)
                        if existing["called_at"] is None or (
                            called_at is not None and called_at < existing["called_at"]
                        ):
                            existing["called_at"] = called_at

        async_value = details.get("async")
        if async_value is not None:
            if not isinstance(async_value, dict):
                findings["malformed_async_job_id"] += 1
            else:
                async_job_id = _nonempty_string(async_value.get("jobId"))
                if async_job_id is None:
                    findings["malformed_async_job_id"] += 1
                elif async_job_id not in progress_ids:
                    findings["unmatched_async_job_id"] += 1
                elif linked and call_id is not None:
                    linked_async_result_ids.add(call_id)

    findings["unmatched_task_call"] += len(set(calls) - linked_result_ids)
    task_jobs: list[_TaskJob] = []
    for job_id, job in sorted(jobs.items()):
        if not job["linked"]:
            findings["unmatched_task_job"] += 1
            continue
        task_jobs.append(
            _TaskJob(
                job_id=job_id,
                task=job["task"],
                agent=job["agent"],
                called_at=job["called_at"],
            )
        )
    return _TaskFacts(
        calls=call_records,
        linked_results=len(linked_result_ids),
        linked_jobs=len(linked_async_result_ids),
        task_jobs=len(jobs),
        jobs=tuple(task_jobs),
        findings=dict(findings),
    )


def _message_role_counts(session: _NativeSession) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in session.records:
        message = _message(record)
        if message is None:
            continue
        role = message.get("role")
        label = (
            role
            if isinstance(role, str) and role in _KNOWN_MESSAGE_ROLES
            else "other"
        )
        counts[label] += 1
    return dict(sorted(counts.items()))


def _parent_link_counts(session: _NativeSession) -> tuple[int, int, int]:
    record_ids = {record["id"] for record in session.records}
    linked = 0
    unresolved = 0
    malformed = 0
    for record in session.records:
        parent_id = record.get("parentId")
        if parent_id is None:
            continue
        if not isinstance(parent_id, str):
            malformed += 1
            unresolved += 1
        elif parent_id in record_ids:
            linked += 1
        else:
            unresolved += 1
    return linked, unresolved, malformed


def _model(session: _NativeSession) -> dict[str, str] | None:
    raw: str | None = None
    for record in session.records:
        if record.get("type") == "model_change" and isinstance(record.get("model"), str):
            raw = record["model"]
    if raw is None and session.session_init is not None:
        value = session.session_init.get("resolvedModel")
        if isinstance(value, str):
            raw = value
    if raw is None or len(raw) > 200:
        return None
    model: dict[str, str] = {"raw": raw, "id": raw, "tier": "reported"}
    if "/" in raw:
        provider = raw.split("/", 1)[0]
        if provider and len(provider) <= 100:
            model["provider"] = provider
    return model


def _effort(session: _NativeSession) -> str | None:
    effort: str | None = None
    for record in session.records:
        if record.get("type") != "thinking_level_change":
            continue
        value = record.get("thinkingLevel", record.get("level"))
        if isinstance(value, str) and len(value) <= 100:
            effort = value
    return effort


def _usage_facts(session: _NativeSession) -> _UsageFacts:
    calls: list[tuple[object, object, object]] = []
    for record in session.records:
        message = _message(record)
        if message is not None and message.get("role") == "assistant":
            calls.append(
                (message.get("provider"), message.get("model"), message.get("usage"))
            )
        elif record.get("type") == "model_usage":
            calls.append(
                (record.get("provider"), record.get("model"), record.get("usage"))
            )

    findings: Counter[str] = Counter()
    groups: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    complete_calls = 0
    for provider_value, model_value, usage_value in calls:
        call_reasons: Counter[str] = Counter()
        provider = _nonempty_string(provider_value)
        if provider is None:
            call_reasons[
                "provider_missing" if provider_value is None else "provider_malformed"
            ] += 1
        elif len(provider) > 100:
            provider = None
            call_reasons["provider_oversized"] += 1
        model = _nonempty_string(model_value)
        if model is None:
            call_reasons[
                "model_missing" if model_value is None else "model_malformed"
            ] += 1
        elif len(model) > 200:
            model = None
            call_reasons["model_oversized"] += 1

        token_values: dict[str, int] = {}
        usage_reasons: Counter[str] = Counter()
        if usage_value is None:
            usage_reasons["usage_missing"] += 1
        elif not isinstance(usage_value, dict):
            usage_reasons["usage_not_object"] += 1
        else:
            for source, target in _TOKEN_FIELDS.items():
                value = usage_value.get(source)
                if value is None:
                    usage_reasons[f"usage_{source}_missing"] += 1
                elif isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    usage_reasons[f"usage_{source}_invalid"] += 1
                else:
                    token_values[target] = value
            total = usage_value.get("totalTokens")
            if total is None:
                usage_reasons["usage_totalTokens_missing"] += 1
            elif isinstance(total, bool) or not isinstance(total, int) or total < 0:
                usage_reasons["usage_totalTokens_invalid"] += 1
            elif len(token_values) == len(_TOKEN_FIELDS) and total != sum(
                token_values.values()
            ):
                usage_reasons["usage_totalTokens_mismatch"] += 1

        key = (provider, model)
        group = groups.setdefault(
            key,
            {
                "calls": 0,
                "complete_calls": 0,
                "tokens": Counter(),
                "reasons": Counter(),
            },
        )
        group["calls"] += 1
        all_reasons = call_reasons + usage_reasons
        findings.update(all_reasons)
        group["reasons"].update(all_reasons)
        if not usage_reasons:
            complete_calls += 1
            group["complete_calls"] += 1
            group["tokens"].update(token_values)

    usage_groups: list[_UsageGroup] = []
    for (provider, model), group in sorted(
        groups.items(), key=lambda item: (item[0][0] or "", item[0][1] or "")
    ):
        usage = None
        if group["complete_calls"] == group["calls"]:
            usage = Usage(
                input_tokens=group["tokens"]["input_tokens"],
                cached_input_tokens=group["tokens"]["cached_input_tokens"],
                cache_creation_tokens=group["tokens"]["cache_creation_tokens"],
                output_tokens=group["tokens"]["output_tokens"],
            )
        usage_groups.append(
            _UsageGroup(
                provider=provider,
                model=model,
                calls=group["calls"],
                complete_calls=group["complete_calls"],
                unknown_calls=group["calls"] - group["complete_calls"],
                usage=usage,
                known_tokens=dict(group["tokens"]),
                unknown_reasons=dict(group["reasons"]),
            )
        )

    aggregate = None
    if calls and complete_calls == len(calls):
        totals: Counter[str] = Counter()
        for group in groups.values():
            totals.update(group["tokens"])
        aggregate = Usage(
            input_tokens=totals["input_tokens"],
            cached_input_tokens=totals["cached_input_tokens"],
            cache_creation_tokens=totals["cache_creation_tokens"],
            output_tokens=totals["output_tokens"],
        )
    return _UsageFacts(
        calls=len(calls),
        complete_calls=complete_calls,
        unknown_calls=len(calls) - complete_calls,
        usage=aggregate,
        groups=tuple(usage_groups),
        findings=dict(findings),
    )
