"""Claude Code's `claude -p --output-format stream-json --verbose` stream, read back: tokens per model, the final
text, Claude Code's own dollar figures, and how the session ended. No pricing here (`ledger.py` prices).

One JSON object per line. The lines this reader uses (Claude Code 2.1.301):

    system, subtype init        the session's model
    assistant                   one per content block of a response: `message.id`, `message.model`,
                                `message.usage` (input_tokens, cache_read_input_tokens,
                                cache_creation_input_tokens, output_tokens). A request that failed comes back as
                                an assistant line with `is_api_error_message` true, the SDK's `error` and the
                                typed `api_error`
    rate_limit_event            `rate_limit_info.status` is "rejected" when the account's own limit refused
    system, subtype api_retry   one per retry the CLI made of a failed request
    result                      the end: `subtype` (success, error_max_turns, error_max_budget_usd,
                                error_during_execution), `is_error`, `result` (the final text), `num_turns`,
                                `total_cost_usd`, `usage` and `modelUsage` {model: {inputTokens,
                                cacheReadInputTokens, cacheCreationInputTokens, outputTokens, costUSD}}

Tokens per model come from `modelUsage`, which has one entry per model the session used, so a subagent on another
model gets its own row. A stream with no `modelUsage` falls back to the assistant lines: one count per response
(message id), the largest seen per stream, since a response's early lines carry placeholder output counts, under
the response's own model. `input` excludes cache reads and writes, as the price table reads it. A placeholder
model (`<synthetic>`, the failed request's own line) with no tokens is no model and gets no row. A session that
ended on an API error and reported no tokens anywhere still made a call: it gets one row on its model with zero
tokens, which the ledger prices at 0, never as a lost call.

The error kind (`contracts.ERROR_KINDS`) is read from error channels only: the result's subtype and `is_error`,
its `api_error_status` and `api_error`, the failed request's SDK error, a rejected rate-limit event and the retry
lines. A result with `is_error` false is the session's outcome whatever its text says. A turn that ended on an API
error has subtype success, `is_error` true and the error's own text as `result`; only then is that text read, and
only for the CLI's usage-limit openings.

A line that does not parse is counted, never fatal; a torn last line is the usual case of a killed session.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .contracts import Usage

__all__ = ["StreamResult", "read_stream", "parse_lines", "classify", "render"]

# The account's own limit: every later session on it would be refused too.
USAGE_LIMIT_TYPED = ("usage_limit_reached",)
USAGE_LIMIT_OPENINGS = ("You've hit your", "You've reached your", "You're out of usage credits",
                        "You're out of extra usage")
# The account's credentials or standing.
AUTH_STATUS = (401, 403)
AUTH_SDK = ("authentication_failed", "billing_error", "oauth_org_not_allowed", "account_on_hold",
            "verification_required")
AUTH_TYPED = ("authentication_error", "permission_error", "api_key_auth_disabled", "provider_credentials")
# The trajectory's own limits: an output cap or a context that no longer fits.
AGENT_LIMIT_TYPED = ("max_output_tokens", "autocompact_thrashing")
CONTEXT_OPENINGS = ("Prompt is too long", "prompt is too long")
# The service or the network.
TRANSIENT_STATUS = (408, 429)
TRANSIENT_TYPED = ("no_response", "overloaded_error", "rate_limit_error", "api_error", "timeout_error")
TRANSIENT_SDK = ("rate_limit", "server_error")
# Result subtypes that are the session's own limits.
LIMIT_SUBTYPES = {"error_max_turns": "the session reached its --max-turns limit",
                  "error_max_budget_usd": "the session reached its --max-budget-usd limit"}

_STREAMS = (("input", "input_tokens", "inputTokens"), ("cache_read", "cache_read_input_tokens", "cacheReadInputTokens"),
            ("cache_write", "cache_creation_input_tokens", "cacheCreationInputTokens"),
            ("output", "output_tokens", "outputTokens"))


@dataclass
class StreamResult:
    """One session's stream, read. `usage` one Usage per model; `has_result` False when the stream has no result
    line (killed, crashed or cut); `error_kind` and `error` from the error channels, empty for a clean end."""

    model: str = ""
    usage: list[Usage] = field(default_factory=list)
    text: str = ""
    is_error: bool = False
    subtype: str = ""
    has_result: bool = False
    num_turns: int | None = None
    cli_cost_usd: float | None = None
    error_kind: str = ""
    error: str = ""
    api_status: int | None = None
    retries: int = 0
    turns: int = 0
    lines: int = 0
    bad_lines: int = 0
    usage_source: str = ""
    api_error: bool = False


def _whole(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return 0
    return int(value)


def _money(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def parse_lines(lines: Iterable[str]) -> tuple[list[dict], int, int]:
    """(events, lines read, lines that did not parse as a JSON object)."""
    events: list[dict] = []
    n = bad = 0
    for raw in lines:
        if not raw.strip():
            continue
        n += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            events.append(obj)
        else:
            bad += 1
    return events, n, bad


def _from_model_usage(mu: Any) -> list[Usage]:
    out = []
    if not isinstance(mu, dict):
        return out
    for model, d in mu.items():
        if not isinstance(d, dict):
            continue
        u = Usage(model=str(model), cli_cost_usd=_money(d.get("costUSD")))
        for name, _, key in _STREAMS:
            setattr(u, name, _whole(d.get(key)))
        if str(model).startswith("<") and not any(getattr(u, name) for name, _, _ in _STREAMS):
            continue  # a placeholder for a request that reached no model
        out.append(u)
    return out


def _from_messages(events: list[dict]) -> list[Usage]:
    """Per model, per response id, the largest count seen per stream."""
    per: dict[tuple[str, str], dict[str, int]] = {}
    order: list[str] = []
    for i, e in enumerate(events):
        if e.get("type") != "assistant" or e.get("is_api_error_message"):
            continue
        msg = e.get("message") if isinstance(e.get("message"), dict) else {}
        use = msg.get("usage")
        if not isinstance(use, dict):
            continue
        model = str(msg.get("model") or "")
        if not model or model.startswith("<"):
            continue
        key = (model, str(msg.get("id") or f"line-{i}"))
        got = per.setdefault(key, {name: 0 for name, _, _ in _STREAMS})
        for name, snake, _ in _STREAMS:
            got[name] = max(got[name], _whole(use.get(snake)))
        if model not in order:
            order.append(model)
    out = []
    for model in order:
        u = Usage(model=model)
        for (m, _), counts in per.items():
            if m == model:
                for name, _, _ in _STREAMS:
                    setattr(u, name, getattr(u, name) + counts[name])
        out.append(u)
    return out


def _starts(text: str, openings: tuple[str, ...]) -> bool:
    return any(text.startswith(o) for o in openings)


def classify(result: dict | None, error_msg: dict | None, *, rejected: bool, retries: int,
             turns: int) -> tuple[str, str]:
    """(error kind, one plain line) for a session's end, from its error channels; ("", "") for a clean end or no
    result line (the executor reads the exit code then)."""
    if not isinstance(result, dict):
        return "", ""
    subtype = str(result.get("subtype") or "")
    if subtype in LIMIT_SUBTYPES:
        return "agent_limit", LIMIT_SUBTYPES[subtype]
    if subtype and subtype != "success":
        return "other", f"the session ended with {subtype}"
    if not result.get("is_error"):
        return "", ""
    status = result.get("api_error_status")
    status = status if isinstance(status, int) and not isinstance(status, bool) else None
    em = error_msg or {}
    typed = result.get("api_error") or em.get("api_error")
    typed = typed if isinstance(typed, str) else (typed.get("type") if isinstance(typed, dict) else None)
    sdk = em.get("error") if isinstance(em.get("error"), str) else None
    text = str(result.get("result") or "").strip()
    head = f"API error (status {status if status is not None else 'none'}" + (f", {typed}" if typed else "") + \
        (f", {sdk}" if sdk else "") + (f", {retries} CLI retries" if retries else "") + ")"
    why = f"{head}: {text[:200]}" if text else head
    if typed in USAGE_LIMIT_TYPED or _starts(text, USAGE_LIMIT_OPENINGS) or (status == 429 and rejected):
        return "usage_limit", why
    if status in AUTH_STATUS or typed in AUTH_TYPED or sdk in AUTH_SDK:
        return "auth", why
    if typed in AGENT_LIMIT_TYPED or sdk in AGENT_LIMIT_TYPED or _starts(text, CONTEXT_OPENINGS):
        # A context that does not fit before the first turn is no trajectory's doing: a setting error.
        return ("agent_limit" if turns else "other"), why
    if (status in TRANSIENT_STATUS or (status is not None and status >= 500) or typed in TRANSIENT_TYPED
            or sdk in TRANSIENT_SDK or (status is None and typed is None)):
        return "transient", why
    return "other", why


def _read(events: list[dict], n: int, bad: int) -> StreamResult:
    out = StreamResult(lines=n, bad_lines=bad)
    result = error_msg = None
    rejected = False
    ids: set[str] = set()
    for i, e in enumerate(events):
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init" and not out.model:
            out.model = str(e.get("model") or "")
        elif t == "system" and e.get("subtype") == "api_retry":
            out.retries += 1
        elif t == "rate_limit_event":
            info = e.get("rate_limit_info")
            if isinstance(info, dict) and info.get("status") == "rejected":
                rejected = True
        elif t == "assistant" and e.get("is_api_error_message"):
            error_msg = e
        elif t == "assistant":
            msg = e.get("message") if isinstance(e.get("message"), dict) else {}
            ids.add(str(msg.get("id") or f"line-{i}"))
        elif t == "result":
            result = e
    out.turns = len(ids)
    if isinstance(result, dict):
        out.has_result = True
        out.subtype = str(result.get("subtype") or "")
        out.is_error = bool(result.get("is_error"))
        text = result.get("result")
        out.text = text if isinstance(text, str) else ""
        nt = result.get("num_turns")
        out.num_turns = nt if isinstance(nt, int) and not isinstance(nt, bool) else None
        st = result.get("api_error_status")
        out.api_status = st if isinstance(st, int) and not isinstance(st, bool) else None
        out.usage = _from_model_usage(result.get("modelUsage"))
        if out.usage:
            out.usage_source = "modelUsage"
        total = _money(result.get("total_cost_usd"))
        figures = [u.cli_cost_usd for u in out.usage if u.cli_cost_usd is not None]
        out.cli_cost_usd = round(sum(figures), 9) if figures else total
        if out.usage and total is not None and not figures and len(out.usage) == 1:
            out.usage[0].cli_cost_usd = total
    if not out.usage:
        out.usage = _from_messages(events)
        if out.usage:
            out.usage_source = "assistant"
        if isinstance(result, dict) and not out.usage and isinstance(result.get("usage"), dict) and out.model:
            u = Usage(model=out.model, cli_cost_usd=out.cli_cost_usd)
            for name, snake, _ in _STREAMS:
                setattr(u, name, _whole(result["usage"].get(snake)))
            out.usage = [u]
            out.usage_source = "result"
    out.error_kind, out.error = classify(result, error_msg, rejected=rejected, retries=out.retries, turns=out.turns)
    out.api_error = bool(isinstance(result, dict) and result.get("is_error") and out.subtype in ("", "success")
                         and (error_msg is not None or out.api_status is not None or result.get("api_error")))
    if out.api_error and not out.usage and out.model:
        out.usage = [Usage(model=out.model, cli_cost_usd=out.cli_cost_usd)]
        out.usage_source = "api_error"
    return out


def read_stream(path: str | Path) -> StreamResult:
    """Read one session's stream file. A missing or unreadable file is an empty result (no result line)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            events, n, bad = parse_lines(fh)
    except OSError:
        return StreamResult()
    return _read(events, n, bad)


def render(event: dict, width: int = 160) -> str | None:
    """One stream event as a readable line for a terminal pane (assistant text, tool calls, the result), or None
    for events a person need not see."""
    t = event.get("type")

    def cut(s: str) -> str:
        s = " ".join(str(s).split())
        return s if len(s) <= width else s[: width - 3] + "..."

    if t == "system" and event.get("subtype") == "init":
        return cut(f"session start: {event.get('model') or '?'} in {event.get('cwd') or '?'}")
    if t == "system" and event.get("subtype") == "api_retry":
        return "api retry"
    if t == "assistant":
        msg = event.get("message") if isinstance(event.get("message"), dict) else {}
        parts = []
        for block in msg.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
            elif block.get("type") == "tool_use":
                arg = block.get("input") if isinstance(block.get("input"), dict) else {}
                hint = next((str(arg[k]) for k in ("command", "file_path", "pattern", "path", "description")
                             if k in arg), "")
                parts.append(f"[{block.get('name') or 'tool'}] {hint}".rstrip())
        return cut(" ".join(parts)) if parts else None
    if t == "result":
        cost = event.get("total_cost_usd")
        bits = [f"result: {event.get('subtype') or '?'}", "error" if event.get("is_error") else "ok"]
        if isinstance(event.get("num_turns"), int):
            bits.append(f"{event['num_turns']} turns")
        if isinstance(cost, (int, float)):
            bits.append(f"{cost:.4f} USD")
        return cut(", ".join(bits))
    return None
