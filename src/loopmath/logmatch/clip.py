"""Clip a `--session self` match to the attempt's own window.

`--session self` names the Claude Code session that runs the command, and
that session usually did other work before and after the attempt. Such an
attempt (lane 7 marks it `ext["dev.loopmath.match"].session_from == "self"`)
is clipped to `[started_at, ended_at]`, inclusive, by transcript entry time:

- a request counts when its first entry falls in the window, with the ingest
  parser's own accounting (per-stream maximum over a `requestId`'s lines,
  lines without one counted directly, the model of its first line that names
  a real one);
- a sub-agent that starts inside the window counts in full, even past the
  end: a sub-agent file by its earliest entry, a sidechain agent embedded in
  the session file by its earliest entry of any type (its prompt, not its
  first response); one that started before the window does not count;
- with no `ended_at` the caller clips at run finish (see `settle`).

Sessions the orchestrator named keep their whole cost. The basis stays
`measured`; `cost.ext["dev.loopmath.logmatch"].clip` records the window.
Only timestamps, request ids, agent ids, models and usage counts are read.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

from ..ingest.base import Tokens, canonical_model, iter_jsonl
from .match import Match, _epoch, _utc

MATCH_EXT = "dev.loopmath.match"


def is_self(attempt: Mapping[str, Any]) -> bool:
    """True when lane 7 recorded the attempt's session as `--session self`."""
    ext = attempt.get("ext")
    rec = ext.get(MATCH_EXT) if isinstance(ext, Mapping) else None
    return isinstance(rec, Mapping) and rec.get("session_from") == "self"


def _inside(epoch: float | None, lo: float | None, hi: float | None) -> bool:
    if lo is None and hi is None:
        return True  # an open window keeps everything, as the parser does
    if epoch is None:
        return False
    return (lo is None or epoch >= lo) and (hi is None or epoch <= hi)


def window_tokens(path: str | Path, lo: float | None, hi: float | None) -> dict[str | None, Tokens]:
    """Four streams per canonical model for the requests of `path` inside `[lo, hi]`.

    `None` leaves that side open, so `window_tokens(p, None, None)` equals the
    parser's `tokens_by_model` for a Claude Code session file. A request is
    placed by its own first entry; an embedded sidechain agent by its earliest
    entry of any type (its prompt usually precedes its first response), and
    then all of its requests go with it.
    """
    agent_start: dict[str, float] = {}
    request_first: dict[str, float | None] = {}
    request_agent: dict[str, str | None] = {}
    request_model: dict[str, str] = {}
    request_max: dict[str, list[int]] = {}
    direct: list[tuple[float | None, str | None, str | None, tuple[int, int, int, int]]] = []
    for obj in iter_jsonl(path):
        epoch = _epoch(obj.get("timestamp"))
        agent = obj.get("agentId") if obj.get("isSidechain") is True else None
        if not isinstance(agent, str) or not agent:
            agent = None
        if agent is not None and epoch is not None and (agent not in agent_start or epoch < agent_start[agent]):
            agent_start[agent] = epoch
        if obj.get("type") != "assistant":
            continue
        msg = obj.get("message") or {}
        raw = msg.get("model")
        line_model = None
        if isinstance(raw, str) and raw.strip() and not raw.strip().startswith("<"):
            line_model = raw.strip()
        usage = msg.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        vals = (
            int(usage.get("input_tokens") or 0),
            int(usage.get("cache_read_input_tokens") or 0),
            int(usage.get("cache_creation_input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )
        request_id = obj.get("requestId")
        if request_id:
            if request_id not in request_first:
                request_first[request_id] = epoch
                request_agent[request_id] = agent
            if line_model is not None and request_id not in request_model:
                request_model[request_id] = line_model
            prev = request_max.setdefault(request_id, [0, 0, 0, 0])
            for i, v in enumerate(vals):
                if v > prev[i]:
                    prev[i] = v
        else:
            direct.append((epoch, agent, line_model, vals))

    def counts(epoch: float | None, agent: str | None) -> bool:
        return _inside(agent_start.get(agent) if agent is not None else epoch, lo, hi)

    by_model: dict[str | None, list[int]] = {}
    for epoch, agent, line_model, vals in direct:
        if counts(epoch, agent):
            part = by_model.setdefault(canonical_model(line_model), [0, 0, 0, 0])
            for i, v in enumerate(vals):
                part[i] += v
    for request_id, streams in request_max.items():
        if counts(request_first[request_id], request_agent[request_id]):
            part = by_model.setdefault(canonical_model(request_model.get(request_id)), [0, 0, 0, 0])
            for i, v in enumerate(streams):
                part[i] += v
    return {
        model: Tokens(in_=v[0], cache_read=v[1], cache_write=v[2], out=v[3])
        for model, v in by_model.items()
        if any(v)
    }


def clip_match(match: Match, start: str | None, end: str) -> Match:
    """`match` cut to `[start, end]` (see the module docstring); `start=None` is open.

    Claude Code only: `--session self` is an error under Codex.
    """
    if match.harness != "claude-code":
        raise ValueError(f"only a claude-code session is clipped, not {match.harness}")
    lo, hi = _epoch(start), _epoch(end)
    at = start or match.started_at
    parts: list[dict[str, Any]] = []
    for model, tokens in window_tokens(match.session_path, lo, hi).items():
        parts.append(
            {
                "session": match.session_id,
                "role": "session",
                "harness": match.harness,
                "model": model,
                "tokens": tokens,
                "path": str(match.session_path),
                "parent": None,
                "started_at": at,
                "ended_at": end,
            }
        )
    children: list[str] = []
    for part in match.parts:
        if part.get("role") == "child" and _inside(_epoch(part.get("started_at")), lo, hi):
            parts.append(part)
            if part["session"] not in children:
                children.append(part["session"])
    if not parts:
        # Nothing in the window: still this session, with no requests.
        parts.append(
            {
                "session": match.session_id,
                "role": "session",
                "harness": match.harness,
                "model": None,
                "tokens": Tokens(),
                "path": str(match.session_path),
                "parent": None,
                "started_at": at,
                "ended_at": end,
            }
        )
    total = Tokens()
    models: list[str] = []
    for part in parts:
        t = part["tokens"]
        total.in_ += t.in_
        total.cache_read += t.cache_read
        total.cache_write += t.cache_write
        total.out += t.out
        if part["model"] and part["model"] not in models:
            models.append(part["model"])
    own = [p for p in parts if p["role"] == "session" and p["model"]]
    model = max(own, key=lambda p: p["tokens"].total)["model"] if own else match.model
    return dataclasses.replace(
        match,
        tokens=total,
        model=model,
        started_at=_utc(lo) if lo is not None else match.started_at,
        ended_at=_utc(hi) if hi is not None else match.ended_at,
        children=children,
        parts=parts,
        models=models,
        artifacts=[],
        clip={"from": start or match.started_at, "to": end},
    )
