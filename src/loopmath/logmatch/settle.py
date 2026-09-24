"""Settle a run's attempts against the session logs (for `run finish`, lane 7).

Owner: lane 02. Spec: design/0.1/ (02 `run finish`, 03 section 6; Analyst D27).

`settle_attempt` matches one attempt and fills what the logs know: `session`,
the four-stream `cost` with dollars, `basis`, tariff and, when the tokens ran
on more than one model, their split by model (Analyst D71), and any missing
`started_at`, `ended_at`, `harness`, `model` and `effort`. Values the
orchestrator already wrote are kept, except that the cost streams are
replaced by the measured ones. A declared model the log never ran is kept
too, and the match reason says which model the log ran (the dollars are
always the log's models). An unmatched attempt comes back unchanged
with the reason. `settle_run` does every attempt, the ones that name their
session first, so a heuristic match never takes a session another attempt
named, and returns the counts `run finish --json` reports.

A `--session self` attempt is clipped to its own window (Analyst D47, see
`clip`). One with no `ended_at` is clipped at run finish: `finished_at`, else
the run's `ended_at`, else the clock; that time becomes its `ended_at` and
the clip record says `to_finish`, which a later settle keeps.

A session that more than one attempt names is counted once (Analyst D87,
D88). Clipped attempts keep their window's cost; the rest of the session
(its whole cost minus those windows) is split equally among the attempts
that name it whole, per model and stream, whole tokens with the remainder
to the earlier attempts. Those attempts get basis `allocated`, keep tier
`verified` and the whole session's `parts` (as measured), carry their share
in `dev.loopmath.model_tokens` even for one model, so no reader reprices the
parts, and carry `shared_session = {"session", "attempts"}` in their logmatch ext;
`summary["shared"]` lists each such session. Attempts that are all clipped
share nothing.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import artifacts_for
from .clip import clip_match, is_self
from .costs import EXT_KEY, MODEL_TOKENS_KEY, cost_record, model_tokens_from_parts
from .match import LogRoots, _attempt_model, _epoch, explain_match

# Harnesses with no session log: a mechanical gate runs a command, not a model.
NO_LOG_HARNESSES = ("command",)
_STREAMS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_tokens",
    "cache_creation_5m_tokens",
    "cache_creation_1h_tokens",
    "output_tokens",
    "reasoning_tokens",
    "requests",
    "usd",
    "basis",
    "tariff",
)


def _merge_cost(old: Any, new: dict[str, Any]) -> dict[str, Any]:
    """`new` over `old`, keeping `old`'s fields that are not measured here."""
    merged = {k: v for k, v in (old or {}).items() if k not in _STREAMS and k != "ext"}
    ext = dict((old or {}).get("ext") or {})
    # The split is measured with the streams; its absence means one model (D71).
    ext.pop(MODEL_TOKENS_KEY, None)
    merged.update(new)
    ext.update(new.get("ext") or {})
    if ext:
        merged["ext"] = ext
    return merged


def _earlier_heuristic(attempt: dict[str, Any]) -> dict[str, Any] | None:
    """The ext record of an earlier heuristic settle whose session is still named."""
    cost = attempt.get("cost")
    ext = ((cost or {}).get("ext") or {}).get(EXT_KEY) if isinstance(cost, dict) else None
    if isinstance(ext, dict) and ext.get("tier") == "heuristic" and ext.get("session") == attempt.get("session"):
        return ext
    return None


def _earlier_clip(attempt: dict[str, Any]) -> dict[str, Any] | None:
    """The clip record of an earlier settle, if any."""
    cost = attempt.get("cost")
    ext = ((cost or {}).get("ext") or {}).get(EXT_KEY) if isinstance(cost, dict) else None
    clip = ext.get("clip") if isinstance(ext, dict) else None
    return clip if isinstance(clip, dict) else None


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def settle_attempt(
    attempt: dict[str, Any],
    *,
    roots: LogRoots,
    exclude: Iterable[str] = (),
    prices: str | Path | None = None,
    finished_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(attempt, report)`: the attempt with log facts filled, and what happened.

    `finished_at` is the run finish time a `--session self` attempt with no
    `ended_at` is clipped at (the clock when None).
    """
    report: dict[str, Any] = {"attempt": attempt.get("id"), "tier": None, "session": None}
    if attempt.get("harness") in NO_LOG_HARNESSES:
        report["reason"] = f"harness {attempt['harness']!r} keeps no session log"
        report["skipped"] = True
        return attempt, report
    match, reason = explain_match(attempt, roots=roots, exclude=exclude)
    earlier = _earlier_heuristic(attempt)
    if match is not None and earlier is not None:
        # The session was written here by an earlier heuristic match, not by
        # the orchestrator, so finding it by id again proves nothing new: the
        # tier and the reason stay those of the heuristic match.
        match.tier = "heuristic"
        reason = match.reason = str(earlier.get("reason") or "an earlier heuristic match")
    report["reason"] = reason
    if match is None:
        return attempt, report

    clip_end = None
    clip_skipped = None
    if is_self(attempt):
        if match.harness != "claude-code":
            clip_skipped = "--session self names a Claude Code session (D26); a codex session is not clipped"
        else:
            end = attempt.get("ended_at")
            if end:
                earlier = _earlier_clip(attempt)
                to_finish = bool(earlier and earlier.get("to_finish") and _epoch(earlier.get("to")) == _epoch(end))
            else:
                end = clip_end = finished_at or _now()
                to_finish = True
            match = clip_match(match, attempt.get("started_at"), end)
            if to_finish:
                match.clip["to_finish"] = True
            report["clip"] = dict(match.clip)

    declared = _attempt_model(attempt)
    if declared and match.models and declared not in match.models:
        # `run attempt` needs --model even with --session, so a wrong guess
        # would otherwise sit beside the log's model without a word.
        reason = f"{reason}; its log ran {', '.join(match.models)}, not the declared {declared}"
        match.reason = report["reason"] = reason

    match.artifacts = artifacts_for(match)
    out = copy.deepcopy(attempt)
    out["session"] = match.session_id
    out["cost"] = _merge_cost(attempt.get("cost"), cost_record(match, path=prices))
    if clip_skipped:
        out["cost"]["ext"][EXT_KEY]["clip_skipped"] = clip_skipped
    if clip_end and not out.get("ended_at"):
        out["ended_at"] = clip_end
    if not out.get("harness"):
        out["harness"] = match.harness
    if not out.get("started_at") and match.started_at:
        out["started_at"] = match.started_at
    if not out.get("ended_at") and match.ended_at:
        out["ended_at"] = match.ended_at
    if not out.get("model") and match.model:
        # Claude Code logs the model per request (verified); Codex logs the
        # configured model per turn (reported).
        out["model"] = {
            "raw": match.model,
            "id": match.model,
            "tier": "verified" if match.harness == "claude-code" else "reported",
        }
    if not out.get("effort") and match.effort:
        out["effort"] = match.effort
    if match.artifacts:
        # Where run finish turns them into OCP artifacts and edges is lane 7's
        # call; here they ride on the attempt under this lane's key.
        attempt_ext = out.setdefault("ext", {})
        attempt_ext[EXT_KEY] = {**(attempt_ext.get(EXT_KEY) or {}), "artifacts": match.artifacts}
    report.update(
        tier=match.tier,
        session=match.session_id,
        children=list(match.children),
        usd=out["cost"].get("usd"),
    )
    return out, report


def settle_run(
    doc: dict[str, Any],
    *,
    roots: LogRoots,
    prices: str | Path | None = None,
    finished_at: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """`(doc, summary)` with every attempt settled.

    `summary` is `{verified, heuristic, unmatched: [{attempt, reason}],
    skipped: [{attempt, reason}], shared: [{session, attempts}], usd}`; `usd`
    sums the attempts that have dollars, each session once (D87), and is None
    when none has. `finished_at` (else the run's `ended_at`, else the clock)
    is the run finish time for D47 clipping.
    """
    finish = finished_at or (doc.get("run") or {}).get("ended_at") or _now()
    out = copy.deepcopy(doc)
    attempts = out.get("attempts") or []
    named = {str(a["session"]) for a in attempts if isinstance(a, dict) and a.get("session")}
    claimed: set[str] = set(named)
    summary: dict[str, Any] = {"verified": 0, "heuristic": 0, "unmatched": [], "skipped": []}
    order = sorted(range(len(attempts)), key=lambda i: 0 if attempts[i].get("session") else 1)
    matched: dict[int, dict[str, Any]] = {}
    for i in order:
        settled, report = settle_attempt(attempts[i], roots=roots, exclude=claimed, prices=prices, finished_at=finish)
        attempts[i] = settled
        if report.get("skipped"):
            summary["skipped"].append({"attempt": report["attempt"], "reason": report["reason"]})
        elif report["tier"] is None:
            summary["unmatched"].append({"attempt": report["attempt"], "reason": report["reason"]})
        else:
            summary[report["tier"]] += 1
            claimed.add(report["session"])
            claimed.update(report.get("children") or [])
            matched[i] = report
    summary["shared"] = _share_sessions(attempts, matched)
    usds = [attempts[i]["cost"].get("usd") for i in sorted(matched)]
    known = [u for u in usds if u is not None]
    summary["usd"] = round(sum(known), 6) if known else None
    return out, summary


def _share_sessions(attempts: list[dict[str, Any]], matched: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Count each session once when several attempts name it (D87); see the module docstring."""
    by_session: dict[str, list[int]] = {}
    for i in sorted(matched):
        by_session.setdefault(str(matched[i]["session"]), []).append(i)
    shared = []
    for session, idx in by_session.items():
        whole = [i for i in idx if not matched[i].get("clip")]
        if len(idx) < 2 or not whole:
            continue
        clipped = [attempts[i]["cost"] for i in idx if matched[i].get("clip")]
        record = {"session": session, "attempts": [attempts[i].get("id") for i in idx]}
        base = attempts[whole[0]]["cost"]
        rest = _minus(_by_model(base), [_by_model(c) for c in clipped])
        usd = _rest_usd(base, clipped)
        for k, i in enumerate(whole):
            share = {m: {f: n // len(whole) + (k < n % len(whole)) for f, n in fields.items()} for m, fields in rest.items()}
            cost = attempts[i]["cost"]
            for f in _TOKEN_FIELDS:
                cost[f] = sum(fields[f] for fields in share.values())
            if usd is None:
                cost.pop("usd", None)
            else:
                cost["usd"] = round(usd / len(whole), 6)
            cost["basis"] = "allocated"
            ext = cost["ext"]
            if ext.get(MODEL_TOKENS_KEY) != {}:  # {}: several models, no split (D71), stays so
                ext.pop(MODEL_TOKENS_KEY, None)
                kept = {m: fields for m, fields in share.items() if any(fields.values())}
                if kept:
                    # Always the share, one model included: without it a reader
                    # falls back to `parts`, the whole session's (review of e71da6f).
                    ext[MODEL_TOKENS_KEY] = kept
            ext[EXT_KEY]["shared_session"] = copy.deepcopy(record)
            matched[i]["usd"] = cost.get("usd")
        shared.append(record)
    return shared


_TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens")


def _by_model(cost: dict[str, Any]) -> dict[str, dict[str, int]]:
    """The attempt's four streams per model, from its measured parts."""
    return model_tokens_from_parts(cost["ext"][EXT_KEY].get("parts") or [])


def _minus(whole: dict[str, dict[str, int]], windows: list[dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    rest = {m: dict(fields) for m, fields in whole.items()}
    for window in windows:
        for model, fields in window.items():
            for f, n in fields.items():
                if model in rest:
                    rest[model][f] = max(0, rest[model][f] - n)
    return rest


def _rest_usd(whole: dict[str, Any], clipped: list[dict[str, Any]]) -> float | None:
    """Whole dollars minus the clipped windows' (a window with no requests is 0); None when unknown."""
    usd = whole.get("usd")
    for cost in clipped:
        if usd is None:
            return None
        if cost.get("usd") is not None:
            usd -= cost["usd"]
        elif not cost["ext"][EXT_KEY].get("no_requests"):
            return None
    return max(0.0, usd) if usd is not None else None
