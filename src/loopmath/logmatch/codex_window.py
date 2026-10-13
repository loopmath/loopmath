"""A Codex session cut to a window (0.2.6).

A Codex rollout logs cumulative usage: each `event_msg` `token_count` line
carries `info.total_token_usage`, the file's counter so far, which a
compaction resets to a lower total. The whole-session parser
(`ingest.codex.parse_session`) adds the last reading of each file and of each
segment. Here each reading's increase over the reading before it on the same
counter (one file, one segment; a counter's first reading counts from zero)
counts at the reading's own time, and a reading counts when its time is
inside `[lo, hi]`, both edges included, as a Claude Code request does
(`clip`). With both sides open the result equals the parser's `tokens` and
`tokens_by_model` exactly.

- Each increase goes to the model of the latest `turn_context` before it,
  the thread's first model before any. The split follows the parser: the one
  model of a thread that ran one, `ingest.codex._split_by_model` for one that
  switched, and no model (unpriced) when that split does not add up.
- A resumed thread's files are read in order, each with its own counter.
- A child thread goes with its start and counts in full (`clip`).
- A window that cannot be placed raises `WindowRefused`, and the session
  counts whole with the reason: a usable reading with no time, or a stream
  whose increases inside the window add up to less than zero.
- A thread that began after the window counts nothing: every line of it,
  timed or not, follows its start (`session_parts`).

Only timestamps, models and usage counts are read.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..ingest.base import Tokens, canonical_model, iter_jsonl_stream
from ..ingest.codex import _USAGE_KEYS, _split_by_model
from .match import UNSPLIT, Match, _epoch


class WindowRefused(Exception):
    """The session cannot be cut to the window; the message says why."""


def window_tokens(
    paths: list[str | Path], lo: float | None, hi: float | None
) -> tuple[Tokens, dict[str | None, Tokens]]:
    """`(tokens, tokens_by_model)` for the readings of the thread's `paths` inside `[lo, hi]`.

    `None` leaves that side open; both open is the whole thread, as
    `parse_session` counts it. Raises `WindowRefused` (see the module
    docstring) only for a window with at least one side.
    """
    bounded = lo is not None or hi is not None
    raw: dict[str | None, list[int]] = {}  # model -> [input incl. cache, cached, cache write, output]
    models: Counter = Counter()
    active: str | None = None
    first: str | None = None
    for path in paths:
        last = None  # the latest reading of this file's counter
        prev_total = None
        for line in iter_jsonl_stream(path):
            payload = line.get("payload")
            if not isinstance(payload, dict):
                continue
            if line.get("type") == "turn_context":
                m = canonical_model(payload.get("model"))
                if m:
                    models[m] += 1
                    active = m
                    if first is None:
                        first = m
                continue
            if line.get("type") != "event_msg" or payload.get("type") != "token_count":
                continue
            info = payload.get("info") or {}
            usage = info.get("total_token_usage") or {}
            total = usage.get("total_tokens")
            if not isinstance(total, (int, float)):
                continue
            base = None if prev_total is not None and total < prev_total else last
            last, prev_total = usage, total
            if bounded:
                at = _epoch(line.get("timestamp"))
                if at is None:
                    raise WindowRefused("a token reading has no time")
                if (lo is not None and at < lo) or (hi is not None and at > hi):
                    continue
            part = raw.setdefault(active, [0, 0, 0, 0])
            for i, key in enumerate(_USAGE_KEYS):
                part[i] += int(usage.get(key) or 0) - (int(base.get(key) or 0) if base else 0)

    acc = [sum(v[i] for v in raw.values()) for i in range(4)]
    if bounded and (min(acc) < 0 or acc[0] < acc[1]):
        raise WindowRefused("a usage counter went down inside the window without a reset")
    tokens = Tokens(in_=max(0, acc[0] - acc[1]), cache_read=acc[1], cache_write=acc[2], out=acc[3])
    if len(models) <= 1:
        return tokens, {next(iter(models), None): tokens}
    return tokens, _split_by_model(raw, first, tokens)


def own_files(match: Match) -> list[str]:
    """Every rollout file of the matched thread itself (a resumed thread has several), children not included."""
    files: list[str] = []
    for part in match.parts:
        if part.get("role") == "session":
            for file in part.get("files") or ([part["path"]] if part.get("path") else []):
                if file not in files:
                    files.append(file)
    return files or [str(match.session_path)]


def session_parts(match: Match, lo: float | None, hi: float | None, at: str, end: str) -> list[dict[str, Any]]:
    """The thread's own parts inside `[lo, hi]`, as `match._parts_of` shapes them (children are `clip`'s).

    Unlike a Claude Code window, an empty one keeps a part with no tokens, so the
    thread's files stay on the match for `artifacts_for` even when a child is all
    the window holds.
    """
    files = own_files(match)
    began = _epoch(match.started_at)
    if hi is not None and began is not None and began > hi:
        tokens, by_model = Tokens(), {}  # began after the window: nothing of it is inside, timed or not
    else:
        tokens, by_model = window_tokens(files, lo, hi)
    unsplit = None
    if not by_model and tokens.total:
        by_model = {None: tokens}
        unsplit = UNSPLIT
    parts = []
    for model, part_tokens in by_model.items():
        if not part_tokens.total:
            continue
        part = {
            "session": match.session_id,
            "role": "session",
            "harness": match.harness,
            "model": model,
            "tokens": part_tokens,
            "path": files[0],
            "files": files,
            "parent": None,
            "started_at": at,
            "ended_at": end,
        }
        if unsplit:
            part["unpriced"] = unsplit
            part["unsplit"] = True
        parts.append(part)
    if not parts:
        parts.append(
            {
                "session": match.session_id,
                "role": "session",
                "harness": match.harness,
                "model": None,
                "tokens": Tokens(),
                "path": files[0],
                "files": files,
                "parent": None,
                "started_at": at,
                "ended_at": end,
            }
        )
    return parts
