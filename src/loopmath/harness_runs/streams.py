"""Token usage read back from a rollout's own event stream (no pricing here).

Two streams, read as harness2's own parsers read them (OpenCode: `session.py:44-49`, `oc_agent.py:353-370`;
Codex: `codex.py:432-444`):

- `opencode`: `opencode run --format json` lines. Each `step_finish` event's part carries
  `tokens {input, output, reasoning, cache {read, write}}` and `cost`; a `step_start` opens each step.
- `codex`: `codex exec --json` lines. Each `turn.completed` event carries
  `usage {input_tokens (cached included), cached_input_tokens, output_tokens}`; `turn.started` opens a
  turn and `turn.failed` ends one with no usage.

The four streams follow OCP and `loopmath.price`. `input` excludes cached reads. `output` includes
reasoning, as `adapters/opencode_store._message_tokens` maps OpenCode's split. Lines that do not parse are
counted, never fatal.

Coverage (spec/harness-runs.md section 4): a step is billable when its tokens or its cost are above 0.
OpenCode's own dollars are the rollout's only when every billable step reports a number and the total is
above 0; a reported 0 is a number, a missing `cost` is not. A step or turn that started and has no usage
after it makes the counts incomplete, and so does a `step_finish` or `turn.completed` whose counts are missing
or not whole numbers: a missing count is never read as 0. A named file that cannot be read makes them
incomplete too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

KINDS = ("opencode", "codex")


def _whole(value: Any) -> int | None:
    """A count as the stream wrote it: a whole number at or above 0, else None (not measured)."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _opencode_step(tok: Any) -> dict[str, int] | None:
    """One `step_finish`'s counts, or None when any is missing or not a whole number. `input`, `output` and
    `cache {read, write}` are required; an absent `reasoning` adds 0, as `_message_tokens` reads it."""
    if not isinstance(tok, dict) or not isinstance(tok.get("cache"), dict):
        return None
    got = (_whole(tok.get("input")), _whole(tok.get("output")), _whole(tok["cache"].get("read")),
           _whole(tok["cache"].get("write")), 0 if tok.get("reasoning") is None else _whole(tok.get("reasoning")))
    if any(v is None for v in got):
        return None
    return {"input": got[0], "output": got[1] + got[4], "cache_read": got[2], "cache_write": got[3]}


def _codex_turn(use: Any) -> dict[str, int] | None:
    """One `turn.completed` usage, or None when `input_tokens`, `cached_input_tokens` or `output_tokens` is
    missing or not a whole number, or the cached part exceeds the input."""
    if not isinstance(use, dict):
        return None
    total, cached, out = (_whole(use.get(k)) for k in ("input_tokens", "cached_input_tokens", "output_tokens"))
    if total is None or cached is None or out is None or cached > total:
        return None
    return {"input": total - cached, "cache_read": cached, "output": out, "cache_write": 0}


def _dollars(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def rollout_usage(paths: str | Path | Iterable[str | Path], kind: str) -> dict[str, Any]:
    """Usage summed over every file of one rollout (JB retries are separate files, and each one is spend).

    Returns `{tokens, usd, usd_source, recovered, complete, incomplete, partial_usd, files, skipped_lines}`
    and, for OpenCode, `uncosted_steps` (billable steps with no `cost`). `tokens` is None when no file held
    a usage event. `usd` is OpenCode's own total when every billable step reported a number and the total
    is above 0, else None (left for pricing); `partial_usd` is then any reported subtotal. Codex streams
    carry no dollars.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown stream kind {kind!r} (known: {', '.join(KINDS)})")
    tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
    reported, costed, uncosted = 0.0, 0, 0
    events, files, skipped, unfinished, failed, unmeasured = 0, 0, 0, 0, 0, 0
    unread: list[str] = []
    for path in _paths(paths):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:     # a named file that cannot be read is spend the counts miss
            unread.append(f"{path.name} ({exc.strerror or type(exc).__name__})")
            continue
        files += 1
        open_unit = False
        for line in text.split("\n"):     # JSON lines: newline only, never U+2028 or U+0085 inside a string
            line = line.strip(" \t\r")
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(ev, dict):
                skipped += 1
                continue
            if kind == "opencode":
                part = ev.get("part") if isinstance(ev.get("part"), dict) else {}
                etype = part.get("type") or ev.get("type")
                if etype in ("step-start", "step_start"):
                    unfinished += open_unit
                    open_unit = True
                    continue
                if etype not in ("step-finish", "step_finish"):
                    continue
                open_unit = False
                step = _opencode_step(part.get("tokens"))
                cost = _dollars(part.get("cost"))
                if step is None:
                    unmeasured += 1
                    if cost:
                        reported += cost
                    continue
                events += 1
                for key, value in step.items():
                    tokens[key] += value
                if not (sum(step.values()) > 0 or (cost or 0) > 0):
                    continue
                if cost is None:
                    uncosted += 1
                else:
                    reported += cost
                    costed += 1
            else:
                etype = ev.get("type")
                if etype == "turn.started":
                    unfinished += open_unit
                    open_unit = True
                elif etype == "turn.failed":
                    failed += 1
                    open_unit = False
                elif etype == "turn.completed":
                    open_unit = False
                    turn = _codex_turn(ev.get("usage"))
                    if turn is None:
                        unmeasured += 1
                        continue
                    events += 1
                    for key, value in turn.items():
                        tokens[key] += value
        unfinished += open_unit
    missing = []
    if unfinished:
        unit = "step" if kind == "opencode" else "turn"
        ends = "step_finish" if kind == "opencode" else "turn.completed usage"
        missing.append(f"{_plural(unfinished, unit)} started with no {ends} after")
    if failed:
        missing.append(f"{_plural(failed, 'turn')} failed (turn.failed, no usage)")
    if unmeasured:
        unit, ends = ("step", "step_finish") if kind == "opencode" else ("turn", "turn.completed")
        missing.append(f"{_plural(unmeasured, unit)} ended ({ends}) with token counts missing or not whole numbers")
    if unread:
        missing.append(f"{_plural(len(unread), 'stream file')} could not be read: {', '.join(unread)}")
    incomplete = "; ".join(missing) or None
    own = kind == "opencode" and uncosted == 0 and costed > 0 and reported > 0
    out: dict[str, Any] = {
        "tokens": tokens if events else None, "usd": round(reported, 6) if own and not incomplete else None,
        "usd_source": "opencode" if own and not incomplete else None, "recovered": bool(events),
        "complete": incomplete is None, "incomplete": incomplete,
        "partial_usd": round(reported, 6) if reported > 0 and (incomplete or not own) else None,
        "files": files, "skipped_lines": skipped}
    if kind == "opencode":
        out["uncosted_steps"] = uncosted
    return out
