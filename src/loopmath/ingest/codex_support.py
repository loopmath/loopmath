"""Support routines for the Codex rollout-log parser."""

from __future__ import annotations

import json
import re
from pathlib import Path


_CHECK_CMD_RE = re.compile(
    r"pytest|\bnpm (run )?test\b|cargo test|go test|vitest|jest|make test|ruff|mypy|tsc",
    re.IGNORECASE,
)
_EXIT_CODE_RE = re.compile(
    r'"exit_code"\s*:\s*(-?\d+)'
    r"|\bexit(?:ed)?(?:\s+with)?\s+code[:\s]+(-?\d+)\b",
    re.IGNORECASE,
)
_FAIL_TEXT_RE = re.compile(
    r"\b[1-9]\d* (?:failed|failures?|errors?)\b"
    r"|\bFAILED\b"
    r"|\bnot ok\b"
    r"|\bTraceback \(most recent call\)"
)
_ZERO_PASSED_RE = re.compile(r"\b0 passed\b")
_PASS_TEXT_RE = re.compile(r"\b[1-9]\d* passed\b" r"|\ball tests passed\b")
_OK_RE = re.compile(r"(?:^|\\n)[ \t]*ok\b", re.MULTILINE)
_AGENT_CLAIM_RE = re.compile(
    r"\b(done|complete|completed|all tests pass|working|fixed)\b", re.IGNORECASE
)
_PATCH_UPDATED_MARKER = "Updated the following files:"
_PATCH_FILE_LINE_RE = re.compile(r"^[AMD] (.+)$", re.MULTILINE)


def _session_id(path: "str | Path") -> "str | None":
    """Return the first persisted Codex thread id in *path*, if present.

    The thread's own `id` comes first. A sub-agent thread carries its own `id`
    and its root parent's id in `session_id` (measured 2026-09-23: 174 of
    4,314 rollout files, every one a `source.subagent` thread), so reading
    `session_id` first grouped each child into its parent as if it were a
    resumed continuation, priced at the parent's model (Analyst D28). No two
    files shared an `id` in that estate; `session_id` remains the fallback
    for a file without an `id`.
    """
    line = _first_session_meta(path)
    if line is None:
        return None
    payload = line.get("payload")
    if not isinstance(payload, dict):
        return None
    value = payload.get("id") or payload.get("session_id")
    return str(value) if value else None


def _first_session_meta(path: "str | Path") -> "dict | None":
    """The first `session_meta` line of *path*, reading no further than it.

    Grouping reads this for every Codex file on every `parse_all`, and it is
    line 0 in practice, so decoding the whole rollout (as `iter_jsonl` does)
    dominated a warm run (lane 09 profile: 15.1 s of 17.9 s over 1,988 files).
    Blank, unparseable and non-object lines are skipped as `iter_jsonl` skips
    them; a line that cannot hold the type is not decoded at all.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if "session_meta" not in raw:
                    continue
                try:
                    obj = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(obj, dict) and obj.get("type") == "session_meta":
                    return obj
    except OSError:
        return None
    return None


def parent_session_id(meta_payload: dict) -> "str | None":
    """The thread a Codex child was spawned from, or None for a root thread.

    `parent_thread_id` names the direct parent (it differs from `session_id`
    for a grandchild, where `session_id` is the root). Without it, a
    `session_id` that differs from the file's own `id` still marks a child.
    """
    own = meta_payload.get("id")
    parent = meta_payload.get("parent_thread_id")
    if parent and str(parent) != str(own):
        return str(parent)
    root = meta_payload.get("session_id")
    if own and root and str(root) != str(own):
        return str(root)
    return None


def group_session_paths(paths: "list[str | Path]") -> "list[list[Path]]":
    """Group rollout files belonging to the same resumed Codex thread.

    Files group by the thread's own id (`_session_id`), so a sub-agent thread
    is its own group, never part of its parent's.

    Input order is retained both between groups and inside each group. The
    ingest discovery layer supplies chronological filename order, so a
    continuation is parsed after the file it resumes. Files without usable
    session metadata remain independent; they must never be merged merely
    because they share a workspace.
    """
    groups: list[list[Path]] = []
    by_session_id: dict[str, list[Path]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        session_id = _session_id(path)
        if session_id is None:
            groups.append([path])
            continue
        group = by_session_id.get(session_id)
        if group is None:
            group = []
            by_session_id[session_id] = group
            groups.append(group)
        group.append(path)
    return groups


def _extract_exit_code(text: str) -> int | None:
    """Structured exit status read directly off the raw output text, or None.

    See the module-level comment above `_EXIT_CODE_RE` for the two real log
    shapes this covers. No JSON parsing is attempted: a substring search over
    the raw text is sufficient for both shapes and is robust to the output
    not being valid JSON at all (e.g. an `apply_patch` failure message).
    """
    m = _EXIT_CODE_RE.search(text)
    if not m:
        return None
    raw = m.group(1) if m.group(1) is not None else m.group(2)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _classify_check_output_text(text: str) -> bool | None:
    """Path 2 (text heuristic) of the check-outcome read. See module docstring."""
    if _FAIL_TEXT_RE.search(text):
        return False
    if _ZERO_PASSED_RE.search(text) and not (_PASS_TEXT_RE.search(text) or _OK_RE.search(text)):
        return None  # "0 passed": nothing ran, no signal either way
    if _PASS_TEXT_RE.search(text) or _OK_RE.search(text):
        return True
    return None


def _classify_check_output(text: str) -> bool | None:
    """Pass/fail verdict for one check command's output text.

    Path 1: a structured exit status (`_extract_exit_code`), when the output
    carries one. This is authoritative: exit 0 is a pass, non-zero is a
    fail, and no text marker is consulted for that command once found.

    Path 2: only when path 1 finds nothing, `_classify_check_output_text`'s
    count-aware text heuristic, which can be wrong on unusual output.

    Either way, returns True/False when a verdict is reached, None when the
    output is inconclusive; the text itself is never retained past this call.
    """
    code = _extract_exit_code(text)
    if code is not None:
        return code == 0
    return _classify_check_output_text(text)


def _tool_call_text(payload: dict) -> str | None:
    text = payload.get("input")
    if text is None:
        text = payload.get("arguments")
    return text if isinstance(text, str) else None


def _tool_output_text(payload: dict) -> str:
    out = payload.get("output")
    if isinstance(out, list):
        return " ".join(b.get("text", "") for b in out if isinstance(b, dict))
    if isinstance(out, str):
        return out
    return ""


def _extract_apply_patch_paths(text: str) -> "list[str]":
    """File paths from a successful `apply_patch` call's own output summary.

    Metadata only, per the module's privacy contract: this reads the tool's
    status-list lines (a change-kind letter plus a path, one file per line),
    never the `input` patch envelope (the diff body) and never any text after
    a failed-patch message. Returns `[]` when the marker is absent (including
    every failed-patch shape sampled).
    """
    idx = text.find(_PATCH_UPDATED_MARKER)
    if idx == -1:
        return []
    # Both real serializations use the tool's OWN newline convention inside
    # this block: the JSON-string shape's embedded newlines are the literal
    # two-character sequence `\n`, never an actual newline byte, in this
    # local estate; normalizing that sequence to a real newline first lets
    # one line-oriented regex cover both shapes.
    tail = text[idx:].replace("\\n", "\n")
    return [m.group(1).strip() for m in _PATCH_FILE_LINE_RE.finditer(tail)]
