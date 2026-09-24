"""One pass over a session transcript collecting the structural facts the graph is
built from. Frozen after task X0: the hook points other tasks fill are in
`bashwrites.py` (A1), `codexio.py` (A2), `launch.py` (L1, L2) and `cache.py` (L2).

`scan_claude_session` collects the assistant `tool_use` blocks that matter: `Task`
calls (subagent spawns, keyed by tool_use id), `Bash` commands (with the harnesses
their text launches, via `launch.detect_launch`, and the files they write, via
`bashwrites.writes_from_command`), and file writes/reads through the Write, Edit,
MultiEdit, NotebookEdit and Read tools with their timestamps. `bash` entries carry
the tool_use timestamp and, when the matching tool_result was found, the timestamp
the command returned; a CLI launched inside that interval (directly or through a
script) is attributed to it. Each entry also carries the transcript line's `cwd`
when present, for relative-path resolution (A1) and script lookup (L1).
"""

from __future__ import annotations

import difflib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import bashwrites
from .cache import link_cache_get, link_cache_put
from .launch import detect_launch

_SPAWN_TOOLS = {"Task", "Agent"}
_WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}
_READ_TOOLS = {"Read"}


def epoch(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def cache_key(path: Path) -> tuple | None:
    """`(path, mtime_ns, size)` of a session file, or None when it cannot be stat'ed."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)


def _tool_path(inp: dict) -> str | None:
    for key in ("file_path", "notebook_path", "path"):
        val = inp.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _claude_artifact_fact(name: str, inp: dict, ts: str | None, path: str) -> dict:
    """Build the private fact retained until this tool call succeeds.

    The public ``writes`` event remains the path-only compatibility contract.  This
    side-channel record carries only measurements present in the transcript: Write
    content supplies its UTF-8 size, while Edit and MultiEdit replacement strings
    permit a line-diff heuristic. ``splitlines`` makes an empty replacement zero
    lines and does not invent an extra line for a terminal newline.
    """
    fact: dict = {
        "ts": ts,
        "path": path,
        "operation": "write" if name == "Write" else "edit",
        "operation_paths": [path],
        "tier": "verified",
        "observed": True,
    }
    if name == "Write" and isinstance(inp.get("content"), str):
        fact["bytes"] = len(inp["content"].encode("utf-8"))
        fact["bytes_tier"] = "verified"
    elif name == "Edit":
        old, new = inp.get("old_string"), inp.get("new_string")
        if isinstance(old, str) and isinstance(new, str) and inp.get("replace_all") is not True:
            fact["lines_added"], fact["lines_removed"] = _replacement_line_delta(old, new)
            fact["lines_added_tier"] = "heuristic"
            fact["lines_removed_tier"] = "heuristic"
    elif name == "MultiEdit":
        edits = inp.get("edits")
        if isinstance(edits, list) and all(
            isinstance(edit, dict)
            and isinstance(edit.get("old_string"), str)
            and isinstance(edit.get("new_string"), str)
            and edit.get("replace_all") is not True
            for edit in edits
        ):
            deltas = [
                _replacement_line_delta(edit["old_string"], edit["new_string"])
                for edit in edits
            ]
            fact["lines_added"] = sum(added for added, _ in deltas)
            fact["lines_added_tier"] = "heuristic"
            fact["lines_removed"] = sum(removed for _, removed in deltas)
            fact["lines_removed_tier"] = "heuristic"
    return fact


def _replacement_line_delta(old: str, new: str) -> tuple[int, int]:
    """Return heuristic added/removed counts for recorded replacement text."""
    added = removed = 0
    for tag, old_start, old_end, new_start, new_end in difflib.SequenceMatcher(
        a=old.splitlines(), b=new.splitlines(), autojunk=False
    ).get_opcodes():
        if tag in {"replace", "insert"}:
            added += new_end - new_start
        if tag in {"replace", "delete"}:
            removed += old_end - old_start
    return added, removed


def scan_claude_session(path: Path) -> dict:
    """Scan one Claude Code transcript, through the link cache (L2)."""
    key = cache_key(path)
    if key is not None:
        hit = link_cache_get(("claude", *key))
        if hit is not None:
            return hit
    out = _scan_claude_session(path)
    if key is not None:
        link_cache_put(("claude", *key), out)
    return out


def _scan_claude_session(path: Path) -> dict:
    out: dict = {"tasks": {}, "bash": [], "writes": [], "reads": [], "excluded": {}}
    pending: dict[str, dict] = {}
    pending_artifact: dict[str, dict] = {}
    excluded: Counter = Counter()  # bashwrites exclusions by reason, merged into Graph.meta by extract.py
    for obj in iter_jsonl(path):
        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        ts = obj.get("timestamp")
        if msg.get("role") == "user":
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    entry = pending.pop(tool_use_id, None)
                    if entry is not None:
                        entry["end_ts"] = ts
                    fact = pending_artifact.pop(tool_use_id, None)
                    if fact is not None and block.get("is_error") is not True:
                        out.setdefault("artifact_facts", []).append(fact)
            continue
        if msg.get("role") != "assistant":
            continue
        cwd = obj.get("cwd") if isinstance(obj.get("cwd"), str) else None
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            inp = block.get("input")
            if not isinstance(inp, dict):
                continue
            if name in _SPAWN_TOOLS:
                tid = block.get("id")
                if tid:
                    out["tasks"][tid] = {
                        "ts": ts,
                        "description": inp.get("description"),
                        "subagent_type": inp.get("subagent_type"),
                        "requested_model": inp.get("model"),
                    }
            elif name == "Bash":
                cmd = inp.get("command")
                if isinstance(cmd, str):
                    entry = {"ts": ts, "end_ts": None, "command": cmd, "launch": detect_launch(cmd), "cwd": cwd}
                    out["bash"].append(entry)
                    if block.get("id"):
                        pending[block["id"]] = entry
                    out["writes"].extend(bashwrites.writes_from_command(cmd, ts, cwd=cwd, exclusions=excluded))
                    # A1 hook: the one line task A1 may add to this file goes here:
                    out["reads"].extend(bashwrites.reads_from_command(cmd, ts, cwd=cwd, exclusions=excluded))
            elif name in _WRITE_TOOLS:
                p = _tool_path(inp)
                if p:
                    out["writes"].append({"ts": ts, "path": p, "tier": "verified", "how": name})
                    if block.get("id"):
                        pending_artifact[block["id"]] = _claude_artifact_fact(name, inp, ts, p)
            elif name in _READ_TOOLS:
                p = _tool_path(inp)
                if p:
                    out["reads"].append({"ts": ts, "path": p, "tier": "verified", "how": name})
    out["excluded"] = dict(excluded)
    return out


def read_meta(session_path: Path) -> dict | None:
    meta_path = session_path.with_suffix(".meta.json")
    try:
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def codex_first_prompt(session_path: Path, limit: int = 600) -> str | None:
    """The user's actual prompt in a codex rollout: the last user message
    before the first assistant message. Codex prepends injected user-role
    messages (`<recommended_plugins>`, environment context) ahead of the real
    prompt, so the first user message is boilerplate. Returns None when the
    rollout has no user message."""
    last = None
    for obj in iter_jsonl(session_path):
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "user_message" and isinstance(payload.get("message"), str):
            last = payload["message"]
            continue
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role == "assistant" and last is not None:
            break
        if role != "user":
            continue
        parts = payload.get("content")
        if isinstance(parts, list):
            chunks = [p.get("text") for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str)]
            text = "\n".join(chunks) if chunks else None
        elif isinstance(parts, str):
            text = parts
        else:
            text = None
        if text and text.strip():
            last = text
    return last.strip()[:limit] if last else None
