"""Artifacts written and read by a matched session.

Owner: lane 02. Spec: design/0.1/ (02 `run finish`: "attaches artifacts found
in the logs"; 03 section 6 `Match.artifacts`).

The graph's own scanners list what each file of the match wrote and read:
Claude Code Write, Edit and Read tool calls and shell commands
(`graph.scan.scan_claude_session`), Codex patches and shell commands
(`graph.codexio.scan_codex_session`). The session and every child are
scanned, and the results are merged per path: one record per path with its
kind (`graph.artifact_kinds.artifact_kind`), how many writes and reads, the
best tier seen and the first and last time. A path inside the session's
working folder is made relative to it. Metadata only: no content, no diff, no
command text.

A match clipped to a `--session self` attempt (Analyst D47) keeps only the
session file's entries inside the window; the sub-agents it kept count in
full. A sidechain agent embedded in the session file is cut by entry time
here, as the scanners do not name the agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..graph.artifact_kinds import artifact_kind
from ..graph.codexio import scan_codex_session
from ..graph.scan import scan_claude_session
from ..ingest.base import parse_ts, ts_epoch
from .match import session_cwd

_TIER_RANK = {"verified": 3, "reported": 2, "heuristic": 1}


def _display(path: str, cwd: str | None) -> str:
    if cwd and os.path.isabs(path):
        base = os.path.realpath(os.path.expanduser(cwd))
        full = os.path.realpath(path)
        if full == base or full.startswith(base + os.sep):
            return os.path.relpath(full, base)
    return path


def artifacts_for(match) -> list[dict[str, Any]]:
    """Files the matched session and its children wrote or read, one record per path."""
    files: list[str] = [str(match.session_path)]
    for part in getattr(match, "parts", None) or []:
        # Every file of the session and of each child (a resumed Codex
        # thread continues in new files).
        for file in part.get("files") or ([part["path"]] if part.get("path") else []):
            if file not in files:
                files.append(file)
    cwd = session_cwd(match.harness, Path(match.session_path))
    scan = scan_claude_session if match.harness == "claude-code" else scan_codex_session
    clip = getattr(match, "clip", None) or {}
    lo, hi = (ts_epoch(parse_ts(clip.get(k))) for k in ("from", "to"))

    by_path: dict[str, dict[str, Any]] = {}
    for file in files:
        found = scan(Path(file))
        for op, entries in (("writes", found.get("writes") or []), ("reads", found.get("reads") or [])):
            for entry in entries:
                raw = entry.get("path")
                if not isinstance(raw, str) or not raw:
                    continue
                if clip and file == str(match.session_path):
                    at = ts_epoch(parse_ts(entry.get("ts")))
                    if at is None or (lo is not None and at < lo) or (hi is not None and at > hi):
                        continue
                path = _display(raw, cwd)
                rec = by_path.setdefault(
                    path, {"path": path, "writes": 0, "reads": 0, "tier": None, "first_at": None, "last_at": None}
                )
                rec[op] += 1
                tier = entry.get("tier")
                if _TIER_RANK.get(tier, 0) > _TIER_RANK.get(rec["tier"], 0):
                    rec["tier"] = tier
                ts = parse_ts(entry.get("ts"))
                if ts is not None:
                    if rec["first_at"] is None or ts < rec["first_at"]:
                        rec["first_at"] = ts
                    if rec["last_at"] is None or ts > rec["last_at"]:
                        rec["last_at"] = ts

    out = []
    for rec in by_path.values():
        rec["kind"] = artifact_kind(rec["path"])[0] or "other"
        out.append({k: v for k, v in rec.items() if v is not None})
    # Written files first, then read-only ones, each by path.
    return sorted(out, key=lambda r: (r["writes"] == 0, r["path"]))
