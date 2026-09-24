"""Codex-origin corroboration for launch lineage.

Kept separate from script and command launch matching so each implementation
module remains small while ``loopmath.graph.launch`` retains its original API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _under(path: str, root: str) -> bool:
    """Whether ``path`` is ``root`` or a descendant, without importing launch."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)

ORIGIN_UNSCANNED = "launched by an unscanned session"
ORIGIN_NOT_LAUNCHED = "not launched: an interactive session by its own report"
ORIGIN_EXEC_OUTSIDE_WORKTREE = "ran as codex exec by its own report, from a cwd outside the record's worktree"
ORIGIN_EXEC_ELSEWHERE = "ran as codex exec by its own report, from a cwd whose workspace name is not the record's"
ORIGIN_EXEC_CWD_UNKNOWN = "ran as codex exec by its own report, but the report has no cwd"
ORIGIN_UNSUPPORTED = "origin not recognised: the reported originator/source pair is not an established one"
ORIGIN_MISSING = "origin unknown: the rollout has no session_meta"
ORIGIN_UNREADABLE = "origin unknown: the rollout could not be read completely"
_ORIGIN_FIELDS = ("originator", "source", "cwd", "cli_version")

# The pairs the CLI writes for a session a person drove (measured over the rollouts
# on the build machine: the TUI, the Rust CLI, the VS Code extension, the desktop
# apps). Anything else is unsupported and counted, never assumed interactive.
INTERACTIVE_ORIGINS = frozenset({
    ("codex-tui", "cli"),
    ("codex_cli_rs", "cli"),
    ("codex_cli", "cli"),
    ("codex_vscode", "vscode"),
    ("Codex Desktop", "vscode"),
    ("codex_work_desktop", "vscode"),
})

# the buckets of `unlaunched_codex`, in the order they are reported
UNLAUNCHED_BUCKETS = (
    "unlaunched_codex_exec_outside_worktree",
    "unlaunched_codex_exec_elsewhere",
    "unlaunched_codex_exec_cwd_unknown",
    "unlaunched_codex_not_launched",
    "unlaunched_codex_origin_unsupported",
    "unlaunched_codex_origin_missing",
    "unlaunched_codex_origin_unreadable",
)
LAUNCHED_BUCKETS = (
    "launches_origin_corroborated",
    "launches_origin_contradicted",
    "launches_origin_unsupported",
    "launches_origin_missing",
    "launches_origin_unreadable",
)

_origin_cache: dict[tuple, dict] = {}


def _origin_record(payload: dict) -> dict:
    return {**{f: payload.get(f) for f in _ORIGIN_FIELDS}, "tier": "reported"}


def read_session_meta(path: Path) -> dict:
    """Read the whole rollout for its `session_meta` record. Returns `{"status":
    "found", "origin": {...}}` at the first `session_meta` with a payload;
    `{"status": "absent", "lines": n}` when every line was read and parsed and none
    was one (an established absence); `{"status": "unreadable", "error": ...,
    "lines": n, "unparsed": k}` when the file could not be opened or read to its end,
    or some lines were not JSON (absence is then not established)."""
    lines = unparsed = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                lines += 1
                try:
                    obj = json.loads(raw)
                except ValueError:
                    unparsed += 1
                    continue
                if not isinstance(obj, dict):
                    unparsed += 1
                    continue
                if obj.get("type") != "session_meta":
                    continue
                payload = obj.get("payload")
                if isinstance(payload, dict):
                    return {"status": "found", "origin": _origin_record(payload)}
                unparsed += 1  # a session_meta without a payload dict: not a report
    except OSError as exc:
        return {"status": "unreadable", "error": f"{type(exc).__name__}: {exc}", "lines": lines, "unparsed": unparsed}
    if unparsed:
        return {"status": "unreadable", "error": f"{unparsed} of {lines} lines are not JSON records", "lines": lines, "unparsed": unparsed}
    return {"status": "absent", "lines": lines}


def scan_origin_is_complete(origin) -> bool:
    """A2's `origin` is a complete report of the rollout's `session_meta`: found
    (not `session_meta: False`), every field a string, none listed missing and none
    filled in from a turn (`cwd_from`). Anything else is re-read from the rollout."""
    if not isinstance(origin, dict) or origin.get("session_meta") is False:
        return False
    if origin.get("missing") or "cwd_from" in origin:
        return False
    return all(isinstance(origin.get(f), str) and origin.get(f) for f in _ORIGIN_FIELDS)


def codex_origin_status(session_path: str | Path, scan: dict | None = None) -> dict:
    """The origin of a codex rollout with how it was established: the scan's `origin`
    when it is a complete report (status `found`), else the rollout's own
    `session_meta` through `read_session_meta` (memoised per process by
    `(path, mtime, size)`)."""
    from .scan import cache_key  # lazy: scan.py imports detect_launch from here

    origin = (scan or {}).get("origin")
    if scan_origin_is_complete(origin):
        return {"status": "found", "origin": _origin_record(origin)}
    path = Path(session_path)
    key = cache_key(path)
    if key is not None and key in _origin_cache:
        return dict(_origin_cache[key])
    result = read_session_meta(path)
    if key is not None:
        _origin_cache[key] = dict(result)
    return result


def codex_origin(session_path: str | Path, scan: dict | None = None) -> dict | None:
    """`{originator, source, cwd, cli_version, tier: "reported"}` for a codex rollout,
    or None when no `session_meta` was found (see `codex_origin_status` for why)."""
    return codex_origin_status(session_path, scan).get("origin")


def origin_pair(origin: dict | None) -> tuple:
    """`(originator, source)` as the record has them; a non-string value is kept as
    written (a dict source is a real shape) but the pair then matches nothing."""
    if not origin:
        return (None, None)
    return (origin.get("originator"), origin.get("source"))


def origin_pair_label(origin: dict | None) -> str:
    """`originator/source` for counting, with a non-string source rendered as JSON."""
    o, s = origin_pair(origin)
    return "/".join(x if isinstance(x, str) else json.dumps(x, sort_keys=True) for x in (o, s))


def origin_says_exec(origin: dict | None) -> bool:
    """The session reports it ran as `codex exec`: both fields say so."""
    return origin_pair(origin) == ("codex_exec", "exec")


def origin_says_interactive(origin: dict | None) -> bool:
    """The session reports an established interactive start."""
    pair = origin_pair(origin)
    return all(isinstance(x, str) for x in pair) and pair in INTERACTIVE_ORIGINS


CWD_IN_WORKTREE = "in_worktree"
CWD_OUTSIDE_WORKTREE = "outside_worktree"
CWD_WORKSPACE_NAME = "workspace_name"
CWD_ELSEWHERE = "elsewhere"
CWD_UNKNOWN = "unknown"


def cwd_placement(cwd: str | None, workspace: str | None, worktree: str | None = None) -> str:
    """Where a reported `cwd` lies relative to the session's record. With a worktree
    on the record: `in_worktree` when the canonical (`realpath`) cwd is the canonical
    worktree or under it, else `outside_worktree`; the workspace name is never
    consulted then. Without a worktree: `workspace_name` when the cwd's workspace
    label (`ingest.base.workspace_name`, the label the ingest derives from a cwd)
    is the record's workspace, else `elsewhere`. `unknown` when there is no cwd or
    no workspace to compare with."""
    if not cwd or not isinstance(cwd, str):
        return CWD_UNKNOWN
    if worktree:
        root = os.path.realpath(worktree)
        return CWD_IN_WORKTREE if _under(os.path.realpath(cwd), root) else CWD_OUTSIDE_WORKTREE
    if not workspace:
        return CWD_UNKNOWN
    from ..ingest.base import workspace_name

    return CWD_WORKSPACE_NAME if workspace_name(cwd) == workspace else CWD_ELSEWHERE


def cwd_in_workspace(cwd: str | None, workspace: str | None, worktree: str | None = None) -> bool:
    """`cwd_placement` qualifies the cwd as the session's workspace."""
    return cwd_placement(cwd, workspace, worktree) in (CWD_IN_WORKTREE, CWD_WORKSPACE_NAME)


def _unknown_origin_record(status: dict) -> dict:
    """The detail record for a rollout whose origin was not found: a verified
    absence, or a counted read failure."""
    if status.get("status") == "absent":
        return {"tier": "verified", "how": ORIGIN_MISSING, "lines_read": status.get("lines", 0)}
    return {"tier": "verified", "how": ORIGIN_UNREADABLE, "read_error": status.get("error"), "lines_read": status.get("lines", 0), "unparsed": status.get("unparsed", 0)}


def corroborate_origins(nodes, scans, records, launch_cmd, edges, meta) -> None:
    """Fold each codex session's reported origin into the launch result (see the
    L2 note above) and recompute the codex aggregates.

    Population: `codex_sessions` = `launched_codex` (a launch edge) +
    `codex_unscanned_launcher` (no launcher in scope, but the session reports
    `codex exec` from a cwd placed in the workspace: launched by an unscanned
    session) + `unlaunched_codex`. Launched sessions split into `LAUNCHED_BUCKETS`
    (`launches_origin_corroborated`: the session reports `codex exec`;
    `..._contradicted`: an established interactive pair; `..._unsupported`: any
    other pair; `..._missing`: no `session_meta`, whole rollout read;
    `..._unreadable`). `unlaunched_codex` is the sum of `UNLAUNCHED_BUCKETS`
    (`unlaunched_codex_exec_outside_worktree`, `..._exec_elsewhere`,
    `..._exec_cwd_unknown`, `..._not_launched`, `..._origin_unsupported`,
    `..._origin_missing`, `..._origin_unreadable`). The reason per session is in
    `unlaunched_codex_detail` and `codex_unscanned_launcher_detail` (sorted by id).
    `origin_unsupported_pairs` counts every unrecognised `originator/source` pair
    over all codex sessions."""
    worktrees = {str(r.get("run_id")): r.get("worktree") for r in records if r.get("run_id")}
    by_dst = {e.dst: e for e in edges if e.kind == "launch"}
    counts = {k: 0 for k in LAUNCHED_BUCKETS + ("codex_unscanned_launcher",) + UNLAUNCHED_BUCKETS}
    unsupported_pairs: dict[str, int] = {}
    detail_unlaunched: dict[str, dict] = {}
    detail_unscanned: dict[str, dict] = {}
    n_codex = n_launched = 0
    for nid in sorted(nodes):
        n = nodes[nid]
        if n.source != "codex":
            continue
        n_codex += 1
        status = codex_origin_status(n.session_path, scans.get(nid))
        origin = status.get("origin")
        if origin is not None and not origin_says_exec(origin) and not origin_says_interactive(origin):
            label = origin_pair_label(origin)
            unsupported_pairs[label] = unsupported_pairs.get(label, 0) + 1
        if nid in launch_cmd:
            n_launched += 1
            e = by_dst.get(nid)
            if origin is None:
                bucket = "launches_origin_missing" if status.get("status") == "absent" else "launches_origin_unreadable"
                counts[bucket] += 1
                rec = {**_unknown_origin_record(status), "corroborates": None}
            elif origin_says_exec(origin):
                counts["launches_origin_corroborated"] += 1
                rec = {**origin, "corroborates": True, "how": "the session reports it ran as codex exec"}
            elif origin_says_interactive(origin):
                counts["launches_origin_contradicted"] += 1
                rec = {**origin, "corroborates": False, "how": "the session reports an interactive start, not codex exec"}
            else:
                counts["launches_origin_unsupported"] += 1
                rec = {**origin, "corroborates": None, "how": ORIGIN_UNSUPPORTED, "pair": origin_pair_label(origin)}
            if e is not None:
                e.detail["origin"] = dict(rec)
            if isinstance(n.launched_by, dict):
                n.launched_by["origin"] = dict(rec)
            continue
        if origin is None:
            bucket = "unlaunched_codex_origin_missing" if status.get("status") == "absent" else "unlaunched_codex_origin_unreadable"
            counts[bucket] += 1
            detail_unlaunched[nid] = _unknown_origin_record(status)
        elif origin_says_exec(origin):
            place = cwd_placement(origin.get("cwd"), n.workspace, worktrees.get(nid))
            if place == CWD_UNKNOWN:
                counts["unlaunched_codex_exec_cwd_unknown"] += 1
                detail_unlaunched[nid] = {**origin, "how": ORIGIN_EXEC_CWD_UNKNOWN}
            elif place in (CWD_IN_WORKTREE, CWD_WORKSPACE_NAME):
                counts["codex_unscanned_launcher"] += 1
                detail_unscanned[nid] = {**origin, "how": ORIGIN_UNSCANNED, "cwd_placement": place}
            elif place == CWD_OUTSIDE_WORKTREE:
                counts["unlaunched_codex_exec_outside_worktree"] += 1
                detail_unlaunched[nid] = {**origin, "how": ORIGIN_EXEC_OUTSIDE_WORKTREE, "worktree": worktrees.get(nid)}
            else:
                counts["unlaunched_codex_exec_elsewhere"] += 1
                detail_unlaunched[nid] = {**origin, "how": ORIGIN_EXEC_ELSEWHERE}
        elif origin_says_interactive(origin):
            counts["unlaunched_codex_not_launched"] += 1
            detail_unlaunched[nid] = {**origin, "how": ORIGIN_NOT_LAUNCHED}
        else:
            counts["unlaunched_codex_origin_unsupported"] += 1
            detail_unlaunched[nid] = {**origin, "how": ORIGIN_UNSUPPORTED, "pair": origin_pair_label(origin)}
    meta.update(counts)
    meta["codex_sessions"] = n_codex
    meta["launched_codex"] = n_launched
    # the aggregate, recomputed after corroboration: what is left once the launched
    # and the unscanned-launched sessions are taken out of the population
    meta["unlaunched_codex"] = n_codex - n_launched - counts["codex_unscanned_launcher"]
    meta["origin_unsupported_pairs"] = dict(sorted(unsupported_pairs.items()))
    meta["unlaunched_codex_detail"] = detail_unlaunched
    meta["codex_unscanned_launcher_detail"] = detail_unscanned
