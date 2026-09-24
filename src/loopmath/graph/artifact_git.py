"""Read-only git fallback for artifact writes."""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Callable

from .scan import epoch
from .schema import GraphNode

GIT_LOG_ARGS = ("log", "--format=%H%x09%at%x09%an", "--name-only")
GIT_REASONS = ("no_cwd", "git_unavailable", "git_failed", "not_a_worktree", "log_failed")
GIT_EXE = "git"
GIT_COMMIT_RE = re.compile(r"\bgit\b[^;&|\n]*\bcommit\b")
NOT_A_WORKTREE_RE = re.compile(r"not a git repository|must be run in a work tree", re.I)
SLACK_S = 1.0
MAX_SAMPLES = 20
MAX_PATHS_IN_SAMPLE = 5
GIT_RULE_CONTAINS = "session interval contains the commit time"
GIT_RULE_COMMIT_CALL = "bash call running git commit at the commit time"


def parse_git_log(text: str) -> list[dict]:
    commits: list[dict] = []
    cur: dict | None = None
    for line in text.splitlines():
        if "\t" in line:
            h, _, rest = line.partition("\t")
            at_s, _, author = rest.partition("\t")
            try:
                at: int | None = int(at_s)
            except ValueError:
                at = None
            cur = {"commit": h.strip(), "at": at, "author": author, "paths": []}
            commits.append(cur)
        elif line.strip() and cur is not None:
            cur["paths"].append(line.strip())
    commits.reverse()
    return commits


def git_from_disk(cwd: str, *, git_exe: str = GIT_EXE, git_log_args: tuple[str, ...] = GIT_LOG_ARGS) -> tuple[str, str] | str:
    """Read a worktree's log without modifying it or contacting a network."""
    if not cwd or not os.path.isdir(cwd):
        return "no_cwd"
    try:
        top = subprocess.run([git_exe, "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return "git_unavailable"
    except (OSError, subprocess.SubprocessError):
        return "git_failed"
    if top.returncode != 0 or not top.stdout.strip():
        return "not_a_worktree" if NOT_A_WORKTREE_RE.search(top.stderr or "") else "git_failed"
    toplevel = top.stdout.strip()
    try:
        log = subprocess.run([git_exe, *git_log_args], cwd=toplevel, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "log_failed"
    return (toplevel, log.stdout) if log.returncode == 0 else "log_failed"


def iso(t: float | int) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cwds_of(scans: dict[str, dict], nids: list[str]) -> list[str]:
    counts: Counter = Counter()
    for nid in nids:
        scan = scans.get(nid) or {}
        for bash in scan.get("bash") or []:
            if isinstance(bash, dict) and isinstance(bash.get("cwd"), str) and bash["cwd"]:
                counts[bash["cwd"]] += 1
        origin = scan.get("origin") or {}
        if isinstance(origin, dict) and isinstance(origin.get("cwd"), str) and origin["cwd"]:
            counts[origin["cwd"]] += 1
    return [cwd for cwd, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def bash_spans(scan: dict) -> list[tuple[float, float, str]]:
    out = []
    for bash in scan.get("bash") or []:
        if not isinstance(bash, dict):
            continue
        start, end = epoch(bash.get("ts")), epoch(bash.get("end_ts"))
        if start is not None and end is not None and end >= start:
            out.append((start, end, str(bash.get("command") or "")))
    return out


def git_fallback_writes(scans: dict[str, dict], nodes: dict[str, GraphNode], *, git: Callable[[str], tuple[str, str] | str] = git_from_disk, meta: dict | None = None) -> dict[str, list[dict]]:
    meta = meta if meta is not None else {}
    counts: Counter = Counter()
    ambiguous_samples: list[dict] = []
    ambiguous_commits: list[str] = []
    outside_samples: list[str] = []
    no_worktree: list[str] = []
    no_worktree_reasons: dict[str, dict[str, str]] = {}
    cwd_failures: Counter = Counter()
    no_cwd: list[str] = []
    by_rule: Counter = Counter()
    writes: dict[str, list[dict]] = defaultdict(list)
    seen_paths = {str(write.get("path")) for scan in scans.values() for write in (scan.get("writes") or []) if write.get("path")}
    by_ws: dict[str, list[str]] = defaultdict(list)
    for nid in sorted(nodes):
        by_ws[str(nodes[nid].workspace)].append(nid)
    intervals: dict[str, tuple[float, float]] = {}
    for nid, node in nodes.items():
        start = epoch(node.ts)
        if start is None or not isinstance(node.wall_s, (int, float)):
            counts["nodes_without_interval"] += 1
            continue
        intervals[nid] = (start, start + float(node.wall_s))
    spans = {nid: bash_spans(scans.get(nid) or {}) for nid in nodes}
    for workspace in sorted(by_ws):
        nids = by_ws[workspace]
        cwds = cwds_of(scans, nids)
        if not cwds:
            no_cwd.append(workspace)
            continue
        worktrees: dict[str, str] = {}
        reasons: dict[str, str] = {}
        for cwd in cwds:
            hit = git(cwd)
            if isinstance(hit, tuple):
                toplevel, text = hit
                worktrees.setdefault(toplevel, text)
            else:
                reason = str(hit) if hit in GIT_REASONS else f"unknown:{hit}"
                reasons[cwd] = reason
                cwd_failures[reason] += 1
        if not worktrees:
            no_worktree.append(workspace)
            no_worktree_reasons[workspace] = reasons
            continue
        counts["worktrees"] += len(worktrees)
        timed = [nid for nid in nids if nid in intervals]
        seen_commits: set[str] = set()
        for toplevel in sorted(worktrees):
            for commit in parse_git_log(worktrees[toplevel]):
                if commit["commit"] in seen_commits:
                    continue
                seen_commits.add(commit["commit"])
                counts["commits"] += 1
                at = commit["at"]
                if at is None:
                    counts["commits_bad_time"] += 1
                    continue
                paths = [os.path.normpath(os.path.join(toplevel, path)) for path in commit["paths"]]
                contain = [nid for nid in timed if intervals[nid][0] - SLACK_S <= at <= intervals[nid][1] + SLACK_S]
                if not contain:
                    counts["commits_outside_sessions"] += 1
                    if len(outside_samples) < MAX_SAMPLES:
                        outside_samples.append(f"{commit['commit'][:12]} {iso(at)}")
                    continue
                unseen = []
                for path in paths:
                    if path in seen_paths:
                        counts["paths_seen_by_scans"] += 1
                    else:
                        unseen.append(path)
                spanning = [nid for nid in contain if any(start - SLACK_S <= at <= end + SLACK_S for start, end, _ in spans[nid])]
                committing = [nid for nid in spanning if any(start - SLACK_S <= at <= end + SLACK_S and GIT_COMMIT_RE.search(command) for start, end, command in spans[nid])]
                if len(contain) == 1:
                    level, rule = contain, GIT_RULE_CONTAINS
                elif len(committing) == 1:
                    level, rule = committing, GIT_RULE_COMMIT_CALL
                else:
                    level, rule = [], ""
                if len(level) == 1:
                    nid = level[0]
                    for path in unseen:
                        writes[nid].append({"ts": iso(at), "path": path, "tier": "heuristic", "how": "git commit", "commit": commit["commit"], "author": commit["author"], "rule": rule, "candidates": len(contain)})
                        counts["writes"] += 1
                        by_rule[rule] += 1
                    continue
                if not unseen:
                    continue
                counts["ambiguous_commits"] += 1
                counts["ambiguous_paths"] += len(unseen)
                ambiguous_commits.append(commit["commit"][:12])
                if len(ambiguous_samples) < MAX_SAMPLES:
                    ambiguous_samples.append({"commit": commit["commit"][:12], "ts": iso(at), "n_paths": len(unseen), "paths": unseen[:MAX_PATHS_IN_SAMPLE], "candidates": contain, "committer_hint": {"tier": "reported", "git_commit_call": committing, "bash_call_running": spanning}})
    meta.update({
        "git_fallback_worktrees": counts["worktrees"], "git_fallback_workspaces_no_cwd": no_cwd,
        "git_fallback_workspaces_no_worktree": no_worktree, "git_fallback_workspaces_no_worktree_reasons": no_worktree_reasons,
        "git_fallback_cwd_failures": {reason: cwd_failures[reason] for reason in sorted(cwd_failures)},
        "git_fallback_commits": counts["commits"], "git_fallback_commits_bad_time": counts["commits_bad_time"],
        "git_fallback_commits_outside_sessions": counts["commits_outside_sessions"], "git_fallback_commits_outside_sessions_sample": outside_samples,
        "git_fallback_paths_seen_by_scans": counts["paths_seen_by_scans"], "git_fallback_ambiguous_commits": counts["ambiguous_commits"],
        "git_fallback_ambiguous_paths": counts["ambiguous_paths"], "git_fallback_ambiguous_commit_ids": ambiguous_commits,
        "git_fallback_ambiguous_sample": ambiguous_samples, "git_fallback_writes": counts["writes"],
        "git_fallback_writes_by_rule": dict(sorted(by_rule.items())), "git_fallback_nodes_without_interval": counts["nodes_without_interval"],
    })
    return dict(writes)
