"""Dataset builder implementation split from :mod:`loopmath.graph.dataset`."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .scan import codex_first_prompt, iter_jsonl, epoch
from .schema import Graph, GraphNode, TIERS

from .dataset_core import *
from .dataset_core import _LAUNCH_MATCH_SLACK_S, _SLACK_S, _weakest

def _established(value, tier: str, how: str, **extra) -> dict:
    return {"value": value, "tier": tier, "how": how, **extra}


def _missing(reason: str, **extra) -> dict:
    return {"value": None, "reason": reason, **extra}


def _span(text: str | None, limit: int) -> dict:
    if text is None:
        return {}
    return {"chars": len(text), "truncated": len(text) > limit}


def _text_of(content) -> str | None:
    """The user-typed text of a Claude Code user message: a plain string, or the text
    blocks of a list (a list holding only tool results is not a prompt)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [b.get("text") for b in content if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)]
        if chunks:
            return "\n".join(chunks)
    return None


# A user line Claude Code writes for a slash command or its output, not a prompt a person typed.
_COMMAND_WRAPPER_RE = re.compile(r"^\s*<(command-name|command-message|command-args|local-command-stdout|local-command-caveat)\b")


def claude_head(path: Path) -> dict:
    """The transcript's own `cwd` (the first line carrying one) and its first user prompt:
    the first user message with text that is neither meta nor a slash-command wrapper
    (those are skipped and counted in `skipped`). Each None when the file has none."""
    cwd = prompt = None
    skipped = 0
    for obj in iter_jsonl(path):
        if cwd is None and isinstance(obj.get("cwd"), str) and obj["cwd"]:
            cwd = obj["cwd"]
        if prompt is None and obj.get("type") == "user" and not obj.get("isMeta"):
            msg = obj.get("message")
            if isinstance(msg, dict):
                text = _text_of(msg.get("content"))
                if text and text.strip():
                    if _COMMAND_WRAPPER_RE.match(text):
                        skipped += 1
                    else:
                        prompt = text.strip()
        if cwd is not None and prompt is not None:
            break
    return {"cwd": cwd, "prompt": prompt, "skipped": skipped}


def codex_head(path: Path) -> dict:
    """Same for a codex rollout: the cwd from `session_meta` (else the first
    `turn_context`), the prompt through `scan.codex_first_prompt`."""
    cwd = None
    for obj in iter_jsonl(path):
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if obj.get("type") == "session_meta" or payload.get("type") == "session_meta":
            if isinstance(payload.get("cwd"), str) and payload["cwd"]:
                cwd = payload["cwd"]
                break
        if obj.get("type") == "turn_context" or payload.get("type") == "turn_context":
            if cwd is None and isinstance(payload.get("cwd"), str) and payload["cwd"]:
                cwd = payload["cwd"]
    return {"cwd": cwd, "prompt": codex_first_prompt(path, limit=10**9), "skipped": 0}


def parse_git_log(text: str) -> tuple[list[dict], int]:
    """`git log --format=%H%x09%at%x09%s` lines as `{commit, at, message}` oldest first,
    plus the count of lines whose time did not parse (dropped, counted)."""
    commits: list[dict] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        h, _, rest = line.partition("\t")
        at_s, _, message = rest.partition("\t")
        try:
            at = int(at_s)
        except ValueError:
            bad += 1
            continue
        commits.append({"commit": h.strip(), "at": at, "message": message})
    commits.reverse()
    return commits, bad


def git_from_disk(cwd: str) -> tuple[str, str] | str:
    """`(toplevel, git log text)` for the worktree containing `cwd`, or one of
    `GIT_REASONS`. Read-only: `rev-parse` and `log` only, no network."""
    if not cwd or not os.path.isdir(cwd):
        return "no_cwd"
    try:
        top = subprocess.run([GIT_EXE, "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return "git_unavailable"
    except (OSError, subprocess.SubprocessError):
        return "git_failed"
    if top.returncode != 0 or not top.stdout.strip():
        if "not a git repository" in (top.stderr or "").lower() or "work tree" in (top.stderr or "").lower():
            return "not_a_worktree"
        return "git_failed"
    toplevel = top.stdout.strip()
    try:
        log = subprocess.run([GIT_EXE, *GIT_LOG_ARGS], cwd=toplevel, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return "log_failed"
    if log.returncode != 0:
        return "log_failed"
    return toplevel, log.stdout


def _iso(t: float | int) -> str:
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Commits:
    """Commit lists per worktree, each `git log` run once; failures cached by cwd."""

    def __init__(self, git: Callable[[str], tuple[str, str] | str], report: Report):
        self.git = git
        self.report = report
        self.by_cwd: dict[str, tuple[str, list[dict]] | str] = {}
        self.by_top: dict[str, list[dict]] = {}

    def for_cwd(self, cwd: str) -> tuple[str, list[dict]] | str:
        if cwd in self.by_cwd:
            return self.by_cwd[cwd]
        hit = self.git(cwd)
        if isinstance(hit, tuple):
            top, text = hit
            if top not in self.by_top:
                commits, bad = parse_git_log(text)
                self.by_top[top] = commits
                self.report.counters["git_worktrees"] += 1
                self.report.counters["git_commits_read"] += len(commits)
                if bad:
                    self.report.warn("git_commits_bad_time", f"worktree {top}: {bad} git log lines whose time did not parse were dropped")
            self.by_cwd[cwd] = (top, self.by_top[top])
        else:
            reason = str(hit) if hit in GIT_REASONS else f"unknown:{hit}"
            self.report.warn(f"git_cwd_{reason}", f"cwd {cwd}: git gave {reason}; no commits for nodes running there")
            self.by_cwd[cwd] = reason
        return self.by_cwd[cwd]


def _interval(n: GraphNode) -> dict:
    start = epoch(n.ts)
    if start is None:
        return _missing("the node's start time is missing or does not parse" if n.ts else "the node has no start time")
    if not isinstance(n.wall_s, (int, float)):
        return _missing("the node's wall clock is unknown, so its end time is unknown")
    return _established({"start": _iso(start), "end": _iso(start + float(n.wall_s)), "start_epoch": start, "end_epoch": start + float(n.wall_s)}, "verified", "record start time plus wall clock")


def _first_prompt(head: dict, n: GraphNode) -> dict:
    p = head.get("prompt")
    skipped = int(head.get("skipped") or 0)
    if not p:
        if skipped:
            return _missing(f"the transcript's only user text is {skipped} slash-command line(s), no prompt", skipped_command_lines=skipped)
        return _missing("the transcript has no user message with text")
    how = "last user message before the first assistant message of the rollout" if n.source == "codex" else "first user message of the transcript"
    if skipped:
        how += f" after {skipped} slash-command line(s)"
    return _established(p[:PROMPT_LIMIT], "verified", how, skipped_command_lines=skipped, **_span(p, PROMPT_LIMIT))


def _spawn_description(n: GraphNode, spawn_tier: str | None) -> dict:
    if n.source != "subagent":
        return _missing(f"the node is {'an' if n.source == 'external' else 'a'} {n.source} session, not a subagent")
    if not isinstance(n.spawn, dict):
        return _missing("the subagent has no spawn record")
    desc = n.spawn.get("description")
    if not isinstance(desc, str) or not desc:
        meta_desc = n.spawn.get("meta_description")
        if isinstance(meta_desc, str) and meta_desc:
            return _established(meta_desc, "reported", "the subagent's meta file description (no matching Task call)")
        return _missing("the spawn record has no description")
    if n.parent is None or spawn_tier is None:
        return _missing("the spawn record has a description but no spawn edge established it")
    how = "the parent's Task call matched by tool_use id" if spawn_tier == "verified" else "the meta file description; parent by path containment"
    return _established(desc, spawn_tier, how, subagent_type=n.spawn.get("subagent_type"))


def _parent_command(n: GraphNode, parent: GraphNode | None, edge_tier: str | None, edge_kind: str | None, tasks: dict, bash: list[dict], report: Report) -> dict:
    """The text the parent issued to start this node: the Task prompt for a spawn (read
    from the parent's transcript by tool_use id), the complete Bash command for a launch
    (read from the parent's transcript by the launch join's call time and command
    prefix). The tier is the edge's, as it is: an absent tier stays absent and is counted
    by `_tier_defects`, never defaulted."""
    if n.parent is None:
        return _missing("the node has no parent")
    if parent is None:
        return _missing(f"the parent {n.parent} is not a node of the graph")
    if edge_kind == "spawn":
        tid = (n.spawn or {}).get("tool_use_id")
        if not tid:
            return _missing("the spawn record has no tool_use id to find the Task call by")
        call = tasks.get(tid)
        if call is None:
            return _missing("the parent's transcript has no Task call with the spawn's tool_use id")
        prompt = call.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            return _missing("the parent's Task call has no prompt text")
        return _established(prompt[:COMMAND_LIMIT], "verified", "the parent's Task call matched by tool_use id", kind="spawn", **_span(prompt, COMMAND_LIMIT))
    if edge_kind == "launch":
        lb = n.launched_by if isinstance(n.launched_by, dict) else {}
        excerpt = lb.get("command")
        if not isinstance(excerpt, str) or not excerpt:
            return _missing("the launch record has no command text to find the Bash call by")
        start = epoch(n.ts)
        lag = lb.get("lag_s")
        if start is None or not isinstance(lag, (int, float)):
            return _missing("the launch record gives no call time (node start or lag_s missing), so the Bash call cannot be found")
        call_at = start - float(lag)
        hits = [b for b in bash if b["at"] is not None and abs(b["at"] - call_at) <= _LAUNCH_MATCH_SLACK_S and b["command"].startswith(excerpt)]
        if not hits:
            near = sum(1 for b in bash if b["at"] is not None and abs(b["at"] - call_at) <= _LAUNCH_MATCH_SLACK_S)
            return _missing(f"the parent's transcript has no Bash call at {_iso(call_at)} starting with the launch record's command text ({near} call(s) at that time)")
        if len(hits) > 1 and any(b["command"] != hits[0]["command"] for b in hits):
            report.warn("launch_calls_ambiguous", f"node {n.id}: {len(hits)} Bash calls of {n.parent} at {_iso(call_at)} start with the launch record's text and differ; the first in transcript order is kept")
        cmd = hits[0]["command"]
        return _established(cmd, edge_tier, f"the parent's Bash call matched by the launch join's call time and command prefix: {lb.get('how')}", kind="launch", chars=len(cmd), truncated=False, tool_use_id=hits[0]["id"])
    return _missing("the parent is set but no spawn or launch edge reaches the node")


def _parent_feature(n: GraphNode, edge_kind: str | None, edge_tier: str | None) -> dict:
    """The skeleton's parent with the tier of the edge that established it; a parent
    the edges do not explain carries no tier and is counted as a defect."""
    if n.parent is None:
        return _missing("the node has no parent")
    if edge_kind is None:
        return _established(n.parent, None, "the skeleton names a parent but no spawn or launch edge reaches the node")
    return _established(n.parent, edge_tier, f"the {edge_kind} edge into the node", kind=edge_kind)


def _bash_calls(path: Path) -> list[dict]:
    """Every Bash tool_use block of a Claude transcript, in order, with its complete
    command text, its time as an epoch (None when it does not parse) and its id."""
    out: list[dict] = []
    for obj in iter_jsonl(path):
        msg = obj.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Bash":
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                if isinstance(inp.get("command"), str):
                    out.append({"ts": obj.get("timestamp"), "at": epoch(obj.get("timestamp")), "command": inp["command"], "id": block.get("id")})
    return out


def _task_calls(path: Path) -> dict[str, dict]:
    """Task/Agent tool_use blocks of a Claude transcript by id, with their prompt text
    (the scanner keeps only the description; the prompt is the command the labeler reads)."""
    out: dict[str, dict] = {}
    for obj in iter_jsonl(path):
        msg = obj.get("message")
        if not isinstance(msg, dict) or msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") in ("Task", "Agent") and block.get("id"):
                inp = block.get("input") if isinstance(block.get("input"), dict) else {}
                out[block["id"]] = {"prompt": inp.get("prompt"), "description": inp.get("description")}
    return out


def _commits_in(interval: dict, cwd: dict, commits: _Commits) -> dict:
    if cwd["value"] is None:
        return _missing("no commits without the node's own cwd: " + cwd["reason"])
    if interval["value"] is None:
        return _missing("no commits without the node's own interval: " + interval["reason"])
    hit = commits.for_cwd(cwd["value"])
    if not isinstance(hit, tuple):
        return _missing(f"git could not read the worktree of {cwd['value']}: {hit}")
    top, lst = hit
    lo, hi = interval["value"]["start_epoch"] - _SLACK_S, interval["value"]["end_epoch"] + _SLACK_S
    inside = [{"commit": c["commit"], "at": _iso(c["at"]), "message": c["message"]} for c in lst if lo <= c["at"] <= hi]
    return _established(inside, "heuristic", "commit time inside the node's interval in the worktree of its cwd, over every local ref", worktree=top)


def _tier_defects(item: dict, report: Report) -> int:
    """Count every established feature value (the parent and the parent's command
    included) and every skeleton value whose tier is absent or outside `schema.TIERS`."""
    defects = 0
    for name, feat in sorted(item["features"].items()):
        if name == "skeleton":
            continue
        if feat.get("value") is not None and feat.get("tier") not in TIERS:
            defects += 1
            report.warn("tier_defects", f"node {item['id']}: feature {name} has tier {feat.get('tier')!r}, not one of {', '.join(TIERS)}")
    sk = item["features"]["skeleton"]
    for name, tier_key in (("role", "role_tier"), ("phase", "phase_tier"), ("model", "model_tier")):
        if sk.get(name) is not None and sk.get(tier_key) not in TIERS:
            defects += 1
            report.warn("tier_defects", f"node {item['id']}: skeleton {name} has tier {sk.get(tier_key)!r}, not one of {', '.join(TIERS)}")
    return defects


def node_items(graph: Graph, labels: Labels, *, workspaces: list[str], git: Callable[[str], tuple[str, str] | str] = git_from_disk, report: Report | None = None) -> list[dict]:
    """One item per graph node, sorted by id. `git(cwd)` is `git_from_disk` or a fake."""
    report = report if report is not None else Report()
    c = report.counters
    ids = Counter(n.id for n in graph.nodes)
    for nid, k in sorted(ids.items()):
        if k > 1:
            report.warn("duplicate_node_ids", f"node id {nid} appears {k} times in the graph; every occurrence is emitted")
    nodes = {n.id: n for n in graph.nodes}
    in_edge: dict[str, tuple[str, str]] = {}
    for e in graph.edges:
        if e.kind in ("spawn", "launch") and e.dst not in in_edge:
            in_edge[e.dst] = (e.kind, e.tier)
    commits = _Commits(git, report)
    heads: dict[str, dict] = {}
    tasks: dict[str, dict] = {}
    bash: dict[str, list[dict]] = {}
    items: list[dict] = []
    for n in sorted(graph.nodes, key=lambda n: n.id):
        path = Path(n.session_path)
        if n.id not in heads:
            heads[n.id] = codex_head(path) if n.source == "codex" else claude_head(path)
        head = heads[n.id]
        edge_kind, edge_tier = in_edge.get(n.id, (None, None))
        parent = nodes.get(n.parent) if n.parent else None
        if parent is not None and edge_kind == "spawn" and n.parent not in tasks:
            tasks[n.parent] = _task_calls(Path(parent.session_path)) if parent.source != "codex" else {}
        if parent is not None and edge_kind == "launch" and n.parent not in bash:
            bash[n.parent] = _bash_calls(Path(parent.session_path)) if parent.source != "codex" else []
        cwd = _established(head["cwd"], "verified", "the transcript's own cwd") if head["cwd"] else _missing("the transcript carries no cwd")
        interval = _interval(n)
        feats = {
            "skeleton": asdict(n),
            "parent": _parent_feature(n, edge_kind, edge_tier),
            "cwd": cwd,
            "interval": interval,
            "spawn_description": _spawn_description(n, edge_tier if edge_kind == "spawn" else None),
            "first_prompt": _first_prompt(head, n),
            "commits_in_interval": _commits_in(interval, cwd, commits),
            "parent_command": _parent_command(n, parent, edge_tier, edge_kind, tasks.get(n.parent or "", {}), bash.get(n.parent or "", []), report),
        }
        for name, feat in feats.items():
            if name != "skeleton" and feat["value"] is None:
                c[f"feature_missing:{name}"] += 1
                c[f"feature_missing_reason:{name}:{feat['reason']}"] += 1
        gold, elsewhere = labels.node_labels(n.id, n.workspace)
        if elsewhere:
            report.warn("labels_in_other_workspace", f"node {n.id} (workspace {n.workspace}): the id is labeled under workspace(s) {', '.join(elsewhere)}, not its own; those labels are not used")
        if n.workspace is None:
            c["nodes_without_workspace"] += 1
        for key in NODE_LABEL_KEYS:
            c[f"gold_labeled:{key}" if key in gold else f"gold_unlabeled:{key}"] += 1
            if key in gold:
                c[f"gold_tier:{key}:{gold[key]['tier']}"] += 1
        item = {
            "item": "node",
            "id": n.id,
            "workspace": n.workspace,
            "features": feats,
            "gold": {k: gold[k] for k in NODE_LABEL_KEYS if k in gold},
            "gold_missing": [k for k in NODE_LABEL_KEYS if k not in gold],
            "gold_workspace": n.workspace if gold else None,
        }
        item["tier_defects"] = _tier_defects(item, report)
        items.append(item)
    c["items_node"] = len(items)
    c["nodes_labeled"] = sum(1 for it in items if it["gold"])
    c["nodes_unlabeled"] = sum(1 for it in items if not it["gold"])
    for ws in workspaces:
        for nid in sorted((labels.swarms.get(ws) or {}).get("nodes", {})):
            if nid not in nodes:
                report.warn("labels_without_node", f"workspace {ws}: labeled node {nid} is not in the graph")
    return items


# --- edge items (D1b) --------------------------------------------------------------------
