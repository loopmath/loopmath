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
from .dataset_core import _weakest
from .dataset_nodes import *
from .dataset_edges import *

GOLD_KEYS = (*NODE_LABEL_KEYS, "verdict", *ATTEMPT_LABEL_KEYS)


def build_dataset(graph: Graph, labels: Labels, *, workspaces: list[str], git: Callable[[str], tuple[str, str] | str] = git_from_disk, report: Report | None = None) -> list[dict]:
    """Node items with the E2 and contract-v3 gold attached, then edge items; the
    labeled versus unlabeled node split is recounted after the attachments."""
    report = report if report is not None else Report()
    c = report.counters
    nodes = node_items(graph, labels, workspaces=workspaces, git=git, report=report)
    attach_e2(nodes, labels, report)
    attach_contract(nodes, labels, report)
    # Recount every gold key from scratch: the E2 pass can replace a swarms.json role
    # with a stronger one, so the tiers node_items counted are stale by now.
    for key in [k for k in c if k.startswith(("gold_labeled:", "gold_unlabeled:", "gold_tier:"))]:
        del c[key]
    for key in GOLD_KEYS:
        c[f"gold_labeled:{key}"] = sum(1 for it in nodes if key in it["gold"])
        c[f"gold_unlabeled:{key}"] = sum(1 for it in nodes if key not in it["gold"])
        for it in nodes:
            if key in it["gold"]:
                c[f"gold_tier:{key}:{it['gold'][key]['tier']}"] += 1
    c["nodes_labeled"] = sum(1 for it in nodes if it["gold"])
    c["nodes_unlabeled"] = sum(1 for it in nodes if not it["gold"])
    return nodes + edge_items(graph, labels, workspaces=workspaces, report=report)


def labeled_workspaces(labels: Labels) -> dict[str, list[str]]:
    """Every workspace a label file names, with the files naming it: the keys of
    `swarms.json`, the `workspace` of every E2 run, and the first path component of
    the contract-v3 `source` (the ledger's own workspace)."""
    out: dict[str, list[str]] = {}
    for ws in labels.swarms:
        out.setdefault(ws, []).append("swarms.json")
    for run in labels.e2_arms:
        ws = run.get("workspace")
        if isinstance(ws, str) and ws and "e2-arms.json" not in out.setdefault(ws, []):
            out[ws].append("e2-arms.json")
    src = (labels.contract_v3 or {}).get("source")
    if isinstance(src, str) and src:
        head = src.strip("/").split("/")[0]
        if head and "contract-v3.json" not in out.setdefault(head, []):
            out[head].append("contract-v3.json")
    return {ws: out[ws] for ws in sorted(out)}


def transcript_cwd(path: Path, harness: str | None) -> str | None:
    """The transcript's own cwd read from its head: for codex the `session_meta` (else
    the first `turn_context`), for Claude Code the first line carrying one."""
    if harness == "codex":
        cwd = None
        for obj in iter_jsonl(path):
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                continue
            kind = obj.get("type") if obj.get("type") in ("session_meta", "turn_context") else payload.get("type")
            if kind == "session_meta" and isinstance(payload.get("cwd"), str) and payload["cwd"]:
                return payload["cwd"]
            if kind == "turn_context" and cwd is None and isinstance(payload.get("cwd"), str) and payload["cwd"]:
                cwd = payload["cwd"]
        return cwd
    for obj in iter_jsonl(path):
        if isinstance(obj.get("cwd"), str) and obj["cwd"]:
            return obj["cwd"]
    return None


def resolve_workspaces(records: list[dict], workspaces: list[str], report: Report, *, cwd_of: Callable[[Path, str | None], str | None] = transcript_cwd) -> list[dict]:
    """Records for `extract`: a requested workspace no record carries as its ingest
    label is matched against a path component of each transcript's own cwd (ingest
    labels `.../Workspace/e2-runs/e2-T1-S` as `e2-runs`; the label file says `e2-T1-S`).
    Matched records are copied with the requested name as `workspace` and the ingest's
    label kept as `workspace_ingest`; everything is counted and printed. Records are
    never mutated."""
    c = report.counters
    have = Counter(r.get("workspace") for r in records)
    unmatched = [ws for ws in workspaces if have[ws] == 0]
    for ws in workspaces:
        c[f"workspace_records:{ws}"] = have[ws]
    if not unmatched:
        return records
    out: list[dict] = []
    matched: Counter = Counter()
    unresolved: dict[str, Counter] = {}  # ingest label -> reason -> records

    def leave(r: dict, reason: str) -> None:
        """A record that needed cwd resolution and got none: left under its ingest label,
        which `extract` will exclude; counted by ingest label and reason, never silent."""
        out.append(r)
        c["records_unresolved"] += 1
        c[f"records_unresolved_reason:{reason}"] += 1
        unresolved.setdefault(str(r.get("workspace")), Counter())[reason] += 1

    for r in records:
        path = r.get("session_path")
        if r.get("workspace") in workspaces:
            out.append(r)
            continue
        if not path:
            leave(r, "no_session_path")
            continue
        cwd = cwd_of(Path(str(path)), r.get("harness"))
        if not cwd:
            leave(r, "no_cwd_in_transcript")
            continue
        hits = [ws for ws in unmatched if ws in set(Path(cwd).parts)]
        if len(hits) == 1:
            out.append({**r, "workspace": hits[0], "workspace_ingest": r.get("workspace")})
            matched[hits[0]] += 1
            c[f"workspace_resolved_by_cwd:{hits[0]}"] += 1
        elif hits:
            report.warn("workspace_resolution_ambiguous", f"record {r.get('run_id')}: its cwd {cwd} names {len(hits)} requested workspaces ({', '.join(hits)}); left under its ingest label {r.get('workspace')}")
            leave(r, "cwd_names_several_requested_workspaces")
        else:
            leave(r, "cwd_names_no_requested_workspace")
    for ws in unmatched:
        if matched[ws]:
            report.warnings.append(f"workspace {ws}: no record carries that ingest label; {matched[ws]} record(s) matched by a path component of the transcript's own cwd carry it for this dataset")
        else:
            report.warn("workspaces_without_records", f"workspace {ws}: no record carries that ingest label and no transcript cwd has it as a path component; the workspace has no nodes")
    for label in sorted(unresolved):
        reasons = unresolved[label]
        c[f"records_unresolved_by_label:{label}"] = sum(reasons.values())
        report.warnings.append(f"ingest label {label}: {sum(reasons.values())} record(s) matched no requested workspace by transcript cwd (" + ", ".join(f"{reasons[k]} {k}" for k in sorted(reasons)) + "); they stay under that label and are excluded from extraction")
    return out


def resolution_lines(report: Report) -> list[str]:
    """The workspace resolution printout: records per requested workspace by ingest
    label and by transcript cwd, and every record the resolution left unresolved, by
    ingest label and by reason. Printed before extraction runs."""
    c = report.counters
    keys = [k for k in sorted(c) if k.startswith(("workspace_records:", "workspace_resolved_by_cwd:"))]
    out = ["workspace records (by ingest label; resolved by transcript cwd):" + ("" if keys else " none")]
    out.extend(f"  {k.split(':', 1)[1]}: {c[k]}" + (" resolved by cwd" if k.startswith("workspace_resolved") else " by ingest label") for k in keys)
    out.append(f"records unresolved (needed a cwd match and got none; excluded from extraction): {c['records_unresolved']}")
    out.extend(f"  reason {k.split(':', 1)[1]}: {c[k]}" for k in sorted(c) if k.startswith("records_unresolved_reason:"))
    out.extend(f"  ingest label {k.split(':', 1)[1]}: {c[k]}" for k in sorted(c) if k.startswith("records_unresolved_by_label:"))
    return out


def resolution_keys(report: Report) -> set[str]:
    return {k for k in report.counters if k.startswith(("workspace_records:", "workspace_resolved_by_cwd:", "records_unresolved"))}


def workspace_summary(items: list[dict], report: Report) -> dict[str, Counter]:
    """Per workspace (and `total`): every count the total prints, under the same name.
    From the items: node and edge items, the labeled versus unlabeled split per gold key
    with its tiers, the edge outcomes by kind. From the workspace-qualified counters
    (`ws:<workspace>:<key>`, kept by the E2, contract-v3 and missing-edge passes): the
    missing edges, the E2 and the contract-v3 outcomes. `total` is the sum over the
    workspaces, so what it prints equals the sum of what each workspace prints."""
    out: dict[str, Counter] = {}
    for it in items:
        ws = it["workspace"] or "(none)"
        for key in (ws, "total"):
            s = out.setdefault(key, Counter())
            if it["item"] == "node":
                s["nodes"] += 1
                s["nodes_labeled" if it["gold"] else "nodes_unlabeled"] += 1
                for gk in GOLD_KEYS:
                    if gk in it["gold"]:
                        s[f"gold_labeled:{gk}"] += 1
                        s[f"gold_tier:{gk}:{it['gold'][gk]['tier']}"] += 1
                    else:
                        s[f"gold_unlabeled:{gk}"] += 1
            else:
                s["edges"] += 1
                s[f"edges:{it['kind']}"] += 1
                outcome = it["gold_edge"]["value"] or "unlabeled"
                s[f"edge_{outcome}"] += 1
                s[f"edge_{outcome}:{it['kind']}"] += 1
                if outcome == "positive":
                    s[f"edge_positive_tier:{it['kind']}:{it['gold_edge'].get('tier')}"] += 1
    for key, n in report.counters.items():
        if key.startswith("ws:"):
            _, ws, name = key.split(":", 2)
            out.setdefault(ws, Counter())[name] += n
            out.setdefault("total", Counter())[name] += n
    return out


# --- printing and CLI --------------------------------------------------------------------


def meta_lines(meta: dict) -> list[str]:
    """Every `Graph.meta` counter, generically: integer keys as they are, list keys by
    length, dict keys by their sorted contents; nothing chosen by name."""
    lines = []
    for key in sorted(meta):
        v = meta[key]
        if isinstance(v, bool):
            lines.append(f"  {key}: {v}")
        elif isinstance(v, int):
            lines.append(f"  {key}: {v}")
        elif isinstance(v, float):
            lines.append(f"  {key}: {v}")
        elif isinstance(v, list):
            lines.append(f"  {key}: {len(v)} entries")
        elif isinstance(v, dict):
            lines.append(f"  {key}: {json.dumps(v, sort_keys=True)}")
        elif v is None:
            lines.append(f"  {key}: None")
        else:
            lines.append(f"  {key}: {v}")
    return lines


def _by_reason(c: Counter, prefix: str, shown: set[str]) -> list[str]:
    """`prefix:<kind>:<reason>` counters as `kind: n: reason` lines; the keys used go
    into `shown` so the generic tail prints only what no structured line carried."""
    keys = [k for k in sorted(c) if k.startswith(prefix)]
    shown.update(keys)
    return [f"  {k.split(':', 2)[1]}: {c[k]}: {k.split(':', 2)[2]}" for k in keys] or ["  none"]


def _tiers(c: Counter, prefix: str, shown: set[str]) -> str:
    keys = [k for k in sorted(c) if k.startswith(prefix)]
    shown.update(keys)
    return ", ".join(f"{k.split(':')[2]} {c[k]}" for k in keys)


def _e2_line(get: Callable[[str], int]) -> str:
    return (f"E2: runs {get('e2_runs')}, verdicts attached {get('e2_verdicts_attached')}, verdicts weakened by a weaker dev-session match {get('e2_verdicts_weakened')}, "
            f"dev sessions absent from the graph {get('e2_dev_sessions_absent')}, runs without a dev session label {get('e2_runs_without_dev_session')}, "
            f"runs without a verdict {get('e2_runs_without_verdict')}, dev sessions shared by runs {get('e2_dev_sessions_shared')}, roles attached {get('e2_roles_attached')}, "
            f"role sessions absent {get('e2_role_sessions_absent')}, referee sessions named {get('e2_referee_sessions_named')}")


def _contract_line(get: Callable[[str], int]) -> str:
    return (f"contract-v3: attempts {get('attempts')}, attached {get('attempts_attached')}, sessions absent from the ledger's workspace {get('attempts_session_absent')}, "
            f"sessions with a same-id node in another workspace {get('attempts_session_in_other_workspace')}, without a session label {get('attempts_without_session')}, "
            f"source workspace unknown {get('attempts_source_workspace_unknown')}, "
            + ", ".join(f"{key} disagreements {get(f'attempt_labels_disagree:{key}')}, {key} merged {get(f'attempt_labels_merged:{key}')}, {key} missing on an attempt {get(f'attempt_labels_missing:{key}')}" for key in ATTEMPT_LABEL_KEYS))


_E2_CONTRACT_KEYS = ("e2_runs", "e2_verdicts_attached", "e2_verdicts_weakened", "e2_dev_sessions_absent", "e2_runs_without_dev_session", "e2_runs_without_verdict", "e2_dev_sessions_shared", "e2_roles_attached", "e2_role_sessions_absent", "e2_referee_sessions_named",
                    "attempts", "attempts_attached", "attempts_session_absent", "attempts_session_in_other_workspace", "attempts_without_session", "attempts_source_workspace_unknown",
                    *(f"attempt_labels_{what}:{key}" for key in ATTEMPT_LABEL_KEYS for what in ("disagree", "merged", "missing")))


def summary_lines(name: str, s: Counter, indent: str = "  ") -> list[str]:
    """One workspace's (or the total's) summary block: the same lines, with the same
    names, for every workspace and for the total; any count the structured lines did
    not carry prints under `other`."""
    shown: set[str] = set()

    def get(key: str) -> int:
        shown.add(key)
        return s[key]

    sub = indent + "  "
    edges = ", ".join(f"{kind} {get(f'edges:{kind}')} (positive {get(f'edge_positive:{kind}')}, negative {get(f'edge_negative:{kind}')}, unlabeled {get(f'edge_unlabeled:{kind}')})" for kind in EDGE_GOLD_KEYS)
    out = [f"{indent}{name}:",
           f"{sub}nodes {get('nodes')} (labeled {get('nodes_labeled')}, unlabeled {get('nodes_unlabeled')}); edges {get('edges')} (positive {get('edge_positive')}, negative {get('edge_negative')}, unlabeled {get('edge_unlabeled')}); {edges}"]
    for key in GOLD_KEYS:
        tiers = _tiers(s, f"gold_tier:{key}:", shown)
        out.append(f"{sub}gold {key}: labeled {get(f'gold_labeled:{key}')}, unlabeled {get(f'gold_unlabeled:{key}')}" + (f" (tiers: {tiers})" if tiers else ""))
    for kind in EDGE_GOLD_KEYS:
        tiers = _tiers(s, f"edge_positive_tier:{kind}:", shown)
        if tiers:
            out.append(f"{sub}{kind} positive tiers: {tiers}")
    out.append(f"{sub}edges the labels have and the graph lacks: parent {get('edges_missing:parent')} (present {get('edges_labeled_present:parent')}), artifact {get('edges_missing:artifact')} (present {get('edges_labeled_present:artifact')})")
    out.append(sub + _e2_line(get))
    out.append(sub + _contract_line(get))
    other = [k for k in sorted(s) if k not in shown]
    if other:
        out.append(f"{sub}other: " + ", ".join(f"{k} {s[k]}" for k in other))
    return out


def report_lines(graph: Graph, labels: Labels, report: Report, items: list[dict] | None = None, *, resolution: bool = True) -> list[str]:
    """The full printout: label files, their counters and every warning, the graph's
    meta, the item counts with the labeled versus unlabeled split per label kind, the
    edge outcomes and the missing edges by reason, the E2 and contract-v3 attachment
    counts, and (when `items` is given) the same counts per workspace and in total.
    Every counter no structured line carried prints under `dataset counters`: the
    structured lines record the keys they used, nothing is hidden by name prefix."""
    out = [f"label files under {labels.directory}: present {', '.join(labels.present) or 'none'}; missing {', '.join(labels.missing) or 'none'}"]
    lc = labels.report.counters
    out.append("label loading counters:" + ("" if lc else " none"))
    out.extend(f"  {k}: {lc[k]}" for k in sorted(lc))
    out.append(f"label loading warnings: {len(labels.report.warnings)}")
    out.extend(f"  {w}" for w in labels.report.warnings)
    out.append("graph meta:")
    out.extend(meta_lines(graph.meta))
    c = report.counters
    shown: set[str] = set()

    def n(key: str) -> int:
        shown.add(key)
        return c[key]

    out.append(f"items: node {n('items_node')} (nodes labeled {n('nodes_labeled')}, unlabeled {n('nodes_unlabeled')})")
    out.append(f"items: edge {n('items_edge')}")
    for key in GOLD_KEYS:
        tiers = _tiers(c, f"gold_tier:{key}:", shown)
        out.append(f"  gold {key}: labeled {n(f'gold_labeled:{key}')}, unlabeled {n(f'gold_unlabeled:{key}')}" + (f" (tiers: {tiers})" if tiers else ""))
    out.append("edge items by kind (positive, negative, unlabeled):")
    for kind in EDGE_GOLD_KEYS:
        tiers = _tiers(c, f"edge_positive_tier:{kind}:", shown)
        out.append(f"  {kind}: {n(f'edges:{kind}')} (positive {n(f'edge_positive:{kind}')}, negative {n(f'edge_negative:{kind}')}, unlabeled {n(f'edge_unlabeled:{kind}')})" + (f" (positive tiers: {tiers})" if tiers else ""))
    out.append("edge negatives (count by reason):")
    out.extend(_by_reason(c, "edge_negative_reason:", shown))
    out.append("edge unlabeled (count by reason):")
    out.extend(_by_reason(c, "edge_unlabeled_reason:", shown))
    out.append(f"edges the labels have and the graph lacks: parent {n('edges_missing:parent')} (present {n('edges_labeled_present:parent')}), artifact {n('edges_missing:artifact')} (present {n('edges_labeled_present:artifact')})")
    out.extend(_by_reason(c, "edges_missing_reason:", shown))
    out.append(_e2_line(n))
    out.append(_contract_line(n))
    out.append("features missing (count by reason):")
    out.extend(_by_reason(c, "feature_missing_reason:", shown))
    shown.update(k for k in c if k.startswith("feature_missing:"))  # the per-feature totals of the reasons above
    out.append("edge features missing (count by reason):")
    out.extend(_by_reason(c, "edge_feature_missing_reason:", shown))
    shown.update(k for k in c if k.startswith("edge_feature_missing:"))
    if resolution:
        out.extend(resolution_lines(report))
    shown.update(resolution_keys(report))
    if items is not None:
        out.append("per workspace (the same counts, under the same names, for every workspace and in total):")
        summary = workspace_summary(items, report)
        for ws in sorted(summary):
            if ws != "total":
                out.extend(summary_lines(ws, summary[ws]))
        out.extend(summary_lines("total", summary.get("total", Counter())))
    shown.update(k for k in c if k.startswith("ws:"))  # printed per workspace above
    other = [k for k in sorted(c) if k not in shown]
    out.append("dataset counters:" + ("" if other else " none"))
    out.extend(f"  {k}: {c[k]}" for k in other)
    out.append(f"dataset warnings: {len(report.warnings)}")
    out.extend(f"  {w}" for w in report.warnings)
    return out


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, sort_keys=True, ensure_ascii=False) + "\n")


def _load_records(args) -> list[dict]:
    from .. import grade as grade_mod, ingest, price as price_mod

    since = None if args.all else args.since
    if args.logs is not None:
        since = None
    records, _diag = ingest.parse_all(args.logs, limit=None, use_cache=not args.no_cache, progress=None, since_days=since)
    records, _cov = grade_mod.grade_all(records)
    records, _warn = price_mod.price_all(records, price_mod.load_prices(None))
    return records


def main(argv: list[str] | None = None) -> int:
    from ..cli import DEFAULT_SINCE_DAYS, _resolve_out
    from .extract import extract

    p = argparse.ArgumentParser(prog="python -m loopmath.graph.dataset", description="write the labeling eval dataset (node and edge items) for one or more workspaces")
    p.add_argument("--workspace", action="append", default=[], help="workspace name to include (repeatable; required unless --all)")
    p.add_argument("--labels", required=True, help="directory holding swarms.json, e2-arms.json, contract-v3.json (any subset)")
    p.add_argument("--out", default=None, help="write the items as jsonl here (inside the git worktree of the current directory)")
    p.add_argument("--logs", default=None, help="parse this path instead of the default log roots")
    p.add_argument("--since", type=float, default=DEFAULT_SINCE_DAYS, help=f"only read log files modified in the last N days (default {DEFAULT_SINCE_DAYS:g})")
    p.add_argument("--all", action="store_true", help="read every log file, ignoring --since, and include every workspace a label file names in addition to --workspace")
    p.add_argument("--no-cache", action="store_true", help="reparse everything, ignoring the cache")
    args = p.parse_args(argv)
    if not args.workspace and not args.all:
        p.error("--workspace or --all is required")

    out_path = None
    if args.out:
        out_path, err = _resolve_out(args.out)
        if err:
            print(err, file=sys.stderr)
            return 2
    labels = load_labels(args.labels)
    requested = list(dict.fromkeys(args.workspace))
    named = labeled_workspaces(labels) if args.all else {}
    workspaces = requested + [ws for ws in named if ws not in requested]
    report = Report()
    records = resolve_workspaces(_load_records(args), workspaces, report)
    print(f"workspaces requested: {', '.join(requested) or 'none'}; from the label files: " + (", ".join(f"{ws} ({', '.join(named[ws])})" for ws in named if ws not in requested) or ("none" if args.all else "not asked (no --all)")))
    print(f"records parsed: {len(records)}")
    print("\n".join(resolution_lines(report)))
    print(f"workspace resolution warnings: {len(report.warnings)}")
    print("\n".join(f"  {w}" for w in report.warnings))
    graph = extract(records, workspaces=workspaces)
    items = build_dataset(graph, labels, workspaces=workspaces, report=report)
    if out_path is not None:
        write_jsonl(items, out_path)
    print(f"graph nodes: {len(graph.nodes)}; graph edges: {len(graph.edges)}")
    print("\n".join(report_lines(graph, labels, report, items, resolution=False)))
    if out_path is not None:
        print(f"wrote {len(items)} items to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
