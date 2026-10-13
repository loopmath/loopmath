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
from .dataset_core import _pick, _rank, _weakest
from .dataset_nodes import *
from .dataset_nodes import _established, _interval, _missing, _tier_defects

EDGE_GOLD_KEYS = {"spawn": ("parent",), "launch": ("parent",), "artifact": tuple(ARTIFACT_LABEL_KEYS)}


def edge_id(e) -> str:
    """`kind:src->dst`, plus `:path` for an artifact edge; the item's id."""
    path = e.detail.get("path") if isinstance(e.detail, dict) else None
    return f"{e.kind}:{e.src}->{e.dst}" + (f":{path}" if e.kind == "artifact" and path else "")


def _node_feature(nid: str, nodes: dict[str, GraphNode], end: str) -> dict:
    n = nodes.get(nid)
    if n is None:
        return _missing(f"the edge's {end} {nid} is not a node of the graph")
    return _established(asdict(n), "verified", f"the skeleton record of the edge's {end}")


def _artifact_labels(labels: Labels, path: str, workspace: str | None) -> tuple[dict[str, dict], list[str]]:
    """The artifact's labels under `workspace`, plus the other workspaces labeling the
    same path (counted by the caller, never used: a swarm's labels describe its own
    sessions and the edge's nodes belong to `workspace`)."""
    found = labels.swarms.get(workspace, {}).get("artifacts", {}).get(path, {}) if workspace else {}
    elsewhere = sorted(ws for ws in labels.swarms if ws != workspace and path in labels.swarms[ws]["artifacts"])
    return found, elsewhere


def _parent_edge_gold(e, gold: dict[str, dict]) -> dict:
    lab = gold.get("parent")
    if lab is None:
        return _missing("the labels carry no parent for the destination node")
    if lab["value"] == e.src:
        return _established("positive", lab["tier"], f"the parent label of {e.dst} names {e.src}")
    return _established("negative", lab["tier"], f"the labels name {lab['value']} as the parent of {e.dst}, not {e.src}", reason="the labeled parent is another node")


def _artifact_edge_gold(e, gold: dict[str, dict], path: str) -> dict:
    prod, cons = gold.get("producer"), gold.get("consumers")
    if prod is None and cons is None:
        return _missing("the labels carry no artifact at the path" if not gold else "the labels for the path carry neither producer nor consumers")
    if cons is not None and e.dst not in cons["value"]:
        return _established("negative", cons["tier"], f"the labels do not list {e.dst} among the consumers of {path}", reason="the destination is not a labeled consumer")
    if prod is not None and prod["value"] != e.src:
        return _established("negative", prod["tier"], f"the labels name {prod['value']} as the producer of {path}, not {e.src}", reason="the labeled producer is another writer")
    if prod is None or cons is None:
        return _missing(f"the labels for the path carry only {'consumers' if prod is None else 'a producer'}, so the flow is not confirmed on both ends")
    return _established("positive", _weakest(prod["tier"], cons["tier"]), f"the labels name {e.src} as the producer of {path} and list {e.dst} among its consumers (the weaker of the two tiers)")


def _missing_edges(graph: Graph, labels: Labels, workspaces: list[str], report: Report) -> None:
    """Count every edge the labels have and the graph lacks, by reason: a parent label
    with no spawn or launch edge from that parent, a producer-to-consumer flow with no
    artifact edge. Never emitted as items; counted and printed."""
    c = report.counters
    nodes = {n.id: n for n in graph.nodes}
    parent_edges = {(e.src, e.dst) for e in graph.edges if e.kind in ("spawn", "launch")}
    parents_of: dict[str, set[str]] = {}
    for src, dst in parent_edges:
        parents_of.setdefault(dst, set()).add(src)
    art_edges = {(e.src, e.dst, e.detail.get("path")) for e in graph.edges if e.kind == "artifact"}
    writers_into: dict[tuple[str, str], set[str]] = {}
    for src, dst, path in art_edges:
        writers_into.setdefault((dst, path), set()).add(src)
    art_paths = {a.id for a in graph.artifacts}

    def in_ws(nid: str, ws: str) -> str | None:
        """None when `nid` is a node of workspace `ws`, else the reason it is not."""
        n = nodes.get(nid)
        if n is None:
            return "is not in the graph"
        if n.workspace != ws:
            return f"is in the graph under workspace {n.workspace}, not {ws}"
        return None

    def count(kind: str, ws: str, reason: str, text: str) -> None:
        _count_ws(report, ws, f"edges_missing:{kind}")
        c[f"edges_missing_reason:{kind}:{reason}"] += 1
        report.warnings.append(f"missing {kind} edge, workspace {ws}: {text}")

    for ws in workspaces:
        entry = labels.swarms.get(ws) or {"nodes": {}, "artifacts": {}}
        for nid in sorted(entry["nodes"]):
            labs = entry["nodes"][nid]
            if "parent" not in labs:
                continue
            p = labs["parent"]["value"]
            if (p, nid) in parent_edges and in_ws(nid, ws) is None:
                _count_ws(report, ws, "edges_labeled_present:parent")
                continue
            why = in_ws(nid, ws)
            if why is not None:
                count("parent", ws, "the labeled node is not a node of the workspace", f"labeled node {nid} {why}; its parent label ({p}) has no edge to match")
            elif p not in nodes:
                count("parent", ws, "the labeled parent is not in the graph", f"the labeled parent {p} of {nid} is not in the graph")
            elif nid in parents_of:
                count("parent", ws, "the graph gives the node a different parent", f"the labels name {p} as the parent of {nid}; the graph's edge comes from {', '.join(sorted(parents_of[nid]))}")
            else:
                count("parent", ws, "the graph has no spawn or launch edge into the node", f"the labels name {p} as the parent of {nid}; the graph has no spawn or launch edge into {nid}")
        for path in sorted(entry["artifacts"]):
            labs = entry["artifacts"][path]
            if "producer" not in labs or "consumers" not in labs:
                report.warn("artifact_labels_incomplete", f"workspace {ws}: artifact {path} is labeled without {'a producer' if 'producer' not in labs else 'consumers'}; no flow to check against the graph")
                continue
            p = labs["producer"]["value"]
            for cons in sorted(labs["consumers"]["value"]):
                if (p, cons, path) in art_edges:
                    _count_ws(report, ws, "edges_labeled_present:artifact")
                    continue
                others = writers_into.get((cons, path))
                if others:
                    count("artifact", ws, "the graph's edge into the consumer comes from another writer, not the labeled producer", f"{path}: the labels name {p} as the producer and {cons} as a consumer; the graph's artifact edge into {cons} comes from {', '.join(sorted(others))}")
                elif in_ws(p, ws) is not None or in_ws(cons, ws) is not None:
                    who = p if in_ws(p, ws) is not None else cons
                    count("artifact", ws, "a labeled node is not a node of the workspace", f"{path}: labeled {'producer' if who == p else 'consumer'} {who} {in_ws(who, ws)}")
                elif path not in art_paths:
                    count("artifact", ws, "the graph has no artifact at the path", f"{path}: labeled producer {p} and consumer {cons}, but the graph has no artifact at the path")
                else:
                    count("artifact", ws, "the graph has no artifact edge into the consumer", f"{path}: labeled producer {p} and consumer {cons}, but the graph has no artifact edge into {cons} on that path")


def edge_items(graph: Graph, labels: Labels, *, workspaces: list[str], report: Report | None = None) -> list[dict]:
    """One item per candidate edge of the graph, sorted by (kind, src, dst, path); every
    occurrence is emitted, repeated ids counted. Gold is looked up under the destination
    node's workspace (the same rule as node items); the missing-edge counters cover the
    labels of `workspaces`."""
    report = report if report is not None else Report()
    c = report.counters
    nodes = {n.id: n for n in graph.nodes}
    ordered = sorted(graph.edges, key=lambda e: (e.kind, e.src, e.dst, str(e.detail.get("path") or "")))
    ids = Counter(edge_id(e) for e in ordered)
    for eid, k in sorted(ids.items()):
        if k > 1:
            report.warn("duplicate_edge_ids", f"edge {eid} appears {k} times in the graph; every occurrence is emitted")
    items: list[dict] = []
    for e in ordered:
        if e.kind not in EDGE_GOLD_KEYS:
            report.warn("edges_unknown_kind", f"edge {e.src}->{e.dst} has kind {e.kind!r}, not one of {', '.join(EDGE_GOLD_KEYS)}; emitted without gold")
        dst = nodes.get(e.dst)
        src = nodes.get(e.src)
        workspace = dst.workspace if dst is not None else (src.workspace if src is not None else None)
        path = e.detail.get("path") if isinstance(e.detail, dict) else None
        feats = {
            "src": _node_feature(e.src, nodes, "source"),
            "dst": _node_feature(e.dst, nodes, "destination"),
            "edge": _established({"kind": e.kind, "src": e.src, "dst": e.dst, "detail": dict(e.detail)}, e.tier, f"the graph's {e.kind} edge with its own evidence"),
            "src_interval": _interval(src) if src is not None else _missing(f"the edge's source {e.src} is not a node of the graph"),
            "dst_interval": _interval(dst) if dst is not None else _missing(f"the edge's destination {e.dst} is not a node of the graph"),
        }
        if e.kind == "artifact":
            art = next((a for a in graph.artifacts if a.id == path), None) if path else None
            feats["artifact"] = _established(asdict(art), e.tier, "the graph's artifact record; only as strong as the edge, its weaker end") if art is not None else _missing(f"the graph has no artifact record at {path}" if path else "the artifact edge names no path")
        for name, feat in feats.items():
            if feat["value"] is None:
                c[f"edge_feature_missing:{name}"] += 1
                c[f"edge_feature_missing_reason:{name}:{feat['reason']}"] += 1
        gold: dict[str, dict] = {}
        if e.kind in ("spawn", "launch"):
            labs, elsewhere = labels.node_labels(e.dst, workspace)
            gold = {k: labs[k] for k in EDGE_GOLD_KEYS[e.kind] if k in labs}
            verdict = _parent_edge_gold(e, gold)
        elif e.kind == "artifact" and path:
            labs, elsewhere = _artifact_labels(labels, path, workspace)
            gold = {k: labs[k] for k in EDGE_GOLD_KEYS[e.kind] if k in labs}
            verdict = _artifact_edge_gold(e, gold, path)
        else:
            elsewhere = []
            verdict = _missing("the artifact edge names no path to look up" if e.kind == "artifact" else f"no gold is defined for edge kind {e.kind!r}")
        if elsewhere:
            report.warn("edge_labels_in_other_workspace", f"edge {edge_id(e)} (workspace {workspace}): labeled under workspace(s) {', '.join(elsewhere)}, not its own; those labels are not used")
        outcome = verdict["value"] or "unlabeled"
        c[f"edge_{outcome}:{e.kind}"] += 1
        c[f"edges:{e.kind}"] += 1
        if outcome == "negative":
            c[f"edge_negative_reason:{e.kind}:{verdict['reason']}"] += 1
        else:
            c[f"edge_{outcome}_tier:{e.kind}:{verdict.get('tier')}" if outcome == "positive" else f"edge_unlabeled_reason:{e.kind}:{verdict['reason']}"] += 1
        item = {
            "item": "edge",
            "id": edge_id(e),
            "kind": e.kind,
            "workspace": workspace,
            "features": feats,
            "gold": gold,
            "gold_missing": [k for k in EDGE_GOLD_KEYS.get(e.kind, ()) if k not in gold],
            "gold_edge": verdict,
            "gold_workspace": workspace if gold else None,
        }
        item["tier_defects"] = _edge_tier_defects(item, report)
        items.append(item)
    c["items_edge"] = len(items)
    _missing_edges(graph, labels, workspaces, report)
    return items


def _edge_tier_defects(item: dict, report: Report) -> int:
    """Every established feature (the edge itself included) and every tiered skeleton
    value of both ends whose tier is absent or outside `schema.TIERS`, counted."""
    defects = 0
    for name, feat in sorted(item["features"].items()):
        if feat.get("value") is not None and feat.get("tier") not in TIERS:
            defects += 1
            report.warn("tier_defects", f"edge {item['id']}: feature {name} has tier {feat.get('tier')!r}, not one of {', '.join(TIERS)}")
        if name in ("src", "dst") and feat.get("value") is not None:
            sk = feat["value"]
            for key, tier_key in (("role", "role_tier"), ("phase", "phase_tier"), ("model", "model_tier")):
                if sk.get(key) is not None and sk.get(tier_key) not in TIERS:
                    defects += 1
                    report.warn("tier_defects", f"edge {item['id']}: {name} skeleton {key} has tier {sk.get(tier_key)!r}, not one of {', '.join(TIERS)}")
    return defects


# --- E2 verdicts and contract-v3 attempts (D1b) ------------------------------------------


def _by_id(items: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for it in items:
        if it["item"] == "node":
            out.setdefault(it["id"], []).append(it)
    return out


def _add_gold(item: dict, key: str, label: dict, where: str, report: Report) -> None:
    """Attach `label` as gold `key` on a node item; a second label for the key is a
    conflict resolved and counted the way the loader resolves duplicated keys."""
    if key in item["gold"]:
        item["gold"][key] = _pick([item["gold"][key], label], where, report)
    else:
        item["gold"][key] = label
    if key in item["gold_missing"]:
        item["gold_missing"].remove(key)


def _count_ws(report: Report, ws: str | None, key: str, n: int = 1) -> None:
    """One counter twice: globally and qualified by workspace (`ws:<ws>:<key>`), so the
    per-workspace summary carries every count the total does, under the same name."""
    report.counters[key] += n
    report.counters[f"ws:{ws or '(none)'}:{key}"] += n


def _warn_ws(report: Report, ws: str | None, key: str, text: str) -> None:
    _count_ws(report, ws, key)
    report.warnings.append(text)


def attach_e2(items: list[dict], labels: Labels, report: Report) -> None:
    """The verdict of every E2 run onto its dev session's node item (there is no referee
    session to carry it: spec section 8), with the tier the weaker of the verdict's
    and the dev-session mapping's (a heuristic mapping cannot make a verified verdict a
    verified fact about the node; both original tiers are kept beside it), the run's
    roles onto the named sessions; every session absent from the graph under the run's
    workspace counted and printed. Every counter is also kept per workspace."""
    by_id = _by_id(items)
    for run in sorted(labels.e2_arms, key=lambda a: (str(a.get("workspace")), str(a.get("task")), str(a.get("arm")))):
        ws, task, variant = run.get("workspace"), run.get("task"), run.get("arm")
        wsn = ws if isinstance(ws, str) and ws else "(none)"
        name = f"E2 run {task}/{variant} (workspace {ws})"
        _count_ws(report, wsn, "e2_runs")
        dev = run.get("dev_session")
        if "referee_session" in run:
            _count_ws(report, wsn, "e2_referee_sessions_named")
        if dev is None:
            _warn_ws(report, wsn, "e2_runs_without_dev_session", f"{name}: no dev session label; its verdict has nowhere to attach")
        else:
            targets = [it for it in by_id.get(dev["value"], []) if it["workspace"] == ws]
            elsewhere = sorted({it["workspace"] or "(none)" for it in by_id.get(dev["value"], []) if it["workspace"] != ws})
            if not targets:
                _warn_ws(report, wsn, "e2_dev_sessions_absent", f"{name}: dev session {dev['value']} is not in the graph under that workspace" + (f" (it is under {', '.join(elsewhere)})" if elsewhere else ""))
            elif "verdict" not in run:
                _warn_ws(report, wsn, "e2_runs_without_verdict", f"{name}: the dev session is in the graph but the run carries no verdict")
            else:
                verdict = run["verdict"]
                tier = _weakest(verdict["tier"], dev["tier"])
                for it in targets:
                    if "e2" in it:
                        _warn_ws(report, wsn, "e2_dev_sessions_shared", f"{name}: dev session {dev['value']} is also the dev session of E2 run {it['e2']['task']}/{it['e2']['arm']}")
                    it["e2"] = {"task": task, "arm": variant, "workspace": ws, "dev_session_tier": dev["tier"], "dev_session_source": dev["source"]}
                    label = {"value": verdict["value"], "tier": tier, "source": f"{verdict['source']} ({name}; dev session matched at tier {dev['tier']}: {dev['source']})", "verdict_tier": verdict["tier"], "dev_session_tier": dev["tier"]}
                    _add_gold(it, "verdict", label, f"{name}: verdict on {dev['value']}", report)
                    _count_ws(report, wsn, "e2_verdicts_attached")
                    if tier != verdict["tier"]:
                        _warn_ws(report, wsn, "e2_verdicts_weakened", f"{name}: verdict tier {verdict['tier']} lowered to {tier} because the dev session {dev['value']} was matched at tier {dev['tier']}")
        for rid in sorted(run.get("roles", {})):
            targets = [it for it in by_id.get(rid, []) if it["workspace"] == ws]
            if not targets:
                _warn_ws(report, wsn, "e2_role_sessions_absent", f"{name}: session {rid} labeled {run['roles'][rid]['value']} is not in the graph under that workspace")
                continue
            for it in targets:
                _add_gold(it, "role", run["roles"][rid], f"{name}: role of {rid}", report)
                _count_ws(report, wsn, "e2_roles_attached")


def contract_workspace(labels: Labels) -> str | None:
    """The workspace the contract-v3 attempts belong to: the first path component of the
    file's `source` (the ledger's own workspace), None when the file does not say."""
    src = (labels.contract_v3 or {}).get("source")
    if isinstance(src, str) and src.strip("/"):
        return src.strip("/").split("/")[0]
    return None


def attach_contract(items: list[dict], labels: Labels, report: Report) -> None:
    """Every contract-v3 attempt onto its session's node item under the ledger's own
    workspace (session ids are workspace-scoped: a node with the same id under another
    workspace is counted, never labeled): the attempt under `attempts`, its send_back and
    approved labels as gold with the weaker of the session match's tier and the label's
    own tier (the original tiers kept beside it). Attempts on one session that disagree
    on a key leave that key off the gold and are counted. Every counter is also kept per
    workspace (the ledger's)."""
    by_id = _by_id(items)
    attempts = (labels.contract_v3 or {}).get("attempts", [])
    src_ws = contract_workspace(labels)
    wsn = src_ws or "(none)"
    per_item: dict[int, dict[str, list[dict]]] = {}
    for att in attempts:
        _count_ws(report, wsn, "attempts")
        name = f"contract-v3 attempt {att.get('task')} n{att.get('n')}"
        session = att.get("session")
        if session is None:
            _warn_ws(report, wsn, "attempts_without_session", f"{name}: no session label; its labels have nowhere to attach")
            continue
        if src_ws is None:
            _warn_ws(report, wsn, "attempts_source_workspace_unknown", f"{name}: contract-v3.json names no source workspace (source {(labels.contract_v3 or {}).get('source')!r}), so session {session['value']} cannot be looked up under one; not attached")
            continue
        all_targets = by_id.get(session["value"], [])
        targets = [it for it in all_targets if it["workspace"] == src_ws]
        elsewhere = sorted({it["workspace"] or "(none)" for it in all_targets if it["workspace"] != src_ws})
        if elsewhere:
            _count_ws(report, wsn, "attempts_session_in_other_workspace")
        if not targets:
            _warn_ws(report, wsn, "attempts_session_absent", f"{name}: session {session['value']} is not in the graph under workspace {src_ws}" + (f" (a node with that id is under {', '.join(elsewhere)}; it gets no label from the attempt)" if elsewhere else ""))
            continue
        if elsewhere:
            report.warnings.append(f"{name}: session {session['value']} is also in the graph under workspace(s) {', '.join(elsewhere)}, not the ledger's {src_ws}; those nodes get no label from the attempt")
        entry = {k: att.get(k) for k in ("task", "n", "actor", "model", "started_at", "ended_at", "result", "cause")}
        entry["session_tier"], entry["session_source"] = session["tier"], session["source"]
        entry["labels"] = {k: dict(v) for k, v in att["labels"].items()}
        for it in targets:
            it.setdefault("attempts", []).append(entry)
            for key in ATTEMPT_LABEL_KEYS:
                if key not in att["labels"]:
                    _count_ws(report, wsn, f"attempt_labels_missing:{key}")
                    continue
                lab = att["labels"][key]
                merged = {"value": lab["value"], "tier": _weakest(lab["tier"], session["tier"]), "source": f"{lab['source']} ({name}; session matched at tier {session['tier']}: {session['source']})", "label_tier": lab["tier"], "session_tier": session["tier"]}
                per_item.setdefault(id(it), {}).setdefault(key, []).append(merged)
        _count_ws(report, wsn, "attempts_attached")
    for it in items:
        cands = per_item.get(id(it))
        if not cands:
            continue
        for key in ATTEMPT_LABEL_KEYS:
            labs = cands.get(key, [])
            if not labs:
                continue
            values = {lab["value"] for lab in labs}
            if len(values) > 1:
                _warn_ws(report, wsn, f"attempt_labels_disagree:{key}", f"node {it['id']}: {len(labs)} attempts disagree on {key} ({', '.join(str(v) for v in sorted(values))}); no gold value for it, every attempt is listed on the item")
                continue
            if len(labs) > 1:
                _count_ws(report, wsn, f"attempt_labels_merged:{key}")
            _add_gold(it, key, sorted(labs, key=_rank)[0], f"node {it['id']}: {key}", report)


# --- the whole dataset and the workspace resolution (D1b) --------------------------------
