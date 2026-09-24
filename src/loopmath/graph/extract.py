"""Deterministic extraction of the workflow graph from parsed sessions.

Input is the list of priced run records `ingest.parse_all` -> `grade_all` ->
`price_all` produces (plain dicts, one per session). Output is a `Graph`. No
model calls; every relation comes from a structural fact in the logs, and the
modules below are named after them (EXTRACTOR-SPEC.md section 2):

1. `scan.py`: one pass over each transcript (Task calls, Bash commands, file
   writes and reads); `bashwrites.py` (A1) adds Bash-written files, `codexio.py`
   (A2) the codex side, `cache.py` (L2) the link cache.
2. Spawn edges (here): a subagent transcript sits at
   `<project>/<sid>/subagents/agent-<hash>.jsonl` next to a `.meta.json` whose
   `toolUseId` is the id of the parent's `Task` block. On the three 08-31
   swarms this matched 38/38 (A) and 27/27 (B) subagents, so the edge is
   tier `verified`; when the meta file is missing the parent falls back to
   the enclosing session by path, tier `heuristic`.
3. Launch edges: `launch.py` (L1, L2).
4. Artifact edges: `artifacts.py` (A3).

Roles and phases: `labels.py`. This module is composition only; after X0 the
orchestrator edits it for glue, nobody else.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .artifacts import build_artifacts
from .cache import link_cache_get, link_cache_put
from .codexio import scan_codex_session
from .labels import label_roles_and_phases
from .launch import join_launches
from .scan import cache_key, epoch, read_meta, scan_claude_session
from .schema import Graph, GraphEdge, GraphNode
from .token_completeness import missing_token_streams


def node_from_record(r: dict) -> GraphNode:
    sp = str(r.get("session_path") or "")
    harness = str(r.get("harness") or "")
    if harness == "codex":
        source, model_tier = "codex", "reported"
    elif "/subagents/" in sp:
        source, model_tier = "subagent", "verified"
    else:
        source, model_tier = "top", "verified"
    usd = r.get("usd")
    return GraphNode(
        id=str(r["run_id"]),
        harness=harness,
        source=source,
        session_path=sp,
        model=r.get("model"),
        model_tier=model_tier if r.get("model") else None,
        effort=r.get("effort"),
        workspace=r.get("workspace"),
        ts=r.get("ts"),
        wall_s=float(r["wall_s"]) if isinstance(r.get("wall_s"), (int, float)) else None,
        tokens=dict(r["tokens"]) if isinstance(r.get("tokens"), dict) else None,
        usd=float(usd) if isinstance(usd, (int, float)) else None,
    )


def _scan_codex(path: Path) -> dict:
    """A codex node's scan, shaped like a Claude scan (no tasks, no bash) plus
    `origin`, through the link cache."""
    key = cache_key(path)
    hit = link_cache_get(("codex", *key)) if key is not None else None
    if hit is None:
        io = scan_codex_session(path)
        hit = {
            "tasks": {},
            "bash": [],
            "writes": list(io.get("writes") or []),
            "reads": list(io.get("reads") or []),
            "artifact_facts": list(io.get("artifact_facts") or []),
            "origin": dict(io.get("origin") or {}),
            "meta": dict(io.get("meta") or {}),
        }
        if key is not None:
            link_cache_put(("codex", *key), hit)
    return hit


def _join_spawns(nodes: dict[str, GraphNode], scans: dict[str, dict], meta: dict) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    by_path = {n.session_path: nid for nid, n in nodes.items()}
    task_index: dict[str, tuple[str, dict]] = {}
    for nid in nodes:
        for tid, t in scans[nid]["tasks"].items():
            task_index[tid] = (nid, t)
    for nid, n in nodes.items():
        if n.source != "subagent":
            continue
        sp = Path(n.session_path)
        m = read_meta(sp) or {}
        tid = m.get("toolUseId")
        hit = task_index.get(tid) if tid else None
        base_spawn = {
            "tool_use_id": tid,
            "agent_type": m.get("agentType"),
            "spawn_depth": m.get("spawnDepth"),
            "meta_description": m.get("description"),
        }
        if hit:
            pnid, t = hit
            n.parent = pnid
            n.spawn = {**base_spawn, "description": t.get("description"), "subagent_type": t.get("subagent_type"), "requested_model": t.get("requested_model")}
            edges.append(GraphEdge(pnid, nid, "spawn", "verified", {"tool_use_id": tid, "description": t.get("description")}))
            continue
        enclosing = by_path.get(str(sp.parent.parent) + ".jsonl")
        if enclosing:
            n.parent = enclosing
            n.spawn = {**base_spawn, "description": m.get("description")}
            edges.append(GraphEdge(enclosing, nid, "spawn", "heuristic", {"tool_use_id": tid, "reason": "path containment; no matching Task tool_use"}))
        else:
            n.spawn = base_spawn
            meta["unlinked_subagents"] += 1
    return edges


def _attribute_pending_writes(scans: dict[str, dict], edges: list[GraphEdge], launch_cmd: dict[str, str], meta: dict) -> None:
    """A1 emits `codex exec ... -o f` as a write by the calling session flagged
    `pending_producer`; the producer is the codex session that call launched (spec
    section 3, A2). Move the write to that session when the launch join names
    exactly one codex session whose launch command contains the path; otherwise
    the write stays with the caller, still flagged, and is counted."""
    launched: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e.kind == "launch":
            launched[e.src].append(e.dst)
    moved = unresolved = 0
    for nid in sorted(scans):
        keep = []
        for w in scans[nid]["writes"]:
            if not w.get("pending_producer"):
                keep.append(w)
                continue
            needles = [n for n in (w["path"], w.get("raw")) if isinstance(n, str) and n]
            cands = sorted(d for d in launched.get(nid, []) if d in scans and any(n in launch_cmd.get(d, "") for n in needles))
            if len(cands) == 1:
                scans[cands[0]]["writes"].append({**w, "pending_producer": False, "tier": "heuristic", "how": w.get("how", "codex -o") + " via launch join", "launched_by": nid})
                moved += 1
            else:
                keep.append(w)
                unresolved += 1
        scans[nid]["writes"] = keep
    meta["pending_writes_attributed"] = moved
    meta["pending_writes_unresolved"] = unresolved


def extract(records: list[dict], *, workspaces: list[str] | None = None) -> Graph:
    """Build the graph for `records` (optionally restricted to `workspaces`)."""
    wanted = set(workspaces) if workspaces else None
    nodes: dict[str, GraphNode] = {}
    # Records without a run_id or a session file cannot become nodes; they are
    # counted, never dropped silently (spec section 0, rule 1). Records outside
    # the requested workspaces are out of scope, not excluded, and are not counted.
    skipped_no_id_or_path = 0
    for r in records:
        if not r.get("run_id") or not r.get("session_path"):
            if wanted is None or r.get("workspace") in wanted:
                skipped_no_id_or_path += 1
            continue
        if wanted is not None and r.get("workspace") not in wanted:
            continue
        nodes[str(r["run_id"])] = node_from_record(r)

    meta: dict = {"unlinked_subagents": 0, "unmatched_launches": 0, "unlaunched_codex": 0, "records_skipped_no_id_or_path": skipped_no_id_or_path}
    scans: dict[str, dict] = {}
    for nid, n in nodes.items():
        s = _scan_codex(Path(n.session_path)) if n.source == "codex" else scan_claude_session(Path(n.session_path))
        # Own copies of the lists this function mutates, so a cache hit (L2) is never altered.
        scans[nid] = {**s, "writes": list(s["writes"]), "reads": list(s["reads"])}
        # Bash commands A1 inspected but could not turn into a path, by reason (never a fake path).
        for reason, count in (s.get("excluded") or {}).items():
            meta[f"bashwrites_excluded_{reason}"] = meta.get(f"bashwrites_excluded_{reason}", 0) + count
        # Codex-side counters A2 returns in its scan's `meta` (failed calls, missing session_meta, ...).
        for key, count in (s.get("meta") or {}).items():
            if isinstance(count, int) and not isinstance(count, bool):
                meta[f"codex_{key}"] = meta.get(f"codex_{key}", 0) + count

    edges: list[GraphEdge] = _join_spawns(nodes, scans, meta)
    launch_cmd, external, launch_edges = join_launches(nodes, scans, records, wanted, meta=meta)
    edges.extend(launch_edges)
    # Unknown durations and token counts stay None on the node and are counted here,
    # after the launch join so the external launcher nodes it adds are included.
    meta["nodes_missing_wall_s"] = sum(1 for n in nodes.values() if n.wall_s is None)
    meta["nodes_missing_tokens"] = sum(
        bool(missing_token_streams(node.tokens)) for node in nodes.values()
    )
    _attribute_pending_writes(scans, edges, launch_cmd, meta)
    artifacts, art_edges = build_artifacts(scans, nodes, meta=meta)  # A3: git fallback and untimed-event counters land in meta
    edges.extend(art_edges)
    # Reads of paths no node ever wrote cannot become artifact edges; counted, not dropped.
    art_paths = {a.id for a in artifacts}
    meta["reads_without_writes"] = sum(1 for s in scans.values() for rd in s["reads"] if rd.get("path") not in art_paths)
    # Events whose timestamp does not parse never reach the artifact join; reads that precede
    # the path's first observed write have no producer to join to. Both are counted here.
    first_write = {a.id: epoch(a.first_write_ts) for a in artifacts}
    meta["events_invalid_ts"] = sum(1 for s in scans.values() for ev in s["writes"] + s["reads"] if epoch(ev.get("ts")) is None)
    meta["reads_before_first_write"] = sum(1 for s in scans.values() for rd in s["reads"] if rd.get("path") in first_write and epoch(rd.get("ts")) is not None and first_write[rd["path"]] is not None and epoch(rd["ts"]) < first_write[rd["path"]])
    label_roles_and_phases(nodes, edges, launch_cmd)

    g = Graph(nodes=list(nodes.values()), edges=edges, artifacts=artifacts, meta=meta)
    meta.update(summarize(g))
    return g


def summarize(g: Graph) -> dict:
    nodes_by_source: dict[str, int] = defaultdict(int)
    roles: dict[str, int] = defaultdict(int)
    phases: dict[str, int] = defaultdict(int)
    for n in g.nodes:
        nodes_by_source[n.source] += 1
        roles[n.role or "unlabeled"] += 1
        phases[n.phase or "unknown"] += 1
    edges_by_kind: dict[str, int] = defaultdict(int)
    for e in g.edges:
        edges_by_kind[f"{e.kind}/{e.tier}"] += 1
    return {
        "n_nodes": len(g.nodes),
        "nodes_by_source": dict(nodes_by_source),
        "roles": dict(roles),
        "phases": dict(phases),
        "edges_by_kind_tier": dict(edges_by_kind),
        "n_artifacts": len(g.artifacts),
        "n_artifacts_consumed": sum(1 for a in g.artifacts if a.consumers),
        "usd_total": round(sum(n.usd or 0.0 for n in g.nodes), 2),
        "usd_unpriced_nodes": sum(1 for n in g.nodes if n.usd is None),
    }
