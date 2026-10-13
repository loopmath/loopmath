"""OCP document emission from a workflow graph."""

from __future__ import annotations

import copy

from .ocp_support import (
    Artifact, COUNTERS, EXT_KEY, GraphEdge, OCP_VERSION, TIERS, _NODE_KIND, _Text, _artifact_identity, _cost, _edge_evidence, _edge_record, _fmt_ts, _label, _node_title, _origin, _parse_ts, _phase, _privacy, _producer, _role, attempt_id, datetime, timedelta, Graph,
)

_ARTIFACT_EXT_KEY = "dev.loopmath.artifact"
_CORE_CAPABILITIES = {
    "groups": True, "events": True, "artifacts": True, "edges_dep": True,
    "edges_spawn": True, "edges_launch": True, "edges_artifact": True,
    "cost_usd": True, "cost_tokens": True, "outcome_evidence": True,
}


def _artifact_recording(art: Artifact) -> dict:
    """The complete Q7 artifact extension, including explicit unknowns."""
    return {
        "bytes": art.bytes,
        "bytes_tier": art.bytes_tier,
        "lines_added": art.lines_added,
        "lines_added_tier": art.lines_added_tier,
        "lines_removed": art.lines_removed,
        "lines_removed_tier": art.lines_removed_tier,
        "language": art.language,
        "language_tier": art.language_tier,
        "tests_touched": art.tests_touched,
        "tests_touched_tier": art.tests_touched_tier,
        "fate": art.fate,
        "fate_tier": art.fate_tier,
        "meta": copy.deepcopy(art.meta),
    }

def to_ocp(graph: Graph, *, producer=None, privacy="metadata_only") -> dict:
    """One OCP v0.3 document for `graph`. `producer` is a name, a dict merged
    over the extractor's defaults, or None; `privacy` is a profile name or a
    privacy record. Under `metadata_only` no transcript text leaves the graph:
    titles stay synthesized, role evidence is reduced to rule names, and spawn
    descriptions and launch commands appear only under `full`. Boolean producer
    capability extensions survive; core capability values are emitter-owned."""
    prod = _producer(producer)
    supplied_capabilities = prod.get("capabilities", {})
    if not isinstance(supplied_capabilities, dict):
        raise TypeError("producer capabilities must be a dict of boolean declarations")
    for name, value in supplied_capabilities.items():
        if not isinstance(value, bool):
            raise ValueError(
                f"producer capability {name!r} must be boolean, got {value!r}"
            )
    extensions = {
        name: value
        for name, value in supplied_capabilities.items()
        if name not in _CORE_CAPABILITIES
    }
    prod["capabilities"] = {**_CORE_CAPABILITIES, **extensions}
    priv = _privacy(privacy)
    full = priv["profile"] == "full"
    counters: dict = {k: 0 for k in COUNTERS}
    t = _Text(counters)

    nodes = sorted(graph.nodes, key=lambda n: n.id)
    by_id = {n.id: n for n in nodes}
    node_ids = set(by_id)
    # Edges that will be emitted, by target: the origin of a node copies its
    # launch edge's tier exactly, and an edge held back cannot back an origin.
    spawn_tier: dict[str, str] = {}
    launch_in: dict[str, tuple[str, str]] = {}
    for e in sorted(graph.edges, key=lambda e: (e.src, e.dst)):
        if e.src not in node_ids or e.dst not in node_ids or e.tier not in TIERS:
            continue
        if e.kind == "spawn":
            spawn_tier.setdefault(e.dst, e.tier)
        elif e.kind == "launch":
            lb = by_id[e.dst].launched_by
            wanted = str(lb.get("id")) if isinstance(lb, dict) and lb.get("id") else None
            if e.dst not in launch_in or e.src == wanted:
                launch_in[e.dst] = (e.src, e.tier)
    raw_spawn_tier = {e.dst: e.tier for e in graph.edges if e.kind == "spawn"}

    # Groups cover every workspace a node sits in; the run is named after the ones in
    # scope (external launchers sit in their own workspace but are not the run).
    groups = [{"id": ws, "title": t(f"workspace {ws}")} for ws in sorted({n.workspace for n in nodes if n.workspace})]
    workspaces = sorted({n.workspace for n in nodes if n.workspace and n.source != "external"})

    ocp_nodes: list[dict] = []
    attempts: list[dict] = []
    events: list[tuple[str, int, str, dict]] = []
    starts: list[datetime] = []
    ends: list[datetime] = []
    for n in nodes:
        kind = _NODE_KIND.get(n.role or "", "unknown")
        title, node_ext = _node_title(n, full, t)
        preserved_ext = graph._ocp_node_ext.get(n.id)
        if isinstance(preserved_ext, dict):
            # The reader carries only this emitter's own namespace.  Generated
            # values win if the graph has since changed; otherwise provenance
            # such as a metadata-only withheld-title note survives verbatim.
            preserved_ext = copy.deepcopy(preserved_ext)
            preserved_ext.pop("title_withheld" if full else "title_source", None)
            if "title_withheld" in preserved_ext and "title_withheld" not in node_ext:
                counters["titles_withheld_metadata_only"] += 1
            node_ext = {**preserved_ext, **node_ext}
        node: dict = {"id": n.id, "kind": kind, "title": title, "state": "settled_unverified"}
        if n.workspace:
            node["group"] = n.workspace
        node["labels"] = {"harness": n.harness, "source": n.source}
        if node_ext:
            node["ext"] = {EXT_KEY: node_ext}
        ocp_nodes.append(node)

        a: dict = {"id": attempt_id(n.id), "node": n.id, "n": 1, "actor": n.role or n.source, "harness": n.harness}
        ext: dict = {}
        if n.model:
            a["model"] = {"raw": t(n.model)}
            if n.model_tier in TIERS:
                a["model"]["tier"] = n.model_tier
            else:
                counters["model_tiers_omitted_invalid"] += 1
                ext["model_tier_omitted"] = f"graph gave model tier {n.model_tier!r}, not one of {'/'.join(TIERS)}; the label is kept, its tier is unknown"
        if n.effort:
            a["effort"] = t(n.effort)
        a["cause"] = {"type": "initial"}
        a["origin"] = _origin(n, launch_in.get(n.id), spawn_tier.get(n.id, raw_spawn_tier.get(n.id)), t)
        a["role"] = _role(n, full, t)
        a["phase"] = _phase(n, t)
        a["status"] = "settled_unverified"
        st = _parse_ts(n.ts)
        if st is None:
            counters["attempts_without_start_ts"] += 1
            ext["started_at_missing"] = "no start time in the session record" if n.ts is None else t(f"start time {str(n.ts)[:60]!r} does not parse as ISO 8601")
        else:
            a["started_at"] = _fmt_ts(st)
            starts.append(st)
            events.append((a["started_at"], 0, a["id"], {"type": "attempt_started", "attempt": a["id"], "node": n.id}))
            if n.wall_s is None:
                counters["attempts_missing_wall_s"] += 1
                ext["ended_at_missing"] = "no wall-clock duration in the session record"
            else:
                en = st + timedelta(seconds=float(n.wall_s))
                a["ended_at"] = _fmt_ts(en)
                ends.append(en)
                events.append((a["ended_at"], 1, a["id"], {"type": "attempt_settled", "attempt": a["id"], "node": n.id}))
        a["outcome"] = {"result": "settled_unverified", "evidence": "heuristic", "reason": "session file parsed to its end; the graph carries no acceptance signal"}
        cost, missing = _cost(n, counters)
        if cost is not None:
            a["cost"] = cost
        if missing:
            ext["cost_unknown"] = {k: None for k in missing}
            ext["cost_missing"] = {k: t(v) for k, v in missing.items()}
        a["session"] = t(n.session_path.rsplit("/", 1)[-1]) if n.session_path else n.id
        if full:
            if n.spawn:
                ext["spawn"] = dict(n.spawn)
            if isinstance(n.launched_by, dict) and n.launched_by.get("command"):
                ext["launch_command"] = n.launched_by["command"]
        if ext:
            a["ext"] = {EXT_KEY: dict(sorted(ext.items()))}
        attempts.append(a)

    art_by_path: dict[str, tuple[str, Artifact]] = {}
    artifacts: list[dict] = []
    for art in sorted(graph.artifacts, key=lambda a: a.id):
        aid, shown, art_ext = _artifact_identity(art.id, counters)
        art_by_path[art.id] = (aid, art)
        fw = _parse_ts(art.first_write_ts)
        if fw is None and art.first_write_ts is not None:
            counters["artifacts_first_write_ts_unparseable"] += 1
        kind_value, kind_tier, note = _label(art.kind, art.kind_tier, "artifact kind", counters)
        kind_ev = note if note else ("path pattern" if kind_value else "path pattern; no rule matched")
        rec = {
            "id": aid,
            "path": shown,
            "kind": {"value": t(kind_value) if kind_value else None, "tier": kind_tier, "evidence": t(kind_ev)},
            "producer": attempt_id(art.producer),
            "writers": [attempt_id(w) for w in art.writers],
            "consumers": [attempt_id(c) for c in art.consumers],
            "first_write_at": _fmt_ts(fw) if fw is not None else None,
            "n_writes": int(art.n_writes) if art.n_writes else None,
            "n_reads": int(art.n_reads) if isinstance(art.n_reads, int) else None,
        }
        rec["ext"] = {_ARTIFACT_EXT_KEY: _artifact_recording(art)}
        if art_ext:
            rec["ext"][EXT_KEY] = art_ext
        artifacts.append(rec)

    edges: list[dict] = []
    not_emitted: list[dict] = []

    def _hold(e: GraphEdge, counter: str, reason: str) -> None:
        counters[counter] += 1
        not_emitted.append({**_edge_record(e, full, t), "reason": reason})

    for e in sorted(graph.edges, key=lambda e: (e.kind, e.src, e.dst, str((e.detail or {}).get("path") or ""))):
        if e.src not in node_ids or e.dst not in node_ids:
            _hold(e, "edges_not_emitted_unknown_endpoint", "an endpoint is not a node of the graph")
            continue
        if e.tier not in TIERS:
            _hold(e, "edges_not_emitted_invalid_tier", t(f"tier {e.tier!r} is not one of {'/'.join(TIERS)}; the emitter does not pick one"))
            continue
        rec: dict = {"from": e.src, "to": e.dst, "kind": e.kind, "tier": e.tier, "from_attempt": attempt_id(e.src), "to_attempt": attempt_id(e.dst), "evidence": _edge_evidence(e, t)}
        if e.kind == "artifact":
            path = str((e.detail or {}).get("path") or "")
            hit = art_by_path.get(path)
            if hit is None:
                _hold(e, "edges_not_emitted_unknown_artifact", "the edge's path is not an artifact of the graph")
                continue
            aid, art = hit
            rec["artifact"] = aid
            if e.src not in art.writers:
                _hold(e, "edges_not_emitted_writer_not_listed", "the edge's writer is not among the artifact's writers")
                continue
            if e.dst not in art.consumers:
                _hold(e, "edges_not_emitted_reader_not_consumer", "the edge's reader is not among the artifact's consumers")
                continue
            if art.producer != e.src:
                # The graph joins a read to the latest writer before it; OCP's artifact
                # edge runs from the producer (first writer). Same tier, later writer
                # named, rewrite counted.
                counters["artifact_edges_rerouted_from_later_writer"] += 1
                rec["from"] = art.producer
                rec["from_attempt"] = attempt_id(art.producer)
                rec["evidence"] = t(f"read joined to later writer {attempt_id(e.src)} in the graph ({rec['evidence']}); OCP runs the artifact edge from the producer {attempt_id(art.producer)}")
                rec["ext"] = {EXT_KEY: {"graph_from": e.src, "graph_from_attempt": attempt_id(e.src), "rerouted": "later writer to producer"}}
        elif full and e.kind == "launch" and (e.detail or {}).get("command"):
            rec["ext"] = {EXT_KEY: {"command": e.detail["command"]}}
        edges.append(rec)

    events.sort(key=lambda tt: (tt[0], tt[1], tt[2]))
    run: dict = {"id": "loopmath-graph:" + ("+".join(workspaces) if workspaces else "no-workspace"), "title": t("workflow graph of " + (", ".join(workspaces) if workspaces else "sessions without a workspace")), "workspace": "+".join(workspaces)[:500]}
    if starts:
        run["started_at"] = _fmt_ts(min(starts))
    if ends:
        run["ended_at"] = _fmt_ts(max(ends))

    harnesses = sorted({n.harness for n in nodes if n.harness})
    prod.setdefault("framework", "+".join(harnesses) if harnesses else "unknown")

    emitted_meta = graph.meta
    if (
        graph._ocp_loaded_meta is not None
        and graph.meta == graph._ocp_loaded_meta
        and graph._ocp_source_meta is not None
    ):
        emitted_meta = copy.deepcopy(graph.meta)
        for key in graph._ocp_source_meta["absent"]:
            emitted_meta.pop(key, None)
        emitted_meta.update(graph._ocp_source_meta["values"])
    doc = {
        "ocp": OCP_VERSION,
        "producer": prod,
        "privacy": priv,
        "run": run,
        "groups": groups,
        "nodes": ocp_nodes,
        "edges": edges,
        "attempts": attempts,
        "artifacts": artifacts,
        "events": [dict(at=at, **ev) for at, _, _, ev in events],
        "ext": {EXT_KEY: {"meta": dict(sorted(emitted_meta.items())), "emitter": dict(sorted(counters.items())), "edges_not_emitted": not_emitted}},
    }
    return doc
