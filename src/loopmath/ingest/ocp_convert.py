"""Convert OCP v0.2 documents into loopmath graphs and analysis rows."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path

from .base import file_kind
from .ocp_artifact import _artifact_extension
from .ocp_common import (
    OCPError,
    _OUTCOME_TIERS,
    _REROUTED_RE,
    _TERMINAL_FALSE,
    _TERMINAL_TRUE,
    _attempt_graph_ids,
    _duration,
    _ext,
    _lag,
    _meta_snapshots,
    _objects,
    _tokens,
    _unique,
    _validate_v02,
    read_document,
)
from .ocp_projection import _extension_loss_counters, _projection_key
from ..graph.extract import summarize
from ..graph.schema import Artifact, Graph, GraphEdge, GraphNode
from ..graph.token_completeness import missing_token_streams
from ..ocp.version import version_at_least


def _source_scoped_extension_projection(document: dict, projected: dict) -> dict:
    """Exclude derived meta additions while comparing supplied source members."""
    source_meta = _ext(document).get("meta")
    projected_meta = _ext(projected).get("meta")
    if not isinstance(source_meta, dict) or not isinstance(projected_meta, dict):
        return projected
    scoped = copy.deepcopy(projected)
    scoped_meta = _ext(scoped)["meta"]
    for key in tuple(scoped_meta):
        if key not in source_meta:
            del scoped_meta[key]
    return scoped


def _projection_loss_counters(document: dict, graph: Graph) -> dict[str, int]:
    """Count input entities and members absent from a fresh Graph projection."""
    from ..graph.ocp import to_ocp

    projected = to_ocp(graph)
    counters: Counter = Counter()
    missing = object()
    for family in ("producer", "privacy", "run"):
        before = document.get(family)
        after = projected.get(family)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        counters[f"ocp_projection_{family}_members_unmodeled"] += sum(
            key != "ext" and after.get(key, missing) != value
            for key, value in before.items()
        )

    for collection, family in (
        ("groups", "group"),
        ("nodes", "node"),
        ("edges", "edge"),
        ("attempts", "attempt"),
        ("artifacts", "artifact"),
        ("events", "event"),
    ):
        available: dict[object, list[dict]] = defaultdict(list)
        for record in projected.get(collection, []):
            if isinstance(record, dict):
                available[_projection_key(collection, record)].append(record)
        for record in document.get(collection, []):
            if not isinstance(record, dict):
                continue
            candidates = available.get(_projection_key(collection, record), [])
            if not candidates:
                counters[f"ocp_projection_{collection}_unmodeled"] += 1
                continue
            exact = next(
                (index for index, candidate in enumerate(candidates) if candidate == record),
                0,
            )
            matched = candidates.pop(exact)
            counters[f"ocp_projection_{family}_members_unmodeled"] += sum(
                key != "ext" and matched.get(key, missing) != value
                for key, value in record.items()
            )

    known_root = {
        "ocp", "producer", "privacy", "run", "groups", "nodes", "edges",
        "attempts", "artifacts", "events", "ext",
    }
    counters["ocp_projection_root_members_unmodeled"] += sum(
        key not in known_root and projected.get(key, missing) != value
        for key, value in document.items()
    )

    extension_projection = projected
    privacy = document.get("privacy")
    if isinstance(privacy, dict) and privacy.get("profile") == "full":
        extension_projection = to_ocp(graph, privacy=privacy)
    extension_projection = _source_scoped_extension_projection(
        document, extension_projection
    )
    counters.update(_extension_loss_counters(document, extension_projection))
    return {key: value for key, value in counters.items() if value}


def _token_completeness_counters(nodes: list[GraphNode]) -> dict[str, int]:
    counters: Counter = Counter()
    for node in nodes:
        missing = missing_token_streams(node.tokens)
        if not isinstance(node.tokens, dict):
            counters["ocp_nodes_without_cost"] += 1
        elif missing:
            counters["ocp_nodes_partial_tokens"] += 1
        if missing:
            counters["nodes_missing_tokens"] += 1
            counters["ocp_token_streams_missing"] += len(missing)
            for ocp_name in missing:
                counters[f"ocp_missing_{ocp_name}"] += 1
    counters.setdefault("nodes_missing_tokens", 0)
    return dict(counters)


def _settle_projection_loss_counters(
    document: dict, graph: Graph, source_meta: dict
) -> None:
    """Apply projection counters after every derived metadata value is visible.

    The projection counters are themselves emitted metadata.  Re-snapshotting
    before each pass lets the closed compatibility policy hide only its named
    legacy delta while a later overwrite of any other source member remains
    visible to extension-loss accounting.  One pass derives ordinary losses,
    a second observes any overwritten source projection counter, and the third
    confirms the result is stable.
    """
    missing = object()
    for _ in range(3):
        graph._ocp_source_meta, graph._ocp_loaded_meta = _meta_snapshots(
            source_meta, graph.meta
        )
        counters = _projection_loss_counters(document, graph)
        if all(
            graph.meta.get(key, missing) == value
            for key, value in counters.items()
        ):
            return
        graph.meta.update(counters)
    raise OCPError("projection loss counters did not stabilize")


def _graph_node(
    graph_id: str,
    logical: dict,
    attempt: dict | None,
    *,
    incoming_kinds: set[str],
    attempt_to_graph: dict[str, str],
    launch_edges: list[dict],
    run_workspace: str | None,
    producer_framework: str | None,
) -> GraphNode:
    attempt = attempt or {}
    labels = logical.get("labels") if isinstance(logical.get("labels"), dict) else {}
    harness = attempt.get("harness") or labels.get("harness") or producer_framework or "unknown"
    source = labels.get("source")
    if not isinstance(source, str) or not source:
        if "spawn" in incoming_kinds:
            source = "subagent"
        elif harness == "codex":
            source = "codex"
        else:
            source = "top"

    role = attempt.get("role") if isinstance(attempt.get("role"), dict) else {}
    phase = attempt.get("phase") if isinstance(attempt.get("phase"), dict) else {}
    origin = attempt.get("origin") if isinstance(attempt.get("origin"), dict) else {}
    model = attempt.get("model") if isinstance(attempt.get("model"), dict) else {}
    cost = attempt.get("cost") if isinstance(attempt.get("cost"), dict) else None
    workspace = origin.get("workspace")
    if workspace is None:
        workspace = logical.get("group") or run_workspace

    launched_by = None
    launcher_attempt = origin.get("launched_by")
    if isinstance(launcher_attempt, str) and launcher_attempt in attempt_to_graph:
        lag_s = next(
            (
                _lag(edge.get("evidence"))
                for edge in launch_edges
                if edge.get("to_attempt") == attempt.get("id")
                and edge.get("from_attempt") == launcher_attempt
            ),
            None,
        )
        launched_by = {
            "id": attempt_to_graph[launcher_attempt],
            "workspace": workspace,
            "how": origin.get("how"),
            "tier": origin.get("tier"),
            "evidence": origin.get("evidence"),
            "lag_s": lag_s,
        }
    elif origin.get("external") is True:
        launched_by = {
            "id": None,
            "external": True,
            "workspace": workspace,
            "how": origin.get("how"),
            "tier": origin.get("tier"),
            "evidence": origin.get("evidence"),
        }

    attempt_ext = _ext(attempt)
    spawn = copy.deepcopy(attempt_ext.get("spawn")) if isinstance(attempt_ext.get("spawn"), dict) else None
    session = attempt.get("session")
    session_path = str(session) if isinstance(session, str) and session else str(attempt.get("id") or graph_id)
    usd = cost.get("usd") if cost is not None else None
    return GraphNode(
        id=graph_id,
        harness=str(harness),
        source=source,
        session_path=session_path,
        model=model.get("raw") or model.get("id") or None,
        model_tier=model.get("tier"),
        effort=attempt.get("effort"),
        workspace=workspace,
        ts=attempt.get("started_at"),
        wall_s=_duration(attempt.get("started_at"), attempt.get("ended_at")),
        tokens=_tokens(cost),
        usd=float(usd) if isinstance(usd, (int, float)) and not isinstance(usd, bool) else None,
        spawn=spawn,
        launched_by=launched_by,
        role=role.get("value"),
        role_tier=role.get("tier"),
        role_evidence=role.get("evidence"),
        phase=phase.get("value"),
        phase_tier=phase.get("tier"),
    )


def _convert(document: dict) -> tuple[Graph, list[dict]]:
    if not isinstance(document, dict) or not version_at_least(document, "0.2"):
        version = document.get("ocp") if isinstance(document, dict) else None
        raise OCPError(f"only OCP v0.2 documents are supported, or later, got {version!r}")
    reader_counters = _validate_v02(document)

    nodes = _objects(document, "nodes")
    attempts = _objects(document, "attempts")
    edges_in = _objects(document, "edges")
    artifacts_in = _objects(document, "artifacts")
    node_by_id = _unique(nodes, "id", "node")
    attempt_to_graph, primary, attempts_by_node = _attempt_graph_ids(nodes, attempts)

    incoming: dict[str, set[str]] = defaultdict(set)
    launch_edges: list[dict] = []
    for edge in edges_in:
        if isinstance(edge.get("to"), str) and isinstance(edge.get("kind"), str):
            incoming[edge["to"]].add(edge["kind"])
        if edge.get("kind") == "launch":
            launch_edges.append(edge)

    run = document.get("run") if isinstance(document.get("run"), dict) else {}
    producer = document.get("producer") if isinstance(document.get("producer"), dict) else {}
    graph_nodes: list[GraphNode] = []
    ocp_node_ext: dict[str, dict] = {}
    for logical_id in sorted(node_by_id):
        logical = node_by_id[logical_id]
        ordered = sorted(
            attempts_by_node.get(logical_id, []),
            key=lambda a: (
                a.get("n") if isinstance(a.get("n"), int) and not isinstance(a.get("n"), bool) else 10**12,
                a["id"],
            ),
        )
        if not ordered:
            graph_node = _graph_node(
                primary[logical_id], logical, None,
                incoming_kinds=incoming.get(logical_id, set()),
                attempt_to_graph=attempt_to_graph, launch_edges=launch_edges,
                run_workspace=run.get("workspace"), producer_framework=producer.get("framework"),
            )
            graph_nodes.append(graph_node)
            if _ext(logical):
                ocp_node_ext[graph_node.id] = copy.deepcopy(_ext(logical))
        else:
            for attempt in ordered:
                graph_node = _graph_node(
                    attempt_to_graph[attempt["id"]], logical, attempt,
                    incoming_kinds=incoming.get(logical_id, set()),
                    attempt_to_graph=attempt_to_graph, launch_edges=launch_edges,
                    run_workspace=run.get("workspace"), producer_framework=producer.get("framework"),
                )
                graph_nodes.append(graph_node)
                if _ext(logical):
                    ocp_node_ext[graph_node.id] = copy.deepcopy(_ext(logical))
    graph_node_by_id = {node.id: node for node in graph_nodes}

    def endpoint(edge: dict, side: str) -> str:
        attempt_ref = edge.get(f"{side}_attempt")
        if isinstance(attempt_ref, str) and attempt_ref in attempt_to_graph:
            return attempt_to_graph[attempt_ref]
        logical = edge.get("from" if side == "from" else "to")
        if logical not in primary:
            raise OCPError(f"OCP edge names unknown {side} node {logical!r}")
        return primary[str(logical)]

    artifact_path: dict[str, str] = {}
    graph_artifacts: list[Artifact] = []
    used_paths: set[str] = set()
    for record in sorted(artifacts_in, key=lambda a: str(a.get("id") or "")):
        artifact_id = record.get("id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise OCPError("every OCP artifact must have a non-empty string id")
        full_path = _ext(record).get("path") or record.get("path")
        if not isinstance(full_path, str) or not full_path:
            raise OCPError(f"OCP artifact {artifact_id!r} has no usable path")
        if full_path in used_paths:
            raise OCPError(
                f"OCP artifacts cannot be represented by path: duplicate path {full_path!r}"
            )
        used_paths.add(full_path)
        artifact_path[artifact_id] = full_path
        kind = record.get("kind") if isinstance(record.get("kind"), dict) else {}
        artifact_extension = _artifact_extension(record, artifact_id)

        def attempt_nodes(field: str) -> list[str]:
            refs = record.get(field, [])
            if not isinstance(refs, list):
                raise OCPError(f"OCP artifact {artifact_id!r} field {field!r} must be a list")
            try:
                return [attempt_to_graph[ref] for ref in refs]
            except KeyError as exc:
                raise OCPError(
                    f"OCP artifact {artifact_id!r} names unknown attempt {exc.args[0]!r}"
                ) from exc

        producer_ref = record.get("producer")
        if producer_ref not in attempt_to_graph:
            raise OCPError(f"OCP artifact {artifact_id!r} names unknown producer {producer_ref!r}")
        graph_artifacts.append(
            Artifact(
                id=full_path,
                producer=attempt_to_graph[producer_ref],
                writers=attempt_nodes("writers"),
                consumers=attempt_nodes("consumers"),
                first_write_ts=record.get("first_write_at"),
                n_writes=record.get("n_writes"),
                n_reads=record.get("n_reads"),
                kind=kind.get("value"),
                kind_tier=kind.get("tier"),
                **artifact_extension,
            )
        )

    graph_edges: list[GraphEdge] = []
    for record in edges_in:
        kind = record.get("kind", "dep")
        src, dst = endpoint(record, "from"), endpoint(record, "to")
        detail: dict = {}
        evidence = record.get("evidence")
        if isinstance(evidence, str):
            detail["evidence"] = evidence
        if kind == "artifact":
            original_artifact = record.get("artifact")
            if original_artifact not in artifact_path:
                raise OCPError(f"OCP artifact edge names unknown artifact {original_artifact!r}")
            detail["path"] = artifact_path[original_artifact]
            ext = _ext(record)
            graph_from_attempt = ext.get("graph_from_attempt")
            graph_from = ext.get("graph_from")
            if isinstance(graph_from_attempt, str) and graph_from_attempt in attempt_to_graph:
                src = attempt_to_graph[graph_from_attempt]
            elif isinstance(graph_from, str) and graph_from in primary:
                src = primary[graph_from]
            rerouted = _REROUTED_RE.match(evidence or "")
            if rerouted:
                detail["evidence"] = rerouted.group(1)
        elif kind == "launch":
            lag_s = _lag(evidence)
            if lag_s is not None:
                detail["lag_s"] = lag_s
            launched_by = graph_node_by_id.get(dst).launched_by if graph_node_by_id.get(dst) else None
            if isinstance(launched_by, dict) and isinstance(launched_by.get("how"), str):
                detail["how"] = launched_by["how"]
            command = _ext(record).get("command")
            if isinstance(command, str):
                detail["command"] = command
                if graph_node_by_id.get(dst) and isinstance(graph_node_by_id[dst].launched_by, dict):
                    graph_node_by_id[dst].launched_by["command"] = command
        graph_edges.append(GraphEdge(src, dst, str(kind), record.get("tier"), detail))
        if kind == "spawn" and dst in graph_node_by_id:
            graph_node_by_id[dst].parent = src
        elif kind == "launch" and dst in graph_node_by_id:
            launched_by = graph_node_by_id[dst].launched_by
            if isinstance(launched_by, dict) and launched_by.get("id") == src:
                graph_node_by_id[dst].parent = src

    root_ext = _ext(document)
    raw_meta = root_ext.get("meta")
    meta = copy.deepcopy(raw_meta) if isinstance(raw_meta, dict) else {}
    source_meta = copy.deepcopy(meta)
    graph = Graph(
        nodes=graph_nodes,
        edges=graph_edges,
        artifacts=graph_artifacts,
        meta=meta,
        _ocp_node_ext=ocp_node_ext,
    )
    for key, value in summarize(graph).items():
        graph.meta.setdefault(key, value)
    graph.meta.setdefault("nodes_missing_wall_s", sum(node.wall_s is None for node in graph.nodes))
    graph.meta.update(_token_completeness_counters(graph.nodes))
    graph.meta.update(reader_counters)
    _settle_projection_loss_counters(document, graph, source_meta)

    # Build the analysis rows from the exact attempt records while the mapping
    # from logical nodes and attempt ids is still available.
    kinds_by_attempt: dict[str, Counter] = defaultdict(Counter)
    dirs_by_attempt: dict[str, set[str]] = defaultdict(set)
    for artifact in artifacts_in:
        path = _ext(artifact).get("path") or artifact.get("path")
        if not isinstance(path, str):
            continue
        kind = file_kind(path)
        parent = str(Path(path).parent)
        for writer in artifact.get("writers", []) if isinstance(artifact.get("writers"), list) else []:
            if kind:
                kinds_by_attempt[writer][kind] += 1
            if parent not in ("", "."):
                dirs_by_attempt[writer].add(parent)

    rows: list[dict] = []
    for attempt in attempts:
        graph_id = attempt_to_graph[attempt["id"]]
        node = graph_node_by_id[graph_id]
        outcome = attempt.get("outcome") if isinstance(attempt.get("outcome"), dict) else {}
        result = outcome.get("result") or attempt.get("status")
        if result in _TERMINAL_TRUE:
            accepted = True
        elif result in _TERMINAL_FALSE:
            accepted = False
        else:
            accepted = None
        evidence = outcome.get("evidence")
        if evidence in _OUTCOME_TIERS:
            tier = evidence
        elif result in _TERMINAL_TRUE | _TERMINAL_FALSE:
            tier = "asserted"
        else:
            tier = "ungraded"
        signal = outcome.get("reason") or outcome.get("receipt") or f"OCP outcome {result or 'unknown'}"
        original_n = attempt.get("n")
        row = {
            "run_id": graph_id,
            "harness": node.harness,
            "model": node.model,
            "effort": node.effort,
            "tokens": copy.deepcopy(node.tokens),
            "wall_s": node.wall_s,
            "ts": node.ts,
            "workspace": node.workspace,
            "attempt": original_n if isinstance(original_n, int) and not isinstance(original_n, bool) else 1,
            "attempt_rule": "ocp-v0.2",
            "attempt_clause": "imported",
            "touched_dirs": sorted(dirs_by_attempt[attempt["id"]]),
            "written_file_kinds": dict(sorted(kinds_by_attempt[attempt["id"]].items())),
            "session_path": node.session_path,
            "subagents": 0,
            "usd": node.usd,
            "grade": {"accepted": accepted, "tier": tier, "signal": signal},
        }
        rows.append(row)
    return graph, rows


def from_ocp(document: dict) -> Graph:
    """Convert one parsed OCP v0.2 document to a :class:`~loopmath.graph.Graph`."""
    return _convert(document)[0]


def load_ocp(path: str | Path) -> Graph:
    """Read and convert one OCP v0.2 JSON file."""
    return from_ocp(read_document(path))


def records_from_ocp(document: dict) -> list[dict]:
    """Return already-graded, already-priced analysis rows for an OCP document."""
    return _convert(document)[1]
