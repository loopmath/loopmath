"""Compact graph data for the self-contained HTML visualizer."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from .dataset_core import ROLE_WORDS
from .ocp import sanitize
from .ocp_support import _COST_FIELDS
from .schema import EDGE_KINDS, Graph, GraphEdge, GraphNode
from .token_completeness import TOKEN_STREAMS


_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_LAG = re.compile(r"([0-9]+(?:\.[0-9]+)?) s later(?:\)|$)")
_REVIEW = re.compile(r"review\.sh\s+(\S+)\s+(\S+)")
_EXTERNAL_URL = re.compile(
    r"(?i)\bhttps?://[^\s\"'<>]+|(?<![:/])//[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s\"'<>]*)?"
)
_TOKEN_TOTAL_STREAMS = tuple(key for key, _ in _COST_FIELDS)
_MAX_NUMBER = 9_007_199_254_740_991


def _role_prefix(role: str | None) -> str:
    if not role:
        return "top"
    if len(role) <= 4:
        return role
    return role[:4] if role.endswith("ner") else role[:3]


def _parse_ts(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fmt_ts(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _primary_workspace(nodes: list[GraphNode]) -> str:
    values = [
        node.workspace for node in nodes if node.workspace and node.source != "external"
    ]
    if not values:
        values = [node.workspace for node in nodes if node.workspace]
    if not values:
        return "unlabeled"
    counts = Counter(values)
    return min(counts, key=lambda item: (-counts[item], item))


def _short_path(path: str, workspace: str) -> str:
    marker = f"/Workspace/{workspace}/"
    if marker in path:
        return path.split(marker, 1)[1]
    scratch = re.search(
        rf"^/private/tmp/[^/]+/-[^/]*-Workspace-{re.escape(workspace)}/(.*)$",
        path,
    )
    if scratch:
        return "scratch/" + scratch.group(1)
    return _UUID.sub(lambda match: match.group(0)[:8], path)


def _title(node: GraphNode, by_id: dict[str, GraphNode]) -> str:
    spawn = node.spawn if isinstance(node.spawn, dict) else {}
    if spawn.get("description"):
        return str(spawn["description"])
    launched = node.launched_by if isinstance(node.launched_by, dict) else {}
    command = str(launched.get("command") or "")
    match = _REVIEW.search(command)
    if match:
        return f"codex review of {match.group(1)} at {match.group(2)[:7]}"
    if "Reply with exactly one line" in command:
        return "codex smoke call"
    if "claude -p" in command:
        parent = by_id.get(node.parent or "")
        workspace = parent.workspace if parent and parent.workspace else launched.get("workspace")
        return "claude -p review launched from " + str(workspace or "another workspace")
    if node.role == "lead":
        return "lead session"
    if node.source == "external":
        return "external launcher session (not in the workspace)"
    if command.startswith("codex"):
        return "codex session"
    return node.role or node.source or "session"


def _lag(edge: GraphEdge) -> float | None:
    detail = edge.detail if isinstance(edge.detail, dict) else {}
    raw = _number(detail.get("lag_s"), maximum=86_400_000_000_000)
    if raw is not None:
        return float(raw)
    evidence = detail.get("evidence")
    if isinstance(evidence, str):
        match = _LAG.search(evidence)
        if match:
            return _number(float(match.group(1)), maximum=86_400_000_000_000)
    return None


def _edge_path(edge: GraphEdge) -> str | None:
    value = edge.detail.get("path") if isinstance(edge.detail, dict) else None
    return value if isinstance(value, str) and value else None


def _attempt_id(node_id: str) -> str:
    return node_id if re.search(r"\.a[1-9][0-9]*$", node_id) else node_id + ".a1"


def _number(value, *, maximum: float | None = _MAX_NUMBER):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        invalid = not math.isfinite(value)
    except OverflowError:
        invalid = True
    if invalid or value < 0 or (maximum is not None and value > maximum):
        return None
    return value


def _resolve(ref, positions: dict[str, list[int]]) -> tuple[int | None, str | None]:
    if not ref:
        return None, "unavailable"
    matches = positions.get(ref, [])
    if len(matches) == 1:
        return matches[0], None
    return None, "outside_graph" if not matches else "ambiguous_node_id"


def _clean(value, counters: Counter):
    if isinstance(value, float) and not math.isfinite(value):
        counters["nonfinite_values_replaced_with_null"] += 1
        return None
    if isinstance(value, str):
        text = sanitize(value, counters)
        text, replaced = _EXTERNAL_URL.subn("[external URL withheld]", text)
        if replaced:
            counters["strings_with_external_url_replaced"] += 1
        return text
    if isinstance(value, list):
        return [_clean(item, counters) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item, counters) for key, item in value.items()}
    return value


def build_html_data(graph: Graph) -> dict:
    """Return the compact data contract consumed by the inline scripts."""
    nodes = graph.nodes
    issues: Counter = Counter()
    by_id = {node.id: node for node in nodes}
    parsed = [_parse_ts(node.ts) for node in nodes]
    durations = []
    costs = []
    token_rows = []
    for node, stamp in zip(nodes, parsed):
        if stamp is None:
            issues["attempt_timestamps_unavailable"] += 1
        duration = _number(node.wall_s, maximum=86_400_000_000_000)
        if duration is None:
            issues["attempt_durations_unavailable"] += 1
        durations.append(duration)
        cost = _number(node.usd)
        if cost is None:
            issues["attempt_costs_unavailable"] += 1
        costs.append(cost)
        tokens = node.tokens if isinstance(node.tokens, dict) else None
        if tokens is None:
            issues["attempt_token_records_unavailable"] += 1
        token_row = {}
        missing_streams = 0
        for _, stream in TOKEN_STREAMS:
            value = _number(tokens.get(stream)) if tokens is not None else None
            if value is None:
                missing_streams += 1
            token_row[stream] = value
        if missing_streams:
            issues["attempts_with_incomplete_token_streams"] += 1
            issues["token_stream_values_unavailable"] += missing_streams
        token_rows.append(token_row)

    timed = [value for value in parsed if value is not None]
    start = min(timed) if timed else datetime(1970, 1, 1, tzinfo=timezone.utc)
    end = start
    for pos, (duration, stamp) in enumerate(zip(durations, parsed)):
        if stamp is None:
            continue
        end = max(end, stamp)
        if duration is not None:
            try:
                end = max(end, stamp + timedelta(seconds=duration))
            except OverflowError:
                durations[pos] = None
                issues["attempt_durations_out_of_range"] += 1

    order = sorted(range(len(nodes)), key=lambda pos: (nodes[pos].ts or "", nodes[pos].id))
    positions: dict[str, list[int]] = defaultdict(list)
    for rank, pos in enumerate(order):
        positions[nodes[pos].id].append(rank)
    issues["duplicate_node_ids"] = sum(len(items) - 1 for items in positions.values())
    role_totals = Counter(node.role for node in nodes)
    role_seen: Counter = Counter()
    out_nodes = []
    for rank, pos in enumerate(order):
        node = nodes[pos]
        role = node.role
        prefix = _role_prefix(role)
        if role in {"dev", "reviewer"}:
            role_seen[role] += 1
            width = len(str(role_totals[role]))
            label = f"{prefix}-{role_seen[role]:0{width}d}"
        else:
            label = prefix
        stamp = parsed[pos]
        relative = (stamp - start).total_seconds() if stamp is not None else None
        spawn = node.spawn if isinstance(node.spawn, dict) else None
        launched = node.launched_by if isinstance(node.launched_by, dict) else None
        if node.spawn is not None and spawn is None:
            issues["attempt_spawn_records_unavailable"] += 1
        if node.launched_by is not None and launched is None:
            issues["attempt_launch_records_unavailable"] += 1
        issues["attempt_statuses_unavailable"] += 1
        for field, value in (
            ("models", node.model),
            ("model_tiers", node.model_tier),
            ("effort_values", node.effort),
            ("sources", node.source),
            ("harnesses", node.harness),
            ("workspaces", node.workspace),
            ("roles", node.role),
            ("role_tiers", node.role_tier),
            ("role_evidence_values", node.role_evidence),
            ("phases", node.phase),
            ("phase_tiers", node.phase_tier),
            ("session_paths", node.session_path),
        ):
            if value is None or value == "":
                issues[f"attempt_{field}_unavailable"] += 1
        parent, parent_reason = (
            _resolve(node.parent, positions) if node.parent else (None, None)
        )
        if parent_reason:
            issues[f"attempt_parent_refs_{parent_reason}"] += 1
        launch_lag = _number(launched.get("lag_s")) if launched else None
        if launched and launch_lag is None:
            issues["attempt_launch_lags_unavailable"] += 1
        out_nodes.append(
            {
                "i": rank,
                "id": node.id,
                "aid": _attempt_id(node.id),
                "lbl": label,
                "title": _title(node, by_id),
                "role": role or "unlabeled",
                "role_source": role,
                "rt": node.role_tier,
                "rtd": node.role_tier if role else None,
                "re": node.role_evidence,
                "model": node.model,
                "mt": node.model_tier,
                "effort": node.effort,
                "src": node.source,
                "harness": node.harness,
                "phase": node.phase,
                "pt": node.phase_tier,
                "ws": node.workspace,
                "ts": node.ts,
                "t0": relative,
                "dur": durations[pos],
                "usd": costs[pos],
                "tok": token_rows[pos],
                "tok_record": isinstance(node.tokens, dict),
                "parent": parent,
                "parent_source": node.parent,
                "parent_ref": node.parent if parent_reason else None,
                "parent_reason": parent_reason,
                "spawn": spawn,
                "launch": launched,
                "session": node.session_path,
                "status": None,
            }
        )

    workspace = _primary_workspace(nodes)
    artifact_positions: dict[str, list[int]] = defaultdict(list)
    for pos, artifact in enumerate(graph.artifacts):
        artifact_positions[artifact.id].append(pos)
    issues["duplicate_artifact_ids"] = sum(
        len(items) - 1 for items in artifact_positions.values()
    )
    artifact_index = {
        artifact_id: items[0]
        for artifact_id, items in artifact_positions.items()
        if len(items) == 1
    }
    out_artifacts = []
    for pos, artifact in enumerate(graph.artifacts):
        written = _parse_ts(artifact.first_write_ts)
        if written is None:
            issues["artifact_first_write_times_unavailable"] += 1
        full_path = str(artifact.id) if artifact.id else "[artifact id unavailable]"
        if not artifact.id:
            issues["artifact_ids_unavailable"] += 1
        short_path = _short_path(full_path, workspace)
        if short_path != full_path:
            issues["artifact_paths_shortened_for_display"] += 1

        def participants(refs, relation):
            resolved = []
            missing = []
            seen = set()
            for ref in refs or []:
                node_pos, reason = _resolve(ref, positions)
                if reason:
                    issues[f"artifact_{relation}_refs_{reason}"] += 1
                    missing.append({"id": ref, "reason": reason})
                elif node_pos in seen:
                    issues[f"artifact_{relation}_refs_duplicated"] += 1
                else:
                    seen.add(node_pos)
                    resolved.append(node_pos)
            return resolved, missing

        writers, missing_writers = participants(artifact.writers, "writer")
        consumers, missing_consumers = participants(artifact.consumers, "consumer")
        producer, producer_reason = _resolve(artifact.producer, positions)
        if producer_reason:
            issues[f"artifact_producer_refs_{producer_reason}"] += 1
        out_artifacts.append(
            {
                "i": pos,
                "path": short_path,
                "kind": artifact.kind,
                "kt": artifact.kind_tier,
                "prod": producer,
                "pm": {"id": artifact.producer, "reason": producer_reason}
                if producer_reason
                else None,
                "w": writers,
                "wm": missing_writers,
                "c": consumers,
                "cm": missing_consumers,
                "t": (written - start).total_seconds() if written else None,
                "nw": _number(artifact.n_writes),
                "nr": _number(artifact.n_reads),
                "lang": artifact.language,
                "bytes": _number(artifact.bytes),
                "la": _number(artifact.lines_added),
                "lr": _number(artifact.lines_removed),
                "hint": artifact.hint,
            }
        )

    kind_rank = {kind: rank for rank, kind in enumerate(EDGE_KINDS)}
    ordered_edges = sorted(
        enumerate(graph.edges),
        key=lambda pair: (
            kind_rank.get(pair[1].kind, len(EDGE_KINDS)),
            artifact_index.get(_edge_path(pair[1]), len(graph.artifacts))
            if pair[1].kind == "artifact"
            else pair[0],
            pair[0],
        ),
    )
    out_edges = []
    for _, edge in ordered_edges:
        source, source_reason = _resolve(edge.src, positions)
        target, target_reason = _resolve(edge.dst, positions)
        endpoint_reasons = {source_reason, target_reason} - {None}
        if endpoint_reasons:
            reason = next(
                item
                for item in ("unavailable", "outside_graph", "ambiguous_node_id")
                if item in endpoint_reasons
            )
            issues[f"edges_excluded_{reason}"] += 1
            continue
        if edge.kind not in EDGE_KINDS:
            issues["edges_excluded_unknown_kind"] += 1
            continue
        tier = edge.tier or "unknown"
        if not edge.tier:
            issues["edge_tiers_unavailable"] += 1
        record = [source, target, edge.kind, tier]
        if edge.kind == "artifact":
            path = _edge_path(edge)
            matches = artifact_positions.get(path, []) if path else []
            if not path:
                issues["artifact_edges_excluded_missing_path"] += 1
                continue
            if not matches:
                issues["artifact_edges_excluded_unmatched_path"] += 1
                continue
            if len(matches) != 1:
                issues["artifact_edges_excluded_ambiguous_path"] += 1
                continue
            if source not in out_artifacts[matches[0]]["w"]:
                issues["artifact_edges_with_unlisted_writers"] += 1
            if target not in out_artifacts[matches[0]]["c"]:
                issues["artifact_edges_with_unlisted_consumers"] += 1
            lag = _lag(edge)
            if lag is None:
                issues["artifact_edge_lags_unavailable"] += 1
            record.extend([matches[0], lag])
        elif edge.kind == "launch":
            lag = _lag(edge)
            if lag is None:
                issues["launch_edge_lags_unavailable"] += 1
            record.extend([None, lag])
        else:
            record.extend([None, None])
        out_edges.append(record)

    issues["edges_excluded"] = len(graph.edges) - len(out_edges)

    artifact_edges = defaultdict(list)
    for edge_pos, edge in enumerate(out_edges):
        if edge[2] == "artifact":
            artifact_edges[(edge[4], edge[0], "w")].append(edge_pos)
            artifact_edges[(edge[4], edge[1], "c")].append(edge_pos)
    for artifact in out_artifacts:
        artifact["we"] = [artifact_edges[(artifact["i"], item, "w")] for item in artifact["w"]]
        artifact["ce"] = [artifact_edges[(artifact["i"], item, "c")] for item in artifact["c"]]
        issues["artifact_writer_connections_without_edges"] += sum(
            not items for items in artifact["we"]
        )
        issues["artifact_consumer_connections_without_edges"] += sum(
            not items for items in artifact["ce"]
        )

    span = max((end - start).total_seconds(), 0.0)
    if span < 1:
        issues["run_spans_padded_for_plot"] += 1
    roles = Counter((node.role or "unlabeled") for node in nodes)
    edge_counts = Counter(f"{edge[2]}/{edge[3]}" for edge in out_edges)
    data = {
        "run": {
            "workspace": workspace,
            "start": _fmt_ts(start) if timed else None,
            "end": _fmt_ts(end) if timed else None,
            "clock_base": _fmt_ts(start),
            "span_s": span if timed else None,
            "plot_span_s": max(span, 1.0),
            "n_nodes": len(out_nodes),
            "usd_total": graph.meta.get("usd_total"),
            "usd_unpriced_nodes": graph.meta.get("usd_unpriced_nodes"),
            "n_artifacts": len(out_artifacts),
            "n_consumed": sum(bool(artifact["c"]) for artifact in out_artifacts),
            "n_source_edges": len(graph.edges),
            "edges_by_kind_tier": dict(sorted(edge_counts.items())),
            "roles": dict(roles),
            "role_vocabulary": list(ROLE_WORDS),
            # The viewer draws and names one line style per kind and takes the
            # vocabulary from here, so the page never carries a second copy of
            # `EDGE_KINDS`.
            "edge_kinds": list(EDGE_KINDS),
            "attempt_source_fields": list(GraphNode.__dataclass_fields__),
            "attempt_field_omissions": [],
            "token_streams": [key for _, key in TOKEN_STREAMS],
            "token_total_streams": list(_TOKEN_TOTAL_STREAMS),
            "source": f"loopmath graph --workspace {workspace} --format html",
        },
        "nodes": out_nodes,
        "edges": out_edges,
        "artifacts": out_artifacts,
    }
    cleaned = _clean(data, issues)
    cleaned["run"]["accounting"] = dict(
        sorted((key, count) for key, count in issues.items() if count)
    )
    return cleaned
