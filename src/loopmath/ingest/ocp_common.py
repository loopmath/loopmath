"""Shared validation and parsing helpers for the OCP v0.2 reader."""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..ocp.version import version_at_least


class OCPError(ValueError):
    """An OCP source cannot be represented safely as a loopmath graph."""


_COST_TO_GRAPH = {
    "input_tokens": "in",
    "cached_input_tokens": "cache_read",
    "cache_creation_tokens": "cache_write",
    "cache_creation_5m_tokens": "cache_write_5m",
    "cache_creation_1h_tokens": "cache_write_1h",
    "output_tokens": "out",
}
_TERMINAL_TRUE = {"done"}
_TERMINAL_FALSE = {"failed", "rejected", "canceled", "lost"}
_OUTCOME_TIERS = {"verified", "reported", "heuristic", "asserted"}
_LAG_RE = re.compile(r"; lag (-?(?:\d+(?:\.\d*)?|\.\d+)) s(?:$|[;)])")
_REROUTED_RE = re.compile(
    r"^read joined to later writer .+? in the graph \((.*)\); "
    r"OCP runs the artifact edge from the producer .+$"
)
_CONFORMANCE_VALIDATE = None
# Historical native-emitter documents can carry a nonnegative integer
# nodes_missing_tokens summary which the stricter shared completeness rule
# recomputes differently.  An unchanged round trip restores only that typed
# legacy value; current writers agree, so they contribute no value state.
_OCP_META_VALUE_KEYS = ("nodes_missing_tokens",)
# These diagnostics do not exist in admitted native-emitter metadata.  Their
# source state is therefore absence, represented as closed removal markers.
# Keeping this list closed prevents an unrelated twelfth difference from
# silently becoming private round-trip state.
_OCP_META_ABSENT_KEYS = (
    "ocp_missing_cache_creation_1h_tokens",
    "ocp_missing_cache_creation_5m_tokens",
    "ocp_missing_cache_creation_tokens",
    "ocp_missing_cached_input_tokens",
    "ocp_missing_input_tokens",
    "ocp_missing_output_tokens",
    "ocp_nodes_partial_tokens",
    "ocp_nodes_without_cost",
    "ocp_reader_retired_prefix_findings_ignored",
    "ocp_token_streams_missing",
)


def _is_legacy_meta_value(key: str, source_value, loaded: dict) -> bool:
    """Return whether a changed source value has the old writer's summary shape."""
    if key != "nodes_missing_tokens":
        return False
    loaded_value = loaded.get(key)
    return (
        type(source_value) is int
        and source_value >= 0
        and type(loaded_value) is int
        and loaded_value >= 0
        and loaded_value != source_value
    )


def _meta_snapshots(source: dict, loaded: dict) -> tuple[dict, dict]:
    """Return the closed compatibility delta and complete loaded-meta guard.

    Current-writer documents need the nine named absence states.  Historical
    documents add the one named value state only when both source and loaded
    values are nonnegative, non-boolean integers that differ.  No other
    observed difference is absorbed.  Equal and unlisted members continue
    through ordinary public ``Graph.meta``.
    """
    delta = {
        "values": {
            key: copy.deepcopy(value)
            for key in _OCP_META_VALUE_KEYS
            if key in source
            for value in (source[key],)
            if _is_legacy_meta_value(key, value, loaded)
        },
        "absent": tuple(
            key for key in _OCP_META_ABSENT_KEYS if key in loaded and key not in source
        ),
    }
    return delta, copy.deepcopy(loaded)


def _validate_v02(
    document: dict, source: str = "OCP document"
) -> dict[str, int]:
    """Reject unreadable errors and count reader-compatible E181 findings."""
    global _CONFORMANCE_VALIDATE
    if _CONFORMANCE_VALIDATE is None:
        from ..ocp.conformance import validate_doc  # the checker is package code (spec 01 section 3)

        _CONFORMANCE_VALIDATE = validate_doc
    findings = _CONFORMANCE_VALIDATE(document)
    ignored_retired_prefixes = sum(
        finding.level == "error" and finding.code == "E181"
        for finding in findings
    )
    errors = [
        finding
        for finding in findings
        if finding.level == "error" and finding.code != "E181"
    ]
    if errors:
        detail = "; ".join(
            f"{finding.code} {finding.path}: {finding.message}" for finding in errors
        )
        raise OCPError(
            f"{source} fails OCP v{document.get('ocp')} conformance with {len(errors)} error(s): {detail}"
        )
    return {
        "ocp_reader_retired_prefix_findings_ignored": ignored_retired_prefixes,
    }


def _objects(document: dict, key: str) -> list[dict]:
    value = document.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise OCPError(f"OCP field {key!r} must be a list of objects")
    return value


def _unique(records: list[dict], key: str, entity: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in records:
        ident = record.get(key)
        if not isinstance(ident, str) or not ident:
            raise OCPError(f"every OCP {entity} must have a non-empty string {key}")
        if ident in out:
            raise OCPError(f"duplicate OCP {entity} {key} {ident!r}")
        out[ident] = record
    return out


def read_document(path: str | Path) -> dict:
    """Read one JSON OCP v0.2 document, with source-labelled errors."""
    source = Path(path).expanduser()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OCPError(f"cannot read OCP source {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise OCPError(f"OCP source {source} has a non-object top level")
    if not version_at_least(document, "0.2"):
        raise OCPError(
            f"OCP source {source} has version {document.get('ocp')!r}; only OCP v0.2 is supported, or later"
        )
    _validate_v02(document, f"OCP source {source}")
    return document


def _attempt_graph_ids(
    nodes: list[dict], attempts: list[dict]
) -> tuple[dict[str, str], dict[str, str], dict[str, list[dict]]]:
    node_ids = _unique(nodes, "id", "node")
    attempt_ids = _unique(attempts, "id", "attempt")
    by_node: dict[str, list[dict]] = defaultdict(list)
    for attempt_id, attempt in attempt_ids.items():
        logical = attempt.get("node")
        if logical not in node_ids:
            raise OCPError(f"OCP attempt {attempt_id!r} names unknown node {logical!r}")
        by_node[str(logical)].append(attempt)

    attempt_to_graph: dict[str, str] = {}
    primary: dict[str, str] = {}
    used: set[str] = set()
    for logical in sorted(node_ids):
        ordered = sorted(
            by_node.get(logical, []),
            key=lambda a: (
                a.get("n") if isinstance(a.get("n"), int) and not isinstance(a.get("n"), bool) else 10**12,
                a["id"],
            ),
        )
        if not ordered:
            graph_ids = [logical]
        elif len(ordered) == 1:
            graph_ids = [logical]
            attempt_to_graph[ordered[0]["id"]] = logical
        else:
            graph_ids = [a["id"] for a in ordered]
            for attempt, graph_id in zip(ordered, graph_ids):
                attempt_to_graph[attempt["id"]] = graph_id
        for graph_id in graph_ids:
            if graph_id in used:
                raise OCPError(
                    f"OCP attempts cannot be expanded without duplicate graph node id {graph_id!r}"
                )
            used.add(graph_id)
        # A node-level edge has no exact attempt endpoint.  The latest retry is
        # the least surprising projection of the logical node's current state.
        primary[logical] = graph_ids[-1]
    return attempt_to_graph, primary, by_node


def _duration(started_at, ended_at) -> float | None:
    if not isinstance(started_at, str) or not isinstance(ended_at, str):
        return None
    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds >= 0 else None


def _tokens(cost) -> dict | None:
    if not isinstance(cost, dict):
        return None
    return {
        graph_key: cost.get(ocp_key)
        for ocp_key, graph_key in _COST_TO_GRAPH.items()
    }


def _ext(record: dict) -> dict:
    ext = record.get("ext")
    if not isinstance(ext, dict):
        return {}
    # The graph writer's namespace, else the one it wrote before 0.3 (W182).
    value = ext.get("dev.loopmath.graph", ext.get("dev.dagr.graph"))
    return value if isinstance(value, dict) else {}


def _lag(evidence) -> float | None:
    if not isinstance(evidence, str):
        return None
    match = _LAG_RE.search(evidence)
    return float(match.group(1)) if match else None
