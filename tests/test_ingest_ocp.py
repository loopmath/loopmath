"""OCP v0.2 reader and OCP-backed CLI sources."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from loopmath import ingest
from loopmath.cli import main
from loopmath.graph import Artifact, Graph, GraphEdge, GraphNode, extract, to_dot, to_ocp
from loopmath.graph.ocp_support import _cost
from loopmath.ingest.ocp import (
    OCPError,
    coverage_from_records,
    from_ocp,
    read_document,
    records_from_ocp,
)
from loopmath.ingest.ocp_common import _OCP_META_ABSENT_KEYS, _OCP_META_VALUE_KEYS
from loopmath.ingest.ocp_projection import _extension_loss_counters


ROOT = Path(__file__).resolve().parent.parent
SWARM = ROOT / "spec" / "examples" / "swarm-v02.ocp.json"
NATIVE_SWARMS = ROOT / "tests" / "fixtures" / "graph" / "roundtrip"


def _native_eval_graph(name: str) -> Graph:
    envelope = json.loads(
        (NATIVE_SWARMS / f"native-swarm-{name}.json").read_bytes()
    )
    assert envelope.get("encoding") == "gzip+base64"
    compressed = base64.b64decode(envelope["data"], validate=True)
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == envelope["sha256"]
    assert len(raw) == envelope["uncompressed_bytes"]
    payload = json.loads(raw)
    assert payload["dagr_graph"] == 1
    return Graph(
        nodes=[GraphNode(**node) for node in payload["nodes"]],
        edges=[GraphEdge(**edge) for edge in payload["edges"]],
        artifacts=[Artifact(**artifact) for artifact in payload["artifacts"]],
        meta=payload["meta"],
    )


def _canonical(document: dict) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _attempt(node: str, *, n: int = 1, suffix: str = "a1", model: str = "gpt-test") -> dict:
    return {
        "id": f"{node}.{suffix}",
        "node": node,
        "n": n,
        "harness": "codex",
        "model": {"raw": model, "tier": "reported"},
        "effort": "high",
        "status": "done",
        "started_at": "2026-09-02T20:00:00Z",
        "ended_at": "2026-09-02T20:02:00Z",
        "outcome": {"result": "done", "evidence": "verified", "receipt": "synthetic gate"},
        "cost": {
            "input_tokens": 10,
            "cached_input_tokens": 20,
            "cache_creation_tokens": 3,
            "cache_creation_5m_tokens": 1,
            "cache_creation_1h_tokens": 2,
            "output_tokens": 4,
            "usd": 0.25,
            "basis": "measured",
        },
        "session": f"session-{node}-{suffix}",
        "origin": {"launched_by": None, "workspace": f"ws-{node}", "external": False, "how": None, "tier": "verified"},
        "role": {"value": "reviewer", "tier": "reported", "evidence": "synthetic role"},
        "phase": {"value": "build", "tier": "heuristic", "evidence": "synthetic phase"},
    }


def _doc(node: str, *, attempts: list[dict] | None = None) -> dict:
    return {
        "ocp": "0.2",
        "producer": {"name": "test-adapter", "framework": "codex"},
        "privacy": {"profile": "metadata_only"},
        "run": {"id": f"run-{node}", "workspace": f"ws-{node}"},
        "nodes": [{"id": node, "kind": "review", "labels": {"harness": "codex", "source": "codex"}}],
        "edges": [],
        "attempts": attempts if attempts is not None else [_attempt(node)],
        "artifacts": [],
    }


def test_reader_preserves_attempt_cost_identity_labels_edges_and_artifacts():
    document = read_document(SWARM)
    graph = from_ocp(document)
    assert len(graph.nodes) == len(document["attempts"]) == 6
    lead = graph.node("S-lead")
    assert (lead.model, lead.model_tier, lead.effort) == ("claude-fable-5-1", "verified", "medium")
    assert lead.tokens == {
        "in": 4200,
        "cache_read": 910000,
        "cache_write": 62000,
        "cache_write_5m": 41000,
        "cache_write_1h": 21000,
        "out": 18000,
    }
    assert (lead.ts, lead.wall_s, lead.session_path) == (
        "2026-08-31T09:00:00Z",
        5400.0,
        "0f3c2a1e-lead",
    )
    assert (lead.role, lead.role_tier, lead.phase, lead.phase_tier) == (
        "lead", "heuristic", "build", "heuristic"
    )
    review = graph.node("S-rev")
    assert review.launched_by["id"] == "S-lead"
    assert review.launched_by["tier"] == "heuristic"
    assert graph.node("S-plan").parent == "S-lead"
    assert [(edge.kind, edge.tier) for edge in graph.edges].count(("spawn", "verified")) == 2
    assert {(artifact.id, artifact.producer) for artifact in graph.artifacts} == {
        ("docs/BUILD-PLAN.md", "S-plan"),
        ("src/feature.py", "S-dev"),
        ("reviews/dev-A.md", "S-rev"),
    }


def test_cost_usd_and_outcomes_feed_analysis_without_repricing():
    document = _doc("one")
    graph = from_ocp(document)
    node = graph.node("one")
    assert node.tokens == {
        "in": 10,
        "cache_read": 20,
        "cache_write": 3,
        "cache_write_5m": 1,
        "cache_write_1h": 2,
        "out": 4,
    }
    assert node.usd == 0.25 and node.wall_s == 120.0
    records = records_from_ocp(document)
    assert records[0]["usd"] == 0.25
    assert records[0]["grade"] == {
        "accepted": True,
        "tier": "verified",
        "signal": "synthetic gate",
    }
    coverage = coverage_from_records(records)
    assert (coverage["n_total"], coverage["n_graded"], coverage["n_accepted"]) == (1, 1, 1)


def test_reader_counts_valid_ocp_facts_that_the_graph_cannot_model():
    document = to_ocp(from_ocp(_doc("loss")))
    document["future_root"] = None
    document["producer"]["future_member"] = "producer fact"
    document["producer"]["future_null_member"] = None
    document["privacy"]["future_member"] = "privacy fact"
    document["run"]["future_member"] = "run fact"
    document["groups"][0]["future_member"] = "group fact"
    document["groups"].append({"id": "unreferenced-group"})
    document["nodes"][0]["future_member"] = "node fact"
    document["attempts"][0]["future_member"] = "attempt fact"
    document["attempts"][0]["future_null_member"] = None
    document["events"].append(
        {
            "at": "2026-09-02T20:01:00Z",
            "type": "note",
            "node": "loss",
            "attempt": "loss.a1",
            "detail": "event with no Graph representation",
        }
    )
    document["ext"]["example.run"] = {"kept": False}
    document["nodes"][0].setdefault("ext", {})["example.node"] = {"kept": False}

    graph = from_ocp(document)

    assert {
        key: value
        for key, value in graph.meta.items()
        if key.startswith("ocp_projection_")
    } == {
        "ocp_projection_attempt_members_unmodeled": 2,
        # Normalizing before mutation left stale source projection counters;
        # replacing that supplied meta member is itself an accounted loss.
        "ocp_projection_dagr_graph_members_unmodeled": 1,
        "ocp_projection_extension_namespaces_unmodeled": 2,
        "ocp_projection_group_members_unmodeled": 1,
        "ocp_projection_groups_unmodeled": 1,
        "ocp_projection_events_unmodeled": 1,
        "ocp_projection_node_members_unmodeled": 1,
        "ocp_projection_privacy_members_unmodeled": 1,
        "ocp_projection_producer_members_unmodeled": 2,
        "ocp_projection_root_members_unmodeled": 1,
        "ocp_projection_run_members_unmodeled": 1,
    }


def test_reader_counts_unpreserved_dagr_graph_extension_members():
    fresh = to_ocp(from_ocp(_doc("loopmath-ext-loss")))
    assert "ocp_projection_dagr_graph_members_unmodeled" not in from_ocp(fresh).meta

    fresh["ext"]["dev.loopmath.graph"]["future_root_member"] = {"lost": True}
    fresh["attempts"][0].setdefault("ext", {}).setdefault("dev.loopmath.graph", {})[
        "future_attempt_member"
    ] = {"lost": True}
    fresh["nodes"][0].setdefault("ext", {}).setdefault("dev.loopmath.graph", {})[
        "preserved_node_member"
    ] = {"kept": True}

    graph = from_ocp(fresh)

    assert graph.meta["ocp_projection_dagr_graph_members_unmodeled"] == 2
    projected_node = to_ocp(graph)["nodes"][0]
    assert projected_node["ext"]["dev.loopmath.graph"]["preserved_node_member"] == {
        "kept": True
    }


def test_reader_counts_explicit_null_dagr_meta_overwritten_by_projection_counter():
    document = to_ocp(from_ocp(_doc("projection-null")))
    document["ext"]["dev.loopmath.graph"]["meta"][
        "ocp_projection_root_members_unmodeled"
    ] = None
    document["future_root"] = None

    graph = from_ocp(document)

    assert graph.meta["ocp_projection_root_members_unmodeled"] == 1
    assert graph.meta.get("ocp_projection_dagr_graph_members_unmodeled") == 1


def test_partial_cost_has_complete_none_shape_and_missing_stream_counters():
    document = _doc("partial")
    del document["attempts"][0]["cost"]["cache_creation_1h_tokens"]
    del document["attempts"][0]["cost"]["output_tokens"]

    graph = from_ocp(document)

    assert type(graph.node("partial").tokens) is dict
    assert graph.node("partial").tokens == {
        "in": 10,
        "cache_read": 20,
        "cache_write": 3,
        "cache_write_5m": 1,
        "cache_write_1h": None,
        "out": None,
    }
    assert graph.meta["nodes_missing_tokens"] == 1
    assert graph.meta["ocp_nodes_partial_tokens"] == 1
    assert graph.meta["ocp_token_streams_missing"] == 2
    assert graph.meta["ocp_missing_cache_creation_1h_tokens"] == 1
    assert graph.meta["ocp_missing_output_tokens"] == 1


def test_reader_counts_explicit_null_dagr_meta_overwritten_by_token_counter():
    document = _doc("token-null")
    del document["attempts"][0]["cost"]["output_tokens"]
    document = to_ocp(from_ocp(document))
    document["ext"]["dev.loopmath.graph"]["meta"]["ocp_missing_output_tokens"] = None

    graph = from_ocp(document)
    reemitted = to_ocp(graph)

    assert graph.meta["ocp_missing_output_tokens"] == 1
    assert reemitted["ext"]["dev.loopmath.graph"]["meta"]["ocp_missing_output_tokens"] == 1
    assert graph.meta.get("ocp_projection_dagr_graph_members_unmodeled") == 1


def test_reader_counts_explicit_null_nodes_missing_tokens_overwrite():
    document = _doc("nodes-missing-null")
    del document["attempts"][0]["cost"]["output_tokens"]
    document = to_ocp(from_ocp(document))
    assert "ocp_projection_dagr_graph_members_unmodeled" not in from_ocp(document).meta
    document["ext"]["dev.loopmath.graph"]["meta"]["nodes_missing_tokens"] = None

    graph = from_ocp(document)
    reemitted = to_ocp(graph)

    assert graph.meta["nodes_missing_tokens"] == 1
    assert (
        graph.meta.get("ocp_projection_dagr_graph_members_unmodeled"),
        reemitted["ext"]["dev.loopmath.graph"]["meta"]["nodes_missing_tokens"],
    ) == (1, 1)


@pytest.mark.parametrize("source_value", [False, -1, "1"])
def test_reader_does_not_restore_nonlegacy_missing_token_summary(source_value):
    document = _doc("nodes-missing-nonlegacy")
    del document["attempts"][0]["cost"]["output_tokens"]
    document = to_ocp(from_ocp(document))
    document["ext"]["dev.loopmath.graph"]["meta"]["nodes_missing_tokens"] = source_value

    graph = from_ocp(document)

    assert graph.meta["nodes_missing_tokens"] == 1
    assert graph._ocp_source_meta["values"] == {}
    assert graph.meta["ocp_projection_dagr_graph_members_unmodeled"] == 1


def test_reader_meta_snapshot_preserves_source_until_current_meta_changes():
    admitted = to_ocp(_native_eval_graph("c"))

    graph = from_ocp(admitted)

    assert admitted["ext"]["dev.loopmath.graph"]["meta"]["nodes_missing_tokens"] == 1
    assert graph.meta["nodes_missing_tokens"] == 4
    assert graph.meta["ocp_nodes_partial_tokens"] == 3
    assert to_ocp(graph)["ext"]["dev.loopmath.graph"]["meta"] == admitted["ext"]["dev.loopmath.graph"]["meta"]
    assert Graph(meta={"same": True}) == Graph(
        meta={"same": True},
        _ocp_source_meta={"values": {"private": True}, "absent": ()},
        _ocp_loaded_meta={"same": True, "private": True},
    )

    loaded_missing = graph.meta["nodes_missing_tokens"]
    graph.meta["nodes_missing_tokens"] = 99
    assert to_ocp(graph)["ext"]["dev.loopmath.graph"]["meta"]["nodes_missing_tokens"] == 99
    graph.meta["nodes_missing_tokens"] = loaded_missing
    assert to_ocp(graph)["ext"]["dev.loopmath.graph"]["meta"] == admitted["ext"]["dev.loopmath.graph"]["meta"]

    del graph.meta["n_nodes"]
    graph.meta["nodes_missing_tokens"] = None
    graph.meta["operator_fact"] = "changed after import"
    changed = to_ocp(graph)["ext"]["dev.loopmath.graph"]["meta"]
    assert changed == graph.meta
    assert "n_nodes" not in changed
    assert changed["nodes_missing_tokens"] is None


@pytest.mark.parametrize(
    ("swarm", "source_missing", "reader_missing", "source_size", "loaded_size", "ordinary_size"),
    [
        ("a", 1, 111, 87, 97, 86),
        ("b", 1, 42, 83, 93, 82),
        ("c", 1, 4, 83, 93, 82),
    ],
)
def test_historical_native_reader_private_meta_state_is_exactly_eleven_fields(
    swarm, source_missing, reader_missing, source_size, loaded_size, ordinary_size
):
    admitted = to_ocp(_native_eval_graph(swarm))
    source_meta = admitted["ext"]["dev.loopmath.graph"]["meta"]
    graph = from_ocp(admitted)
    expected_value_keys = {"nodes_missing_tokens"}
    expected_absent_keys = {
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
    }
    missing = object()
    observed_delta = {
        key
        for key in source_meta.keys() | graph.meta.keys()
        if source_meta.get(key, missing) != graph.meta.get(key, missing)
    }

    assert set(_OCP_META_VALUE_KEYS) == expected_value_keys
    assert set(_OCP_META_ABSENT_KEYS) == expected_absent_keys
    assert observed_delta == expected_value_keys | expected_absent_keys
    assert len(source_meta) == source_size
    assert len(graph.meta) == loaded_size
    assert len(set(graph.meta) - observed_delta) == ordinary_size
    assert (source_meta["nodes_missing_tokens"], graph.meta["nodes_missing_tokens"]) == (
        source_missing,
        reader_missing,
    )
    assert graph._ocp_source_meta["values"] == {
        "nodes_missing_tokens": source_missing,
    }
    assert set(graph._ocp_source_meta["absent"]) == expected_absent_keys
    assert len(graph._ocp_source_meta["values"]) + len(graph._ocp_source_meta["absent"]) == 11
    assert graph._ocp_loaded_meta == graph.meta
    assert not any(key.startswith("_ocp_") for key in graph.to_dict())


def test_current_writer_matching_counter_has_exactly_ten_field_delta(tmp_path):
    paths = [tmp_path / "without-cost.jsonl", tmp_path / "partial.jsonl"]
    for path in paths:
        path.write_text("")
    graph = extract(
        [
            {
                "run_id": "without-cost",
                "harness": "codex",
                "session_path": str(paths[0]),
                "workspace": "current-writer",
                "tokens": None,
            },
            {
                "run_id": "partial",
                "harness": "codex",
                "session_path": str(paths[1]),
                "workspace": "current-writer",
                "tokens": {
                    "in": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                    "cache_write_5m": 0,
                    "cache_write_1h": None,
                    "out": 0,
                },
            },
        ]
    )
    admitted = to_ocp(graph)
    source_meta = admitted["ext"]["dev.loopmath.graph"]["meta"]
    loaded = from_ocp(admitted)
    missing = object()
    observed_delta = {
        key
        for key in source_meta.keys() | loaded.meta.keys()
        if source_meta.get(key, missing) != loaded.meta.get(key, missing)
    }

    assert graph.meta["nodes_missing_tokens"] == 2
    assert loaded.meta["nodes_missing_tokens"] == 2
    assert observed_delta == set(_OCP_META_ABSENT_KEYS)
    assert loaded._ocp_source_meta["values"] == {}
    assert set(loaded._ocp_source_meta["absent"]) == set(_OCP_META_ABSENT_KEYS)
    assert len(loaded._ocp_source_meta["absent"]) == 10


def _cost_for_retention(tokens):
    counters = Counter()
    _record, missing = _cost(
        GraphNode("cost", "codex", "codex", "cost.jsonl", tokens=tokens),
        counters,
    )
    return missing, counters


def test_cost_absent_retention_bucket_uses_no_split_evidence():
    absent, counters = _cost_for_retention(
        {"in": 1, "cache_read": 2, "cache_write": 3, "out": 4}
    )
    for field in ("cache_creation_5m_tokens", "cache_creation_1h_tokens"):
        assert "no retention split" in absent[field]
    assert counters["cache_retention_buckets_missing"] == 2


def test_cost_explicit_none_retention_bucket_uses_same_unavailable_evidence():
    base = {"in": 1, "cache_read": 2, "cache_write": 3, "out": 4}
    absent, _absent_counts = _cost_for_retention(dict(base))
    explicit_none, counters = _cost_for_retention(
        {**base, "cache_write_5m": None, "cache_write_1h": None}
    )
    assert explicit_none["cache_creation_5m_tokens"] == absent["cache_creation_5m_tokens"]
    assert explicit_none["cache_creation_1h_tokens"] == absent["cache_creation_1h_tokens"]
    assert counters["cache_retention_buckets_missing"] == 2


def test_cost_malformed_non_none_retention_keeps_invalid_value_evidence():
    malformed, counters = _cost_for_retention(
        {
            "in": 1,
            "cache_read": 2,
            "cache_write": 3,
            "cache_write_5m": "unknown",
            "cache_write_1h": False,
            "out": 4,
        }
    )
    for field in ("cache_creation_5m_tokens", "cache_creation_1h_tokens"):
        assert "not a count" in malformed[field]
        assert "no retention split" not in malformed[field]
    assert counters["cache_retention_buckets_missing"] == 2


def test_all_reader_entrypoints_reject_nonconforming_v02(tmp_path):
    document = _doc("invalid")
    document["attempts"][0]["status"] = "teleporting"
    path = tmp_path / "invalid.ocp.json"
    path.write_text(json.dumps(document))

    for read in (lambda: read_document(path), lambda: from_ocp(document), lambda: records_from_ocp(document)):
        with pytest.raises(OCPError, match=r"conformance.*E010"):
            read()


def test_reader_preserves_distinct_node_ids_when_attempt_sessions_match():
    document = _doc("one")
    document["nodes"].append(
        {"id": "two", "kind": "review", "labels": {"harness": "codex", "source": "codex"}}
    )
    document["attempts"].append(_attempt("two"))
    for attempt in document["attempts"]:
        attempt["session"] = "shared-vendor-session"

    graph = from_ocp(document)

    assert {node.id for node in graph.nodes} == {"one", "two"}
    assert len(graph.nodes) == 2
    assert graph.node("one").session_path == "shared-vendor-session"
    assert graph.node("two").session_path == "shared-vendor-session"


def test_retries_expand_to_attempt_nodes_and_node_edges_choose_latest_attempt():
    first = _attempt("retry", n=1, suffix="a1")
    second = _attempt("retry", n=2, suffix="a2")
    document = _doc("retry", attempts=[first, second])
    document["nodes"].append({"id": "gate", "kind": "gate"})
    document["edges"].append({"from": "retry", "to": "gate", "kind": "dep", "tier": "reported", "evidence": "declared dependency"})
    graph = from_ocp(document)
    assert {node.id for node in graph.nodes} == {"retry.a1", "retry.a2", "gate"}
    assert [(edge.src, edge.dst, edge.kind, edge.tier) for edge in graph.edges] == [
        ("retry.a2", "gate", "dep", "reported")
    ]
    assert "style=solid" in to_dot(graph)


def test_reader_output_can_be_reemitted_and_read_with_same_core_graph():
    original = from_ocp(read_document(SWARM))
    emitted = to_ocp(original)
    rebuilt = from_ocp(emitted)
    assert {node.id for node in rebuilt.nodes} == {node.id for node in original.nodes}
    for node in original.nodes:
        again = rebuilt.node(node.id)
        assert (again.model, again.effort, again.tokens, again.usd, again.session_path) == (
            node.model, node.effort, node.tokens, node.usd, node.session_path
        )
        assert (again.role, again.role_tier, again.phase, again.phase_tier) == (
            node.role, node.role_tier, node.phase, node.phase_tier
        )
    assert sorted((edge.src, edge.dst, edge.kind, edge.tier) for edge in rebuilt.edges) == sorted(
        (edge.src, edge.dst, edge.kind, edge.tier) for edge in original.edges
    )
    assert sorted((artifact.id, artifact.producer, artifact.writers, artifact.consumers) for artifact in rebuilt.artifacts) == sorted(
        (artifact.id, artifact.producer, artifact.writers, artifact.consumers) for artifact in original.artifacts
    )


def _doc_with_artifact(extension) -> dict:
    document = _doc("writer")
    document["artifacts"] = [
        {
            "id": "artifact-1",
            "path": "src/empty.py",
            "kind": {"value": "code", "tier": "heuristic"},
            "producer": "writer.a1",
            "writers": ["writer.a1"],
            "consumers": [],
            "first_write_at": "2026-09-02T20:00:00Z",
            "n_writes": 1,
            "n_reads": 0,
            "ext": {"dev.dagr.artifact": copy.deepcopy(extension)},
        }
    ]
    return document


def test_reader_validates_and_round_trips_artifact_recording_and_arbitrary_meta():
    extension = {
        "bytes": 0,
        "bytes_tier": "verified",
        "lines_added": 3,
        "lines_added_tier": "reported",
        "lines_removed": 0,
        "lines_removed_tier": "verified",
        "language": "py",
        "language_tier": "heuristic",
        "tests_touched": 0,
        "tests_touched_tier": "heuristic",
        "fate": "kept",
        "fate_tier": "reported",
        "meta": {
            "future.consumer.key": {
                "nested": [0, False, None, {"punctuation key": "αλφα"}]
            }
        },
    }
    document = _doc_with_artifact(extension)
    expected_canonical = _canonical(extension)
    graph = from_ocp(document)
    artifact = graph.artifacts[0]
    assert "ocp_projection_extension_namespaces_unmodeled" not in graph.meta
    assert (
        artifact.bytes,
        artifact.lines_added,
        artifact.lines_removed,
        artifact.language,
        artifact.tests_touched,
        artifact.fate,
    ) == (0, 3, 0, "py", 0, "kept")
    assert _canonical(artifact.meta) == _canonical(extension["meta"])
    document["artifacts"][0]["ext"]["dev.dagr.artifact"]["meta"]["future.consumer.key"]["nested"].append("changed")
    assert _canonical(artifact.meta) == _canonical(extension["meta"])

    reemitted = to_ocp(graph)
    round_tripped = reemitted["artifacts"][0]["ext"]["dev.loopmath.artifact"]
    assert _canonical(round_tripped) == expected_canonical
    round_tripped["meta"]["future.consumer.key"]["nested"].append("changed again")
    assert _canonical(artifact.meta) == _canonical(extension["meta"])

    artifact.meta["future.consumer.key"]["nested"].append("graph changed")
    graph_mutation = to_ocp(graph)["artifacts"][0]["ext"]["dev.loopmath.artifact"]
    assert graph_mutation["meta"]["future.consumer.key"]["nested"][-1] == "graph changed"


def test_artifact_projection_distinguishes_explicit_null_from_absence():
    supplied_null = {
        "artifacts": [
            {"id": "a", "ext": {"dev.dagr.artifact": {"bytes": None}}}
        ]
    }
    projected_null = copy.deepcopy(supplied_null)
    assert _extension_loss_counters(supplied_null, projected_null) == {}

    projected_absent = {"artifacts": [{"id": "a"}]}
    assert _extension_loss_counters(supplied_null, projected_absent) == {
        "ocp_projection_dagr_artifact_members_unmodeled": 1
    }
    assert _extension_loss_counters(projected_absent, projected_null) == {}

    foreign = {"artifacts": [{"id": "a", "ext": {"vendor.example": {"x": 1}}}]}
    assert _extension_loss_counters(foreign, {"artifacts": [{"id": "a"}]}) == {
        "ocp_projection_extension_namespaces_unmodeled": 1
    }


def test_reader_counts_a_dagr_artifact_member_lost_by_projection(monkeypatch):
    extension = {
        "bytes": None,
        "bytes_tier": None,
        "lines_added": None,
        "lines_added_tier": None,
        "lines_removed": None,
        "lines_removed_tier": None,
        "language": None,
        "language_tier": None,
        "tests_touched": None,
        "tests_touched_tier": None,
        "fate": "unknown",
        "fate_tier": None,
        "meta": {},
    }
    document = _doc_with_artifact(extension)
    document["artifacts"][0]["id"] = "src/empty.py"

    from loopmath.graph import ocp as graph_ocp

    real_to_ocp = graph_ocp.to_ocp

    def lose_bytes(graph, **kwargs):
        projected = real_to_ocp(graph, **kwargs)
        projected["artifacts"][0]["ext"]["dev.loopmath.artifact"].pop("bytes")
        return projected

    monkeypatch.setattr(graph_ocp, "to_ocp", lose_bytes)
    graph = from_ocp(document)

    assert graph.meta["ocp_projection_dagr_artifact_members_unmodeled"] == 1
    assert "ocp_projection_extension_namespaces_unmodeled" not in graph.meta


@pytest.mark.parametrize(
    ("extension", "message"),
    [
        ({"bytes": True, "bytes_tier": "verified"}, "nonnegative integer"),
        ({"lines_added": -1, "lines_added_tier": "verified"}, "nonnegative integer"),
        ({"tests_touched": 1, "tests_touched_tier": "asserted"}, "must be verified"),
        ({"tests_touched": 1, "tests_touched_tier": []}, "must be verified"),
        ({"lines_removed": 0, "lines_removed_tier": None}, "must be null together"),
        ({"language": None, "language_tier": "heuristic"}, "must be null together"),
        ({"language": "PY", "language_tier": "heuristic"}, "lowercase extension identifier"),
        ({"language": "not.an.extension", "language_tier": "heuristic"}, "lowercase extension identifier"),
        ({"fate": "accepted", "fate_tier": "reported"}, "must be kept"),
        ({"fate": [], "fate_tier": None}, "must be kept"),
        ({"fate": "kept", "fate_tier": {}}, "must be verified"),
        ({"fate": "unknown", "fate_tier": "reported"}, "must use unknown and null together"),
        ({"fate": "deleted", "fate_tier": None}, "must use unknown and null together"),
        ({"meta": []}, "must be an object"),
        ({"future": 1}, "unknown fields"),
    ],
)
def test_reader_rejects_invalid_artifact_recording(extension, message):
    with pytest.raises(OCPError, match=message):
        from_ocp(_doc_with_artifact(extension))


def test_reader_rejects_non_object_artifact_recording_namespace():
    with pytest.raises(OCPError, match="must be an object"):
        from_ocp(_doc_with_artifact([]))


def test_reader_rejects_incomplete_artifact_recording_namespace():
    with pytest.raises(OCPError, match="missing required fields"):
        from_ocp(_doc_with_artifact({"bytes": None}))


def test_generic_swarm_example_loss_surface_is_measured():
    source = read_document(SWARM)
    reemitted = to_ocp(from_ocp(source))

    assert tuple(len(source[entity]) for entity in ("nodes", "edges", "attempts", "artifacts", "events")) == (
        6, 7, 6, 3, 21
    )
    assert tuple(len(reemitted[entity]) for entity in ("nodes", "edges", "attempts", "artifacts", "events")) == (
        6, 7, 6, 3, 12
    )
    assert "labels" not in reemitted["run"]
    assert all(set(attempt["model"]) == {"raw", "tier"} for attempt in reemitted["attempts"])
    assert all("requests" not in attempt.get("cost", {}) for attempt in reemitted["attempts"])
    assert all("reasoning_tokens" not in attempt.get("cost", {}) for attempt in reemitted["attempts"])
    assert all("receipt" not in attempt["outcome"] for attempt in reemitted["attempts"])
    artifact = next(
        artifact for artifact in reemitted["artifacts"] if artifact["id"] == "reviews/dev-A.md"
    )
    assert set(artifact["ext"]) == {"dev.loopmath.artifact"}


@pytest.mark.parametrize(
    ("swarm", "expected_graph_counts", "expected_ocp_counts"),
    [
        ("a", (111, 914, 434), (2, 111, 914, 111, 434, 220)),
        ("b", (42, 319, 120), (2, 42, 319, 42, 120, 82)),
        ("c", (4, 27, 49), (2, 4, 27, 4, 49, 6)),
    ],
)
def test_native_eval_swarm_ocp_round_trip_is_canonically_lossless(
    swarm, expected_graph_counts, expected_ocp_counts
):
    graph = _native_eval_graph(swarm)
    assert (len(graph.nodes), len(graph.edges), len(graph.artifacts)) == expected_graph_counts

    emitted = to_ocp(graph)
    reemitted = to_ocp(from_ocp(emitted))

    assert tuple(
        len(emitted[entity])
        for entity in ("groups", "nodes", "edges", "attempts", "artifacts", "events")
    ) == expected_ocp_counts
    assert _canonical(reemitted) == _canonical(emitted)
    assert all(
        set(record.get("ext", {})) >= {"dev.loopmath.artifact"}
        for record in emitted["artifacts"]
    )
    assert all(
        record["ext"]["dev.loopmath.artifact"]["fate"] == "unknown"
        and record["ext"]["dev.loopmath.artifact"]["fate_tier"] is None
        for record in emitted["artifacts"]
    )
    for entity in ("nodes", "attempts", "artifacts"):
        assert [record["id"] for record in reemitted[entity]] == [
            record["id"] for record in emitted[entity]
        ]


def test_reader_rejects_non_v02_and_ambiguous_artifact_paths():
    with pytest.raises(OCPError, match="only OCP v0.2"):
        from_ocp({"ocp": "0.1"})
    document = _doc("dup")
    attempt_id = document["attempts"][0]["id"]
    artifact = {
        "id": "a",
        "path": "same.txt",
        "kind": {"value": "doc", "tier": "heuristic"},
        "producer": attempt_id,
        "writers": [attempt_id],
        "consumers": [],
        "first_write_at": None,
        "n_writes": 1,
        "n_reads": 0,
    }
    document["artifacts"] = [artifact, {**artifact, "id": "b"}]
    with pytest.raises(OCPError, match="duplicate path"):
        from_ocp(document)


def test_graph_cli_accepts_multiple_ocp_sources_without_native_scan(tmp_path, monkeypatch, capsys):
    paths = []
    for node in ("one", "two"):
        path = tmp_path / f"{node}.json"
        path.write_text(json.dumps(_doc(node)))
        paths.append(path)

    def native_scan_forbidden(*args, **kwargs):
        raise AssertionError("OCP-only graph must not scan native logs")

    monkeypatch.setattr(ingest, "parse_all", native_scan_forbidden)
    rc = main([
        "graph", "--ocp", str(paths[0]), "--ocp", str(paths[1]),
        "--format", "json", "--quiet",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    graph = json.loads(captured.out)
    assert {node["id"] for node in graph["nodes"]} == {"one", "two"}
    assert graph["meta"]["n_nodes"] == 2
    assert "imported 2 OCP document(s)" in captured.err


def test_analyze_ocp_only_is_repeatable_and_does_not_scan_or_reprice(tmp_path, monkeypatch, capsys):
    paths = []
    for node in ("one", "two"):
        path = tmp_path / f"{node}.json"
        path.write_text(json.dumps(_doc(node)))
        paths.append(path)

    monkeypatch.setattr(ingest, "discover", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no native discovery")))
    monkeypatch.setattr(ingest, "parse_all", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no native parse")))
    rc = main([
        "analyze", "--ocp", str(paths[0]), "--ocp", str(paths[1]),
        "--min-n", "1", "--boot", "4", "--quiet",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "2 OCP document(s), 2 attempts imported" in captured.out
    assert "OCP costs from 2 document(s) were preserved as supplied" in captured.out
    assert "gpt-test.high" in captured.out


def test_cli_reports_bad_ocp_as_a_usage_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text('{"ocp": "0.1"}')
    assert main(["graph", "--ocp", str(bad), "--quiet"]) == 2
    assert "only OCP v0.2 is supported" in capsys.readouterr().err
