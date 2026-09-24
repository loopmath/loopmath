"""N1B regression coverage for full and metadata-only run export."""

from __future__ import annotations

import json

from loopmath.graph import to_ocp
from loopmath.graph.ocp import EXT_KEY, RUN_EXT_KEY
from loopmath.graph.runfile import to_runfile
from loopmath.graph.schema import Artifact, Graph, GraphEdge, GraphNode


LAUNCH_COMMAND = "codex exec --full-auto 'audit the run exporter'"


def _capable_graph(*, attempt_command: bool = True, edge_command: bool = True) -> Graph:
    """A local graph with a described subagent and a linked CLI launch."""
    nodes = [
        GraphNode(
            id="lead",
            harness="claude-code",
            source="top",
            session_path="/fixture/lead.jsonl",
            workspace="/fixture/workspace",
            ts="2026-09-03T08:00:00Z",
            wall_s=120.0,
            model="claude-opus-5",
            model_tier="verified",
            effort="high",
            tokens={
                "in": 100,
                "cache_read": 20,
                "cache_write": 30,
                "cache_write_5m": 10,
                "cache_write_1h": 20,
                "out": 50,
            },
            usd=1.25,
            role="lead",
            role_tier="heuristic",
            role_evidence="top-level session",
            phase="build",
            phase_tier="heuristic",
        ),
        GraphNode(
            id="planner",
            harness="claude-code",
            source="subagent",
            session_path="/fixture/planner.jsonl",
            workspace="/fixture/workspace",
            ts="2026-09-03T08:00:10Z",
            wall_s=30.0,
            parent="lead",
            spawn={"description": "Map every exported field", "subagent_type": "Plan"},
            role="planner",
            role_tier="reported",
            role_evidence="declared type 'Plan'",
            phase="build",
            phase_tier="heuristic",
        ),
        GraphNode(
            id="reviewer",
            harness="codex",
            source="codex",
            session_path="/fixture/reviewer.jsonl",
            workspace="/fixture/workspace",
            ts="2026-09-03T08:01:05Z",
            wall_s=45.0,
            parent="lead",
            launched_by={
                "id": "lead",
                "workspace": "/fixture/workspace",
                "how": "Bash launch",
                "lag_s": 5.0,
                **({"command": LAUNCH_COMMAND} if attempt_command else {}),
            },
            role="reviewer",
            role_tier="heuristic",
            role_evidence="launch command requests an audit",
            phase="post",
            phase_tier="heuristic",
        ),
    ]
    edges = [
        GraphEdge("lead", "planner", "spawn", "verified", {"how": "Task call"}),
        GraphEdge(
            "lead",
            "reviewer",
            "launch",
            "heuristic",
            {
                "how": "Bash launch",
                "lag_s": 5.0,
                **({"command": LAUNCH_COMMAND} if edge_command else {}),
            },
        ),
        GraphEdge(
            "planner",
            "reviewer",
            "artifact",
            "verified",
            {
                "path": "/fixture/workspace/plan.py",
                "lag_s": 25.0,
                "write_tier": "verified",
                "read_tier": "verified",
            },
        ),
    ]
    artifacts = [
        Artifact(
            id="/fixture/workspace/plan.py",
            producer="planner",
            writers=["planner"],
            consumers=["reviewer"],
            first_write_ts="2026-09-03T08:00:40Z",
            n_writes=1,
            n_reads=1,
            kind="source",
            kind_tier="heuristic",
            hint="def exported_field():",
            writes=[
                {
                    "node": "planner",
                    "ts": "2026-09-03T08:00:40Z",
                    "tier": "verified",
                    "how": "Write",
                }
            ],
            bytes=321,
            bytes_tier="verified",
            lines_added=12,
            lines_added_tier="reported",
            lines_removed=2,
            lines_removed_tier="verified",
            language="py",
            language_tier="heuristic",
            tests_touched=1,
            tests_touched_tier="reported",
            fate="edited",
            fate_tier="verified",
            meta={"reviewed": True},
        )
    ]
    return Graph(
        nodes=nodes,
        edges=edges,
        artifacts=artifacts,
        meta={"fixture": "N1B A6", "nested": {"complete": True}},
    )


def _task(document: dict, task_id: str) -> dict:
    return next(task for task in document["tasks"] if task["id"] == task_id)


def _attempt(document: dict, node_id: str) -> dict:
    return next(attempt for attempt in document["attempts"] if attempt["node"] == node_id)


def _documents(*, attempt_command: bool, edge_command: bool, privacy: str) -> tuple[dict, dict]:
    graph = _capable_graph(attempt_command=attempt_command, edge_command=edge_command)
    ocp = to_ocp(graph, privacy=privacy)
    run = to_runfile(
        graph,
        ocp=ocp,
        finish={},
        generated_at="2026-09-03T09:00:00Z",
    )
    return ocp, run


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value), set())
    return set()


def _launch_edge(document: dict) -> dict:
    return next(edge for edge in document["edges"] if edge["kind"] == "launch")


def _launch_dep(document: dict) -> dict:
    return next(dep for dep in _task(document, "reviewer")["ext"][RUN_EXT_KEY]["deps"] if dep["kind"] == "launch")


def test_full_run_keeps_attempt_and_edge_commands_without_metadata_leak():
    attempt_ocp, attempt_run = _documents(
        attempt_command=True,
        edge_command=False,
        privacy="full",
    )
    assert _attempt(attempt_ocp, "reviewer")["ext"][EXT_KEY]["launch_command"] == LAUNCH_COMMAND
    assert _task(attempt_run, "reviewer")["ext"][RUN_EXT_KEY]["launch_command"] == LAUNCH_COMMAND
    assert json.dumps(attempt_ocp).count(LAUNCH_COMMAND) == 1
    assert json.dumps(attempt_run).count(LAUNCH_COMMAND) == 1

    edge_ocp, edge_run = _documents(
        attempt_command=False,
        edge_command=True,
        privacy="full",
    )
    assert _launch_edge(edge_ocp)["ext"][EXT_KEY]["command"] == LAUNCH_COMMAND
    assert _launch_dep(edge_run)["ext"][RUN_EXT_KEY]["command"] == LAUNCH_COMMAND
    assert json.dumps(edge_ocp).count(LAUNCH_COMMAND) == 1
    assert json.dumps(edge_run).count(LAUNCH_COMMAND) == 1

    both_ocp, both_run = _documents(
        attempt_command=True,
        edge_command=True,
        privacy="full",
    )
    assert json.dumps(both_ocp).count(LAUNCH_COMMAND) == 2
    assert json.dumps(both_run).count(LAUNCH_COMMAND) == 2
    lead_attempt = _attempt(both_ocp, "lead")
    assert lead_attempt["model"] == {"raw": "claude-opus-5", "tier": "verified"}
    assert lead_attempt["effort"] == "high"
    assert set(lead_attempt["cost"]) == {
        "input_tokens",
        "cached_input_tokens",
        "cache_creation_tokens",
        "cache_creation_5m_tokens",
        "cache_creation_1h_tokens",
        "output_tokens",
        "basis",
        "usd",
    }
    planner = _task(both_run, "planner")
    assert planner["title"] == "planner: Map every exported field"
    assert planner["ext"][RUN_EXT_KEY]["role"]["evidence"] == "declared type 'Plan'"
    artifact = both_ocp["artifacts"][0]
    assert artifact["first_write_at"] == "2026-09-03T08:00:40.000Z"
    assert artifact["n_writes"] == artifact["n_reads"] == 1
    assert all(value is not None for value in artifact["ext"]["dev.loopmath.artifact"].values())
    assert both_run["ext"][RUN_EXT_KEY]["artifacts"] == both_ocp["artifacts"]
    assert both_run["ext"][RUN_EXT_KEY]["artifact_edges"] == [
        edge for edge in both_ocp["edges"] if edge["kind"] == "artifact"
    ]

    metadata_ocp, metadata_run = _documents(
        attempt_command=True,
        edge_command=True,
        privacy="metadata_only",
    )
    assert metadata_ocp["privacy"]["profile"] == "metadata_only"
    assert metadata_run["ext"][RUN_EXT_KEY]["privacy"]["profile"] == "metadata_only"
    assert LAUNCH_COMMAND not in json.dumps({"ocp": metadata_ocp, "run": metadata_run})
    assert {"launch_command", "command"}.isdisjoint(_keys(metadata_ocp))
    assert {"launch_command", "command"}.isdisjoint(_keys(metadata_run))
