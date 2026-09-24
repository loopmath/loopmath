"""Static and data-contract tests for the inline graph visualizer."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from loopmath.cli import main
from loopmath.graph.html_common import COMMON_JS
from loopmath.graph.html_css import CSS
from loopmath.graph.html_data import build_html_data
from loopmath.graph.html_render import to_html
from loopmath.graph.render import _EDGE_STYLE
from loopmath.graph.schema import EDGE_KINDS, Artifact, Graph, GraphEdge, GraphNode
from loopmath.graph.token_completeness import TOKEN_STREAMS
from loopmath.ingest.ocp import from_ocp, load_ocp
from tests.graph_html_static_probe import LIMIT, network_findings
from tests.test_ingest_ocp import _native_eval_graph


ROOT = Path(__file__).resolve().parent.parent


def _tokens(**values) -> dict[str, int | None]:
    tokens = {key: (index + 1) * 10 for index, (_, key) in enumerate(TOKEN_STREAMS)}
    tokens.update(values)
    return tokens


def _node(node_id: str = "one", **values) -> GraphNode:
    fields = {
        "id": node_id,
        "harness": "fixture",
        "source": "top",
        "session_path": f"{node_id}.jsonl",
        "model": "fixture-model",
        "model_tier": "verified",
        "effort": "high",
        "workspace": "fixture",
        "ts": "2026-09-03T12:00:00Z",
        "wall_s": 30.0,
        "tokens": _tokens(),
        "usd": 1.25,
        "role": "lead",
        "role_tier": "heuristic",
        "role_evidence": "fixture rule",
        "phase": "build",
        "phase_tier": "heuristic",
    }
    fields.update(values)
    return GraphNode(**fields)


def test_html_is_single_offline_document_with_table_before_three_views():
    page = to_html(Graph(nodes=[_node()], meta={"usd_total": 1.25, "usd_unpriced_nodes": 0}))

    assert page.startswith("<!doctype html>") and page.count("<html") == 1
    assert network_findings(page) == []
    assert page.index("data-attempt-table") < page.index('data-view="swim"')
    assert re.findall(r'<details class="viewsec" data-view="([^"]+)" open>', page) == [
        "swim",
        "force",
        "cost",
    ]
    for label in (
        "attempt",
        "role",
        "model",
        "tier",
        "cost",
        "tokens",
        "duration",
        "status",
        "artifacts written",
        "artifacts read",
    ):
        assert f"label: '{label}'" in page
    assert "th[data-key]" in page and "state.sortDir" in page


def test_one_attempt_without_edges_has_valid_data_and_all_layouts():
    graph = Graph(nodes=[_node()], edges=[], artifacts=[], meta={})
    data = build_html_data(graph)
    page = to_html(graph)

    assert len(data["nodes"]) == 1 and data["edges"] == []
    assert data["run"]["span_s"] == 30.0
    assert "const Swimlanes" in page
    assert "const Force" in page
    assert "const CostCurve" in page
    _check_missing_measurements()
    _check_invalid_measurements()
    _check_unresolved_relationships()


def _check_missing_measurements():
    node = _node(
        wall_s=None,
        tokens=_tokens(cache_read=None),
        phase_tier="reported",
    )

    data = build_html_data(Graph(nodes=[node], meta={}))
    attempt = data["nodes"][0]

    assert attempt["dur"] is None
    assert attempt["tok"] == _tokens(cache_read=None)
    assert data["run"]["token_streams"] == [key for _, key in TOKEN_STREAMS]
    assert attempt["pt"] == "reported"
    assert data["run"]["accounting"]["attempt_durations_unavailable"] == 1
    assert data["run"]["accounting"]["attempts_with_incomplete_token_streams"] == 1
    assert data["run"]["accounting"]["token_stream_values_unavailable"] == 1

    absent = build_html_data(Graph(nodes=[_node(tokens=None)], meta={}))
    assert absent["nodes"][0]["tok"] == {key: None for _, key in TOKEN_STREAMS}
    assert absent["nodes"][0]["tok_record"] is False
    assert absent["run"]["accounting"]["token_stream_values_unavailable"] == len(
        TOKEN_STREAMS
    )


def _check_invalid_measurements():
    node = _node(
        ts="not-a-timestamp",
        wall_s=float("nan"),
        usd=float("inf"),
        tokens=_tokens(**{"in": -1, "cache_write": 10**400}),
        launched_by={"how": "fixture", "lag_s": float("nan")},
    )
    artifact = Artifact(
        id="/tmp/item.txt",
        producer="one",
        writers=["one"],
        first_write_ts="invalid",
        n_writes=10**400,
        bytes=-1,
        lines_added=float("inf"),
        lines_removed=float("nan"),
    )
    edge = GraphEdge("one", "one", "launch", "verified", {"lag_s": float("inf")})
    graph = Graph(
        nodes=[node], edges=[edge], artifacts=[artifact], meta={"usd_total": float("nan")}
    )

    data = build_html_data(graph)
    attempt, item, counts = data["nodes"][0], data["artifacts"][0], data["run"]["accounting"]

    assert attempt["t0"] is None and attempt["dur"] is None and attempt["usd"] is None
    assert attempt["tok"]["in"] is None and attempt["tok"]["cache_write"] is None
    assert attempt["launch"]["lag_s"] is None
    assert item["t"] is None and item["nw"] is None and item["bytes"] is None
    assert item["la"] is None and item["lr"] is None
    assert data["edges"] == [[0, 0, "launch", "verified", None, None]]
    assert data["run"]["span_s"] is None
    assert counts["attempt_timestamps_unavailable"] == 1
    assert counts["attempt_durations_unavailable"] == 1
    assert counts["attempt_costs_unavailable"] == 1
    assert counts["attempt_launch_lags_unavailable"] == 1
    assert counts["launch_edge_lags_unavailable"] == 1
    assert counts["nonfinite_values_replaced_with_null"] == 2
    assert counts["run_spans_padded_for_plot"] == 1
    assert "NaN" not in to_html(graph) and "Infinity" not in to_html(graph)


def test_complete_nested_attempt_mappings_are_preserved():
    spawn = {
        "tool_use_id": "tool-1",
        "agent_type": "general-purpose",
        "spawn_depth": 2,
        "meta_description": "fixture task",
        "description": "fixture task",
        "subagent_type": None,
        "requested_model": "fixture-model",
    }
    launched = {
        "id": "launcher",
        "workspace": "fixture",
        "how": "fixture launch",
        "tier": "reported",
        "evidence": "fixture evidence",
        "lag_s": 2.5,
        "command": "fixture command",
    }
    node = _node(
        parent="launcher",
        role=None,
        role_tier="heuristic",
        spawn=spawn,
        launched_by=launched,
    )

    data = build_html_data(Graph(nodes=[node], meta={}))
    attempt = data["nodes"][0]

    assert attempt["spawn"] == spawn
    assert attempt["launch"] == launched
    assert attempt["parent_source"] == "launcher"
    assert attempt["role_source"] is None
    assert attempt["rt"] == "heuristic"
    assert data["run"]["attempt_field_omissions"] == []


def _check_unresolved_relationships():
    first = _node("one", parent="missing-parent")
    second = _node("two", role="dev", ts="2026-09-03T12:01:00Z")
    artifact = Artifact(
        id="/tmp/item.txt",
        producer="missing-producer",
        writers=["one", "missing-writer"],
        consumers=["two", "missing-consumer"],
        first_write_ts="2026-09-03T12:00:10Z",
        n_writes=1,
        n_reads=1,
        hint="fixture hint",
    )
    edges = [
        GraphEdge("one", "two", "artifact", "verified", {"path": artifact.id, "lag_s": 12}),
        GraphEdge("two", "one", "artifact", "verified", {"path": artifact.id, "lag_s": 13}),
        GraphEdge("missing-node", "two", "spawn", "verified"),
        GraphEdge("", "two", "spawn", "verified"),
        GraphEdge("one", "two", "artifact", "verified", {"path": "/tmp/missing.txt"}),
        GraphEdge("one", "two", "artifact", "verified", {}),
        GraphEdge("one", "two", "other", "verified"),
    ]

    data = build_html_data(Graph(nodes=[first, second], edges=edges, artifacts=[artifact]))
    attempt = data["nodes"][0]
    item = data["artifacts"][0]
    counts = data["run"]["accounting"]

    assert data["edges"] == [
        [0, 1, "artifact", "verified", 0, 12.0],
        [1, 0, "artifact", "verified", 0, 13.0],
    ]
    assert data["run"]["n_source_edges"] == 7
    assert attempt["parent"] is None
    assert attempt["parent_ref"] == "missing-parent"
    assert item["prod"] is None
    assert item["pm"] == {"id": "missing-producer", "reason": "outside_graph"}
    assert item["w"] == [0] and item["c"] == [1]
    assert item["wm"] == [{"id": "missing-writer", "reason": "outside_graph"}]
    assert item["cm"] == [{"id": "missing-consumer", "reason": "outside_graph"}]
    assert item["hint"] == "fixture hint"
    assert item["we"] == [[0]] and item["ce"] == [[0]]
    assert counts["attempt_parent_refs_outside_graph"] == 1
    assert counts["artifact_producer_refs_outside_graph"] == 1
    assert counts["artifact_writer_refs_outside_graph"] == 1
    assert counts["artifact_consumer_refs_outside_graph"] == 1
    assert counts["edges_excluded_outside_graph"] == 1
    assert counts["edges_excluded_unavailable"] == 1
    assert counts["edges_excluded_unknown_kind"] == 1
    assert counts["edges_excluded"] == 5
    assert counts["artifact_edges_excluded_missing_path"] == 1
    assert counts["artifact_edges_excluded_unmatched_path"] == 1
    assert counts["artifact_edges_with_unlisted_writers"] == 1
    assert counts["artifact_edges_with_unlisted_consumers"] == 1


def test_artifact_lag_is_recovered_from_ocp_evidence():
    writer = _node("writer", role="dev")
    reader = _node("reader", role="reviewer", ts="2026-09-03T12:01:00Z")
    artifact = Artifact(
        id="/tmp/item.txt",
        producer="writer",
        writers=["writer"],
        consumers=["reader"],
        first_write_ts="2026-09-03T12:00:10Z",
        n_writes=1,
        n_reads=1,
    )
    edge = GraphEdge(
        "writer",
        "reader",
        "artifact",
        "verified",
        {"path": artifact.id, "evidence": "write (verified) then read (verified) 42.5 s later"},
    )

    data = build_html_data(Graph(nodes=[writer, reader], edges=[edge], artifacts=[artifact]))
    assert data["edges"] == [[0, 1, "artifact", "verified", 0, 42.5]]


def test_embedded_data_cannot_close_its_script():
    graph = Graph(nodes=[_node(spawn={"description": "</script><b>bad</b>"})])
    page = to_html(graph)

    assert "</script><b>bad</b>" not in page
    assert "\\\\u003c/script\\\\u003e" in page


def test_ocp_attempt_status_is_unavailable_and_accounted():
    graph = load_ocp(ROOT / "spec" / "examples" / "golden" / "minimal-run.ocp.json")
    data = build_html_data(graph)
    page = to_html(graph)

    assert data["nodes"][0]["status"] is None
    assert data["run"]["accounting"]["attempt_statuses_unavailable"] == 1
    assert "session parsed to its end" not in page
    assert "Every status is settled_unverified" not in page
    assert "status unavailable; the graph model carries no acceptance signal" in page
    assert "Attempt status is unavailable because the graph model carries no acceptance signal" in page


def test_embedded_data_preserves_special_mapping_key_as_own_property():
    hostile = "</script><script>Object.prototype.viewerChanged = true</script>\u2028\u2029&"
    page = to_html(Graph(nodes=[_node(spawn={"__proto__": hostile})]))
    match = re.search(r"<script>(const DATA = .*?;)</script>", page)
    assert match is not None
    probe = f"""{match.group(1)}
const mapping = DATA.nodes[0].spawn;
process.stdout.write(JSON.stringify({{
  own: Object.prototype.hasOwnProperty.call(mapping, '__proto__'),
  value: mapping['__proto__'],
  mappingPrototypeIntact: Object.getPrototypeOf(mapping) === Object.prototype,
  pagePrototypeIntact: !Object.prototype.hasOwnProperty.call(Object.prototype, 'viewerChanged'),
}}));
"""
    result = subprocess.run(["node", "-e", probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "own": True,
        "value": hostile,
        "mappingPrototypeIntact": True,
        "pagePrototypeIntact": True,
    }


def test_corrupt_ocp_reports_an_error_and_nonzero_exit(tmp_path, capsys):
    source = tmp_path / "corrupt.ocp.json"
    source.write_text("{broken", encoding="utf-8")

    result = main(["graph", "--ocp", str(source), "--format", "html", "--quiet"])
    captured = capsys.readouterr()

    assert result != 0
    assert captured.out == ""
    assert "error:" in captured.err
    assert str(source) in captured.err


def _page_data(page: str) -> dict:
    """The payload the page carries, decoded the way the browser decodes it."""
    match = re.search(r"<script>const DATA = JSON\.parse\((.*?)\);</script>", page, re.S)
    assert match is not None
    return json.loads(json.loads(match.group(1)))


def _scheduling_document() -> dict:
    """One OCP document with exactly one `dep` edge and one `fan_in` edge."""

    def attempt(node: str, minute: int) -> dict:
        return {
            "id": f"{node}.a1",
            "node": node,
            "status": "done",
            "outcome": {"result": "done", "evidence": "verified"},
            "started_at": f"2026-09-03T12:0{minute}:00Z",
            "ended_at": f"2026-09-03T12:0{minute}:30Z",
        }

    return {
        "ocp": "0.2",
        "producer": {"name": "fixture", "version": "1"},
        "privacy": {"profile": "metadata_only"},
        "run": {"id": "scheduling-fixture"},
        "nodes": [
            {"id": "a-build", "kind": "impl", "state": "done"},
            {"id": "b-review", "kind": "review", "state": "done"},
            {"id": "c-gate", "kind": "gate", "state": "done"},
        ],
        "attempts": [attempt("a-build", 0), attempt("b-review", 2), attempt("c-gate", 4)],
        "edges": [
            {
                "from": "a-build",
                "to": "b-review",
                "kind": "dep",
                "tier": "reported",
                "evidence": "declared dependency",
            },
            {
                "from": "b-review",
                "to": "c-gate",
                "kind": "fan_in",
                "tier": "reported",
                "evidence": "declared gate input",
            },
        ],
    }


def test_scheduling_edges_are_read_carried_into_the_page_and_never_excluded():
    """`dep` and `fan_in` are OCP core kinds, so the reader keeps them, the
    viewer payload carries them, and no counter calls them an unknown kind."""
    graph = from_ocp(_scheduling_document())
    data = build_html_data(graph)
    page = to_html(graph)
    counts = data["run"]["accounting"]

    assert [(edge.src, edge.dst, edge.kind, edge.tier) for edge in graph.edges] == [
        ("a-build", "b-review", "dep", "reported"),
        ("b-review", "c-gate", "fan_in", "reported"),
    ]
    assert data["edges"] == [
        [0, 1, "dep", "reported", None, None],
        [1, 2, "fan_in", "reported", None, None],
    ]
    assert counts.get("edges_excluded_unknown_kind", 0) == 0
    assert counts.get("edges_excluded", 0) == 0
    assert data["run"]["n_source_edges"] == 2
    assert data["run"]["edges_by_kind_tier"] == {"dep/reported": 1, "fan_in/reported": 1}
    assert _page_data(page)["edges"] == data["edges"]


def test_every_edge_kind_has_one_line_style_named_in_the_legend():
    """The vocabulary lives once, in `EDGE_KINDS`. The legend iterates the copy
    the payload carries, and the styles agree with it rather than restating it."""
    legend_block = re.search(r"const EDGE_LEGEND = \{(.*?)\n  \};", COMMON_JS, re.S)
    named = tuple(re.findall(r"^    (\w+): \{", legend_block.group(1), re.M))
    styled = tuple(re.findall(r"^\.e-([a-z_]+) \{", CSS, re.M))
    page = to_html(from_ocp(_scheduling_document()))

    assert named == EDGE_KINDS
    assert sorted(styled) == sorted(EDGE_KINDS)
    assert tuple(_EDGE_STYLE) == EDGE_KINDS  # the DOT renderer's per-kind styles
    assert "RUN.edge_kinds.forEach" in COMMON_JS
    assert _page_data(page)["run"]["edge_kinds"] == list(EDGE_KINDS)
    assert "dep, dotted: the target cannot start before the source settles" in page
    assert "fan_in, dash-dot: the source is an acceptance input of the target gate" in page
    assert ".e-dep { stroke: #7a6a9c; stroke-width: 1.2; stroke-dasharray: 1 3;" in page
    assert ".e-fan_in { stroke: #7a6a9c; stroke-width: 1.2; stroke-dasharray: 7 2 1 2;" in page
    # Swimlanes route a scheduling edge from the end of the source box; the force
    # view draws every edge that is not a hidden handoff, so it needs no branch.
    assert "edge[2] === 'dep' || edge[2] === 'fan_in'" in page
    assert "dep or fan_in, runs from the end of the source box" in page
    assert "the declared scheduling edges dep and fan_in are drawn but do not pull" in page


def test_network_scan_reads_live_markup_and_not_attempt_text(capsys, tmp_path):
    """The offline scan is about markup, not about words in a transcript.

    The adversarial fixture's attempt text names all three network APIs and the
    page stays clean; the same page with one external script tag planted in it
    fails, in the probe as well as here."""
    adversarial = _node(
        spawn={"description": "replaced the fetch( call, dropped XMLHttpRequest"},
        role_evidence="reviewer note: the WebSocket fallback stays out",
    )
    page = to_html(Graph(nodes=[adversarial], meta={}))
    planted = page.replace(
        "</body>", '<script src="https://example.com/x.js"></script></body>', 1
    )
    clean_page, dirty_page = tmp_path / "clean.html", tmp_path / "planted.html"
    clean_page.write_text(page, encoding="utf-8")
    dirty_page.write_text(planted, encoding="utf-8")
    probe = ROOT / "tests" / "graph_html_static_probe.py"
    clean_run = subprocess.run(
        [sys.executable, str(probe), str(clean_page)], capture_output=True, text=True
    )
    dirty_run = subprocess.run(
        [sys.executable, str(probe), str(dirty_page)], capture_output=True, text=True
    )
    with capsys.disabled():
        print("\n  adversarial fixture: " + (clean_run.stdout.splitlines() or [""])[0])
        print("  planted script fixture: " + (dirty_run.stdout.splitlines() or [""])[0])

    # The fixture is adversarial only if the words really do reach the page.
    assert all(token in page for token in ("fetch(", "XMLHttpRequest", "WebSocket"))
    assert network_findings(page) == []
    assert network_findings(planted) == [
        "<script> with src https://example.com/x.js",
        "absolute URL in <script> src: https://example.com/x.js",
    ]
    assert clean_run.returncode == 0
    assert "PASS offline content: no live element or attribute" in clean_run.stdout
    assert dirty_run.returncode == 1
    assert "FAIL offline content: <script> with src https://example.com/x.js" in dirty_run.stdout


def test_swarm_a_page_is_offline_and_stays_under_the_size_limit():
    page = to_html(_native_eval_graph("a"))

    assert network_findings(page) == []
    assert len(page.encode("utf-8")) < LIMIT
