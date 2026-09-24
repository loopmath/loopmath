"""`graph/runfile.py`: the workflow graph as a herdr-dagr contract v3 run file (spec section 5, P3).

The skeleton fixture graph (tests/fixtures/graph/skeleton) supplies a lead with
spawned subagents, launched codex sessions, an external launcher with no start
time of its own, and unlabeled sessions; a hand-built graph pins the cases the
fixture does not reach (a dependency that would close a cycle). Every assertion
names a value the exporter must produce; when the herdr-dagr binary is on this
machine the documents are also run through `dagr check --strict`.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from loopmath import grade, ingest, price
from loopmath.cli import main
from loopmath.graph import extract, to_ocp
from loopmath.graph.runfile import COUNTERS, HARNESS_WITHOUT_MARKER, codex_thread_files, read_finish_markers, to_runfile
from loopmath.graph.schema import Graph, GraphEdge, GraphNode
from tests.test_cli_graph import COVERAGE, DIAG, PRICE_WARNINGS, _git_tree
from tests.test_graph_extract import ALPHA, skeleton_records

# The vocabulary the graph once withheld; the filter was lifted in 0.1.0 (Q3).
LIFTED_TERMS = ("knowledge gradient", "posterior", "prior", "experimental design", "value of information", "bandit", "arms")

# The herdr dagr checker, a separate program (not this package): the loopmath rename must not
# touch its name, or these strict checks skip for good.
DAGR_BIN = Path(os.environ.get("DAGR_BIN") or Path.home() / ".local" / "bin" / "dagr")

# Grading signals as `ingest.parse_all` attaches them under `_signals`: the lead's
# last turn ended normally (a turn-level fact, not a session-level one), dev's
# last turn hit max_tokens, the codex review was interrupted, plan's log has no
# usable marker; every other node has no entry.
SIGNALS = {
    "lead": {"exit_ok": True, "timed_out": False},
    "dev": {"exit_ok": False, "timed_out": False},
    "codex-review": {"exit_ok": True, "timed_out": True},
    "plan": {"exit_ok": None, "timed_out": False},
}
CLAUDE_NO_MARKER = HARNESS_WITHOUT_MARKER["claude-code"]
NO_ACCEPTANCE = "; the graph carries no acceptance signal"

# One codex rollout that ends on the harness's task_complete event (the shape
# 368 of 400 sampled real rollouts end on), and one that ends mid-turn.
CODEX_HEAD = [
    {"timestamp": "2026-08-31T10:00:00Z", "type": "session_meta", "payload": {"id": "thread-1", "timestamp": "2026-08-31T10:00:00Z", "cwd": "/w", "originator": "codex_exec", "cli_version": "0.40.0", "source": "exec"}},
    {"timestamp": "2026-08-31T10:00:01Z", "type": "event_msg", "payload": {"type": "task_started", "turn_id": "t1"}},
    {"timestamp": "2026-08-31T10:00:02Z", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}},
]
TASK_COMPLETE = {"timestamp": "2026-08-31T10:00:03Z", "type": "event_msg", "payload": {"type": "task_complete", "turn_id": "t1", "last_agent_message": "ok"}}


def _write_jsonl(path: Path, records: list, trailer: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + trailer)
    return path


def _two_node_graph(codex_path: Path, claude_path: Path = Path("/s/claude.jsonl")) -> Graph:
    nodes = [
        GraphNode(id="cx", harness="codex", source="codex", session_path=str(codex_path), workspace="w", ts="2026-08-31T10:00:00Z", wall_s=3.0, model="gpt-5", model_tier="reported", effort="high", role="reviewer", role_tier="heuristic"),
        GraphNode(id="cc", harness="claude-code", source="top", session_path=str(claude_path), workspace="w", ts="2026-08-31T10:00:10Z", wall_s=50.0, model="claude-opus-5", model_tier="verified"),
    ]
    return Graph(nodes=nodes, edges=[], artifacts=[], meta={})


def strict_check(doc: dict, path: Path) -> list:
    """Findings from `dagr check --strict --json`, or a skip when the binary is absent."""
    if not DAGR_BIN.exists():
        pytest.skip(f"herdr-dagr binary not found at {DAGR_BIN}")
    path.write_text(json.dumps(doc, ensure_ascii=False))
    r = subprocess.run([str(DAGR_BIN), "check", str(path), "--strict", "--json"], capture_output=True, text=True, timeout=60)
    assert r.returncode in (0, 1), r.stderr
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def g() -> Graph:
    return extract(skeleton_records(), workspaces=[ALPHA])


@pytest.fixture(scope="module")
def run(g) -> dict:
    return to_runfile(g, signals=SIGNALS)


def _task(run, task_id):
    return next(t for t in run["tasks"] if t["id"] == task_id)


def test_document_is_contract_v3_and_strict_clean(run, tmp_path):
    assert run["dagr"] == 3
    assert run["run"] == {"id": "loopmath-graph:/ws/alpha", "title": "workflow graph of /ws/alpha", "started_at": "2026-08-31T10:00:00.000Z"}
    assert run["projects"] == [{"id": "/ws/alpha", "title": "workspace /ws/alpha"}, {"id": "/ws/other", "title": "workspace /ws/other"}]
    assert strict_check(run, tmp_path / "skeleton.run.json") == []


def test_one_task_and_one_attempt_per_node_with_kind_project_and_deps(g, run):
    assert [t["id"] for t in run["tasks"]] == sorted(n.id for n in g.nodes)
    for t in run["tasks"]:
        assert len(t["attempts"]) == 1
        a = t["attempts"][0]
        assert a["id"] == f"{t['id']}.a1" and a["n"] == 1 and a["cause"] == {"type": "initial"}
        assert a["state"] == t["state"] == a["outcome"]["result"]
        assert t["state"] in ("done", "settled_unverified")
    assert _task(run, "dev")["kind"] == "impl" and _task(run, "plan")["kind"] == "plan" and _task(run, "codex-review")["kind"] == "review"
    assert _task(run, "lead")["kind"] == "ops" and _task(run, "reader")["kind"] == "unknown"
    assert _task(run, "dev")["project"] == ALPHA and _task(run, "analyst")["project"] == "/ws/other"
    assert _task(run, "dev")["title"] == "dev subagent (claude-code)" and _task(run, "dev")["owner"] == "dev"
    # deps are the spawn and launch parents only; artifact edges are handoffs, not dependencies.
    assert _task(run, "dev")["deps"] == ["lead"] and _task(run, "plan")["deps"] == ["lead"]
    assert _task(run, "codex-review")["deps"] == ["lead"] and _task(run, "codex-audit")["deps"] == ["analyst"]
    assert _task(run, "reader")["deps"] == [] and _task(run, "lead")["deps"] == []
    assert _task(run, "dev")["ext"]["dev.dagr.graph"]["deps"] == [{"task": "lead", "kind": "spawn", "tier": "verified"}]
    assert _task(run, "codex-review")["ext"]["dev.dagr.graph"]["deps"] == [{"task": "lead", "kind": "launch", "tier": "heuristic"}]


def test_no_session_is_done_without_a_session_level_clean_finish_marker(run):
    """The skeleton's claude-code logs have no session-end record (none exist for
    that harness) and its codex rollouts end on an assistant message, not on
    task_complete: every task is settled_unverified, each with the reason, and
    lead's exit_ok=True (a turn-level fact) promotes nothing."""
    assert {t["state"] for t in run["tasks"]} == {"settled_unverified"}
    lead = _task(run, "lead")["attempts"][0]
    assert lead["outcome"] == {"result": "settled_unverified", "evidence": "heuristic", "reason": CLAUDE_NO_MARKER + NO_ACCEPTANCE}
    assert _task(run, "dev")["attempts"][0]["outcome"]["reason"] == CLAUDE_NO_MARKER + NO_ACCEPTANCE
    review = _task(run, "codex-review")["attempts"][0]
    assert review["outcome"]["reason"] == "the rollout's last record is response_item/message, not the harness's task_complete event" + NO_ACCEPTANCE
    assert _task(run, "codex-review")["ext"]["dev.dagr.graph"]["finish"] == {"harness": "codex", "marker": False, "last_record": "response_item/message", "evidence": "the rollout's last record is response_item/message, not the harness's task_complete event"}
    assert _task(run, "lead")["ext"]["dev.dagr.graph"]["finish"] == {"harness": "claude-code", "marker": None, "last_record": None, "evidence": CLAUDE_NO_MARKER}
    ext = run["ext"]["dev.dagr.graph"]
    assert ext["harnesses_without_finish_marker"] == {"claude-code": {"sessions": 7, "reason": CLAUDE_NO_MARKER}}
    ex = ext["exporter"]
    assert ex["tasks_done_clean_finish"] == 0 and ex["tasks_settled_unverified_harness_without_finish_marker"] == 7
    assert ex["tasks_settled_unverified_no_finish_marker"] == 3 and ex["tasks_settled_unverified_no_finish_signal"] == 0 and ex["tasks_settled_unverified_unclean_finish"] == 0


def test_codex_done_verified_only_when_the_rollout_ends_on_task_complete_and_nothing_vetoes(tmp_path):
    clean = _write_jsonl(tmp_path / "clean.jsonl", CODEX_HEAD + [TASK_COMPLETE])
    g = _two_node_graph(clean)
    ok = {"cx": {"exit_ok": True, "timed_out": False}, "cc": {"exit_ok": True, "timed_out": False}}
    run = to_runfile(g, signals=ok)
    cx = _task(run, "cx")["attempts"][0]
    assert cx["state"] == "done"
    assert cx["outcome"] == {"result": "done", "evidence": "verified", "receipt": "harness log: the rollout's last record is the harness's task_complete event; the final task_started has a matching task_complete and no turn was aborted"}
    assert _task(run, "cx")["ext"]["dev.dagr.graph"]["finish"] == {"harness": "codex", "marker": True, "last_record": "event_msg/task_complete", "evidence": "the rollout's last record is the harness's task_complete event"}
    # The claude-code node with the same signals stays settled_unverified: its harness has no marker.
    cc = _task(run, "cc")["attempts"][0]
    assert cc["state"] == "settled_unverified" and cc["outcome"]["reason"] == CLAUDE_NO_MARKER + NO_ACCEPTANCE
    assert run["ext"]["dev.dagr.graph"]["harnesses_without_finish_marker"] == {"claude-code": {"sessions": 1, "reason": CLAUDE_NO_MARKER}}
    assert run["ext"]["dev.dagr.graph"]["exporter"]["tasks_done_clean_finish"] == 1
    assert strict_check(run, tmp_path / "clean.run.json") == []

    def reason(signals=None, finish=None):
        return _task(to_runfile(g, signals=signals, finish=finish), "cx")["attempts"][0]["outcome"]["reason"]

    # The ingest's signals veto: an aborted turn, an unmatched final task, no task at all.
    assert reason({"cx": {"exit_ok": True, "timed_out": True}}) == "a turn was aborted (timeout or interruption); the log ends on task_complete but that is not a clean finish, and the graph carries no acceptance signal"
    assert reason({"cx": {"exit_ok": False, "timed_out": False}}) == "the final task_started has no matching task_complete; the log ends on task_complete but that is not a clean finish, and the graph carries no acceptance signal"
    assert reason({"cx": {"exit_ok": None, "timed_out": False}}) == "the rollout has no task_started at all; the log ends on task_complete but that is not a clean finish, and the graph carries no acceptance signal"
    # No signals, or markers not read: unknown, never clean.
    assert reason() == "no grading signals supplied for this session, so the ingest's final-task pairing and abort flag are unknown; the graph carries no acceptance signal"
    assert reason(ok, finish={}) == "finish markers were not read for this session; the graph carries no acceptance signal"
    ex = to_runfile(g, signals=ok, finish={})["ext"]["dev.dagr.graph"]["exporter"]
    assert ex["tasks_settled_unverified_finish_markers_not_read"] == 1
    assert ex["tasks_settled_unverified_no_finish_signal"] == 0 and ex["tasks_done_clean_finish"] == 0
    # The same rollout ending mid-turn, on a truncated line, or unreadable: named, not promoted.
    midturn = _write_jsonl(tmp_path / "midturn.jsonl", CODEX_HEAD)
    r = to_runfile(_two_node_graph(midturn), signals=ok)
    assert _task(r, "cx")["attempts"][0]["outcome"]["reason"] == "the rollout's last record is response_item/message, not the harness's task_complete event" + NO_ACCEPTANCE
    assert r["ext"]["dev.dagr.graph"]["exporter"]["tasks_settled_unverified_no_finish_marker"] == 1
    truncated = _write_jsonl(tmp_path / "truncated.jsonl", CODEX_HEAD + [TASK_COMPLETE], trailer='{"timestamp":"2026-08-31T10:00:04Z","type":"event_msg","payload":{"ty')
    assert _task(to_runfile(_two_node_graph(truncated), signals=ok), "cx")["attempts"][0]["outcome"]["reason"] == "the last line of the session file is not a JSON record (truncated write?)" + NO_ACCEPTANCE
    missing = _task(to_runfile(_two_node_graph(tmp_path / "absent.jsonl"), signals=ok), "cx")["attempts"][0]
    assert missing["state"] == "settled_unverified" and missing["outcome"]["reason"] == "session file not readable (FileNotFoundError)" + NO_ACCEPTANCE


def test_final_jsonl_record_larger_than_tail_chunk_is_read_completely(tmp_path):
    huge = {**TASK_COMPLETE, "padding": "x" * 70_000}
    path = _write_jsonl(tmp_path / "large-final-record.jsonl", CODEX_HEAD + [huge])
    g = _two_node_graph(path)
    finish = read_finish_markers(g)
    assert finish["cx"]["marker"] is True
    run = to_runfile(g, signals={"cx": {"exit_ok": True, "timed_out": False}}, finish=finish)
    assert _task(run, "cx")["state"] == "done"


def test_resumed_codex_thread_is_read_to_its_last_file(tmp_path):
    """A resumed thread spans files with the same session_meta id; the ingest keeps
    the first as the session path, the session's end is in the last."""
    first = _write_jsonl(tmp_path / "rollout-1.jsonl", CODEX_HEAD)
    second = _write_jsonl(tmp_path / "rollout-2.jsonl", CODEX_HEAD + [TASK_COMPLETE])
    other = _write_jsonl(tmp_path / "rollout-3.jsonl", [{**CODEX_HEAD[0], "payload": {**CODEX_HEAD[0]["payload"], "id": "thread-2"}}])
    no_meta = _write_jsonl(tmp_path / "rollout-4.jsonl", CODEX_HEAD[1:])
    files = codex_thread_files([first, second, other, no_meta])
    assert files == {str(first): [str(first), str(second)], str(other): [str(other)], str(no_meta): [str(no_meta)]}
    g = _two_node_graph(first)
    assert read_finish_markers(g)["cx"]["marker"] is False  # the first file alone ends mid-turn
    finish = read_finish_markers(g, session_files=files)
    assert finish["cx"]["marker"] is True and finish["cx"]["last_record"] == "event_msg/task_complete"
    run = to_runfile(g, signals={"cx": {"exit_ok": True, "timed_out": False}}, finish=finish)
    assert _task(run, "cx")["state"] == "done"


def test_timestamps_model_chip_and_events(g, run):
    lead = _task(run, "lead")["attempts"][0]
    assert lead["started_at"] == "2026-08-31T10:00:00.000Z" and lead["ended_at"] == "2026-08-31T10:10:00.000Z"
    assert lead["model"] == "claude-opus-5·high" and lead["actor"] == "lead"
    assert _task(run, "plan")["attempts"][0]["model"] == "claude-opus-5"  # no effort: no chip suffix
    # The chip is display only; the label and its evidence tier travel in the task's ext.
    assert _task(run, "lead")["ext"]["dev.dagr.graph"]["model"] == {"raw": "claude-opus-5", "tier": "verified"}
    assert _task(run, "lead")["ext"]["dev.dagr.graph"]["effort"] == "high"
    assert _task(run, "codex-review")["ext"]["dev.dagr.graph"]["model"] == {"raw": "gpt-5", "tier": "reported"}
    assert "effort" not in _task(run, "plan")["ext"]["dev.dagr.graph"]
    assert "model" not in _task(run, "analyst")["attempts"][0] and "model" not in _task(run, "analyst")["ext"]["dev.dagr.graph"]
    assert _task(run, "analyst")["ext"]["dev.dagr.graph"]["model_missing"] == "no model label in the session record"
    assert _task(run, "plan")["note"] == "phase build (heuristic)"
    ats = [e["at"] for e in run["events"]]
    assert ats == sorted(ats) and len(run["events"]) == 2 * len(g.nodes)
    assert run["events"][0] == {"at": "2026-08-31T10:00:00.000Z", "type": "attempt_started", "task": "lead", "attempt": "lead.a1", "actor": "lead"}
    settled = [e for e in run["events"] if e["type"] == "attempt_settled" and e["task"] == "lead"]
    assert settled == [{"at": "2026-08-31T10:10:00.000Z", "type": "attempt_settled", "task": "lead", "attempt": "lead.a1", "actor": "lead", "detail": "settled_unverified (heuristic)"}]
    ex = run["ext"]["dev.dagr.graph"]["exporter"]
    assert ex["attempts_without_model_chip"] == 1 and ex["attempts_without_start_ts"] == 0 and ex["attempts_without_end_ts"] == 0


def test_external_launcher_is_timestamped_from_its_launch_calls(g, run):
    """Analyst has no start time or duration in the graph (it is outside the requested
    workspaces); it launched codex-audit at 10:08:30 (codex-audit starts 10:08:32, lag
    2 s). Both timestamps are that call, tier heuristic, said so everywhere."""
    assert g.node("analyst").ts is None
    task = _task(run, "analyst")
    a = task["attempts"][0]
    assert a["started_at"] == "2026-08-31T10:08:30.000Z" and a["ended_at"] == "2026-08-31T10:08:30.000Z"
    why = "started_at and ended_at are the earliest and latest of the 1 launch call(s) this session made (heuristic); the extractor carries no start time or duration for it"
    assert task["ext"]["dev.dagr.graph"]["timestamps"] == {"tier": "heuristic", "evidence": why}
    assert task["note"] == "phase external (heuristic); timestamps from launch calls (heuristic, 1 call(s))"
    assert a["outcome"]["reason"].endswith("; " + why)
    ev = [e for e in run["events"] if e["task"] == "analyst"]
    assert [e["type"] for e in ev] == ["attempt_started", "attempt_settled"] and ev[1]["detail"] == "settled_unverified (heuristic); ended_at is the latest launch call (heuristic)"
    assert run["ext"]["dev.dagr.graph"]["exporter"]["attempts_timestamped_from_launch_calls"] == 1
    # Two launch calls: the window spans them.
    g2 = extract(skeleton_records(), workspaces=[ALPHA])
    g2.edges.append(GraphEdge("analyst", "reader", "launch", "heuristic", {"how": "test", "lag_s": 30.0}))
    a2 = _task(to_runfile(g2), "analyst")["attempts"][0]
    assert a2["started_at"] == "2026-08-31T10:08:30.000Z" and a2["ended_at"] == "2026-08-31T10:19:30.000Z"


def test_a_session_with_no_time_at_all_gets_none_and_is_counted():
    g = extract(skeleton_records(), workspaces=[ALPHA])
    g.edges = [e for e in g.edges if not (e.kind == "launch" and e.src == "analyst")]
    run = to_runfile(g)
    a = _task(run, "analyst")["attempts"][0]
    assert "started_at" not in a and "ended_at" not in a
    assert _task(run, "analyst")["ext"]["dev.dagr.graph"]["started_at_missing"] == "no start time in the session record"
    ex = run["ext"]["dev.dagr.graph"]["exporter"]
    assert ex["attempts_without_start_ts"] == 1 and ex["attempts_without_end_ts"] == 1 and ex["attempts_timestamped_from_launch_calls"] == 0
    assert not any(e["task"] == "analyst" for e in run["events"])


def test_generated_at_is_the_latest_end_unless_supplied(run, g):
    assert run["generated_at"] == "2026-08-31T10:31:00.000Z"  # codex-stray, the last session to end
    assert run["ext"]["dev.dagr.graph"]["generated_at_basis"].startswith("the latest attempt end in the document")
    given = to_runfile(g, generated_at="2026-09-01T00:00:00Z")
    assert given["generated_at"] == "2026-09-01T00:00:00Z" and given["ext"]["dev.dagr.graph"]["generated_at_basis"] == "supplied by the caller"


def test_ext_carries_meta_emitter_counters_labels_and_origin(g, run):
    ext = run["ext"]["dev.dagr.graph"]
    assert ext["meta"] == g.meta
    assert ext["emitter"] == to_ocp(g)["ext"]["dev.loopmath.graph"]["emitter"]
    assert set(ext["exporter"]) == set(COUNTERS)
    assert ext["privacy"]["profile"] == "metadata_only" and ext["producer"]["name"] == "loopmath"
    t = _task(run, "codex-review")["ext"]["dev.dagr.graph"]
    assert t["origin"]["launched_by"] == "lead.a1" and t["origin"]["tier"] == "heuristic"
    assert t["role"] == {"value": "reviewer", "tier": "heuristic", "evidence": "launch command mentions review"}
    assert t["phase"]["value"] == "build" and t["harness"] == "codex"


def test_ext_carries_artifacts_artifact_edges_and_costs_verbatim_and_counted(g, run):
    """Contract v3 has no place for artifacts, artifact edges or cost records;
    they go under ext exactly as the OCP document has them, and are counted."""
    doc = to_ocp(g)
    ext = run["ext"]["dev.dagr.graph"]
    assert ext["artifacts"] == doc["artifacts"] and len(ext["artifacts"]) == 3
    assert ext["artifact_edges"] == [e for e in doc["edges"] if e["kind"] == "artifact"] and len(ext["artifact_edges"]) == 3
    assert all(e["tier"] in ("verified", "heuristic", "reported") for e in ext["artifact_edges"])
    priced = {a["id"]: a["cost"] for a in doc["attempts"] if "cost" in a}
    assert ext["costs"] == [{"attempt": f"{t['id']}.a1", "task": t["id"], "cost": priced[f"{t['id']}.a1"]} for t in run["tasks"] if f"{t['id']}.a1" in priced]
    assert len(ext["costs"]) == 9 and all(c["cost"]["basis"] == "measured" for c in ext["costs"])
    analyst = next(a for a in doc["attempts"] if a["id"] == "analyst.a1")
    assert "cost" not in analyst and _task(run, "analyst")["ext"]["dev.dagr.graph"]["cost_unknown"] == analyst["ext"]["dev.loopmath.graph"]["cost_unknown"]
    ex = ext["exporter"]
    assert ex["ext_artifacts_carried"] == 3 and ex["ext_artifact_edges_carried"] == 3 and ex["ext_cost_records_carried"] == 9 and ex["attempts_without_cost_record"] == 1
    # No task carries a cost on the contract side: the validator would not know the field.
    assert not any("cost" in a for t in run["tasks"] for a in t["attempts"])


def test_title_from_role_and_spawn_description_under_full_privacy(g):
    doc = to_ocp(g, privacy="full")
    run = to_runfile(g, ocp=doc)
    assert _task(run, "plan")["title"] == "planner: Plan the extractor"
    assert run["ext"]["dev.dagr.graph"]["privacy"]["profile"] == "full"
    meta_only = json.dumps(to_runfile(g))
    assert "Plan the extractor" not in meta_only


def _cyclic_graph() -> Graph:
    nodes = [
        GraphNode(id="a", harness="claude-code", source="top", session_path="/s/a.jsonl", workspace="w", ts="2026-08-31T10:00:00Z", wall_s=100.0, role="lead", role_tier="heuristic", phase="build", phase_tier="heuristic"),
        GraphNode(id="b", harness="codex", source="codex", session_path="/s/b.jsonl", workspace="w", ts="2026-08-31T10:00:10Z", wall_s=50.0, model="gpt-5", effort="high"),
        GraphNode(id="c", harness="claude-code", source="subagent", session_path="/s/c.jsonl", workspace="w", ts="2026-08-31T10:00:20Z", wall_s=10.0, parent="b"),
    ]
    edges = [
        GraphEdge("a", "b", "launch", "heuristic", {"how": "cli", "lag_s": 1.0}),
        GraphEdge("b", "c", "spawn", "verified", {}),
        GraphEdge("c", "a", "launch", "heuristic", {"how": "cli", "lag_s": 1.0}),  # closes a cycle
    ]
    return Graph(nodes=nodes, edges=edges, artifacts=[], meta={})


def test_a_dependency_that_would_close_a_cycle_is_held_with_the_reason(tmp_path):
    """Edges are admitted from the earliest-started parent on, so the link from c
    (started last) back to a is the one held, whatever the node order."""
    run = to_runfile(_cyclic_graph())
    assert _task(run, "b")["deps"] == ["a"] and _task(run, "c")["deps"] == ["b"] and _task(run, "a")["deps"] == []
    held = run["ext"]["dev.dagr.graph"]["deps_not_emitted"]
    assert held == [{"task": "a", "dep": "c", "kind": "launch", "tier": "heuristic", "reason": "this dependency would close a cycle; the contract requires a DAG"}]
    assert run["ext"]["dev.dagr.graph"]["exporter"]["deps_not_emitted_cycle"] == 1
    assert _task(run, "b")["attempts"][0]["model"] == "gpt-5·high"
    # A model label whose tier the graph did not give: the label is kept, the missing tier is said.
    b_ext = _task(run, "b")["ext"]["dev.dagr.graph"]
    assert b_ext["model"] == {"raw": "gpt-5"} and b_ext["effort"] == "high"
    assert b_ext["model_tier_omitted"] == "graph gave model tier None, not one of verified/heuristic/reported; the label is kept, its tier is unknown"
    assert strict_check(run, tmp_path / "cycle.run.json") == []


def test_em_dashes_never_reach_the_run_file_and_the_lifted_vocabulary_passes(g):
    g2 = extract(skeleton_records(), workspaces=[ALPHA])
    g2.node("plan").spawn = {"description": "Plan the bandit arms \u2014 posterior first"}
    run = to_runfile(g2, ocp=to_ocp(g2, privacy="full"))
    text = json.dumps(run, ensure_ascii=False)
    assert "\u2014" not in text and "[term withheld]" not in text
    assert _task(run, "plan")["title"] == "planner: Plan the bandit arms - posterior first"


def test_nested_meta_strings_are_sanitized_recursively_and_every_replacement_is_counted():
    """Graph.meta is copied into ext; an em-dash nested at every depth (a string,
    a list, a dict in a list, a key) never reaches the run file, and the counters
    say how many strings and how many replacements. The lifted vocabulary (Q3)
    passes through unchanged and is no longer counted."""
    g = extract(skeleton_records(), workspaces=[ALPHA])
    g.meta["note"] = "scan \u2014 done"
    g.meta["terms"] = [f"the {term} here" for term in LIFTED_TERMS]
    g.meta["nested"] = {"deep": [{"why": "Bandit arms \u2014 posterior \u2014 prior"}, 7, None, True]}
    g.meta["bandit \u2014 key"] = "prior \u2014 value"
    g.meta["untouched"] = {"n": 3, "flag": False, "words": "a plain reason"}
    run = to_runfile(g)
    text = json.dumps(run, ensure_ascii=False)
    assert "\u2014" not in text
    meta = run["ext"]["dev.dagr.graph"]["meta"]
    assert meta["note"] == "scan - done"
    assert meta["terms"] == [f"the {term} here" for term in LIFTED_TERMS]
    assert meta["nested"] == {"deep": [{"why": "Bandit arms - posterior - prior"}, 7, None, True]}
    assert meta["bandit - key"] == "prior - value"
    assert meta["untouched"] == {"n": 3, "flag": False, "words": "a plain reason"}  # non-strings and clean strings untouched
    ex = run["ext"]["dev.dagr.graph"]["exporter"]
    assert ex["strings_with_em_dash_replaced"] == 4 and ex["em_dashes_replaced"] == 5
    assert "forbidden_terms_replaced" not in ex and "strings_with_forbidden_term_replaced" not in ex
    # Sanitizing changed nothing the skeleton itself carries: the counters are exactly these.
    clean = to_runfile(extract(skeleton_records(), workspaces=[ALPHA]))["ext"]["dev.dagr.graph"]["exporter"]
    assert clean["em_dashes_replaced"] == 0 and clean["strings_with_em_dash_replaced"] == 0


def test_identifier_sanitization_remaps_every_task_attempt_project_and_reference(tmp_path):
    # Both identifiers sanitize to "lead - one"; sorted, the parent comes first.
    parent_id = "lead \u2014 one"
    child_id = "lead\u2014one"
    workspace = "arms\u2014workspace"
    g = Graph(
        nodes=[
            GraphNode(id=parent_id, harness="claude-code", source="top", session_path="/s/a", workspace=workspace, ts="2026-08-31T10:00:00Z", wall_s=10.0, role="lead"),
            GraphNode(id=child_id, harness="claude-code", source="subagent", session_path="/s/b", workspace=workspace, ts="2026-08-31T10:00:01Z", wall_s=5.0, role="bandit\u2014actor", parent=parent_id),
            GraphNode(id="z", harness="claude-code", source="top", session_path="/s/z", workspace=workspace, ts="2026-08-31T10:00:20Z", wall_s=1.0),
        ],
        edges=[GraphEdge(parent_id, child_id, "spawn", "verified", {})],
        artifacts=[],
        meta={},
    )
    doc = to_ocp(g)
    parent_node = next(n for n in doc["nodes"] if n["id"] == parent_id)
    child_node = next(n for n in doc["nodes"] if n["id"] == child_id)
    parent_node["kind"] = parent_id
    child_node["kind"] = child_id
    next(a for a in doc["attempts"] if a["node"] == parent_id)["actor"] = parent_id
    child_attempt = next(a for a in doc["attempts"] if a["node"] == child_id)
    child_attempt["actor"] = child_id
    next(a for a in doc["attempts"] if a["node"] == "z")["cause"] = {"type": "followup", "ref": f"{child_id}.a1"}
    # These are data, not references. Matching an identifier must not give
    # paths, labels or evidence the identifier namespace's collision suffix.
    g.meta["matching_free_text"] = {"path": child_id, "label": child_id, "evidence": child_id, "from": child_id, "actor": child_id, "kind": child_id}
    run = to_runfile(g, ocp=doc)

    parent = "lead - one"
    child = "lead - one~2"
    project = "arms - workspace"
    assert {t["id"] for t in run["tasks"]} == {parent, child, "z"}
    assert _task(run, child)["deps"] == [parent]
    assert _task(run, child)["project"] == project
    assert _task(run, parent)["kind"] == parent
    assert _task(run, child)["kind"] == child
    assert run["projects"] == [{"id": project, "title": f"workspace {project}"}]
    attempt = _task(run, child)["attempts"][0]
    assert attempt["id"] == f"{child}.a1"
    assert attempt["actor"] == child and _task(run, child)["owner"] == child
    assert _task(run, "z")["attempts"][0]["cause"] == {"type": "followup", "ref": f"{child}.a1"}
    child_events = [e for e in run["events"] if e["task"] == child]
    assert {e["attempt"] for e in child_events} == {f"{child}.a1"}
    assert {e["actor"] for e in child_events} == {child}
    assert _task(run, child)["ext"]["dev.dagr.graph"]["deps"] == [{"task": parent, "kind": "spawn", "tier": "verified"}]
    assert run["ext"]["dev.dagr.graph"]["meta"]["matching_free_text"] == {"path": parent, "label": parent, "evidence": parent, "from": parent, "actor": parent, "kind": parent}
    text = json.dumps(run, ensure_ascii=False).lower()
    assert "\u2014" not in text
    assert run["ext"]["dev.dagr.graph"]["exporter"]["em_dashes_replaced"] > 0
    # One collision each in the task, actor and kind namespaces.
    assert run["ext"]["dev.dagr.graph"]["exporter"]["identifier_collisions_disambiguated"] == 3
    assert strict_check(run, tmp_path / "sanitized-identifiers.run.json") == []


def test_sanitized_dictionary_key_collisions_are_disambiguated_and_counted():
    g = extract(skeleton_records(), workspaces=[ALPHA])
    g.meta["colliding_keys"] = {"lead\u2014one": "first record", "lead - one": "second record"}
    run = to_runfile(g)
    meta = run["ext"]["dev.dagr.graph"]["meta"]
    assert meta["colliding_keys"] == {"lead - one": "first record", "lead - one~2": "second record"}
    assert run["ext"]["dev.dagr.graph"]["exporter"]["dictionary_key_collisions_disambiguated"] == 1


def test_deterministic(g):
    assert json.dumps(to_runfile(g, signals=SIGNALS)) == json.dumps(to_runfile(g, signals=SIGNALS))


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    def fake_parse_all(logs, limit=None, use_cache=True, progress=None, since_days=None, **kw):
        records = skeleton_records()
        for r in records:
            if r["run_id"] in SIGNALS:
                r["_signals"] = dict(SIGNALS[r["run_id"]])
        return records, json.loads(json.dumps(DIAG))

    monkeypatch.setattr(ingest, "parse_all", fake_parse_all)
    monkeypatch.setattr(ingest, "discover", lambda logs, since_days=None, **kw: {"claude-code": []})
    monkeypatch.setattr(grade, "grade_all", lambda records: (records, json.loads(json.dumps(COVERAGE))))
    monkeypatch.setattr(price, "price_all", lambda records, table: (records, json.loads(json.dumps(PRICE_WARNINGS))))


def test_cli_format_run_writes_a_strict_clean_file_and_prints_exporter_counters(stubbed_pipeline, tmp_path, monkeypatch, capsys):
    """The codex audit's rollout is moved to a copy that ends on task_complete
    (with exit_ok from the ingest) so the CLI path is seen promoting exactly one
    session; every claude-code session is counted as lacking a marker, and the
    per-harness count is printed with its reason."""
    audit = next(r for r in skeleton_records() if r["run_id"] == "codex-audit")
    clean_copy = tmp_path / "rollout-audit.jsonl"
    clean_copy.write_text(Path(audit["session_path"]).read_text() + json.dumps(TASK_COMPLETE) + "\n")
    real_parse_all = ingest.parse_all

    def parse_all_with_clean_audit(*a, **kw):
        records, diag = real_parse_all(*a, **kw)
        for r in records:
            if r["run_id"] == "codex-audit":
                r["session_path"] = str(clean_copy)
                r["_signals"] = {"exit_ok": True, "timed_out": False}
        return records, diag

    monkeypatch.setattr(ingest, "parse_all", parse_all_with_clean_audit)
    tree = _git_tree(tmp_path / "tree")
    monkeypatch.chdir(tree)
    rc = main(["graph", "--workspace", ALPHA, "--all", "--format", "run", "--out", "out/alpha.run.json", "--quiet"])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "  emitter.run.tasks_done_clean_finish: 1\n" in captured.err
    assert "  emitter.run.tasks_settled_unverified_harness_without_finish_marker: 7\n" in captured.err
    assert "  emitter.run.tasks_settled_unverified_no_finish_marker: 2\n" in captured.err
    assert f"  emitter.run.sessions_without_finish_marker.claude-code: 7 ({CLAUDE_NO_MARKER})\n" in captured.err
    assert "  emitter.run.attempts_timestamped_from_launch_calls: 1\n" in captured.err
    assert "  emitter.run.ext_artifacts_carried: 3\n" in captured.err
    assert "  emitter.run.ext_artifact_edges_carried: 3\n" in captured.err
    assert "  emitter.run.ext_cost_records_carried: 9\n" in captured.err
    assert "  emitter.run.attempts_without_cost_record: 1\n" in captured.err
    assert "  emitter.run.dictionary_key_collisions_disambiguated: 0\n" in captured.err
    assert "wrote " in captured.err and "(run)" in captured.err
    doc = json.loads((tree / "out" / "alpha.run.json").read_text())
    assert doc["dagr"] == 3 and _task(doc, "codex-audit")["state"] == "done" and _task(doc, "codex-audit")["attempts"][0]["outcome"]["evidence"] == "verified"
    assert _task(doc, "lead")["state"] == "settled_unverified" and _task(doc, "dev")["state"] == "settled_unverified"
    assert _task(doc, "codex-review")["attempts"][0]["outcome"]["reason"].startswith("the rollout's last record is response_item/message")
    assert doc["ext"]["dev.dagr.graph"]["meta"]["ingest_files_seen"] == 15  # the pipeline counters travel with the graph
    assert strict_check(doc, tmp_path / "cli.run.json") == []
