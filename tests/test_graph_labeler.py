"""The labeler's prompt, parser, scorer, cost figures and the grid runner's dry run.

The dataset under test is the synthetic delta workspace of `tests/test_graph_dataset.py`
built with its `labels-full/` files (three nodes: `delta-lead` with two agreeing contract
attempts, `delta-impl` with two disagreeing ones and an E2 verdict, `delta-reviewer` with
nothing but swarm labels). The model answer is a recorded fake (`FAKE_RESPONSE`): prose
around a fenced JSON object with one wrong role, one value outside the vocabulary, one
missing key, one bad boundary time, one confidence out of range, one id the batch never
asked about and one id answered twice. No model is called anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from loopmath.graph.dataset import Report, build_dataset, load_labels, write_jsonl
from loopmath.graph.extract import extract
from loopmath.graph.labeler import (
    BATCH_SIZE,
    COST_BOUND,
    PREDICTION_KEYS,
    PROMPT_VERSION,
    PROMPT_VERSIONS,
    LabelerError,
    batches,
    build_prompt,
    candidate_parent_metadata,
    candidate_parents,
    claude_json_result,
    cost_figures,
    find_session_files,
    gold_boundaries,
    load_dataset,
    load_predictions,
    main,
    node_view,
    parse_response,
    price_sessions,
    report_lines,
    score,
    session_evidence,
    session_id_from_events,
    workflow_evidence,
    write_predictions,
)
from test_graph_dataset import FIX, WS, fake_git, fixture_records

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"knowledge gradient|posterior|\bprior\b|experimental design|value of information|bandit|\barms?\b", re.I)

FAKE_RESPONSE = """Here are the labels you asked for.

```json
{"labels": [
  {"id": "delta-impl", "role": "dev", "fine_role": "implement", "parent": "delta-lead",
   "boundaries": ["2026-08-31T12:05:30Z"], "send_back": true, "approved": false,
   "confidence": 0.8, "evidence": "spawn description says implement the parser; a second commit round follows"},
  {"id": "delta-lead", "role": "lead", "fine_role": "orchestrate", "parent": null,
   "boundaries": [], "send_back": false, "approved": true, "confidence": 0.9,
   "evidence": "first prompt names it the lead; it spawned the implementer"},
  {"id": "delta-reviewer", "role": "planner", "fine_role": "bogus",
   "boundaries": ["not a time"], "send_back": null, "approved": true, "confidence": 1.5,
   "evidence": "first prompt says review"},
  {"id": "delta-ghost", "role": "dev", "fine_role": null, "parent": null, "boundaries": [], "send_back": null, "approved": null, "confidence": 0.1},
  {"id": "delta-lead", "role": "dev", "fine_role": null, "parent": null, "boundaries": [], "send_back": null, "approved": null, "confidence": 0.2}
]}
```

Let me know if you need anything else.
"""

MODEL, EFFORT = "gpt-5.6-luna", "low"


@pytest.fixture(scope="module")
def items():
    g = extract(fixture_records(), workspaces=[WS])
    labels = load_labels(FIX / "labels-full")
    return build_dataset(g, labels, workspaces=[WS], git=fake_git, report=Report())


@pytest.fixture(scope="module")
def nodes(items):
    return [it for it in items if it["item"] == "node"]


@pytest.fixture(scope="module")
def parsed(nodes):
    return parse_response(FAKE_RESPONSE, nodes, model=MODEL, effort=EFFORT, batch_index=0)


@pytest.fixture(scope="module")
def dataset_path(tmp_path_factory, items):
    p = tmp_path_factory.mktemp("d2") / "dataset.jsonl"
    write_jsonl(items, p)
    return p


# ---- dataset and batches ---------------------------------------------------------------


def test_load_dataset_keeps_file_order_and_counts(dataset_path, items):
    nodes, edges, c = load_dataset(dataset_path)
    assert [n["id"] for n in nodes] == ["delta-impl", "delta-lead", "delta-reviewer"]
    assert len(edges) == 2 and c["items_node"] == 3 and c["items_edge"] == 2
    with pytest.raises(LabelerError, match="does not exist"):
        load_dataset(dataset_path.parent / "nope.jsonl")
    bad = dataset_path.parent / "bad.jsonl"
    bad.write_text('{"item": "node", "id": "x"}\nnot json\n', encoding="utf-8")
    with pytest.raises(LabelerError, match="line 2: not JSON"):
        load_dataset(bad)


def test_batches_hold_up_to_the_size_in_order(nodes):
    assert BATCH_SIZE == 20
    fake = [{"id": f"n{i:02d}"} for i in range(45)]
    bs = batches(fake)
    assert [len(b) for b in bs] == [20, 20, 5]
    assert [it["id"] for it in bs[2]] == ["n40", "n41", "n42", "n43", "n44"]
    assert len(batches(nodes)) == 1 and [len(b) for b in batches(nodes, 2)] == [2, 1]
    with pytest.raises(ValueError):
        batches(nodes, 0)


# ---- prompt ----------------------------------------------------------------------------


def test_node_view_carries_spans_and_reasons_but_no_gold(nodes):
    by_id = {n["id"]: n for n in nodes}
    v = node_view(by_id["delta-impl"])
    assert v["id"] == "delta-impl" and v["source"] == "subagent" and v["wall_clock_s"] == 120.0
    assert v["extractor_parent"] == {"value": "delta-lead", "tier": "verified"}
    assert v["spawn_description"] == {"value": "Implement the parser", "tier": "verified"}
    assert v["first_prompt"]["value"].startswith("Implement /ws/delta/src/parser.py") and v["first_prompt"]["tier"] == "verified"
    assert v["parent_command"]["tier"] == "verified" and "parser" in v["parent_command"]["value"]
    assert v["extractor_role"] == {"value": "dev", "tier": "heuristic", "evidence": "subagent with no planner/reviewer signal"}
    text = json.dumps(v)
    for word in ("gold", "attempts", "verdict", "send_back", "approved"):
        assert word not in text
    r = node_view(by_id["delta-reviewer"])
    assert r["spawn_description"] == {"value": None, "reason": "the node is a top session, not a subagent"}
    assert r["extractor_parent"] == {"value": None, "reason": "the node has no parent"}
    assert r["commits_in_interval"] == {"value": None, "reason": "no commits without the node's own cwd: the transcript carries no cwd"}
    lead = node_view(by_id["delta-lead"])
    assert lead["commits_in_interval"]["count"] == 1 and lead["commits_in_interval"]["value"] == ["2026-08-31T12:05:00Z delta: parser and notes"]


def test_prompt_names_version_count_candidates_and_every_node(nodes, items):
    edges = [it for it in items if it["item"] == "edge"]
    text = build_prompt(nodes, all_nodes=nodes, edges=edges)
    assert f"PROMPT_VERSION: {PROMPT_VERSION}" in text and "NODE_COUNT: 3" in text
    body = json.loads(text[text.index("NODE_COUNT: 3") + len("NODE_COUNT: 3"):])
    assert body["CANDIDATE_PARENTS"] == ["delta-impl", "delta-lead", "delta-reviewer"]
    assert body["PROMPT_METADATA"]["parent_candidates"] == {
        "included": 3,
        "excluded": 2,
        "excluded_by_reason": {"edge_source_duplicate_id": 1, "edge_wrong_kind": 1},
        "source_records": 5,
    }
    assert [n["id"] for n in body["NODES"]] == ["delta-impl", "delta-lead", "delta-reviewer"]
    head = text[: text.index("PROMPT_VERSION")]
    for word in ("lead", "planner", "dev", "reviewer", "smoke", "solo", "external", "implement", "repair", "send_back", "approve", "orchestrate", "boundaries", "confidence", "evidence"):
        assert word in head
    assert "—" not in head and not FORBIDDEN.search(head)
    assert "verified" in head and "heuristic" in head and "reported" in head
    assert PROMPT_VERSIONS == ("v1", "v2", "v3", "v3-strict", "v4")
    with pytest.raises(LabelerError, match="not implemented"):
        build_prompt(nodes, version="v5", all_nodes=nodes, edges=edges)
    # Dataset-wide inputs are deliberately mandatory. A one-batch fallback hides valid
    # candidates outside that batch, which is the behavior this fix removes.
    with pytest.raises(LabelerError, match="requires dataset-wide nodes"):
        build_prompt(nodes)
    with pytest.raises(LabelerError, match="requires dataset-wide candidate edges"):
        build_prompt(nodes, all_nodes=nodes)


def test_v1_prompt_is_byte_stable(nodes, items):
    edges = [it for it in items if it["item"] == "edge"]
    text = build_prompt(nodes, version="v1", all_nodes=nodes, edges=edges)
    assert len(text) == 6541
    assert hashlib.sha256(text.encode()).hexdigest() == "b5988c8a0225dc9590df0b3c9df113c17c22dc2e39aea9e4c58064876458d60e"


@pytest.mark.parametrize(
    ("version", "length", "digest"),
    [
        ("v1", 6541, "b5988c8a0225dc9590df0b3c9df113c17c22dc2e39aea9e4c58064876458d60e"),
        ("v2", 10878, "747678781bb6513f8bd11f59b9eeea05f426964b860cce6fac407dd4ee883b22"),
        ("v3", 8454, "2d55da9df823c1a36fd087fa9f489c12006edfb2f660fabd7e138f9babe4ae45"),
        ("v3-strict", 7099, "e0d68d61fe29eb103115e04382b65d75c2d5dc5d12cc572f9b6db1129cf99b7c"),
    ],
)
def test_pre_v4_prompts_remain_byte_stable(nodes, items, version, length, digest):
    edges = [it for it in items if it["item"] == "edge"]
    text = build_prompt(nodes, version=version, all_nodes=nodes, edges=edges)
    assert len(text) == length
    assert hashlib.sha256(text.encode()).hexdigest() == digest


def test_v2_prompt_exposes_only_label_free_attempt_evidence(nodes, items):
    edges = [it for it in items if it["item"] == "edge"]
    text = build_prompt(nodes, version="v2", all_nodes=nodes, edges=edges)
    assert "PROMPT_VERSION: v2" in text
    v2_head = text.split("\nPROMPT_VERSION: v2", 1)[0]
    assert len(v2_head) == 4567
    assert hashlib.sha256(v2_head.encode()).hexdigest() == "1c516b6bcf3e271e61f0a0c87748751825fdae0ad48872e7009364d8531614be"
    assert "ATTEMPT_EVIDENCE" in text and "ATTEMPT_LEDGER" in text
    assert "after the earliest" in text and "cause.type\n\"sent_back\"" in text
    assert "A session assigned one coding task is dev" in text
    assert "Use solo only for a\ntop-level session" in text
    body = json.loads(text[text.index("NODE_COUNT: 3") + len("NODE_COUNT: 3"):])
    impl = next(node for node in body["NODES"] if node["id"] == "delta-impl")
    assert impl["attempt_evidence"] == [
        {
            "session_id": "delta-impl", "task": "P1", "attempt_number": 1,
            "attempt_id": "P1·a1",
            "started_at": "2026-08-31T12:01:00Z", "ended_at": "2026-08-31T12:03:00Z",
            "result": "done", "cause": {"type": "initial", "ref": None},
        },
        {
            "session_id": "delta-impl", "task": "P1", "attempt_number": 2,
            "attempt_id": "P1·a2",
            "started_at": "2026-08-31T12:05:00Z", "ended_at": "2026-08-31T12:08:00Z",
            "result": "done", "cause": {"type": "sent_back", "ref": "P1"},
        },
    ]
    ledger = body["ATTEMPT_LEDGER"]
    assert [row["attempt_id"] for row in ledger] == ["W0·a1", "P1·a1", "P1·a2", "W0·a2"]
    evidence = json.dumps({"nodes": [{"attempt_evidence": n["attempt_evidence"]} for n in body["NODES"]], "ledger": ledger})
    for forbidden in ("gold", "labels", "verdict", "label_tier", "label_source", "session_source", "actor", "model"):
        assert forbidden not in evidence
    assert body["PROMPT_METADATA"]["attempt_evidence"] == {
        "nodes": 3, "nodes_with_attempts_field": 2, "nodes_without_attempts": 1,
        "nodes_with_empty_attempts": 0, "attempt_containers_invalid": 0,
        "attempt_records": 4, "attempt_records_included": 4,
        "attempt_records_excluded": 0, "excluded_by_reason": {},
        "missing_fields": {"cause.ref": 2}, "invalid_fields": {},
        "normalized_fields": {"cause.ref": {"null": 2}},
    }


def test_v3_keeps_only_per_node_session_and_times_with_no_ledger(nodes, items):
    edges = [it for it in items if it["item"] == "edge"]
    v2 = build_prompt(nodes, version="v2", all_nodes=nodes, edges=edges)
    text = build_prompt(nodes, version="v3", all_nodes=nodes, edges=edges)
    assert "PROMPT_VERSION: v3" in text and "ATTEMPT_EVIDENCE" in text
    assert "shown attempt start after the earliest" in text
    assert "use only the ordinary node skeleton and text spans" in text
    role_guard = """For role,
fine_role and parent, apply the v1 definitions to the skeleton and text spans exactly
as if the attempt evidence were absent. A session assigned one coding task is dev,
even when it has no children or implemented that task alone. Use solo only for a
top-level session that handled the whole end-to-end job rather than one assigned task."""
    assert role_guard in v2 and role_guard in text
    body = json.loads(text[text.index("NODE_COUNT: 3") + len("NODE_COUNT: 3"):])
    assert "ATTEMPT_LEDGER" not in body
    impl = next(node for node in body["NODES"] if node["id"] == "delta-impl")
    assert impl["attempt_evidence"] == [
        {
            "session_id": "delta-impl",
            "started_at": "2026-08-31T12:01:00Z",
            "ended_at": "2026-08-31T12:03:00Z",
        },
        {
            "session_id": "delta-impl",
            "started_at": "2026-08-31T12:05:00Z",
            "ended_at": "2026-08-31T12:08:00Z",
        },
    ]
    assert body["PROMPT_METADATA"]["attempt_evidence"]["missing_fields"] == {}
    assert body["PROMPT_METADATA"]["attempt_evidence"]["invalid_fields"] == {}
    projected = json.dumps({
        "rows": [node["attempt_evidence"] for node in body["NODES"]],
        "metadata": body["PROMPT_METADATA"]["attempt_evidence"],
    })
    for hidden in ('"task"', '"n"', "attempt_number", "attempt_id", '"result"', '"cause"'):
        assert hidden not in projected


def test_v3_strict_has_no_attempt_block_and_uses_spans_for_all_outcomes(nodes, items):
    edges = [it for it in items if it["item"] == "edge"]
    v2 = build_prompt(nodes, version="v2", all_nodes=nodes, edges=edges)
    text = build_prompt(nodes, version="v3-strict", all_nodes=nodes, edges=edges)
    assert "PROMPT_VERSION: v3-strict" in text
    assert "ATTEMPT_EVIDENCE" not in text and "ATTEMPT_LEDGER" not in text
    assert "For boundaries, send_back and approved, use only the ordinary node skeleton" in text
    role_guard = """For role,
fine_role and parent, apply the v1 definitions to the skeleton and text spans exactly
as if the attempt evidence were absent. A session assigned one coding task is dev,
even when it has no children or implemented that task alone. Use solo only for a
top-level session that handled the whole end-to-end job rather than one assigned task."""
    assert role_guard in v2 and role_guard in text
    body = json.loads(text[text.index("NODE_COUNT: 3") + len("NODE_COUNT: 3"):])
    assert "attempt_evidence" not in body["PROMPT_METADATA"]
    assert "ATTEMPT_LEDGER" not in body
    assert all("attempt_evidence" not in node for node in body["NODES"])


def test_v4_projects_raw_session_and_workflow_evidence_with_accounted_caps(tmp_path, nodes, items):
    projected = json.loads(json.dumps(nodes[:2]))
    by_id = {node["id"]: node for node in projected}
    lead_path = tmp_path / "lead.jsonl"
    child_path = tmp_path / "agent-impl.jsonl"
    long_prompt = "P" * 1000 + " repair requested " + "Q" * 1000
    long_followup = "F" * 1000 + " still working " + "G" * 1000
    long_closing = "A" * 1000 + " approved at checkpoint " + "Z" * 1000
    records = [
        {"type": "user", "timestamp": "2026-09-01T00:00:00Z", "message": {"role": "user", "content": "Open the workflow."}},
        {
            "type": "user",
            "timestamp": "2026-09-01T00:00:00.500Z",
            "message": {"role": "user", "content": "<task-notification>review child completed</task-notification>"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-09-01T00:00:01Z",
            "message": {"role": "assistant", "content": [{
                "type": "tool_use", "id": "toolu_impl", "name": "Task",
                "input": {"description": "Implement raw evidence", "prompt": long_prompt},
            }]},
        },
        {
            "type": "assistant",
            "timestamp": "2026-09-01T00:00:02Z",
            "message": {"role": "assistant", "content": [{
                "type": "tool_use", "id": "toolu_resume", "name": "Agent",
                "input": {"resume": "agent-impl", "description": "Follow up", "prompt": "Address the raw review text."},
            }]},
        },
    ]
    for index in range(10):
        followup = long_followup if index == 0 else f"operator followup {index}"
        records.extend([
            {"type": "user", "timestamp": f"2026-09-01T00:01:{index:02d}Z", "message": {"role": "user", "content": followup}},
            {
                "type": "assistant",
                "timestamp": f"2026-09-01T00:02:{index:02d}Z",
                "message": {"role": "assistant", "content": [{
                    "type": "tool_use", "id": f"toolu_review_{index}", "name": "Bash",
                    "input": {"command": f"tools/review.sh R{index} src/module_{index}.py"},
                }]},
            },
            {
                "type": "user",
                "timestamp": f"2026-09-01T00:03:{index:02d}Z",
                "message": {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": f"toolu_review_{index}",
                    "content": f"VERDICT: {'FLAGS' if index < 5 else 'APPROVE'} {index}",
                }]},
            },
        ])
    for index in range(70):
        records.append({
            "type": "assistant",
            "timestamp": f"2026-09-01T00:04:{index % 60:02d}Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"checkpoint {index} passed"}]},
        })
    records.append({
        "type": "assistant",
        "timestamp": "2026-09-01T00:06:00Z",
        "message": {"role": "assistant", "content": [{"type": "text", "text": long_closing}]},
    })
    lead_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    child_path.write_text(
        "\n".join([
            json.dumps({"type": "user", "timestamp": "2026-09-01T00:00:02Z", "message": {"role": "user", "content": "Implement raw evidence."}}),
            json.dumps({"type": "assistant", "timestamp": "2026-09-01T00:00:03Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Child complete."}]}}),
        ]) + "\n",
        encoding="utf-8",
    )
    by_id["delta-lead"]["features"]["skeleton"]["session_path"] = str(lead_path)
    by_id["delta-impl"]["features"]["skeleton"]["session_path"] = str(child_path)
    edges = [it for it in items if it["item"] == "edge"]
    text = build_prompt(projected, version="v4", all_nodes=projected, edges=edges)
    assert "PROMPT_VERSION: v4" in text and "ATTEMPT_LEDGER" not in text
    assert "SESSION_EVIDENCE and WORKFLOW_EVIDENCE" in text
    body = json.loads(text[text.index("NODE_COUNT: 2") + len("NODE_COUNT: 2"):])
    lead = next(node for node in body["NODES"] if node["id"] == "delta-lead")
    assert set(lead["attempt_evidence"][0]) == {"session_id", "started_at", "ended_at"}
    evidence = lead["session_evidence"]
    assert evidence["closing_text"]["value"] == long_closing[:800] + long_closing[-800:]
    assert evidence["closing_text"]["truncated"] is True
    assert [row["value"] for row in evidence["followup_user_text"]][1:4] == [
        "operator followup 1", "operator followup 2", "operator followup 3",
    ]
    assert [row["value"] for row in evidence["followup_user_text"]][-4:] == [
        "operator followup 6", "operator followup 7", "operator followup 8", "operator followup 9",
    ]
    assert [row["command"].split()[1] for row in evidence["tool_events"]] == [
        "R0", "R1", "R2", "R3", "R6", "R7", "R8", "R9",
    ]
    assert evidence["metadata"]["followup_user_text"] == {"available": 10, "included": 8}
    assert evidence["metadata"]["tool_events"] == {"available": 10, "included": 8}
    assert evidence["metadata"]["accounting_unit"] == "parsed_session_record"
    assert evidence["metadata"]["source_records"] == (
        evidence["metadata"]["included_records"] + evidence["metadata"]["excluded_records"]
    )
    assert evidence["metadata"]["excluded_by_reason"]["followup_list_cap"] == 2
    assert evidence["metadata"]["excluded_by_reason"]["tool_event_list_cap"] == 4
    assert evidence["metadata"]["excluded_by_reason"]["generated_system_text"] == 1
    assert evidence["metadata"]["truncated_values"] == 2
    workflow = body["WORKFLOW_EVIDENCE"]
    assert len(workflow["dispatches"]) == 2
    assert workflow["dispatches"][0]["prompt_excerpt"] == long_prompt[:800] + long_prompt[-800:]
    assert workflow["dispatches"][0]["prompt_accounting"]["truncated"] is True
    assert workflow["dispatches"][1]["child_session_id"] is None
    assert workflow["dispatches"][1]["runtime_child_id"] == "agent-impl"
    assert workflow["metadata"]["decision_text"] == {"available": 71, "included": 64}
    assert workflow["metadata"]["excluded_by_reason"]["decision_text_list_cap"] == 7
    assert workflow["metadata"]["excluded_by_reason"]["generated_system_text"] == 1
    assert workflow["metadata"]["accounting_unit"] == "parsed_session_record"
    assert workflow["metadata"]["source_records"] == (
        workflow["metadata"]["included_records"] + workflow["metadata"]["excluded_records"]
    )
    assert body["PROMPT_METADATA"]["session_evidence"]["node_collections"] == 2
    assert body["PROMPT_METADATA"]["session_evidence"]["source_records"] == (
        body["PROMPT_METADATA"]["session_evidence"]["included_records"]
        + body["PROMPT_METADATA"]["session_evidence"]["excluded_records"]
    )
    assert body["PROMPT_METADATA"]["workflow_evidence"] == workflow["metadata"]
    evidence_projection = {"nodes": [node["session_evidence"] for node in body["NODES"]], "workflow": workflow}
    rejected_keys = {
        "task", "n", "attempt_number", "attempt_id", "cause", "labels",
        "source", "tier", "same_task", "later_retry", "final", "is_rework",
        "review_rejected", "gate_passed", "merged", "accepted", "approval_status",
    }

    def all_keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(all_keys(child) for child in value.values()), set())
        if isinstance(value, list):
            return set().union(*(all_keys(child) for child in value), set())
        return set()

    assert not all_keys(evidence_projection) & rejected_keys


def test_v4_reads_codex_visible_text_and_disposition_tool_output(tmp_path, nodes):
    path = tmp_path / "codex.jsonl"
    records = [
        {
            "timestamp": "2026-09-01T01:00:00Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Review the patch."}]},
        },
        {
            "timestamp": "2026-09-01T01:00:01Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Reopen it after the findings."}]},
        },
        {
            "timestamp": "2026-09-01T01:00:02Z",
            "type": "response_item",
            "payload": {
                "type": "function_call", "name": "exec_command", "call_id": "call_review",
                "arguments": json.dumps({"cmd": "tools/review.sh C1 src/codex.py"}),
            },
        },
        {
            "timestamp": "2026-09-01T01:00:03Z",
            "type": "response_item",
            "payload": {"type": "function_call_output", "call_id": "call_review", "output": "VERDICT: FLAGS"},
        },
        {
            "timestamp": "2026-09-01T01:00:04Z",
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Review complete with flags."}]},
        },
        {
            "timestamp": "2026-09-01T01:00:05Z",
            "type": "response_item",
            "payload": {"type": "reasoning", "summary": []},
        },
        {
            "timestamp": "2026-09-01T01:00:06Z",
            "type": "response_item",
            "payload": {"type": "task_complete"},
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    item = json.loads(json.dumps(nodes[0]))
    item["features"]["skeleton"]["session_path"] = str(path)
    evidence = session_evidence(item)
    assert evidence["closing_text"]["value"] == "Review complete with flags."
    assert [row["value"] for row in evidence["followup_user_text"]] == ["Reopen it after the findings."]
    assert evidence["tool_events"] == [{
        "at": "2026-09-01T01:00:02Z",
        "tool": "exec_command",
        "command": "tools/review.sh C1 src/codex.py",
        "command_accounting": {
            "original_chars": 31,
            "shown_chars": 31,
            "not_shown_chars": 0,
            "truncated": False,
        },
        "result": "VERDICT: FLAGS",
        "result_accounting": {
            "original_chars": 14,
            "shown_chars": 14,
            "not_shown_chars": 0,
            "truncated": False,
        },
    }]
    assert evidence["metadata"]["accounting_unit"] == "parsed_session_record"
    assert evidence["metadata"]["source_records"] == 7
    assert evidence["metadata"]["included_records"] == 4
    assert evidence["metadata"]["excluded_records"] == 3
    assert evidence["metadata"]["excluded_by_reason"] == {
        "first_user_text": 1,
        "unsupported_response_item_type": 2,
    }
    assert evidence["metadata"]["source_line_exclusions_by_reason"] == {}
    assert evidence["metadata"]["source_unavailable_by_reason"] == {}


def test_v4_separates_line_failures_from_exhaustive_parsed_record_accounting(tmp_path, nodes):
    path = tmp_path / "mixed-session.jsonl"
    valid_records = [
        {"type": "user", "message": {"role": "user", "content": "Start the task."}},
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Task complete."}]},
        },
    ]
    path.write_text(
        "\nnot json\n[]\n" + "\n".join(json.dumps(record) for record in valid_records) + "\n",
        encoding="utf-8",
    )
    item = json.loads(json.dumps(nodes[0]))
    item["features"]["skeleton"]["session_path"] = str(path)
    metadata = session_evidence(item)["metadata"]
    assert metadata["accounting_unit"] == "parsed_session_record"
    assert metadata["source_lines"] == 5
    assert metadata["source_records"] == 2
    assert metadata["included_records"] == 1
    assert metadata["excluded_records"] == 1
    assert metadata["excluded_by_reason"] == {"first_user_text": 1}
    assert metadata["source_line_exclusions_by_reason"] == {
        "blank_line": 1,
        "invalid_json": 1,
        "not_object": 1,
    }


def test_v4_retains_parsed_records_after_partial_session_read_error(tmp_path, nodes, monkeypatch):
    path = tmp_path / "partial-session.jsonl"
    path.touch()
    lines = iter([
        json.dumps({"type": "user", "message": {"role": "user", "content": "Start the task."}}) + "\n",
        json.dumps({
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "Task complete."}]},
        }) + "\n",
    ])

    class PartialRead:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(lines)
            except StopIteration:
                raise OSError("simulated read failure")

    original_open = Path.open

    def partial_open(self, *args, **kwargs):
        return PartialRead() if self == path else original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", partial_open)
    item = json.loads(json.dumps(nodes[0]))
    item["features"]["skeleton"]["session_path"] = str(path)
    evidence = session_evidence(item)
    assert evidence["closing_text"]["value"] == "Task complete."
    metadata = evidence["metadata"]
    assert metadata["source_lines"] == 2
    assert metadata["source_records"] == 2
    assert metadata["included_records"] == 1
    assert metadata["excluded_records"] == 1
    assert metadata["excluded_by_reason"] == {"first_user_text": 1}
    assert metadata["source_unavailable_by_reason"] == {"session_file_partial_read": 1}


def test_v4_dispatch_text_preserves_null_and_counts_every_field_issue(tmp_path, nodes):
    path = tmp_path / "invalid-dispatches.jsonl"
    dispatch_inputs = [
        {},
        {"description": None, "prompt": None},
        {"description": 7, "prompt": ["not text"]},
        {"description": "", "prompt": ""},
    ]
    records = [
        {
            "type": "assistant",
            "timestamp": f"2026-09-01T00:00:0{index}Z",
            "message": {"role": "assistant", "content": [{
                "type": "tool_use", "id": f"toolu_{index}", "name": "Task", "input": tool_input,
            }]},
        }
        for index, tool_input in enumerate(dispatch_inputs)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    item = json.loads(json.dumps(nodes[0]))
    item["features"]["skeleton"]["session_path"] = str(path)
    item["features"]["skeleton"]["role"] = "lead"
    workflow = workflow_evidence([item])
    assert [(row["description"], row["prompt_excerpt"]) for row in workflow["dispatches"]] == [
        (None, None), (None, None), (None, None), (None, None),
    ]
    for row, reason in zip(workflow["dispatches"], ("missing", "null", "invalid_type", "invalid_value")):
        expected = {
            "original_chars": None,
            "shown_chars": None,
            "not_shown_chars": None,
            "truncated": None,
            "reason": reason,
        }
        assert row["description_accounting"] == expected
        assert row["prompt_accounting"] == expected
    assert workflow["metadata"]["dispatch_text_fields_by_reason"] == {
        "description": {"invalid_type": 1, "invalid_value": 1, "missing": 1, "null": 1},
        "prompt": {"invalid_type": 1, "invalid_value": 1, "missing": 1, "null": 1},
    }
    assert workflow["metadata"]["original_characters"] == 0
    assert workflow["metadata"]["shown_characters"] == 0
    assert workflow["metadata"]["truncated_values"] == 0


def test_v2_prompt_counts_malformed_attempt_evidence(nodes, items):
    edges = [it for it in items if it["item"] == "edge"]
    broken = json.loads(json.dumps(nodes))
    broken[0]["attempts"].append("not an object")
    broken[1]["attempts"] = {"not": "a list"}
    text = build_prompt(broken, version="v2", all_nodes=broken, edges=edges)
    body = json.loads(text[text.index("NODE_COUNT: 3") + len("NODE_COUNT: 3"):])
    meta = body["PROMPT_METADATA"]["attempt_evidence"]
    assert meta["nodes_without_attempts"] == 1
    assert meta["attempt_containers_invalid"] == 1
    assert meta["attempt_records"] == 3 and meta["attempt_records_included"] == 2
    assert meta["attempt_records_excluded"] == 1
    assert meta["excluded_by_reason"] == {"attempt_not_object": 1, "attempts_not_list": 1}
    assert meta["missing_fields"] == {"cause.ref": 1}
    assert meta["invalid_fields"] == {}
    assert meta["normalized_fields"] == {"cause.ref": {"null": 1}}


@pytest.fixture
def malformed_attempt_nodes(nodes):
    broken = json.loads(json.dumps(nodes[:2]))
    del broken[0]["id"]
    broken[0]["attempts"] = [
        {},
        {
            "task": "T-ok",
            "n": 1,
            "started_at": "2026-08-31T12:01:00Z",
            "ended_at": "2026-08-31T12:02:00Z",
            "result": "done",
            "cause": {},
        },
    ]
    broken[1]["id"] = False
    broken[1]["attempts"] = [
        {
            "task": "",
            "n": True,
            "started_at": [],
            "ended_at": "not-a-time",
            "result": {},
            "cause": [],
        },
        {
            "task": "T-valid",
            "n": 2,
            "started_at": "2026-08-31T12:03:00",
            "ended_at": "2026-08-31",
            "result": "done",
            "cause": {"type": [], "ref": False},
        },
    ]
    return broken


def test_v2_normalizes_every_malformed_projected_field_by_field_and_reason(malformed_attempt_nodes):
    text = build_prompt(
        malformed_attempt_nodes,
        version="v2",
        all_nodes=malformed_attempt_nodes,
        edges=[],
    )
    body = json.loads(text[text.index("NODE_COUNT: 2") + len("NODE_COUNT: 2"):])
    meta = body["PROMPT_METADATA"]["attempt_evidence"]
    assert meta["attempt_records"] == meta["attempt_records_included"] == 4
    assert meta["attempt_records_excluded"] == 0
    assert meta["missing_fields"] == {
        "attempt_id": 1,
        "cause": 1,
        "cause.ref": 2,
        "cause.type": 2,
        "ended_at": 1,
        "n": 1,
        "result": 1,
        "session_id": 2,
        "started_at": 1,
        "task": 1,
    }
    assert meta["invalid_fields"] == {
        "attempt_id": 1,
        "cause": 1,
        "cause.ref": 2,
        "cause.type": 2,
        "ended_at": 2,
        "n": 1,
        "result": 1,
        "session_id": 2,
        "started_at": 2,
        "task": 1,
    }
    assert meta["normalized_fields"] == {
        "attempt_id": {
            "task_invalid_value+n_invalid_type": 1,
            "task_missing+n_missing": 1,
        },
        "cause": {"invalid_type": 1, "missing": 1},
        "cause.ref": {
            "cause_invalid_type": 1,
            "cause_missing": 1,
            "invalid_type": 1,
            "missing": 1,
        },
        "cause.type": {
            "cause_invalid_type": 1,
            "cause_missing": 1,
            "invalid_type": 1,
            "missing": 1,
        },
        "ended_at": {"invalid_value": 2, "missing": 1},
        "n": {"invalid_type": 1, "missing": 1},
        "result": {"invalid_type": 1, "missing": 1},
        "session_id": {"invalid_type": 2, "missing": 2},
        "started_at": {"invalid_type": 1, "invalid_value": 1, "missing": 1},
        "task": {"invalid_value": 1, "missing": 1},
    }
    missing_node, invalid_node = body["NODES"]
    assert missing_node["attempt_evidence"][0] == {
        "session_id": None,
        "task": None,
        "attempt_number": None,
        "attempt_id": None,
        "started_at": None,
        "ended_at": None,
        "result": None,
        "cause": {"type": None, "ref": None},
    }
    assert invalid_node["attempt_evidence"][0] == missing_node["attempt_evidence"][0]
    assert invalid_node["attempt_evidence"][1]["started_at"] is None
    assert invalid_node["attempt_evidence"][1]["ended_at"] is None


def test_v3_malformed_metadata_names_only_its_three_projected_fields(malformed_attempt_nodes):
    text = build_prompt(
        malformed_attempt_nodes,
        version="v3",
        all_nodes=malformed_attempt_nodes,
        edges=[],
    )
    body = json.loads(text[text.index("NODE_COUNT: 2") + len("NODE_COUNT: 2"):])
    meta = body["PROMPT_METADATA"]["attempt_evidence"]
    assert meta["missing_fields"] == {"ended_at": 1, "session_id": 2, "started_at": 1}
    assert meta["invalid_fields"] == {"ended_at": 2, "session_id": 2, "started_at": 2}
    assert meta["normalized_fields"] == {
        "ended_at": {"invalid_value": 2, "missing": 1},
        "session_id": {"invalid_type": 2, "missing": 2},
        "started_at": {"invalid_type": 1, "invalid_value": 1, "missing": 1},
    }
    projected = json.dumps({
        "rows": [node["attempt_evidence"] for node in body["NODES"]],
        "metadata": meta,
    })
    for hidden in ('"task"', '"n"', "attempt_number", "attempt_id", '"result"', '"cause"'):
        assert hidden not in projected


def test_circularity_script_prints_v2_receipts_and_scoped_v3_v4_proof():
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/check_label_circularity.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "send_back reproduced by cause.type and cause.ref alone: 49/49" in result.stdout
    assert "approved  reproduced by result and later cause.type alone: 49/49" in result.stdout
    assert "v3 projected fields: session_id, started_at, ended_at" in result.stdout
    assert "v3 dataset-wide ATTEMPT_LEDGER: absent" in result.stdout
    assert "v3 twin-world contract validation: 196/196 worlds pass dagr check --strict" in result.stdout
    assert "v3 twin-world send_back non-determination: 49/49" in result.stdout
    assert "v3 twin-world approved non-determination:  49/49" in result.stdout
    assert "v3-strict twin-world non-determination" in result.stdout
    assert "v4 projected contract fields: session_id, started_at, ended_at" in result.stdout
    assert "no contract result, cause, task identity, attempt number, labels, or derived outcome flags" in result.stdout
    assert "v4 twin-world send_back non-determination: 49/49" in result.stdout
    assert "v4 twin-world approved non-determination:  49/49" in result.stdout
    assert "with nonempty raw session and workflow evidence" in result.stdout
    assert "approved fitted by task+n final-number shortcut: 48/49" in result.stdout
    assert "send_back fitted by n>1 shortcut: 31/49" in result.stdout
    assert "global_start_rank>=9): 49/49" in result.stdout
    assert "injectivity and fitted shortcuts are not definitional leakage" in result.stdout
    assert "establish only" in result.stdout and "v4 raw evidence do not determine either label" in result.stdout
    assert "—" not in result.stdout


def test_each_batch_prompt_has_all_node_ids_and_parent_edge_sources(nodes, items):
    edges = [it for it in items if it["item"] == "edge"]
    launch = json.loads(json.dumps(next(it for it in edges if it["kind"] == "spawn")))
    launch["kind"] = "launch"
    launch["features"]["edge"]["value"].update({"kind": "launch", "src": "outside-launcher"})
    artifact = next(it for it in edges if it["kind"] == "artifact")
    candidates = candidate_parents(nodes, edges + [launch, artifact])
    assert candidates == ["delta-impl", "delta-lead", "delta-reviewer", "outside-launcher"]
    text = build_prompt(nodes[:1], all_nodes=nodes, edges=edges + [launch, artifact])
    body = json.loads(text[text.index("NODE_COUNT: 1") + len("NODE_COUNT: 1"):])
    # This pins the dataset-wide list: the one-node batch can still choose either node
    # outside its batch and the launch source that has no node item.
    assert body["CANDIDATE_PARENTS"] == candidates
    assert [node["id"] for node in body["NODES"]] == ["delta-impl"]


def test_candidate_parent_exclusions_are_counted_by_reason(nodes, items):
    spawn = json.loads(json.dumps(next(it for it in items if it.get("kind") == "spawn")))
    malformed = [
        None,
        {},
        {"kind": "artifact"},
        {"kind": "spawn"},
        {"kind": "spawn", "features": []},
        {"kind": "spawn", "features": {}},
        {"kind": "spawn", "features": {"edge": []}},
        {"kind": "spawn", "features": {"edge": {}}},
        {"kind": "spawn", "features": {"edge": {"value": []}}},
        {"kind": "spawn", "features": {"edge": {"value": {}}}},
        {"kind": "spawn", "features": {"edge": {"value": {"src": 3}}}},
        spawn,
    ]
    all_nodes = nodes + [None, {}, {"id": ""}, {"id": "delta-lead"}]
    assert candidate_parent_metadata(all_nodes, malformed) == {
        "included": 3,
        "excluded": 16,
        "excluded_by_reason": {
            "edge_invalid_edge_feature": 1,
            "edge_invalid_features": 1,
            "edge_invalid_value": 1,
            "edge_missing_edge_feature": 1,
            "edge_missing_features": 1,
            "edge_missing_kind": 1,
            "edge_missing_value": 1,
            "edge_not_object": 1,
            "edge_source_duplicate_id": 1,
            "edge_source_invalid_id": 1,
            "edge_source_missing_id": 1,
            "edge_wrong_kind": 1,
            "node_duplicate_id": 1,
            "node_invalid_id": 1,
            "node_missing_id": 1,
            "node_not_object": 1,
        },
        "source_records": 19,
    }


# ---- parsing ---------------------------------------------------------------------------


def test_parse_keeps_valid_values_rejects_the_rest_and_counts_everything(parsed):
    preds, c, warnings = parsed
    assert [p["id"] for p in preds] == ["delta-impl", "delta-lead", "delta-reviewer"]
    assert all(p["tier"] == "reported" and p["model"] == MODEL and p["effort"] == EFFORT and p["prompt_version"] == PROMPT_VERSION and p["batch"] == 0 for p in preds)
    impl = preds[0]
    assert impl["labels"] == {"role": "dev", "fine_role": "implement", "parent": "delta-lead", "boundaries": ["2026-08-31T12:05:30Z"], "send_back": True, "approved": False}
    assert impl["confidence"] == 0.8 and impl["rejected"] == {} and "parser" in impl["evidence"]
    rev = preds[2]
    assert rev["labels"] == {"role": "planner", "fine_role": None, "parent": None, "boundaries": None, "send_back": None, "approved": True}
    assert rev["rejected"]["fine_role"]["value"] == "bogus" and "not one of" in rev["rejected"]["fine_role"]["reason"]
    assert rev["rejected"]["boundaries"] == {"value": ["not a time"], "reason": "1 entries are not ISO 8601 times"}
    assert rev["confidence"] is None
    # The repeated delta-lead entry is the second one; the first (role lead) is kept.
    assert preds[1]["labels"]["role"] == "lead"
    assert c["nodes_asked"] == 3 and c["nodes_answered"] == 3 and c["nodes_unanswered"] == 0
    assert c["entries_unknown_id"] == 1 and c["entries_repeated_id"] == 1
    assert c["values_rejected:fine_role"] == 1 and c["values_rejected:boundaries"] == 1 and c["values_rejected:confidence"] == 1
    assert c["keys_missing:parent"] == 1
    assert any("delta-ghost" in w for w in warnings) and any("answered again" in w for w in warnings) and any("bogus" in w for w in warnings)
    assert set(PREDICTION_KEYS) == set(impl["labels"])


def test_parse_counts_an_unanswered_node_and_refuses_an_answer_without_json(nodes):
    preds, c, warnings = parse_response('{"labels": [{"id": "delta-lead", "role": "lead"}]}', nodes, model=MODEL, effort=EFFORT, batch_index=3)
    assert [p["id"] for p in preds] == ["delta-lead"] and c["nodes_unanswered"] == 2
    assert preds[0]["labels"]["fine_role"] is None and c["keys_missing:fine_role"] == 1 and c["keys_missing:boundaries"] == 1
    assert sum(1 for w in warnings if "has no entry" in w) == 2
    with pytest.raises(LabelerError, match="no JSON object with a 'labels' key"):
        parse_response("I cannot label these.", nodes, model=MODEL, effort=EFFORT, batch_index=0)
    with pytest.raises(LabelerError, match="'labels' is not a list"):
        parse_response('{"labels": {"delta-lead": "lead"}}', nodes, model=MODEL, effort=EFFORT, batch_index=0)


def test_parse_counts_and_retains_missing_or_invalid_confidence_and_evidence(nodes):
    response = json.dumps({"labels": [
        {"id": "delta-impl", "role": "dev", "fine_role": None, "parent": "delta-lead", "boundaries": [], "send_back": None, "approved": None},
        {"id": "delta-lead", "role": "lead", "fine_role": None, "parent": None, "boundaries": [], "send_back": None, "approved": None, "confidence": "high", "evidence": []},
    ]})
    preds, counts, _warnings = parse_response(response, nodes[:2], model=MODEL, effort=EFFORT, batch_index=0)
    by_id = {pred["id"]: pred for pred in preds}
    assert counts["keys_missing:confidence"] == 1 and counts["keys_missing:evidence"] == 1
    assert counts["values_rejected:confidence"] == 1 and counts["values_rejected:evidence"] == 1
    assert by_id["delta-impl"]["rejected"]["confidence"] == {"value": None, "reason": "key is missing"}
    assert by_id["delta-impl"]["rejected"]["evidence"] == {"value": None, "reason": "key is missing"}
    assert by_id["delta-lead"]["rejected"]["confidence"] == {"value": "high", "reason": "not a number from 0 to 1"}
    assert by_id["delta-lead"]["rejected"]["evidence"] == {"value": [], "reason": "not a non-empty string"}


def test_predictions_round_trip_and_repeated_ids_are_counted(tmp_path, parsed):
    preds, _c, _w = parsed
    p = tmp_path / "preds.jsonl"
    write_predictions(preds, p)
    write_predictions(preds[:1], p, append=True)
    loaded, c, warnings = load_predictions(p)
    assert [x["id"] for x in loaded] == ["delta-impl", "delta-lead", "delta-reviewer"]
    assert c["predictions"] == 3 and c["predictions_repeated_id"] == 1 and "delta-impl" in warnings[0]
    assert loaded == preds
    with pytest.raises(LabelerError, match="does not exist"):
        load_predictions(tmp_path / "none.jsonl")


# ---- scoring ---------------------------------------------------------------------------


def test_gold_boundaries_are_attempt_starts_after_the_first(nodes):
    by_id = {n["id"]: n for n in nodes}
    assert gold_boundaries(by_id["delta-impl"]) == ([1788177900.0], 0)  # P1 n2 at 12:05
    assert gold_boundaries(by_id["delta-lead"]) == ([1788178200.0], 0)  # W0 n2 at 12:10
    assert gold_boundaries(by_id["delta-reviewer"]) == ([], 0)
    # One unparsed start makes the gold unknown (it could have been the first start): None, not a shorter list.
    assert gold_boundaries({"attempts": [{"started_at": "soon"}, {"started_at": "2026-08-31T12:00:00Z"}]}) == (None, 1)
    assert gold_boundaries({"attempts": [{"started_at": "2026-08-31T12:00:00Z"}, {"started_at": "2026-08-31T12:05:00Z"}, {"started_at": None}]}) == (None, 1)


def test_score_gives_every_figure_with_its_counts(items, parsed):
    preds, _c, _w = parsed
    s = score(items, preds)
    assert s["counts"] == {"edge_items": 2, "items_other": 0, "nodes": 3, "nodes_with_prediction": 3, "nodes_without_any_gold": 0, "nodes_without_prediction": 0, "predictions": 3}
    assert s["configurations"] == [{"model": MODEL, "effort": EFFORT, "prompt_version": PROMPT_VERSION, "predictions": 3}]
    assert s["prediction_tiers"] == {"reported": 3}
    r = s["role"]
    assert r["accuracy"] == {"value": 2 / 3, "numerator": 2, "denominator": 3}
    assert (r["gold_labeled"], r["predicted"], r["scored"], r["correct"]) == (3, 3, 3, 2)
    assert r["accuracy_answered"] == r["accuracy"] and r["abstained"] == 0 and r["wrong"] == 1
    assert r["by_gold_value"] == {"dev": {"n": 1, "correct": 1, "abstained": 0}, "lead": {"n": 1, "correct": 1, "abstained": 0}, "reviewer": {"n": 1, "correct": 0, "abstained": 0}}
    assert r["confusions"] == {"reviewer->planner": 1}
    assert r["gold_tiers"] == {"heuristic": 1, "verified": 2}
    f = s["fine_role"]
    # The reviewer's fine role came back outside the vocabulary, so it is null: an abstention,
    # wrong for the score (2 of 3 gold-labeled nodes), and counted apart from the answered figure.
    assert f["accuracy"] == {"value": 2 / 3, "numerator": 2, "denominator": 3} and f["accuracy_answered"] == {"value": 1.0, "numerator": 2, "denominator": 2}
    assert f["scored"] == 2 and f["abstained"] == 1 and f["predicted_null"] == 1 and f["gold_predicted_null"] == 1 and f["gold_tiers"] == {"hand": 3}
    assert f["by_gold_value"]["review"] == {"n": 1, "correct": 0, "abstained": 1}
    b = s["boundary"]
    assert b["gold_boundaries"] == 2 and b["nodes_with_attempts"] == 2 and b["nodes_with_gold_boundaries"] == 2
    assert b["predicted_boundaries"] == 1 and b["matched_by_count"] == 1 and b["matched_by_time"] == 1 and b["predicted_null"] == 1
    assert b["recall_by_count"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    assert b["recall_by_time"]["value"] == 0.5 and b["precision_by_count"]["value"] == 1.0
    # 30 s off the gold start: inside the default tolerance, outside a 10 s one.
    assert score(items, preds, tolerance_s=10)["boundary"]["matched_by_time"] == 0
    e = s["edge"]
    assert e["precision"] == {"value": 1.0, "numerator": 1, "denominator": 1} and e["recall"]["value"] == 1.0
    assert (e["gold_edges"], e["predicted_edges"], e["correct"], e["wrong"], e["predicted_none"], e["gold_predicted_none"], e["abstained"]) == (1, 1, 1, 0, 2, 0, 0)
    # The dataset's candidate edge items: the spawn edge is gold positive and asserted (the
    # prediction for delta-impl names delta-lead); the artifact edge is not something the
    # labeler predicts, so it is counted under its kind and gold, never scored.
    ce = s["candidate_edges"]
    assert ce["precision"] == {"value": 1.0, "numerator": 1, "denominator": 1} and ce["recall"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert (ce["edges"], ce["gold_labeled"], ce["gold_positive"], ce["gold_negative"], ce["tp"], ce["fp"], ce["fn"], ce["tn"], ce["abstained"]) == (2, 1, 1, 0, 1, 0, 0, 0, 0)
    assert ce["by_kind"] == {"artifact": 1, "spawn": 1} and ce["unscorable"] == {"artifact:gold_positive": 1} and ce["unscorable_kind"] == 1 and ce["gold_unknown"] == 0
    assert ce["gold_tiers"] == {"verified": 1}
    sb = s["send_back"]
    assert sb["precision"]["value"] is None and "no predicted true with a gold label" in sb["precision"]["reason"]
    assert sb["recall"]["value"] is None and sb["accuracy"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert (sb["tp"], sb["fp"], sb["fn"], sb["tn"], sb["predicted_without_gold"], sb["predicted_null"]) == (0, 0, 0, 1, 1, 1)
    ap = s["approved"]
    assert ap["precision"]["value"] == 1.0 and ap["recall"]["value"] == 1.0 and (ap["tp"], ap["predicted_without_gold"], ap["gold_true"], ap["gold_false"]) == (1, 2, 1, 0)
    # Attempt-level labels: delta-lead's two attempts (send_back false, approved true, twice)
    # and delta-impl's two disagreeing ones (P1 n1 sent back and not approved, P1 n2 the fix),
    # each scored against the one prediction for its session (send_back true, approved false
    # for delta-impl: right on n1, wrong on n2).
    sba, apa = s["send_back_attempts"], s["approved_attempts"]
    assert (sba["attempts"], sba["nodes_with_attempts"], sba["gold_labeled"], sba["gold_true"], sba["gold_false"], sba["attempts_on_disagreeing_sessions"]) == (4, 2, 4, 1, 3, 2)
    assert (sba["tp"], sba["fp"], sba["fn"], sba["tn"], sba["abstained"]) == (1, 1, 0, 2, 0)
    assert sba["precision"] == {"value": 0.5, "numerator": 1, "denominator": 2} and sba["recall"]["value"] == 1.0 and sba["accuracy"] == {"value": 0.75, "numerator": 3, "denominator": 4}
    assert (apa["tp"], apa["fp"], apa["fn"], apa["tn"], apa["gold_true"], apa["gold_false"]) == (2, 0, 1, 1, 3, 1)
    assert apa["recall"] == {"value": 2 / 3, "numerator": 2, "denominator": 3} and apa["gold_tiers"] == {"verified": 4}


def test_score_reports_coverage_correctness_and_majority_baseline_for_every_outcome(items, parsed):
    preds, _c, _w = parsed
    scores = score(items, preds)
    expected = {
        "role": (3, 3, 2, "dev", 1, 3),
        "fine_role": (2, 3, 2, "implement", 1, 3),
        "boundary": (2, 3, 1, 1, 2, 3),
        "edge": (1, 1, 1, "delta-lead", 1, 1),
        "candidate_edges": (1, 1, 1, True, 1, 1),
        "send_back": (1, 1, 1, False, 1, 1),
        "approved": (1, 1, 1, True, 1, 1),
        "send_back_attempts": (4, 4, 3, False, 3, 4),
        "approved_attempts": (4, 4, 3, True, 3, 4),
    }
    for key, (covered, total, correct, prediction, baseline_correct, baseline_total) in expected.items():
        metric = scores[key]
        assert metric["coverage"] == {"value": covered / total, "numerator": covered, "denominator": total}
        assert metric["correctness"] == {"value": correct / covered, "numerator": correct, "denominator": covered}
        baseline = metric["majority_class_baseline"]
        assert baseline["prediction"] == prediction
        assert baseline["accuracy"] == {
            "value": baseline_correct / baseline_total,
            "numerator": baseline_correct,
            "denominator": baseline_total,
        }

    text = "\n".join(report_lines(scores))
    assert text.count("majority-class baseline: always predict") == len(expected)
    assert "fine_role coverage: 0.667 (2/3); correctness among covered: 1.000 (2/2)" in text
    assert "boundary count coverage: 0.667 (2/3); correctness among covered: 0.500 (1/2)" in text
    assert "send-back attempt coverage: 1.000 (4/4); correctness among covered: 0.750 (3/4)" in text


def test_score_counts_nodes_without_predictions_and_wrong_edges(items, parsed):
    preds, _c, _w = parsed
    wrong = json.loads(json.dumps(preds[0]))
    wrong["labels"]["parent"] = "delta-reviewer"
    s = score(items, [wrong, {"item": "prediction", "id": "not-here", "labels": {}, "tier": "reported"}])
    assert s["counts"]["predictions_for_unknown_ids"] == 1 and s["counts"]["nodes_without_prediction"] == 2
    # Two gold-labeled nodes without a prediction are wrong for the score: 1 of 3, not 1 of 1.
    r = s["role"]
    assert r["gold_without_prediction"] == 2 and r["abstained"] == 2 and r["accuracy"] == {"value": 1 / 3, "numerator": 1, "denominator": 3}
    assert r["accuracy_answered"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert r["by_gold_value"]["lead"] == {"n": 1, "correct": 0, "abstained": 1}
    e = s["edge"]
    assert e["wrong"] == 1 and e["precision"]["value"] == 0.0 and e["recall"] == {"value": 0.0, "numerator": 0, "denominator": 1}
    ce = s["candidate_edges"]
    assert (ce["tp"], ce["fn"], ce["fn_other_parent"], ce["fn_null_parent"], ce["fn_no_prediction"]) == (0, 1, 1, 0, 0) and ce["recall"]["value"] == 0.0
    assert ce["precision"]["value"] is None and "no asserted candidate edge" in ce["precision"]["reason"]
    assert s["boundary"]["gold_without_prediction"] == 1
    # The approved gold on delta-lead (true) has no prediction at all: a false negative, counted as an abstention.
    ap = s["approved"]
    assert (ap["gold_true"], ap["fn"], ap["fn_abstained"], ap["abstained"], ap["gold_without_prediction"]) == (1, 1, 1, 1, 1)
    assert ap["recall"] == {"value": 0.0, "numerator": 0, "denominator": 1} and ap["accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 1}
    apa = s["approved_attempts"]
    assert (apa["attempts_without_prediction"], apa["fn_abstained"], apa["gold_false_abstained"], apa["fn"], apa["tn"]) == (2, 2, 0, 3, 1)
    empty = score([], [])
    assert empty["role"]["accuracy"]["value"] is None and empty["edge"]["recall"]["value"] is None and empty["boundary"]["recall_by_count"]["value"] is None
    assert empty["candidate_edges"]["recall"]["value"] is None and empty["send_back_attempts"]["recall"]["value"] is None
    for key in ("role", "fine_role", "boundary", "edge", "candidate_edges", "send_back", "approved", "send_back_attempts", "approved_attempts"):
        assert empty[key]["coverage"]["value"] is None
        assert empty[key]["correctness"]["value"] is None
        assert empty[key]["majority_class_baseline"]["prediction"] is None
        assert empty[key]["majority_class_baseline"]["class_counts"] == []
    empty_report = "\n".join(report_lines(empty))
    assert "majority-class baseline: always predict null" not in empty_report
    assert empty_report.count("majority-class baseline: unavailable") == 9


def test_null_predictions_are_wrong_answers_and_counted_as_abstentions(items, parsed):
    """Abstaining (null) never lifts a score: role accuracy keeps every gold-labeled node
    in its denominator, and a null on a gold positive is a false negative."""
    preds, _c, _w = parsed
    silent = json.loads(json.dumps(preds))
    for p in silent:
        p["labels"] = {k: None for k in PREDICTION_KEYS}
    s = score(items, silent)
    r = s["role"]
    assert r["accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 3} and r["abstained"] == 3 and r["gold_predicted_null"] == 3 and r["scored"] == 0
    assert r["accuracy_answered"]["value"] is None and "non-null prediction" in r["accuracy_answered"]["reason"]
    ap = s["approved"]  # delta-lead: gold approved true
    assert (ap["gold_true"], ap["fn"], ap["fn_abstained"], ap["tp"], ap["tn"], ap["abstained"]) == (1, 1, 1, 0, 0, 1)
    assert ap["recall"] == {"value": 0.0, "numerator": 0, "denominator": 1} and ap["accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 1}
    sb = s["send_back"]  # delta-lead: gold send_back false; a null is wrong for accuracy but not a false positive
    assert (sb["gold_false"], sb["gold_false_abstained"], sb["fp"], sb["tn"], sb["abstained"]) == (1, 1, 0, 0, 1)
    assert sb["accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 1} and sb["precision"]["value"] is None
    sba = s["send_back_attempts"]  # four attempts: one gold true (P1 n1), three gold false
    assert (sba["attempts_predicted_null"], sba["fn_abstained"], sba["gold_false_abstained"], sba["abstained"]) == (4, 1, 3, 4)
    assert sba["recall"] == {"value": 0.0, "numerator": 0, "denominator": 1} and sba["accuracy"] == {"value": 0.0, "numerator": 0, "denominator": 4}
    ce = s["candidate_edges"]  # the spawn edge is gold positive; a null parent on its destination misses it
    assert (ce["fn"], ce["fn_null_parent"], ce["abstained"], ce["tp"]) == (1, 1, 1, 0) and ce["recall"] == {"value": 0.0, "numerator": 0, "denominator": 1}
    assert s["edge"]["abstained"] == 1 and s["edge"]["recall"] == {"value": 0.0, "numerator": 0, "denominator": 1}
    assert s["boundary"]["predicted_null"] == 3 and s["boundary"]["gold_predicted_null"] == 2 and s["boundary"]["recall_by_count"] == {"value": 0.0, "numerator": 0, "denominator": 2}


def test_candidate_edges_score_negatives_unknowns_and_other_kinds(items, parsed):
    preds, _c, _w = parsed
    by_id = {p["id"]: p for p in preds}
    edges = [it for it in items if it["item"] == "edge"]
    spawn = next(e for e in edges if e["kind"] == "spawn")

    def edge(eid, kind, src, dst, gold_edge):
        e = json.loads(json.dumps(spawn))
        e["id"], e["kind"], e["gold_edge"] = eid, kind, gold_edge
        e["features"]["edge"]["value"].update({"kind": kind, "src": src, "dst": dst})
        return e

    synthetic = [
        spawn,  # positive, asserted: tp
        edge("spawn:delta-reviewer->delta-impl", "spawn", "delta-reviewer", "delta-impl", {"value": "negative", "tier": "verified", "how": "x", "reason": "the labeled parent is another node"}),  # not asserted (other parent): tn
        edge("launch:delta-impl->delta-lead", "launch", "delta-impl", "delta-lead", {"value": "negative", "tier": "heuristic", "how": "x", "reason": "the labeled parent is another node"}),  # lead predicted a null parent: tn
        edge("launch:delta-lead->delta-reviewer", "launch", "delta-lead", "delta-reviewer", {"value": "positive", "tier": "heuristic", "how": "x"}),  # reviewer's parent key was missing from the answer: fn
        edge("spawn:delta-lead->ghost", "spawn", "delta-lead", "ghost", {"value": "positive", "tier": "verified", "how": "x"}),  # destination not in the dataset
        edge("spawn:delta-impl->delta-lead", "spawn", "delta-impl", "delta-lead", {"value": None, "reason": "the labels carry no parent for the destination node"}),
        edge("artifact:delta-lead->delta-impl:/p", "artifact", "delta-lead", "delta-impl", {"value": "negative", "tier": "hand", "how": "x", "reason": "the destination is not a labeled consumer"}),
    ]
    nodes = [it for it in items if it["item"] == "node"]
    ce = score(nodes + synthetic, preds)["candidate_edges"]
    assert (ce["edges"], ce["gold_labeled"], ce["gold_positive"], ce["gold_negative"]) == (7, 4, 2, 2)
    assert (ce["tp"], ce["fp"], ce["fn"], ce["tn"]) == (1, 0, 1, 2)
    assert (ce["fn_null_parent"], ce["tn_other_parent"], ce["tn_null_parent"], ce["abstained"]) == (1, 1, 1, 2)
    assert (ce["dst_not_in_dataset"], ce["gold_unknown"], ce["unscorable_kind"]) == (1, 1, 1)
    assert ce["gold_unknown_reasons"] == {"the labels carry no parent for the destination node": 1} and ce["unscorable"] == {"artifact:gold_negative": 1}
    assert ce["by_kind"] == {"artifact": 1, "launch": 2, "spawn": 4} and ce["gold_tiers"] == {"heuristic": 2, "verified": 2}
    assert ce["precision"] == {"value": 1.0, "numerator": 1, "denominator": 1} and ce["recall"] == {"value": 0.5, "numerator": 1, "denominator": 2}
    # Asserting a gold-negative edge is a false positive.
    wrong = json.loads(json.dumps(by_id["delta-impl"]))
    wrong["labels"]["parent"] = "delta-reviewer"
    ce2 = score(nodes + synthetic[:2], [wrong])["candidate_edges"]
    assert (ce2["tp"], ce2["fp"], ce2["fn"], ce2["fn_other_parent"]) == (0, 1, 1, 1) and ce2["precision"] == {"value": 0.0, "numerator": 0, "denominator": 1}


def test_unparsed_attempt_start_makes_the_boundary_gold_unavailable(items, parsed):
    preds, _c, _w = parsed
    broken = json.loads(json.dumps(items))
    lead = next(it for it in broken if it["id"] == "delta-lead")
    lead["attempts"][0]["started_at"] = "soon"
    b = score(broken, preds)["boundary"]
    # delta-lead's one boundary is unknown, not silently zero: it leaves every denominator.
    assert (b["nodes_gold_unavailable"], b["attempt_starts_unparsed"], b["nodes_with_attempts"], b["nodes_with_gold_boundaries"]) == (1, 1, 2, 1)
    assert b["gold_boundaries"] == 1 and b["recall_by_count"] == {"value": 1.0, "numerator": 1, "denominator": 1}
    assert b["predicted_boundaries_on_unavailable"] == 0 and b["predicted_boundaries"] == 1
    lead["attempts"][0]["started_at"] = "2026-08-31T12:00:00Z"
    impl = next(it for it in broken if it["id"] == "delta-impl")
    impl["attempts"][1]["started_at"] = None
    b = score(broken, preds)["boundary"]
    assert (b["nodes_gold_unavailable"], b["attempt_starts_unparsed"], b["predicted_boundaries_on_unavailable"], b["gold_boundaries"]) == (1, 1, 1, 1)
    assert b["recall_by_count"] == {"value": 0.0, "numerator": 0, "denominator": 1} and b["predicted_boundaries"] == 0
    assert "nodes_gold_unavailable" in b["note"]
    assert "gold unavailable on 1 nodes with 1 attempt starts unparsed, 1 boundaries predicted on them" in "\n".join(report_lines(score(broken, preds)))


# ---- cost ------------------------------------------------------------------------------


def test_cost_figures_measure_the_bound_and_never_estimate(items, parsed):
    preds, _c, _w = parsed
    s = score(items, preds)
    good = [{"ref": "a", "usd": 0.004, "wall_s": 30.0}, {"ref": "b", "usd": 0.002, "wall_s": 20.0}]
    c = cost_figures(s, items, good)
    assert c["labeled_sessions"] == 3 and c["labeled_sessions_unpriced"] == 0 and c["labeled_usd"] == pytest.approx(1.7)
    assert c["labeler_usd"] == pytest.approx(0.006) and c["labeler_wall_s"] == 50.0 and c["bound"] == COST_BOUND == 0.005
    assert c["ratio"] == pytest.approx(0.006 / 1.7) and c["within_bound"] is True
    assert c["role_correct"] == 2 and c["role_correct_per_usd"] == pytest.approx(2 / 0.006) and c["role_correct_per_second"] == pytest.approx(2 / 50)
    over = cost_figures(s, items, [{"ref": "a", "usd": 0.5, "wall_s": 30.0}])
    assert over["within_bound"] is False and over["ratio"] == pytest.approx(0.5 / 1.7)
    part = cost_figures(s, items, [good[0], {"ref": "cx_missing", "usd": None, "reason": "no log file found for the reference under the log roots"}])
    assert part["ratio"] is None and "within_bound" not in part and "cx_missing" in part["ratio_reason"] and part["labeler_sessions_unpriced"] == 1
    assert part["role_correct_per_usd"] is None and "not fully known" in part["per_usd_reason"]
    none = cost_figures(s, items, [])
    assert none["ratio"] is None and "no labeler session" in none["ratio_reason"] and none["role_correct_per_second"] is None
    # An unpriced labeled session is counted and the labeled total marked a lower bound.
    partial = json.loads(json.dumps(items))
    partial[0]["features"]["skeleton"]["usd"] = None
    c2 = cost_figures(s, partial, good)
    assert c2["labeled_sessions_unpriced"] == 1 and c2["labeled_usd_is_lower_bound"] is True and c2["labeled_usd"] == pytest.approx(1.2)
    # A lower bound is no denominator: no ratio and no bound verdict, with the reason.
    assert c2["ratio"] is None and "within_bound" not in c2 and "1 of 3 labeled sessions are unpriced" in c2["ratio_reason"]
    # The labeler's own cost is still known, so the per-dollar figure stays.
    assert c2["labeler_usd"] == pytest.approx(0.006) and c2["role_correct_per_usd"] == pytest.approx(2 / 0.006)
    t = "\n".join(report_lines(s, dict(c2, sessions=[])))
    assert "cost bound 0.500%: not measured: 1 of 3 labeled sessions are unpriced" in t and "(a lower bound: some are unpriced)" in t and "within" not in t.split("cost bound")[1]


def test_price_sessions_says_when_a_session_cannot_be_found():
    assert find_session_files("no-uuid-here") == []
    out = price_sessions(["no-uuid-here"])
    assert out == [{"ref": "no-uuid-here", "path": None, "usd": None, "reason": "no log file found for the reference under the log roots"}]


def test_price_sessions_selects_one_matching_record_and_leaves_no_match_unpriced(tmp_path, monkeypatch):
    from loopmath import grade as grade_mod, ingest, price as price_mod
    from loopmath.graph import labeler

    session_file = tmp_path / "multi.jsonl"
    session_file.write_text("{}\n", encoding="utf-8")
    records = [
        {"run_id": "cx_other", "session_path": "/logs/other.jsonl", "model": MODEL, "effort": EFFORT, "wall_s": 2.0, "tokens": {}, "usd": 0.02},
        {"run_id": "cx_target", "session_path": "/logs/target.jsonl", "model": MODEL, "effort": EFFORT, "wall_s": 3.0, "tokens": {}, "usd": 0.03},
    ]
    monkeypatch.setattr(labeler, "find_session_files", lambda _ref: [session_file])
    monkeypatch.setattr(ingest, "parse_all", lambda *_args, **_kwargs: (records, {}))
    monkeypatch.setattr(grade_mod, "grade_all", lambda value: (value, {}))
    monkeypatch.setattr(price_mod, "load_prices", lambda _path: {})
    monkeypatch.setattr(price_mod, "price_all", lambda value, _table: (value, []))

    matched = price_sessions(["cx_target"])[0]
    assert matched["run_id"] == "cx_target" and matched["usd"] == 0.03 and matched["records_in_file"] == 2
    unmatched = price_sessions(["cx_missing"])[0]
    assert unmatched["usd"] is None and unmatched["records_in_file"] == 2
    assert "none matches the invocation" in unmatched["reason"]


def test_report_prints_every_figure_and_count(items, parsed):
    preds, _c, _w = parsed
    s = score(items, preds)
    cost = cost_figures(s, items, [{"ref": "a", "usd": 0.004, "wall_s": 30.0, "path": "/x/a.jsonl", "run_id": "cx_a", "model": "gpt-5.6-luna", "effort": "low"}, {"ref": "b", "usd": None, "reason": "no log file found for the reference under the log roots", "path": None}])
    cost["sessions"] = [{"ref": "a", "usd": 0.004, "wall_s": 30.0, "path": "/x/a.jsonl", "run_id": "cx_a", "model": "gpt-5.6-luna", "effort": "low"}, {"ref": "b", "usd": None, "reason": "no log file found for the reference under the log roots", "path": None}]
    text = "\n".join(report_lines(s, cost))
    assert "role accuracy: 0.667 (2/3) over every gold-labeled node (among answered 0.667 (2/3))" in text and "confusions (gold->predicted): reviewer->planner 1" in text
    assert "fine_role accuracy: 0.667 (2/3) over every gold-labeled node (among answered 1.000 (2/2)); gold labeled 3, predicted 2, scored 2, correct 2, wrong 0, abstained 1" in text
    assert "candidate edge precision: 1.000 (1/1); recall: 1.000 (1/1); edge items 2 (artifact 1, spawn 1)" in text and "not predicted by the labeler (kind:gold): artifact:gold_positive 1" in text
    assert "send-back detection per contract-v3 attempt: precision 0.500 (1/2), recall 1.000 (1/1), accuracy 0.750 (3/4); tp 1, fp 1, fn 0 (abstained 0), tn 2" in text
    assert "approval detection per contract-v3 attempt: precision 1.000 (2/2), recall 0.667 (2/3)" in text and "on sessions whose attempts disagree 2" in text
    assert "gold unavailable on 0 nodes with 0 attempt starts unparsed" in text
    assert "boundary recall by count: 0.500 (1/2)" in text and "gold boundaries 2 on 2 nodes" in text
    assert "edge precision: 1.000 (1/1); recall: 1.000 (1/1)" in text
    assert "send-back detection: precision None (no predicted true with a gold label; 0/0)" in text
    assert "approval detection: precision 1.000 (1/1), recall 1.000 (1/1)" in text
    assert "session b: not priced: no log file found" in text and "usd unknown" in text
    assert "cost bound 0.500%: not measured: 1 of 2 labeler sessions could not be priced" in text
    assert "prediction tiers: reported 3" in text and "gpt-5.6-luna low prompt v3: 3 predictions" in text
    assert "—" not in text and not FORBIDDEN.search(text)
    within = cost_figures(s, items, [{"ref": "a", "usd": 0.004, "wall_s": 30.0}])
    within["sessions"] = []
    t2 = "\n".join(report_lines(s, within))
    assert "labeler over labeled 0.23529%: within the bound" in t2 and "per dollar 500.00; per second 0.0667 (labeler wall clock 30.0 s)" in t2


# ---- session ids the calls leave behind -------------------------------------------------


def test_session_ids_from_codex_events_and_claude_json(tmp_path):
    ev = tmp_path / "events.jsonl"
    ev.write_text('not json\n{"type":"item.started","item":{"id":"item_0"}}\n{"type":"thread.started","thread_id":"01a05f87-a900-7943-8341-f59c96324e7a"}\n', encoding="utf-8")
    assert session_id_from_events(ev) == "01a05f87-a900-7943-8341-f59c96324e7a"
    ev.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
    assert session_id_from_events(ev) is None
    cj = tmp_path / "answer.json"
    cj.write_text(json.dumps({"type": "result", "session_id": "9d3c1a2b-1111-4222-8333-444455556666", "result": FAKE_RESPONSE, "total_cost_usd": 0.01}), encoding="utf-8")
    assert claude_json_result(cj) == ("9d3c1a2b-1111-4222-8333-444455556666", FAKE_RESPONSE)
    cj.write_text("{}", encoding="utf-8")
    assert claude_json_result(cj) == (None, None)
    cj.write_text("nope", encoding="utf-8")
    with pytest.raises(LabelerError):
        claude_json_result(cj)


# ---- CLI round trip and the grid runner's dry run ---------------------------------------------


def test_cli_prompt_parse_score_round_trip(tmp_path, dataset_path, capsys):
    prompt = tmp_path / "prompt-0.txt"
    assert main(["prompt", "--dataset", str(dataset_path), "--batch", "0", "--out", str(prompt)]) == 0
    loaded_nodes, loaded_edges, _ = load_dataset(dataset_path)
    assert prompt.read_text(encoding="utf-8") == build_prompt(loaded_nodes, all_nodes=loaded_nodes, edges=loaded_edges)
    prompt_out = capsys.readouterr().out
    assert "parent candidates included 3, excluded 2 by reason: edge_source_duplicate_id 1, edge_wrong_kind 1" in prompt_out
    assert main(["batches", "--dataset", str(dataset_path)]) == 0
    out = capsys.readouterr().out
    assert "batches 1" in out and "batch 0: 3 nodes, 3 parent candidates: delta-impl, delta-lead, delta-reviewer" in out
    assert main(["prompt", "--dataset", str(dataset_path), "--batch", "1", "--out", str(prompt)]) == 1
    assert "batch 1 does not exist; the dataset has 1 batches" in capsys.readouterr().err
    answer = tmp_path / "answer-0.txt"
    answer.write_text(FAKE_RESPONSE, encoding="utf-8")
    preds = tmp_path / "preds.jsonl"
    assert main(["parse", "--dataset", str(dataset_path), "--batch", "0", "--response", str(answer), "--model", MODEL, "--effort", EFFORT, "--out", str(preds)]) == 0
    out = capsys.readouterr().out
    assert "asked 3, answered 3, unanswered 0" in out and "entries_unknown_id 1" in out and "appended 3 predictions" in out
    assert "parent candidates included 3, excluded 2 by reason: edge_source_duplicate_id 1, edge_wrong_kind 1" in out
    prediction_items = [json.loads(line) for line in preds.read_text(encoding="utf-8").splitlines()]
    assert all(item["prompt_metadata"]["parent_candidates"]["excluded_by_reason"] == {"edge_source_duplicate_id": 1, "edge_wrong_kind": 1} for item in prediction_items)
    assert all(item["prompt_version"] == item["prompt_metadata"]["prompt_version"] == "v3" for item in prediction_items)
    v2_prompt = tmp_path / "prompt-v2.txt"
    assert main(["prompt", "--dataset", str(dataset_path), "--batch", "0", "--prompt-version", "v2", "--out", str(v2_prompt)]) == 0
    assert "PROMPT_VERSION: v2" in v2_prompt.read_text(encoding="utf-8")
    assert "wrote prompt v2" in capsys.readouterr().out
    v2_preds = tmp_path / "preds-v2.jsonl"
    assert main(["parse", "--dataset", str(dataset_path), "--batch", "0", "--response", str(answer), "--model", MODEL, "--effort", EFFORT, "--prompt-version", "v2", "--out", str(v2_preds)]) == 0
    assert all(item["prompt_version"] == item["prompt_metadata"]["prompt_version"] == "v2" for item in map(json.loads, v2_preds.read_text(encoding="utf-8").splitlines()))
    assert all(item["prompt_metadata"]["attempt_evidence"]["attempt_records_included"] == 4 for item in map(json.loads, v2_preds.read_text(encoding="utf-8").splitlines()))
    for version in ("v3", "v3-strict", "v4"):
        version_prompt = tmp_path / f"prompt-{version}.txt"
        assert main(["prompt", "--dataset", str(dataset_path), "--batch", "0", "--prompt-version", version, "--out", str(version_prompt)]) == 0
        assert f"PROMPT_VERSION: {version}" in version_prompt.read_text(encoding="utf-8")
        assert f"wrote prompt {version}" in capsys.readouterr().out
        version_preds = tmp_path / f"preds-{version}.jsonl"
        assert main(["parse", "--dataset", str(dataset_path), "--batch", "0", "--response", str(answer), "--model", MODEL, "--effort", EFFORT, "--prompt-version", version, "--out", str(version_preds)]) == 0
        parsed_versions = list(map(json.loads, version_preds.read_text(encoding="utf-8").splitlines()))
        assert all(item["prompt_version"] == item["prompt_metadata"]["prompt_version"] == version for item in parsed_versions)
        if version in ("v3", "v4"):
            assert all(item["prompt_metadata"]["attempt_evidence"]["attempt_records_included"] == 4 for item in parsed_versions)
            if version == "v4":
                assert all(item["prompt_metadata"]["session_evidence"]["node_collections"] == 3 for item in parsed_versions)
                assert all("workflow_evidence" in item["prompt_metadata"] for item in parsed_versions)
        else:
            assert all("attempt_evidence" not in item["prompt_metadata"] for item in parsed_versions)
    capsys.readouterr()
    bad = tmp_path / "bad.txt"
    bad.write_text("no labels here", encoding="utf-8")
    assert main(["parse", "--dataset", str(dataset_path), "--batch", "0", "--response", str(bad), "--model", MODEL, "--effort", EFFORT, "--out", str(tmp_path / "unused.jsonl")]) == 1
    assert "no JSON object with a 'labels' key" in capsys.readouterr().err
    assert not (tmp_path / "unused.jsonl").exists()
    cj = tmp_path / "answer.json"
    cj.write_text(json.dumps({"session_id": "9d3c1a2b-1111-4222-8333-444455556666", "result": FAKE_RESPONSE}), encoding="utf-8")
    assert main(["session", "--claude-json", str(cj)]) == 0 and capsys.readouterr().out.strip() == "9d3c1a2b-1111-4222-8333-444455556666"
    scores = tmp_path / "scores.json"
    assert main(["score", "--dataset", str(dataset_path), "--predictions", str(preds), "--session", "no-uuid-here", "--out", str(scores)]) == 0
    out = capsys.readouterr().out
    assert "role accuracy: 0.667 (2/3)" in out and "session no-uuid-here: not priced: no log file found" in out and "cost bound 0.500%: not measured" in out
    doc = json.loads(scores.read_text(encoding="utf-8"))
    assert doc["scores"]["role"]["correct"] == 2 and doc["cost"]["ratio"] is None and doc["cost"]["sessions"][0]["ref"] == "no-uuid-here"
    assert set(doc) == {"dataset", "predictions", "scores", "cost"}  # no clock: the same inputs write the same bytes
    again = tmp_path / "scores-again.json"
    assert main(["score", "--dataset", str(dataset_path), "--predictions", str(preds), "--session", "no-uuid-here", "--out", str(again)]) == 0
    capsys.readouterr()
    assert again.read_bytes() == scores.read_bytes()


def test_grid_dry_run_prints_the_command_and_the_batch_count_without_calling(tmp_path, dataset_path):
    script = ROOT / "tools" / "grid.sh"
    r = subprocess.run(["bash", str(script), "--dry-run", "--dataset", str(dataset_path), "--out-dir", str(tmp_path / "grid"), "gpt-5.6-luna", "low"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "batches: 1 (up to 20 node items each)" in r.stdout
    assert "prompt version v3" in r.stdout and "--prompt-version v3" in r.stdout
    assert "codex exec --model gpt-5.6-luna -c model_reasoning_effort=low --sandbox read-only --skip-git-repo-check --cd " in r.stdout
    assert "--json -o " in r.stdout and "prompt-0.txt > " in r.stdout and "events-0.jsonl" in r.stdout
    assert "score output contract: coverage, correctness among covered, and majority-class baseline for all 9 outcomes" in r.stdout
    assert "dry run: nothing was called, nothing was written" in r.stdout
    assert not (tmp_path / "grid").exists()
    r = subprocess.run(["bash", str(script), "--dry-run", "--dataset", str(dataset_path), "--out-dir", str(tmp_path / "grid"), "claude-haiku-4-5", "default"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "claude -p --model claude-haiku-4-5 --output-format json --tools ''" in r.stdout and "--effort" not in r.stdout.split("exact model command")[1]
    r = subprocess.run(["bash", str(script), "--dry-run", "--prompt-version", "v2", "--dataset", str(dataset_path), "--out-dir", str(tmp_path / "grid-v2"), "gpt-5.6-terra", "high"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "prompt version v2" in r.stdout and "labeler prompt --dataset" in r.stdout and "--prompt-version v2" in r.stdout
    assert not (tmp_path / "grid-v2").exists()
    r = subprocess.run(["bash", str(script), "--dry-run", "--dataset", str(tmp_path / "missing.jsonl"), "gpt-5.6-terra", "xhigh"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "FAILED: dataset" in r.stderr and "does not exist" in r.stderr
    r = subprocess.run(["bash", str(script), "--dry-run", "gpt-5.6-luna", "ultra"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "effort ultra is not one of" in r.stderr
    r = subprocess.run(["bash", str(script), "--dry-run", "gpt-4", "low"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "model gpt-4 is not one of" in r.stderr
    for version in ("v3", "v3-strict", "v4"):
        r = subprocess.run(["bash", str(script), "--dry-run", "--prompt-version", version, "--dataset", str(dataset_path), "--out-dir", str(tmp_path / f"grid-{version}"), "gpt-5.6-luna", "low"], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        assert f"prompt version {version}" in r.stdout and f"--prompt-version {version}" in r.stdout
        assert not (tmp_path / f"grid-{version}").exists()
    r = subprocess.run(["bash", str(script), "--dry-run", "--prompt-version", "v5", "gpt-5.6-luna", "low"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "prompt version v5 is not one of v1, v2, v3, v3-strict, v4" in r.stderr


@pytest.mark.parametrize("model,effort", [("gpt-5.6-terra", "high"), ("gpt-5.6-luna", "low")])
def test_grid_v4_dry_run_covers_requested_paid_configurations(tmp_path, dataset_path, model, effort):
    script = ROOT / "tools" / "grid.sh"
    out_dir = tmp_path / f"grid-v4-{model}-{effort}"
    result = subprocess.run(
        [
            "bash", str(script), "--dry-run", "--prompt-version", "v4",
            "--dataset", str(dataset_path), "--out-dir", str(out_dir), model, effort,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert f"model {model}, effort {effort}" in result.stdout
    assert "prompt version v4" in result.stdout and "--prompt-version v4" in result.stdout
    assert "dry run: nothing was called, nothing was written" in result.stdout
    assert not out_dir.exists()


def test_grid_dry_run_resolves_relative_out_dir_before_model_working_directory(tmp_path, dataset_path):
    script = ROOT / "tools" / "grid.sh"
    r = subprocess.run(
        ["bash", str(script), "--dry-run", "--dataset", str(dataset_path), "--out-dir", "grid-rel", "gpt-5.6-luna", "low"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 0, r.stderr
    absolute = tmp_path / "grid-rel"
    assert f"predictions would be written to: {absolute}/gpt-5.6-luna-low.jsonl" in r.stdout
    assert f"work directory: {absolute}/work/gpt-5.6-luna-low" in r.stdout
    assert f"--cd {absolute}/work/gpt-5.6-luna-low" in r.stdout
    assert not absolute.exists()


def test_grid_success_prints_score_contract_after_scoring_and_writes_all_outcomes(tmp_path, dataset_path):
    script = ROOT / "tools" / "grid.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_codex = fake_bin / "codex"
    fake_codex.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("-o") + 1])
out.write_text(os.environ["LOOPMATH_TEST_LABELER_RESPONSE"], encoding="utf-8")
print('{"type":"thread.started","thread_id":"00000000-0000-4000-8000-000000000001"}')
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    out_dir = tmp_path / "grid"
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["LOOPMATH_TEST_LABELER_RESPONSE"] = FAKE_RESPONSE
    result = subprocess.run(
        [
            "bash", str(script), "--dataset", str(dataset_path),
            "--out-dir", str(out_dir), "gpt-5.6-luna", "low",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    contract = "grid: score output contract satisfied: coverage, correctness among covered, and majority-class baseline for all 9 outcomes"
    assert contract in result.stdout
    assert result.stdout.index("wrote scores to") < result.stdout.index(contract)
    assert result.stdout.count("majority-class baseline: always predict") == 9
    assert "role coverage:" in result.stdout and "correctness among covered:" in result.stdout
    document = json.loads((out_dir / "gpt-5.6-luna-low.scores.json").read_text(encoding="utf-8"))
    outcomes = (
        "role", "fine_role", "boundary", "edge", "candidate_edges",
        "send_back", "approved", "send_back_attempts", "approved_attempts",
    )
    for outcome in outcomes:
        assert set(document["scores"][outcome]) >= {
            "coverage", "correctness", "majority_class_baseline",
        }
