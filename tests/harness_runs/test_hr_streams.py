"""Usage read back from OpenCode and Codex streams: a missing or malformed count is never read as 0.

Lines are written as the tools write them (codex `turn.completed` usage, `substrate/codex.py:432-444` in
harness2; OpenCode `step_finish` parts, `third_party/lab_runner/opencode/session.py:44-49`).
"""

from __future__ import annotations

import json

import pytest

from loopmath import harness_runs as HR

FULL_TURN = {"input_tokens": 3000, "cached_input_tokens": 1000, "output_tokens": 200, "reasoning_output_tokens": 50}
FULL_STEP = {"input": 1200, "output": 80, "reasoning": 20, "cache": {"read": 400, "write": 10}}


def _codex(tmp_path, *usages) -> dict:
    lines = [{"type": "thread.started", "thread_id": "th"}]
    for use in usages:
        lines += [{"type": "turn.started"}, {"type": "turn.completed", "usage": use}]
    path = tmp_path / "codex.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    return HR.rollout_usage(path, "codex")


def _opencode(tmp_path, *steps) -> dict:
    lines = []
    for tok, cost in steps:
        part = {"type": "step-finish", "tokens": tok}
        if cost is not None:
            part["cost"] = cost
        lines += [{"type": "step_start", "part": {"type": "step-start"}}, {"type": "step_finish", "part": part}]
    path = tmp_path / "oc.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    return HR.rollout_usage(path, "opencode")


def test_a_full_codex_turn_is_complete(tmp_path):
    got = _codex(tmp_path, FULL_TURN)
    assert got["complete"] is True and got["incomplete"] is None
    assert got["tokens"] == {"input": 2000, "cache_read": 1000, "cache_write": 0, "output": 200}


@pytest.mark.parametrize("usage", [
    {},
    {"input_tokens": 3000, "cached_input_tokens": 1000},
    {"input_tokens": 3000, "output_tokens": 200},
    {"input_tokens": 3000, "cached_input_tokens": 1000, "output_tokens": None},
    {"input_tokens": 3000, "cached_input_tokens": 1000, "output_tokens": 2.5},
    {"input_tokens": 3000, "cached_input_tokens": 1000, "output_tokens": -1},
    {"input_tokens": 3000, "cached_input_tokens": 1000, "output_tokens": True},
    {"input_tokens": 100, "cached_input_tokens": 1000, "output_tokens": 200},
    None,
], ids=["empty", "no-output", "no-cached", "null-output", "float", "negative", "bool", "cached-over-input",
        "no-usage"])
def test_a_codex_turn_with_missing_counts_is_incomplete_never_zero(tmp_path, usage):
    got = _codex(tmp_path, FULL_TURN, usage)
    assert got["complete"] is False and got["usd"] is None
    assert "1 turn ended (turn.completed) with token counts missing or not whole numbers" in got["incomplete"]
    assert got["tokens"] == {"input": 2000, "cache_read": 1000, "cache_write": 0, "output": 200}   # the good turn


def test_a_codex_stream_whose_only_turn_has_empty_usage_is_unknown_and_unpriced(tmp_path):
    path = tmp_path / "codex.jsonl"
    path.write_text(json.dumps({"type": "turn.started"}) + "\n"
                    + json.dumps({"type": "turn.completed", "usage": {}}) + "\n", encoding="utf-8")
    cost = HR.rollout_cost(path, "codex", model="gpt-5.4")
    assert cost["complete"] is False and cost["usd"] is None and cost["tokens"] is None
    assert cost["usd_source"] is None and "incomplete" in cost["unpriced"]


def test_a_full_opencode_step_is_complete_and_an_absent_reasoning_adds_0(tmp_path):
    no_reasoning = {k: v for k, v in FULL_STEP.items() if k != "reasoning"}
    got = _opencode(tmp_path, (FULL_STEP, 0.01), (no_reasoning, 0.02))
    assert got["complete"] is True and got["usd"] == pytest.approx(0.03)
    assert got["tokens"] == {"input": 2400, "cache_read": 800, "cache_write": 20, "output": 180}


@pytest.mark.parametrize("tokens", [
    None,
    {},
    {"input": 1200, "reasoning": 20, "cache": {"read": 400, "write": 10}},
    {"input": 1200, "output": 80, "reasoning": 20},
    {"input": 1200, "output": 80, "reasoning": 20, "cache": {"read": 400}},
    {"input": 1200, "output": "80", "reasoning": 20, "cache": {"read": 400, "write": 10}},
    {"input": 1200, "output": 80, "reasoning": -5, "cache": {"read": 400, "write": 10}},
], ids=["none", "empty", "no-output", "no-cache", "no-cache-write", "string", "negative-reasoning"])
def test_an_opencode_step_with_missing_counts_is_incomplete_never_zero(tmp_path, tokens):
    got = _opencode(tmp_path, (FULL_STEP, 0.01), (tokens, 0.02))
    assert got["complete"] is False and got["usd"] is None and got["usd_source"] is None
    assert "1 step ended (step_finish) with token counts missing or not whole numbers" in got["incomplete"]
    assert got["partial_usd"] == pytest.approx(0.03)     # both steps' reported dollars, as a note only


def test_text_with_line_separators_does_not_split_a_stream_line(tmp_path):
    text = "one\u2028two\u2029three\u0085four"
    path = tmp_path / "codex.jsonl"
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in [
        {"type": "turn.started"}, {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": FULL_TURN}]), encoding="utf-8")
    got = HR.rollout_usage(path, "codex")
    assert got["skipped_lines"] == 0 and got["complete"] is True
    assert got["tokens"] == {"input": 2000, "cache_read": 1000, "cache_write": 0, "output": 200}


def test_measured_zeros_are_a_complete_zero(tmp_path):
    got = _codex(tmp_path, {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0})
    assert got["complete"] is True
    assert got["tokens"] == {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}


def test_a_named_stream_file_that_cannot_be_read_makes_the_counts_incomplete(tmp_path):
    good = tmp_path / "attempt1.jsonl"
    good.write_text(json.dumps({"type": "turn.started"}) + "\n"
                    + json.dumps({"type": "turn.completed", "usage": FULL_TURN}) + "\n", encoding="utf-8")
    gone = tmp_path / "attempt2.jsonl"
    cost = HR.rollout_cost([good, gone], "codex", model="gpt-5.4")
    assert cost["complete"] is False and cost["usd"] is None and cost["usd_source"] is None
    assert "1 stream file could not be read: attempt2.jsonl" in cost["incomplete"]
    assert cost["tokens"] == {"input": 2000, "cache_read": 1000, "cache_write": 0, "output": 200}   # kept, not priced
    assert cost["files"] == 1
    whole = HR.rollout_cost([good], "codex", model="gpt-5.4")
    assert whole["complete"] is True and whole["usd"] > 0


def test_an_unreadable_opencode_file_keeps_the_reported_dollars_as_a_note(tmp_path):
    got = _opencode(tmp_path, (FULL_STEP, 0.01))
    assert got["complete"] is True
    got = HR.rollout_usage([tmp_path / "oc.jsonl", tmp_path / "retry.jsonl"], "opencode")
    assert got["complete"] is False and got["usd"] is None and got["partial_usd"] == pytest.approx(0.01)
