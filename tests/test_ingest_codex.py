"""Tests for the Codex rollout-log parser.

Fixtures are synthetic JSONL committed under ``tests/fixtures/codex`` or built
in ``tmp_path``; nothing here reads the user's real Codex session logs.
"""

import json
from pathlib import Path

from loopmath import ingest
from loopmath.grade import assign_attempts
from loopmath.ingest.codex import _classify_check_output, _extract_apply_patch_paths, parse_session


FIXTURES = Path(__file__).parent / "fixtures" / "codex"


def _write_jsonl(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
    )
    return path


def _session_meta(session_id="019fc9b3-3b31-79b1-b712-9c6010ba0a28", cwd="/repo", ts="2026-08-03T10:00:00.000Z"):
    return {
        "timestamp": ts,
        "type": "session_meta",
        "payload": {
            "session_id": session_id,
            "id": session_id,
            "timestamp": ts,
            "cwd": cwd,
            "originator": "codex_exec",
            "cli_version": "0.144.1",
            "source": "exec",
            "thread_source": "user",
            "model_provider": "openai",
            "base_instructions": {"text": "You are Codex."},
        },
    }


def _turn_context(model, effort, ts="2026-08-03T10:00:01.000Z", cwd="/repo"):
    return {
        "timestamp": ts,
        "type": "turn_context",
        "payload": {
            "turn_id": "turn-1",
            "cwd": cwd,
            "model": model,
            "effort": effort,
        },
    }


def _token_count(
    ts,
    input_tokens,
    cached_input_tokens,
    output_tokens,
    total_tokens,
    cache_write_input_tokens=0,
):
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "cache_write_input_tokens": cache_write_input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": 0,
                    "total_tokens": total_tokens,
                },
                "last_token_usage": {
                    "input_tokens": input_tokens,
                    "cached_input_tokens": cached_input_tokens,
                    "cache_write_input_tokens": cache_write_input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_output_tokens": 0,
                    "total_tokens": total_tokens,
                },
            },
        },
    }


def _patch_apply_end(ts, changes, success=True):
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {
            "type": "patch_apply_end",
            "success": success,
            "changes": changes,
        },
    }


def _task_started(ts, turn_id):
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {"type": "task_started", "turn_id": turn_id, "model_context_window": 258400},
    }


def _task_complete(ts, turn_id, last_agent_message="Done."):
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {"type": "task_complete", "turn_id": turn_id, "last_agent_message": last_agent_message},
    }


def _custom_tool_call(ts, call_id, cmd_text):
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "call_id": call_id, "name": "exec_command", "input": cmd_text},
    }


def _custom_tool_call_output(ts, call_id, output_text):
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "custom_tool_call_output", "call_id": call_id, "output": output_text},
    }


def test_resumed_files_are_one_v1_attempt_with_summed_tokens():
    records, diag = ingest.parse_all(
        logs=FIXTURES,
        harnesses=("codex",),
        use_cache=False,
    )
    records = assign_attempts(records)

    # attempt_rule v1 numbers logical runs, not rollout files. The two fixture
    # files continue the same persisted session id, so together they are the
    # documented first attempt rather than attempts 1 and 2.
    assert diag["files_seen"] == {"codex": 2, "total": 2}
    assert diag["records"] == {"codex": 1, "total": 1}
    assert len(records) == 1
    record = records[0]
    assert record["run_id"] == "cx_019fc9b3-3b31-79b1-b712-9c6010ba0a28"
    assert record["tokens"] == {
        "in": 2300,
        "cache_read": 300,
        "cache_write": 0,
        "out": 120,
    }
    assert record["attempt"] == 1
    assert record["attempt_rule"] == "v1"
    assert record["attempt_clause"] == "wall"


def test_cumulative_token_usage_only_last_line_counts(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 1000, 200, 50, 1050),
        _token_count("2026-08-03T10:00:03.000Z", 4000, 900, 300, 4300),
        # Last line is authoritative: cumulative, not a sum across lines.
        _token_count("2026-08-03T10:00:04.000Z", 5000, 2000, 800, 5800),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    tokens = rec.tokens.as_dict()
    # in_ excludes the cached part that input_tokens already includes.
    assert tokens == {
        "in": 5000 - 2000,
        "cache_read": 2000,
        "cache_write": 0,
        "out": 800,
    }


def test_four_streams_land_in_distinct_places(tmp_path):
    # Four distinct, mutually prime-ish values so that a transposed field or
    # an accidental sum of two fields both produce a visibly wrong number
    # rather than a coincidentally-correct one.
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count(
            "2026-08-03T10:00:02.000Z",
            input_tokens=9000,
            cached_input_tokens=1000,
            output_tokens=77,
            total_tokens=9077,
            cache_write_input_tokens=300,
        ),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    tokens = rec.tokens.as_dict()
    # in_ = input_tokens - cached_input_tokens = 9000 - 1000 = 8000.
    # cache_write_input_tokens (300) is a distinct field, read straight
    # through, never folded into in_ or summed with cache_read.
    assert tokens == {"in": 8000, "cache_read": 1000, "cache_write": 300, "out": 77}


def test_cache_write_is_zero_for_a_normal_codex_session(tmp_path):
    # Pinning finding, not a placeholder: sampling the real ~/.codex/sessions
    # estate (about 3,300 rollout files, 421k token_count lines, both old and
    # new CLI builds) found `cache_write_input_tokens` reading 0 every single
    # time it appeared, and older builds don't carry the field at all (as
    # here). OpenAI does not bill for cache writes, and no sampled session
    # ever reported a nonzero one, so 0 is the honest value, not a default
    # standing in for missing data.
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 1000, 200, 50, 1050),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.tokens.as_dict()["cache_write"] == 0


def test_compaction_reset_guard_adds_segments(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        # Segment 1 grows, then the cumulative counter is reset by compaction.
        _token_count("2026-08-03T10:00:02.000Z", 6000, 2000, 200, 6200),
        _token_count("2026-08-03T10:00:03.000Z", 10000, 4000, 500, 10500),
        {"timestamp": "2026-08-03T10:00:04.000Z", "type": "compacted", "payload": {"message": ""}},
        # Segment 2 starts back near zero: total_tokens drops, which is the
        # reset signal.
        _token_count("2026-08-03T10:00:05.000Z", 100, 20, 10, 130),
        _token_count("2026-08-03T10:00:06.000Z", 3000, 1000, 300, 3300),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    tokens = rec.tokens.as_dict()
    # Segment 1's last reading (10000/4000/500) must be folded in, not lost,
    # plus segment 2's last reading (3000/1000/300).
    acc_in = 10000 + 3000
    acc_cache = 4000 + 1000
    acc_out = 500 + 300
    assert tokens == {
        "in": acc_in - acc_cache,
        "cache_read": acc_cache,
        "cache_write": 0,
        "out": acc_out,
    }


def test_model_and_effort_majority_selection(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh", ts="2026-08-03T10:00:01.000Z"),
        _turn_context("gpt-5.6-sol", "xhigh", ts="2026-08-03T10:00:02.000Z"),
        _turn_context("gpt-5.6-sol", "xhigh", ts="2026-08-03T10:00:03.000Z"),
        _turn_context("gpt-5.6-terra", "low", ts="2026-08-03T10:00:04.000Z"),
        _token_count("2026-08-03T10:00:05.000Z", 100, 0, 10, 100),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.model == "gpt-5.6-sol"
    assert rec.effort == "xhigh"


def test_touched_dirs_and_written_file_kinds_ignore_failed_patch(tmp_path):
    lines = [
        _session_meta(cwd="/repo"),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _patch_apply_end(
            "2026-08-03T10:00:03.000Z",
            {
                "/repo/src/pkg/a.py": {"type": "add", "unified_diff": "irrelevant"},
                "/repo/src/pkg/b.md": {"type": "update", "unified_diff": "irrelevant"},
            },
            success=True,
        ),
        _patch_apply_end(
            "2026-08-03T10:00:04.000Z",
            {"/repo/should/not/count.txt": {"type": "add", "unified_diff": "irrelevant"}},
            success=False,
        ),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.touched_dirs == ["src/pkg"]
    assert rec.written_file_kinds == {"md": 1, "py": 1}


def test_wall_s_and_ts(tmp_path):
    lines = [
        _session_meta(ts="2026-08-03T10:00:00.000Z"),
        _turn_context("gpt-5.6-sol", "xhigh", ts="2026-08-03T10:01:00.000Z"),
        _token_count("2026-08-03T10:05:30.000Z", 100, 0, 10, 100),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.ts == "2026-08-03T10:00:00Z"
    assert rec.wall_s == 330.0  # 5 minutes 30 seconds after the earliest timestamp


def test_empty_or_meta_only_file_returns_none(tmp_path):
    # A genuinely empty file: nothing to parse at all.
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    assert parse_session(empty_path) is None

    # A file with lines but none of them a session_meta or a token_count:
    # nothing to price or grade.
    meta_only_lines = [
        {
            "timestamp": "2026-08-03T10:00:00.000Z",
            "type": "world_state",
            "payload": {"note": "nothing interesting"},
        },
    ]
    meta_only_path = _write_jsonl(tmp_path, "meta_only.jsonl", meta_only_lines)
    assert parse_session(meta_only_path) is None


def test_session_meta_present_but_no_token_usage_returns_none(tmp_path):
    # Flag 1 regression: a session_meta line by itself, with no token_count
    # event anywhere in the file, must not become a zero-token record that
    # would silently price at $0.00. The old gate only returned None when
    # BOTH session_meta and token usage were missing.
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    assert parse_session(path) is None


# --- Flag 2: check pass/fail classification -------------------------------


def test_classify_check_output_zero_passed_is_no_signal():
    # "0 passed" means nothing ran: neither a pass nor a failure.
    assert _classify_check_output("0 passed") is None


def test_classify_check_output_not_ok_is_failure():
    assert _classify_check_output("not ok 1 - build") is False


def test_classify_check_output_singular_error_with_passed_is_failure():
    # A real failure must win even though "passed" also appears in the same
    # summary line.
    assert _classify_check_output("1 error, 2 passed") is False


def test_classify_check_output_zero_failed_count_is_pass():
    # The zero-count exemption: "0 failed" must not be misread as a failure.
    assert _classify_check_output("5 passed, 0 failed") is True


def test_classify_check_output_zero_error_count_alone_is_no_signal():
    assert _classify_check_output("Ran suite. 0 errors.") is None


def test_classify_check_output_tap_ok_is_pass():
    assert _classify_check_output("ok 1 - build succeeded") is True


def test_classify_check_output_structured_exit_code_zero_wins_over_failing_text():
    # Path 1 (structured exit status) is authoritative: it must win even
    # when the same text also contains failing-looking markers.
    text = "Traceback (most recent call last):\n  ...\nProcess exited with code 0"
    assert _classify_check_output(text) is True


def test_classify_check_output_structured_exit_code_nonzero_wins_over_passing_text():
    text = '{"output":"1 passed\\n","metadata":{"exit_code":1,"duration_seconds":0.1}}'
    assert _classify_check_output(text) is False


def test_classify_check_output_plain_exit_code_zero_wins_over_failed_marker():
    assert _classify_check_output("noise\nexit code 0\nmore noise, FAILED") is True


def test_check_exit_zero_from_structured_exit_code_end_to_end(tmp_path):
    # The custom_tool_call_output shape actually seen in real rollout files:
    # a JSON-encoded string with a metadata.exit_code field.
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _custom_tool_call("2026-08-03T10:00:03.000Z", "call-1", 'tools.exec_command({cmd: "pytest"})'),
        _custom_tool_call_output(
            "2026-08-03T10:00:04.000Z",
            "call-1",
            '{"output":"5 passed\\n","metadata":{"exit_code":0,"duration_seconds":1.2}}',
        ),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.check_exit_zero is True


# --- Flag 4: exit_ok from the final task lifecycle only --------------------


def test_exit_ok_true_when_last_task_started_has_matching_complete(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _task_started("2026-08-03T10:00:03.000Z", "turn-1"),
        _task_complete("2026-08-03T10:00:04.000Z", "turn-1"),
        _task_started("2026-08-03T10:00:05.000Z", "turn-2"),
        _task_complete("2026-08-03T10:00:06.000Z", "turn-2"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.exit_ok is True


def test_exit_ok_false_when_last_task_started_never_completes(tmp_path):
    # An earlier completed task must not paper over the last task never
    # finishing.
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _task_started("2026-08-03T10:00:03.000Z", "turn-1"),
        _task_complete("2026-08-03T10:00:04.000Z", "turn-1"),
        _task_started("2026-08-03T10:00:05.000Z", "turn-2"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.exit_ok is False


def test_exit_ok_none_when_no_task_started_at_all(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.exit_ok is None


def test_exit_ok_ignores_failed_patch_apply_anywhere(tmp_path):
    # Flag 4 regression: a failed-then-retried patch is normal work, not a
    # dirty exit, and must not flip a clean final task completion to False.
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _task_started("2026-08-03T10:00:03.000Z", "turn-1"),
        _patch_apply_end(
            "2026-08-03T10:00:03.500Z",
            {"/repo/a.py": {"type": "update", "unified_diff": "irrelevant"}},
            success=False,
        ),
        _task_complete("2026-08-03T10:00:04.000Z", "turn-1"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.exit_ok is True


# --- Flag 1: red-to-green requires the SAME normalized command -------------


def _apply_patch_call(ts, call_id, patch_text="*** Begin Patch\n*** Update File: /repo/a.py\n*** End Patch\n"):
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "custom_tool_call", "call_id": call_id, "name": "apply_patch", "input": patch_text},
    }


def _apply_patch_success_output(ts, call_id, updated_lines):
    # Matches the real JSON-encoded shape sampled from rollout files: embedded
    # newlines inside this string are the literal two-character sequence
    # `\n`, never an actual newline byte.
    files_block = "\\n".join(updated_lines)
    output = (
        '{"output":"Success. Updated the following files:\\n'
        + files_block
        + '\\n","metadata":{"exit_code":0,"duration_seconds":0.0}}'
    )
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "custom_tool_call_output", "call_id": call_id, "output": output},
    }


def _apply_patch_failed_output(ts, call_id):
    # Real shape: no "Updated the following files" marker, and the message
    # quotes surrounding source lines -- exactly the diff-adjacent content
    # this parser must never read into a record.
    text = (
        "apply_patch verification failed: Failed to find expected lines in "
        "/repo/a.py:\ndef existing_function():\n    pass\n"
    )
    return {
        "timestamp": ts,
        "type": "response_item",
        "payload": {"type": "custom_tool_call_output", "call_id": call_id, "output": text},
    }


def test_red_to_green_does_not_fire_across_different_commands(tmp_path):
    # Regression for the real false case the reviewer found: a failing check
    # followed by a passing DIFFERENT check must not count as verification.
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _custom_tool_call("2026-08-03T10:00:03.000Z", "call-1", 'tools.exec_command({cmd: "vitest run"})'),
        _custom_tool_call_output("2026-08-03T10:00:04.000Z", "call-1", "1 failed, 2 passed"),
        _custom_tool_call("2026-08-03T10:00:05.000Z", "call-2", 'tools.exec_command({cmd: "tsc --noEmit"})'),
        _custom_tool_call_output("2026-08-03T10:00:06.000Z", "call-2", "5 passed, 0 failed"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.tests_red_to_green is False
    # The passing tsc check still sets check_exit_zero True: that field is
    # command-agnostic on purpose.
    assert rec.check_exit_zero is True


def test_red_to_green_fires_on_same_command_retry(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _custom_tool_call("2026-08-03T10:00:03.000Z", "call-1", 'tools.exec_command({cmd: "pytest -q"})'),
        _custom_tool_call_output("2026-08-03T10:00:04.000Z", "call-1", "1 failed, 2 passed"),
        _custom_tool_call("2026-08-03T10:00:05.000Z", "call-2", 'tools.exec_command({cmd: "pytest -q"})'),
        _custom_tool_call_output("2026-08-03T10:00:06.000Z", "call-2", "5 passed, 0 failed"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.tests_red_to_green is True
    assert rec.check_exit_zero is True


# --- Flag 2: check_exit_zero is tri-state -----------------------------------


def test_check_exit_zero_is_none_when_no_check_ever_observed(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.check_exit_zero is None


def test_check_exit_zero_is_false_when_every_observed_check_failed(tmp_path):
    lines = [
        _session_meta(),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _custom_tool_call("2026-08-03T10:00:03.000Z", "call-1", 'tools.exec_command({cmd: "pytest -q"})'),
        _custom_tool_call_output("2026-08-03T10:00:04.000Z", "call-1", "1 failed, 2 passed"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.check_exit_zero is False


# --- `_OK_RE` tightening: an actual status token, not "ok" inside prose -----


def test_ok_re_does_not_match_ok_inside_prose():
    assert _classify_check_output("looks ok, moving on to the next step") is None


def test_ok_re_still_matches_go_test_status_line():
    assert _classify_check_output("ok  \tpackage/path\t0.005s") is True


# --- Flag 4: `apply_patch` write metadata (the modern shape) ----------------


def test_apply_patch_success_extracts_write_metadata(tmp_path):
    lines = [
        _session_meta(cwd="/repo"),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _apply_patch_call("2026-08-03T10:00:03.000Z", "call-1"),
        _apply_patch_success_output(
            "2026-08-03T10:00:04.000Z",
            "call-1",
            ["M /repo/src/pkg/a.py", "A /repo/src/pkg/b.md"],
        ),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.touched_dirs == ["src/pkg"]
    assert rec.written_file_kinds == {"md": 1, "py": 1}


def test_apply_patch_failure_contributes_no_write_metadata(tmp_path):
    lines = [
        _session_meta(cwd="/repo"),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _apply_patch_call("2026-08-03T10:00:03.000Z", "call-1"),
        _apply_patch_failed_output("2026-08-03T10:00:04.000Z", "call-1"),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    assert rec.touched_dirs == []
    assert rec.written_file_kinds == {}


def test_apply_patch_input_never_treated_as_a_check_command(tmp_path):
    # A patch body that happens to mention a check-like word (e.g. patching a
    # test file) must not be classified as a check command: `input` on an
    # `apply_patch` call is a diff body, never a shell command.
    patch_text = (
        "*** Begin Patch\n*** Update File: /repo/tests/test_x.py\n@@\n"
        "-def test_pytest_marker(): pass\n+def test_pytest_marker(): assert True\n"
        "*** End Patch\n"
    )
    lines = [
        _session_meta(cwd="/repo"),
        _turn_context("gpt-5.6-sol", "xhigh"),
        _token_count("2026-08-03T10:00:02.000Z", 100, 0, 10, 100),
        _apply_patch_call("2026-08-03T10:00:03.000Z", "call-1", patch_text=patch_text),
        _apply_patch_success_output(
            "2026-08-03T10:00:04.000Z", "call-1", ["M /repo/tests/test_x.py"]
        ),
    ]
    path = _write_jsonl(tmp_path, "session.jsonl", lines)
    rec = parse_session(path)
    assert rec is not None
    # No check was ever observed, so this must be None (unobserved), never a
    # false pass/fail derived from reading the diff body.
    assert rec.check_exit_zero is None
    assert rec.tests_red_to_green is False


# --- `_extract_apply_patch_paths` unit tests --------------------------------


def test_extract_apply_patch_paths_handles_json_encoded_shape():
    text = (
        '{"output":"Success. Updated the following files:\\nA /repo/a.py\\n'
        'M /repo/b.py\\n","metadata":{"exit_code":0,"duration_seconds":0.0}}'
    )
    assert _extract_apply_patch_paths(text) == ["/repo/a.py", "/repo/b.py"]


def test_extract_apply_patch_paths_handles_plain_text_shape():
    text = "Exit code: 0\nWall time: 0 seconds\nOutput:\nSuccess. Updated the following files:\nM /repo/c.py\n"
    assert _extract_apply_patch_paths(text) == ["/repo/c.py"]


def test_extract_apply_patch_paths_empty_when_marker_absent():
    text = "apply_patch verification failed: Failed to find expected lines in /repo/a.py:\nsome code\n"
    assert _extract_apply_patch_paths(text) == []
