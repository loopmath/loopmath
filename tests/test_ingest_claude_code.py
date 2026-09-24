"""Tests for the Claude Code session parser.

All fixtures are small synthetic JSONL files built in `tmp_path`; none of this
depends on the user's real `~/.claude/projects` logs.
"""

import json
from pathlib import Path

from loopmath.ingest.claude_code import _check_outcome, _normalize_check_cmd, parse_session

CWD = "/Users/x/Workspace/proj-a"
FIXTURES = Path(__file__).parent / "fixtures"


def _write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def _assistant(
    *,
    ts="2026-08-31T10:00:00Z",
    session_id="sess-1",
    cwd=CWD,
    request_id=None,
    model="claude-opus-5",
    effort="xhigh",
    content=None,
    usage=None,
    stop_reason="tool_use",
):
    line = {
        "type": "assistant",
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {
            "model": model,
            "role": "assistant",
            "content": content if content is not None else [{"type": "text", "text": "ok"}],
            "stop_reason": stop_reason,
            "usage": usage or {},
        },
    }
    if request_id is not None:
        line["requestId"] = request_id
    if effort is not None:
        line["effort"] = effort
    return line


def _user(*, ts="2026-08-31T10:00:01Z", session_id="sess-1", cwd=CWD, content=None, tool_use_result=None):
    line = {
        "type": "user",
        "timestamp": ts,
        "sessionId": session_id,
        "cwd": cwd,
        "message": {"role": "user", "content": content if content is not None else "hi"},
    }
    if tool_use_result is not None:
        line["toolUseResult"] = tool_use_result
    return line


def test_token_summing_four_streams_and_request_id_dedup(tmp_path):
    p = tmp_path / "session.jsonl"
    usage_a = {
        "input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 5,
        "output_tokens": 50,
    }
    lines = [
        # Split physical lines of one logical turn: same requestId, usage
        # repeated on both, must count only once.
        _assistant(request_id="req_A", usage=usage_a, content=[{"type": "thinking", "thinking": ""}]),
        _assistant(request_id="req_A", usage=usage_a, content=[{"type": "text", "text": "hi"}]),
        # A different request: counts separately.
        _assistant(
            request_id="req_B",
            usage={
                "input_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 10,
                "output_tokens": 80,
            },
        ),
        # No requestId at all: always counts.
        _assistant(
            usage={
                "input_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 5,
            },
        ),
        # req_A once more, now far from its first two lines. Deduplication is
        # over the whole session, not over adjacent lines, so this third copy
        # must not move any total either.
        _assistant(request_id="req_A", usage=usage_a, content=[{"type": "text", "text": "again"}]),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    d = rec.to_dict()["tokens"]
    # Four streams land one-to-one, nothing summed across streams: `in` no
    # longer absorbs `cache_creation_input_tokens`.
    assert d["in"] == 100 + 200 + 10
    assert d["cache_read"] == 5 + 10 + 0
    assert d["cache_write"] == 20 + 0 + 0
    assert d["out"] == 50 + 80 + 5


def test_streamed_request_growing_usage_keeps_per_stream_max(tmp_path):
    p = tmp_path / "session.jsonl"
    lines = [
        # Streamed response shape: the request's early physical lines carry a
        # placeholder output_tokens (1, then 2) and only the final line has
        # the real cumulative figure. The dedup must keep 120, not 1.
        _assistant(
            request_id="req_S",
            usage={
                "input_tokens": 300,
                "cache_creation_input_tokens": 40,
                "cache_read_input_tokens": 900,
                "output_tokens": 1,
            },
            content=[{"type": "thinking", "thinking": ""}],
        ),
        _assistant(
            request_id="req_S",
            usage={
                "input_tokens": 300,
                "cache_creation_input_tokens": 40,
                "cache_read_input_tokens": 900,
                "output_tokens": 2,
            },
            content=[{"type": "text", "text": "hi"}],
        ),
        _assistant(
            request_id="req_S",
            usage={
                "input_tokens": 300,
                "cache_creation_input_tokens": 40,
                "cache_read_input_tokens": 900,
                "output_tokens": 120,
            },
            content=[{"type": "text", "text": "done"}],
        ),
        # A line whose usage block is missing entirely must not pin the
        # request's streams at zero when a later line does carry usage.
        _assistant(request_id="req_T", usage={}, content=[{"type": "thinking", "thinking": ""}]),
        _assistant(
            request_id="req_T",
            usage={
                "input_tokens": 50,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 30,
            },
            content=[{"type": "text", "text": "late"}],
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    d = rec.to_dict()["tokens"]
    assert d["in"] == 300 + 50
    assert d["cache_read"] == 900 + 0
    assert d["cache_write"] == 40 + 0
    assert d["out"] == 120 + 30


def test_sidechain_usage_is_attributed_to_parent_once_and_subagents_recorded():
    rec = parse_session(FIXTURES / "claude_code_sidechain.jsonl")

    assert rec is not None
    record = rec.to_dict()
    # Parent usage plus three distinct sidechain requests. req-side-1 occurs
    # on two physical lines but contributes exactly once.
    assert record["tokens"] == {
        "in": 100 + 200 + 300 + 400,
        "cache_read": 20 + 50 + 80 + 110,
        "cache_write": 10 + 40 + 70 + 100,
        "out": 30 + 60 + 90 + 120,
    }
    assert record["subagents"] == 2


def test_four_streams_land_in_four_right_places(tmp_path):
    # Four distinct, pairwise-prime-ish values so any transposition (e.g.
    # cache_read and cache_write swapped) or any accidental summation (e.g.
    # cache_write folded into `in`) produces a detectably wrong number.
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(
            request_id="r1",
            usage={
                "input_tokens": 1001,
                "cache_read_input_tokens": 2003,
                "cache_creation_input_tokens": 3007,
                "output_tokens": 4009,
            },
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    d = rec.to_dict()["tokens"]
    assert d["in"] == 1001
    assert d["cache_read"] == 2003
    assert d["cache_write"] == 3007
    assert d["out"] == 4009


def test_usage_block_missing_cache_creation_field_yields_cache_write_zero(tmp_path):
    # The cache_creation field being simply absent (no cache write happened)
    # is a real, present value of 0, distinct from the whole usage block
    # being missing (which is still None, per
    # test_no_usage_anywhere_returns_none).
    p = tmp_path / "session.jsonl"
    usage = {
        "input_tokens": 50,
        "cache_read_input_tokens": 7,
        "output_tokens": 20,
        # cache_creation_input_tokens intentionally absent
    }
    lines = [_assistant(request_id="r1", usage=usage)]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    d = rec.to_dict()["tokens"]
    assert d["cache_write"] == 0
    assert d["in"] == 50
    assert d["cache_read"] == 7
    assert d["out"] == 20


def test_model_and_effort_majority_ignores_synthetic(tmp_path):
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(request_id="r1", model="claude-opus-5", effort="xhigh"),
        _assistant(request_id="r2", model="claude-opus-5", effort="xhigh"),
        _assistant(request_id="r3", model="claude-opus-5", effort="xhigh"),
        _assistant(request_id="r4", model="claude-sonnet-5", effort="high"),
        _assistant(request_id="r5", model="claude-sonnet-5", effort="high"),
        # Synthetic marker lines must never win the model vote, however many
        # of them there are.
        _assistant(request_id="r6", model="<synthetic>", effort=None),
        _assistant(request_id="r7", model="<synthetic>", effort=None),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.model == "opus-5"
    assert rec.effort == "xhigh"


def test_touched_dirs_and_written_file_kinds(tmp_path):
    p = tmp_path / "session.jsonl"
    write_block = {
        "type": "tool_use",
        "id": "toolu_write",
        "name": "Write",
        "input": {"file_path": f"{CWD}/src/foo.py", "content": "x"},
    }
    edit_block = {
        "type": "tool_use",
        "id": "toolu_edit",
        "name": "Edit",
        "input": {"file_path": f"{CWD}/src/bar.py", "old_string": "a", "new_string": "b"},
    }
    edit_block_2 = {
        "type": "tool_use",
        "id": "toolu_edit2",
        "name": "Edit",
        "input": {"file_path": f"{CWD}/tests/test_foo.py", "old_string": "a", "new_string": "b"},
    }
    read_block = {
        "type": "tool_use",
        "id": "toolu_read",
        "name": "Read",
        "input": {"file_path": f"{CWD}/README.md"},
    }
    lines = [
        _assistant(
            request_id="r1",
            content=[write_block, edit_block, edit_block_2, read_block],
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.touched_dirs == ["src", "tests"]
    assert rec.written_file_kinds == {"py": 3}


def test_wall_s_and_ts(tmp_path):
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(request_id="r1", ts="2026-08-31T10:00:00Z"),
        _user(ts="2026-08-31T10:02:00Z"),
        # No timestamp at all: must not affect min/max.
        {"type": "mode", "sessionId": "sess-1"},
        _assistant(request_id="r2", ts="2026-08-31T10:05:30Z"),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.ts == "2026-08-31T10:00:00Z"
    assert rec.wall_s == 330.0


def test_no_assistant_lines_returns_none(tmp_path):
    # Both cases below are counted as skips by `ingest.parse_all`, which is
    # where the exclusion is surfaced (see the report's first beat), not here.
    p = tmp_path / "session.jsonl"
    lines = [
        _user(),
        {"type": "system", "timestamp": "2026-08-31T10:00:00Z", "sessionId": "sess-1", "cwd": CWD},
        {"type": "mode", "sessionId": "sess-1"},
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is None


def test_no_usage_anywhere_returns_none(tmp_path):
    # Assistant lines exist, but none of them ever carried a `message.usage`
    # block at all (the key is absent, not merely empty): this session is
    # not analysable for cost, so it must return None rather than a record
    # that would silently price at $0.00. Counted as a skip by
    # `ingest.parse_all`, same as the no-assistant-lines case above.
    p = tmp_path / "session.jsonl"
    line = _assistant(request_id="r1")
    del line["message"]["usage"]
    lines = [line]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is None


def test_check_command_red_to_green(tmp_path):
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(
            request_id="r1",
            ts="2026-08-31T10:00:00Z",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "pytest -q"},
                }
            ],
        ),
        _user(
            ts="2026-08-31T10:00:05Z",
            content=[
                {
                    "tool_use_id": "toolu_1",
                    "type": "tool_result",
                    "content": "1 failed, 2 passed",
                    "is_error": True,
                }
            ],
            tool_use_result={"stdout": "1 failed, 2 passed", "stderr": "", "interrupted": False},
        ),
        _assistant(
            request_id="r2",
            ts="2026-08-31T10:01:00Z",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "Bash",
                    "input": {"command": "pytest -q"},
                }
            ],
        ),
        _user(
            ts="2026-08-31T10:01:05Z",
            content=[
                {
                    "tool_use_id": "toolu_2",
                    "type": "tool_result",
                    "content": "5 passed",
                    "is_error": False,
                }
            ],
            tool_use_result={"stdout": "5 passed", "stderr": "", "interrupted": False},
        ),
        _assistant(
            request_id="r3",
            ts="2026-08-31T10:02:00Z",
            content=[{"type": "text", "text": "All tests pass now."}],
            stop_reason="end_turn",
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.tests_red_to_green is True
    assert rec.check_exit_zero is True
    # bonus: the last assistant line's stop_reason and text are also read here
    assert rec.exit_ok is True
    assert rec.agent_claims_success is True


def _outcome_for_text(text, is_error=None):
    block = {"type": "tool_result", "content": text}
    if is_error is not None:
        block["is_error"] = is_error
    tool_result = {"stdout": text, "stderr": "", "interrupted": False}
    return _check_outcome(block, tool_result)


def test_check_outcome_zero_count_forms_are_not_failures():
    # Flag B regression: a count-aware read must not misclassify a clean run
    # as a failure just because "failed"/"error" appears as a substring.
    assert _outcome_for_text("5 passed, 0 failed") is True
    assert _outcome_for_text("Tests: 12 passed, 0 failed") is True
    assert _outcome_for_text("1 failed, 4 passed") is False
    # "0 errors" alone is not a failure, but it is also not an explicit pass
    # marker, so with no is_error flag present it falls through to None.
    assert _outcome_for_text("0 errors") is None
    assert _outcome_for_text("0 errors", is_error=False) is True
    assert _outcome_for_text("0 errors", is_error=True) is False


def test_check_command_cross_command_does_not_set_red_to_green(tmp_path):
    # Flag E regression: a failing pytest run followed by a passing ruff run
    # is not the same check, so tests_red_to_green must stay False even
    # though check_exit_zero is True (some recognized check did pass).
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(
            request_id="r1",
            ts="2026-08-31T10:00:00Z",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "pytest tests/a.py"},
                }
            ],
        ),
        _user(
            ts="2026-08-31T10:00:05Z",
            content=[
                {
                    "tool_use_id": "toolu_1",
                    "type": "tool_result",
                    "content": "1 failed, 2 passed",
                    "is_error": True,
                }
            ],
            tool_use_result={"stdout": "1 failed, 2 passed", "stderr": "", "interrupted": False},
        ),
        _assistant(
            request_id="r2",
            ts="2026-08-31T10:01:00Z",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "Bash",
                    "input": {"command": "ruff check ."},
                }
            ],
        ),
        _user(
            ts="2026-08-31T10:01:05Z",
            content=[
                {
                    "tool_use_id": "toolu_2",
                    "type": "tool_result",
                    "content": "All checks passed!",
                    "is_error": False,
                }
            ],
            tool_use_result={"stdout": "All checks passed!", "stderr": "", "interrupted": False},
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.check_exit_zero is True
    assert rec.tests_red_to_green is False


def test_check_command_same_command_normalized_red_to_green(tmp_path):
    # Flag E regression: the same underlying command, dressed up with a
    # trailing "2>&1 | tail -N" on the failing run and extra whitespace on
    # the passing run, still counts as the same check going from red to
    # green. (Case-folding of the command itself is covered separately by
    # test_normalize_check_cmd_folds_case_whitespace_and_trailing_noise:
    # _CHECK_CMD_RE, which decides whether a command is a check at all, is
    # itself case-sensitive to lowercase tool names like "pytest".)
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(
            request_id="r1",
            ts="2026-08-31T10:00:00Z",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Bash",
                    "input": {"command": "pytest -q 2>&1 | tail -20"},
                }
            ],
        ),
        _user(
            ts="2026-08-31T10:00:05Z",
            content=[
                {
                    "tool_use_id": "toolu_1",
                    "type": "tool_result",
                    "content": "1 failed, 2 passed",
                    "is_error": True,
                }
            ],
            tool_use_result={"stdout": "1 failed, 2 passed", "stderr": "", "interrupted": False},
        ),
        _assistant(
            request_id="r2",
            ts="2026-08-31T10:01:00Z",
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "Bash",
                    "input": {"command": "pytest   -q"},
                }
            ],
        ),
        _user(
            ts="2026-08-31T10:01:05Z",
            content=[
                {
                    "tool_use_id": "toolu_2",
                    "type": "tool_result",
                    "content": "5 passed, 0 failed",
                    "is_error": False,
                }
            ],
            tool_use_result={"stdout": "5 passed, 0 failed", "stderr": "", "interrupted": False},
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.tests_red_to_green is True
    assert rec.check_exit_zero is True


def test_normalize_check_cmd_folds_case_whitespace_and_trailing_noise():
    assert _normalize_check_cmd("pytest -q") == _normalize_check_cmd("PYTEST   -q  2>&1 | tail -20")
    assert _normalize_check_cmd("pytest tests/a.py") != _normalize_check_cmd("ruff check .")


def test_write_path_fallback_notebook_and_generic_path_keys(tmp_path):
    # Flag C regression: NotebookEdit keys its path as `notebook_path`, and a
    # generic `path` key is accepted as a last-resort fallback.
    p = tmp_path / "session.jsonl"
    notebook_block = {
        "type": "tool_use",
        "id": "toolu_nb",
        "name": "NotebookEdit",
        "input": {
            "notebook_path": f"{CWD}/notebooks/analysis.ipynb",
            "cell_id": "1",
            "new_source": "x",
        },
    }
    generic_path_block = {
        "type": "tool_use",
        "id": "toolu_generic",
        "name": "Write",
        "input": {"path": f"{CWD}/scripts/run.sh", "content": "x"},
    }
    lines = [
        _assistant(request_id="r1", content=[notebook_block, generic_path_block]),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.touched_dirs == ["notebooks", "scripts"]
    assert rec.written_file_kinds == {"ipynb": 1, "sh": 1}


def test_max_tokens_last_turn_is_exit_ok_false_and_never_timed_out(tmp_path):
    # Flag D regression: `stop_reason` describes only the last model turn.
    # `max_tokens` there means exit_ok is False and nothing else; these logs
    # never record a session timeout, so timed_out is always False.
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(request_id="r1", stop_reason="max_tokens"),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.exit_ok is False
    assert rec.timed_out is False


# --- Flag 5: run_id is unique per file, not per sessionId -------------------


def test_run_id_distinct_for_parent_and_subagent_sharing_session_id(tmp_path):
    # A sub-agent transcript at <project>/<sessionId>/subagents/agent-*.jsonl
    # carries the PARENT session's sessionId on every line, just like the
    # measured real estate. Both files must still get distinct run_ids.
    sid = "f3a33cad-a2b9-4ed1-b134-f6403e86af53"

    parent_path = tmp_path / f"{sid}.jsonl"
    _write_jsonl(parent_path, [_assistant(request_id="r1", session_id=sid)])

    subagents_dir = tmp_path / "subagents"
    subagents_dir.mkdir()
    sub_path = subagents_dir / "agent-a1a2d821eeb18073d.jsonl"
    _write_jsonl(sub_path, [_assistant(request_id="r2", session_id=sid)])

    parent_rec = parse_session(parent_path)
    sub_rec = parse_session(sub_path)
    assert parent_rec is not None
    assert sub_rec is not None
    assert parent_rec.run_id != sub_rec.run_id
    # The parent session stays identifiable from the sub-agent's run_id.
    assert sid in sub_rec.run_id
    assert parent_rec.run_id == f"cc_{sid}"


def test_run_id_reproducible_across_reparse(tmp_path):
    sid = "f3a33cad-a2b9-4ed1-b134-f6403e86af53"
    subagents_dir = tmp_path / "subagents"
    subagents_dir.mkdir()
    sub_path = subagents_dir / "agent-a1a2d821eeb18073d.jsonl"
    _write_jsonl(sub_path, [_assistant(request_id="r1", session_id=sid)])

    first = parse_session(sub_path)
    second = parse_session(sub_path)
    assert first is not None and second is not None
    assert first.run_id == second.run_id


# --- Flag 6: zero-token synthetic sessions are flagged explicitly ----------


def test_zero_token_synthetic_session_is_flagged(tmp_path):
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(
            request_id="r1",
            model="<synthetic>",
            usage={
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.zero_token_synthetic is True
    assert rec.tokens.total == 0
    assert rec.model is None


def test_real_model_zero_usage_session_is_not_flagged_synthetic(tmp_path):
    # A real, recognized model with genuinely zero usage must not be
    # confused with the harness's own synthetic bookkeeping turn: the flag
    # comes from the literal "<synthetic>" marker, never inferred from
    # "tokens happened to be zero".
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(
            request_id="r1",
            model="claude-opus-5",
            usage={
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.zero_token_synthetic is False


def test_mixed_synthetic_and_real_model_session_is_not_flagged_synthetic(tmp_path):
    # A session that mixes a real model with a synthetic bookkeeping line is
    # real work, not a synthetic session.
    p = tmp_path / "session.jsonl"
    lines = [
        _assistant(
            request_id="r1",
            model="claude-opus-5",
            usage={
                "input_tokens": 10,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 5,
            },
        ),
        _assistant(
            request_id="r2",
            model="<synthetic>",
            usage={
                "input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 0,
            },
        ),
    ]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.zero_token_synthetic is False
    assert rec.model == "opus-5"


# --- Flag 2: check_exit_zero is tri-state -----------------------------------


def test_check_exit_zero_none_when_no_check_ever_observed(tmp_path):
    p = tmp_path / "session.jsonl"
    lines = [_assistant(request_id="r1")]
    _write_jsonl(p, lines)

    rec = parse_session(p)
    assert rec is not None
    assert rec.check_exit_zero is None
