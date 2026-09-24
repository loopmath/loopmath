"""Session matching on the anonymized fixtures (spec 03 section 6).

`fixtures/` holds four real sessions from the build machine rewritten by
`make_fixtures.py`: every id fake, every folder `/work/repo`, no text, token
counts kept. `expected.json` records what the builder matched.

- Claude Code `...0001`: parent on fable-5 with seven sub-agents on fable-5,
  sonnet-5 and opus-5, 2026-09-20 17:02Z.
- Codex `...0001`: `codex exec` on gpt-6-sol, 18:00Z.
- Codex `...0002`: `codex-tui` on gpt-5.6-sol, 19:00Z.
- Codex `...0003`: Codex Desktop on gpt-6-astra, 20:00Z, with three child
  threads `...0004` to `...0006`.
"""

import json
import shutil
from pathlib import Path

import pytest

from loopmath.logmatch.match import (
    LogRoots,
    SessionError,
    default_roots,
    explain_match,
    match_attempt,
    resolve_session,
)

FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))
CC = "00000000-0000-4000-8000-000000000001"
CX_EXEC = "019f0000-0000-7000-8000-000000000001"
CX_TUI = "019f0000-0000-7000-8000-000000000002"
CX_DESKTOP = "019f0000-0000-7000-8000-000000000003"
CX_CHILD = "019f0000-0000-7000-8000-000000000004"


@pytest.fixture
def roots():
    return LogRoots(claude=[FIXTURES / "claude" / "projects"], codex=[FIXTURES / "codex" / "sessions"])


@pytest.mark.parametrize("sid", sorted(EXPECTED))
def test_exact_match_by_session_id_is_verified(roots, sid):
    want = EXPECTED[sid]
    match = match_attempt({"session": sid, "harness": want["harness"]}, roots=roots)
    assert match is not None
    assert match.tier == "verified"
    assert match.session_id == sid
    assert match.harness == want["harness"]
    assert match.tokens.as_dict() == want["tokens"]
    assert match.models == want["models"]
    assert len(match.children) == want["children"]
    assert (match.started_at, match.ended_at) == (want["started_at"], want["ended_at"])
    assert match.effort == want["effort"]


def test_exact_match_without_harness_finds_either_log(roots):
    assert match_attempt({"session": CC}, roots=roots).harness == "claude-code"
    assert match_attempt({"session": CX_EXEC}, roots=roots).harness == "codex"


def test_claude_children_are_the_subagent_files_each_at_its_own_model(roots):
    match = match_attempt({"session": CC, "harness": "claude-code"}, roots=roots)
    folder = FIXTURES / "claude" / "projects" / "-work-repo" / CC / "subagents"
    assert sorted(match.children) == sorted(p.stem[len("agent-"):] for p in folder.glob("agent-*.jsonl"))
    assert match.model == "fable-5"
    child_models = {p["model"] for p in match.parts if p["role"] == "child"}
    assert child_models == {"fable-5", "sonnet-5", "opus-5"}
    total = [0, 0, 0, 0]
    for part in match.parts:
        for i, v in enumerate(part["tokens"].as_dict().values()):
            total[i] += v
    assert total == list(match.tokens.as_dict().values())


def test_codex_child_threads_add_into_the_parent_and_are_listed(roots):
    match = match_attempt({"session": CX_DESKTOP, "harness": "codex"}, roots=roots)
    assert sorted(match.children) == [
        "019f0000-0000-7000-8000-000000000004",
        "019f0000-0000-7000-8000-000000000005",
        "019f0000-0000-7000-8000-000000000006",
    ]
    assert {p["parent"] for p in match.parts if p["role"] == "child"} == {CX_DESKTOP}
    parent_only = [p for p in match.parts if p["role"] == "session"]
    assert len(parent_only) == 1
    assert parent_only[0]["tokens"].total < match.tokens.total


def test_a_child_thread_named_directly_matches_alone(roots):
    match = match_attempt({"session": CX_CHILD, "harness": "codex"}, roots=roots)
    assert match is not None and match.session_id == CX_CHILD
    assert match.children == []


def test_a_missing_session_id_is_no_match_and_never_a_guess(roots):
    attempt = {
        "session": "019f0000-0000-7000-8000-00000000dead",
        "harness": "codex",
        "cwd": "/work/repo",
        "started_at": "2026-09-20T18:00:00Z",
    }
    match, reason = explain_match(attempt, roots=roots)
    assert match is None
    assert "not found" in reason


def test_self_in_a_run_file_is_not_resolved_at_match_time(roots):
    match, reason = explain_match({"session": "self", "harness": "claude-code"}, roots=roots)
    assert match is None
    assert "CLAUDE_CODE_SESSION_ID" in reason


def test_an_unsafe_session_value_is_rejected(roots):
    match, reason = explain_match({"session": "../*", "harness": "claude-code"}, roots=roots)
    assert match is None
    assert "not a session id" in reason


# --- heuristic ----------------------------------------------------------------


def test_one_candidate_in_folder_window_and_model_is_heuristic(roots):
    attempt = {
        "harness": "codex",
        "cwd": "/work/repo",
        "started_at": "2026-09-20T18:00:10Z",
        "ended_at": "2026-09-20T18:01:00Z",
        "model": {"raw": "gpt-6-sol"},
    }
    match, reason = explain_match(attempt, roots=roots)
    assert match is not None and match.session_id == CX_EXEC
    assert match.tier == "heuristic"
    assert reason.startswith("one session")


def test_claude_heuristic_uses_the_first_timestamp(roots):
    attempt = {"harness": "claude-code", "cwd": "/work/repo", "started_at": "2026-09-20T17:01:00Z"}
    match = match_attempt(attempt, roots=roots)
    assert match is not None and match.session_id == CC and match.tier == "heuristic"
    assert len(match.children) == 7


def test_more_than_one_candidate_is_no_match(roots):
    attempt = {
        "harness": "codex",
        "cwd": "/work/repo",
        "started_at": "2026-09-20T18:00:00Z",
        "ended_at": "2026-09-20T19:05:00Z",
    }
    match, reason = explain_match(attempt, roots=roots)
    assert match is None
    assert "2 candidate sessions" in reason


def test_the_model_narrows_candidates(roots):
    attempt = {
        "harness": "codex",
        "cwd": "/work/repo",
        "started_at": "2026-09-20T18:00:00Z",
        "ended_at": "2026-09-20T19:05:00Z",
        "model": "gpt-5.6-sol",
    }
    match = match_attempt(attempt, roots=roots)
    assert match is not None and match.session_id == CX_TUI


def test_a_wrong_model_or_folder_or_window_is_no_match(roots):
    base = {"harness": "codex", "cwd": "/work/repo", "started_at": "2026-09-20T18:00:10Z", "model": "gpt-6-sol"}
    assert match_attempt({**base, "model": "gpt-6-luna"}, roots=roots) is None
    assert match_attempt({**base, "cwd": "/work/other"}, roots=roots) is None
    assert match_attempt({**base, "started_at": "2026-09-20T18:10:00Z"}, roots=roots) is None


def test_child_threads_are_never_candidates(roots):
    # The three children start inside this window in the same folder.
    attempt = {"harness": "codex", "cwd": "/work/repo", "started_at": "2026-09-20T19:59:30Z", "ended_at": "2026-09-20T20:01:00Z"}
    match = match_attempt(attempt, roots=roots)
    assert match is not None and match.session_id == CX_DESKTOP
    assert len(match.children) == 3


def test_sessions_already_matched_are_excluded(roots):
    attempt = {
        "harness": "codex",
        "cwd": "/work/repo",
        "started_at": "2026-09-20T18:00:00Z",
        "ended_at": "2026-09-20T19:05:00Z",
    }
    match, _ = explain_match(attempt, roots=roots, exclude={CX_EXEC})
    assert match is not None and match.session_id == CX_TUI


def test_heuristic_needs_a_window_and_a_folder(roots):
    assert "no started_at" in explain_match({"harness": "codex", "cwd": "/work/repo"}, roots=roots)[1]
    assert "no cwd" in explain_match({"harness": "codex", "started_at": "2026-09-20T18:00:00Z"}, roots=roots)[1]


def test_a_harness_without_logs_is_no_match(roots):
    match, reason = explain_match({"harness": "command", "started_at": "2026-09-20T18:00:00Z"}, roots=roots)
    assert match is None and "no session logs" in reason


def test_archived_flat_codex_folder_is_searched(tmp_path):
    flat = tmp_path / "archived_sessions"
    flat.mkdir()
    for path in (FIXTURES / "codex" / "sessions").rglob(f"*{CX_EXEC}.jsonl"):
        shutil.copy(path, flat / path.name)
    match = match_attempt({"session": CX_EXEC, "started_at": "2026-09-20T18:00:00Z"}, roots=LogRoots(codex=[flat]))
    assert match is not None and match.tokens.as_dict() == EXPECTED[CX_EXEC]["tokens"]


# --- self and roots -------------------------------------------------------------


def test_self_is_exact_under_claude_code():
    env = {"CLAUDECODE": "1", "CLAUDE_CODE_SESSION_ID": CC}
    assert resolve_session("self", "claude-code", env=env) == CC
    assert resolve_session("self", None, env=env) == CC


def test_self_under_codex_is_an_error_asking_for_the_id():
    with pytest.raises(SessionError, match="thread id"):
        resolve_session("self", "codex", env={"CLAUDE_CODE_SESSION_ID": CC})


def test_self_without_the_variable_is_an_error():
    with pytest.raises(SessionError, match="CLAUDE_CODE_SESSION_ID"):
        resolve_session("self", "claude-code", env={})


def test_a_plain_id_passes_through():
    assert resolve_session(f" {CX_EXEC} ", "codex", env={}) == CX_EXEC


def test_default_roots_honor_the_config_variables(tmp_path):
    roots = default_roots({"CLAUDE_CONFIG_DIR": str(tmp_path / "c"), "CODEX_HOME": str(tmp_path / "x")})
    assert roots.claude == [tmp_path / "c" / "projects"]
    assert roots.codex == [tmp_path / "x" / "sessions", tmp_path / "x" / "archived_sessions"]


def test_a_found_session_with_no_usage_is_a_verified_zero_token_match(tmp_path):
    sid = "019f0000-0000-7000-8000-0000000000aa"
    folder = tmp_path / "2026" / "09" / "20"
    folder.mkdir(parents=True)
    meta = {"id": sid, "session_id": sid, "timestamp": "2026-09-20T18:00:00.000Z", "cwd": "/work/repo"}
    (folder / f"rollout-2026-09-20T18-00-00-{sid}.jsonl").write_text(
        json.dumps({"timestamp": "2026-09-20T18:00:00.000Z", "type": "session_meta", "payload": meta}) + "\n"
        + json.dumps({"timestamp": "2026-09-20T18:00:05.000Z", "type": "event_msg", "payload": {"type": "task_started"}}) + "\n",
        encoding="utf-8",
    )
    match = match_attempt({"session": sid, "harness": "codex"}, roots=LogRoots(codex=[tmp_path]))
    assert match is not None and match.tier == "verified"
    assert match.tokens.total == 0 and match.model is None
    assert (match.started_at, match.ended_at) == ("2026-09-20T18:00:00Z", "2026-09-20T18:00:05Z")
