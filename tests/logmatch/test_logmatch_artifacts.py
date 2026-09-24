"""Artifacts a matched session wrote and read (synthetic sessions in tmp_path)."""

import json

from loopmath.logmatch.match import LogRoots, match_attempt
from loopmath.logmatch.settle import settle_attempt

SID = "00000000-0000-4000-8000-0000000000a1"
CX = "019f0000-0000-7000-8000-0000000000a1"


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def _assistant(ts, request, blocks, sidechain=False, session=SID):
    return {
        "type": "assistant",
        "sessionId": session,
        "requestId": request,
        "timestamp": ts,
        "cwd": "/work/repo",
        "isSidechain": sidechain,
        "message": {
            "role": "assistant",
            "model": "claude-opus-5-5",
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 5},
            "content": blocks,
        },
    }


def _result(ts, tool_id):
    return {
        "type": "user",
        "sessionId": SID,
        "timestamp": ts,
        "cwd": "/work/repo",
        "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]},
    }


def _tool(tool_id, name, **inp):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": inp}


def test_claude_writes_and_reads_merge_per_path_across_children(tmp_path):
    project = tmp_path / "projects" / "-work-repo"
    _write(
        project / f"{SID}.jsonl",
        [
            _assistant("2026-09-20T17:00:00Z", "r1", [_tool("t1", "Read", file_path="/work/repo/src/app.py")]),
            _result("2026-09-20T17:00:01Z", "t1"),
            _assistant("2026-09-20T17:00:02Z", "r2", [_tool("t2", "Edit", file_path="/work/repo/src/app.py", old_string="a", new_string="b")]),
            _result("2026-09-20T17:00:03Z", "t2"),
        ],
    )
    _write(
        project / SID / "subagents" / "agent-a1.jsonl",
        [
            _assistant("2026-09-20T17:01:00Z", "r3", [_tool("t3", "Write", file_path="/work/repo/docs/notes.md", content="x")], sidechain=True),
            _result("2026-09-20T17:01:01Z", "t3"),
            _assistant("2026-09-20T17:01:02Z", "r4", [_tool("t4", "Read", file_path="/elsewhere/ref.txt")], sidechain=True),
        ],
    )
    match = match_attempt({"session": SID, "harness": "claude-code"}, roots=LogRoots(claude=[tmp_path / "projects"]))
    assert match is not None
    by_path = {a["path"]: a for a in match.artifacts}
    assert set(by_path) == {"src/app.py", "docs/notes.md", "/elsewhere/ref.txt"}
    app = by_path["src/app.py"]
    assert (app["writes"], app["reads"], app["kind"], app["tier"]) == (1, 1, "code", "verified")
    assert (app["first_at"], app["last_at"]) == ("2026-09-20T17:00:00Z", "2026-09-20T17:00:02Z")
    assert by_path["docs/notes.md"]["writes"] == 1  # written by the sub-agent
    # Written files come first; nothing but metadata is kept.
    assert [a["writes"] > 0 for a in match.artifacts] == [True, True, False]
    assert set().union(*match.artifacts) <= {"path", "writes", "reads", "tier", "first_at", "last_at", "kind"}


def test_codex_patch_writes_are_found_and_settle_puts_them_on_the_attempt(tmp_path):
    folder = tmp_path / "sessions" / "2026" / "09" / "20"
    meta = {"id": CX, "session_id": CX, "timestamp": "2026-09-20T18:00:00.000Z", "cwd": "/work/repo"}
    usage = {"input_tokens": 100, "cached_input_tokens": 0, "output_tokens": 10, "total_tokens": 110}
    _write(
        folder / f"rollout-2026-09-20T18-00-00-{CX}.jsonl",
        [
            {"timestamp": "2026-09-20T18:00:00.000Z", "type": "session_meta", "payload": meta},
            {"timestamp": "2026-09-20T18:00:01.000Z", "type": "turn_context", "payload": {"turn_id": "t", "cwd": "/work/repo", "model": "gpt-6-sol", "effort": "high"}},
            {"timestamp": "2026-09-20T18:00:02.000Z", "type": "response_item", "payload": {"type": "custom_tool_call", "name": "apply_patch", "call_id": "c", "input": "*** Begin Patch\n*** Add File: y.md\n+hi\n*** End Patch"}},
            {"timestamp": "2026-09-20T18:00:03.000Z", "type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "c", "output": "Success. Updated the following files:\nA y.md\n"}},
            {"timestamp": "2026-09-20T18:00:04.000Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage, "last_token_usage": usage}}},
        ],
    )
    roots = LogRoots(codex=[tmp_path / "sessions"])
    match = match_attempt({"session": CX, "harness": "codex"}, roots=roots)
    assert [(a["path"], a["writes"], a["kind"]) for a in match.artifacts] == [("y.md", 1, "doc")]

    settled, _ = settle_attempt({"id": "a1", "session": CX, "ext": {"x.other": 1}}, roots=roots)
    assert settled["ext"]["dev.loopmath.logmatch"]["artifacts"][0]["path"] == "y.md"
    assert settled["ext"]["x.other"] == 1
