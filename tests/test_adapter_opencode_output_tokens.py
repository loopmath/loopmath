"""OpenCode's output count leaves reasoning out; OCP's output_tokens includes it."""

from __future__ import annotations

import json
from pathlib import Path

from loopmath.adapters import Selection
from loopmath.adapters.opencode import OpenCodeAdapter
from loopmath.adapters.opencode_store import _message_tokens

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "adapters" / "opencode"
NS = "dev.dagr.adapter.opencode"


def _message(n: int, model: str, tokens: dict) -> dict:
    return {"info": {"id": f"msg_synthetic_{n}", "sessionID": "ses_synthetic", "role": "assistant",
                     "providerID": "openai", "modelID": model, "cost": 0.001, "tokens": tokens,
                     "time": {"created": 1704326460000 + n, "completed": 1704326520000 + n},
                     "finish": "stop"},
            "parts": []}


def _session(tmp_path: Path, messages: list[dict]) -> dict:
    """A synthetic OpenCode export session, emitted through the adapter."""
    export = {"info": {"id": "ses_synthetic", "directory": "/synthetic/workspaces/gamma", "version": "1.17.11",
                       "cost": 0.001 * len(messages), "time": {"created": 1704326400000, "updated": 1704328200000}},
              "messages": messages}
    (tmp_path / "session.json").write_text(json.dumps(export))
    doc = OpenCodeAdapter().emit(Selection(stores=(tmp_path,)))
    (attempt,) = doc["attempts"]
    return attempt


def test_output_tokens_is_output_plus_reasoning_on_the_fixture():
    doc = OpenCodeAdapter().emit(Selection(stores=(FIXTURE,), session_ids=("ses_fixture_root",)))
    (root,) = doc["attempts"]
    # Two messages: output 20 and 10, reasoning 5 and 2.
    assert root["cost"]["output_tokens"] == 37 and root["cost"]["reasoning_tokens"] == 7
    usage = {item["model"]: item for item in root["ext"][NS]["model_usage"]}
    assert usage["gpt-5.3-codex-spark"]["output_tokens"] == 25
    assert usage["gpt-5.4-mini"]["output_tokens"] == 12
    assert usage["gpt-5.4-mini"]["reasoning_tokens"] == 2


def test_a_message_without_reasoning_adds_zero(tmp_path):
    attempt = _session(tmp_path, [
        _message(1, "gpt-5.4-mini", {"input": 10, "output": 4, "reasoning": 3, "cache": {"read": 0, "write": 0}}),
        _message(2, "gpt-5.4-mini", {"input": 10, "output": 6, "cache": {"read": 0, "write": 0}}),
    ])
    assert attempt["cost"]["output_tokens"] == 13
    assert "reasoning_tokens" not in attempt["cost"]  # one message has no count, so no total
    incomplete = attempt["ext"][NS]["usage_incomplete_messages"]
    assert incomplete == {"reasoning_tokens": 1}


def test_message_tokens_rules():
    tokens = {"input": 10, "output": 4, "reasoning": 3, "cache": {"read": 1, "write": 2}}
    assert _message_tokens(tokens, "output_tokens") == 7
    assert _message_tokens(tokens, "reasoning_tokens") == 3
    assert _message_tokens(tokens, "input_tokens") == 10
    assert _message_tokens({"output": 4}, "output_tokens") == 4
    assert _message_tokens({"output": 4, "reasoning": 0}, "output_tokens") == 4
    # A reasoning count that is there but unreadable leaves the total unknown, not short.
    assert _message_tokens({"output": 4, "reasoning": -1}, "output_tokens") is None
    assert _message_tokens({"output": 4, "reasoning": "3"}, "output_tokens") is None
    assert _message_tokens({"reasoning": 3}, "output_tokens") is None
    assert _message_tokens(None, "output_tokens") is None
