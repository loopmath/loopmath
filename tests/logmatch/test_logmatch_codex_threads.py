"""Codex threads that switch models or resume in new files (lane 02 review, findings 1 and 2)."""

import json

import pytest

from loopmath.ingest import codex
from loopmath.logmatch.costs import EXT_KEY, cost_record
from loopmath.logmatch.match import LogRoots, match_attempt
from loopmath.logmatch.settle import settle_attempt

SID = "019f0000-0000-7000-8000-0000000000b2"


def _rollout(root, day, sid, events, cwd="/work/repo"):
    """A rollout in `root/2026/09/<day>`; `events`: ("model", name) or ("usage", in, out), cumulative."""
    stamp = f"2026-09-{day:02d}T18:00:00.000Z"
    lines = [{"timestamp": stamp, "type": "session_meta", "payload": {"id": sid, "session_id": sid, "timestamp": stamp, "cwd": cwd}}]
    for n, event in enumerate(events, start=1):
        ts = f"2026-09-{day:02d}T18:{n:02d}:00.000Z"
        if event[0] == "model":
            lines.append({"timestamp": ts, "type": "turn_context", "payload": {"turn_id": f"t{n}", "cwd": cwd, "model": event[1], "effort": "high"}})
        else:
            usage = {"input_tokens": event[1], "cached_input_tokens": 0, "output_tokens": event[2], "total_tokens": event[1] + event[2]}
            lines.append({"timestamp": ts, "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}})
    path = root / "2026" / "09" / f"{day:02d}" / f"rollout-2026-09-{day:02d}T11-00-00-{sid}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


@pytest.fixture
def switched(tmp_path):
    root = tmp_path / "sessions"
    _rollout(root, 20, SID, [("model", "gpt-6-sol"), ("usage", 1_000, 0), ("model", "gpt-6-astra"), ("usage", 2_000, 0)])
    return LogRoots(codex=[root])


def test_a_model_switch_prices_each_model_and_hides_neither(switched):
    cost = cost_record(match_attempt({"session": SID}, roots=switched))
    # 1000 input on sol at $2/Mtok plus 1000 on astra at $10/Mtok.
    assert cost["usd"] == pytest.approx(0.012)
    parts = {p["model"]: p["tokens"]["in"] for p in cost["ext"][EXT_KEY]["parts"]}
    assert parts == {"gpt-6-sol": 1_000, "gpt-6-astra": 1_000}


def test_a_thread_that_cannot_be_split_withholds_dollars_and_says_why(switched, monkeypatch):
    monkeypatch.setattr(codex, "_split_by_model", lambda *args: {})
    cost = cost_record(match_attempt({"session": SID}, roots=switched))
    assert "usd" not in cost and cost["input_tokens"] == 2_000
    (part,) = cost["ext"][EXT_KEY]["parts"]
    assert part["model"] is None and "switched models" in part["unpriced"]


def test_a_named_session_counts_a_continuation_days_later_whatever_the_start(tmp_path):
    root = tmp_path / "sessions"
    _rollout(root, 20, SID, [("model", "gpt-6-sol"), ("usage", 1_000, 10)])
    _rollout(root, 23, SID, [("model", "gpt-6-sol"), ("usage", 1_000, 10)])  # resumed, own counter
    roots = LogRoots(codex=[root])
    without = match_attempt({"session": SID}, roots=roots)
    with_start = match_attempt({"session": SID, "started_at": "2026-09-20T18:00:00Z"}, roots=roots)
    assert without.tokens.in_ == with_start.tokens.in_ == 2_000
    assert without.tokens.out == with_start.tokens.out == 20


def test_a_heuristic_match_is_one_candidate_per_thread_and_takes_every_file(tmp_path):
    root = tmp_path / "sessions"
    _rollout(root, 20, SID, [("model", "gpt-6-sol"), ("usage", 1_000, 10)])
    _rollout(root, 21, SID, [("model", "gpt-6-sol"), ("usage", 500, 5)])
    attempt = {"id": "a", "harness": "codex", "cwd": "/work/repo", "started_at": "2026-09-20T18:00:00Z", "ended_at": "2026-09-21T19:00:00Z"}
    out, report = settle_attempt(attempt, roots=LogRoots(codex=[root]))
    assert report["tier"] == "heuristic", report["reason"]
    assert out["cost"]["input_tokens"] == 1_500


def test_artifacts_come_from_every_file_of_a_resumed_thread(tmp_path):
    root = tmp_path / "sessions"
    _rollout(root, 20, SID, [("model", "gpt-6-sol"), ("usage", 1_000, 10)])
    second = _rollout(root, 23, SID, [("model", "gpt-6-sol"), ("usage", 1_000, 10)])
    patch = [
        {"timestamp": "2026-09-23T18:30:00.000Z", "type": "response_item", "payload": {"type": "custom_tool_call", "name": "apply_patch", "call_id": "c", "input": "*** Begin Patch\n*** Add File: later.md\n+hi\n*** End Patch"}},
        {"timestamp": "2026-09-23T18:30:01.000Z", "type": "response_item", "payload": {"type": "custom_tool_call_output", "call_id": "c", "output": "Success. Updated the following files:\nA later.md\n"}},
    ]
    with second.open("a", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(line) + "\n" for line in patch))
    match = match_attempt({"session": SID, "started_at": "2026-09-20T18:00:00Z"}, roots=LogRoots(codex=[root]))
    assert [a["path"] for a in match.artifacts] == ["later.md"]
