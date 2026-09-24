"""A Codex thread that switched models splits its tokens by model (lane 02 review, finding 1).

The counters are cumulative per segment, so each reading's increase is given
to the model of the latest turn_context; the parts add up to the thread's
totals exactly. Every rollout is synthetic and built in `tmp_path`.
"""

import json

from loopmath.ingest import codex

SID = "019f0000-0000-7000-8000-0000000000b1"


def _rollout(path, events, sid=SID):
    """`events`: ("model", name) or ("usage", input incl. cache, cached, output), cumulative."""
    lines = [{"timestamp": "2026-09-20T18:00:00.000Z", "type": "session_meta", "payload": {"id": sid, "session_id": sid, "timestamp": "2026-09-20T18:00:00.000Z", "cwd": "/work/repo"}}]
    for n, event in enumerate(events, start=1):
        ts = f"2026-09-20T18:{n:02d}:00.000Z"
        if event[0] == "model":
            lines.append({"timestamp": ts, "type": "turn_context", "payload": {"turn_id": f"t{n}", "cwd": "/work/repo", "model": event[1], "effort": "high"}})
        else:
            _, tin, cached, tout = event
            usage = {"input_tokens": tin, "cached_input_tokens": cached, "output_tokens": tout, "total_tokens": tin + tout}
            lines.append({"timestamp": ts, "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def _split(record):
    return {m: (t.in_, t.cache_read, t.cache_write, t.out) for m, t in record.tokens_by_model.items()}


def _sums_to_total(record):
    for name in ("in_", "cache_read", "cache_write", "out"):
        assert sum(getattr(t, name) for t in record.tokens_by_model.values()) == getattr(record.tokens, name)


def test_a_mid_thread_switch_splits_by_the_active_model(tmp_path):
    path = _rollout(tmp_path / "a.jsonl", [("model", "gpt-6-sol"), ("usage", 1_000, 400, 10), ("model", "gpt-6-astra"), ("usage", 3_000, 1_000, 30)])
    record = codex.parse_session(path)
    assert _split(record) == {"gpt-6-sol": (600, 400, 0, 10), "gpt-6-astra": (1_400, 600, 0, 20)}
    _sums_to_total(record)


def test_a_resumed_thread_on_another_model_splits_across_its_files(tmp_path):
    first = _rollout(tmp_path / "1.jsonl", [("model", "gpt-6-sol"), ("usage", 1_000, 0, 10)])
    second = _rollout(tmp_path / "2.jsonl", [("model", "gpt-6-astra"), ("usage", 500, 0, 5)])  # its own counter
    record = codex.parse_session([first, second])
    assert _split(record) == {"gpt-6-sol": (1_000, 0, 0, 10), "gpt-6-astra": (500, 0, 0, 5)}
    _sums_to_total(record)


def test_a_counter_reset_after_a_switch_keeps_both_segments(tmp_path):
    events = [("model", "gpt-6-sol"), ("usage", 1_000, 0, 10), ("usage", 2_000, 0, 20), ("model", "gpt-6-astra"), ("usage", 300, 0, 3)]
    record = codex.parse_session(_rollout(tmp_path / "a.jsonl", events))
    assert _split(record) == {"gpt-6-sol": (2_000, 0, 0, 20), "gpt-6-astra": (300, 0, 0, 3)}
    _sums_to_total(record)


def test_a_reading_before_any_turn_context_goes_to_the_first_model(tmp_path):
    events = [("usage", 100, 0, 1), ("model", "gpt-6-sol"), ("usage", 1_000, 0, 10), ("model", "gpt-6-astra"), ("usage", 1_500, 0, 15)]
    record = codex.parse_session(_rollout(tmp_path / "a.jsonl", events))
    assert _split(record) == {"gpt-6-sol": (1_000, 0, 0, 10), "gpt-6-astra": (500, 0, 0, 5)}


def test_a_single_model_thread_is_unchanged(tmp_path):
    record = codex.parse_session(_rollout(tmp_path / "a.jsonl", [("model", "gpt-6-sol"), ("usage", 1_000, 400, 10), ("model", "gpt-6-sol"), ("usage", 2_000, 400, 20)]))
    assert _split(record) == {"gpt-6-sol": (1_600, 400, 0, 20)}
