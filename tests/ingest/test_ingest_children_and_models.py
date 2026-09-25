"""Child sessions and per-model tokens in ingest (lane 02).

A Codex sub-agent thread is its own record, priced at its own model, with its
parent in `parent_session`; a Claude Code sub-agent transcript names its
parent the same way; a Claude Code session that ran two models splits its
tokens by model. Every fixture is synthetic and built in `tmp_path`.
"""

import json

from loopmath import ingest
from loopmath.ingest import claude_code, codex
from loopmath.ingest.base import Tokens


ROOT = "019fc9b3-0000-7000-8000-000000000001"
CHILD = "019fc9b3-0000-7000-8000-000000000002"
GRANDCHILD = "019fc9b3-0000-7000-8000-000000000003"


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def _codex_rollout(tmp_path, stamp, own_id, *, model, tokens, root=None, parent=None):
    source = "exec"
    if parent is not None:
        source = {"subagent": {"thread_spawn": {"parent_thread_id": parent, "depth": 1}}}
    meta = {
        "id": own_id,
        "session_id": root or own_id,
        "timestamp": f"2026-09-23T{stamp}.000Z",
        "cwd": "/repo",
        "originator": "codex_exec",
        "source": source,
    }
    if parent is not None:
        meta["parent_thread_id"] = parent
    inp, cached, out = tokens
    usage = {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "output_tokens": out,
        "total_tokens": inp + out,
    }
    lines = [
        {"timestamp": f"2026-09-23T{stamp}.000Z", "type": "session_meta", "payload": meta},
        {
            "timestamp": f"2026-09-23T{stamp}.100Z",
            "type": "turn_context",
            "payload": {"turn_id": "t1", "cwd": "/repo", "model": model, "effort": "high"},
        },
        {
            "timestamp": f"2026-09-23T{stamp}.200Z",
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"total_token_usage": usage, "last_token_usage": usage},
            },
        },
    ]
    name = f"rollout-2026-09-23T{stamp.replace(':', '-')}-{own_id}.jsonl"
    return _write(tmp_path / "2026" / "09" / "23" / name, lines)


def test_codex_child_threads_are_their_own_records_at_their_own_model(tmp_path):
    _codex_rollout(tmp_path, "10:00:00", ROOT, model="gpt-6-astra", tokens=(1000, 400, 100))
    _codex_rollout(
        tmp_path, "10:01:00", CHILD, model="gpt-6-luna", tokens=(500, 0, 50), root=ROOT, parent=ROOT
    )
    _codex_rollout(
        tmp_path,
        "10:02:00",
        GRANDCHILD,
        model="gpt-6-sol",
        tokens=(300, 100, 30),
        root=ROOT,
        parent=CHILD,
    )

    records, diag = ingest.parse_all(logs=tmp_path, harnesses=("codex",), use_cache=False)

    assert diag["records"] == {"codex": 3, "total": 3}
    by_id = {r["run_id"]: r for r in records}
    root = by_id[f"cx_{ROOT}"]
    child = by_id[f"cx_{CHILD}"]
    grandchild = by_id[f"cx_{GRANDCHILD}"]
    assert root["model"] == "gpt-6-astra"
    assert root["tokens"] == {"in": 600, "cache_read": 400, "cache_write": 0, "out": 100}
    assert "parent_session" not in root
    assert child["model"] == "gpt-6-luna"
    assert child["tokens"] == {"in": 500, "cache_read": 0, "cache_write": 0, "out": 50}
    assert child["parent_session"] == ROOT
    # A grandchild names its direct parent, not the root in `session_id`.
    assert grandchild["model"] == "gpt-6-sol"
    assert grandchild["parent_session"] == CHILD


def test_codex_child_without_parent_thread_id_still_names_its_root(tmp_path):
    path = _codex_rollout(tmp_path, "10:01:00", CHILD, model="gpt-6-luna", tokens=(10, 0, 1), root=ROOT)
    record = codex.parse_session(path)
    assert record is not None
    assert record.run_id == f"cx_{CHILD}"
    assert record.parent_session == ROOT
    assert record.tokens_by_model == {"gpt-6-luna": record.tokens}


def test_codex_files_sharing_an_id_still_group_as_one_resumed_thread(tmp_path):
    first = _codex_rollout(tmp_path, "10:00:00", ROOT, model="gpt-6-sol", tokens=(100, 0, 10))
    second = _codex_rollout(tmp_path, "10:05:00", ROOT, model="gpt-6-sol", tokens=(200, 0, 20))
    groups = codex.group_session_paths([first, second])
    assert groups == [[first, second]]


def _cc_line(session_id, request_id, model, usage, *, sidechain=False):
    return {
        "type": "assistant",
        "sessionId": session_id,
        "requestId": request_id,
        "timestamp": "2026-09-23T10:00:00.000Z",
        "cwd": "/repo",
        "isSidechain": sidechain,
        "message": {
            "model": model,
            "usage": {
                "input_tokens": usage[0],
                "cache_read_input_tokens": usage[1],
                "cache_creation_input_tokens": usage[2],
                "output_tokens": usage[3],
            },
            "content": [],
            "stop_reason": "end_turn",
        },
    }


def test_claude_code_tokens_split_by_model_and_sum_to_the_total(tmp_path):
    sid = "5b1e0c3a-0000-4000-8000-000000000001"
    path = _write(
        tmp_path / "proj" / f"{sid}.jsonl",
        [
            _cc_line(sid, "r1", "claude-opus-5-5", (10, 100, 20, 5)),
            # A streamed request writes more than one line; the per-stream max
            # counts once, under the request's model.
            _cc_line(sid, "r1", "claude-opus-5-5", (10, 100, 20, 9)),
            _cc_line(sid, "r2", "claude-haiku-4-5-20251001", (3, 30, 6, 2)),
            _cc_line(sid, "r3", "claude-opus-5-5", (1, 10, 0, 1)),
        ],
    )
    record = claude_code.parse_session(path)
    assert record is not None
    assert record.tokens_by_model == {
        "claude-opus-5-5": Tokens(in_=11, cache_read=110, cache_write=20, out=10),
        "haiku-4.5": Tokens(in_=3, cache_read=30, cache_write=6, out=2),
    }
    total = Tokens()
    for part in record.tokens_by_model.values():
        total.in_ += part.in_
        total.cache_read += part.cache_read
        total.cache_write += part.cache_write
        total.out += part.out
    assert total == record.tokens
    # The split joins the record dict, for shared pricing.
    assert record.to_dict()["tokens_by_model"] == [
        {"model": "claude-opus-5-5", "tokens": {"in": 11, "cache_read": 110, "cache_write": 20, "out": 10}},
        {"model": "haiku-4.5", "tokens": {"in": 3, "cache_read": 30, "cache_write": 6, "out": 2}},
    ]
    assert record.parent_session is None
    assert "parent_session" not in record.to_dict()


def test_claude_code_subagent_transcript_names_its_parent(tmp_path):
    sid = "5b1e0c3a-0000-4000-8000-000000000001"
    path = _write(
        tmp_path / "proj" / sid / "subagents" / "agent-a0000000000000001.jsonl",
        [_cc_line(sid, "r1", "claude-haiku-4-5-20251001", (1, 2, 3, 4), sidechain=True)],
    )
    record = claude_code.parse_session(path)
    assert record is not None
    assert record.parent_session == sid
    assert record.to_dict()["parent_session"] == sid


def test_the_thread_id_is_read_from_the_first_session_meta_line_only(tmp_path):
    """Lane 09 perf note: grouping stops at the first session_meta line."""
    from loopmath.ingest.codex_support import _session_id

    path = tmp_path / "rollout.jsonl"
    path.write_text(
        "\n"
        + '{"type": "session_meta", "payload": \n'  # truncated: skipped, as iter_jsonl skips it
        + json.dumps({"type": "event_msg", "payload": {"type": "session_meta"}})  # not the type
        + "\n"
        + json.dumps({"type": "session_meta", "payload": {"id": CHILD, "session_id": ROOT}})
        + "\n"
        + json.dumps({"type": "session_meta", "payload": {"id": GRANDCHILD}})
        + "\n",
        encoding="utf-8",
    )
    assert _session_id(path) == CHILD
    assert _session_id(tmp_path / "missing.jsonl") is None
