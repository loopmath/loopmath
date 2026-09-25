"""The per-model token split settle writes and `price_model_tokens` reprices."""

import json
from pathlib import Path

import pytest

from loopmath.ingest import codex
from loopmath.ingest.base import Tokens
from loopmath.logmatch.costs import EXT_KEY, MODEL_TOKENS_KEY, cost_record, model_tokens_from_parts
from loopmath.logmatch.match import LogRoots, Match
from loopmath.logmatch.settle import settle_run
from loopmath.price import load_prices, price_model_tokens

FIXTURES = Path(__file__).parent / "fixtures"
CC = "00000000-0000-4000-8000-000000000001"  # fable-5, opus-5 and sonnet-5
CX_TUI = "019f0000-0000-7000-8000-000000000002"  # one model
SID = "019f0000-0000-7000-8000-0000000000c7"
FIELDS = {"input_tokens", "cached_input_tokens", "cache_creation_tokens", "output_tokens"}
STREAMS = sorted(FIELDS)


def _switched(root):
    """A Codex thread: 1,000 input tokens on gpt-6-sol, then 1,000 on gpt-6-astra."""
    lines = [{"timestamp": "2026-09-20T18:00:00.000Z", "type": "session_meta", "payload": {"id": SID, "timestamp": "2026-09-20T18:00:00.000Z", "cwd": "/work/repo"}}]
    for n, model in enumerate(["gpt-6-sol", "gpt-6-astra"], start=1):
        lines.append({"timestamp": f"2026-09-20T18:0{n}:00.000Z", "type": "turn_context", "payload": {"turn_id": f"t{n}", "cwd": "/work/repo", "model": model, "effort": "low"}})
        usage = {"input_tokens": 1000 * n, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 1000 * n}
        lines.append({"timestamp": f"2026-09-20T18:0{n}:30.000Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}})
    path = root / "2026" / "09" / "20" / f"rollout-2026-09-20T11-00-00-{SID}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return LogRoots(codex=[root])


def _settled_cost(session, roots, cost=None):
    attempt = {"id": "a1", "node": "n1", "status": "done", "session": session}
    if cost is not None:
        attempt["cost"] = cost
    doc, _ = settle_run({"ocp": "0.3", "attempts": [attempt]}, roots=roots)
    return doc["attempts"][0]["cost"]


def _sums(split):
    return {name: sum(fields[name] for fields in split.values()) for name in STREAMS}


def test_a_mixed_thread_is_split_under_the_cost_field_names_and_reprices_to_its_dollars(tmp_path):
    cost = _settled_cost(SID, _switched(tmp_path / "sessions"))
    split = cost["ext"][MODEL_TOKENS_KEY]
    one = {"input_tokens": 1000, "cached_input_tokens": 0, "cache_creation_tokens": 0, "output_tokens": 0}
    assert split == {"gpt-6-sol": one, "gpt-6-astra": one}
    assert _sums(split) == {name: cost[name] for name in STREAMS}
    priced = price_model_tokens(split, load_prices())
    # gpt-6-sol at $2 and gpt-6-astra at $10 per million input tokens.
    assert priced["priced"] and priced["usd"] == pytest.approx(0.012) == pytest.approx(cost["usd"])


def test_a_claude_session_with_three_models_is_split_and_holds_ids_and_counts_only():
    cost = _settled_cost(CC, LogRoots(claude=[FIXTURES / "claude" / "projects"]))
    split = cost["ext"][MODEL_TOKENS_KEY]
    assert set(split) == {"fable-5", "opus-5", "sonnet-5"}
    assert all(set(fields) == FIELDS and all(isinstance(n, int) for n in fields.values()) for fields in split.values())
    assert _sums(split) == {name: cost[name] for name in STREAMS}
    assert price_model_tokens(split, load_prices())["usd"] == pytest.approx(cost["usd"], abs=1e-5)


def test_a_single_model_attempt_has_no_split_and_a_stale_one_is_dropped():
    roots = LogRoots(codex=[FIXTURES / "codex" / "sessions"])
    assert MODEL_TOKENS_KEY not in _settled_cost(CX_TUI, roots)["ext"]
    stale = {"ext": {MODEL_TOKENS_KEY: {"gpt-6-sol": {name: 1 for name in FIELDS}}, "x.other": {"k": 1}}}
    ext = _settled_cost(CX_TUI, roots, cost=stale)["ext"]
    assert MODEL_TOKENS_KEY not in ext and ext["x.other"] == {"k": 1}


def test_a_thread_that_could_not_be_split_is_an_empty_split(tmp_path, monkeypatch):
    monkeypatch.setattr(codex, "_split_by_model", lambda *args: {})
    cost = _settled_cost(SID, _switched(tmp_path / "sessions"))
    assert cost["ext"][MODEL_TOKENS_KEY] == {} and "usd" not in cost
    priced = price_model_tokens({}, load_prices())
    assert priced["usd"] is None and "could not be split by model" in priced["reason"]


def _match(parts):
    total = Tokens()
    for part in parts:
        for stream in ("in_", "cache_read", "cache_write", "out"):
            setattr(total, stream, getattr(total, stream) + getattr(part["tokens"], stream))
    return Match(
        harness="claude-code",
        session_path=Path("/nowhere"),
        session_id="s1",
        tier="verified",
        tokens=total,
        model=parts[0]["model"],
        effort=None,
        started_at="2026-09-23T12:00:00Z",
        ended_at="2026-09-23T12:00:00Z",
        children=[p["session"] for p in parts if p["role"] == "child"],
        parts=parts,
        reason="test",
    )


def _part(model, tokens, role="session", session="s1"):
    return {"session": session, "role": role, "harness": "claude-code", "model": model, "tokens": tokens}


def test_unlabelled_tokens_go_under_unknown_and_are_never_priced():
    cost = cost_record(_match([_part("sonnet-5", Tokens(in_=10)), _part(None, Tokens(in_=5), role="child", session="c1")]))
    split = cost["ext"][MODEL_TOKENS_KEY]
    assert set(split) == {"sonnet-5", "unknown"} and split["unknown"]["input_tokens"] == 5
    priced = price_model_tokens(split, load_prices())
    assert priced["usd"] is None and priced["unpriced_models"] == ["unknown"]
    assert "no model label" in priced["reason"]


def test_one_model_across_session_and_child_is_summed_and_zero_parts_are_left_out():
    parts = [
        _part("sonnet-5", Tokens(in_=10, out=1)),
        _part("sonnet-5", Tokens(in_=5, cache_read=2), role="child", session="c1"),
        _part("haiku-5", Tokens(), role="child", session="c2"),
    ]
    cost = cost_record(_match(parts))
    assert MODEL_TOKENS_KEY not in cost["ext"]  # one model with tokens: no split
    split = model_tokens_from_parts(cost["ext"][EXT_KEY]["parts"])
    assert split == {"sonnet-5": {"input_tokens": 15, "cached_input_tokens": 2, "cache_creation_tokens": 0, "output_tokens": 1}}


def test_price_model_tokens_reads_the_cost_field_names():
    table = load_prices()
    base = {"input_tokens": 1_000_000, "cached_input_tokens": 0, "output_tokens": 0}
    assert price_model_tokens({"gpt-6-sol": base}, table)["usd"] == pytest.approx(2.0)
    # cache_creation_tokens is optional: absent, it is the 5m and 1h halves.
    halves = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "cache_creation_5m_tokens": 1_000_000, "cache_creation_1h_tokens": 1_000_000}
    whole = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "cache_creation_tokens": 2_000_000}
    assert price_model_tokens({"opus-5": halves}, table)["usd"] == pytest.approx(price_model_tokens({"opus-5": whole}, table)["usd"])
    assert price_model_tokens({"gpt-6-sol": {"input_tokens": 1}}, table)["usd"] is None  # a missing count
    assert price_model_tokens({"gpt-9-unreleased": base}, table)["unpriced_models"] == ["gpt-9-unreleased"]
    assert price_model_tokens(["gpt-6-sol"], table)["usd"] is None
