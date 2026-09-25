"""Shared pricing prices each model of a session at its own rate.

`RunRecord.to_dict` keeps the per-model split, the ingest cache keeps it, and
`price.price_run` sums the parts or withholds dollars when a part has no
price. Every session is synthetic and built in `tmp_path`.
"""

import json

import pytest

from loopmath import ingest
from loopmath.graph.extract import node_from_record
from loopmath.ingest import codex
from loopmath.ingest.base import RunRecord, Tokens
from loopmath.price import load_prices, price_all, price_run

SID = "019f0000-0000-7000-8000-0000000000d1"
TOKENS = {"in": 1000, "cache_read": 0, "cache_write": 0, "out": 0}


def _mixed_rollout(path):
    """1,000 input tokens on gpt-6-sol, then 1,000 on gpt-6-astra (the reviewer's case)."""
    lines = [{"timestamp": "2026-09-20T18:00:00.000Z", "type": "session_meta", "payload": {"id": SID, "timestamp": "2026-09-20T18:00:00.000Z", "cwd": "/work/repo"}}]
    for n, model in enumerate(["gpt-6-sol", "gpt-6-astra"], start=1):
        lines.append({"timestamp": f"2026-09-20T18:0{n}:00.000Z", "type": "turn_context", "payload": {"turn_id": f"t{n}", "cwd": "/work/repo", "model": model, "effort": "low"}})
        usage = {"input_tokens": 1000 * n, "cached_input_tokens": 0, "output_tokens": 0, "total_tokens": 1000 * n}
        lines.append({"timestamp": f"2026-09-20T18:0{n}:30.000Z", "type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": usage}}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path


def test_a_mixed_session_is_priced_per_model_through_to_the_graph_node(tmp_path):
    record = codex.parse_session(_mixed_rollout(tmp_path / "rollout-2026-09-20T18-00-00-mixed.jsonl")).to_dict()
    assert record["tokens_by_model"] == [{"model": "gpt-6-sol", "tokens": TOKENS}, {"model": "gpt-6-astra", "tokens": TOKENS}]
    result = price_run(record, load_prices())
    assert result["priced"] and result["usd"] == pytest.approx(0.012)
    # The onboarding path: price_all, then the graph node the emitter reads.
    priced, warnings = price_all([record], load_prices())
    assert priced[0]["usd"] == pytest.approx(0.012) and warnings["n_unpriced_runs"] == 0
    assert node_from_record(priced[0]).usd == pytest.approx(0.012)


def test_one_unpriced_model_withholds_the_dollars_and_is_named(tmp_path):
    record = {
        "run_id": "r",
        "model": "gpt-6-sol",
        "tokens": {"in": 2000, "cache_read": 0, "cache_write": 0, "out": 0},
        "tokens_by_model": [{"model": "gpt-6-sol", "tokens": TOKENS}, {"model": "gpt-9-unreleased", "tokens": TOKENS}],
    }
    result = price_run(record, load_prices())
    assert (result["usd"], result["usd_breakdown"], result["priced"]) == (None, None, False)
    assert result["unpriced_models"] == ["gpt-9-unreleased"]
    assert "'gpt-9-unreleased'" in result["reason"] and result["model"] == "gpt-6-sol"
    _, warnings = price_all([record], load_prices())
    # Tallied under the model with no price, not the session's main model.
    assert warnings["unpriced_models"] == {"gpt-9-unreleased": 1}


def test_tokens_under_no_model_are_never_priced_as_another_model():
    record = {"model": "gpt-6-sol", "tokens": TOKENS, "tokens_by_model": [{"model": "gpt-6-sol", "tokens": TOKENS}, {"model": None, "tokens": TOKENS}]}
    result = price_run(record, load_prices())
    assert result["usd"] is None and result["unpriced_models"] == [None]
    assert "no model label" in result["reason"]


def test_a_session_that_could_not_be_split_has_no_dollars():
    record = RunRecord(run_id="r", harness="codex", model="gpt-6-sol", effort=None, tokens=Tokens(in_=2000), wall_s=1.0, ts=None, workspace=None)
    body = record.to_dict()
    assert body["tokens_by_model"] == []
    result = price_run(body, load_prices())
    assert result["usd"] is None and "could not be split by model" in result["reason"]


def test_a_single_model_record_keeps_its_shape_and_its_price(tmp_path):
    tokens = Tokens(in_=1000)
    record = RunRecord(run_id="r", harness="codex", model="gpt-6-sol", effort=None, tokens=tokens, wall_s=1.0, ts=None, workspace=None, tokens_by_model={"gpt-6-sol": tokens})
    body = record.to_dict()
    assert "tokens_by_model" not in body
    result = price_run(body, load_prices())
    assert result["usd"] == pytest.approx(0.002)  # gpt-6-sol alone, $2 per million input tokens


def test_the_split_survives_the_ingest_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    _mixed_rollout(tmp_path / "logs" / "rollout-2026-09-20T18-00-00-mixed.jsonl")
    fresh, diag = ingest.parse_all(logs=tmp_path / "logs", use_cache=True)
    assert diag["cache_hits"] == 0
    cached, diag = ingest.parse_all(logs=tmp_path / "logs", use_cache=True)
    assert diag["cache_hits"] == 1
    assert cached == fresh and cached[0]["tokens_by_model"] == fresh[0]["tokens_by_model"] != []
    assert price_run(cached[0], load_prices())["usd"] == pytest.approx(0.012)
