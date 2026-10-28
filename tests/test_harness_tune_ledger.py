"""The cost ledger: a session's price per model against the price table (within 1e-9 of a hand sum), unknown
models priced at None and never 0, one calls.jsonl row per session per model, and the summary's gap check."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from loopmath.harness.tune.contracts import Ledger, Limits, SessionEnd, SessionRequest, TuneDir, Usage, Workspace
from loopmath.harness.tune.ledger import FIELDS, PriceLedger, row_cost, summary
from loopmath.price import load_prices

TUNE_ID = "tune-20261024-101233-7f3a"
# $/Mtok: input, cache_read, cache_write, output
HAIKU55 = (0.10, 0.01, 0.20, 0.50)
SONNET55 = (2.00, 0.10, 4.00, 10.00)
OPUS55 = (4.00, 0.20, 8.00, 20.00)


def hand(u: Usage, r: tuple[float, float, float, float]) -> float:
    return (u.input * r[0] + u.cache_read * r[1] + u.cache_write * r[2] + u.output * r[3]) / 1_000_000


@pytest.fixture
def tune(tmp_path):
    t = TuneDir(tmp_path / TUNE_ID)
    t.root.mkdir()
    return t


def req_of(tune, lab="base-1", role="executor", model="claude-haiku-5-5", **meta) -> SessionRequest:
    ws = Workspace(label=lab, role=role, folder=tune.worktree(lab), cfg_dir=tune.cfg(lab), run_dir=tune.run(lab))
    return SessionRequest(label=lab, role=role, stage="base", prompt="p", workspace=ws,
                          limits=Limits(model=model, effort="high"), meta=dict(meta))


A = Usage(model="claude-haiku-5-5", input=12_345, cache_read=987_654, cache_write=45_678, output=6_789,
          cli_cost_usd=0.0205)
B = Usage(model="claude-sonnet-5-5", input=1_001, cache_read=20_002, cache_write=3_003, output=4_004,
          cli_cost_usd=0.06)
C = Usage(model="claude-opus-5-5", input=7, cache_read=3_000_000, cache_write=111_111, output=99_999,
          cli_cost_usd=3.5)


def test_the_price_table_has_the_new_rows_marked_unconfirmed():
    t = load_prices()
    for model, r in (("claude-haiku-5-5", HAIKU55), ("claude-sonnet-5-5", SONNET55)):
        rate = t.rate(model)
        assert (rate["input"], rate["cache_read"], rate["cache_write"], rate["output"]) == r
        assert t.is_todo(model) and t.source_for(model).startswith("unconfirmed:")
    assert not t.is_todo("claude-opus-5-5")


def test_price_is_within_1e9_of_a_hand_sum(tune):
    led = PriceLedger(tune)
    assert isinstance(led, Ledger)
    got = led.price([A, B, C])
    want = hand(A, HAIKU55) + hand(B, SONNET55) + hand(C, OPUS55)
    assert abs(got - want) < 1e-9
    assert abs(led.price([A]) - hand(A, HAIKU55)) < 1e-9


def test_an_unknown_model_prices_at_none_never_zero(tune):
    led = PriceLedger(tune)
    odd = Usage(model="claude-unreleased-9", input=10, output=10)
    assert led.price([A, odd]) is None
    usd, note = row_cost(odd, led.table)
    assert usd is None and "no price entry" in note
    assert led.price([]) == 0.0  # no tokens at all is a real zero


def test_a_zero_token_api_error_row_is_a_call_priced_zero(tune):
    led = PriceLedger(tune)
    for model in ("claude-haiku-5-5", "<synthetic>", "claude-unreleased-9"):
        assert row_cost(Usage(model=model), led.table) == (0.0, "")  # no tokens: 0 for any model, never None
        assert row_cost(Usage(model=model), None) == (0.0, "")
    assert abs(led.price([A, Usage(model="<synthetic>")]) - hand(A, HAIKU55)) < 1e-9
    end = SessionEnd(label="base-1", role="executor", exit=1, is_error=True, error_kind="transient", wall_s=40.0,
                     error="API error (status 529, overloaded_error): API Error: 529 Overloaded",
                     usage=[Usage(model="claude-haiku-5-5", cli_cost_usd=0.0)], cli_cost_usd=0.0)
    assert led.price(end.usage) == 0.0
    rows = led.record(req_of(tune), end, "t1") + led.record(req_of(tune, retry=1), end, "t1")
    assert [(r["cost_usd"], r["error_kind"], r.get("retry")) for r in rows] == [(0.0, "transient", None),
                                                                                (0.0, "transient", 1)]
    assert all("price_note" not in r and r["input_tokens"] == r["output_tokens"] == 0 for r in rows)
    s = summary(tune.calls)
    assert s["rows"] == 2 and s["unpriced"] == 0 and s["cost_usd"] == 0.0 and s["by_role"]["executor"]["rows"] == 2


def test_record_writes_one_row_per_model_with_the_named_fields(tune):
    led = PriceLedger(tune)
    end = SessionEnd(label="base-1", role="executor", exit=0, wall_s=12.5, usage=[A, B], cli_cost_usd=0.0805)
    rows = led.record(req_of(tune), end, "t0123456789a")
    assert len(rows) == 2
    on_disk = [json.loads(x) for x in tune.calls.read_text().splitlines()]
    assert on_disk == rows
    for r in rows:
        assert set(FIELDS) <= set(r)
        assert r["tune"] == TUNE_ID and r["task_id"] == "t0123456789a" and r["label"] == "base-1"
        assert r["role"] == "executor" and r["effort"] == "high" and r["exit"] == 0 and r["is_error"] is False
    a, b = rows
    assert (a["model"], a["input_tokens"], a["cache_read_tokens"], a["cache_write_tokens"], a["output_tokens"]) == \
        ("claude-haiku-5-5", 12_345, 987_654, 45_678, 6_789)
    assert abs(a["cost_usd"] - hand(A, HAIKU55)) < 1e-9 and abs(b["cost_usd"] - hand(B, SONNET55)) < 1e-9
    assert a["cli_cost_usd"] == 0.0205 and b["cli_cost_usd"] == 0.06
    assert a["wall_s"] == 12.5 and b["wall_s"] == 0.0  # the session's wall time once
    assert a["price_note"] == "placeholder price row"
    assert abs(sum(r["cost_usd"] for r in rows) - led.price([A, B])) < 1e-9


def test_a_session_with_no_tokens_still_gets_a_row(tune):
    led = PriceLedger(tune)
    end = SessionEnd(label="base-1", role="executor", exit=None, is_error=True, error_kind="other",
                     error="the session ended on a signal without a result line")
    rows = led.record(req_of(tune, model="claude-opus-5-5"), end, "t1")
    assert len(rows) == 1 and rows[0]["model"] == "claude-opus-5-5" and rows[0]["cost_usd"] == 0.0
    assert rows[0]["error_kind"] == "other" and rows[0]["is_error"] is True and rows[0]["exit"] is None
    assert "price_note" not in rows[0]


def test_an_unpriced_row_says_why_and_a_retry_is_marked(tune):
    led = PriceLedger(tune)
    end = SessionEnd(label="base-1", role="executor", exit=1, is_error=True, error_kind="transient",
                     usage=[Usage(model="claude-unreleased-9", input=5)])
    row = led.record(req_of(tune, retry=1), end, "t1")[0]
    assert row["cost_usd"] is None and "no price entry" in row["price_note"] and row["retry"] == 1


def test_rows_from_parallel_sessions_stay_whole(tune):
    led = PriceLedger(tune)

    def one(i):
        end = SessionEnd(label=f"probe-{i:02d}", role="executor", exit=0, wall_s=1.0, usage=[A, B, C])
        led.record(req_of(tune, f"probe-{i:02d}"), end, "t1")

    threads = [threading.Thread(target=one, args=(i,)) for i in range(1, 13)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [json.loads(x) for x in tune.calls.read_text().splitlines()]
    assert len(rows) == 36 and len({r["label"] for r in rows}) == 12


def test_a_price_table_by_path(tune, tmp_path):
    from loopmath import price

    led = PriceLedger(tune, prices=Path(price.__file__).with_name("prices.toml"))
    assert abs(led.price([C]) - hand(C, OPUS55)) < 1e-9


def test_summary_totals_and_flags_gaps_over_five_percent(tune):
    led = PriceLedger(tune)
    exact = Usage(model="claude-opus-5-5", input=1_000_000, cli_cost_usd=4.0)
    off = Usage(model="claude-opus-5-5", input=1_000_000, cli_cost_usd=4.5)
    led.record(req_of(tune, "base-1"), SessionEnd(label="base-1", role="executor", exit=0, usage=[exact]), "t")
    led.record(req_of(tune, "judge-a1", role="judge"), SessionEnd(label="judge-a1", role="judge", exit=0,
                                                                   usage=[off]), "t")
    led.record(req_of(tune, "base-2"), SessionEnd(label="base-2", role="executor", exit=0,
                                                  usage=[Usage(model="nope-1", input=3)]), "t")
    with open(tune.calls, "a") as fh:
        fh.write("{torn\n")
    s = summary(tune.calls)
    assert s["rows"] == 3 and s["bad_lines"] == 1 and s["unpriced"] == 1
    assert abs(s["cost_usd"] - 8.0) < 1e-9 and abs(s["cli_cost_usd"] - 8.5) < 1e-9
    assert s["by_role"]["executor"]["rows"] == 2 and s["by_role"]["judge"]["cost_usd"] == pytest.approx(4.0)
    assert s["by_model"]["claude-opus-5-5"]["input_tokens"] == 2_000_000
    assert [g["label"] for g in s["gaps"]] == ["judge-a1"] and s["gaps"][0]["gap"] == pytest.approx(0.1111, abs=1e-4)
    assert summary(tune.root / "absent.jsonl")["rows"] == 0
