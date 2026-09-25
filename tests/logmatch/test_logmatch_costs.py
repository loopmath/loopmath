"""Tariffs and four-stream costs (spec 01 section 2.5, 03 section 6)."""

import hashlib
from pathlib import Path

import pytest

from loopmath.ingest.base import Tokens
from loopmath.logmatch.costs import EXT_KEY, cost_record, price_attempt
from loopmath.logmatch.match import LogRoots, Match, match_attempt
from loopmath.logmatch.tariff import UnpricedModel, table_id, tariff_for
from loopmath.price import DEFAULT_PRICES

FIXTURES = Path(__file__).parent / "fixtures"
AT = "2026-09-23T12:00:00Z"

TABLE = """as_of = "2026-08-31"

["checked-model"]
input = 1.00
cache_read = 0.10
cache_write = 2.00
output = 5.00
source = "official list, https://example.invalid/pricing, checked 2026-09-23"

["dated-by-table"]
input = 3.00
cache_read = 0.30
cache_write = 6.00
output = 15.00
source = "derived from billing"

["guessed-model"]
input = 2.00
cache_read = 0.20
cache_write = 0.00
zero_ok = true
output = 8.00
todo = true
source = "guess: like dated-by-table"
"""


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "prices.toml"
    path.write_text(TABLE, encoding="utf-8")
    return path


def _match(parts, tier="verified", harness="claude-code"):
    total = Tokens()
    for part in parts:
        t = part["tokens"]
        total.in_ += t.in_
        total.cache_read += t.cache_read
        total.cache_write += t.cache_write
        total.out += t.out
    return Match(
        harness=harness,
        session_path=Path("/nowhere"),
        session_id="s1",
        tier=tier,
        tokens=total,
        model=parts[0]["model"],
        effort=None,
        started_at=AT,
        ended_at=AT,
        children=[p["session"] for p in parts if p["role"] == "child"],
        parts=parts,
        reason="test",
    )


def _part(model, tokens, role="session", session="s1", harness="claude-code"):
    return {"session": session, "role": role, "harness": harness, "model": model, "tokens": tokens}


def test_tariff_id_is_the_table_sha256_prefix_and_follows_its_bytes(table):
    assert table_id(table) == hashlib.sha256(table.read_bytes()).hexdigest()[:12]
    before = table_id(table)
    table.write_text(TABLE.replace("output = 5.00", "output = 5.50"), encoding="utf-8")
    assert table_id(table) != before


def test_tariff_date_is_the_rows_checked_date_else_the_tables_as_of(table):
    assert tariff_for("checked-model", AT, path=table)["date"] == "2026-09-23"
    tariff = tariff_for("dated-by-table", AT, path=table)
    assert tariff == {"id": table_id(table), "date": "2026-08-31", "source": "derived from billing"}


def test_an_unknown_model_raises_and_is_never_priced_at_zero(table):
    with pytest.raises(UnpricedModel):
        tariff_for("no-such-model", AT, path=table)
    with pytest.raises(UnpricedModel):
        price_attempt(Tokens(in_=1), "no-such-model", AT, path=table)


def test_a_bad_timestamp_is_refused(table):
    with pytest.raises(ValueError):
        tariff_for("checked-model", "yesterday", path=table)


def test_four_streams_are_priced_separately(table):
    tokens = Tokens(in_=1_000_000, cache_read=2_000_000, cache_write=3_000_000, out=4_000_000)
    usd, tariff = price_attempt(tokens, "checked-model", AT, path=table)
    assert usd == pytest.approx(1.00 + 0.20 + 6.00 + 20.00)
    assert tariff["date"] == "2026-09-23"


def test_the_packaged_rows_price_the_new_models():
    one = Tokens(in_=1_000_000, cache_read=1_000_000, cache_write=1_000_000, out=1_000_000)
    for model, want in (
        ("claude-opus-5-5", 4.00 + 0.20 + 8.00 + 20.00),
        ("opus-5.5", 4.00 + 0.20 + 8.00 + 20.00),
        ("gpt-6-astra", 10.00 + 1.00 + 12.50 + 50.00),
        ("gpt-6-sol", 2.00 + 0.20 + 2.50 + 10.00),
        ("gpt-6-luna", 0.10 + 0.01 + 0.125 + 0.50),
        ("claude-haiku-4-5-20251001", 1.00 + 0.10 + 2.00 + 5.00),
        ("claude-sonnet-5", 2.00 + 0.20 + 4.00 + 10.00),
    ):
        usd, tariff = price_attempt(one, model, AT)
        assert usd == pytest.approx(want), model
        assert tariff["date"] == "2026-09-23", model
        assert tariff["id"] == table_id(DEFAULT_PRICES)
        assert "https://" in tariff["source"], model


def test_a_mixed_model_match_is_priced_per_part(table):
    match = _match(
        [
            _part("checked-model", Tokens(in_=1_000_000)),
            _part("dated-by-table", Tokens(out=1_000_000), role="child", session="c1"),
        ]
    )
    cost = cost_record(match, path=table)
    assert cost["usd"] == pytest.approx(1.00 + 15.00)
    assert cost["input_tokens"] == 1_000_000 and cost["output_tokens"] == 1_000_000
    assert cost["basis"] == "measured"
    assert cost["tariff"]["date"] == "2026-09-23"  # the session's row
    ext = cost["ext"][EXT_KEY]
    assert ext["tier"] == "verified" and ext["children"] == ["c1"]
    assert [p["usd"] for p in ext["parts"]] == [pytest.approx(1.0), pytest.approx(15.0)]


def test_a_heuristic_match_is_allocated(table):
    cost = cost_record(_match([_part("checked-model", Tokens(in_=10))], tier="heuristic"), path=table)
    assert cost["basis"] == "allocated"
    assert cost["ext"][EXT_KEY]["tier"] == "heuristic"


def test_an_unpriced_or_guessed_part_leaves_usd_out_and_says_why(table):
    for model, why in (("no-such-model", "no price row"), ("guessed-model", "is a guess"), (None, "no model")):
        match = _match([_part("checked-model", Tokens(in_=10)), _part(model, Tokens(in_=10), role="child", session="c1")])
        cost = cost_record(match, path=table)
        assert "usd" not in cost and "tariff" not in cost, model
        assert cost["input_tokens"] == 20
        child = cost["ext"][EXT_KEY]["parts"][1]
        assert why in child["unpriced"], model
    guessed = cost_record(_match([_part("guessed-model", Tokens(in_=1_000_000))]), path=table)
    assert guessed["ext"][EXT_KEY]["parts"][0]["estimate_usd"] == pytest.approx(2.0)


def test_codex_on_a_model_that_bills_cache_writes_is_marked_a_floor():
    roots = LogRoots(codex=[FIXTURES / "codex" / "sessions"])
    astra = cost_record(match_attempt({"session": "019f0000-0000-7000-8000-000000000003"}, roots=roots))
    assert "usd_is_floor" in astra["ext"][EXT_KEY]
    sol56 = cost_record(match_attempt({"session": "019f0000-0000-7000-8000-000000000002"}, roots=roots))
    assert "usd_is_floor" not in sol56["ext"][EXT_KEY]  # gpt-5.6 bills no cache writes


def test_fixture_claude_session_prices_each_model_at_its_own_row():
    roots = LogRoots(claude=[FIXTURES / "claude" / "projects"])
    match = match_attempt({"session": "00000000-0000-4000-8000-000000000001"}, roots=roots)
    cost = cost_record(match)
    parts = cost["ext"][EXT_KEY]["parts"]
    by_hand = 0.0
    rates = {"fable-5": (10, 1, 20, 50), "opus-5": (5, 0.5, 10, 25), "sonnet-5": (2, 0.2, 4, 10)}
    for part in parts:
        r = rates[part["model"]]
        t = part["tokens"]
        by_hand += (t["in"] * r[0] + t["cache_read"] * r[1] + t["cache_write"] * r[2] + t["out"] * r[3]) / 1e6
    assert cost["usd"] == pytest.approx(by_hand, abs=1e-5)
    assert cost["input_tokens"] + cost["cached_input_tokens"] + cost["cache_creation_tokens"] + cost["output_tokens"] == match.tokens.total


def test_a_zero_token_part_costs_nothing_and_withholds_nothing(table):
    match = _match([_part("checked-model", Tokens(in_=1_000_000)), _part(None, Tokens(), role="child", session="c1")])
    cost = cost_record(match, path=table)
    assert cost["usd"] == pytest.approx(1.0)
    assert cost["ext"][EXT_KEY]["parts"][1]["usd"] == 0.0


def test_a_session_with_no_requests_has_no_dollars(table):
    cost = cost_record(_match([_part(None, Tokens())]), path=table)
    assert "usd" not in cost and "tariff" not in cost
    assert cost["input_tokens"] == 0 and cost["basis"] == "measured"
    assert "no model requests" in cost["ext"][EXT_KEY]["no_requests"]
