"""Tests for loopmath.price (SPEC section 5, amended 08-31): tokens to dollars, four
streams per model (`input`, `cache_read`, `cache_write`, `output`), with loud
warnings.
"""

import builtins
import datetime
import dis
from types import CodeType

import pytest

from loopmath import price as price_module
from loopmath import price_validation
from loopmath.price import (
    STALE_AFTER_DAYS,
    load_prices,
    price_all,
    price_run,
    ratio_pair,
    validate_prices,
    validation_lines,
    warning_lines,
)

# The packaged table's as-of date, used as "now" wherever a test needs a fixed
# today so a staleness check does not drift with the wall clock.
TODAY = datetime.date(2026, 8, 31)


def test_validator_templates_declare_and_share_every_global_dependency():
    def loaded_globals(code):
        names = {
            instruction.argval
            for instruction in dis.get_instructions(code)
            if instruction.opname == "LOAD_GLOBAL"
        }
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                names.update(loaded_globals(constant))
        return names

    templates = price_validation._VALIDATOR_TEMPLATES
    export_names = {template.__name__ for template in templates}
    required_template_names = {"_as_of_date", "validate_prices", "validation_lines"}
    assert export_names == required_template_names, (
        "_VALIDATOR_TEMPLATES must contain exactly the validator facade exports; "
        f"missing={sorted(required_template_names - export_names)}, "
        f"unexpected={sorted(export_names - required_template_names)}"
    )

    missing = object()

    def resolves_to_builtin(name, namespace):
        builtin_object = vars(builtins).get(name, missing)
        if builtin_object is missing:
            return False
        return namespace.get(name, builtin_object) is builtin_object

    dependencies = set()
    for template in templates:
        dependencies.update(
            name
            for name in loaded_globals(template.__code__)
            if name not in export_names
            and not resolves_to_builtin(name, template.__globals__)
        )

    declared = set(price_validation._VALIDATOR_GLOBAL_NAMES)
    assert dependencies <= declared, (
        "validator LOAD_GLOBAL dependencies missing from _VALIDATOR_GLOBAL_NAMES: "
        f"{sorted(dependencies - declared)}"
    )
    for name in dependencies:
        assert name in vars(price_module), f"validator global {name!r} missing from loopmath.price"
        assert name in vars(price_validation), (
            f"validator global {name!r} missing from loopmath.price_validation"
        )
        assert vars(price_module)[name] is vars(price_validation)[name], (
            f"validator global {name!r} is not synchronized between modules"
        )

    for template in templates:
        rebuilt = vars(price_module)[template.__name__]
        assert callable(rebuilt)
        assert rebuilt.__globals__ is vars(price_module)
        assert vars(price_validation)[template.__name__] is rebuilt


def test_load_prices_packaged_table():
    table = load_prices()
    assert table.as_of == "2026-08-31"
    assert table.rates["claude-opus-5"] == {
        "input": 5.00,
        "cache_read": 0.50,
        "cache_write": 10.00,
        "output": 25.00,
    }
    assert table.rates["gpt-5.6-sol"] == {
        "input": 4.00,
        "cache_read": 0.40,
        "cache_write": 0.00,
        "output": 20.00,
    }
    assert "claude-opus-5" not in table.todo
    assert "gpt-5.6-sol" not in table.todo


def test_load_prices_packaged_table_shape_and_todo_count():
    # The Analyst's authoritative table, five OpenCode-observed entries and
    # the rows checked on the provider pages on 2026-09-23: 27 entries, 5
    # still flagged todo (the placeholder section), and the canonical short
    # name resolves to the full-name table entry.
    table = load_prices()
    assert len(table.rates) == 27
    assert len(table.todo) == 5

    rates = table.rate("opus-5")
    assert rates is not None
    assert rates["input"] == pytest.approx(5.00)
    assert rates["cache_read"] == pytest.approx(0.50)
    assert rates["cache_write"] == pytest.approx(10.00)
    assert rates["output"] == pytest.approx(25.00)


def test_todo_set_holds_only_unchecked_rows():
    table = load_prices()
    # Rows checked on a provider page are confirmed; the todo set covers the
    # models no page or billing fit has confirmed.
    assert "gpt-5.5" in table.todo
    assert table.is_todo("gpt-5.5") is True
    assert table.is_todo("opus-4.8") is False
    assert table.is_todo("claude-sonnet-5") is False
    assert table.is_todo("opus-5") is False
    assert table.is_todo("claude-opus-5") is False


def test_model_name_resolution_both_spellings_match():
    # C: the table keys Anthropic models by full name (claude-opus-5); a run
    # record carries the short canonical name (opus-5). Both must resolve to
    # the identical rates, and an exact match must not require canonicalizing.
    table = load_prices()
    full = table.rate("claude-opus-5")
    short = table.rate("opus-5")
    assert full is not None
    assert full == short


def test_price_run_arithmetic():
    table = load_prices()
    record = {
        "model": "opus-5",
        "tokens": {
            "in": 2_000_000,
            "cache_read": 1_000_000,
            "cache_write": 500_000,
            "out": 250_000,
        },
    }
    result = price_run(record, table)
    # Hand-computed against claude-opus-5 rates (5.00 / 0.50 / 10.00 / 25.00):
    # 2*5.00 + 1*0.50 + 0.5*10.00 + 0.25*25.00 = 10 + 0.5 + 5 + 6.25 = 21.75
    assert result["usd"] == pytest.approx(21.75, abs=1e-9)
    assert result["usd_breakdown"]["in"] == pytest.approx(10.0, abs=1e-9)
    assert result["usd_breakdown"]["cache_read"] == pytest.approx(0.5, abs=1e-9)
    assert result["usd_breakdown"]["cache_write"] == pytest.approx(5.0, abs=1e-9)
    assert result["usd_breakdown"]["out"] == pytest.approx(6.25, abs=1e-9)
    assert result["priced"] is True
    assert result["todo"] is False
    assert result["model"] == "opus-5"
    assert result["reason"] is None


def test_price_run_stream_transposition_guard(tmp_path):
    # A: four distinct rates and four distinct token counts, chosen so that
    # transposing any two streams (e.g. pricing cache_write tokens at the
    # cache_read rate) changes the total. This is the canary for the
    # _TOKEN_TO_RATE mapping.
    prices = tmp_path / "prices.toml"
    prices.write_text(
        """
        as_of = "2026-08-31"

        ["test-model"]
        input = 2.0
        cache_read = 3.0
        cache_write = 5.0
        output = 7.0
        source = "test fixture"
        """,
        encoding="utf-8",
    )
    table = load_prices(prices)
    record = {
        "model": "test-model",
        "tokens": {"in": 11, "cache_read": 13, "cache_write": 17, "out": 19},
    }
    result = price_run(record, table)
    expected = (
        (11 / 1_000_000.0) * 2.0
        + (13 / 1_000_000.0) * 3.0
        + (17 / 1_000_000.0) * 5.0
        + (19 / 1_000_000.0) * 7.0
    )
    assert result["usd"] == pytest.approx(expected, rel=1e-12)
    assert result["usd_breakdown"]["in"] == pytest.approx((11 / 1_000_000.0) * 2.0)
    assert result["usd_breakdown"]["cache_read"] == pytest.approx((13 / 1_000_000.0) * 3.0)
    assert result["usd_breakdown"]["cache_write"] == pytest.approx((17 / 1_000_000.0) * 5.0)
    assert result["usd_breakdown"]["out"] == pytest.approx((19 / 1_000_000.0) * 7.0)

    # A transposed mapping (e.g. cache_read priced at the cache_write rate)
    # would produce a different total; make sure our expected total actually
    # depends on getting the mapping right.
    wrong = (
        (11 / 1_000_000.0) * 2.0
        + (13 / 1_000_000.0) * 5.0  # cache_read tokens at cache_write rate
        + (17 / 1_000_000.0) * 3.0  # cache_write tokens at cache_read rate
        + (19 / 1_000_000.0) * 7.0
    )
    assert wrong != pytest.approx(expected, rel=1e-9)


def test_price_run_todo_model_still_prices_but_flags():
    table = load_prices()
    record = {
        "model": "gpt-5.5",
        "tokens": {"in": 1_000_000, "cache_read": 0, "cache_write": 0, "out": 0},
    }
    result = price_run(record, table)
    assert result["usd"] == pytest.approx(2.50, abs=1e-9)
    assert result["priced"] is True
    assert result["todo"] is True


def test_price_run_unknown_model_never_zero():
    table = load_prices()
    record = {
        "model": "unknown-model",
        "tokens": {"in": 1000, "cache_read": 1000, "cache_write": 1000, "out": 1000},
    }
    result = price_run(record, table)
    assert result["usd"] is None
    assert result["usd_breakdown"] is None
    assert result["priced"] is False
    assert result["reason"]
    assert "unknown-model" in result["reason"]


def test_price_run_no_model_label():
    table = load_prices()
    record = {
        "model": None,
        "tokens": {"in": 1000, "cache_read": 1000, "cache_write": 1000, "out": 1000},
    }
    result = price_run(record, table)
    assert result["usd"] is None
    assert result["priced"] is False
    assert result["reason"] == "run has no model label"


def test_price_run_missing_tokens_block_is_unpriced_not_zero():
    table = load_prices()
    record = {"model": "opus-5"}
    result = price_run(record, table)
    assert result["usd"] is None
    assert result["usd_breakdown"] is None
    assert result["priced"] is False
    assert result["reason"]
    assert "token count" in result["reason"]


def test_price_run_missing_stream_names_it_and_is_unpriced():
    table = load_prices()
    record = {"model": "opus-5", "tokens": {"in": 1000, "cache_write": 1000, "out": 1000}}
    result = price_run(record, table)
    assert result["usd"] is None
    assert result["usd_breakdown"] is None
    assert result["priced"] is False
    assert "cache_read" in result["reason"]


def test_price_run_missing_cache_write_stream_names_it_and_is_unpriced():
    # The amendment's new fourth stream: dropping it must be caught exactly
    # like dropping any other stream, naming it by name.
    table = load_prices()
    record = {"model": "opus-5", "tokens": {"in": 1000, "cache_read": 1000, "out": 1000}}
    result = price_run(record, table)
    assert result["usd"] is None
    assert result["usd_breakdown"] is None
    assert result["priced"] is False
    assert "cache_write" in result["reason"]


def test_price_run_explicit_zero_stream_prices_normally():
    table = load_prices()
    record = {
        "model": "opus-5",
        "tokens": {"in": 1, "cache_read": 0, "cache_write": 0, "out": 2},
    }
    result = price_run(record, table)
    # Hand-computed: 1e-6*5.0 + 0 + 0 + 2e-6*25.0, tiny but real and non-None.
    assert result["priced"] is True
    assert result["usd"] is not None
    assert result["usd_breakdown"]["cache_read"] == pytest.approx(0.0, abs=1e-12)
    assert result["usd_breakdown"]["cache_write"] == pytest.approx(0.0, abs=1e-12)
    assert result["reason"] is None


def test_price_run_non_numeric_stream_is_unpriced_and_does_not_raise():
    table = load_prices()
    record = {
        "model": "opus-5",
        "tokens": {"in": 1000, "cache_read": "lots", "cache_write": 0, "out": 1000},
    }
    result = price_run(record, table)
    assert result["usd"] is None
    assert result["usd_breakdown"] is None
    assert result["priced"] is False
    assert "cache_read" in result["reason"]


def test_load_prices_missing_as_of_raises(tmp_path):
    bad = tmp_path / "no-as-of.toml"
    bad.write_text(
        """
        ["opus-5"]
        input = 5.0
        cache_read = 0.50
        cache_write = 10.0
        output = 25.0
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="as_of"):
        load_prices(bad)


def test_load_prices_missing_rate_raises(tmp_path):
    bad = tmp_path / "missing-rate.toml"
    bad.write_text(
        """
        as_of = "2026-08-31"

        ["broken-model"]
        input = 5.0
        cache_read = 0.50
        cache_write = 10.0
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="broken-model") as exc_info:
        load_prices(bad)
    assert "output" in str(exc_info.value)


def test_load_prices_unknown_model_table_keys_raise(tmp_path):
    bad = tmp_path / "unknown-keys.toml"
    bad.write_text(
        """
        as_of = "2026-08-31"

        ["broken-model"]
        input = 5.0
        cache_read = 0.50
        cache_write = 10.0
        output = 25.0
        cache_raed = 0.25
        comment = "not supported metadata"
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="broken-model") as exc_info:
        load_prices(bad)

    message = str(exc_info.value)
    assert "unknown key" in message
    assert "cache_raed" in message
    assert "comment" in message


def test_price_all_warnings_split_todo_and_unpriced():
    table = load_prices()
    records = [
        {
            "model": "opus-5",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 100, "out": 100},
        },
        {
            "model": "gpt-5.5",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 100, "out": 100},
        },
        {
            "model": "gpt-5.5",
            "tokens": {"in": 200, "cache_read": 200, "cache_write": 200, "out": 200},
        },
        {
            "model": "gpt-5.2",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 100, "out": 100},
        },
        {
            "model": "nope-9",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 100, "out": 100},
        },
    ]
    priced, warnings = price_all(records, table)

    # Non-mutating: input records are untouched, output carries the fields.
    assert "usd" not in records[0]
    assert priced[0]["usd"] is not None
    assert priced[4]["usd"] is None

    assert warnings["n_todo_runs"] == 3  # two gpt-5.5 + one gpt-5.2
    assert warnings["todo_models"] == {"gpt-5.5": 2, "gpt-5.2": 1}
    assert warnings["n_unpriced_runs"] == 1
    assert warnings["unpriced_models"] == {"nope-9": 1}
    assert warnings["as_of"] == table.as_of

    lines = warning_lines(warnings, table)
    joined = " ".join(lines)
    assert "3 runs priced from placeholder rates" in joined
    assert "gpt-5.5: 2" in joined
    assert "gpt-5.2: 1" in joined
    assert "1 run could not be priced at all" in joined
    assert "nope-9: 1" in joined
    # No em-dashes and none of the forbidden jargon terms leak into output.
    forbidden = [
        "knowledge gradient",
        "posterior",
        "prior",
        "experimental design",
        "value of information",
        "bandit",
        "—",
    ]
    for term in forbidden:
        assert term not in joined


def test_price_all_warnings_include_unpriced_reasons():
    table = load_prices()
    records = [
        {
            "model": "opus-5",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 100, "out": 100},
        },
        {"model": "opus-5"},  # no tokens block at all
        {
            "model": "opus-5",
            "tokens": {"in": 100, "cache_write": 100, "out": 100},
        },  # missing cache_read
        {
            "model": "opus-5",
            "tokens": {"in": 100, "cache_read": "lots", "cache_write": 100, "out": 100},
        },
    ]
    priced, warnings = price_all(records, table)

    assert priced[0]["usd"] is not None
    assert priced[1]["usd"] is None
    assert priced[2]["usd"] is None
    assert priced[3]["usd"] is None

    assert warnings["n_unpriced_runs"] == 3
    assert warnings["unpriced_reasons"] == {
        "run has no token counts": 1,
        "run is missing the 'cache_read' token count": 1,
        "run has a non-numeric 'cache_read' token count": 1,
    }

    lines = warning_lines(warnings, table)
    joined = " ".join(lines)
    assert "run has no token counts: 1" in joined
    assert "run is missing the 'cache_read' token count: 1" in joined
    assert "run has a non-numeric 'cache_read' token count: 1" in joined
    assert "excluded from every dollar figure, never counted as zero" in joined


def test_usd_breakdown_has_four_token_keys():
    table = load_prices()
    record = {
        "model": "opus-5",
        "tokens": {"in": 1000, "cache_read": 1000, "cache_write": 1000, "out": 1000},
    }
    result = price_run(record, table)
    assert set(result["usd_breakdown"].keys()) == {"in", "cache_read", "cache_write", "out"}


def test_warning_lines_gpt56_caveat_present_when_priced():
    table = load_prices()
    records = [
        {
            "model": "gpt-5.6-sol",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 0, "out": 100},
        },
        {
            "model": "gpt-5.6-terra",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 0, "out": 100},
        },
    ]
    priced, warnings = price_all(records, table)
    assert warnings["n_gpt56_priced_runs"] == 2

    lines = warning_lines(warnings, table)
    joined = " ".join(lines)
    assert "GPT-5.6" in joined
    assert "272k" in joined
    assert "floor" in joined
    # No forbidden jargon, no em-dashes.
    for term in ["knowledge gradient", "posterior", "prior", "bandit", "—"]:
        assert term not in joined


def test_warning_lines_gpt56_caveat_absent_when_not_priced():
    table = load_prices()
    records = [
        {
            "model": "opus-5",
            "tokens": {"in": 100, "cache_read": 100, "cache_write": 100, "out": 100},
        },
    ]
    priced, warnings = price_all(records, table)
    assert warnings["n_gpt56_priced_runs"] == 0

    lines = warning_lines(warnings, table)
    joined = " ".join(lines)
    assert "GPT-5.6" not in joined
    assert "272k" not in joined
def test_warning_lines_omits_age_clause_when_fresh_or_unknown():
    table = load_prices()
    fresh = dict(
        todo_models={},
        unpriced_models={},
        n_todo_runs=0,
        n_unpriced_runs=0,
        n_gpt56_priced_runs=0,
        as_of=table.as_of,
        staleness_days=0,
    )
    assert warning_lines(fresh, table) == ["Price table as of 2026-08-31."]

    unknown_age = dict(fresh, staleness_days=None)
    assert warning_lines(unknown_age, table) == ["Price table as of 2026-08-31."]
def test_ratio_pair_amplifier_example():
    # SPEC amplifier example: 3x tokens on a cheaper model can be ~10x dollars.
    a_tokens = {"in": 3_000_000, "cache_read": 0, "cache_write": 0, "out": 0}
    b_tokens = {"in": 1_000_000, "cache_read": 0, "cache_write": 0, "out": 0}
    result = ratio_pair(a_tokens, b_tokens, a_usd=10.0, b_usd=1.0)
    assert result["token_ratio"] == pytest.approx(3.0, abs=1e-9)
    assert result["dollar_ratio"] == pytest.approx(10.0, abs=1e-9)
    assert result["amplifier"] == pytest.approx(3.333, abs=1e-3)


def test_ratio_pair_sums_all_four_streams():
    # E: the total must include cache_read and cache_write, not just in/out.
    a_tokens = {"in": 1_000_000, "cache_read": 500_000, "cache_write": 250_000, "out": 100_000}
    b_tokens = {"in": 1_000_000, "cache_read": 0, "cache_write": 0, "out": 0}
    result = ratio_pair(a_tokens, b_tokens, a_usd=1.0, b_usd=1.0)
    expected_total = 1_000_000 + 500_000 + 250_000 + 100_000
    assert result["token_ratio"] == pytest.approx(expected_total / 1_000_000.0, abs=1e-9)


def test_ratio_pair_guards_division_by_zero():
    zero_tokens = {"in": 0, "cache_read": 0, "cache_write": 0, "out": 0}
    some_tokens = {"in": 100, "cache_read": 0, "cache_write": 0, "out": 0}
    result = ratio_pair(some_tokens, zero_tokens, a_usd=5.0, b_usd=0.0)
    assert result["token_ratio"] is None
    assert result["dollar_ratio"] is None
    assert result["amplifier"] is None


def test_warning_lines_leave_staleness_to_the_validator():
    """The run-pricing report states the as-of date; it never judges its age.

    Staleness now exists (`validate_prices`, SPEC section 8 item 2) but it
    lives there, not here: `warning_lines` reports on the runs that were
    priced, so it prints the as-of date as a fact and computes no age from it.
    """
    table = load_prices()
    assert not hasattr(table, "staleness_days")
    lines = warning_lines({"as_of": table.as_of}, table)
    assert lines == [f"Price table as of {table.as_of}."]
    joined = " ".join(lines).lower()
    for word in ("days old", "stale", "out of date"):
        assert word not in joined


# ---------------------------------------------------------------------------
# The price table validator (SPEC section 8, item 2): schema, staleness, the
# zero/negative sign rules, and the todo audit.
# ---------------------------------------------------------------------------

VALID_TABLE = """
as_of = "2026-08-31"

["claude-opus-5"]
input = 5.00
cache_read = 0.50
cache_write = 10.00
output = 25.00
source = "derived from billing (exact)"
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_validate_prices_valid_file_passes_silently(tmp_path):
    result = validate_prices(_write(tmp_path, "ok.toml", VALID_TABLE), today=TODAY)
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["warnings"] == []


def test_validate_prices_marked_zero_passes_silently(tmp_path):
    # A real free stream: OpenAI charges nothing for a cache write. Marked
    # zero_ok, it is not a warning and not an error.
    table = VALID_TABLE + """
["gpt-5.6-sol"]
input = 4.00
cache_read = 0.40
cache_write = 0.00
zero_ok = true
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "marked-zero.toml", table), today=TODAY)
    assert result["ok"] is True
    assert result["errors"] == []
    assert result["warnings"] == []


def test_validate_prices_zero_ok_only_suppresses_cache_write_zeros(tmp_path):
    # Keep the established boolean marker compatible, but do not let a cache
    # write exemption make an accidental free input or output rate silent.
    table = VALID_TABLE + """
["broken-free-rates"]
input = 0.00
cache_read = 0.00
cache_write = 0.00
zero_ok = true
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "stream-specific-zero.toml", table), today=TODAY)
    assert result["ok"] is True
    assert result["errors"] == []
    assert len(result["warnings"]) == 1
    warning = result["warnings"][0]
    assert "broken-free-rates.input" in warning
    assert "broken-free-rates.cache_read" in warning
    assert "broken-free-rates.output" not in warning
    assert "broken-free-rates.cache_write" not in warning


def test_validate_prices_a_marked_free_model_passes_silently(tmp_path):
    # Dogfood: the packaged free rows always warned, and their
    # zero_ok could not clear it. Every rate zero and marked is a free model;
    # one dropped digit cannot zero all four rates.
    free = """
["free-model"]
input = 0.00
cache_read = 0.00
cache_write = 0.00
zero_ok = true
output = 0.00
"""
    result = validate_prices(_write(tmp_path, "free.toml", VALID_TABLE + free), today=TODAY)
    assert (result["ok"], result["errors"], result["warnings"]) == (True, [], [])
    unmarked = validate_prices(_write(tmp_path, "unmarked.toml", VALID_TABLE + free.replace("zero_ok = true\n", "")), today=TODAY)
    assert len(unmarked["warnings"]) == 1 and "free-model.output" in unmarked["warnings"][0]


def test_validate_prices_unmarked_zero_warns_naming_model_and_stream(tmp_path):
    table = VALID_TABLE + """
["gpt-5.6-sol"]
input = 4.00
cache_read = 0.40
cache_write = 0.00
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "bare-zero.toml", table), today=TODAY)
    # A zero is legal, so this is a warning and the table stays valid.
    assert result["ok"] is True
    assert result["errors"] == []
    assert len(result["warnings"]) == 1
    warning = result["warnings"][0]
    assert "gpt-5.6-sol.cache_write" in warning
    assert "zero_ok" in warning


def test_validate_prices_negative_rate_is_a_hard_error(tmp_path):
    table = VALID_TABLE + """
["broken-model"]
input = -4.00
cache_read = 0.40
cache_write = 0.00
zero_ok = true
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "negative.toml", table), today=TODAY)
    assert result["ok"] is False
    assert len(result["errors"]) == 1
    assert "broken-model" in result["errors"][0]
    assert "input" in result["errors"][0]
    assert "negative" in result["errors"][0]


def test_validate_prices_zero_ok_does_not_excuse_a_negative_rate(tmp_path):
    # zero_ok says "my zeros are meant", never "skip my sign check".
    table = VALID_TABLE + """
["broken-model"]
input = 4.00
cache_read = 0.40
cache_write = -1.00
zero_ok = true
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "negative-marked.toml", table), today=TODAY)
    assert result["ok"] is False
    assert "cache_write" in result["errors"][0]


def test_validate_prices_missing_stream_is_a_hard_error(tmp_path):
    table = VALID_TABLE + """
["broken-model"]
input = 4.00
cache_read = 0.40
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "missing.toml", table), today=TODAY)
    assert result["ok"] is False
    assert len(result["errors"]) == 1
    assert "broken-model" in result["errors"][0]
    assert "cache_write" in result["errors"][0]


def test_validate_prices_rejects_unexpected_top_level_scalar(tmp_path):
    table = 'currency = "USD"\n' + VALID_TABLE
    result = validate_prices(_write(tmp_path, "top-level-scalar.toml", table), today=TODAY)
    assert result["ok"] is False
    assert len(result["errors"]) == 1
    assert "unexpected top-level field 'currency'" in result["errors"][0]


def test_validate_prices_rejects_every_unknown_model_table_field(tmp_path):
    table = VALID_TABLE + """
["unknown-fields"]
input = 4.00
cache_read = 0.40
cache_write = 0.00
zero_ok = true
output = 20.00
currency = "USD"
cache_wirte = 0.00
"""
    result = validate_prices(_write(tmp_path, "unknown-fields.toml", table), today=TODAY)
    assert result["ok"] is False
    assert len(result["errors"]) == 2
    joined = " ".join(result["errors"])
    assert "unknown-fields" in joined
    assert "unknown field 'currency'" in joined
    assert "unknown field 'cache_wirte'" in joined
    assert "source, todo, zero_ok" in joined


def test_validate_prices_rejects_string_and_boolean_rates(tmp_path):
    table = VALID_TABLE + """
["string-rate"]
input = "4.00"
cache_read = 0.40
cache_write = 0.00
zero_ok = true
output = 20.00

["boolean-rate"]
input = true
cache_read = 0.40
cache_write = 0.00
zero_ok = true
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "typed-rates.toml", table), today=TODAY)
    assert result["ok"] is False
    assert len(result["errors"]) == 2
    joined = " ".join(result["errors"])
    assert "string-rate" in joined
    assert "boolean-rate" in joined
    assert "non-numeric 'input'" in joined


def test_validate_prices_rejects_non_boolean_markers_without_suppressing_zero(tmp_path):
    table = VALID_TABLE + """
["broken-markers"]
input = 4.00
cache_read = 0.40
cache_write = 0.00
zero_ok = "false"
output = 20.00
todo = "true"
"""
    result = validate_prices(_write(tmp_path, "typed-markers.toml", table), today=TODAY)
    assert result["ok"] is False
    assert len(result["errors"]) == 2
    joined = " ".join(result["errors"])
    assert "non-boolean 'zero_ok'" in joined
    assert "non-boolean 'todo'" in joined
    assert result["todo_models"] == []
    assert len(result["warnings"]) == 1
    assert "broken-markers.cache_write" in result["warnings"][0]


def test_validate_prices_reports_every_problem_in_one_pass(tmp_path):
    # Unlike load_prices, which stops at the first thing it will not invent,
    # the validator must not let one broken entry hide the next.
    table = VALID_TABLE + """
["broken-a"]
input = 4.00
cache_read = 0.40
output = 20.00

["broken-b"]
input = -1.00
cache_read = 0.40
cache_write = 0.00
zero_ok = true
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "two-bad.toml", table), today=TODAY)
    assert result["ok"] is False
    assert len(result["errors"]) == 2
    joined = " ".join(result["errors"])
    assert "broken-a" in joined
    assert "broken-b" in joined


def test_validate_prices_missing_as_of_is_a_hard_error(tmp_path):
    table = """
["claude-opus-5"]
input = 5.00
cache_read = 0.50
cache_write = 10.00
output = 25.00
"""
    result = validate_prices(_write(tmp_path, "no-as-of.toml", table), today=TODAY)
    assert result["ok"] is False
    assert "as_of" in result["errors"][0]


def test_validate_prices_unparseable_as_of_is_a_hard_error(tmp_path):
    table = VALID_TABLE.replace('as_of = "2026-08-31"', 'as_of = "last tuesday"')
    result = validate_prices(_write(tmp_path, "bad-date.toml", table), today=TODAY)
    assert result["ok"] is False
    assert "as_of" in result["errors"][0]
    assert "last tuesday" in result["errors"][0]


def test_validate_prices_stale_as_of_warns_with_the_age(tmp_path):
    p = _write(tmp_path, "stale.toml", VALID_TABLE)
    # 2026-08-31 plus 31 days: one day past the threshold.
    result = validate_prices(p, today=datetime.date(2026, 10, 1))
    assert result["ok"] is True
    assert len(result["warnings"]) == 1
    assert "31 days old" in result["warnings"][0]
    assert str(STALE_AFTER_DAYS) in result["warnings"][0]


def test_validate_prices_boundary_age_is_not_yet_stale(tmp_path):
    p = _write(tmp_path, "fresh.toml", VALID_TABLE)
    exactly = datetime.date(2026, 8, 31) + datetime.timedelta(days=STALE_AFTER_DAYS)
    assert validate_prices(p, today=exactly)["warnings"] == []
    assert validate_prices(p, today=exactly + datetime.timedelta(days=1))["warnings"]


def test_validate_prices_lists_todo_entries_loudly(tmp_path):
    table = VALID_TABLE + """
["opus-4.8"]
input = 5.00
cache_read = 0.50
cache_write = 10.00
output = 25.00
todo = true
source = "guess"

["haiku-4.5"]
input = 1.00
cache_read = 0.10
cache_write = 2.00
output = 5.00
todo = true
source = "guess"
"""
    result = validate_prices(_write(tmp_path, "todo.toml", table), today=TODAY)
    assert result["ok"] is True
    assert result["todo_models"] == ["opus-4.8", "haiku-4.5"]
    warning = result["warnings"][0]
    assert "2 of 3" in warning
    assert "opus-4.8" in warning
    assert "haiku-4.5" in warning


def test_validate_prices_unreadable_and_malformed_files(tmp_path):
    missing = validate_prices(tmp_path / "nope.toml", today=TODAY)
    assert missing["ok"] is False
    assert "could not be read" in missing["errors"][0]

    junk = _write(tmp_path, "junk.toml", "as_of = \n")
    bad = validate_prices(junk, today=TODAY)
    assert bad["ok"] is False
    assert "not valid TOML" in bad["errors"][0]


def test_validate_prices_empty_table_is_a_hard_error(tmp_path):
    result = validate_prices(_write(tmp_path, "empty.toml", 'as_of = "2026-08-31"\n'), today=TODAY)
    assert result["ok"] is False
    assert "no model entries" in result["errors"][0]


def test_validate_prices_packaged_table_has_no_hard_error():
    # The shipped table must validate: every GPT cache_write is a real 0.00
    # (OpenAI charges nothing for a cache write) and is marked zero_ok. The
    # three OpenCode-hosted free routes are all zero and marked, so they are
    # free models and no longer warn (from lane 02's dogfood).
    # Staleness is deliberately not asserted here because the packaged as_of
    # ages on its own.
    result = validate_prices()
    assert result["errors"] == []
    assert result["ok"] is True
    assert len(result["todo_models"]) == 5
    assert [warning for warning in result["warnings"] if "rate(s) are zero" in warning] == []


def test_validation_lines_render_errors_first_then_verdict(tmp_path):
    table = VALID_TABLE + """
["broken-model"]
input = -4.00
cache_read = 0.40
cache_write = 0.00
output = 20.00
"""
    result = validate_prices(_write(tmp_path, "render.toml", table), today=TODAY)
    lines = validation_lines(result)
    assert lines[0].startswith("checking ")
    assert lines[1].startswith("ERROR: ")
    assert lines[2].startswith("WARNING: ")
    assert lines[-1] == "price table is INVALID: 1 error."

    clean = validate_prices(_write(tmp_path, "clean.toml", VALID_TABLE), today=TODAY)
    assert validation_lines(clean)[-1] == "price table is valid (no warnings)."


def test_cli_validate_prices_exit_status(tmp_path, capsys):
    from loopmath.cli import main

    assert main(["validate-prices"]) == 0
    assert "price table is valid" in capsys.readouterr().out

    bad = _write(tmp_path, "bad.toml", VALID_TABLE + """
["broken-model"]
input = -4.00
cache_read = 0.40
cache_write = 1.00
output = 20.00
""")
    assert main(["validate-prices", "--prices", str(bad)]) == 1
    # A failed check reports on stderr, so a pipeline reading stdout is not
    # handed a verdict that never came.
    assert "INVALID" in capsys.readouterr().err


def test_warning_lines_pluralize_run_counts():
    table = load_prices()
    base = {
        "todo_models": {"demo": 1},
        "unpriced_models": {},
        "unpriced_reasons": {},
        "n_unpriced_runs": 0,
        "n_gpt56_priced_runs": 0,
    }

    one = warning_lines(dict(base, n_todo_runs=1), table)[0]
    two = warning_lines(dict(base, n_todo_runs=2), table)[0]
    assert "1 run priced from placeholder rates" in one
    assert "2 runs priced from placeholder rates" in two

    unpriced = dict(
        base,
        todo_models={},
        n_todo_runs=0,
        unpriced_models={"demo": 1},
        unpriced_reasons={"run has no token counts": 1},
        n_unpriced_runs=1,
    )
    singular_lines = warning_lines(unpriced, table)
    singular_expected = (
        "1 run could not be priced at all",
        "It is excluded from every dollar figure, never counted as zero",
        "That run failed for a reason other than a missing price entry",
    )
    singular_text = " ".join(singular_lines)
    assert [text for text in singular_expected if text not in singular_text] == []

    plural_lines = warning_lines(
        dict(
            unpriced,
            unpriced_models={"demo": 2},
            unpriced_reasons={"run has no token counts": 2},
            n_unpriced_runs=2,
        ),
        table,
    )
    assert "2 runs could not be priced at all" in plural_lines[0]
    assert "They are excluded from every dollar figure, never counted as zero" in plural_lines[0]
    assert "2 of those runs failed for a reason other than a missing price entry" in plural_lines[1]
