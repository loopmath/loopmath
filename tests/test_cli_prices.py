"""Tests for `loopmath prices` (SPEC section 8, item 6): show the active price table."""

from __future__ import annotations

from loopmath.cli import main

FIXTURE_TABLE = """
as_of = "2026-01-15"

["claude-opus-5"]
input = 5.00
cache_read = 0.50
cache_write = 10.00
output = 25.00
source = "derived from billing (exact)"

["claude-sonnet-5"]
input = 3.00
cache_read = 0.30
cache_write = 6.00
output = 15.00

["gpt-5.6-luna"]
input = 0.10
cache_read = 0.01
cache_write = 0.00
output = 0.40
todo = true
source = "public pricing page, unconfirmed"
"""


def test_prices_verb_lists_every_model_and_as_of_date(tmp_path, capsys):
    prices_path = tmp_path / "prices.toml"
    prices_path.write_text(FIXTURE_TABLE)

    rc = main(["prices", "--prices", str(prices_path)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "claude-opus-5" in out
    assert "claude-sonnet-5" in out
    assert "gpt-5.6-luna" in out
    assert "2026-01-15" in out


def test_prices_verb_marks_todo_entries_and_leaves_confirmed_ones_unmarked(tmp_path, capsys):
    prices_path = tmp_path / "prices.toml"
    prices_path.write_text(FIXTURE_TABLE)

    main(["prices", "--prices", str(prices_path)])
    out = capsys.readouterr().out
    lines = out.splitlines()

    todo_line = next(line for line in lines if "gpt-5.6-luna" in line)
    confirmed_line = next(line for line in lines if "claude-opus-5" in line)
    assert "TODO" in todo_line
    assert "TODO" not in confirmed_line


def test_prices_verb_shows_source_path(tmp_path, capsys):
    prices_path = tmp_path / "prices.toml"
    prices_path.write_text(FIXTURE_TABLE)

    main(["prices", "--prices", str(prices_path)])
    out = capsys.readouterr().out

    assert str(prices_path) in out


def test_prices_verb_missing_file_exits_1_with_clear_message(tmp_path, capsys):
    missing = tmp_path / "does-not-exist.toml"

    rc = main(["prices", "--prices", str(missing)])
    captured = capsys.readouterr()

    assert rc == 1
    assert str(missing) in captured.err
    assert "not found" in captured.err.lower()


def test_prices_verb_default_table_loads_packaged_prices(capsys):
    rc = main(["prices"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "claude-opus-5" in out
    assert "2026-08-31" in out
