"""The one `--since` reader and the fits size in `status`."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from store_helpers import cli

from loopmath.store import Store
from loopmath.store.ids import AmbiguousSince, SinceError, parse_since, since_days

NOW = datetime(2026, 9, 23, 23, 0, tzinfo=timezone(timedelta(hours=-7)))


@pytest.mark.parametrize("text,back", [
    ("36h", timedelta(hours=36)), ("90d", timedelta(days=90)), ("12w", timedelta(weeks=12)),
    ("1.5d", timedelta(hours=36)), (" 90 D ", timedelta(days=90)), ("2W", timedelta(weeks=2)),
])
def test_lengths_count_back_from_now(text, back):
    assert parse_since(text, NOW) == NOW - back


def test_dates_and_times_and_absent():
    assert parse_since("2026-09-01", NOW) == datetime(2026, 9, 1).astimezone()  # local midnight
    assert parse_since("2026-09-01T10:00:00Z", NOW) == datetime(2026, 9, 1, 10, tzinfo=timezone.utc)
    assert parse_since(None, NOW) is None and parse_since("  ", NOW) is None


@pytest.mark.parametrize("text", ["3m", "90M", "3 m", "1.5m"])
def test_m_is_ambiguous_and_exits_2(text):
    with pytest.raises(AmbiguousSince, match="ambiguous: use 90d, 12w or a date") as exc:
        parse_since(text, NOW)
    assert exc.value.exit_code == 2 and isinstance(exc.value, ValueError)


@pytest.mark.parametrize("text", ["90", "3y", "yesterday", "3 months", "d"])
def test_anything_else_unreadable_exits_1(text):
    with pytest.raises(SinceError, match="use 36h, 90d, 12w or a date") as exc:
        parse_since(text, NOW)
    assert exc.value.exit_code == 1 and not isinstance(exc.value, AmbiguousSince)


@pytest.mark.parametrize("text", ["99999999999d", "999999w", "1000000d"])
def test_a_length_past_year_one_is_refused_not_a_traceback(text):
    """Timedelta or the date subtraction overflowed."""
    with pytest.raises(SinceError, match="use 36h, 90d, 12w or a date") as exc:
        parse_since(text, NOW)
    assert exc.value.exit_code == 1 and exc.value.__cause__ is None
    with pytest.raises(SinceError):
        since_days(text, NOW)


def test_the_days_form_for_onboard():
    assert since_days("90d", NOW) == 90 and since_days("36h", NOW) == 1.5 and since_days("2w", NOW) == 14
    assert since_days("2026-09-21T23:00:00-07:00", NOW) == 2
    assert since_days(None, NOW) is None
    with pytest.raises(SinceError, match="not a window in the past"):
        since_days("2026-10-01", NOW)
    with pytest.raises(AmbiguousSince):
        since_days("3m", NOW)


def test_report_since_m_exits_2_and_says_why(tmp_path, capsys):
    home = tmp_path / "lm"
    Store(home)
    code, _, err = cli(capsys, home, "report", "--since", "3m")
    assert code == 2 and "ambiguous: use 90d, 12w or a date" in err
    code, out, _ = cli(capsys, home, "report", "--since", "90M", "--json")
    assert code == 2 and out["ok"] is False and out["exit"] == 2
    code, _, err = cli(capsys, home, "report", "--since", "3y")
    assert code == 1 and "use 36h, 90d, 12w or a date" in err
    code, _, err = cli(capsys, home, "report", "--since", "1.5d")
    assert code == 0, err
    code, _, err = cli(capsys, home, "report", "--since", "99999999999d")
    assert code == 1 and "use 36h, 90d, 12w or a date" in err and "Traceback" not in err


def test_status_names_the_fits_size(tmp_path, capsys):
    store = Store(tmp_path / "lm")
    for name, size in (("fit_20260923210000", 3 * 1024 * 1024), ("fit_20260923220000", 1024 * 1024)):
        folder = store.home / "fits" / name
        folder.mkdir(parents=True)
        (folder / "state.npz").write_bytes(b"\0" * size)
    code, out, err = cli(capsys, store.home, "status", "--json")
    assert code == 0, err
    assert out["size"]["fits_bytes"] == 4 * 1024 * 1024 and out["size"]["bytes"] >= out["size"]["fits_bytes"]
    code, text, err = cli(capsys, store.home, "status")
    assert code == 0, err
    assert ", of which fits 4.0 MB in 2 fit folder(s)" in text
    code, text, err = cli(capsys, tmp_path / "empty", "status")
    assert code == 0 and "of which fits" not in text
