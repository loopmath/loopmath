"""Default `--html` page names from `output.html_path` (recommend, report, workflows show): stamped to the
second, and a second page in the same second gets `-2`, `-3` instead of replacing the first (spec 02 section 1)."""

from __future__ import annotations

import datetime as dt
import types
from pathlib import Path

from loopmath import output


def test_pages_in_the_same_second_each_keep_their_own_name(tmp_path, monkeypatch):
    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime(2026, 9, 24, 19, 45, 7)

    monkeypatch.setattr(output, "_dt", types.SimpleNamespace(datetime=Clock))
    home = str(tmp_path / "home")
    names = []
    for text in ("one", "two", "three"):
        path = output.html_path("recommend", output.HTML_DEFAULT, home)
        output.write_html(path, f"<p>{text}</p>")
        names.append(path.name)
    assert names == ["recommend-20260924-194507.html", "recommend-20260924-194507-2.html",
                     "recommend-20260924-194507-3.html"]
    views = tmp_path / "home" / "views"
    assert [(views / n).read_text() for n in names] == ["<p>one</p>", "<p>two</p>", "<p>three</p>"]


def test_a_given_path_is_used_as_is_and_no_flag_is_no_page(tmp_path):
    given = tmp_path / "page.html"
    given.write_text("old")
    assert output.html_path("recommend", str(given), str(tmp_path)) == given
    assert output.html_path("recommend", None, str(tmp_path)) is None
