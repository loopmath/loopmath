"""The screen's docs page and its help, from a new user's view.

- `docs/screen.md` exists and says the screen is experimental; where the export script is present (it is private, so
  not in the public tree), the page is on its allow list.
- Every `loopmath ...` command in it, and in `screen --help`'s closing flow, parses with the CLI as written.
- It has a worked example with real numbers: 48 sessions, the one edit it named, the best subset's predicted cut and
  range, and the check against no file on other tasks.
- It names the known limits: the range, the effect of having the file at all, ties on success.
- It names no private path (the public export's scan checks private names), and has no long dash.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.cli_errors import JsonErrorParser

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "screen.md"
EXPORT = ROOT / "scripts" / "export-public.sh"
PRIVATE_PATH = re.compile(r"/Users/|/home/|~/|\$HOME")  # the export's own scan covers private names


class Parsed(Exception):
    pass


def _parses(monkeypatch, argv: list[str]) -> None:
    original = JsonErrorParser.parse_args

    def stop_after_parsing(self, *a, **k):
        raise Parsed(original(self, *a, **k))

    monkeypatch.setattr(JsonErrorParser, "parse_args", stop_after_parsing)
    try:
        with pytest.raises(Parsed):
            cli.main(argv)
    finally:
        monkeypatch.setattr(JsonErrorParser, "parse_args", original)


def _commands(text: str) -> list[list[str]]:
    """Each `loopmath ...` line inside a fenced block, split as a shell would (a trailing comment dropped)."""
    out = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.S):
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("$ "):
                line = line[2:]
            if line.startswith("loopmath "):
                out.append(shlex.split(line, comments=True)[1:])
    return out


def test_the_page_exists_and_says_experimental():
    assert DOC.is_file()
    text = DOC.read_text(encoding="utf-8")
    assert text.startswith("# ") and "experimental" in text.lower()


@pytest.mark.skipif(not EXPORT.is_file(), reason="scripts/export-public.sh is private")
def test_the_page_is_on_the_export_list():
    assert '"docs/screen.md",' in EXPORT.read_text(encoding="utf-8")


def test_every_command_on_the_page_parses(monkeypatch):
    commands = _commands(DOC.read_text(encoding="utf-8"))
    verbs = {" ".join(c[:2]) for c in commands}
    for needed in ("screen start", "screen next", "run record", "screen status", "screen apply", "check-change start"):
        assert needed in verbs, needed
    for argv in commands:
        _parses(monkeypatch, argv)


def test_the_page_has_the_worked_example_and_the_limits():
    text = DOC.read_text(encoding="utf-8")
    example = text.split("## Example", 1)[1].split("\n## ", 1)[0]
    for fact in ("48 sessions", "8 edits", "12 pairs", "0.94", "18%", "2% to 30%", "33%", "list-equivalent"):
        assert fact in example, fact
    limits = text.split("## Limits", 1)[1].split("\n## ", 1)[0].lower()
    for topic in ("range", "the file at all", "tie", "12 pairs"):
        assert topic in limits, topic


def test_the_page_names_no_private_path_and_has_no_long_dash():
    text = DOC.read_text(encoding="utf-8")
    assert PRIVATE_PATH.findall(text) == []
    assert "\u2014" not in text


def test_the_screen_help_ends_with_the_flow_and_each_step_parses(capsys, monkeypatch):
    capsys.readouterr()
    with pytest.raises(SystemExit):
        cli.main(["screen", "--help"])
    out = capsys.readouterr().out
    flow = out.split("the flow:", 1)[1]
    steps = [shlex.split(line.strip())[1:] for line in flow.splitlines() if line.strip().startswith("loopmath ")]
    assert [" ".join(s[:2]) for s in steps] == ["screen start", "screen next", "run record", "screen status",
                                                "screen apply", "check-change start"]
    for argv in steps:
        _parses(monkeypatch, argv)
