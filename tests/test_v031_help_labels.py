"""0.3.1 help and README labels.

- `gate`, the harness-part estimates (`recommend --harness-variants` and `--harness`, `posterior --level
  harness`), `adapt harness2` and the ADE adapters say "experimental" in their help and in the README.
- `check-change` is not marked experimental, in its help or in the README.
- `scan`, `prior build` and `research` are off `loopmath --help`, `loopmath prior --help` and the README, and stay
  callable by name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from loopmath import cli

README = Path(__file__).resolve().parent.parent / "README.md"


def _help(capsys, *argv: str) -> str:
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--help"])
    assert exc.value.code == 0
    return " ".join(capsys.readouterr().out.split())


def _listing(capsys, *argv: str) -> tuple[list[str], list[str]]:
    """The names on the usage line's `{a,b,...}` and the names listed under it."""
    with pytest.raises(SystemExit):
        cli.main([*argv, "--help"])
    out = capsys.readouterr().out
    usage = out.split("{", 1)[1].split("}", 1)[0].split(",")
    listed = [line.split()[0] for line in out.splitlines() if line.startswith("    ") and line.split()]
    return usage, listed


def _top_line(capsys, name: str) -> str:
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    lines = capsys.readouterr().out.splitlines()
    i = next(i for i, line in enumerate(lines) if line.strip().startswith(name + " "))
    text = [lines[i]] + [line for line in lines[i + 1:i + 4] if line.startswith(" " * 20)]
    return " ".join(" ".join(text).split())


def test_gate_says_experimental(capsys):
    assert "experimental:" in _top_line(capsys, "gate")
    assert "Experimental:" in _help(capsys, "gate")


@pytest.mark.parametrize("argv", [["check-change"], ["check-change", "start"], ["check-change", "status"],
                                  ["check-change", "from-runs"]])
def test_check_change_is_not_marked_experimental(capsys, argv):
    assert "experimental" not in _help(capsys, *argv).lower()
    if argv == ["check-change"]:
        line = _top_line(capsys, "check-change")
        assert "paired A/B check of a change" in line and "experimental" not in line.lower()


def test_harness_part_estimates_say_experimental(capsys):
    rec = _help(capsys, "recommend")
    assert "--harness-variants [N] experimental:" in rec
    assert "--harness PATH|auto experimental:" in rec
    assert "harness: the harness labels and their parts, experimental" in _help(capsys, "posterior")


def test_adapters_and_harness2_say_experimental(capsys):
    assert "(experimental;" in _top_line(capsys, "adapt")
    out = _help(capsys, "adapt")
    assert "registered adapter name (experimental)" in out and "harness2 (experimental), harness-runs" in out
    assert "Experimental: the ADE adapters and the harness2 importer" in out


@pytest.mark.parametrize("name", ["scan", "research"])
def test_scan_and_research_are_off_the_main_help(capsys, name):
    usage, listed = _listing(capsys)
    assert name not in usage and name not in listed
    assert "gate" in usage and "gate" in listed  # the listing itself was read


def test_prior_build_is_off_the_prior_help(capsys):
    usage, listed = _listing(capsys, "prior")
    assert usage == ["show", "import-shared", "remove-shared"]
    assert "build" not in listed and "show" in listed
    assert "build" not in _top_line(capsys, "prior")


@pytest.mark.parametrize("argv, words", [
    (["scan"], "Scanning is implicit"),
    (["research"], "research data folders"),
    (["research", "fit"], "sweep run records"),
    (["prior", "build"], "Rebuild the packaged prior bundle"),
])
def test_hidden_commands_stay_callable_by_name(capsys, argv, words):
    assert words in _help(capsys, *argv)


def test_scan_still_runs(capsys):
    assert cli.main(["scan"]) == 0
    assert "Scanning is implicit" in capsys.readouterr().out


def _section(text: str, heading: str) -> str:
    return text.split(heading, 1)[1].split("\n## ", 1)[0]


def test_readme_marks_the_four_experimental_and_drops_the_hidden():
    text = README.read_text(encoding="utf-8")
    table = _section(text, "## Commands")
    assert re.search(r"\| Harness testing \| `check-change`, `gate` \(experimental\) \|", table)
    assert "`scan`" not in table
    testing = _section(text, "## Testing a harness change")
    assert "**`check-change`.**" in testing and "**`gate` (experimental).**" in testing
    assert "`check-change` (experimental)" not in text
    listed = text.split("These are experimental: ", 1)[1].split(". ", 1)[0]
    assert "`gate`" in listed and "check-change" not in listed
    for marked in ("`gate`", "`adapt harness2`", "harness-part estimates", "ADE adapters"):
        assert re.search(re.escape(marked) + r"[^.\n]*experimental|experimental[^.\n]*" + re.escape(marked),
                         text), marked
    assert "## Research data" not in text
    assert "loopmath research" not in text and "research fit" not in text and "prior build" not in text
    assert "loopmath scan" not in text
