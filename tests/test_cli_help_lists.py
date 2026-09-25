"""Top-level help keeps up with the commands (FINDINGS-0.2 I21): `run` lists `record`, and `recommend`
says it gives choices, options 1 to 5."""

from __future__ import annotations

import pytest

from loopmath import cli


def _line(capsys, name: str) -> str:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    return " ".join(next(line for line in out.splitlines() if line.strip().startswith(name + " ")).split())


def test_run_lists_record_and_import(capsys):
    line = _line(capsys, "run")
    assert "record" in line.split(": ", 1)[1] and "import" in line and "start" in line


def test_recommend_says_it_gives_choices(capsys):
    line = _line(capsys, "recommend")
    assert "choices" in line and "options 1 to 5" in line


def test_run_import_help_names_brief(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "import", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--brief" in out and "the overlap with the shipped prior" in out


def test_run_start_choice_help_names_most_likely(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "start", "--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "(goal, pair, reference, most_likely, cheapest_run)" in out
