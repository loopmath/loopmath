"""Top-level help keeps up with the commands: `run` lists `record`, and `recommend`
says it gives choices, options 1 to 5. `loopmath` without a command names the commands
and where to start, and exits 2."""

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


def _bare(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_no_command_names_the_commands_and_where_to_start(capsys):
    code, out, err = _bare(capsys)
    assert code == 2 and out == ""
    assert err.startswith("loopmath needs a command, one of:")
    assert "the following arguments are required" not in err and "usage:" not in err
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    usage = capsys.readouterr().out
    listed = " ".join(err.split()).split("one of: ", 1)[1].split(" hint: ")[0].split(", ")
    assert "{" + ",".join(listed) + "}" in "".join(usage.split())  # the same commands as the usage line
    hint = " ".join(err.split()).split(" hint: ", 1)[1]
    assert "`loopmath onboard`" in hint and "`loopmath skill install`" in hint and "`loopmath --help`" in hint


def test_no_command_under_json_prints_one_error_object(capsys):
    import json

    code, out, err = _bare(capsys, "--json")
    assert code == 2 and err.startswith("loopmath needs a command")
    obj = json.loads(out)
    assert obj["schema"] == "loopmath.error/1" and obj["ok"] is False and obj["exit"] == 2
    assert obj["command"] == "loopmath" and obj["error"].startswith("loopmath needs a command, one of: analyze")
    assert "`loopmath onboard`" in obj["hint"]
