"""`loopmath guide eval`, the eval topic (0.4.5), against the commands it teaches.

Every `loopmath eval ...` line in its sh blocks parses with the real parser, and the blocks walk every stage. Its
last section, "The eval commands", names every eval command and each flag the parser gives that command. The
refusal object and the exit codes it states are what a refused and a mistyped command give. The catalog it describes
is the one `eval simulate --list` prints, and the specs it says write a value file are the ones that do. Its practice
block runs as written up to the sampler, and the fit options its limits call not built are refused. The skill tests
(`tests/skill/test_skill_guide.py`, `test_skill_lint.py`) check the topic list, the cross references and that every
flag a guide shows is in its reference. Commands run in this process on simulated studies under tmp_path; nothing
samples.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shlex
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.eval import commands, judges, specs
from loopmath.skill import guide

TEXT = guide.text("eval")
FLAT = " ".join(TEXT.split())  # a wrapped line reads as one
HEADING = "## The eval commands"
SECTION = TEXT.split(f"\n{HEADING}\n", 1)[1]
SH = re.compile(r"```sh\n(.*?)```", re.S)
# The stages the walk goes through; `scores add` and `scores export` are in the last section only.
WALK = ("rubric check", "scores import", "scores check", "fit", "judges", "value", "steps", "labels next", "simulate")
# Flags every command that reads a fit file takes; the last section names them once, after the list.
FIT_READER = {"--study", "--strict", "--allow-unconverged"}


def lm(*argv: object) -> tuple[int, str, str]:
    """`loopmath ARGV` in this process: the exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main([str(word) for word in argv])
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    return code, out.getvalue(), err.getvalue()


def parser() -> argparse.ArgumentParser:
    """The real top-level parser, as `cli.main` builds it."""
    class Built(Exception):
        pass

    def grab(self, *args, **kwargs):
        raise Built(self)

    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["--help"])
    except Built as got:
        return got.args[0]
    finally:
        argparse.ArgumentParser.parse_args = real
    raise AssertionError("cli.main did not build a parser")


def sub(p: argparse.ArgumentParser, word: str) -> argparse.ArgumentParser:
    [action] = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
    return action.choices[word]


def shown() -> list[str]:
    """Every `loopmath eval ...` line of the topic's sh blocks."""
    return [line.strip() for block in SH.findall(TEXT) for line in block.splitlines()
            if line.strip().startswith("loopmath eval ")]


def has_flag(text: str, flag: str) -> bool:
    """`flag` as a whole word in `text`: `--run` is not found in `--runs`."""
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", text) is not None


def test_every_eval_command_in_the_blocks_parses_and_the_walk_covers_each_stage(capsys):
    top, lines = parser(), shown()
    assert len(lines) >= len(WALK), lines
    for line in lines:
        try:
            top.parse_args(shlex.split(line)[1:])
        except SystemExit:
            raise AssertionError(f"does not parse: {line}: {capsys.readouterr().err.strip()[-300:]}")
    for stage in WALK:
        assert any(line.startswith(f"loopmath eval {stage} ") for line in lines), stage


def test_the_last_section_names_every_eval_command_and_each_of_its_flags():
    assert re.findall(r"^## .+$", TEXT, re.M)[-1] == HEADING
    bullets = [b.split("\n\n", 1)[0] for b in SECTION.split("\n- ")[1:]]
    after = SECTION.rstrip("\n").split("\n\n")[-1]  # the paragraph after the list
    eval_parser = sub(parser(), "eval")
    problems = []
    for words in commands.COMMANDS:
        p = eval_parser
        for word in words.split():
            p = sub(p, word)
        mine = [b for b in bullets if b.startswith(f"`loopmath eval {words} ")]
        if len(mine) != 1:
            problems.append(f"eval {words}: {len(mine)} bullets")
            continue
        flags = {s for a in p._actions for s in a.option_strings if s.startswith("--") and s != "--help"}
        for flag in sorted(flags):
            shared = flag in FIT_READER and f"`eval {words}`" in after and has_flag(after, flag)
            if not (has_flag(mine[0], flag) or shared):
                problems.append(f"eval {words}: {flag} is not named")
    assert not problems, "\n".join(problems)
    assert len(bullets) == len(commands.COMMANDS)  # one bullet per command, and no other


def test_a_refusal_and_a_usage_error_exit_as_the_last_section_says(tmp_path):
    code, out, err = lm("eval", "judges", tmp_path / "missing" / "fit.json", "--json")
    assert code == 1 and err.startswith("error: ")
    got = json.loads(out)
    assert list(got) == ["schema", "ok", "command", "error", "exit"]
    assert got["schema"] == "loopmath.error/1" and got["ok"] is False and got["exit"] == 1
    assert "`schema` (`loopmath.error/1`), `ok` (false), `command`, `error` and `exit`" in " ".join(SECTION.split())
    assert "1 a refusal" in FLAT and "2 usage" in FLAT
    code, _, err = lm("eval", "fit", tmp_path)  # no --rubric, no --judges
    assert code == 2 and "--rubric" in err
    assert f"`{judges.SCHEMA}`" in SECTION  # the schema it gives as an example is the judges report's


def test_the_catalog_is_the_one_simulate_lists_and_the_named_specs_write_a_value_file(tmp_path):
    code, out, _ = lm("eval", "simulate", "--list")
    listed = [line.split()[0] for line in out.splitlines() if line.strip()]
    assert code == 0 and "linear" in listed
    assert f"The catalog has {len(listed)} specs" in FLAT
    with_value = set()
    for name in listed:  # each spec on 12 inputs: the files it writes do not depend on the size
        spec = tmp_path / f"{name}.spec.json"
        spec.write_text(json.dumps(specs.get(name, inputs=12)), encoding="utf-8")
        assert lm("eval", "simulate", spec, "--out", tmp_path / name, "--seed", 0)[0] == 0, name
        if (tmp_path / name / "value.json").exists():
            with_value.add(name)
    said = re.search(r"; (`[a-z-]+`(?:, `[a-z-]+`)* and `[a-z-]+`) also write a `value\.json`", FLAT)
    assert said and set(re.findall(r"`([a-z-]+)`", said.group(1))) == with_value


def test_the_practice_block_runs_as_written_up_to_the_sampler(tmp_path, monkeypatch):
    """The block after "Try it without data", with `--check-only` on its fit so nothing samples; without
    `--allow-simulated` the fit is refused, and the fit options the limits call not built are refused by name."""
    monkeypatch.chdir(tmp_path)
    block = SH.findall(TEXT.split("## Try it without data", 1)[1])[0]
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    assert [line.split()[2] for line in lines] == ["simulate", "simulate", "fit"]
    for line in lines[:2]:
        assert lm(*shlex.split(line)[1:])[0] == 0, line
    fit = shlex.split(lines[2])[1:]
    before = sorted(p.name for p in Path("study").iterdir())
    code, out, err = lm(*fit, "--check-only")
    assert code == 0 and out.startswith("design: ") and sorted(p.name for p in Path("study").iterdir()) == before
    code, _, err = lm(*[w for w in fit if w != "--allow-simulated"], "--check-only")
    assert code == 1 and "--allow-simulated" in err
    for flag, value in (("--halo", "full"), ("--cuts", "free")):
        assert f"`eval fit {flag} {value}`" in FLAT or f"`{flag} {value}`" in FLAT
        code, _, err = lm(*fit, "--check-only", flag, value)
        assert code == 1 and "not built" in err, (flag, err)


def test_eval_record_sits_after_eval_and_each_sends_the_agent_to_the_other():
    topics = list(guide.TOPICS)
    assert topics[-3:] == ["eval", "eval-record", "reference"]
    assert "`loopmath guide eval-record`" in TEXT and "`loopmath guide eval`" in guide.text("eval-record")
    assert TEXT.splitlines()[0].startswith("# ") and TEXT.splitlines()[0].endswith("(experimental)")
    for topic in ("eval", "eval-record"):
        assert guide.TOPICS[topic].endswith("(experimental)")
        assert guide.text(topic).startswith("# ") and "experimental" in guide.text(topic).lower()
