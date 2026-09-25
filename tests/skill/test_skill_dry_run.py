"""Spec 07 section 3: a scripted dry run of the skill's flow with fake agents.

The dry run follows SKILL.md as an orchestrator would, in process, on a temp store with empty
log folders: `recommend` saved to rec.json, a pair started from it (`--new-slate`, then
`--slate`), one `--harness command` attempt per piece, verdicts, the blinded preference, and
`run finish` for both. It must leave two valid OCP v0.3 runs, two receipts and a refit that
started and stamped them. It skips, naming the command, while a verb is still a lane stub.

A second test checks that every `loopmath ...` command the skill and the README show names a
real command and only flags that command has; two more check that the shell hands each shown
argument to loopmath intact (a bare `--target heldout_perf>=2400` is a redirection).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from loopmath import cli
from loopmath.skill import install as inst

ROOT = Path(__file__).resolve().parents[2]
BASE = "a" * 40
FIT_WAIT_S = 180


# ---------------------------------------------------------------- the commands the text shows
class _Parser(Exception):
    pass


def _parser(monkeypatch) -> argparse.ArgumentParser:
    """The real top-level parser, as `cli.main` builds it."""
    def grab(self, *a, **k):
        raise _Parser(self)

    with monkeypatch.context() as m:
        m.setattr(argparse.ArgumentParser, "parse_args", grab)
        try:
            cli.main(["--help"])
        except _Parser as got:
            return got.args[0]
    raise AssertionError("cli.main did not build a parser")


def _choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _commands(text: str) -> list[str]:
    """`loopmath ...` lines in sh blocks (backslash continuations joined) and inline code spans."""
    found = []
    for block in re.findall(r"```sh\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("loopmath "):
                found.append(line)
    found += re.findall(r"`(loopmath [^`]+)`", text)
    return found


@pytest.mark.parametrize("source", ["skill", "README.md"])
def test_every_command_shown_parses(monkeypatch, source):
    text = inst.skill_text() if source == "skill" else (ROOT / "README.md").read_text(encoding="utf-8")
    top = _parser(monkeypatch)
    commands = _commands(text)
    assert len(commands) >= 10, commands
    problems = []
    for command in commands:
        words = command.replace("[", " ").replace("]", " ").split()[1:]
        parser, chain = top, []
        while words and words[0] in _choices(parser):
            chain.append(words[0])
            parser = _choices(parser)[words.pop(0)]
        if words and words[0].startswith("<"):
            continue  # `loopmath <command> --help`
        if not chain and not (words and words[0].startswith("--")):
            problems.append(f"{command}: no such command")
            continue
        known = {s for a in parser._actions for s in a.option_strings}
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", " ".join(words)):
            if flag not in known:
                problems.append(f"{command}: `loopmath {' '.join(chain)}` has no {flag}")
    assert not problems, "\n".join(problems)


def _texts() -> dict[str, str]:
    return {"skill": inst.skill_text(), "README.md": (ROOT / "README.md").read_text(encoding="utf-8")}


def _shell_snippets(text: str) -> list[str]:
    """Everything a reader may paste into a shell: sh block lines, and inline spans that are a
    `loopmath ...` command or start with a flag."""
    found = []
    for block in re.findall(r"```sh\n(.*?)```", text, re.S):
        found += [ln.split("#", 1)[0] for ln in block.replace("\\\n", " ").splitlines()]
    found += [s for s in re.findall(r"`([^`\n]+)`", text) if s.startswith(("loopmath ", "--"))]
    return [s.strip() for s in found if s.strip()]


# `<` or `>` inside a word, outside quotes: the shell takes it as a redirection and the word
# is cut there. A placeholder (`<command>`) and a spaced redirect (`> rec.json`) are fine.
IN_WORD_REDIRECT = re.compile(r"[\w\])][<>]=?[\w.]")


@pytest.mark.parametrize("source", ["skill", "README.md"])
def test_no_example_hands_the_shell_a_comparison(source):
    bad = [s for s in _shell_snippets(_texts()[source])
           if IN_WORD_REDIRECT.search(re.sub(r"'[^']*'|\"[^\"]*\"", "''", s))]
    assert not bad, "quote these, the shell would redirect: " + " | ".join(bad)


def _through_bash(bash: str, cwd: Path, words: str) -> list[str]:
    """The argv bash hands to `loopmath` for `loopmath WORDS`, run in `cwd`."""
    out = cwd.parent / "argv"
    script = f'loopmath() {{ printf "%s\\0" "$@" > {shlex.quote(str(out))}; }}; loopmath {words}'
    subprocess.run([bash, "--noprofile", "--norc", "-c", script], cwd=cwd, check=True,
                   env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL, capture_output=True)
    return out.read_bytes().decode().split("\0")[:-1]


def test_target_examples_reach_the_parser_intact(monkeypatch, tmp_path):
    """Each `--target` rule example, pasted into bash, gives argparse the whole expression and
    writes no file."""
    bash = shutil.which("bash") or pytest.skip("no bash on PATH")
    top = _parser(monkeypatch)
    raws = [m for text in _texts().values() for s in _shell_snippets(text)
            if not s.startswith("loopmath skill")
            for m in re.findall(r"--target\s+('[^']*'|\"[^\"]*\"|[^\s`\]]+)", s) if re.search("[<>]", m)]
    assert len(raws) >= 4, raws  # two in the skill, two in the README (`outcome --target X` is a number)
    task = "recommend --type feature --repo acme/web --target "
    for n, raw in enumerate(raws):
        cwd = tmp_path / f"cwd{n}"
        cwd.mkdir()
        argv = _through_bash(bash, cwd, task + raw)
        assert not list(cwd.iterdir()), f"--target {raw} made files: {sorted(p.name for p in cwd.iterdir())}"
        want = shlex.split(raw)[0]
        assert re.fullmatch(r"\w+(>=|<=|>|<)[\d.]+", want), want
        assert top.parse_args(argv).target == want, (raw, argv)

    # The probe itself: the same expression unquoted loses its tail and leaves a file.
    cwd = tmp_path / "bare"
    cwd.mkdir()
    argv = _through_bash(bash, cwd, task + "heldout_perf>=2400")
    assert argv[-1] == "heldout_perf" and [p.name for p in cwd.iterdir()] == ["=2400"]


# ---------------------------------------------------------------- the dry run
@pytest.fixture
def lm(tmp_path, monkeypatch, capsys):
    """Run `loopmath ARGV` in process on a temp store; JSON output parsed. Skips on a lane stub."""
    for name in ("home", "claude", "codex", "cache"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "store"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)

    def run(*argv: str):
        capsys.readouterr()
        try:
            code = cli.main(list(argv))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
        out, err = capsys.readouterr()
        if code == 3 and "not implemented yet" in err:
            pytest.skip(f"`loopmath {' '.join(argv[:2])}` is still a stub in this tree: {err.strip()}")
        assert code == 0, f"loopmath {' '.join(argv)} exited {code}: {err.strip()[-800:]}"
        try:
            return json.loads(out)
        except ValueError:
            return out

    return run


def _wait(pid: int) -> None:
    """Wait for a detached fit this process started (it is our child), within FIT_WAIT_S."""
    end = time.monotonic() + FIT_WAIT_S
    while time.monotonic() < end:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:  # already reaped
            return
        if done:
            return
        time.sleep(0.2)
    raise AssertionError(f"the refit (pid {pid}) did not end within {FIT_WAIT_S} s")


def test_skill_dry_run_pair_with_command_agents(lm, tmp_path):
    types = lm("task-types", "--json")
    assert "bug_fix" in json.dumps(types)
    lm("fit", "--json")  # stands in for onboard's first fit

    # Plan: recommend, saved where run start reads the task from (SKILL.md steps 2 and 5).
    rec = lm("recommend", "--type", "bug_fix", "--repo", "acme/api", "--feature", "size=s",
             "--feature", "lang=python", "--json")
    assert rec["message"] and rec["curve"] and rec["rec"]
    members = rec["pair"]["members"]
    assert len(members) == 2 and members[0] != members[1]
    rec_file = tmp_path / "rec.json"
    rec_file.write_text(json.dumps(rec), encoding="utf-8")

    # Record: the pair from one rec.json and one base commit.
    usual = rec["usual"] or {}  # null without a habit (recommend/2); the baseline is then `reference`
    first_source = "usual" if members[0] == (usual.get("config") or {}).get("id") else "alternative"
    starts = [lm("run", "start", "--task-file", str(rec_file), "--base-commit", BASE, "--config", members[0],
                 "--source", first_source, "--rec", rec["rec"], "--new-slate", "--json")]
    slate = starts[0]["slate"]
    starts.append(lm("run", "start", "--task-file", str(rec_file), "--base-commit", BASE, "--config", members[1],
                     "--source", "exploration", "--rec", rec["rec"], "--slate", slate, "--json"))
    assert starts[1]["slate"] == slate
    assert starts[0]["task"] == starts[1]["task"] == rec["task"]["id"]

    for i, start in enumerate(starts):
        worktree = tmp_path / f"wt{i}"
        worktree.mkdir()
        doc = json.loads(Path(start["path"]).read_text(encoding="utf-8"))
        settings = doc["run"]["configuration"].get("settings") or {}
        last = None
        for piece in start["pieces"]:
            s = settings.get(piece) or {}
            model = s.get("model") if isinstance(s.get("model"), str) else (s.get("model") or {}).get("id")
            att = lm("run", "attempt", "--run", start["run"], "--piece", piece, "--harness", "command",
                     "--model", str(model), "--effort", str(s.get("effort")), "--cwd", str(worktree), "--json")
            lm("run", "attempt", "--run", start["run"], "--end", att["attempt"], "--status", "done", "--json")
            last = att["attempt"]
        lm("outcome", "--run", start["run"], "--signal", "tests=pass", "--kind", "verdict", "--at-attempt", last,
           "--source", "ci", "--tier", "verified", "--json")

    # Pair: the blinded referee's preference, then finish both (each may trigger the refit).
    pref = lm("outcome", "--slate", slate, "--prefer", starts[0]["run"], "--judge", "referee", "--blinded", "--json")
    assert pref["winner"] == starts[0]["run"]
    finishes = [lm("run", "finish", "--run", s["run"], "--json") for s in starts]

    pids = [f["fit"]["pid"] for f in finishes if isinstance(f.get("fit"), dict) and f["fit"].get("started")]
    assert pids, [f.get("fit") for f in finishes]
    for pid in pids:
        _wait(pid)

    # Two valid OCP v0.3 runs.
    check = lm("ocp", "validate", *(s["path"] for s in starts), "--json")
    assert check["ok"], check
    assert [(f["ocp"], f["errors"]) for f in check["files"]] == [("0.3", 0), ("0.3", 0)]

    # Two receipts, both stamped by the refit that finish started.
    store = tmp_path / "store"
    job = json.loads((store / "fits" / "job.json").read_text(encoding="utf-8"))
    assert job["status"] == "done", job
    receipts = sorted((store / "receipts").glob("rct_*.json"))
    assert [f["receipt"] for f in finishes] and len(receipts) == 2
    for path in receipts:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        assert receipt["rec"] == rec["rec"]
        after = receipt["after"].get("fit_after")
        assert after and after != receipt["fit"] and (store / "fits" / after).is_dir(), receipt["after"]
