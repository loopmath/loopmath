"""Spec 07: static checks on the packaged skill and the `loopmath guide` texts.

The skill is named for its folder, has a short description and may run only loopmath and the few
commands the guides show; each job guide sends flags and fields to `loopmath guide reference` (the
flow topic, to its own last section). Every `loopmath ...` command the skill, the guides and the
README show names a real command, only flags that command has and only values its choices allow;
every flag a guide uses is in the reference (a flow command's, in the flow topic's last section).
No example hands the shell a comparison it would read as a redirection. The texts carry no em dash
and none of the private names.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from loopmath.skill import guide
from loopmath.skill import install as inst
from skill_texts import ROOT, PLACEHOLDER, commands, frontmatter, parser, texts, walk, words

SOURCES = [*texts(), "README.md"]
JOBS = [t for t in guide.TOPICS if t != "reference"]
FLOW_COMMANDS = "## The flow commands"


def _text(source: str) -> str:
    return (ROOT / source).read_text(encoding="utf-8") if source == "README.md" else texts()[source]


@pytest.mark.parametrize("name", inst.SKILLS)
def test_the_skill_has_its_folder_name_a_short_description_and_its_tools(name):
    text = inst.skill_text(name)
    meta = frontmatter(text)
    assert meta["name"] == name
    desc = meta["description"]
    assert 0 < len(desc) <= 1024 and "loopmath" in desc, desc
    assert meta["allowed-tools"].split(", ")[0] == "Bash(loopmath:*)", meta["allowed-tools"]
    body = text.split("---\n", 2)[2]
    assert "`loopmath guide`" in body and "`--help`" in body


@pytest.mark.parametrize("topic", JOBS)
def test_each_job_guide_sends_flags_to_its_reference_and_never_to_help(topic):
    text = guide.text(topic)
    if topic == "flow":  # the flow commands' reference is the topic's own last section
        assert ("Their flags, JSON members and exit codes are in the last section of this guide, "
                '"The flow commands".') in " ".join(text.split())
        assert re.findall(r"^## .+$", text, re.M)[-1] == FLOW_COMMANDS
    else:
        assert "Flags and fields are in `loopmath guide reference`." in text
    assert "run `--help`" in text.lower()


@pytest.mark.parametrize("source", SOURCES)
def test_every_command_shown_parses(source):
    top = parser()
    shown = [c for _, c in commands(_text(source))]
    assert shown, f"{source} shows no loopmath command"
    problems = []
    for command in shown:
        chain, p, rest = walk(top, words(command))
        if rest and rest[0].startswith("<"):
            continue  # the README's `loopmath <command> --help`
        if not chain and not (rest and rest[0].startswith("--")):
            problems.append(f"{command}: no such command")
            continue
        flags = {s: a for a in p._actions for s in a.option_strings}
        for i, w in enumerate(rest):
            if not w.startswith("--"):
                continue
            action = flags.get(w)
            if action is None:
                problems.append(f"{command}: `loopmath {' '.join(chain)}` has no {w}")
            elif w == "--help" and source != "README.md":
                problems.append(f"{command}: an agent never runs --help")
            elif action.choices and i + 1 < len(rest) and not PLACEHOLDER.fullmatch(rest[i + 1]) \
                    and rest[i + 1] not in action.choices:
                problems.append(f"{command}: {w} {rest[i + 1]} is not one of {sorted(action.choices)}")
    assert not problems, "\n".join(problems)


def test_every_flag_the_skill_or_a_guide_uses_is_in_the_reference():
    top, ref = parser(), guide.text("reference")
    flow_ref = guide.text("flow").split(FLOW_COMMANDS, 1)[1]
    missing = set()
    for name, text in texts().items():
        for _, command in commands(text):
            chain, _, rest = walk(top, words(command))
            where = flow_ref if chain[:1] == ["flow"] else ref
            missing |= {f"{name}: loopmath {' '.join(chain)} {w}" for w in rest
                        if w.startswith("--") and w not in where}
    assert not missing, sorted(missing)


def test_no_text_names_a_retired_skill_or_command():
    for name, text in texts().items():
        assert not re.findall(r"`(loopmath-[a-z-]+)`", text), name
        assert "loopmath plan " not in text and "loopmath plan`" not in text, name


TOOLS = {"Bash(loopmath:*)", "Bash(git rev-parse:*)", "Bash(git remote get-url:*)", "Bash(open:*)", "Bash(xdg-open:*)"}


def test_the_skill_allows_only_loopmath_git_reads_and_open_and_no_text_names_a_web_tool():
    """The skill may run loopmath, read git and open a page without a prompt; nothing else, and never the web."""
    for name in inst.SKILLS:
        assert set(frontmatter(inst.skill_text(name))["allowed-tools"].split(", ")) <= TOOLS, name
    for name, text in texts().items():
        assert not re.search(r"\b(WebFetch|WebSearch|curl|wget)\b|https?://", text), name


def test_no_em_dash_or_private_name():
    em_dash = "\u2014"
    names_file = ROOT / "scripts" / "private-names.txt"
    private = None
    if names_file.is_file():
        names = [ln.strip() for ln in names_file.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        private = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.I) if names else None
    for name, text in texts().items():
        assert em_dash not in text, name
        if private:
            assert not private.search(text), f"{name} names a private name"


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


@pytest.mark.parametrize("source", SOURCES)
def test_no_example_hands_the_shell_a_comparison(source):
    bad = [s for s in _shell_snippets(_text(source))
           if IN_WORD_REDIRECT.search(re.sub(r"'[^']*'|\"[^\"]*\"", "''", s))]
    assert not bad, "quote these, the shell would redirect: " + " | ".join(bad)


def _through_bash(bash: str, cwd: Path, argv_words: str) -> list[str]:
    """The argv bash hands to `loopmath` for `loopmath WORDS`, run in `cwd`."""
    out = cwd.parent / "argv"
    script = f'loopmath() {{ printf "%s\\0" "$@" > {shlex.quote(str(out))}; }}; loopmath {argv_words}'
    subprocess.run([bash, "--noprofile", "--norc", "-c", script], cwd=cwd, check=True,
                   env={"PATH": "/usr/bin:/bin"}, stdin=subprocess.DEVNULL, capture_output=True)
    return out.read_bytes().decode().split("\0")[:-1]


def test_target_examples_reach_the_parser_intact(tmp_path):
    """Each `--target` rule example, pasted into bash, gives argparse the whole expression and
    writes no file."""
    bash = shutil.which("bash") or pytest.skip("no bash on PATH")
    top = parser()
    raws = [m for source in SOURCES for s in _shell_snippets(_text(source))
            if not s.startswith("loopmath skill")
            for m in re.findall(r"--target\s+('[^']*'|\"[^\"]*\"|[^\s`\]]+)", s) if re.search("[<>]", m)]
    assert len(raws) >= 4, raws  # the plan guide and the reference, and two in the README
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
