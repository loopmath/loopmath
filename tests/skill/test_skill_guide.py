"""`loopmath guide [TOPIC]`: the job instructions the one skill sends the agent to (0.3.4).

The topics are 0.3.3's five job skills and its `reference.md`, without frontmatter, with each
cross reference to another skill or to the reference copy turned into a `loopmath guide` call and
nothing else changed, but for the `update` object 0.3.5 documents in the reference. A guide call needs
no store and writes nothing.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import loopmath
from loopmath import cli
from loopmath.skill import guide
from loopmath.skill import install as inst
from v033 import FIXTURE

SRC = Path(loopmath.__file__).resolve().parents[1]
# The 0.3.3 text each topic was made from.
SOURCE = {**{topic: f"{name}/SKILL.md" for name, topic in inst.RETIRED_SKILLS.items()}, "reference": "reference.md"}
# 0.3.5 adds `status --json`'s `update` object to the reference's status line; the rest is 0.3.4's.
UPDATE_035 = (", `update` (null when the update check is off or pypi.org has not answered yet; else `latest`, "
              "`installed`, `available` (true when a newer loopmath is out), `command` (the upgrade command: "
              "offer it when a newer loopmath is out, and run it only if the user agrees), `checked_at` (when "
              "loopmath last asked pypi.org))")


def _old(topic: str) -> str:
    text = (FIXTURE / "skills" / SOURCE[topic]).read_text(encoding="utf-8")
    if text.startswith("---\n"):
        text = text[text.index("\n---\n", 4) + 5:].lstrip("\n")
    return text


def test_the_topics_are_the_five_jobs_then_the_reference():
    assert list(guide.TOPICS) == ["onboard", "import", "plan", "record", "fit", "reference"]
    assert sorted(SOURCE) == sorted(guide.TOPICS)
    with pytest.raises(KeyError):
        guide.text("loopmath-plan-task")


@pytest.mark.parametrize("topic", list(guide.TOPICS))
def test_each_topic_is_its_0_3_3_text_with_only_the_cross_references_rewritten(topic):
    text = guide.text(topic)
    assert (UPDATE_035 in text) == (topic == "reference")
    text = text.replace(UPDATE_035, "")
    assert not text.startswith("---") and text.startswith("# ") and text.endswith("\n")
    old, new = _old(topic).splitlines(), text.splitlines()
    assert len(old) == len(new)  # line for line: no line added or dropped
    changed = [(a, b) for a, b in zip(old, new) if a != b]
    assert changed, f"{topic}: nothing rewritten"
    for a, b in changed:  # a guide call in place of a skill or the reference copy, or "skills" now "guides"
        renamed = re.sub(r"\bskills\b", "guides", a) == b
        assert renamed or ("`loopmath guide " in b and "`loopmath guide " not in a), (a, b)
    diff = "".join(difflib.ndiff([a for a, _ in changed], [b for _, b in changed]))
    assert len(changed) <= 3, diff


RETIRED_NAMES = re.compile(r"loopmath-(onboard|import-runs|update-fit|plan-task|record-run)|\bloopmath skill\b")
STALE = re.compile(r"reference\.md|SKILL\.md|next to this file|sibling folder|\bskills?\b", re.I)


@pytest.mark.parametrize("topic", list(guide.TOPICS))
def test_no_cross_reference_to_a_skill_or_a_reference_copy_is_left(topic):
    text = guide.text(topic)
    assert not RETIRED_NAMES.search(text), RETIRED_NAMES.search(text)
    assert not STALE.search(text), STALE.search(text)


def test_every_guide_call_in_the_skill_and_the_guides_names_a_topic():
    named = []
    for text in (inst.skill_text("loopmath"), *map(guide.text, guide.TOPICS)):
        named += re.findall(r"`loopmath guide ([a-z-]+)`", text)
    assert set(named) <= set(guide.TOPICS) and set(named) >= set(guide.TOPICS)  # each one is reachable


def test_the_skill_sends_each_job_to_its_guide_and_lists_them_with_guide_alone():
    skill = inst.skill_text("loopmath")
    for topic in guide.TOPICS:
        if topic != "reference":
            assert f"`loopmath guide {topic}`" in skill, topic
    assert "`loopmath guide` alone" in skill and "`loopmath status --json`" in skill


def test_guide_alone_lists_each_topic_with_when_to_use_it(capsys):
    assert cli.main(["guide"]) == 0
    out = capsys.readouterr().out
    assert out == guide.listing()
    rows = [ln.split(None, 1) for ln in out.splitlines() if ln.startswith("  ")]
    assert [name for name, _ in rows] == list(guide.TOPICS)
    assert all(len(when) >= 20 and "\N{EM DASH}" not in when for _, when in rows)
    assert out.rstrip("\n").splitlines()[-1] == "Run `loopmath guide <topic>` and follow it."


@pytest.mark.parametrize("topic", list(guide.TOPICS))
def test_guide_topic_prints_the_packaged_text(capsys, topic):
    assert cli.main(["guide", topic]) == 0
    captured = capsys.readouterr()
    assert captured.out == guide.text(topic) and captured.err == ""


def test_an_unknown_topic_exits_2_names_it_and_lists_the_topics(capsys):
    assert cli.main(["guide", "planning"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("error: no guide 'planning'; the topics are onboard, import, plan, record, fit, reference. "
                            "Run `loopmath guide <topic>` and follow it.\n")


def test_guide_has_no_json_or_home_flag():
    """Plain text only, as `skill show`: the guide is text for the agent, not data."""
    for flag in ("--json", "--home"):
        with pytest.raises(SystemExit) as exc:
            cli.main(["guide", "plan", flag, *(["x"] if flag == "--home" else [])])
        assert exc.value.code == 2


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_a_guide_call_with_no_store_writes_nothing(tmp_path):
    """Under an empty temp HOME, no LOOPMATH_HOME and no cache folder set: no store, no cache, no file."""
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home), "PYTHONPATH": str(SRC), "PYTHONDONTWRITEBYTECODE": "1",
           "LANG": "en_US.UTF-8", "LOOPMATH_NO_UPDATE_CHECK": "1"}
    for argv, code in ((["guide"], 0), (["guide", "plan"], 0), (["guide", "reference"], 0), (["guide", "nope"], 2)):
        res = subprocess.run([sys.executable, "-m", "loopmath", *argv], env=env, cwd=work, capture_output=True,
                             text=True, stdin=subprocess.DEVNULL, timeout=60)
        assert res.returncode == code, (argv, res.stderr)
        assert res.stdout if code == 0 else res.stderr
    assert _tree(tmp_path) == ["home", "work"]
    assert not os.path.exists(home / ".loopmath")
