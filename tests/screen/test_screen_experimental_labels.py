"""The screen ships marked experimental, the way 0.3.1 marks `gate`.

- `loopmath --help` lists `screen` with "experimental:", and each of its verbs' own help starts its description
  with "Experimental:", so a user who lands on `screen status --help` sees it too.
- `run start --screen` and `run record --screen` say experimental.
- The CHANGELOG's 0.3.2 section names `screen` as experimental.
- A check arm with `--screen` and a screen arm with `--check` are refused, naming the arms that fit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from loopmath import cli

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_screen_cli import SOLO, cost, home, lm, start  # noqa: E402,F401


def _help(capsys, *argv: str) -> str:
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        cli.main([*argv, "--help"])
    assert exc.value.code == 0
    return capsys.readouterr().out


def _description(text: str) -> str:
    """The paragraph after the usage lines."""
    body = text.split("\n\n", 1)[1]
    return " ".join(body.split("\n\n", 1)[0].split())


def test_the_top_help_marks_screen(capsys):
    lines = _help(capsys).splitlines()
    i = next(i for i, line in enumerate(lines) if line.strip().startswith("screen "))
    text = " ".join(" ".join([lines[i]] + [x for x in lines[i + 1:i + 3] if x.startswith(" " * 20)]).split())
    assert "experimental:" in text


@pytest.mark.parametrize("argv", [["screen"], ["screen", "start"], ["screen", "next"], ["screen", "status"],
                                  ["screen", "apply"]])
def test_each_screen_help_starts_with_experimental(capsys, argv):
    assert _description(_help(capsys, *argv)).startswith("Experimental:"), argv


@pytest.mark.parametrize("verb", ["start", "record"])
def test_the_run_screen_flag_says_experimental(capsys, verb):
    out = " ".join(_help(capsys, "run", verb).split())
    assert "--screen SCR experimental:" in out


def test_the_changelog_names_the_screen_experimental():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert text.startswith("# Changelog\n\n## 0.3.2")
    section = text.split("## 0.3.2", 1)[1].split("\n## 0.3.1", 1)[0]
    assert "`screen`" in section and "experimental" in section
    for verb in ("screen start", "screen next", "screen status", "screen apply", "run record --screen"):
        assert verb in section, verb


def test_a_check_with_a_screen_arm_is_refused(home, tmp_path, capsys):
    code, out, err = lm(capsys, "run", "record", "--check", "chk_01ABC", "--arm", "A", *SOLO,
                        "--cwd", f"implement={home.parent / 'wt'}", "--since", "1h", "--no-commits", "--no-fit",
                        "--json")
    assert code == 1
    assert "--arm base" in out["error"] and "cand" in out["error"] and "--screen" in out["error"]


def test_a_screen_with_a_check_arm_is_refused_on_run_start(home, tmp_path, capsys):
    scr = start(capsys, tmp_path, 3, 4)["screen"]
    code, out, err = lm(capsys, "run", "start", "--screen", scr, "--arm", "base", *SOLO, "--json")
    assert code == 1 and "--arm A" in out["error"]
    assert json.loads(json.dumps(out))["ok"] is False
