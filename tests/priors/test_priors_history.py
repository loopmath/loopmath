"""The repo-history miner and the shipped task set (lane 11)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from loopmath.priors.history import (classify_files, is_runnable_test, main, mine_repo, select_tasks, task_type,
                                     validate_task)
from loopmath.taskmodel import TASK_TYPES
from loopmath.types import Task

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@example.invalid", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}

CALC = "def add(a, b):\n    return a + b\n"
CALC_FIXED = CALC + "\n\ndef div(a, b):\n    if b == 0:\n        return None\n    return a / b\n"
TEST = "import calc\n\n\ndef test_add():\n    assert calc.add(1, 2) == 3\n"
TEST_FIXED = TEST + "\n\ndef test_div_zero():\n    assert calc.div(1, 0) is None\n"
TEST_MORE = TEST_FIXED + "\n\ndef test_div():\n    assert calc.div(4, 2) == 2\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=ENV).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path) -> Path:
    r = tmp_path / "toy"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _commit(r, {"pyproject.toml": "[tool.pytest.ini_options]\npythonpath = ['src']\n", "src/calc.py": CALC,
                "tests/test_calc.py": TEST}, "Initial calculator")
    _commit(r, {"src/calc.py": CALC_FIXED, "tests/test_calc.py": TEST_FIXED}, "fix(calc): division by zero")
    _commit(r, {"tests/test_calc.py": TEST_MORE}, "Add tests for div")
    _commit(r, {"README.md": "# toy\n"}, "docs: readme")
    _commit(r, {"src/calc.py": CALC_FIXED + "\n# tidy\n"}, "Tidy calc")  # no test file: not a candidate
    return r


def test_classification_rules():
    assert is_runnable_test("tests/test_x.py") and is_runnable_test("app/ui/src/a.test.tsx")
    assert is_runnable_test("crate/tests/it.rs") and not is_runnable_test("tests/conftest.py")
    assert not is_runnable_test("spec/ocp_conformance.py")
    groups = classify_files([{"path": "src/a.py", "added": 3, "deleted": 1, "binary": False},
                             {"path": "tests/test_a.py", "added": 5, "deleted": 0, "binary": False},
                             {"path": "package-lock.json", "added": 900, "deleted": 0, "binary": False}])
    assert [f["path"] for f in groups["source"]] == ["src/a.py"] and groups["runnable"] == ["tests/test_a.py"]
    assert task_type("feat(ui): add a panel", groups) == ("feature", "prefix:feat")
    assert task_type("Repair the parser", groups) == ("bug_fix", "keyword:bug_fix")
    assert task_type("Rename the module", groups) == ("refactor", "keyword:refactor")
    assert task_type("Phase 2: new panel", groups) == ("feature", "default")
    assert task_type("Anything", {**groups, "source": []}) == ("tests", "tests_only")


def test_mine_repo_gives_d14_task_lines(repo):
    tasks = list(mine_repo(repo, name="toy"))
    assert [t["type"] for t in tasks] == ["tests", "bug_fix"]  # newest first; root, docs and tidy left out
    fix = tasks[1]
    h = fix["history"]
    assert fix["source"] == "repo_history" and fix["base_commit"] == h["parent"]
    assert fix["id"] == f"toy@{h['commit'][:12]}" and fix["subtype"] == "calc"
    assert h["test_files"] == ["tests/test_calc.py"] and h["test_cmd"] == "python -m pytest -q tests/test_calc.py"
    assert h["changed_lines"] == 6 and h["test_lines"] == 4 and h["files_changed"] == 2
    assert fix["features"] == {"has_tests": "yes", "lang": "python", "size": "xs", "touches": "one"}
    assert Task.from_dict(fix).extra["history"]["commit"] == h["commit"]


def test_validate_finds_fail_to_pass(repo):
    tasks = {t["type"]: t for t in mine_repo(repo, name="toy")}
    v = validate_task(tasks["bug_fix"], python=sys.executable, timeout=120)
    assert v["status"] == "verified", v
    assert v["fail_to_pass"] == ["tests.test_calc::test_div_zero"]
    assert v["pass_to_pass"] == ["tests.test_calc::test_add"]
    tests_task = validate_task(tasks["tests"], python=sys.executable, timeout=120)
    assert tests_task["status"] == "verified" and tests_task["fail_to_pass"] == []
    assert "tests.test_calc::test_div" in tests_task["pass_to_pass"]
    assert _git(repo, "status", "--porcelain") == ""  # the source repo is untouched


def test_select_is_balanced_and_deterministic(repo):
    tasks = list(mine_repo(repo, name="toy"))
    picked = select_tasks(tasks, per_repo={"toy": 2})
    assert len(picked) == 2 and len({t["type"] for t in picked}) == 2
    assert select_tasks(list(reversed(tasks)), per_repo={"toy": 2}) == picked


def test_select_leaves_out_tasks_that_cannot_fail(repo):
    tasks = {t["type"]: t for t in mine_repo(repo, name="toy")}
    tasks["bug_fix"]["history"]["validation"] = {"status": "verified", "fail_to_pass": ["t::a"], "pass_to_pass": []}
    tasks["tests"]["history"]["validation"] = {"status": "verified", "fail_to_pass": [], "pass_to_pass": ["t::a"]}
    dropped: list[dict] = []
    picked = select_tasks(list(tasks.values()), per_repo={"toy": 2}, require_verified=["toy"], dropped=dropped)
    assert [t["type"] for t in picked] == ["bug_fix"]  # The parent already passes the test-only commit
    assert [t["id"] for t in dropped] == [tasks["tests"]["id"]]
    assert len(select_tasks(list(tasks.values()), per_repo={"toy": 2})) == 2  # an unvalidated pick keeps both


def test_select_command_refuses_mistakes_and_writes_nothing(repo, tmp_path, capsys):
    tasks = {t["type"]: t for t in mine_repo(repo, name="toy")}
    tasks["bug_fix"]["history"]["validation"] = {"status": "verified", "fail_to_pass": ["t::a"], "pass_to_pass": []}
    cands = tmp_path / "toy.validated.jsonl"
    cands.write_text("".join(json.dumps(t) + "\n" for t in tasks.values()))
    out = tmp_path / "tasks.jsonl"
    for args in (["--quota", "tyo=2"],                           # a misspelt repo would pick nothing from it
                 ["--quota", "toy=2", "--verified", "tyo"],      # ... or skip its filter
                 ["--quota", "toy"], ["--quota", "toy=0"], [],   # no count, or no quota at all
                 ["--verified", "toy"]):
        with pytest.raises(SystemExit) as exc:
            main(["select", str(cands), "--out", str(out), *args])
        assert exc.value.code == 2 and not out.exists(), args
    with pytest.raises(SystemExit) as exc:
        main(["select", str(tmp_path / "missing.jsonl"), "--quota", "toy=2", "--out", str(out)])
    assert exc.value.code == 2 and "no such file" in capsys.readouterr().err
    tasks["bug_fix"]["history"]["validation"]["fail_to_pass"] = []
    cands.write_text("".join(json.dumps(t) + "\n" for t in tasks.values()))
    assert main(["select", str(cands), "--quota", "toy=2", "--verified", "toy", "--out", str(out)]) == 1
    assert not out.exists() and "not written" in capsys.readouterr().err
    assert main(["select", str(cands), "--quota", "toy=2", "--out", str(out)]) == 0
    assert f"wrote 2 tasks to {out}" in capsys.readouterr().out and len(out.read_text().splitlines()) == 2


# ---------------------------------------------------------------- the shipped task set
TASKS = Path(__file__).resolve().parents[2] / "design" / "0.1" / "data" / "tasks.jsonl"


def test_em_dashes_leave_titles_and_are_escaped_in_test_ids(tmp_path):
    from loopmath.priors.history import _write_jsonl, plain_title

    em = chr(0x2014)
    assert plain_title(f"Fix the parser {em} again") == "Fix the parser, again"
    row = {"title": plain_title(f"a{em}b"), "history": {"validation": {"fail_to_pass": [f"t.test.ts::x {em} y"]}}}
    out = tmp_path / "t.jsonl"
    _write_jsonl(out, [row])
    assert em not in out.read_text(encoding="utf-8")
    assert json.loads(out.read_text(encoding="utf-8")) == row  # the test id is unchanged


def test_shipped_task_set():
    if not TASKS.is_file():
        pytest.skip("no task set in this checkout")
    text = TASKS.read_text(encoding="utf-8")
    assert chr(0x2014) not in text  # house rule; test ids hold it only as a JSON escape
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(rows) >= 60
    assert len({r["type"] for r in rows}) >= 4
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        task = Task.from_dict(r)
        assert task.type in TASK_TYPES and task.source == "repo_history"
        h = r["history"]
        assert task.base_commit == h["parent"] and len(h["commit"]) == 40
        assert h["test_files"] and h["test_cmd"] and h["changed_lines"] > 0 and h["files_changed"] > 0
        assert r["features"]["size"] in ("xs", "s", "m", "l", "xl")
        assert chr(0x2014) not in r["title"]
        assert h["validation"]["status"] == "verified" and h["validation"]["fail_to_pass"]
