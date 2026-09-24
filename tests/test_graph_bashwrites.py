from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from loopmath.graph.bashwrites import reads_from_command, writes_from_command
from loopmath.graph.scan import _scan_claude_session


FIXTURE = Path(__file__).parent / "fixtures" / "graph" / "bash_commands.json"
TS = "2026-09-01T12:00:00Z"


def _complete(items: list[dict]) -> list[dict]:
    return [{"ts": TS, "tier": "heuristic", **item} for item in items]


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text(encoding="utf-8")))
def test_bash_command_case(case: dict) -> None:
    cwd = case.get("cwd")
    assert writes_from_command(case["command"], TS, cwd=cwd) == _complete(case["writes"])
    assert reads_from_command(case["command"], TS, cwd=cwd) == _complete(case["reads"])


def test_scan_claude_session_collects_bash_reads(tmp_path: Path) -> None:
    session = tmp_path / "session.jsonl"
    obj = {
        "timestamp": TS,
        "cwd": "/workspace",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "cat docs/plan.md"},
                }
            ],
        },
    }
    session.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    scan = _scan_claude_session(session)

    assert scan["reads"] == [
        {
            "ts": TS,
            "path": "/workspace/docs/plan.md",
            "tier": "heuristic",
            "how": "cat",
        }
    ]


def test_parse_failures_and_excluded_constructs_are_counted() -> None:
    exclusions: Counter = Counter()
    commands = [
        "cp only",
        "echo ok 2> errors.log",
        "echo ok > /dev/null",
        "echo >",
        "echo 'unterminated",
        "cat <<EOF\nbody",
        "echo $(cat file",
    ]

    writes = []
    for command in commands:
        writes.extend(writes_from_command(command, TS, cwd="/work", exclusions=exclusions))

    assert writes == []
    assert exclusions == Counter(
        {
            "invalid_cp_mv": 1,
            "excluded_redirect": 1,
            "excluded_path": 1,
            "invalid_redirect": 1,
            "tokenize_error": 1,
            "unparsable_heredoc": 1,
            "unparsable_substitution": 1,
        }
    )


def test_invalid_commands_never_emit_fake_paths_without_counter() -> None:
    commands = [
        "cp only",
        "mv",
        "echo ok 2> errors.log",
        "echo ok > /dev/null",
        "echo >",
        "cat <<EOF\nbody",
        "echo $(cat file",
    ]
    for command in commands:
        assert writes_from_command(command, TS, cwd="/work") == [], command


def test_scan_surfaces_real_bashwrite_exclusions(tmp_path: Path) -> None:
    session = tmp_path / "excluded.jsonl"
    obj = {
        "timestamp": TS,
        "cwd": "/workspace",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "Bash",
                    "input": {"command": "cp only && echo ok 2> errors.log"},
                }
            ],
        },
    }
    session.write_text(json.dumps(obj) + "\n", encoding="utf-8")

    scan = _scan_claude_session(session)

    assert scan["writes"] == []
    assert scan["excluded"] == {"invalid_cp_mv": 1, "excluded_redirect": 1}


def test_command_substitution_cd_is_isolated() -> None:
    command = "cd /base && echo $(cd /elsewhere && cat a) && cat b"

    assert reads_from_command(command, TS, cwd="/work") == _complete(
        [{"path": "/elsewhere/a", "how": "cat"}, {"path": "/base/b", "how": "cat"}]
    )


def test_grep_values_and_pytest_selectors_are_not_paths() -> None:
    assert reads_from_command(
        "grep --exclude skip -A 3 needle docs/spec.md", TS, cwd="/work"
    ) == _complete([{"path": "/work/docs/spec.md", "how": "grep"}])
    assert reads_from_command(
        "pytest tests/test_x.py::test_y", TS, cwd="/work"
    ) == _complete([{"path": "/work/tests/test_x.py", "how": "pytest"}])
