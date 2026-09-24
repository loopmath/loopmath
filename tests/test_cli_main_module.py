"""`python -m loopmath` runs the same CLI as the `loopmath` command (D91)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=str(SRC))
    return subprocess.run([sys.executable, "-m", "loopmath", *args], capture_output=True, text=True, env=env,
                          stdin=subprocess.DEVNULL)


def test_python_dash_m_prints_the_version():
    from loopmath import __version__

    result = _run("--version")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"loopmath {__version__}"


def test_python_dash_m_runs_a_command_and_passes_its_exit_code():
    result = _run("task-types", "--json")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schema"] == "loopmath.task-types/1"
    assert _run("no-such-command").returncode == 2
