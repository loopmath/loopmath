"""Commands other than analyze-e0 never import matplotlib (a font-cache build on first use)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
PROBE = (
    "import sys\n"
    "from loopmath import cli\n"
    "code = cli.main(['task-types', '--json'])\n"
    "print('matplotlib' in sys.modules, code)\n"
)


def test_task_types_does_not_import_matplotlib():
    pytest.importorskip("matplotlib")
    env = dict(os.environ, PYTHONPATH=str(SRC))
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env,
                         check=True).stdout.splitlines()[-1]
    assert out == "False 0"


def test_analyze_e0_is_still_registered(capsys):
    pytest.importorskip("matplotlib")
    from loopmath import cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["analyze-e0", "--help"])
    assert exc.value.code == 0 and "--corpus" in capsys.readouterr().out
