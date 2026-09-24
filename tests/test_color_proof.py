"""The suite passes whatever color settings the caller's shell has.

`tests/conftest.py` scrubs the color switches at import. These tests check the
scrub and run the help-text tests that used to fail under FORCE_COLOR=1 in a
child pytest whose environment forces color.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HELP_TESTS = (
    "tests/test_cli_graph.py::test_graph_format_help_and_module_summary_cover_every_format",
    "tests/test_graph_dataset.py::test_dataset_module_entry_point_reaches_the_cli",
    "tests/test_structcheck.py::test_body_allowance_changes_exit_policy_and_is_documented",
)


def test_color_switches_are_scrubbed_for_tests_and_their_children():
    assert "FORCE_COLOR" not in os.environ
    assert "CLICOLOR_FORCE" not in os.environ
    assert os.environ["NO_COLOR"] == "1"
    assert os.environ["PYTHON_COLORS"] == "0"
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('FORCE_COLOR'), os.environ.get('NO_COLOR'))"],
        capture_output=True, text=True, check=True,
    )
    assert child.stdout.split() == ["None", "1"]


def test_help_tests_pass_when_the_shell_forces_color():
    env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "PYTHON_COLORS")}
    env["FORCE_COLOR"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    present = [t for t in HELP_TESTS if (ROOT / t.split("::")[0]).is_file()]  # the public tree has fewer
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *present],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert f"{len(present)} passed" in result.stdout
