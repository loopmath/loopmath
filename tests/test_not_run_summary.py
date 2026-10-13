"""The suite's last line names the tests it did not run.

`-m "not packaging"` in pyproject.toml deselects the wheel smoke test on every
ordinary run, and pytest's own last line only counts it. The conftest hook adds
one further line with the node ids, so the last thing the suite prints is what
it did not run. This checks both the formatting and the real final line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import not_run_line

ROOT = Path(__file__).resolve().parent.parent
DESELECTED_BY_MARKER = (
    "tests/test_packaging_smoke.py::test_built_wheel_installs_and_runs_from_a_fresh_venv"
)


class _Report:
    """The nodeid-carrying part of a pytest report, and nothing else."""

    def __init__(self, nodeid: str, wasxfail: str | None = None) -> None:
        self.nodeid = nodeid
        if wasxfail is not None:
            self.wasxfail = wasxfail


def test_not_run_line_counts_and_names_both_groups():
    stats = {
        "deselected": [_Report("tests/test_a.py::test_one"), _Report("tests/test_a.py::test_two")],
        "skipped": [_Report("tests/test_b.py::test_three")],
        "passed": [_Report("tests/test_c.py::test_four")],
    }

    assert not_run_line(stats) == (
        "2 deselected: tests/test_a.py::test_one, tests/test_a.py::test_two; "
        "1 skipped: tests/test_b.py::test_three"
    )


def test_not_run_line_is_empty_when_everything_ran_and_skips_xfail():
    assert not_run_line({"passed": [_Report("tests/test_a.py::test_one")]}) == ""
    assert not_run_line({"skipped": [_Report("tests/test_a.py::test_one", wasxfail="")]}) == ""


def test_final_line_names_the_test_the_marker_deselects():
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--color=no", "-p", "no:cacheprovider",
         DESELECTED_BY_MARKER.split("::")[0]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "300", "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert run.returncode == 5, run.stdout  # everything collected was deselected
    assert run.stdout.strip().splitlines()[-1] == f"1 deselected: {DESELECTED_BY_MARKER}"
