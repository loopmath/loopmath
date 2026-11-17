"""OCP signals, preferences and harness-runs rows to and from the score tables. Not built yet.

Not a command: `scores.py` calls it. The only module of `loopmath.eval` that may import `loopmath.ocp` and `loopmath.harness_runs`.
"""

from __future__ import annotations

import argparse

from ..commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
