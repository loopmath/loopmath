"""`eval rubric check`: validate a rubric and the files that name its components. Not built yet.

With `--fit` it adds the rubric health report.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
