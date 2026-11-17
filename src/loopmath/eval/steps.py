"""`eval steps`: which step drives the final value. Not built yet.

Reads the fit file, a value file and `runs.jsonl`.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
