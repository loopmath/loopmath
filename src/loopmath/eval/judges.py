"""`eval judges`: which judge, and how many repeats, from a fit. Not built yet.

Reads only the fit file.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
