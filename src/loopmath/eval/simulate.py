"""`eval simulate`: a synthetic study with known truth. Not built yet.

The catalog lives in `specs/`.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
