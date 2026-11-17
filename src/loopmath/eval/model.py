"""The design of a fit: rows to arrays and the identification report (pure, no engine). Not built yet.

Not a command: `fit.py` calls it.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
