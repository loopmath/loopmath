"""The design-based check used by `eval value --check design`. Not built yet.

Not a command: `value.py` calls it.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
