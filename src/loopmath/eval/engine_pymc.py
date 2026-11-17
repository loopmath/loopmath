"""The sampler: the only module of `loopmath.eval` that may import pymc. Not built yet.

Not a command: `fit.py` calls it. Import pymc inside functions or at the top of this module only.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
