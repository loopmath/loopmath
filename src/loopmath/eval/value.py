"""`eval value`: value and gate pass probability per configuration, from a fit. Not built yet.

Reads the fit file and a value file; `quality.compile_value` and `quality.new_runs` are the shared pieces.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
