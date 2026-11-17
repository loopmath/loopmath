"""`eval fit`: runs `model.py`, then the engine, then writes the fit file with `fitfile.assemble` and `fitfile.write_fit`. Not built yet.

Without the [bayes] extra it prints the install line and writes nothing.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
