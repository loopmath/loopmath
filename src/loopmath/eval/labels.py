"""`eval labels next`: what a human reviewer should score or compare next. Not built yet.

Writes a labels plan (`contracts.validate_labels`) and an optional blank form.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
