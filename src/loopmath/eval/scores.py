"""`eval scores import | check | add | export`. Not built yet.

`args.eval_command` is `scores import`, `scores check`, `scores add` or `scores export`.
"""

from __future__ import annotations

import argparse

from .commands import not_built


def run(args: argparse.Namespace) -> int:
    return not_built(args)
