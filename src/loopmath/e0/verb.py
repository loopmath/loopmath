"""`loopmath research analyze-e0` arguments, importable without the walkdown.

The walkdown in cli_e0 draws figures, so importing it pulls in pandas and
matplotlib: half a second on every command, and a font-cache build of ten
seconds or more the first time matplotlib runs on a machine. Registering the
verb from here means only `analyze-e0` itself pays for that. It is registered
with or without matplotlib, so `--help` and the README agree; without the `e0`
extra, running it says which extra to install.
"""

from __future__ import annotations

import importlib.util
import sys

E0_HINT = "analyze-e0 draws figures and needs the e0 extra: pip install 'loopmath[e0]'"
HELP = "walkdown of a session corpus: tokens per accepted run for each workflow configuration (needs the e0 extra)"


def register(sub, *, listed: bool = True) -> None:
    """Attach the corpus walkdown verb to `sub`; `listed=False` leaves it out of that help list."""
    p = sub.add_parser("analyze-e0", **({"help": HELP} if listed else {}), description=HELP)
    p.add_argument("--corpus", default=None,
                   help="corpus directory, read only (default: LOOPMATH_E0_CORPUS, then research.e0_corpus in config)")
    p.add_argument("--out", required=True, help="output directory for report.md and figures")
    p.add_argument("--min-n", type=int, default=20, help="minimum sessions per eligible workflow configuration")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap draws (design: 2000)")
    p.add_argument("--seed", type=int, default=20260830, help="bootstrap seed (recorded)")
    p.set_defaults(func=_analyze)


def _analyze(args) -> int:
    if importlib.util.find_spec("matplotlib") is None:
        print(f"error: {E0_HINT}", file=sys.stderr)
        return 1
    from .cli_e0 import analyze

    return analyze(args)
