"""`loopmath analyze-e0` arguments, importable without the walkdown.

The walkdown in cli_e0 draws figures, so importing it pulls in pandas and
matplotlib: half a second on every command, and a font-cache build of ten
seconds or more the first time matplotlib runs on a machine. Registering the
verb from here means only `analyze-e0` itself pays for that.
"""

from __future__ import annotations


def register(sub) -> None:
    """Attach the E0 corpus verb to the top-level parser."""
    p = sub.add_parser(
        "analyze-e0",
        help="E0 corpus walkdown: tokens-per-accepted per workflow configuration",
    )
    p.add_argument("--corpus", default=None,
                   help="corpus directory, read only (default: LOOPMATH_E0_CORPUS, then research.e0_corpus in config)")
    p.add_argument("--out", required=True, help="output directory for report.md and figures")
    p.add_argument("--min-n", type=int, default=20, help="minimum sessions per eligible workflow configuration")
    p.add_argument("--boot", type=int, default=2000, help="bootstrap draws (design: 2000)")
    p.add_argument("--seed", type=int, default=20260830, help="bootstrap seed (recorded)")
    p.set_defaults(func=_analyze)


def _analyze(args) -> int:
    from .cli_e0 import analyze

    return analyze(args)
