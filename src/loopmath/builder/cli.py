"""The `loopmath builder` parser, added by `cli_registry.register` (spec 02, `builder`).

It takes the task and rule flags of `recommend`, from the registry's own helpers, and imports nothing
heavy: the handler is `builder.server:serve`, loaded only when the command runs.
"""

from __future__ import annotations


def add(sub) -> None:
    from ..cli_registry import FIT_HELP, _common, _lazy, _task_args

    p = sub.add_parser("builder", help="build your own workflow on a local page and see its estimates change",
                       description="Serves the workflow builder on 127.0.0.1: pick a shape, change models, efforts, "
                                   "widths and pieces, and see the estimated numbers for the task, the same ones "
                                   "recommend gives. Ctrl+C stops it.")
    _task_args(p)
    rule = p.add_mutually_exclusive_group()
    rule.add_argument("--rule", default=None, metavar="RULE", help="a named acceptance rule from config")
    rule.add_argument("--target", default=None, metavar="NAME>=X", help="score-target rule, quoted: e.g. 'heldout_perf>=2400'")
    p.add_argument("--goal", default=None, choices=["default", "p50", "p70", "p80", "p90", "p95", "p99"],
                   help="the pick to aim for, as for recommend (default: config goal)")
    p.add_argument("--usual", default=None, metavar="CFG", help="a configuration id to treat as the usual one, as for recommend")
    p.add_argument("--workflow", action="append", default=[], metavar="FILE.toml", help="also consider this workflow (repeatable)")
    p.add_argument("--models", default=None, metavar="M,M,...", help="restrict candidate models")
    p.add_argument("--fit", default=None, metavar="ID", help=FIT_HELP)
    p.add_argument("--rec", default=None, metavar="REC", help="start from a saved recommendation: its task, rule and fit "
                   "(task flags given as well override its task)")
    p.add_argument("--start", default=None, metavar="CFG", help="open this configuration first: a cfg_ id from the "
                   "recommendation, an earlier one or your runs")
    p.add_argument("--port", type=int, default=0, metavar="N", help="port on 127.0.0.1 (default 0: a free port)")
    p.add_argument("--no-open", action="store_true", help="print the URL without opening the browser")
    _common(p, json_flag=False)
    p.set_defaults(func=_lazy("loopmath.builder.server:serve"))
