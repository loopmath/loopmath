"""`loopmath gate winner`: the handler and its text output (the parser is `cli_registry._add_gate_winner`).

The command reads a file of candidate scores or harness-runs rows, never the store.
"""

from __future__ import annotations

import argparse
import math
import os


def _spread(text: str | None):
    from ..store.commands import UserError

    if text is None:
        return None
    try:
        parts = [float(x) for x in text.split(":")]
    except ValueError:
        raise UserError(f"--spread {text}: give a number, or LO:HI") from None
    if len(parts) > 2 or not all(math.isfinite(x) for x in parts):
        raise UserError(f"--spread {text}: give a number, or LO:HI")
    return parts[0] if len(parts) == 1 else tuple(parts)


def _read(flag: str, path: str):
    from ..store.commands import UserError
    from ..store.home import NotFound

    if not os.path.exists(path):
        raise NotFound(f"{flag} {path}: no such file")
    try:
        from . import inputs

        return inputs.read_rows(path)
    except OSError as exc:
        raise UserError(f"{flag} {path}: cannot read: {exc.strerror or exc}") from None


def _n(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:+.{nd}g}"


def _text(v) -> str:
    g = v.gain
    n = v.n_candidates
    pick = "the best of" if v.reasons and v.reasons[0].startswith("the lead") else "picked from"
    unscored = f" ({n['censored']} unscored)" if n["censored"] else ""
    lines = [f"gate winner: {v.winner} leads {v.parent} by {_n(v.observed_lead)}, {pick} {n['total']} candidates"
             f"{unscored}; its gain {_n(g['mean'])} (80% "
             f"{_n(g['bands']['80'][0])} to {_n(g['bands']['80'][1])}, 95% {_n(g['bands']['95'][0])} to "
             f"{_n(g['bands']['95'][1])})"]
    s = v.spread
    spread = (f"spread {s['range'][0]:g} to {s['range'][1]:g} per evaluation (assumed range)" if "range" in s else
              f"spread {s['value']:g} per evaluation ({s['source']})" if s.get("value") is not None else
              "each score's own se")
    shrink = "" if v.shrinkage is None else f"; shrinkage {v.shrinkage:.2f}"
    lines.append(f"P(beats parent) {v.p_better:.2f}{shrink}; {spread}")
    if v.p_above_floor is not None:
        lines.append(f"P(gain above the floor {v.floor:g}) {v.p_above_floor:.2f}")
    if v.gain_confirmed:
        c = v.gain_confirmed
        lines.append(f"with the {c['pairs']} confirmatory {'pair' if c['pairs'] == 1 else 'pairs'}: gain "
                     f"{_n(c['mean'])} (80% {_n(c['bands']['80'][0])} to {_n(c['bands']['80'][1])}), "
                     f"P(beats parent) {c['p_better']:.2f}")
    gate = v.gate
    if gate["action"] == "replicate":
        lines.append(f"gate: replicate {gate['n_more']} more {'pair' if gate['n_more'] == 1 else 'pairs'} "
                     "(on the confirmatory runs; the search's own scores never adopt)")
    else:
        lines.append(f"gate: {gate['action']} (on the confirmatory runs)")
    b, p = v.buy, v.projection
    buy = f"buy: {b['now']} now, at least {b['at_least']}, at most {b['at_most']}"
    if p and not p.get("withheld") and p.get("draws"):
        buy += (f"; expected {b['expected']:.1f} more, P(adopt) {p['p_adopt']:.2f} over {p['draws']} draws"
                f"{' (feasible draws only)' if p.get('conditioned_on') else ''}")
    lines.append(buy)
    for row in v.by_spread:
        lines.append(f"  at spread {row['spread']:g}: gain {_n(row['gain']['mean'])} (80% "
                     f"{_n(row['gain']['bands']['80'][0])} to {_n(row['gain']['bands']['80'][1])}), "
                     f"P(beats parent) {row['p_better']:.2f}")
    lines += [f"  {r}" for r in v.reasons]
    return "\n".join(lines) + "\n"


def gate_winner(args: argparse.Namespace) -> int:
    from ..output import EXIT_NOT_FOUND, EXIT_OK, emit_json
    from ..store.commands import UserError, handler
    from ..store.home import NotFound
    from . import inputs, winner
    from .commands import _flags

    @handler("gate-winner")
    def run(args: argparse.Namespace) -> int:
        spread = _spread(args.spread)
        try:
            policy = winner.default_policy(rule=args.rule, floor=args.floor, r_min=args.r_min, r_max=args.r_max,
                                           budget_usd=args.budget_usd)
        except ValueError as exc:
            raise UserError(_flags(str(exc))) from None
        try:
            if args.scores:
                if args.parent != "arm:base":
                    raise UserError("--parent selects the parent arm of --runs; a scores file marks its parent "
                                    "with role parent")
                if not os.path.exists(args.scores):
                    raise NotFound(f"--scores {args.scores}: no such file")
                search = winner.read_scores(args.scores)
            else:
                if isinstance(spread, tuple):
                    raise UserError("--spread: with --runs, give one number, one run's SD on one task")
                search = winner.search_from_runs(_read("--runs", args.runs), parent=args.parent, rule=args.rule,
                                                 spread=spread)
                spread = None  # the runs' standard errors carry it
            confirm = None
            if args.confirm:
                rows = _read("--confirm", args.confirm)
                picked = []
                for flag, sel in (("--confirm-base", args.confirm_base), ("--confirm-cand", args.confirm_cand)):
                    got = inputs.select(rows, sel)
                    if not got:
                        raise UserError(f"{flag} {sel} matches no run in {args.confirm}; the runs there have "
                                        f"{inputs.present(rows)}")
                    picked.append(got)
                confirm = (picked[0], picked[1])
            v = winner.assess(search, spread=spread, winner=args.winner, policy=policy, confirm=confirm,
                              prior=args.prior, draws=args.draws, seed=args.seed)
        except winner.SpreadNeeded as exc:  # a missing input, as a missing argument: exit 2
            err = UserError(f"{exc}: pass --spread")
            err.exit_code = EXIT_NOT_FOUND
            raise err from None
        except (TypeError, ValueError) as exc:
            raise UserError(_flags(str(exc))) from None
        if args.json:
            emit_json(winner.SCHEMA, v.to_json())
        else:
            print(_text(v), end="")
        return EXIT_OK

    return run(args)
