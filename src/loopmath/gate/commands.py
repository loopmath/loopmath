"""`loopmath gate`: adopt, reject or replicate N more, for a base and a candidate harness."""

from __future__ import annotations

import argparse
import functools
import os
import re

from ..output import EXIT_NOT_IMPLEMENTED, EXIT_OK, emit_json, fmt_usd, home
from ..store.commands import UserError, _error, handler
from ..store.home import NotFound
from . import core, inputs

FLAGS = (("r_min", "--r-min"), ("r_max", "--r-max"), ("budget_usd", "--budget-usd"),
         ("max_cost_ratio", "--max-cost-ratio"))


def _flags(message: str) -> str:
    """The policy's field names, as the flags that set them."""
    for name, flag in FLAGS:
        message = re.sub(rf"\b{name}\b", flag, message)
    return re.sub(r"^floor\b", "--floor", message)


def _headline(d: core.Decision) -> str:
    if d.action == "replicate":
        return f"gate: replicate {d.n_more} more {'pair' if d.n_more == 1 else 'pairs'}"
    if d.action == "adopt":
        return "gate: adopt the candidate"
    return "gate: reject the candidate, keep the base"


def _band_text(what: str, iv) -> str:
    mean = core._mean(iv)
    if mean is None:  # no mean, no bands: compare states a mean from two scored pairs on
        return f"{what} n/a"
    lo80, hi80 = core._band(iv, "80")
    lo95, hi95 = core._band(iv, "95")
    return (
        f"{what} {core._num(mean)} (80% {core._num(lo80)} to {core._num(hi80)}, "
        f"95% {core._num(lo95)} to {core._num(hi95)})"
    )


def _usd(x) -> str:
    return "unknown" if x is None else fmt_usd(x)


def _text(d: core.Decision) -> str:
    c = d.comparison
    if c:
        measure = d.measure or "gain"
        imp = c.get("improvement") or c.get("gain")
        p = c.get("p_better")
        p_txt = "n/a" if p is None else f"{p:.2f}"
        n = c.get("n_pairs", 0)
        line1 = _band_text(measure, imp)
        if measure == "cut in cost per accepted result":  # the acceptance gain beside it, which the floor is not on
            line1 += "; " + _band_text("gain in the chance of acceptance", c.get("gain"))
        line1 += f"; P(better) {p_txt}, {n} {'pair' if n == 1 else 'pairs'}"
        floor = c.get("floor")
        floor = floor if isinstance(floor, (int, float)) else (d.policy.get("floor") or 0.0)
        line1 += f"; floor {floor:g} on the {measure} ({d.unit})"
    else:
        line1 = "no comparison yet: " + "; ".join(r for r in d.reasons if "no scored run" in r)
    k = d.cost
    spent = k.get("spent_usd")
    spent_txt = (_usd(spent) if spent is not None or not k.get("recorded_usd")
                 else f"unknown ({fmt_usd(k['recorded_usd'])} recorded)")
    line2 = (
        f"cost per run: base {_usd(k['base_usd_per_run'])}, candidate {_usd(k['cand_usd_per_run'])}; "
        f"spent {spent_txt}; next {_usd(k['next_usd'])}"
    )
    if k.get("unrecorded"):
        line2 += f"; {k['unrecorded']} {'run' if k['unrecorded'] == 1 else 'runs'} with no recorded cost"
    lines = [_headline(d), line1, line2]
    lines += [f"  {r}" for r in d.reasons]
    return "\n".join(lines) + "\n"


def _no_pair(policy: core.GatePolicy, base: list, cand: list, d: core.Decision, compare) -> str:
    """Why no pair is scored: compare's own reason when it has one (it names the rules that score the runs),
    else what each arm has."""
    from ..compare import rule_label

    c = d.comparison
    if not c:
        kw = {"objective": "gain"} if inputs._score_name(policy.rule) is None else {}
        c = core._comparison_dict(compare(base, cand, rule=policy.rule, max_pairs=policy.r_max, lookahead=False,
                                          **kw))
    said = [r for r in c.get("reasons") or [] if isinstance(r, str) and r.startswith("no pair is scored")]
    if said:
        return said[0]
    arms = [f"{d.inputs[k]['scored']} of {d.inputs[k]['n']} {arm} runs are scored"
            for k, arm in (("base", "base"), ("cand", "candidate"))]
    why = ("no scored base run shares a task or pair key with a scored candidate run" if all(
        d.inputs[k]["scored"] for k in ("base", "cand")) else "an arm with no scored run makes no pair")
    return f"no pair is scored under the rule {rule_label(policy.rule)}: {' and '.join(arms)}, and {why}"


@handler("gate")
def gate(args: argparse.Namespace) -> int:
    for flag, sel in (("--base", args.base), ("--cand", args.cand)):
        try:
            inputs.parse_selector(sel)
        except ValueError as exc:
            raise UserError(f"{flag}: {exc}") from None
    try:
        policy = core.GatePolicy(
            rule=args.rule,
            floor=args.floor,
            r_min=args.r_min,
            r_max=args.r_max,
            budget_usd=args.budget_usd,
            max_cost_ratio=args.max_cost_ratio,
        )
    except ValueError as exc:
        raise UserError(_flags(str(exc))) from None
    from ..store import Store

    store = Store(home(args.home))
    config = store.config()  # the rules it defines, as check-change reads them; also under --runs
    if args.runs:
        where = args.runs
        if not os.path.exists(args.runs):
            raise NotFound(f"--runs {args.runs}: no such file")
        try:
            runs = inputs.read_rows(args.runs)
        except OSError as exc:
            raise UserError(f"--runs {args.runs}: cannot read: {exc.strerror or exc}") from None
    else:
        where = str(store.home)
        runs = list(store.finished_docs())
    if not runs:
        raise UserError(f"no run in {where}" if args.runs else f"no finished run in {where}")
    if args.task:
        tasks = sorted({t for t in map(inputs.task_of, runs) if t})
        runs = [x for x in runs if inputs.task_of(x) == args.task]
        if not runs:
            shown = ", ".join(tasks[:8]) + (f" and {len(tasks) - 8} more" if len(tasks) > 8 else "")
            raise UserError(f"--task {args.task}: no run in {where} has that task; the tasks there: {shown or 'none'}")
    picked = []
    for flag, sel in (("--base", args.base), ("--cand", args.cand)):
        try:
            picked.append(inputs.select(runs, sel))
        except ValueError as exc:  # a digest prefix of more than one digest
            raise UserError(f"{flag}: {exc}") from None
    base, cand = picked
    empty = [f"{flag} {sel}" for flag, sel, got in (("--base", args.base, base), ("--cand", args.cand, cand))
             if not got]
    if empty:
        task = f" of task {args.task}" if args.task else ""
        raise UserError(f"{' and '.join(empty)} {'match' if len(empty) > 1 else 'matches'} no run{task} in {where}; "
                        f"the runs there have {inputs.present(runs)}")
    try:
        compare = core.default_compare()
    except ImportError:
        return _error(args, "gate", "loopmath gate needs loopmath.compare, which this install does not have",
                      EXIT_NOT_IMPLEMENTED)
    # one config for every reading of the rule: check_rule, each run's observation, compare and the no-pair reason
    compare = functools.partial(compare, config=config)
    observe = core.default_observe()
    observe = observe and functools.partial(observe, config=config)
    from ..compare import check_rule, rule_label

    try:
        check_rule(policy.rule, base, cand, config)  # check-change's check: a rule these runs cannot be read under
    except ValueError as exc:
        raise UserError(f"no pair is scored under the rule {rule_label(policy.rule)}: {exc}") from None
    try:
        d = core.decide(base, cand, policy, compare=compare, observe=observe)
    except (TypeError, ValueError) as exc:
        raise UserError(_flags(str(exc))) from None
    if not (d.comparison or {}).get("n_pairs"):  # no evidence is an input error, never a replicate
        raise UserError(_no_pair(policy, base, cand, d, compare))
    if args.json:
        emit_json(core.SCHEMA, d.to_json())
    else:
        print(_text(d), end="")
    return EXIT_OK
