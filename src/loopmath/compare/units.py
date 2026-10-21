"""What a comparison pairs: runs, or cases.

Runs of one case (one task in one experiment) share its task, so for a claim about new cases they are not
independent pairs. Under a score rule, `check-change from-runs` therefore compares cases when any case has
more than one run on an arm: one pair per case, each arm's mean score over its scored runs. Acceptance rows
stay paired run by run. `--unit` chooses either.
"""

from __future__ import annotations

import json
from datetime import datetime
from statistics import fmean
from typing import Any, Iterable

from ..output import counted
from . import Observation, _observations, _one_direction, _one_scale, score_rule

UNITS = ("case", "run")


def case_key(o: Observation) -> tuple[str | None, str] | None:
    """The case a run belongs to: (its experiment, its task), kept apart, so experiment `a/b` with task `c` and
    experiment `a` with task `b/c` are two cases; None when the run names no task."""
    if not o.task:
        return None
    return (o.experiment or None, o.task)


def most_runs(base: list[Observation], cand: list[Observation]) -> int:
    """The most runs one case has on one arm (0 when no run names a task)."""
    most = 0
    for runs in (base, cand):
        n: dict[tuple[str | None, str], int] = {}
        for o in runs:
            k = case_key(o)
            if k is not None:
                n[k] = n.get(k, 0) + 1
        most = max([most, *n.values()])
    return most


def choose(rule: Any, base: list[Observation], cand: list[Observation], asked: str | None = None) -> dict[str, Any]:
    """`{unit, from, why, most_runs}`: the unit `asked` names, else case under a score rule when a case has
    more than one run on an arm, else run. Raises `ValueError` when the case unit cannot be read."""
    score = score_rule(rule) is not None
    most = most_runs(base, cand)
    untasked = sum(1 for o in base + cand if case_key(o) is None)
    if asked is not None and asked not in UNITS:
        raise ValueError(f"--unit: case or run, not {asked!r}")
    if asked == "case":
        if not score:
            raise ValueError("--unit case compares each case's mean score and needs a score rule (score:NAME); "
                             "acceptance rows are compared run by run")
        if untasked:
            raise ValueError(f"--unit case: {counted(untasked, 'run')} {'names' if untasked == 1 else 'name'} no "
                             f"task, so {'its case is' if untasked == 1 else 'their cases are'} not known")
        return {"unit": "case", "from": "--unit", "why": "given", "most_runs": most}
    if asked == "run":
        return {"unit": "run", "from": "--unit", "why": "given", "most_runs": most}
    if not score:
        why = "acceptance rows are compared run by run"
    elif most <= 1:
        why = "one run per arm in each case"
    elif untasked:
        why = (f"{counted(untasked, 'run')} {'names' if untasked == 1 else 'name'} no task, so cases are not "
               f"known; runs of one case are not independent pairs for new cases")
    else:
        return {"unit": "case", "from": "auto", "most_runs": most,
                "why": (f"score rows with up to {most} runs per arm in a case: runs of one case are not independent "
                        f"pairs for new cases, so each case is one pair, the mean score per arm; --unit run "
                        f"compares runs")}
    return {"unit": "run", "from": "auto", "why": why, "most_runs": most}


def _common(values: Iterable[Any]) -> Any:
    got = {v for v in values if v is not None}
    return got.pop() if len(got) == 1 else None


def _case(key: tuple[str | None, str], arm: str, runs: list[Observation]) -> Observation:
    scored = [float(o.outcome) for o in runs if o.outcome is not None]
    usd = [o.usd for o in runs]
    tokens = [o.tokens for o in runs]
    experiment, task = key
    name = task if experiment is None else json.dumps([experiment, task], ensure_ascii=False)  # one id per case
    return Observation(
        task=runs[0].task, arm=arm, outcome=fmean(scored) if scored else None,
        usd=fmean(usd) if None not in usd else None,  # each case's mean cost per run; unknown if any run's is
        tokens=round(fmean(tokens)) if None not in tokens else None,
        harness=_common(o.harness for o in runs), run=f"case:{name}#{arm}", failed=all(o.failed for o in runs),
        # no start: compare orders a task's runs by start, then id, and ids order cases alike on both arms
        started_at=None, experiment=runs[0].experiment,
        base_commit=_common(o.base_commit for o in runs), scale=_common(o.scale for o in runs),
        better=_common(o.better for o in runs))


def by_case(base_runs: Iterable[Any], cand_runs: Iterable[Any], *, rule: Any, now: datetime | None = None,
            config: Any = None) -> tuple[list[Observation], list[Observation]]:
    """One observation per case per arm, for `paired`: the mean of the case's scored runs on that arm
    (unscored when none is scored), and their mean dollars and tokens per run (unknown when any run's is).
    Every run must name a task. Cases pair on their task, so the pairs are the cases both arms ran."""
    if score_rule(rule) is None:
        raise ValueError("cases are compared on a score rule (score:NAME); acceptance rows are compared run by run")
    base = _observations(base_runs, rule, now, config)
    cand = _observations(cand_runs, rule, now, config)
    _one_scale(base, cand)
    _one_direction(base, cand, rule)
    out: list[list[Observation]] = []
    for arm, runs in (("base", base), ("cand", cand)):
        groups: dict[tuple[str | None, str], list[Observation]] = {}
        for o in runs:
            k = case_key(o)
            if k is None:
                raise ValueError(f"run {o.run or '?'} names no task, so its case is not known")
            groups.setdefault(k, []).append(o)
        order = sorted(groups, key=lambda k: (k[0] is not None, k[0] or "", k[1]))
        out.append([_case(k, arm, groups[k]) for k in order])
    return out[0], out[1]
