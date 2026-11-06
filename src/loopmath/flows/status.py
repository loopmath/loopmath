"""`loopmath flow status`: what a flow's recorded runs show, read from the store and the plan, writing nothing.

`read` gives the report, keys `STATUS_KEYS`, from the store's runs of the flow id (`docs`):

- **One fit.** `plan.quality_and_fit` once, on this revision's runs; every section and every C call below takes
  it, so `judges.fit` runs once per status.
- **The plan.** With no plan.json, a stale one or one made for another flow id or version (`commands._usable_plan`'s
  refusals), `plan` is None and `plan_problem` its words; so is a ValueError of the calls below. Otherwise the
  plan is brought up to date with `plan.advance` and not saved; `experiments` are `check.experiment_states` on
  it, `wait` the screen readout's wait, `adopted` the adopted arm once every experiment is done.
- **Dollars.** One reader (`attempt_dollars`) for `cost` and `dollars`: an attempt's `cost.usd` is priced when a
  finite float holds it, past a float when it is a number none holds (an integer past it, either sign, or an
  infinity or NaN), else unpriced; an attempt that is not an object is skipped. A known sum is `math.fsum` of the
  priced attempts; one past a float is None. Unknown is None with a `reason`, never 0 or an infinity. A run with
  an attempt past a float, or whose own sum is, adds `cost:<run id>` to `notes`.
- **Checks.** The verdict names the converter gives a run's checks (`harness_runs.ocp._check_name` over
  `_check_rules` of the run's stored configuration), plus the acceptance rule's `requires` (`tests` when it
  names none). Each run counts once per check, by its newest verdict of that name (by parsed `observed_at`, an
  unread time oldest, full ties in document order): pass or fail by `belief.outcome`'s sets on the lowered text,
  else other; with none, missing. A configuration, the flow's or a run's, that names a workflow the catalog
  does not hold raises `UnresolvedWorkflow`, which `flow status` refuses by name.
- **Quality.** Per arm, the mean of the finite pooled q (`judges.combined_quality`) of the runs that have one.
- **Notes.** The judge read lines, per arm, from `judges.judged` under the rule's score name (else the
  objective's) and `belief.outcome.unreadable_signals`, in compare's words after the counts; then the cost notes.

Arms are in plan order (`base` first), then those the plan does not name, sorted.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..output import counted
from .folder import FlowFolder

STATUS_SCHEMA = "loopmath.flow-status/1"
STATUS_KEYS = ("schema", "flow", "plan", "plan_problem", "runs", "other_revisions", "cost", "checks", "quality",
               "judges", "experiments", "wait", "adopted", "dollars", "notes")
COST_KEYS = ("step", "model", "attempts", "priced", "usd_known", "usd", "reason")
DOLLAR_KEYS = ("runs", "priced_runs", "attempts", "priced", "usd_known", "usd", "reason")
CHECK_ROW_KEYS = ("check", "arm", "runs", "pass", "fail", "other", "missing", "rate")
QUALITY_KEYS = ("arm", "runs", "scored", "pooled_mean")
PAST_FLOAT = "past a float"  # `attempt_dollars` for a number no finite float holds
SUM_PAST = "the sum is past what a float holds"
TAILS = {  # compare's words after the run counts (`compare._judge_warnings`), by the reason's kind
    "unranged": "read without judge {what}, whose linear score has no range: {w}",
    "outside-range": "read without judge {what}, whose newest score lies outside its range: {w}",
    "nonfinite": "read without the scores of judge {what} that are not finite as a float: {w}",
    "nonnumeric": "read without the scores of judge {what} that are not numbers: {w}",
    "unreadable": "read with no signals: run.signals is not a list: {w}",
}


def read(folder: FlowFolder, docs: Sequence[dict]) -> dict[str, Any]:
    """The status report of `folder` over `docs`, the store's runs of its flow id; writes nothing."""
    from . import check as check_mod
    from . import plan as plan_mod
    from .commands import _usable_plan, _waiting

    runs = plan_mod.flow_runs(folder, docs)
    quality, fit = plan_mod.quality_and_fit(folder.flow, runs, None, None)
    plan, problem, experiments, wait, adopted = None, None, [], None, None
    try:
        _usable_plan(folder)
        advanced = plan_mod.advance(folder, docs, quality=quality, fit=fit)
        moved = dataclasses.replace(folder, plan=advanced)
        experiments = check_mod.experiment_states(moved, docs, quality=quality, fit=fit)
        wait = _waiting(folder, advanced, docs, quality, fit)
        adopted = plan_mod.adopted(advanced) if all(e.state == "done" for e in advanced.experiments) else None
        plan = advanced
    except ValueError as exc:
        problem, experiments, wait, adopted = str(exc), [], None, None
    arms = _arm_order(plan, runs)
    cost, dollars, cost_notes = _cost(folder, runs)
    return {"schema": STATUS_SCHEMA,
            "flow": {"id": folder.flow.id, "version": folder.flow.version, "digest": folder.digest},
            "plan": plan.id if plan is not None else None, "plan_problem": problem, "runs": len(runs),
            "other_revisions": sum(1 for d in docs if _run_id(d) not in runs), "cost": cost,
            "checks": check_rows(folder, runs, arms), "quality": quality_rows(folder, runs, arms, fit),
            "judges": judges_view(fit), "experiments": experiments, "wait": wait, "adopted": adopted,
            "dollars": dollars, "notes": judge_lines(folder, runs, arms) + cost_notes}


def _run_id(doc: Any) -> Any:
    run = doc.get("run") if isinstance(doc, dict) else None
    return run.get("id") if isinstance(run, dict) else None


def arm_of(doc: dict) -> str:
    """The run's arm, `run.flow.arm`; `base` when it names none."""
    arm = doc["run"]["flow"].get("arm")
    return arm if isinstance(arm, str) else "base"


def _arm_order(plan: Any, runs: Mapping[str, dict]) -> list[str]:
    """The arms of `runs`: the plan's in its order, `base` first, then any other sorted."""
    present = {arm_of(doc) for doc in runs.values()}
    planned = ["base"] + [a.id for a in (plan.arms if plan is not None else ()) if a.id != "base"]
    return [a for a in planned if a in present] + sorted(present - set(planned))


def _by_arm(runs: Mapping[str, dict], arms: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {arm: [] for arm in arms}
    for doc in runs.values():
        grouped[arm_of(doc)].append(doc)
    return grouped


# ------------------------------------------------------------------ dollars

def attempt_dollars(a: dict) -> float | str | None:
    """An attempt's `cost.usd`: its float when a finite float holds it, `PAST_FLOAT` when it is a number none holds
    (an integer past `sys.float_info.max` either way, compared before any conversion, or an infinity or NaN), None
    when it is no number (no cost, no usd, a bool or any other value)."""
    cost = a.get("cost") if isinstance(a.get("cost"), dict) else {}
    usd = cost.get("usd")
    if isinstance(usd, bool) or not isinstance(usd, (int, float)):
        return None
    if isinstance(usd, int) and not -sys.float_info.max <= usd <= sys.float_info.max:
        return PAST_FLOAT
    value = float(usd)
    return value if math.isfinite(value) else PAST_FLOAT


def _sum(values: list[float]) -> float | None:
    """`math.fsum` of finite floats; None when the sum is past what a float holds."""
    try:
        total = math.fsum(values)
    except OverflowError:
        return None
    return total if math.isfinite(total) else None


def _have(n: int) -> str:
    return "has" if n == 1 else "have"


def _attempts(docs: Sequence[dict]) -> list[tuple[str, dict]]:
    return [(doc["run"]["id"], a) for doc in docs for a in doc.get("attempts") or [] if isinstance(a, dict)]


def _model(a: dict) -> str:
    model = a.get("model") if isinstance(a.get("model"), dict) else {}
    for key in ("id", "raw"):
        if isinstance(model.get(key), str) and model[key]:
            return model[key]
    return "unknown"


class UnresolvedWorkflow(Exception):
    """A configuration, the flow's or a stored run's, names a workflow the catalog does not hold; `flow status`
    refuses by name rather than read it as one with no steps or checks."""


def _resolved(configuration: dict, whose: str, cannot: str) -> Any:
    """`configuration_from_ocp` through the catalog. The resolver's own LookupError (the workflow is not in the
    catalog) is UnresolvedWorkflow, in `flow show`'s words; any other error is raised as it is."""
    from ..workflows.ocp import catalog_resolver, configuration_from_ocp

    try:
        return configuration_from_ocp(configuration, catalog_resolver)
    except LookupError as exc:
        if type(exc) is not LookupError:  # a KeyError or IndexError is a bug, not a missing workflow
            raise
        ref = configuration.get("workflow") or {}
        raise UnresolvedWorkflow(f"{whose} workflow {ref.get('ref')} version {ref.get('version')} is not in the "
                                 f"catalog, so flow status cannot {cannot}") from None


def _piece_order(folder: FlowFolder) -> list[str]:
    try:
        return [p.id for p in _resolved(folder.flow.configuration, "the flow's", "list its steps").workflow.pieces]
    except (AttributeError, KeyError, TypeError, ValueError):
        return []


def _cost(folder: FlowFolder, runs: Mapping[str, dict]) -> tuple[list[dict], dict, list[str]]:
    """(the cost rows, the dollars, the `cost:<run id>` notes), from one reading of every attempt."""
    docs = list(runs.values())
    read = [(run_id, a, attempt_dollars(a)) for run_id, a in _attempts(docs)]
    groups: dict[tuple[str, str], list[float | str | None]] = {}
    for _, a, usd in read:
        step = a.get("node") if isinstance(a.get("node"), str) else "unknown"
        groups.setdefault((step, _model(a)), []).append(usd)
    order = {piece: n for n, piece in enumerate(_piece_order(folder))}
    rows = []
    for (step, model), values in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], len(order)),
                                                                         kv[0][0] if kv[0][0] not in order else "",
                                                                         kv[0][1])):
        priced = [v for v in values if isinstance(v, float)]
        none, past, m = values.count(None), values.count(PAST_FLOAT), len(values)
        known = _sum(priced)
        usd = known if not none and not past else None
        reasons = ([f"{none} of {counted(m, 'attempt')} {_have(none)} no dollars"] if none else []) + (
            [f"{past} of {counted(m, 'attempt')} {_have(past)} dollars past what a float holds"] if past else []) + (
            [SUM_PAST] if known is None else [])
        rows.append(dict(zip(COST_KEYS, (step, model, m, len(priced), known, usd,
                                         "; ".join(reasons) if usd is None else None))))
    notes, unpriced, beyond, empty = [], 0, 0, 0
    for doc in docs:
        own = [attempt_dollars(a) for a in doc.get("attempts") or [] if isinstance(a, dict)]
        if PAST_FLOAT in own or _sum([v for v in own if isinstance(v, float)]) is None:
            notes.append(f"cost:{doc['run']['id']}")
        unpriced += None in own
        beyond += PAST_FLOAT in own
        empty += not own
    priced = [usd for _, _, usd in read if isinstance(usd, float)]
    known = _sum(priced)
    priced_runs = len(docs) - sum(1 for doc in docs if not _all_priced(doc))
    n = len(docs)
    reasons = ([f"{unpriced} of {counted(n, 'run')} {_have(unpriced)} an attempt with no dollars"] if unpriced
               else []) + (
        [f"{beyond} of {counted(n, 'run')} {_have(beyond)} an attempt with dollars past what a float holds"]
        if beyond else []) + (
        [f"{empty} of {counted(n, 'run')} {_have(empty)} no attempts"] if empty else []) + (
        [SUM_PAST] if known is None else [])
    usd = known if priced_runs == n else None
    dollars = dict(zip(DOLLAR_KEYS, (n, priced_runs, len(read), len(priced), known, usd,
                                     "; ".join(reasons) if usd is None else None)))
    return rows, dollars, notes


def _all_priced(doc: dict) -> bool:
    """A priced run: one attempt at least, each priced."""
    own = [attempt_dollars(a) for a in doc.get("attempts") or [] if isinstance(a, dict)]
    return bool(own) and all(isinstance(v, float) for v in own)


# ------------------------------------------------------------------ checks

def _check_names(folder: FlowFolder, doc: dict) -> set[str]:
    """The verdict names of the run's checks: the converter's name for each check of its stored configuration,
    and the acceptance rule's `requires` (`tests` when the rule names none)."""
    from ..harness_runs.ocp import _check_name, _check_rules, _score_names

    rule = folder.flow.acceptance_rule if isinstance(folder.flow.acceptance_rule, dict) else {}
    requires = rule.get("requires")
    if requires is None:
        names = {"tests"}
    else:
        names = {r for r in requires if isinstance(r, str)} if isinstance(requires, list) else set()
    stored = doc["run"].get("configuration")
    if isinstance(stored, dict):
        try:
            rules = _check_rules(_resolved(stored, f"run {doc['run'].get('id')}'s", "name its checks"))
        except (AttributeError, KeyError, TypeError, ValueError):
            rules = {}
        names |= {_check_name(cid, rules, requires, _score_names(folder)) for cid in rules}
    return names


def _instant(signal: dict) -> datetime | None:
    from ..belief.outcome import parse_ts

    try:
        return parse_ts(signal.get("observed_at"))
    except (OverflowError, ValueError):  # a naive year-edge time no local time holds
        return None


def newest_verdict(signals: list[dict]) -> dict | None:
    """The newest of `signals` by parsed `observed_at`, an unread time oldest, full ties in document order (the
    last), as `check._latest` reads them."""
    if not signals:
        return None
    far_past = datetime.min.replace(tzinfo=timezone.utc)
    return max(enumerate(signals), key=lambda p: (_instant(p[1]) or far_past, p[0]))[1]


def check_rows(folder: FlowFolder, runs: Mapping[str, dict], arms: list[str]) -> list[dict]:
    """One row per check and arm (`CHECK_ROW_KEYS`), check names sorted, arms in `arms` order."""
    from ..belief.outcome import FAIL, PASS, signal_list

    grouped = _by_arm(runs, arms)
    named = {arm: set().union(*(_check_names(folder, doc) for doc in docs)) if docs else set()
             for arm, docs in grouped.items()}
    rows = []
    for check in sorted(set().union(*named.values())):
        for arm in arms:
            if check not in named[arm]:
                continue
            counts = {"pass": 0, "fail": 0, "other": 0, "missing": 0}
            for doc in grouped[arm]:
                verdict = newest_verdict([s for s in signal_list(doc["run"]) if isinstance(s, dict)
                                          and s.get("kind") == "verdict" and s.get("name") == check])
                value = verdict.get("value") if verdict is not None else None
                text = value.lower() if isinstance(value, str) else None
                counts["missing" if verdict is None else "pass" if text in PASS else "fail" if text in FAIL
                       else "other"] += 1
            decided = counts["pass"] + counts["fail"]
            rows.append(dict(zip(CHECK_ROW_KEYS, (check, arm, len(grouped[arm]), counts["pass"], counts["fail"],
                                                  counts["other"], counts["missing"],
                                                  counts["pass"] / decided if decided else None))))
    return rows


# ------------------------------------------------------------------ quality and judges

def quality_rows(folder: FlowFolder, runs: Mapping[str, dict], arms: list[str], fit: Mapping[str, Any]) -> list[dict]:
    """Per arm (`QUALITY_KEYS`): its runs, those with a finite pooled q under `fit`, and their mean (None with
    none). No fit, no pooled q; a judge's raw score never stands in."""
    from . import judges

    rows = []
    for arm, docs in _by_arm(runs, arms).items():
        qs = []
        for doc in docs if fit else ():
            q = judges.combined_quality(folder.flow, doc, fit)[0]
            if not isinstance(q, bool) and isinstance(q, (int, float)) and math.isfinite(q):
                qs.append(float(q))
        total = _sum(qs)
        rows.append(dict(zip(QUALITY_KEYS, (arm, len(docs), len(qs),
                                            total / len(qs) if qs and total is not None else None))))
    return rows


def judges_view(fit: Mapping[str, Any]) -> dict | None:
    """The fit's `FIT_KEYS` with `refused` and `anchor`, each judge's `JUDGE_KEYS`; None with no fit."""
    from .judges import FIT_KEYS, JUDGE_KEYS

    if not fit:
        return None
    view = {k: fit.get(k) for k in FIT_KEYS + ("refused", "anchor")}
    view["judges"] = [{k: j.get(k) for k in JUDGE_KEYS} for j in fit.get("judges") or [] if isinstance(j, dict)]
    return view


def score_name(folder: FlowFolder) -> str | None:
    """The score name `judges.fit` reads: the acceptance rule's score, else the objective's."""
    rule = folder.flow.acceptance_rule if isinstance(folder.flow.acceptance_rule, dict) else {}
    score = rule.get("score") if isinstance(rule.get("score"), dict) else {}
    name = score.get("name") or folder.flow.objective.score
    return name if isinstance(name, str) else None


def _runs_text(per_arm: Mapping[str, int], arms: list[str]) -> str:
    return " and ".join(f"{counted(per_arm[a], 'run')} of arm {a}" for a in arms if per_arm.get(a))


def judge_lines(folder: FlowFolder, runs: Mapping[str, dict], arms: list[str], most: int = 5) -> list[str]:
    """The judge read lines: per reason a run's scores carry (`judges.judged(...).reasons`, then
    `unreadable:signals` for a run whose `run.signals` is not a list), its runs per arm, in compare's words after
    the counts; the scores that name no judge share one line, each counted."""
    from ..belief.outcome import SIGNALS_NOT_LIST, unreadable_signals
    from . import judges

    name = score_name(folder)
    seen: dict[str, dict[str, int]] = {}
    for arm, docs in _by_arm(runs, arms).items():
        for doc in docs:
            notes = list(judges.judged(doc, name).reasons) if name is not None else []
            if unreadable_signals(doc) == SIGNALS_NOT_LIST:
                notes.append("unreadable:signals")
            for w in notes:
                per = seen.setdefault(w, {})
                per[arm] = per.get(arm, 0) + 1
    anon = sorted(w for w in seen if w.startswith("unattributed:"))
    out = []
    for w in sorted(seen):
        kind, _, what = w.partition(":")
        if kind == "unattributed":
            if w == anon[0]:
                where = " and ".join(f"{sum(seen[x].get(a, 0) for x in anon)} in arm {a} runs" for a in arms
                                     if any(seen[x].get(a) for x in anon))
                named = ", ".join(anon[:most]) + (f" and {len(anon) - most} more" if len(anon) > most else "")
                out.append(f"judge scores that name no judge are not read ({where}): {named}; record a judge's "
                           "score with loopmath flow score")
        elif kind in TAILS:
            out.append(f"{_runs_text(seen[w], arms)} {TAILS[kind].format(what=what, w=w)}")
    return out
