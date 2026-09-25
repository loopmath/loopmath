"""Feature levels with few tasks, and the run horizon (spec 04 section 1).

A feature value enters the fit as `feature:<key>=<value>` only when enough distinct tasks carry it
(rule M4): a value needs `features.min_tasks` tasks (default 2), a key needs two admitted values
(one is a constant, which only restates the intercept), and a key that splits the tasks exactly
as an earlier key does is dropped as its alias (order: built-ins, then declaration order). A run
with a value that is not admitted keeps it in its document; the value just gets no node.

At prediction a value the fit has covered adds nothing: a constant key's value (the intercept
holds it) and an alias key's value that pairs with the kept key's value as in the fitted tasks.
Any other declared value the fit has no node for widens a new task's range (`state.py`).

The horizon (`task.features.horizon_s`) is two fixed terms, `horizon:timebox` and `horizon:log2h`,
with an informative prior on each head that has timeboxed rows. `log2h` is relative to the fit's
reference horizon (`horizon_coding`), so it moves nothing while every run shares that horizon, and
`timebox` enters only when one source has both timeboxed and open-ended runs; otherwise it would
stand in for that source. The fit records, per task, subtype and (type, repo), the horizon their runs
agree on, so a prediction without one inherits it.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from ..taskmodel import FeatureSet, horizon_text
from ..types import Task
from .design import HORIZON_NODES, Term, horizon_terms, task_horizon
from .priors import FactorSpec

LN2 = math.log(2.0)
# (mean, sd) per doubling of the horizon and of the timebox flag, by head kind (spec 04 section 1):
# cost and tokens scale with an elasticity of 1 +- 0.5; logit and score heads start flat.
HORIZON_PRIOR = {
    "cost": {"horizon:log2h": (LN2, 0.35), "horizon:timebox": (0.0, 1.5)},
    "logit": {"horizon:log2h": (0.0, 0.25), "horizon:timebox": (0.0, 1.0)},
    "score": {"horizon:log2h": (0.0, 0.25), "horizon:timebox": (0.0, 1.0)},
}


TIMEBOX_MIN_RUNS = 2  # `horizon:timebox` needs a source with this many timeboxed and open-ended runs


def horizon_specs(head: str, kind: str, nodes: set[str]) -> list[FactorSpec]:
    """The horizon priors of one head, as factors on the horizon terms its rows have (`horizon_nodes`)."""
    return [FactorSpec(head, [(node, None, 1.0)], mean, sd ** 2, f"{node} prior N({mean:.2f}, {sd}^2)")
            for node, (mean, sd) in HORIZON_PRIOR[kind].items() if node in nodes]


def horizon_nodes(rows: list[list[Term]]) -> set[str]:
    return {node for row in rows for node, _, _ in row if node in HORIZON_NODES}


def add_horizon(rows: list[list[Term]], runs: list[str], horizon_of: dict[str, float | None],
                coding: dict[str, Any]) -> list[list[Term]]:
    """Each row with its run's horizon terms under the fit's coding (`design.horizon_terms`)."""
    if not coding.get("reference_s"):
        return rows
    return [row + horizon_terms(horizon_of.get(run), coding) for row, run in zip(rows, runs)]


def _repo_key(task: Task) -> str:
    return f"{task.type or 'unknown'}/{task.repo or 'unknown'}"


def _subtype_key(task: Task) -> str | None:
    return f"{_repo_key(task)}/{task.subtype}" if task.subtype else None


class FeatureTally:
    """What the fit's runs say about features and horizons, one task at a time."""

    def __init__(self, features: FeatureSet):
        self.features = features
        self.values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # key -> value -> tasks
        self.repos: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # key -> value -> repos
        self.by_task: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # task -> key -> values
        self.horizons: dict[str, dict[str, set]] = {"task": defaultdict(set), "subtype": defaultdict(set),
                                                    "repo": defaultdict(set)}
        self.timeboxed = 0
        self.run_horizon: dict[str, float | None] = {}  # run id -> horizon_s, None when open-ended
        self.kinds: dict[str, Counter] = defaultdict(Counter)  # source -> {"timeboxed": n, "open": n}
        self.nodes: set[str] = set()  # set by collect_rows from admit()
        self.report: dict[str, Any] = {}

    def add(self, task: Task, feats: tuple[tuple[str, str], ...], run: str = "", source: str = "") -> None:
        """One run's task and its feature levels (`design.task_features` with the fit's set)."""
        for key, value in feats:
            self.values[key][value].add(task.id)
            self.repos[key][value].add(_repo_key(task))
            self.by_task[task.id][key].add(value)
        secs = task_horizon(task)
        self.timeboxed += secs is not None
        self.run_horizon[run] = secs
        self.kinds[source]["open" if secs is None else "timeboxed"] += 1
        self.horizons["task"][task.id].add(secs)
        sub = _subtype_key(task)
        if sub:
            self.horizons["subtype"][sub].add(secs)
        self.horizons["repo"][_repo_key(task)].add(secs)

    def admit(self) -> tuple[set[str], dict[str, Any]]:
        """The admitted feature node ids, and the report kept in `meta.json` `features`."""
        min_tasks = self.features.min_tasks
        support: dict[str, dict[str, dict[str, int]]] = {}
        admitted: dict[str, list[str]] = {}
        dropped: dict[str, str] = {}
        absorbed: dict[str, str] = {}  # constant key -> the value every counted task has
        aliases: dict[str, dict] = {}  # alias key -> {"of": kept key, "map": {value: kept value}}
        splits: list[tuple[str, frozenset]] = []
        for key in self.features.specs:  # built-ins, then declaration order
            seen = self.values.get(key)
            if not seen:
                continue
            support[key] = {v: {"tasks": len(t), "repos": len(self.repos[key][v])} for v, t in sorted(seen.items())}
            ok = sorted(v for v, t in seen.items() if len(t) >= min_tasks)
            if not ok:
                dropped[key] = "below min_tasks"
                continue
            if len(ok) == 1:
                dropped[key] = "constant"
                absorbed[key] = ok[0]
                continue
            split = frozenset(frozenset(seen[v]) for v in ok)
            alias = next((k for k, s in splits if s == split), None)
            if alias is not None:
                dropped[key] = f"alias of {alias}"
                of = {frozenset(t): v for v, t in self.values[alias].items() if v in admitted[alias]}
                aliases[key] = {"of": alias, "map": {v: of[frozenset(seen[v])] for v in ok}}
                continue
            splits.append((key, split))
            admitted[key] = ok
        nodes = {f"feature:{k}={v}" for k, vs in admitted.items() for v in vs}
        report = {"declared": self.features.to_json(), "digest": self.features.digest(), "min_tasks": min_tasks,
                  "support": support, "admitted": admitted, "dropped": dropped, "absorbed": absorbed,
                  "aliases": aliases}
        return nodes, report

    def task_values(self, admitted: dict[str, list[str]]) -> dict[str, dict[str, str]]:
        """Per task, the admitted values its runs agree on: what a prediction for it inherits."""
        out: dict[str, dict[str, str]] = {}
        for task_id, keys in self.by_task.items():
            vals = {k: next(iter(vs)) for k, vs in keys.items()
                    if len(vs) == 1 and next(iter(vs)) in admitted.get(k, ())}
            if vals:
                out[task_id] = vals
        return out

    def horizon_coding(self) -> dict[str, Any]:
        """How the fit codes the horizon terms, the one place it is decided (spec 04 section 1).

        - `reference_s`: the fit's reference horizon, the lower median of its timeboxed runs' horizons
          (an observed one); None without timeboxed runs, and then there are no horizon terms.
        - `distinct`: how many horizons the timeboxed runs have. Below 2 the slope is the prior's.
        - `timebox`: whether `horizon:timebox` enters, only when one source has at least
          TIMEBOX_MIN_RUNS timeboxed and as many open-ended runs."""
        secs = sorted(h for h in self.run_horizon.values() if h is not None)
        both = [src for src, n in sorted(self.kinds.items())
                if n["timeboxed"] >= TIMEBOX_MIN_RUNS and n["open"] >= TIMEBOX_MIN_RUNS]
        return {"reference_s": statistics.median_low(secs) if secs else None, "distinct": len(set(secs)),
                "timebox": bool(both), "timebox_sources": both}

    def horizon_report(self) -> dict[str, Any]:
        """Per level, the horizon in seconds every run agrees on, for the levels that have one, and the
        fit's coding (`horizon_coding`)."""
        out: dict[str, Any] = {"timeboxed_runs": self.timeboxed, **self.horizon_coding()}
        for level, groups in self.horizons.items():
            out[level] = {k: next(iter(v)) for k, v in sorted(groups.items()) if len(v) == 1 and next(iter(v))}
        return out


def drop_features(rows: list[list[Term]], admitted: set[str]) -> list[list[Term]]:
    """Rows without the feature terms whose value is not admitted."""
    out = []
    for row in rows:
        if any(n.startswith("feature:") and n not in admitted for n, _, _ in row):
            row = [t for t in row if not (t[0].startswith("feature:") and t[0] not in admitted)]
        out.append(row)
    return out


def features_line(report: dict[str, Any] | None) -> str | None:
    """`features: output 2 values (2+2 tasks), ...; dropped input_items (alias of output), lang (constant)`."""
    if not report:
        return None
    parts = []
    for key, values in (report.get("admitted") or {}).items():
        counts = [report["support"][key][v]["tasks"] for v in values]
        tasks = "+".join(str(n) for n in counts) if len(counts) <= 3 else str(sum(counts))
        parts.append(f"{key} {len(values)} values ({tasks} tasks)")
    dropped = [f"{k} ({why})" for k, why in (report.get("dropped") or {}).items()]
    if not parts and not dropped and not report.get("error"):
        return None
    text = "features: " + (", ".join(parts) if parts else "none used")
    if dropped:
        text += "; dropped " + ", ".join(dropped)
    if report.get("error"):
        text += f"; config [features] not used: {report['error']}"
    return text


def horizon_line(report: dict[str, Any] | None) -> str | None:
    """`horizon: 44 timeboxed runs (feature/ale-bench 2 h); reference 2 h, the slope is the prior's (one
    horizon); open-ended is priced as the reference (no source has both kinds)`."""
    hz = report or {}
    if not hz.get("timeboxed_runs"):
        return None
    repos = ", ".join(f"{k} {horizon_text(v)}" for k, v in list((hz.get("repo") or {}).items())[:3])
    text = f"horizon: {hz['timeboxed_runs']} timeboxed runs" + (f" ({repos})" if repos else "")
    if hz.get("reference_s"):
        text += f"; reference {horizon_text(hz['reference_s'])}"
        if hz.get("distinct", 0) < 2:
            text += ", the slope is the prior's (one horizon)"
        if not hz.get("timebox"):
            text += "; open-ended is priced as the reference (no source has both kinds)"
    return text
