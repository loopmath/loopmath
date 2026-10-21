"""Runs to the history bank and the candidate's results, as `tasks pick` reads them.

A run is an OCP run document from the store or a `loopmath.harness-runs/0.1` row. Its outcome comes from
`compare.observe` under the rule, as `check-change from-runs` reads it: accepted or not, or a score from 0 to 1
(counted as that share of one accepted run). A task is `run.task.id` (`task_id` in a row). A history arm is the
run's arm label, else its configuration id and harness digest.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..compare import observe, score_rule
from .fit import Bank

SELECTOR_FORMS = ("KEY=VALUE for a run label (the --label of run record), arm=<name> or arm:<name>, a cfg_ "
                  "configuration id, or sha256:<hex> or digest:sha256:<hex> (a prefix of one digest will do)")
TASK_KEYS = ("id", "type", "repo", "subtype", "title", "base_commit", "features", "org")


@dataclass
class Run:
    """One run as `tasks pick` reads it."""

    task: str | None
    value: float | None  # accepted (1 or 0), or a score from 0 to 1; None: unscored
    name: str
    arm: str | None = None
    digest: str | None = None
    config: str | None = None
    labels: dict = field(default_factory=dict)
    spec: dict | None = None  # the task object, for a task file (store documents only)


def _value(o, rule) -> float | None:
    if o.outcome is None:
        return None
    if isinstance(o.outcome, bool):
        return float(o.outcome)
    v = float(o.outcome)
    sr = score_rule(rule)
    if (sr is not None and sr[1]) or o.better == "lower":
        raise ValueError(f"{rule}: tasks pick reads a score where higher is better, from 0 to 1")
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"run {o.run}: score {v} is outside 0 to 1; tasks pick reads acceptance, or a score from "
                         "0 to 1 (a share of tests passed, say)")
    return v


def read_runs(docs: list, rule: Any, config: Any = None) -> list[Run]:
    out = []
    for i, x in enumerate(docs):
        o = observe(x, rule, config=config)
        meta = x.get("run") if isinstance(x, dict) and isinstance(x.get("run"), dict) else None
        labels = (meta or x).get("labels") if isinstance(x, dict) else None
        cfg = (meta.get("configuration") or {}).get("id") if meta else None
        spec = None
        if meta and isinstance(meta.get("task"), dict):
            t = meta["task"]
            spec = {k: t[k] for k in TASK_KEYS if t.get(k) not in (None, "", {})}
        out.append(Run(task=o.task, value=_value(o, rule), name=str(o.run or f"run {i + 1}"), arm=o.arm,
                       digest=o.harness, config=cfg if isinstance(cfg, str) else None,
                       labels={str(k): str(v) for k, v in labels.items()} if isinstance(labels, dict) else {},
                       spec=spec))
    return out


def label_form(sel: str) -> tuple[str, str] | None:
    """(KEY, VALUE) for KEY=VALUE, arm:NAME or a bare word (arm=WORD); None for a cfg_ id or a digest."""
    sel = sel.strip()
    if sel.startswith("cfg_") or sel.startswith("sha256:") or sel.startswith("digest:"):
        return None
    if sel.startswith("arm:"):
        sel = "arm=" + sel[len("arm:"):]
    key, eq, value = sel.partition("=")
    if not eq:
        key, value = "arm", sel
    if not key or not value or any(c.isspace() for c in key):
        raise ValueError(f"selector {sel!r}: give {SELECTOR_FORMS}")
    return key, value


def members(runs: list[Run], sel: str) -> list[bool]:
    """Which runs `sel` selects. ValueError for a bad selector or a digest prefix of more than one digest."""
    form = label_form(sel)
    if form is not None:
        key, value = form
        return [r.labels.get(key) == value or (key == "arm" and r.arm == value) for r in runs]
    if sel.startswith("cfg_"):
        return [r.config == sel for r in runs]
    digest = sel[len("digest:"):] if sel.startswith("digest:") else sel
    if not digest.startswith("sha256:") or digest == "sha256:":
        raise ValueError(f"selector {sel!r}: give {SELECTOR_FORMS}")
    digests = sorted({r.digest for r in runs if r.digest})
    hits = [d for d in digests if d.startswith(digest)]
    if digest not in digests and len(hits) > 1:
        raise ValueError(f"selector {sel!r} matches {len(hits)} digests: {', '.join(hits)}; give more of one")
    whole = digest if digest in digests or not hits else hits[0]
    return [r.digest == whole for r in runs]


def record_label(sel: str) -> str | None:
    """The `--label KEY=VALUE` that makes a recorded run match `sel`, when one does."""
    try:
        form = label_form(sel)
    except ValueError:
        return None
    return f"{form[0]}={form[1]}" if form else None


def arm_key(r: Run) -> str:
    if r.arm:
        return f"arm={r.arm}"
    if r.config or r.digest:
        return "@".join(x for x in (r.config, (r.digest or "")[:19]) if x)
    return "unlabelled"


def present(runs: list[Run]) -> str:
    """The arms and their run counts, for a message."""
    seen: dict[str, int] = {}
    for r in runs:
        seen[arm_key(r)] = seen.get(arm_key(r), 0) + 1
    return ", ".join(f"{k} ({n} run{'' if n == 1 else 's'})" for k, n in sorted(seen.items())) or "no run"


@dataclass
class Split:
    bank: Bank
    results: dict[str, tuple[float, float]]  # the candidate's: per task, (accepted, runs)
    cand_runs: int
    unscored: int  # runs with no outcome under the rule, left out
    task_specs: dict[str, dict] = field(default_factory=dict)  # per task: the task object of a history run


def results_of(runs: list[Run], mask: list[bool]) -> dict[str, tuple[float, float]]:
    """Per task, (accepted, runs) of the scored runs in `mask`."""
    out: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r, m in zip(runs, mask):
        if m and r.value is not None and r.task:
            out[r.task][0] += r.value
            out[r.task][1] += 1
    return {t: (s, n) for t, (s, n) in out.items()}


def split(runs: list[Run], is_cand: list[bool], keep: list[bool] | None = None) -> Split:
    """The history bank (runs not of the candidate, and in `keep` when given) and the candidate's results."""
    hist: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    specs: dict[str, dict] = {}
    unscored = 0
    for i, r in enumerate(runs):
        if r.value is None or not r.task:
            unscored += 1
            continue
        if is_cand[i] or (keep is not None and not keep[i]):
            continue
        cell = hist[arm_key(r)][r.task]
        cell[0] += r.value
        cell[1] += 1
        if r.task not in specs and r.spec:
            specs[r.task] = r.spec
    arms = sorted(hist)
    tasks = sorted({t for a in arms for t in hist[a]})
    S = np.zeros((len(arms), len(tasks)))
    N = np.zeros_like(S)
    col = {t: j for j, t in enumerate(tasks)}
    for i, a in enumerate(arms):
        for t, (s, n) in hist[a].items():
            S[i, col[t]], N[i, col[t]] = s, n
    results = results_of(runs, is_cand)
    return Split(bank=Bank(arms=arms, tasks=tasks, successes=S, trials=N), results=results,
                 cand_runs=int(sum(n for _, n in results.values())), unscored=unscored, task_specs=specs)
