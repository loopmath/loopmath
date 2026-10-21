"""The gate at a DSPy optimizer's accept step (experimental): is the child prompt's gain over its parent real?

GEPA scores a parent and its child on the same few training instances and keeps the child when the child's
sum is higher. MIPROv2 keeps a candidate when its full validation score is above the best so far. Both are
the plain "the mean went up" rule on noisy scores. This module turns the per-instance scores into pairs and
asks `loopmath.gate.decide`, which answers accept, reject, or "score N more instances first".

    decide(parent_scores, child_scores, ...) -> Verdict      # one look
    run(parent_scores, child_scores, more, ...) -> Verdict   # look, score more, repeat until decided
    rows(parent_scores, child_scores, ...)                    # the pairs as loopmath.harness-runs/0.1 rows
    GEPAAcceptance(more=..., ...)                             # GEPA's acceptance criterion, duck-typed
    MIPROv2Gate(more=...).full_evaluation(method, ...)        # around MIPROv2's full evaluation
    scores_of(result)                                         # per-instance scores from a DSPy or GEPA result

It imports neither dspy nor gepa; it reads their objects by field name. The fields it reads, at DSPy 3.4.1
(stanfordnlp/dspy 099982b3) and GEPA 0.1.4 (gepa-ai/gepa 8b0ce6cd), the version DSPy 3.4.1 pins:

- GEPA's `CandidateProposal` (`src/gepa/proposer/base.py:30-43`): `subsample_scores_before` and
  `subsample_scores_after`, the parent's and the child's score on each minibatch instance, else
  `eval_before.scores` and `eval_after.scores` (`SubsampleEvaluation`, `base.py:11-27`); `parent_program_ids`;
  `candidate`, the child's texts; `tag`.
- GEPA's state (`src/gepa/core/state.py`): `program_candidates` (:157), the parent's texts, and
  `increment_evals` (:286), which counts the extra instances against GEPA's metric call budget.
- A DSPy `EvaluationResult` (`dspy/evaluate/evaluate.py:48-58`): `results`, (example, prediction, score)
  triples; a GEPA `EvaluationBatch` (`src/gepa/core/adapter.py:16-35`): `scores`.

**The minibatch is a screen, not evidence, by default.** GEPA mutates a parent only when its minibatch is
not all perfect (`src/gepa/proposer/reflective_mutation/reflective_mutation.py:395-413`), and it writes the
child from the parent's failures on those same instances, then scores the child on them (`:420-550`). Both
raise the child's minibatch score with no real gain behind it. So `GEPAAcceptance` keeps GEPA's own rule as
a first screen (a minibatch loss or tie rejects at once, as GEPA would) and lets the gate decide on fresh
instances it asks `more` for. `use_minibatch=True` counts the minibatch pairs as evidence too.

**Kinds.** Scores that are all 0, 1, True or False are binary: the gate reads them as accepted or not and
uses compare's paired binary model; the measure is the gain in the chance of success, floor 0.05. Any other
scores are graded: a score named `metric`, on the `fraction` scale when every score is in [0, 1], with the
paired score model; the measure is the gain in the metric, floor 0. Pass `kind=` when a graded metric can
happen to give only 0 and 1. The objective is always the gain in the metric (the quantity GEPA's rule
compares), never the cost per accepted result; costs, when given, feed the budget, the cost ratio cap and
the next pairs' value, which stops the gate only when asked (`value_stop=True`, see `decide`).

A `None` score is unscored, never 0. DSPy itself writes `failure_score` (0.0 by default) for an instance
that raised (`dspy/teleprompt/gepa/gepa_utils.py:301-309`, `dspy/evaluate/evaluate.py:181`); this module
cannot tell that 0 from a real one.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "loopmath.integrations.dspy/1"
METRIC = "metric"  # the score's name in the rows, and the rule `score:metric`
R_MAX = 24  # pairs at most per decision: GEPA's 3, then looks at 4, 8, 16 and 24
KINDS = ("binary", "graded")
ACTIONS = {"adopt": "accept", "reject": "reject", "replicate": "more"}
ARMS = (("base", "parent"), ("candidate", "child"))
KEYWORDS = {"kind", "floor", "r_min", "r_max", "budget_usd", "sunk_usd", "max_cost_ratio", "on_unresolved",
            "value_stop", "horizon_runs", "value_usd"}  # what GEPAAcceptance and MIPROv2Gate pass to decide


@dataclass(frozen=True)
class Verdict:
    """The gate's answer, in the optimizer's words: accept the child, reject it (keep the parent), or score
    `n_more` more instances with both and ask again."""

    action: str  # accept | reject | more
    n_more: int
    kind: str  # binary | graded
    n_pairs: int  # pairs given, scored or not
    decision: Any  # the loopmath.gate Decision, whole
    notes: tuple[str, ...] = ()  # this module's own reasons: after the gate's, or first where they override it

    @property
    def reasons(self) -> tuple[str, ...]:
        gate = tuple(self.decision.reasons)
        if ACTIONS.get(self.decision.action) != self.action:  # this module overrode the gate: its reason leads
            return self.notes + gate
        return gate + self.notes

    @property
    def scored_pairs(self) -> int:
        n = (self.decision.comparison or {}).get("n_pairs")
        return int(n) if isinstance(n, int) else 0

    def to_json(self) -> dict:
        return {"schema": SCHEMA, "action": self.action, "n_more": self.n_more, "kind": self.kind,
                "n_pairs": self.n_pairs, "scored_pairs": self.scored_pairs, "reasons": list(self.reasons),
                "decision": self.decision.to_json()}


# ---------------------------------------------------------------- reading scores
def _value(x: Any, where: str) -> bool | float | None:
    """One score: None (unscored), a bool, or a finite number."""
    if x is None or isinstance(x, bool):
        return x
    if isinstance(x, (int, float)) and math.isfinite(x):
        return x
    raise ValueError(f"{where}: a score is a finite number, True, False or None, not {x!r}")


def _scores(xs: Any, name: str) -> list:
    if isinstance(xs, (str, bytes)) or not isinstance(xs, Sequence):
        raise ValueError(f"{name}: a list of per-instance scores, not {type(xs).__name__}")
    return [_value(x, f"{name}[{i}]") for i, x in enumerate(xs)]


def _usd(xs: Any, n: int, name: str) -> list | None:
    if xs is None:
        return None
    if isinstance(xs, (str, bytes)) or not isinstance(xs, Sequence):
        raise ValueError(f"{name}: a list of per-instance dollar costs, or None")
    if len(xs) != n:
        raise ValueError(f"{name}: {len(xs)} costs for {n} instances; give one cost (or None) per instance")
    out = []
    for i, x in enumerate(xs):
        if x is not None and (isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                              or x < 0):
            raise ValueError(f"{name}[{i}]: a dollar cost is a number, 0 or more, or None, not {x!r}")
        out.append(None if x is None else float(x))
    return out


def _pairs_given(parent: Any, child: Any) -> tuple[list, list]:
    p, c = _scores(parent, "parent_scores"), _scores(child, "child_scores")
    if len(p) != len(c):
        raise ValueError(f"parent_scores has {len(p)} instances and child_scores {len(c)}: instance i of "
                         "each is one pair, so give both on the same instances, in the same order")
    if not p:
        raise ValueError("parent_scores and child_scores are empty: score at least one instance with both")
    return p, c


def kind_of(parent: Sequence[Any], child: Sequence[Any]) -> str:
    """`binary` when every score given is 0, 1, True or False; `graded` otherwise."""
    seen = [x for x in list(parent) + list(child) if x is not None]
    return "binary" if all(isinstance(x, bool) or x in (0, 1) for x in seen) else "graded"


def _kind(parent: list, child: list, kind: str | None) -> str:
    found = kind_of(parent, child)
    if kind is None:
        return found
    if kind not in KINDS:
        raise ValueError(f"kind: binary, graded or None (found from the scores), not {kind!r}")
    if kind == "binary" and found != "binary":
        bad = next(x for x in parent + child if x is not None and not (isinstance(x, bool) or x in (0, 1)))
        raise ValueError(f"kind binary needs scores of 0, 1, True or False, and {bad!r} is none of them")
    return kind


def _score_of(x: Any, where: str) -> float | None:
    if x is None or isinstance(x, bool):
        return None if x is None else float(x)
    if isinstance(x, (int, float)):
        return float(_value(x, where))
    s = getattr(x, "score", None)  # a Prediction or ScoreWithFeedback (dspy/teleprompt/gepa/gepa_utils.py:76-78)
    if s is None and isinstance(x, Mapping):
        s = x.get("score")
    if s is None:
        raise ValueError(f"{where}: no score in {type(x).__name__}")
    return _score_of(s, where)


def scores_of(result: Any) -> list[float | None]:
    """Per-instance scores from a DSPy `EvaluationResult` (its `results` triples, `dspy/evaluate/evaluate.py:48-58,
    226-229`), a GEPA `EvaluationBatch` (its `scores`, `src/gepa/core/adapter.py:16-35`), or a plain list. A bool
    is 1.0 or 0.0; a score that is an object or a mapping gives its `score`. The order is the evaluation's."""
    items = getattr(result, "results", None)
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
        return [_score_of(it[2] if isinstance(it, (tuple, list)) and len(it) == 3 else it, f"results[{i}]")
                for i, it in enumerate(items)]
    scores = getattr(result, "scores", None)
    if isinstance(scores, Sequence) and not isinstance(scores, (str, bytes)):
        return [_score_of(s, f"scores[{i}]") for i, s in enumerate(scores)]
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        return [_score_of(s, f"[{i}]") for i, s in enumerate(result)]
    raise ValueError(f"scores_of: no results or scores on {type(result).__name__}")


# ---------------------------------------------------------------- the pairs as harness-runs rows
def rows(parent_scores: Sequence[Any], child_scores: Sequence[Any], *, parent_usd: Sequence[Any] | None = None,
         child_usd: Sequence[Any] | None = None, ids: Sequence[str] | None = None, kind: str | None = None,
         experiment: str | None = None) -> tuple[list[dict], list[dict]]:
    """The parent's and the child's `loopmath.harness-runs/0.1` rows, one per instance. Instance i of each is one
    pair (`pair` is its position, so a repeated instance id is still its own pair); `task_id` is `ids[i]`, else
    `i<i>`. Binary scores become verdicts (`outcome.accepted`), graded ones a score named `metric`. Row ids are
    `parent-<i>` and `child-<i>`, after `<experiment>:` when an experiment is given, so the rows of two
    experiments never share an id (or a run id). Without one the ids repeat from one comparison to the next,
    so rows of more than one comparison kept in one file or one store need a distinct `experiment` each:
    `loopmath run import` refuses a run it holds with other content and takes an identical one as already
    stored. The rows work as they are with `loopmath gate` and `loopmath adapt harness-runs`."""
    p, c = _pairs_given(parent_scores, child_scores)
    return _rows(p, c, parent_usd=parent_usd, child_usd=child_usd, ids=ids, kind=kind, experiment=experiment)


def _rows(p: list, c: list, *, parent_usd: Sequence[Any] | None, child_usd: Sequence[Any] | None,
          ids: Sequence[str] | None, kind: str | None, experiment: str | None) -> tuple[list[dict], list[dict]]:
    """`rows` on scores already read, of equal length, possibly none."""
    from loopmath import harness_runs

    k = _kind(p, c, kind)
    usd = {"base": _usd(parent_usd, len(p), "parent_usd"), "candidate": _usd(child_usd, len(c), "child_usd")}
    names = list(ids or [])
    if len(names) > len(p):
        raise ValueError(f"ids: {len(names)} ids for {len(p)} instances")
    graded = [float(x) for x in p + c if x is not None]
    scale = "fraction" if all(0.0 <= x <= 1.0 for x in graded) else "linear"
    out: dict[str, list[dict]] = {"base": [], "candidate": []}
    for arm, word in ARMS:
        values = p if arm == "base" else c
        for i, v in enumerate(values):
            task = str(names[i]) if i < len(names) and names[i] is not None else f"i{i}"
            if v is None:
                outcome = None
            elif k == "binary":
                outcome = {"accepted": bool(v), "by": METRIC}
            else:
                outcome = {"name": METRIC, "score": float(v), "scale": scale, "better": "higher"}
            rid = f"{word}-{i}" if experiment is None else f"{experiment}:{word}-{i}"
            fields: dict[str, Any] = {"id": rid, "arm": arm, "pair": str(i), "outcome": outcome}
            if usd[arm] is not None and usd[arm][i] is not None:
                fields["cost"] = {"usd": usd[arm][i]}
            if experiment is not None:
                fields["experiment"] = experiment
            out[arm].append(harness_runs.row(task, **fields))
    return out["base"], out["candidate"]


# ---------------------------------------------------------------- one look
def decide(parent_scores: Sequence[Any], child_scores: Sequence[Any], *, parent_usd: Sequence[Any] | None = None,
           child_usd: Sequence[Any] | None = None, ids: Sequence[str] | None = None, kind: str | None = None,
           floor: float | None = None, r_min: int = 2, r_max: int = R_MAX, budget_usd: float | None = None,
           sunk_usd: float | None = 0.0, max_cost_ratio: float | None = None, on_unresolved: str = "reject",
           experiment: str | None = None, value_stop: bool = False, horizon_runs: float | None = None,
           value_usd: float | None = None) -> Verdict:
    """Accept the child, reject it, or score `n_more` more instances with both first.

    The statistics and the policy are `loopmath.gate.decide`'s, on the rows `rows()` makes, with compare's
    objective set to the gain. `r_max` caps the pairs one decision looks at (scored or not); `floor` is the
    smallest gain worth accepting, in the metric's units (default 0.05 for binary scores, 0 for graded);
    `sunk_usd` is spend that is not evidence (the reflection call that wrote the child, say), which counts
    against `budget_usd`. With costs unknown the gate decides on the scores alone.

    The next pairs' value is the gate's, over `horizon_runs` future runs (compare's 50 when not given), with a
    success worth `value_usd` dollars (by default what the parent pays for one; a graded metric has a value
    only with it, in dollars a run per unit of the metric). It is shown and stops nothing unless
    `value_stop=True`; then, with a cost on every instance, it can stop the gate from 4 scored pairs on, and
    `horizon_runs` is required: the runs an accepted prompt will serve have no default."""
    p, c = _pairs_given(parent_scores, child_scores)
    return _look(p, c, parent_usd=parent_usd, child_usd=child_usd, ids=ids, kind=kind, floor=floor, r_min=r_min,
                 r_max=r_max, budget_usd=budget_usd, sunk_usd=sunk_usd, max_cost_ratio=max_cost_ratio,
                 on_unresolved=on_unresolved, experiment=experiment, value_stop=value_stop,
                 horizon_runs=horizon_runs, value_usd=value_usd)


def _need_horizon(value_stop: Any, horizon_runs: Any, who: str = "") -> None:
    """Refuse the value stop without a horizon: the future runs an accepted prompt serves have no default."""
    if value_stop is True and horizon_runs is None:
        raise ValueError(f"{who}value_stop=True needs horizon_runs=, the number of future runs an accepted prompt "
                         "will serve, over which more pairs are valued; a prompt's horizon has no default")


def _look(p: list, c: list, *, parent_usd: Sequence[Any] | None = None, child_usd: Sequence[Any] | None = None,
          ids: Sequence[str] | None = None, kind: str | None = None, floor: float | None = None, r_min: int = 2,
          r_max: int = R_MAX, budget_usd: float | None = None, sunk_usd: float | None = 0.0,
          max_cost_ratio: float | None = None, on_unresolved: str = "reject", experiment: str | None = None,
          value_stop: bool = False, horizon_runs: float | None = None, value_usd: float | None = None) -> Verdict:
    """`decide` on scores already read, of equal length. With none, this is the gate's own first look: the
    policy is checked and the budget honored before any instance is bought (a $0 budget rejects)."""
    _need_horizon(value_stop, horizon_runs)
    from loopmath.compare import observe, paired
    from loopmath.gate import GatePolicy
    from loopmath.gate import decide as gate_decide

    k = _kind(p, c, kind)
    base, cand = _rows(p, c, parent_usd=parent_usd, child_usd=child_usd, ids=ids, kind=k, experiment=experiment)
    policy = GatePolicy(rule="accept" if k == "binary" else f"score:{METRIC}", floor=floor, r_min=r_min,
                        r_max=r_max, budget_usd=budget_usd, on_unresolved=on_unresolved,
                        max_cost_ratio=max_cost_ratio, value_stop=value_stop, horizon_runs=horizon_runs,
                        value_usd=value_usd)
    d = gate_decide(base, cand, policy, compare=partial(paired, objective="gain"), observe=observe,
                    sunk_usd=sunk_usd)
    return Verdict(action=ACTIONS[d.action], n_more=d.n_more, kind=k, n_pairs=len(p), decision=d)


# ---------------------------------------------------------------- look, score more, repeat
def _more_result(got: Any, n: int) -> tuple[list, list, list | None, list | None, list | None]:
    """What a `more(n)` callable returned: (parent, child), (parent, child, parent_usd, child_usd), or a mapping
    with `parent`, `child` and optional `parent_usd`, `child_usd` and `ids`."""
    ids = None
    if isinstance(got, Mapping):
        p, c = got.get("parent"), got.get("child")
        pu, cu, ids = got.get("parent_usd"), got.get("child_usd"), got.get("ids")
    elif isinstance(got, (tuple, list)) and len(got) in (2, 4):
        p, c = got[0], got[1]
        pu, cu = (got[2], got[3]) if len(got) == 4 else (None, None)
    else:
        raise ValueError("more(n) returns (parent_scores, child_scores), (parent_scores, child_scores, parent_usd, "
                         "child_usd), or a mapping with parent and child")
    p, c = _scores(p, "more: parent_scores"), _scores(c, "more: child_scores")
    if len(p) != len(c):
        raise ValueError(f"more: {len(p)} parent and {len(c)} child scores; instance i of each is one pair")
    if len(p) > n:
        raise ValueError(f"more: {len(p)} instances returned for {n} asked; the gate's cap counts every one, so "
                         "score at most the number asked")
    return p, c, _usd(pu, len(p), "more: parent_usd"), _usd(cu, len(c), "more: child_usd"), \
        None if ids is None else [None if x is None else str(x) for x in ids]


def _join(a: list | None, b: list | None, n_a: int, n_b: int) -> list | None:
    if a is None and b is None:
        return None
    return (a if a is not None else [None] * n_a) + (b if b is not None else [None] * n_b)


def run(parent_scores: Sequence[Any], child_scores: Sequence[Any], more: Callable[[int], Any], *,
        parent_usd: Sequence[Any] | None = None, child_usd: Sequence[Any] | None = None,
        ids: Sequence[str] | None = None, **kw: Any) -> Verdict:
    """`decide`, and while it says "more", `more(n)` scores n fresh instances with both programs; then decide
    again on every pair so far. Ends at accept or reject: the gate never asks past `r_max` pairs.

    `parent_scores` and `child_scores` may be empty, so the gate sees fresh instances only. Its first look is
    then on no pairs, so the policy is checked and the budget honored before `more` is first called (a $0
    budget, or one the sunk spend has used up, rejects at once). The first purchase is then `r_min` pairs, the
    fewest the gate can accept on, bought before any pair's price is known; a budget under that price can be
    overspent by it. When `more` returns no instance before the gate has decided (the fresh pool is spent),
    the policy stays as it was, the cap and the thresholds included, so the decision is unresolved: the parent
    is kept, or the child accepted under `on_unresolved="adopt"`.

    `more(n)` returns (parent_scores, child_scores), optionally with (parent_usd, child_usd), or a mapping."""
    p, c = _scores(parent_scores, "parent_scores"), _scores(child_scores, "child_scores")
    if len(p) != len(c):
        _pairs_given(p, c)  # names the lengths
    pu, cu = _usd(parent_usd, len(p), "parent_usd"), _usd(child_usd, len(c), "child_usd")
    names = None if ids is None else [None if x is None else str(x) for x in ids]
    while True:
        v = _look(p, c, parent_usd=pu, child_usd=cu, ids=names, **kw)
        if v.action != "more":
            return v
        # on no pairs the gate asks for 1, and on 1 priced pair it stops short of r_min: buy r_min at once
        n = v.n_more if p else max(v.n_more, kw.get("r_min", 2))
        p2, c2, pu2, cu2, ids2 = _more_result(more(n), n)
        if not p2:
            note = (f"the fresh instances ran out after {len(p)} pairs, before the gate decided under its policy "
                    f"(r_max {kw.get('r_max', R_MAX)})")
            if kw.get("on_unresolved", "reject") == "adopt":
                return Verdict("accept", 0, v.kind, v.n_pairs, v.decision,
                               (note + "; accepting, as on_unresolved = adopt",))
            return Verdict("reject", 0, v.kind, v.n_pairs, v.decision, (note + "; keeping the parent",))
        pu, cu = _join(pu, pu2, len(p), len(p2)), _join(cu, cu2, len(c), len(c2))
        if ids2 is not None and len(ids2) != len(p2):
            raise ValueError(f"more: {len(ids2)} ids for {len(p2)} instances")
        if names is not None or ids2 is not None:
            names = (names or [])[:len(p)]
            names = names + [None] * (len(p) - len(names)) + (ids2 or [None] * len(p2))
        p, c = p + p2, c + c2


# ---------------------------------------------------------------- GEPA's acceptance criterion
def _list(x: Any) -> list | None:
    return list(x) if isinstance(x, Sequence) and not isinstance(x, (str, bytes)) else None


def _proposal_scores(proposal: Any) -> tuple[list | None, list | None]:
    """The parent's and the child's minibatch scores: `subsample_scores_before/after`, else
    `eval_before/eval_after.scores` (gepa `src/gepa/proposer/base.py:36-40`)."""
    out = []
    for flat, rich in (("subsample_scores_before", "eval_before"), ("subsample_scores_after", "eval_after")):
        xs = _list(getattr(proposal, flat, None))
        if xs is None:
            xs = _list(getattr(getattr(proposal, rich, None), "scores", None))
        out.append(xs)
    return out[0], out[1]


def _sum(xs: list | None) -> float:
    return float(sum(float(x) for x in (xs or []) if x is not None))


REFLECTIVE = "reflective_mutation"  # the reflective proposer's tag (reflective_mutation.py:636); a merge's is "merge"


def _refusal(proposal: Any, state: Any, before: list | None, after: list | None, need_parent: bool) -> str | None:
    """Why this proposal is not one the criterion can judge, or None. The proposal's declared kind and its
    parent count come first; only then is the per-instance evidence read."""
    tag = getattr(proposal, "tag", None)
    if tag == "merge":
        return ("a merge proposal (src/gepa/proposer/merge.py:394-401) is not supported: its scores before are the "
                "parents' sums, not per-instance pairs; run DSPy's GEPA with use_merge=False")
    if tag != REFLECTIVE:
        return (f"tag {tag!r} is not GEPA's reflective proposal ({REFLECTIVE!r}); this criterion judges reflective "
                "proposals only")
    parents = _list(getattr(proposal, "parent_program_ids", None))
    if parents is None or len(parents) != 1:
        return (f"{len(parents or [])} parent ids: a reflective proposal has exactly one "
                "(reflective_mutation.py:621)")
    if need_parent:
        pool = getattr(state, "program_candidates", None)
        i = parents[0]
        if isinstance(i, bool) or not isinstance(i, int) or not isinstance(pool, Sequence) \
                or not 0 <= i < len(pool):
            return f"parent {i!r} is not in state.program_candidates"
    if before is None or after is None:
        return "no per-instance scores before or after (subsample_scores_before/after, eval_before/after.scores)"
    if not before or not after:
        return "an empty minibatch: no pair to judge"
    if len(before) != len(after):
        return (f"{len(before)} scores before and {len(after)} after, on one minibatch: the length differs, so "
                "the scores are not per-instance pairs")
    try:
        for name, xs in (("before", before), ("after", after)):
            for j, x in enumerate(_scores(xs, f"scores {name}")):
                if x is None:
                    return f"scores {name}[{j}] is None: GEPA scores every minibatch instance"
    except ValueError as e:
        return str(e)
    return None


class GEPAAcceptance:
    """GEPA's `AcceptanceCriterion` (`src/gepa/strategies/acceptance.py:10-41`), duck-typed: pass an instance as
    `acceptance_criterion`, in DSPy through `dspy.GEPA(..., use_merge=False, gepa_kwargs={"acceptance_criterion":
    ...})` (`dspy/teleprompt/gepa/gepa.py:366, 417, 685`). GEPA asks `should_accept(proposal, state)` once per
    reflective proposal (memoized, `src/gepa/core/engine.py:55-75, 464`) and `reject_reason(proposal, state)` for
    its log.

    It judges reflective proposals only. GEPA judges a merge by its own rule and never asks the criterion
    (`src/gepa/core/engine.py:767-840`), and DSPy turns merges on by default (`gepa.py:417`), so pass
    `use_merge=False`: with merges on, merged programs join the pool ungated. The selection strategy applies the
    criterion; GEPA's own strategies honor it, and a custom one may pass over it (`engine.py:444-450`).

    - `more(parent_candidate, child_candidate, n)` scores n fresh training instances with the parent's and the
      child's texts (dicts of component name to text) and returns (parent_scores, child_scores), optionally
      with (parent_usd, child_usd). Fresh means outside the minibatch the child was written from, and no
      instance twice for one proposal.
    - `screen=True`: GEPA's own rule first; a minibatch sum that is not above the parent's rejects without
      scoring anything more.
    - `use_minibatch=False`: the minibatch pairs are a screen, not evidence (see the module docstring). With
      `True` they count, and with no `more` the gate judges the minibatch alone: a "more" answer then keeps
      the parent.
    - The extra instances are counted with `state.increment_evals` (`src/gepa/core/state.py:286`), so GEPA's
      `max_metric_calls` budget sees them; `count_evals=False` leaves the count alone.
    - Refused, never accepted and never spent on: a proposal not tagged `reflective_mutation` (a merge
      included, whatever its lengths), one without exactly one parent in the pool, and one whose before and
      after scores are missing, empty, of different lengths, or not finite numbers. `should_accept` returns
      False and the reason starts "refused". It does not raise: DSPy runs GEPA with `raise_on_exception=True`
      (`gepa.py:682`), where a raise ends the whole run (`engine.py:853-867`).

    Other keywords (`kind`, `floor`, `r_min`, `r_max`, `budget_usd`, `max_cost_ratio`, `on_unresolved`,
    `value_stop`, `horizon_runs`, `value_usd`) go to `decide`; `value_stop=True` without `horizon_runs` is
    refused here. `history` keeps one JSON record per proposal judged."""

    def __init__(self, more: Callable[[dict, dict, int], Any] | None = None, *, screen: bool = True,
                 use_minibatch: bool = False, count_evals: bool = True, **kw: Any):
        if more is None and not use_minibatch:
            raise ValueError("GEPAAcceptance needs more= to score fresh instances, or use_minibatch=True to judge "
                             "the minibatch alone")
        bad = set(kw) - KEYWORDS
        if bad:
            raise TypeError(f"GEPAAcceptance: unknown keyword {', '.join(sorted(bad))}")
        _need_horizon(kw.get("value_stop"), kw.get("horizon_runs"), "GEPAAcceptance: ")
        self.more, self.screen, self.use_minibatch, self.count_evals, self.kw = more, screen, use_minibatch, \
            count_evals, kw
        self.history: list[dict] = []
        self._judged: dict[int, tuple[Any, dict]] = {}

    def _remember(self, proposal: Any, entry: dict) -> None:
        try:
            ref: Any = weakref.ref(proposal)
        except TypeError:
            ref = None
        self._judged = {k: v for k, v in self._judged.items() if v[0] is None or v[0]() is not None}
        self._judged[id(proposal)] = (ref, entry)
        self.history.append(entry)

    @staticmethod
    def _entry(proposal: Any, state: Any, after: list | None) -> dict:
        i = getattr(state, "i", None)
        parents = _list(getattr(proposal, "parent_program_ids", None))
        return {"iteration": i + 1 if isinstance(i, int) else None, "tag": getattr(proposal, "tag", None),
                "parent_ids": parents or [], "minibatch": len(after or []), "minibatch_sum": None,
                "minibatch_pairs": 0, "fresh_pairs": 0, "metric_calls": 0, "accepted": False, "refused": False,
                "action": None, "reasons": [], "verdict": None}

    def should_accept(self, proposal: Any, state: Any) -> bool:
        before, after = _proposal_scores(proposal)
        entry = self._entry(proposal, state, after)
        why = _refusal(proposal, state, before, after, need_parent=self.more is not None)
        if why is not None:
            entry.update(refused=True, action="reject", reasons=[f"refused: {why}"])
            self._remember(proposal, entry)
            return False
        old, new = _sum(before), _sum(after)
        entry["minibatch_sum"] = [old, new]
        if self.screen and not new > old:
            entry.update(action="reject", reasons=[
                f"screen: the child's minibatch sum {new:g} is not above the parent's {old:g} on {len(after)} "
                "instances, so it is rejected without scoring more, as GEPA's own rule would"])
            self._remember(proposal, entry)
            return False
        ids = _list(getattr(proposal, "subsample_indices", None))
        p0, c0 = (before, after) if self.use_minibatch else ([], [])
        names = [f"mb:{x}" for x in ids] if self.use_minibatch and ids and len(ids) == len(after) else None
        if self.use_minibatch:
            entry["minibatch_pairs"] = len(after)
        if self.more is None:
            v = decide(p0, c0, ids=names, **self.kw)
            if v.action == "more":
                v = Verdict("reject", 0, v.kind, v.n_pairs, v.decision,
                            (f"the gate asks for {v.n_more} more instances and no more= is set; keeping the parent",))
        else:
            parent = self._parent(proposal, state)
            child = dict(getattr(proposal, "candidate", None) or {})
            calls = [0]

            def fresh(n: int) -> Any:
                got = self.more(parent, child, n)
                p2, c2, *_ = _more_result(got, n)
                spent = len(p2) + len(c2)
                calls[0] += spent
                if self.count_evals and spent and callable(getattr(state, "increment_evals", None)):
                    state.increment_evals(spent)
                return got

            v = run(p0, c0, fresh, ids=names, **self.kw)
            entry["metric_calls"] = calls[0]
        entry.update(fresh_pairs=v.n_pairs - entry["minibatch_pairs"], accepted=v.action == "accept",
                     action=v.action, reasons=list(v.reasons), verdict=v.to_json())
        self._remember(proposal, entry)
        return v.action == "accept"

    @staticmethod
    def _parent(proposal: Any, state: Any) -> dict:
        """The parent's texts; `_refusal` has checked that the one parent id is in the pool."""
        return dict(state.program_candidates[proposal.parent_program_ids[0]])

    def reject_reason(self, proposal: Any, state: Any) -> str:
        """The reason recorded when this proposal was judged (GEPA calls this after `should_accept`)."""
        ref, entry = self._judged.get(id(proposal), (None, None))
        if entry is None or (ref is not None and ref() is not proposal):
            return "loopmath gate: this proposal was not judged by this criterion"
        verdict = entry.get("verdict") or {}
        word = "refused" if entry.get("refused") else entry["action"]
        lead = (f"loopmath gate: {word} after {verdict.get('n_pairs', 0)} pairs"
                if verdict else f"loopmath gate: {word}")
        return f"{lead}: {'; '.join(entry['reasons'][:3])}"


# ---------------------------------------------------------------- MIPROv2's full evaluation
_FULL_EVAL_ARGS = ("trial_num", "adjusted_num_trials", "param_score_dict", "fully_evaled_param_combos", "evaluate",
                   "valset", "trial_logs", "total_eval_calls", "score_data", "best_score", "best_program", "study",
                   "instruction_candidates", "demo_candidates")  # mipro_optimizer_v2.py:800-816, after self


class MIPROv2Gate:
    """The gate at MIPROv2's accept step. MIPROv2 has no acceptance criterion to plug in: every few minibatch
    trials, `_perform_full_evaluation` (`dspy/teleprompt/mipro_optimizer_v2.py:800-866`) scores the
    top-averaging program on the whole valset, and it replaces the incumbent when `full_eval_score >
    best_score` (`:855-858`). It returns (best_score, best_program, total_eval_calls) (`:866`). A subclass
    wraps that method:

        class GatedMIPROv2(dspy.MIPROv2):
            def __init__(self, *args, gate, **kwargs):
                super().__init__(*args, **kwargs)
                self.gate = gate

            def _perform_full_evaluation(self, *args, **kwargs):
                return self.gate.full_evaluation(super()._perform_full_evaluation, *args, **kwargs)

    When the method would replace the incumbent, `more(incumbent_program, challenger_program, n)` scores n fresh
    instances with both (outside the trainset and the valset, no instance twice) and `run` decides. On accept
    the challenger stays; otherwise the incumbent's score and program come back. Either way the valset
    evaluations are kept in `total_eval_calls` and the confirmation calls are added to it. The valset scores
    themselves are not evidence here: the challenger was picked as the best of many on that valset, so its
    valset score carries a selection gain.

    This covers the minibatch path only (`minibatch=True`, `:635-654`). Without minibatching, MIPROv2 replaces
    the incumbent inside the trial loop (`:592-595`); there, gate the compiled program against the student on a
    held-out split after `compile`, with `run([], [], more)`.

    Keywords (`kind`, `floor`, `r_min`, `r_max`, `budget_usd`, `max_cost_ratio`, `on_unresolved`, `value_stop`,
    `horizon_runs`, `value_usd`) go to `decide`; `value_stop=True` without `horizon_runs` is refused here.
    `history` keeps one JSON record per full evaluation."""

    def __init__(self, more: Callable[[Any, Any, int], Any], **kw: Any):
        if not callable(more):
            raise ValueError("MIPROv2Gate needs more(incumbent, challenger, n) to score fresh instances")
        bad = set(kw) - KEYWORDS
        if bad:
            raise TypeError(f"MIPROv2Gate: unknown keyword {', '.join(sorted(bad))}")
        _need_horizon(kw.get("value_stop"), kw.get("horizon_runs"), "MIPROv2Gate: ")
        self.more, self.kw = more, kw
        self.history: list[dict] = []

    @staticmethod
    def _arg(name: str, args: tuple, kwargs: dict) -> Any:
        if name in kwargs:
            return kwargs[name]
        i = _FULL_EVAL_ARGS.index(name)
        if i >= len(args):
            raise ValueError(f"MIPROv2Gate: {name} is not among the arguments of _perform_full_evaluation")
        return args[i]

    def full_evaluation(self, method: Callable[..., tuple], *args: Any, **kwargs: Any) -> tuple:
        """Call MIPROv2's `_perform_full_evaluation` as `method(*args, **kwargs)`; gate a replacement."""
        old_score, old_program = self._arg("best_score", args, kwargs), self._arg("best_program", args, kwargs)
        trial = self._arg("trial_num", args, kwargs)
        new_score, new_program, calls = method(*args, **kwargs)
        entry = {"trial": trial, "incumbent_score": old_score, "challenger_score": new_score, "fresh_pairs": 0,
                 "metric_calls": 0, "replaced": new_program is not old_program, "accepted": False, "action": None,
                 "reasons": [], "verdict": None}
        if new_program is old_program:
            entry.update(action="keep", reasons=["the full evaluation kept the incumbent; nothing to gate"])
            self.history.append(entry)
            return new_score, new_program, calls
        spent = [0]

        def fresh(n: int) -> Any:
            got = self.more(old_program, new_program, n)
            p2, c2, *_ = _more_result(got, n)
            spent[0] += len(p2) + len(c2)
            return got

        v = run([], [], fresh, **self.kw)
        entry.update(fresh_pairs=v.n_pairs, metric_calls=spent[0], accepted=v.action == "accept", action=v.action,
                     reasons=list(v.reasons), verdict=v.to_json())
        self.history.append(entry)
        if v.action == "accept":
            return new_score, new_program, calls + spent[0]
        return old_score, old_program, calls + spent[0]
