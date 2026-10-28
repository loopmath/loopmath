# DSPy: the gate at GEPA's and MIPROv2's accept step (experimental)

`loopmath.integrations.dspy` is experimental in 0.4.0rc1: it works and has tests, written from DSPy's and GEPA's
source, but it has not yet run with either installed, and its output may change.

`loopmath.integrations.dspy` asks loopmath's gate whether a DSPy optimizer's new prompt is really better
than the one it would replace. It answers accept, reject, or "score N more instances with both first".
It never imports dspy or gepa: it reads their objects by field name, and you install and run them as
usual. The fields and lines below are from DSPy 3.4.1 (commit
[`099982b3`](https://github.com/stanfordnlp/dspy/tree/099982b3c3eec9fc4ec4796d49915af42367a253)) and
GEPA 0.1.4 (commit [`8b0ce6cd`](https://github.com/gepa-ai/gepa/tree/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975)),
the version DSPy 3.4.1 pins.

```python
from loopmath.integrations import dspy as gate_dspy

v = gate_dspy.decide(parent_scores, child_scores)        # one look: v.action is accept, reject or more
v = gate_dspy.run(parent_scores, child_scores, more)     # look, call more(n), repeat until accept or reject
base_rows, cand_rows = gate_dspy.rows(parent_scores, child_scores)   # loopmath.harness-runs/0.1 rows
```

Instance `i` of the parent and instance `i` of the child are one pair. Scores that are all 0, 1, True or
False are binary (the gain in the chance of success, floor 0.05); any others are graded (the gain in the
metric, floor 0). Pass `kind="graded"` when a graded metric can happen to give only 0 and 1 on a small
batch. A `None` score is unscored, never 0. DSPy itself writes `failure_score` (0.0 by default) for an
instance that raised
([`evaluate.py:181`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/evaluate/evaluate.py#L181)),
and the gate cannot tell that 0 from a real one. Costs per instance, when you have them
(`parent_usd=`, `child_usd=`), feed the gate's budget, its cost ratio cap and the value below; the
objective is always the gain in the metric.

The next pairs' value stops nothing by default, as in `loopmath check-change`. To let it stop the gate,
pass `value_stop=True` with `horizon_runs=`, the number of future runs an accepted prompt will serve (the
optimizer's later calls and the program's deployment); without `horizon_runs` it is refused, since a
prompt's horizon has no default. With a cost on every instance, the gate then prices more pairs as
`loopmath gate` does: from 4 scored pairs on, it stops, keeping the parent (or accepting the child under
`on_unresolved="adopt"`), when the next pairs are certified not worth their cost over that horizon, or
when no look left could change the answer; `v.decision.stop` is then `value`. A success on a binary
metric is valued at what the parent pays for one (its mean cost per instance over its chance of success)
unless `value_usd=` says what one is worth; a graded metric has a value only with `value_usd=`, the
dollars a run gains per unit of the metric. A value that is unresolved, or a cost that is unknown, stops
nothing. With the stop off, the decision still shows the value and its price, over `horizon_runs` or 50
future runs.

`rows` gives one row per instance and arm, with ids `parent-<i>` and `child-<i>`. The ids repeat from one
comparison to the next, so when you keep the rows of more than one comparison in one file or one store,
name each with `experiment=` (the ids become `<experiment>:parent-<i>`): `loopmath run import` refuses a
run it already holds with other content, and takes an identical one as already stored.

## GEPA

GEPA's accept step is a pluggable `AcceptanceCriterion`: `should_accept(proposal, state) -> bool`, with an
optional `reject_reason`
([`acceptance.py:10-41`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/strategies/acceptance.py#L10-L41)).
The default keeps a child when its sum on the reflection minibatch is above its parent's
([`acceptance.py:44-53`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/strategies/acceptance.py#L44-L53)),
on 3 instances by default
([`gepa.py:408`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/teleprompt/gepa/gepa.py#L408)).
`GEPAAcceptance` is a criterion that asks the gate instead:

```python
crit = gate_dspy.GEPAAcceptance(more=more)    # r_max=24 by default
optimizer = dspy.GEPA(metric=metric, reflection_lm=reflection_lm, use_merge=False,
                      gepa_kwargs={"acceptance_criterion": crit})
```

**`use_merge=False` is required.** GEPA judges a merged program by its own rule, `new_sum >=
max(parent_sums)`, and adds it to the pool without asking the criterion
([`engine.py:767-840`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/core/engine.py#L767-L840)),
and DSPy turns merges on by default
([`gepa.py:417`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/teleprompt/gepa/gepa.py#L417)).
With merges on, the gate's guarantee covers reflective proposals only; merged programs join the pool
ungated.

**The selection strategy must honor the criterion.** GEPA's selection strategy decides which proposals go
on; its built-in ones (`AllImprovements`, the default, `BestImprovement` and `TopKImprovements`) apply the
criterion, and a custom one may pass over it
([`engine.py:444-450`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/core/engine.py#L444-L450)).

**What the gate sees.** The proposal holds the parent's and the child's score on each minibatch instance
(`subsample_scores_before` and `subsample_scores_after`, or `eval_before.scores` and `eval_after.scores`,
[`base.py:11-43`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/proposer/base.py#L11-L43)),
its one parent (`parent_program_ids`) and the child's texts (`candidate`). The parent's texts are
`state.program_candidates[parent]`
([`state.py:157`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/core/state.py#L157)).

**The minibatch is a screen, not evidence.** GEPA mutates a parent only when its minibatch is not all
perfect
([`reflective_mutation.py:395-413`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/proposer/reflective_mutation/reflective_mutation.py#L395-L413)),
then writes the child from the parent's failures on those instances and scores the child on the same
instances (`:420-550`). Both raise the child's minibatch score with no real gain behind it. So by default
the criterion applies GEPA's own rule first (a minibatch loss or tie rejects at once, with nothing more
scored), and then decides on fresh instances only, which it asks `more` for. `use_minibatch=True` counts
the minibatch pairs as evidence too; the simulation below shows what that costs.

**`more(parent_candidate, child_candidate, n)`** scores n fresh instances with the parent's and the child's
texts and returns `(parent_scores, child_scores)`, or `(parent_scores, child_scores, parent_usd,
child_usd)`. Fresh means outside the trainset GEPA draws its reflection minibatches from, preferably
outside the valset too, and no instance twice for one proposal. In DSPy, build the programs as GEPA's DSPy
adapter does
([`gepa_utils.py:208-221`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/teleprompt/gepa/gepa_utils.py#L208-L221)):

```python
import random

rng = random.Random(0)
evaluate = dspy.Evaluate(devset=confirm_set, metric=metric, num_threads=8, display_progress=False)
orders = {}  # one shuffled order of the confirmation set per child

def build(candidate):
    program = student.deepcopy()
    for name, predictor in program.named_predictors():
        if name in candidate:
            predictor.signature = predictor.signature.with_instructions(candidate[name])
    return program

def more(parent_candidate, child_candidate, n):
    order = orders.setdefault(tuple(sorted(child_candidate.items())),
                              {"next": 0, "examples": rng.sample(confirm_set, len(confirm_set))})
    batch = order["examples"][order["next"]:order["next"] + n]
    order["next"] += len(batch)
    if not batch:
        return [], []
    parent = evaluate(build(parent_candidate), devset=batch)
    child = evaluate(build(child_candidate), devset=batch)
    return gate_dspy.scores_of(parent), gate_dspy.scores_of(child)
```

With a raw GEPA adapter, `adapter.evaluate(batch, candidate).scores` gives the same lists
([`adapter.py:16-35`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/core/adapter.py#L16-L35)).
A confirmation set at least `r_max` long avoids running out. When `more` returns nothing before the gate
has decided, the cap and the thresholds stay as they were, so the decision is unresolved and the parent
is kept (`on_unresolved="adopt"` accepts instead); that says the gate could not decide, not that the child
is no better. The first look is the gate's own, on no pairs: a bad
policy raises, and a `budget_usd` of 0 (or one `sunk_usd` has used up) rejects, before `more` is first
called. The first purchase is then `r_min` pairs (2 by default), before any price is known.

**Budget.** Each fresh pair is two metric calls, and the criterion counts them with
`state.increment_evals`
([`state.py:286`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/core/state.py#L286)),
so GEPA's `max_metric_calls` (and DSPy's `auto` budget) sees them. `count_evals=False` leaves the count
alone.

**Refused proposals.** The criterion judges GEPA's reflective proposals only: tag `"reflective_mutation"`
([`reflective_mutation.py:636`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/proposer/reflective_mutation/reflective_mutation.py#L636)),
one parent in the pool, and before and after scores that are present, non-empty, of equal length and
finite. Anything else, a merge included whatever its lengths, is refused: `should_accept` returns False,
nothing is scored or counted, and `reject_reason` starts "loopmath gate: refused" and names the shape. It
does not raise, since DSPy runs GEPA with `raise_on_exception=True`
([`gepa.py:682`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/teleprompt/gepa/gepa.py#L682)),
where a raise would end the run.

**What it does not cover.** The guarantee is per accepted proposal, on fresh instances. It says nothing
about GEPA's final pick: the program with the best valset score among many
([`engine.py:517`](https://github.com/gepa-ai/gepa/blob/8b0ce6cd99a234f6b74daf37558a2ac0ce18f975/src/gepa/core/engine.py#L517)).
That pick is a best of many on one valset and carries a selection gain; confirm it on held-out data as in
the post-compile pattern below. `crit.history` keeps one JSON record per proposal judged.

## MIPROv2

MIPROv2 has no criterion to plug in. Its minibatch trials score each candidate on a fresh random minibatch
of 35, by its mean only
([`mipro_optimizer_v2.py:121, 586-589`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/teleprompt/mipro_optimizer_v2.py#L586-L589)),
never against a parent. Every 5 trials, `_perform_full_evaluation` scores the top-averaging candidate on
the whole valset and replaces the incumbent when `full_eval_score > best_score`
([`:800-866`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/teleprompt/mipro_optimizer_v2.py#L800-L866),
the compare at `:855`). `MIPROv2Gate` wraps that method:

```python
class GatedMIPROv2(dspy.MIPROv2):
    def __init__(self, *args, gate, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate

    def _perform_full_evaluation(self, *args, **kwargs):
        return self.gate.full_evaluation(super()._perform_full_evaluation, *args, **kwargs)

optimizer = GatedMIPROv2(metric=metric, auto="light", gate=gate_dspy.MIPROv2Gate(more_programs))
```

```python
taken = {}  # instances used so far, per challenger

def more_programs(incumbent, challenger, n):
    order = taken.setdefault(id(challenger), {"next": 0, "examples": rng.sample(confirm_set, len(confirm_set))})
    batch = order["examples"][order["next"]:order["next"] + n]
    order["next"] += len(batch)
    if not batch:
        return [], []
    return (gate_dspy.scores_of(evaluate(incumbent, devset=batch)),
            gate_dspy.scores_of(evaluate(challenger, devset=batch)))
```

MIPROv2's own method runs first, unchanged, and returns `(best_score, best_program, total_eval_calls)`.
When it would replace the incumbent, `more_programs(incumbent, challenger, n)` scores n fresh instances
with both programs, outside the trainset and the valset, and the gate decides. On accept the challenger
stays; otherwise the incumbent's score and program come back. Either way `total_eval_calls` keeps the
valset evaluations already spent, plus two calls per confirmation pair. `_perform_full_evaluation` is a
private method; a DSPy release that renames or reshapes it breaks the wrapper, and the gate raises when it
cannot find `best_score` or `best_program` among the arguments.

The valset scores are not used as evidence. A DSPy callback can see them per instance
([`callback.py:232-253`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/utils/callback.py#L232-L253)),
but the challenger was chosen as the best of many on that valset, so its valset score carries a selection
gain. The confirmation data must be fresh.

This covers MIPROv2's minibatch path only (`minibatch=True`, the default). Without minibatching, MIPROv2
replaces the incumbent inside the trial loop
([`:592-595`](https://github.com/stanfordnlp/dspy/blob/099982b3c3eec9fc4ec4796d49915af42367a253/dspy/teleprompt/mipro_optimizer_v2.py#L592-L595)).
There, and for any optimizer, gate the result after `compile`:

```python
compiled = optimizer.compile(student, trainset=trainset, valset=valset)
v = gate_dspy.run([], [], lambda n: more_programs(student, compiled, n))
program = compiled if v.action == "accept" else student
```

## What the simulation shows

A zero-spend simulation, with no model calls, compares GEPA's rule with the gate at GEPA's minibatch of 3.
Each proposal is a child of one parent on a pool of 300 instances (the parent averages 0.6), with its true
gain on the pool set exactly. Scores are binary, or graded as the mean of 4 rubric items. In the fair
scenario the minibatch is a random draw and nothing favors the child on it. In the GEPA-like one, a parent
whose minibatch is all perfect is skipped, and the child is boosted by kappa (in logits) on exactly the
minibatch instances where its parent scored below perfect, standing in for a child written from those
failures; the boost does not carry to other instances. The gate ran with `r_max=24`, fixed before
any run; fresh instances come from outside the minibatch. Rates are accepts per 100 proposals over 5
seeds of 300 proposals each (1500 per cell); exact 95% intervals are within about 2.6 points of each
rate.

| scores | scenario | rule | gain -0.10 | gain 0 | gain +0.10 | gain +0.20 |
|---|---|---|---|---|---|---|
| binary | fair | GEPA's rule (sum above) | 18.6 | 28.8 | 38.9 | 53.6 |
| binary | fair | gate, `use_minibatch=True` | 0.0 | 0.0 | 0.4 | 4.7 |
| binary | fair | gate, default | 0.0 | 0.0 | 0.1 | 2.9 |
| binary | GEPA-like, kappa 2 | GEPA's rule | 49.3 | 60.5 | 71.1 | 82.0 |
| binary | GEPA-like, kappa 2 | gate, `use_minibatch=True` | 0.0 | 0.1 | 1.1 | 7.1 |
| binary | GEPA-like, kappa 2 | gate, default | 0.0 | 0.0 | 0.4 | 3.7 |
| graded | fair | GEPA's rule | 20.1 | 38.5 | 64.9 | 81.6 |
| graded | fair | gate, `use_minibatch=True` | 0.0 | 0.1 | 8.7 | 53.3 |
| graded | fair | gate, default | 0.0 | 0.0 | 6.7 | 49.6 |
| graded | GEPA-like, kappa 2 | GEPA's rule | 70.4 | 85.1 | 91.9 | 96.7 |
| graded | GEPA-like, kappa 2 | gate, `use_minibatch=True` | 0.2 | 0.9 | 17.5 | 66.7 |
| graded | GEPA-like, kappa 2 | gate, default | 0.1 | 0.1 | 9.3 | 58.1 |

- **GEPA's rule keeps many children that are no better.** At a true gain of 0 it accepts 29% (binary) and
  39% (graded) of children on a fair minibatch, and 61% and 85% when the child is tailored to the minibatch.
  It keeps children that are 0.10 worse 19% to 70% of the time. Accepting ties (`>=`) is worse: 69% and 57%
  at gain 0 on a fair minibatch.
- **The gate's false accepts stay low.** Over every cell with a true gain at or below 0 (both scenarios,
  kappa 0, 1 and 2, both score kinds), the highest false accept rate is 0.3% [0.1, 0.7] for the default and
  1.1% [0.7, 1.8] with `use_minibatch=True`, against the gate's 2.5%. Both highest rates are in a tailored
  cell with graded scores; that is why the default does not count the minibatch. For binary scores at
  `r_max=24` the gate states no bound and uses a stricter fallback threshold; the highest binary rate was
  2 in 1500.
- **Power is low at these sizes.** With 24 fresh pairs, a real +0.20 gain is accepted 3% to 4% of the time
  on binary scores, where the gain must clear the 0.05 floor, and 50% to 58% on graded ones; +0.10 is
  accepted 6% to 9% on graded scores. A rejected child costs a lost improvement, not a regression, and a
  reject does not show that the child is no better.
- **More pairs buy power.** On the fair scenario at `r_max=48`, +0.20 is accepted 13.5% (binary) and 77.6%
  (graded), and +0.10 on graded scores 15.2%; at `r_max=12`, +0.20 on graded scores drops to 18.5%. When the
  budget allows, a graded metric and `r_max=48` with a confirmation set at least that long help most. The
  default stays 24.
- **Calls.** For an assumed mix of true gains (mostly at or near 0), the gate spends about half the metric
  calls of GEPA's rule per 100 proposals (for example 1795 against 3656, binary, fair), counting a valset of
  100 per accepted child, since it accepts so few children.

A paired minibatch of 35 (MIPROv2's `minibatch_size`), as an illustration rather than a replay of MIPROv2's
unpaired trials: a sum-above rule on 35 pairs accepts 45% (binary) and 48% (graded) of children with no
gain. Counting those 35 pairs (`use_minibatch=True`, `r_max=48`) accepts +0.20 at 21% (binary) and 94%
(graded), against 5% and 59% for the default on 24 fresh pairs; false accepts stay at or below 0.3%.

The simulation judges one proposal at a time. It does not run GEPA or MIPROv2, so it says nothing about how
a gated run's final program compares with an ungated one's, and the tailoring is a stand-in for a real
reflection model.
