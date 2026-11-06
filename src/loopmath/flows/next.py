"""The next runs a plan asks for, as specs.

Each spec names the run id, the input, the arm, the configuration id, each step's setting, the slate for
pairs, the judges to apply, whether a person scores it, and how to record it. Deterministic for a plan state and
seed. Inside the package, import this module as `from . import next as next_mod`.

`quality` maps a run id to its unpooled estimate and sd, and `fit` is the judges fit; when they are None they
are computed as `plan` says, only when `docs` holds a run of the flow with a measured judge score.

- **Order.** The experiments of the plan brought up to date (`plan.advance`), in order, skipping done ones; an
  experiment whose arms are not known yet (the subset check before the screen is read) stops the list. Within
  one, its pairs in order, up to its pair cap. A run already in `docs` is not asked for again. Runs come in
  whole pairs, so `n` 7 gives 3 pairs.
- **Pairs.** Pair i of an experiment has the key `<plan id>.<experiment id>.<i:03d>`, the run ids
  `run_<key>.a` and `run_<key>.b` and the slate `slt_<key>`. Both runs take input i, cycling through the
  holdout inputs for a check and the train inputs for a screen. The train inputs are the ids not held out, or
  every id when `inputs.holdout` is empty; a screen with every id held out is refused, never run on held-out
  inputs. Each spec also carries `slate_members`, the
  pair's two run ids, so a run asked for alone (its partner already recorded) still writes the whole slate.
- **Seed.** It decides which of a pair's two runs is listed first, and nothing else: ids, inputs, slates and
  marks never depend on it.
- **Spot checks.** For run r and judge j, u = the first 13 hex digits of SHA-256 over `<plan seed>|<r>|<j>`
  over 16^13. A judge that is not a person judge is applied when u is below its rate (1 by default); the run is
  marked for a person score when u is below a person judge's rate. The marks depend on the plan seed and the run
  id only.
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Mapping, Sequence

from ..types import FlowRef
from . import plan as plan_mod
from .folder import FlowFolder
from .records import Flow, RunSpec, judge_rate

RUN_ID = re.compile(r"^run_[A-Za-z0-9._:-]{1,196}$")
RECORD = "loopmath flow record FLOW --ocp RUN.json"


def next_runs(folder: FlowFolder, *, n: int, seed: int, docs: Sequence[dict] = (),
              quality: Mapping[str, tuple[float, float | None]] | None = None,
              fit: Mapping[str, Any] | None = None) -> list[RunSpec]:
    """Up to `n` runs, in order."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError(f"n {n!r}: runs come in pairs, so n is an integer, 2 or more")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed {seed!r}: the seed is an integer")
    plan = plan_mod.advance(folder, docs, quality=quality, fit=fit)
    flow = folder.flow
    if not flow.inputs.ids:
        raise ValueError(f"flow {flow.id!r} lists no inputs.ids, so a run's input cannot be named")
    recorded = plan_mod.recorded_runs(folder, plan, docs)
    arms = {a.id: a for a in plan.arms}
    specs: list[RunSpec] = []
    for exp in plan.experiments:
        if exp.state == "done":
            continue
        keys = plan_mod.pairs(plan, exp)
        if not keys:
            break
        split = exp.split or "train"
        pool = input_pool(flow, split)
        if not pool:
            raise ValueError(f"experiment {exp.id!r} runs on the train inputs, but flow {flow.id!r} holds out "
                             "every input in inputs.ids; leave some out of inputs.holdout")
        for i, (key, arm_a, arm_b) in enumerate(keys):
            members = plan_mod.run_ids(key)
            runs = [(run_id, arm) for run_id, arm in zip(members, (arm_a, arm_b)) if run_id not in recorded]
            if not runs:
                continue
            if len(specs) + len(runs) > n:
                return specs
            if hashlib.sha256(f"{seed}|{key}".encode()).digest()[0] & 1:
                runs.reverse()
            for run_id, arm_id in runs:
                specs.append(_spec(folder, plan, exp.id, split, arms, run_id, arm_id, pool[i % len(pool)],
                                   f"slt_{key}", members))
    return specs


def input_pool(flow: Flow, split: str) -> list[str]:
    """The inputs a split cycles through: the holdout inputs for `holdout`, else the train inputs (see Pairs)."""
    ids = list(flow.inputs.ids or ())
    holdout = list(flow.inputs.holdout or ())
    if split == "holdout":
        return holdout or ids
    return [i for i in ids if i not in set(holdout)] if holdout else ids


def _spec(folder: FlowFolder, plan: Any, experiment: str, split: str, arms: Mapping[str, Any], run_id: str,
          arm_id: str, task: str, slate: str, members: tuple[str, str]) -> RunSpec:
    for member in members:
        if not RUN_ID.fullmatch(member):
            raise ValueError(f"run id {member!r} does not match {RUN_ID.pattern}")
    if arm_id not in arms:
        raise ValueError(f"experiment {experiment!r} names arm {arm_id!r}, which is not an arm of plan {plan.id}")
    configuration = arms[arm_id].configuration
    flow = folder.flow
    applied, person = [], False
    for judge in flow.judges or ():
        hit = _draw(plan.seed, run_id, judge.id) < judge_rate(judge)
        if judge.kind == "person":
            person = person or hit
        elif hit:
            applied.append(judge.id)
    return RunSpec(run_id=run_id, input=task, arm=arm_id, configuration_id=configuration["id"],
                   settings=copy.deepcopy(configuration["settings"]), experiment=experiment, split=split,
                   slate=slate, judges=tuple(applied), person=person,
                   flow=FlowRef(id=flow.id, version=flow.version, digest=folder.digest, arm=arm_id, plan=plan.id,
                                split=split),
                   record=RECORD, extra={"slate_members": list(members)})


def _draw(plan_seed: int, run_id: str, judge_id: str) -> float:
    """A uniform draw in [0, 1) for one run and judge, from the plan seed."""
    return int(hashlib.sha256(f"{plan_seed}|{run_id}|{judge_id}".encode()).hexdigest()[:13], 16) / 16 ** 13
