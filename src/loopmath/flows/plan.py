"""The plan: arms and experiments for a flow, and the margin price table.

With 3 or more allowed edits the plan starts with a screen over them; then the best subset, or a single edit,
against the base on the holdout inputs; then, if that check does not adopt, the subset's edits one at a time.
The edits the hypothesis names go first.

`quality` maps a run id to its unpooled estimate and sd, and `fit` is the judges fit; when they are None they
are computed from `docs` with one `judges.fit` and `judges.unpooled_quality` under it, and only when a run of the
flow holds a measured score signal a judge writes (a number; a declared null score is not one); with none,
`judges` is not called and a missing one is empty. A run of the flow is a run document (kind `run`, or no kind)
whose `run.flow` has the flow's id and version and, when it states one, the flow's digest; other documents and
other revisions are ignored, for the spreads, the run cost and the progress.

How the first plan is made:

- **Factors.** Each allowed edit other than a merge belongs to a dimension: a piece's model, effort, option
  key or width, a piece's drop, the round budget, one added check or one removed check. A dimension is one
  two-level factor: the base value and one alternative, the value the hypothesis names, else the first value
  `allow` lists. Its other values are queued. The factors the hypothesis names come first, in its order, then
  the rest in `allow` order.
- **Arms** are whole configurations. An arm of several edits applies them to the base in factor order, drops
  last; an edit at a piece the same arm drops is left out of that arm. One edit keeps its candidate's id;
  several get `set-` and their 1-based factor numbers (`set-1-3-4`), or `set-` and 12 hex digits of a hash of
  their edits when that is over 64 characters or is a candidate's id.
- **The screen** (3 or more factors) is a fold-over design (`screen/design.py`) on the train inputs: arm A of a
  pair runs the factors at +1, arm B those at -1. Its pairs are the margin table's interval figure, rounded up
  to a block whose columns are orthogonal, so 2 standard errors of each effect are about the margin.
- **Checks** run on the holdout inputs, one arm against its reference, capped at the check figure: one per
  merge, against the base, and the subset check, whose arm is known once the screen is done. With 1 or 2
  factors there is no screen, and the check of all of them is planned at once. A margin whose check figure, at
  the spreads in use, passes `MAX_PAIRS` is refused by name before any design is built. `margin_problem` is
  that test, and the margin's own, as a function that returns the reason and never raises, for a reader that
  checks a margin before there is a plan.
- **Spreads** are assumed (s_run 0.12, s_judge 0.10, k the judges that are not person judges) until 8 scored
  pairs exist and the fit gives the run noise and the model judges' noise; then fitted. A scored pair is a
  slate of two runs of the flow on one input (the same string `task.id`), each carrying the slate with the
  same two members and a number quality: a pair the judges fit can read the run noise from.
- **Price** is runs times the mean dollars of the flow's recorded runs (base runs when there are any), or None
  with no recorded run. When the mean or the product is past what a float holds, the price is None with a
  `reason`, so the plan holds no infinite number.
- **Ids.** `seed` is the first 8 hex digits of SHA-256 over `loopmath.flow-plan|<flow digest>`; it seeds the
  design and the spot checks. `id` is `pln_` and 12 hex digits of SHA-256 over the plan's flow digest, seed,
  margin, arm configuration ids and experiment layout, so the same flow and margin give the same plan, and a
  new layout gives new run ids.

How `advance` brings a plan up to date (the plan id never changes, so no run id moves):

- **States and verdicts.** A screen is `planned` while none of its runs is recorded, `done` with every pair
  recorded, else `running`. A check with no run recorded is `planned`; otherwise its verdict is
  `check.check_experiment`'s: adopt or reject make it `done` (with `stop`, and the check's `nonfinite`,
  `cost_range` and `unreadable` when it gives them), more keeps it `running` with verdict `more`. A done
  experiment keeps its state and verdict.
- **The subset.** Once the screen is done, `check.screen_readout` reads it. The set of factors it chooses
  becomes the subset check's arm, sized at the spreads of that moment; the readout is kept in the screen's
  `readout`. When it chooses nothing (no set, too few pairs, effects not identified, too many factors, a run
  with no dollars or dollars no float holds, or a number of the fit past what a float holds) the subset check is
  done with verdict reject and that `stop`, and nothing follows.
- **The single checks.** When the subset check rejects, the subset's factors are checked one at a time, in
  the readout's ranking (factor order without a screen), each against the arm adopted so far, starting from
  the base. The candidate is the adopted factors plus the next one, as one canonical set; a factor that
  changes nothing there (a setting on a piece the adopted arm drops) is skipped and listed in the subset's
  `skipped`. Each single is added once the one before it is done; its id is `single-<arm id>`, with `-2`, `-3`
  when that id is taken. The subset's `chain` lists the singles walked, in order, and `adopted` reads it, so a
  single the chain did not reach adopts nothing. A subset of one factor has no singles. Merge checks stand apart.
- **New experiments** go after the existing ones, and new arms after the existing arms.

Each experiment's `design` holds `pairs`, `runs`, `price` (`usd`, `per_run`, `from_runs`, and `reason` when the
price is unknown that way) and `spreads`
(`s_run`, `s_judge`, `k`, `sd_pair`, `label`, `scored_pairs`). A screen adds `factors`, `removed_checks` (per
factor), `rows`, `layout`, `orthogonal`, `seed` and `pair_arms`; a merge check adds `removed_checks`; the
subset check adds `after`.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from ..belief.outcome import signal_list
from ..store.runs import run_cost
from ..types import FlowRef
from . import edits, judges
from .folder import PLAN_FILE, FlowFolder
from .records import PLAN_SCHEMA, Arm, Edit, Experiment, Flow, Plan, hypothesis_edits, judge_score

S_RUN = 0.12  # assumed run noise
S_JUDGE = 0.10  # assumed judge noise
FITTED_PAIRS = 8  # scored pairs before the spreads are fitted
MARGIN = 0.10  # when neither the caller nor the objective gives one
MAX_PAIRS = 10 ** 6  # the most check pairs a margin may need
Z_CHECK = NormalDist().inv_cdf(0.95) + NormalDist().inv_cdf(0.80)
_ARM_LEN = 64
_SETTING_KINDS = ("model", "effort", "option", "width")


def make_plan(folder: FlowFolder, *, docs: Sequence[dict] = (), budget_usd: float | None = None,
              margin: float | None = None, quality: Mapping[str, tuple[float, float | None]] | None = None,
              fit: Mapping[str, Any] | None = None) -> Plan:
    """The first plan for a flow."""
    flow = folder.flow
    if margin is None:
        margin = flow.objective.margin if flow.objective.margin is not None else MARGIN
    why = _margin_why(margin)
    if why is not None:
        raise ValueError(why)
    if budget_usd is not None and not _number(budget_usd):
        raise ValueError(f"budget_usd {budget_usd!r}: a budget is a number of dollars, 0 or more")
    runs = flow_runs(folder, docs)
    quality, fit = quality_and_fit(flow, runs, quality, fit)
    spreads = _spreads(flow, runs, quality, fit)
    check_pairs = margin_table(margins=(margin,), s_run=spreads["s_run"], s_judge=spreads["s_judge"],
                               k=spreads["k"])[0]["check_pairs"]
    per_run, from_runs = _run_cost(runs, sizing=True)
    seed = int(hashlib.sha256(f"loopmath.flow-plan|{folder.digest}".encode()).hexdigest()[:8], 16)

    candidates, excluded = edits.candidates(flow)
    hypothesis = hypothesis_edits(flow)
    factors, queued = _factors(candidates, hypothesis)
    merges = _ranked([a for a in candidates if a.edits[0].kind == "merge"], hypothesis)
    maker = _Arms(flow, Arm(id="base", configuration=copy.deepcopy(flow.configuration), edits=()), factors,
                  {a.id for a in candidates})

    size = _Size(check_pairs, per_run, from_runs, spreads)
    experiments: list[Experiment] = []
    rows = None
    if len(factors) >= 3:
        rows, n_pairs, layout = _screen_rows(len(factors), margin, spreads, seed)
        pair_arms = [[maker.arm([j for j, x in enumerate(row) if x > 0]).id,
                      maker.arm([j for j, x in enumerate(row) if x < 0]).id] for row in rows]
        experiments.append(Experiment(
            id="screen", kind="screen", arms=tuple(dict.fromkeys(a for pair in pair_arms for a in pair)),
            reference="base", split="train",
            design={"factors": [a.edits[0].to_dict() for a in factors],
                    "removed_checks": [_removed(flow, a.edits[0]) for a in factors],
                    "rows": rows, "layout": layout["kind"], "orthogonal": layout["orthogonal"], "seed": seed,
                    "pair_arms": pair_arms, **size.design(n_pairs)},
            pair_cap=n_pairs, state="planned"))
    for merge in merges:
        maker.add(merge)
        experiments.append(size.check(merge.id, "merge", (merge.id,), removed_checks=_removed(flow, merge.edits[0])))
    if len(factors) >= 3:
        experiments.append(size.check("subset", "subset", (), after="screen"))
    elif factors:
        arm = maker.arm(list(range(len(factors))))
        experiments.append(size.check("subset", "subset", (arm.id,)) if len(factors) == 2
                           else size.check(f"single-{arm.id}", "single", (arm.id,)))

    arms = tuple(maker.arms)
    plan_id = "pln_" + _hash({"flow": folder.digest, "seed": seed, "margin": margin,
                              "arms": [[a.id, a.configuration.get("id")] for a in arms],
                              "experiments": [[e.id, e.kind, list(e.arms), e.reference, e.split, e.pair_cap,
                                               rows if e.kind == "screen" else None] for e in experiments]})[:12]
    return Plan(schema=PLAN_SCHEMA, id=plan_id, flow=FlowRef(id=flow.id, version=flow.version, digest=folder.digest),
                seed=seed, margin=margin, arms=arms, experiments=tuple(experiments), queued=tuple(queued),
                excluded=tuple(excluded), budget_usd=budget_usd)


def advance(folder: FlowFolder, docs: Sequence[dict], *,
            quality: Mapping[str, tuple[float, float | None]] | None = None,
            fit: Mapping[str, Any] | None = None) -> Plan:
    """The plan with experiment states and verdicts brought up to date, plus the experiments that follow once
    earlier ones finish (see the module docstring). The plan id never changes, so no run id moves."""
    from . import check as check_mod

    plan = current_plan(folder)
    flow = folder.flow
    runs = flow_runs(folder, docs)
    quality, fit = quality_and_fit(flow, runs, quality, fit)
    step = _Step(folder, docs, runs, quality, fit, check_mod)
    for _ in range(4 * (len(plan.experiments) + len(plan.arms)) + 16):
        settled = step.follow(step.verdicts(plan))
        if settled == plan:
            return plan
        plan = settled
    raise RuntimeError(f"plan {plan.id} did not settle")


def adopted(plan: Plan) -> str:
    """The arm adopted last along the chain: the subset check's arm when it adopts, else the arm of the last
    single check that adopts in the subset's `chain`, the singles `advance` walked, in order, each against the arm
    adopted before it (`base` at first). With no subset the plan's single check is the chain. A single that is not
    in the chain adopts nothing, and neither does a merge check, whose verdict is its own. A chain entry that is not
    a single check against the arm adopted before it is refused."""
    by_id = {e.id: e for e in plan.experiments}
    subset = next((e for e in plan.experiments if e.kind == "subset"), None)
    if subset is None:
        chain = [e.id for e in plan.experiments if e.kind == "single"][:1]
    elif subset.verdict == "adopt" and subset.arms:
        return subset.arms[0]
    else:
        chain = subset.extra.get("chain") or []
    arm = "base"
    for exp_id in chain:
        single = by_id.get(exp_id)
        if single is None or single.kind != "single" or not single.arms or (single.reference or "base") != arm:
            raise ValueError(f"plan {plan.id}: {exp_id!r} in the subset's chain is not a single check against {arm!r}; "
                             "run loopmath flow plan")
        if single.verdict == "adopt":
            arm = single.arms[0]
    return arm


def margin_table(*, margins: Sequence[float] = (0.15, 0.10, 0.05), s_run: float = 0.12, s_judge: float = 0.10,
                 k: int = 2) -> list[dict]:
    """Pairs and price per margin: the interval figure and the check figure, which is the pair cap.

    sd_pair = sqrt(2 (s_run^2 + s_judge^2 / k)) for k judges averaged. The interval figure, ceil((2 sd_pair /
    m)^2), is the pairs at which 2 standard errors equal the margin m. The check figure, ceil(((z_0.95 + z_0.80)
    sd_pair / m)^2), gives 80% power to adopt a change with no true quality loss at the 0.95 threshold; `runs`
    is twice it. There is no run cost here: the dollars are `runs` times `mean_run_cost`.

    A margin whose check figure passes `MAX_PAIRS` is refused by name, with the pairs it would need, and so is a
    spread whose square, or whose pair sd, is past what a float holds: each with `margin_problem`'s reason."""
    sd = _pair_sd(s_run, s_judge, k)
    table = []
    for m in margins:
        why = margin_problem(m, k=k, s_run=s_run, s_judge=s_judge)
        if why is not None:
            raise ValueError(why)
        check = _figure(Z_CHECK * sd / m)
        table.append({"margin": m, "s_run": s_run, "s_judge": s_judge, "k": k, "sd_pair": sd,
                      "interval_pairs": _figure(2 * sd / m), "check_pairs": check, "runs": 2 * check})
    return table


def margin_problem(margin: Any, *, k: Any, s_run: Any = S_RUN, s_judge: Any = S_JUDGE) -> str | None:
    """Why the planner refuses `margin` at these spreads, in the words it raises, or None; it never raises.

    The tests are `margin_table`'s, in its order: each spread a number 0 or more, `k` an integer 1 or more, a
    spread's square and the pair sd within what a float holds, the margin a number above 0, and its check figure
    at most `MAX_PAIRS`. A reader that checks a margin before any plan (validation, `flow init`) passes `k` the
    flow's judges that are not person judges, at least 1, and keeps the default spreads: those are the assumed
    spreads of a first plan, so it refuses exactly what that plan refuses. A plan at fitted spreads may still
    refuse the margin, by name, since its spreads are no longer these."""
    why = _spread_why(s_run, s_judge, k)
    if why is None:
        why = _margin_why(margin)
    return why if why is not None else _limit_why(margin, sd_pair(s_run, s_judge, k))


def sd_pair(s_run: float, s_judge: float, k: int) -> float:
    """The sd of one pair's quality difference."""
    return math.sqrt(2 * (s_run ** 2 + s_judge ** 2 / k))


def mean_run_cost(folder: FlowFolder, docs: Sequence[dict]) -> float | None:
    """The mean dollars of the flow's recorded runs (dollars complete), over its base runs when there are any;
    None with none."""
    return _run_cost(flow_runs(folder, docs))[0]


# Reading runs and pairs (also used by `next`)


def flow_runs(folder: FlowFolder, docs: Sequence[dict]) -> dict[str, dict]:
    """Run id to run document, for the runs of this flow revision in `docs` (see the module docstring)."""
    flow, runs = folder.flow, {}
    for doc in docs:
        if not isinstance(doc, dict) or doc.get("kind", "run") != "run" or not isinstance(doc.get("run"), dict):
            continue
        run = doc["run"]
        ref = run.get("flow")
        if isinstance(ref, dict) and ref.get("id") == flow.id and ref.get("version") == flow.version \
                and ref.get("digest") in (None, folder.digest) and isinstance(run.get("id"), str):
            runs[run["id"]] = doc
    return runs


def quality_and_fit(flow: Flow, runs: Mapping[str, dict], quality: Mapping[str, Any] | None,
                    fit: Mapping[str, Any] | None) -> tuple[Mapping, Mapping]:
    """`quality` and `fit` as given; a missing one is computed with `judges` from the flow's runs, but only when
    one of them holds a judge's score signal, and is empty otherwise."""
    if quality is not None and fit is not None:
        return quality, fit
    if not any(_scored(flow, doc) for doc in runs.values()):
        return quality or {}, fit or {}
    if fit is None:
        fit = judges.fit(flow, list(runs.values()))
    if quality is None:
        quality = {run_id: judges.unpooled_quality(flow, doc, fit) for run_id, doc in runs.items()}
    return quality, fit


def current_plan(folder: FlowFolder) -> Plan:
    """The folder's plan; refused when there is none or it was made for another flow digest."""
    if folder.plan is None:
        raise ValueError(f"there is no {PLAN_FILE} in {folder.dir}; run loopmath flow plan first")
    if folder.plan_stale:
        raise ValueError(f"{folder.dir / PLAN_FILE} is stale: it was made for flow digest {folder.plan.flow.digest}, "
                         f"the flow's digest is {folder.digest}; run loopmath flow plan")
    return folder.plan


def pairs(plan: Plan, exp: Experiment) -> list[tuple[str, str, str]]:
    """(pair key, arm of run a, arm of run b) for each pair of an experiment, in order: a screen pair runs arm A
    and arm B of its row, a check pair the candidate and the reference. Empty until the arms are known."""
    if not exp.arms:
        return []
    if exp.kind == "screen":
        arm_pairs = [tuple(p) for p in (exp.design or {}).get("pair_arms") or []]
    else:
        arm_pairs = [(exp.arms[0], exp.reference or "base")] * (exp.pair_cap or 0)
    return [(f"{plan.id}.{exp.id}.{i:03d}", a, b) for i, (a, b) in enumerate(arm_pairs, start=1)]


def run_ids(key: str) -> tuple[str, str]:
    """The two run ids of the pair `key`."""
    return f"run_{key}.a", f"run_{key}.b"


def recorded_runs(folder: FlowFolder, plan: Plan, docs: Sequence[dict]) -> set[str]:
    """The run ids of this plan's runs in `docs`."""
    return {run_id for run_id, doc in flow_runs(folder, docs).items() if doc["run"]["flow"].get("plan") == plan.id}


# Helpers


def _margin_why(m: Any) -> str | None:
    if isinstance(m, bool) or not isinstance(m, (int, float)) or not _isfinite(m) or m <= 0:
        return f"margin {_shown(m)}: a margin is a number above 0"
    return None


def _shown(value: Any) -> str:
    """`repr(value)` for a reason, or a stand-in that names the value where repr cannot complete: an int past
    the interpreter's digit limit for its decimal text, or an object whose repr raises. A reason never raises."""
    try:
        return repr(value)
    except Exception:
        if isinstance(value, int):
            digits = int(int.bit_length(value) * math.log10(2)) + 1
            return f"<{'negative ' if value < 0 else ''}int of about {digits} digits>"
        return f"<{type(value).__name__} with no repr>"


def _figure(x: float) -> int:
    """ceil(x^2), at least 1; the small allowance keeps a float such as 387.99999999999994 at 388."""
    return max(1, math.ceil(x * x - 1e-9))


def _limit_why(m: float, sd: float) -> str | None:
    """Why margin `m` at pair sd `sd` is refused: its check figure is past `MAX_PAIRS`. The float `x * x` is inf
    where it overflows, so the test never raises, and ceil(v) is at most `MAX_PAIRS` exactly when v is."""
    x = Z_CHECK * sd / m
    need = x * x - 1e-9
    if not need <= MAX_PAIRS:
        count = (str(_figure(x)) if need < 1e15 else f"about {need:.2g}") if math.isfinite(need) else "more than 1e308"
        return f"margin {_shown(m)}: needs {count} check pairs at sd_pair {sd:.4g}, above the limit of {MAX_PAIRS}"
    return None


def _spread_why(s_run: Any, s_judge: Any, k: Any) -> str | None:
    """Why the spreads give no pair sd: a spread that is not a number 0 or more, a `k` that is not an integer 1
    or more, or where `sd_pair` would raise OverflowError or give inf: a spread's square, or the pair sd, past
    what a float holds."""
    for name, value in (("s_run", s_run), ("s_judge", s_judge)):
        if not _number(value):
            return f"{name} {_shown(value)}: a spread is a number, 0 or more"
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return f"k {_shown(k)}: the number of judges averaged is an integer, 1 or more"
    for name, value in (("s_run", s_run), ("s_judge", s_judge)):
        if not math.isfinite(float(value) * float(value)):
            return f"{name} {_shown(value)}: a spread's square is past what a float holds"
    try:
        sd = sd_pair(s_run, s_judge, k)
    except OverflowError:
        sd = math.inf
    if not math.isfinite(sd):
        return (f"s_run {_shown(s_run)}, s_judge {_shown(s_judge)}: the pair sd of these spreads is past what a "
                "float holds")
    return None


def _pair_sd(s_run: float, s_judge: float, k: int) -> float:
    """`sd_pair` for spreads `_spread_why` passes, refused by name with its reason otherwise."""
    why = _spread_why(s_run, s_judge, k)
    if why is not None:
        raise ValueError(why)
    return sd_pair(s_run, s_judge, k)


def _scored(flow: Flow, doc: dict) -> bool:
    """The run holds a measured score signal (a finite number, not null) under a name a judge of the flow
    writes; a `run.signals` that is not a list holds none (`signal_list`)."""
    names = {judge_score(j, flow.objective) for j in flow.judges or ()} | {flow.objective.score}
    return any(isinstance(s, dict) and s.get("kind") == "score" and s.get("name") in names - {None}
               and _finite(s.get("value")) for s in signal_list(doc["run"]))


def _dollars(doc: dict) -> float | None:
    """The run's dollars (`run_cost`); inf when an attempt's dollars are an integer no float holds, which the store's
    reader cannot convert: the mean is past what a float holds then too."""
    try:
        return run_cost(doc)["usd"]
    except OverflowError:
        return math.inf


def _finite(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and _isfinite(x)


def _number(x: Any) -> bool:
    """A finite number, 0 or more."""
    return not isinstance(x, bool) and isinstance(x, (int, float)) and _isfinite(x) and x >= 0


def _isfinite(x: int | float) -> bool:
    """math.isfinite, and False for an int no float can hold, which it would raise on."""
    try:
        return math.isfinite(x)
    except OverflowError:
        return False


def _spreads(flow: Flow, runs: Mapping[str, dict], quality: Mapping[str, Any], fit: Mapping[str, Any]) -> dict:
    """The spreads the pair counts use, labelled assumed or fitted (see the module docstring)."""
    scored = _scored_pairs(runs, quality)
    noise = [float(j["noise"]) for j in (fit.get("judges") or [])
             if isinstance(j, Mapping) and j.get("kind") == "model" and _number(j.get("noise"))]
    if scored >= FITTED_PAIRS and _number(fit.get("s_run")) and noise:
        square = sum(n * n for n in noise) / len(noise)
        if not math.isfinite(square):
            raise ValueError(f"judge noise {noise!r}: the model judges' mean square noise is past what a float holds")
        s_run, s_judge, k = float(fit["s_run"]), math.sqrt(square), len(noise)
        label = "fitted"
    else:
        s_run, s_judge, label = S_RUN, S_JUDGE, "assumed"
        k = max(1, sum(1 for j in flow.judges or () if j.kind != "person"))
    return {"s_run": s_run, "s_judge": s_judge, "k": k, "sd_pair": _pair_sd(s_run, s_judge, k), "label": label,
            "scored_pairs": scored}


def _scored_pairs(runs: Mapping[str, dict], quality: Mapping[str, Any]) -> int:
    """Slates whose two members are both runs of this flow that carry the slate, list the same two members, have
    a number quality and name one input, the same string `task.id`. A run listing a member that does not carry its
    slate counts for nothing, and so does a pair on two inputs or with an input missing."""
    def slate(run_id: str) -> tuple[str, frozenset] | None:
        s = runs[run_id]["run"].get("slate")
        members = s.get("members") if isinstance(s, dict) else None
        if not isinstance(s.get("id") if isinstance(s, dict) else None, str) or not isinstance(members, list) \
                or len(members) != 2 or not all(isinstance(m, str) for m in members) or len(set(members)) != 2:
            return None
        return s["id"], frozenset(members)

    def scored(run_id: str) -> bool:
        q = quality.get(run_id)
        return isinstance(q, (tuple, list)) and len(q) >= 1 and _finite(q[0])

    def input_of(run_id: str) -> str | None:
        task = runs[run_id]["run"].get("task")
        return task.get("id") if isinstance(task, dict) and isinstance(task.get("id"), str) else None

    slates = set()
    for run_id in runs:
        own = slate(run_id)
        if own is not None and run_id in own[1] and all(m in runs and slate(m) == own and scored(m) for m in own[1]) \
                and input_of(run_id) is not None and all(input_of(m) == input_of(run_id) for m in own[1]):
            slates.add(own[0])
    return len(slates)


def _run_cost(runs: Mapping[str, dict], *, sizing: bool = False) -> tuple[float | None, int]:
    """The mean dollars per run, and how many runs it is over. A check's sizing (`sizing`) reads each run through
    `_dollars`, so an attempt's dollars no float holds give a mean past what a float holds, which the price names;
    `mean_run_cost` reads `run_cost` as it is."""
    priced = []
    for doc in runs.values():
        usd = _dollars(doc) if sizing else run_cost(doc)["usd"]
        if usd is not None:
            priced.append((doc["run"]["flow"].get("arm") in (None, "base"), usd))
    pool = [usd for is_base, usd in priced if is_base] or [usd for _, usd in priced]
    return (sum(pool) / len(pool), len(pool)) if pool else (None, 0)


def _dimension(edit: Edit) -> tuple:
    """What one factor varies: a piece's setting or width, a drop, the budget, or one check."""
    if edit.kind == "check_add":
        value = edit.value if isinstance(edit.value, dict) else {}
        return ("check_add", value.get("id") or json.dumps(edit.value, sort_keys=True, default=str))
    if edit.kind == "check_remove":
        return ("check_remove", edit.value)
    return (edit.kind, edit.piece, edit.key)


def _same(a: Edit, b: Edit) -> bool:
    """The same edit; models match on their id."""
    if (a.kind, a.piece, a.key) != (b.kind, b.piece, b.key):
        return False
    if a.kind == "model" and isinstance(a.value, dict) and isinstance(b.value, dict):
        return a.value.get("id") == b.value.get("id")
    return a.value == b.value


def _named(arm: Arm, hypothesis: Sequence[Edit]) -> int | None:
    """The position of the arm's edit among the hypothesis edits, or None."""
    return next((i for i, h in enumerate(hypothesis) if _same(h, arm.edits[0])), None)


def _ranked(arms: list[Arm], hypothesis: Sequence[Edit]) -> list[Arm]:
    """The arms the hypothesis names, in its order, then the rest in their own order."""
    named = sorted((a for a in arms if _named(a, hypothesis) is not None), key=lambda a: _named(a, hypothesis))
    return named + [a for a in arms if _named(a, hypothesis) is None]


def _factors(candidates: list[Arm], hypothesis: Sequence[Edit]) -> tuple[list[Arm], list[Edit]]:
    """One candidate arm per dimension (the factor's alternative), ranked; the other values, queued."""
    groups: dict[tuple, list[Arm]] = {}
    for arm in candidates:
        if arm.edits[0].kind != "merge":
            groups.setdefault(_dimension(arm.edits[0]), []).append(arm)
    chosen = []
    for group in groups.values():
        named = [a for a in group if _named(a, hypothesis) is not None]
        chosen.append(min(named, key=lambda a: _named(a, hypothesis)) if named else group[0])
    ids = {a.id for a in chosen}
    queued = [a.edits[0] for a in candidates if a.edits[0].kind != "merge" and a.id not in ids]
    return _ranked(chosen, hypothesis), queued


def _removed(flow: Flow, edit: Edit) -> list[dict]:
    """The checks one edit removes from the base, as `{name, rule}`."""
    return [{"name": name, "rule": rule} for name, rule in edits.removed_checks(flow.configuration, edit, flow)]


def _screen_rows(p: int, margin: float, spreads: dict, seed: int) -> tuple[list[list[int]], int, dict]:
    """The fold-over rows, the pairs and the layout (`design.plan`). The screen's design is imported here, so
    importing the plan does not load the screen."""
    from ..screen import design as screen_design

    interval = margin_table(margins=(margin,), s_run=spreads["s_run"], s_judge=spreads["s_judge"],
                            k=spreads["k"])[0]["interval_pairs"]
    block = next((s for s in screen_design.SIZES if s >= p + 1), None)
    if block is not None:
        m = block * max(1, math.ceil(interval / block))
    else:
        m = max(interval, p + 1)
        m += m % 2
    rows = [[int(x) for x in row] for row in screen_design.fold_over(p, m, seed)]
    return rows, m, screen_design.plan(p, m)


class _Arms:
    """The plan's arms: the base, then each set of factor edits made once, in the order first asked for.

    A set's id depends on the set and the reserved ids (the base and every candidate's id) only, never on the
    order the sets are asked for."""

    def __init__(self, flow: Flow, base: Arm, factors: list[Arm], reserved: set[str]) -> None:
        self.flow, self.factors = flow, factors
        self.reserved = reserved | {base.id}
        self.arms: list[Arm] = [base]
        self._by_set: dict[tuple[int, ...], Arm] = {(): base}

    def add(self, arm: Arm) -> Arm:
        if any(a.id == arm.id for a in self.arms):
            raise ValueError(f"arm id {arm.id!r} is used twice")
        self.arms.append(arm)
        return arm

    def kept(self, on: Sequence[int]) -> tuple[int, ...]:
        """The factors an arm with the factors `on` (indices, any order) keeps: sorted, each once, without an
        edit at a piece the same arm drops."""
        on = sorted(set(on))
        edit = [f.edits[0] for f in self.factors]
        dropped = {edit[j].piece for j in on if edit[j].kind == "drop"}
        return tuple(j for j in on if edit[j].kind not in _SETTING_KINDS or edit[j].piece not in dropped)

    def arm(self, on: Sequence[int]) -> Arm:
        """The arm with the factors `on` (indices, any order; see `kept`)."""
        edit = [f.edits[0] for f in self.factors]
        kept = self.kept(on)
        if kept in self._by_set:
            return self._by_set[kept]
        if len(kept) == 1:
            arm = self.factors[kept[0]]
        else:
            order = [j for j in kept if edit[j].kind != "drop"] + [j for j in kept if edit[j].kind == "drop"]
            arm = Arm(id=self._id(kept, order), configuration=self._apply([edit[j] for j in order]),
                      edits=tuple(edit[j] for j in order))
        self._by_set[kept] = arm
        return self.add(arm)

    def _apply(self, steps: list[Edit]) -> dict:
        configuration = self.flow.configuration
        labels = ", ".join(json.dumps(e.to_dict(), sort_keys=True) for e in steps)
        try:
            for step in steps:
                configuration = edits.apply(configuration, step, self.flow)
        except ValueError as exc:
            raise ValueError(f"the arm with the edits {labels} cannot be made: {exc}") from exc
        errors = [f for f in edits.validate(configuration) if f.level == "error"]
        if errors:
            raise ValueError(f"the arm with the edits {labels} is not a valid configuration: "
                             + "; ".join(f"{f.code} at {f.path}" for f in errors))
        return configuration

    def _id(self, kept: tuple[int, ...], order: list[int]) -> str:
        """`set-` and the factor numbers; `set-` and 12 hex digits of the edits' hash when that is too long or
        reserved, with `-2`, `-3` only if the hashed id is reserved too. Two sets never get one id: numbered ids
        differ by set, and a hashed id's first field is 12 hex digits, never a factor number."""
        stem = "set-" + "-".join(str(j + 1) for j in kept)
        if len(stem) > _ARM_LEN or stem in self.reserved:
            stem = "set-" + _hash([self.factors[j].edits[0].to_dict() for j in order])[:12]
        arm_id, n = stem, 1
        while arm_id in self.reserved:
            n += 1
            arm_id = f"{stem}-{n}"
        return arm_id


class _Size:
    """A check's design at given spreads: pairs, runs and price, as `make_plan` sizes it."""

    def __init__(self, check_pairs: int, per_run: float | None, from_runs: int, spreads: dict) -> None:
        self.check_pairs, self.per_run, self.from_runs, self.spreads = check_pairs, per_run, from_runs, spreads

    def design(self, pairs: int) -> dict:
        """A price past what a float holds is None with a `reason`, so the plan holds no infinite number."""
        price = {"usd": None, "per_run": self.per_run, "from_runs": self.from_runs}
        if self.per_run is not None:
            usd = 2 * pairs * self.per_run
            if not _isfinite(self.per_run):
                price.update(per_run=None, reason=f"the mean dollars of the {self.from_runs} recorded runs is past "
                                                  "what a float holds")
            elif not math.isfinite(usd):
                price["reason"] = (f"the price of {2 * pairs} runs at {self.per_run:.4g} dollars each is past what a "
                                   "float holds")
            else:
                price["usd"] = round(usd, 4)
        return {"pairs": pairs, "runs": 2 * pairs, "price": price, "spreads": dict(self.spreads)}

    def check(self, exp_id: str, kind: str, arms: tuple[str, ...], reference: str = "base", **more: Any) -> Experiment:
        return Experiment(id=exp_id, kind=kind, arms=arms, reference=reference, split="holdout",
                          design={**self.design(self.check_pairs), **more}, pair_cap=self.check_pairs,
                          state="planned")


class _Step:
    """One `advance`: verdicts, then the experiments that follow, until the plan settles."""

    def __init__(self, folder: FlowFolder, docs: Sequence[dict], runs: Mapping[str, dict], quality: Mapping,
                 fit: Mapping, check_mod: Any) -> None:
        self.folder, self.docs, self.runs, self.quality, self.fit = folder, docs, runs, quality, fit
        self.check_mod = check_mod
        self.results: dict[tuple, Any] = {}
        self._size: _Size | None = None
        self._maker: _Arms | None = None

    def verdicts(self, plan: Plan) -> Plan:
        """Each experiment's state, and each check's verdict; a done one is kept as it is."""
        recorded = recorded_runs(self.folder, plan, self.docs)
        experiments = []
        for exp in plan.experiments:
            current = exp.state or "planned"
            if current == "done":
                new = exp
            elif exp.kind == "screen" or not exp.arms:
                new = dataclasses.replace(exp, state=_state(plan, exp, recorded))
            elif not any(run_id in recorded for key, _, _ in pairs(plan, exp) for run_id in run_ids(key)):
                new = dataclasses.replace(exp, state="planned", verdict=None)
            else:
                r = self._check(plan, exp)
                if r.verdict in ("adopt", "reject"):
                    new = dataclasses.replace(exp, state="done", verdict=r.verdict,
                                              extra={**exp.extra, "stop": r.extra["stop"],
                                                     **{k: r.extra[k]
                                                        for k in ("nonfinite", "cost_range", "unreadable")
                                                        if k in r.extra}})
                else:
                    new = dataclasses.replace(exp, state="running", verdict="more")
            experiments.append(exp if new == exp else new)
        return dataclasses.replace(plan, experiments=tuple(experiments))

    def follow(self, plan: Plan) -> Plan:
        """The subset's arm once the screen is read, then the next single check after a subset that does not
        adopt; at most one change per call."""
        exps = list(plan.experiments)
        subset = next((i for i, e in enumerate(exps) if e.kind == "subset"), None)
        if subset is None:
            return plan
        screen = next((i for i, e in enumerate(exps) if e.kind == "screen"), None)
        if not exps[subset].arms:
            if exps[subset].state == "done" or screen is None or exps[screen].state != "done":
                return plan
            return self._read(plan, exps, screen, subset)
        if exps[subset].state != "done" or exps[subset].verdict != "reject":
            return plan
        if screen is None:
            ranking = list(range(len(self.maker().factors)))
        else:
            readout = exps[screen].extra.get("readout") or {}
            ranking = [j - 1 for j in readout.get("ranking") or []]
        if len(ranking) < 2:  # a subset of one factor is already that factor's single check
            return plan
        return self._chain(plan, exps, subset, ranking)

    def _check(self, plan: Plan, exp: Experiment) -> Any:
        key = (exp.id, exp.arms, exp.reference, exp.pair_cap)
        if key not in self.results:
            self.results[key] = self.check_mod.check_experiment(self.folder, plan, exp, self.docs,
                                                                quality=self.quality, fit=self.fit)
        return self.results[key]

    def _read(self, plan: Plan, exps: list[Experiment], screen: int, subset: int) -> Plan:
        readout = exps[screen].extra.get("readout")
        if readout is None:
            readout = self.check_mod.screen_readout(self.folder, plan, exps[screen], self.docs, quality=self.quality,
                                                    fit=self.fit)
            if readout["status"] == "wait":
                return plan
            exps[screen] = dataclasses.replace(exps[screen], extra={**exps[screen].extra, "readout": readout})
        arms = list(plan.arms)
        if readout["status"] == "stop":
            exps[subset] = dataclasses.replace(exps[subset], state="done", verdict="reject",
                                               extra={**exps[subset].extra, "stop": readout["stop"]})
        else:
            maker = self.maker()
            if [a.edits[0].to_dict() for a in maker.factors] != list((exps[screen].design or {}).get("factors") or []):
                raise ValueError(f"plan {plan.id}: the screen's factors are not the flow's; run loopmath flow plan")
            arm = _placed(arms, maker.arm([j - 1 for j in readout["chosen"]]), plan.id)
            size = self.size(plan)
            exps[subset] = dataclasses.replace(exps[subset], arms=(arm.id,), pair_cap=size.check_pairs,
                                               design={**(exps[subset].design or {}), **size.design(size.check_pairs)})
        return dataclasses.replace(plan, arms=tuple(arms), experiments=tuple(exps))

    def _chain(self, plan: Plan, exps: list[Experiment], subset: int, ranking: list[int]) -> Plan:
        """The single checks over the subset's factors, one at a time, each against the arm adopted so far."""
        maker = self.maker()
        arms = list(plan.arms)
        kept: tuple[int, ...] = ()
        reference = "base"
        skipped: list[dict] = []
        chain: list[str] = []
        added = None
        for k in ranking:
            if maker.kept(kept + (k,)) == kept:
                skipped.append({"factor": k + 1, "reason": f"factor {k + 1} changes nothing on arm {reference!r}"})
                continue
            arm = maker.arm(kept + (k,))
            single = next((e for e in exps if e.kind == "single" and e.arms == (arm.id,)
                           and (e.reference or "base") == reference), None)
            if single is None:
                arm = _placed(arms, arm, plan.id)
                exp_id, n = f"single-{arm.id}", 1
                while any(e.id == exp_id for e in exps):
                    n += 1
                    exp_id = f"single-{arm.id}-{n}"
                added = self.size(plan).check(exp_id, "single", (arm.id,), reference=reference)
                chain.append(exp_id)
                break
            chain.append(single.id)
            if single.state != "done":
                break
            if single.verdict == "adopt":
                kept, reference = maker.kept(kept + (k,)), arm.id
        extra = dict(exps[subset].extra)
        if skipped or "skipped" in extra:
            extra["skipped"] = skipped
        extra["chain"] = chain
        if extra != exps[subset].extra:
            exps[subset] = dataclasses.replace(exps[subset], extra=extra)
        if added is not None:
            exps.append(added)
        return dataclasses.replace(plan, arms=tuple(arms), experiments=tuple(exps))

    def maker(self) -> _Arms:
        """The arm maker `make_plan` used, made again from the flow (a set's id depends on the set only)."""
        if self._maker is None:
            flow = self.folder.flow
            candidates, _ = edits.candidates(flow)
            factors, _ = _factors(candidates, hypothesis_edits(flow))
            self._maker = _Arms(flow, Arm(id="base", configuration=copy.deepcopy(flow.configuration), edits=()),
                                factors, {a.id for a in candidates})
        return self._maker

    def size(self, plan: Plan) -> _Size:
        """A new check's size at the current spreads."""
        if self._size is None:
            spreads = _spreads(self.folder.flow, self.runs, self.quality, self.fit)
            check_pairs = margin_table(margins=(plan.margin,), s_run=spreads["s_run"], s_judge=spreads["s_judge"],
                                       k=spreads["k"])[0]["check_pairs"]
            self._size = _Size(check_pairs, *_run_cost(self.runs, sizing=True), spreads)
        return self._size


def _placed(arms: list[Arm], arm: Arm, plan_id: str) -> Arm:
    """The plan's own arm with this id, which must have the same configuration id, else `arm` added."""
    have = next((a for a in arms if a.id == arm.id), None)
    if have is None:
        arms.append(arm)
        return arm
    if (have.configuration or {}).get("id") != (arm.configuration or {}).get("id"):
        raise ValueError(f"plan {plan_id}: arm {arm.id!r} is not the configuration its edits give; run loopmath flow "
                         "plan")
    return have


def _state(plan: Plan, exp: Experiment, recorded: set[str]) -> str:
    """planned, running or done, from the recorded runs (see `advance`)."""
    current = exp.state or "planned"
    keys = pairs(plan, exp)
    if current == "done" or not keys:
        return current
    ids = [run_id for key, _, _ in keys for run_id in run_ids(key)]
    seen = sum(1 for run_id in ids if run_id in recorded)
    if seen == 0:
        return "planned"
    if exp.kind == "screen" and seen == len(ids):
        return "done"
    return "running"


def _hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
