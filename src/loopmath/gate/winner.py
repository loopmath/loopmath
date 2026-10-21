"""`gate winner`: how much of a search loop's winning lead is real, and what to buy before adopting it.

A loop scores N candidates and keeps the best. The best of N noisy scores is high partly because its noise
was high, so the winner's lead over its parent overstates its gain. `assess` shrinks the lead with the
hierarchical prior of `winner_model`, learnt from all N scores (unreported ones enter as censored). It reports
the winner's gain with 80% and 95% ranges and P(beats parent). The action comes from `gate.decide` on the
confirmatory runs, never from the search's own scores. The projection says what the gate would do from here
if the posterior is right.

Units. One evaluation is one complete pass of one arm over the evaluation's tasks, and its score is the task
mean. `spread` is the SD of one evaluation across reruns: same tasks, same scorer. From harness-runs rows, a
candidate's score is the mean of its task means over the task set every arm ran, and its standard error is
sqrt(sum_t sigma_t^2 / r_t) / T, with sigma_t the replicate SD on task t, pooled over the arms. The task set
is fixed: task-sampling variance is not in the estimand.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import inputs
from .core import GatePolicy, decide
from .winner_model import DEFAULT_POINTS, Data, Mixture, combine, fit

SCHEMA = "loopmath.gate-winner/1"
SCORES_SCHEMA = "loopmath.gate-winner-scores/0.1"
SCALES = ("linear", "log", "fraction")
PARENT_ROLES = ("parent", "seed", "base")
RANGE_POINTS = 7  # spreads the headline integrates over, log-uniform across a stated range
SHOWN_POINTS = 5  # spreads `by_spread` shows
INFEASIBLE_MAX = 0.02  # above this share of infeasible bounded draws, the projection is withheld
VALUE_STOP = True  # the next pair's value stops the gate, as `loopmath gate` has it by default
PRECISE_PARENT = 0.5  # a parent se at most this share of the median candidate se is called precise
HEAVY_MIN_N = 32  # scored candidates needed before the tails of their scores are read


class SpreadNeeded(ValueError):
    """No standard error and no spread: the command asks for --spread."""


@dataclass(frozen=True)
class Score:
    """One scored arm. `se` is the standard error of `score`; `sd` the SD of one evaluation (both optional:
    a spread fills them). `reps` is the number of evaluations averaged into the score."""

    id: str
    score: float
    se: float | None = None
    reps: int = 1
    order: Any = None
    sd: float | None = None


@dataclass(frozen=True)
class Censored:
    """`count` candidates whose scores are unknown, each no better than the bound: `at_most` when higher is
    better, `at_least` when lower is."""

    count: int
    at_most: float | None = None
    at_least: float | None = None
    se: float | None = None
    reps: int = 1
    order: Any = None


@dataclass
class Search:
    parent: Score | None
    candidates: list[Score]
    censored: list[Censored] = field(default_factory=list)
    better: str = "higher"
    scale: str = "linear"
    spread: Any = None  # one evaluation's SD, or (lo, hi); from the file or the runs
    spread_source: str | None = None  # given | runs | range
    kind: str = "scores"  # scores | runs (then confirmatory rows are task runs)
    costs: dict = field(default_factory=dict)  # arm id -> mean USD per run, from runs input
    source: dict | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class WinnerVerdict:
    winner: str
    parent: str
    better: str
    scale: str
    n_candidates: dict
    observed_lead: float
    gain: dict
    p_better: float
    p_above_floor: float | None
    floor: float | None
    shrinkage: float | None
    gain_confirmed: dict | None
    spread: dict
    by_spread: list
    prior: dict
    gate: dict
    projection: dict | None
    buy: dict
    reasons: list
    inputs: dict
    source: dict | None

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "winner": self.winner,
            "parent": self.parent,
            "better": self.better,
            "scale": self.scale,
            "n_candidates": self.n_candidates,
            "observed_lead": self.observed_lead,
            "gain": self.gain,
            "p_better": self.p_better,
            "p_above_floor": self.p_above_floor,
            "floor": self.floor,
            "shrinkage": self.shrinkage,
            "gain_confirmed": self.gain_confirmed,
            "spread": self.spread,
            "by_spread": self.by_spread,
            "prior": self.prior,
            "gate": self.gate,
            "projection": self.projection,
            "buy": self.buy,
            "reasons": self.reasons,
            "inputs": self.inputs,
            "source": self.source,
        }


def _r(x: float | None, nd: int = 6) -> float | None:
    return None if x is None else round(float(x), nd)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# Bounded draws: a Beta with the mean and SD asked for, never a clipped normal (clipping moves the mean).


def beta_params(mean: float, sd: float) -> tuple[float, float]:
    """(alpha, beta) of the Beta with this mean and SD; ValueError when no Beta has them."""
    if not (_finite(mean) and _finite(sd)) or not 0 < mean < 1 or sd <= 0 or sd * sd >= mean * (1 - mean):
        raise ValueError(f"no feasible Beta has mean {mean!r} and SD {sd!r}: the mean must be inside (0, 1) and "
                         "the variance under mean x (1 - mean)")
    k = mean * (1 - mean) / (sd * sd) - 1
    return mean * k, (1 - mean) * k


def draw_evaluations(mean: float, sd: float, scale: str, rng: np.random.Generator, size: int) -> np.ndarray:
    """Fresh evaluation scores with this mean and SD: normal on an open scale, Beta on fractions."""
    if scale == "fraction":
        a, b = beta_params(mean, sd)
        return rng.beta(a, b, size)
    return rng.normal(mean, sd, size)


# Inputs.


def _check_search(search: Search) -> None:
    if search.parent is None:
        raise ValueError("the search has no parent: name the score the loop started from (role parent)")
    if not search.candidates:
        raise ValueError("the search has no scored candidate")
    if search.better not in ("higher", "lower"):
        raise ValueError(f"better must be higher or lower (got {search.better!r})")
    if search.scale not in SCALES:
        raise ValueError(f"scale must be one of {', '.join(SCALES)} (got {search.scale!r})")
    seen: set[str] = set()
    for s in [search.parent, *search.candidates]:
        if not _finite(s.score):
            raise ValueError(f"the score of {s.id} is not a finite number ({s.score!r})")
        if s.se is not None and not (_finite(s.se) and s.se > 0):
            raise ValueError(f"the se of {s.id} must be a number above 0 (got {s.se!r})")
        if not isinstance(s.reps, int) or isinstance(s.reps, bool) or s.reps < 1:
            raise ValueError(f"the reps of {s.id} must be a whole number, 1 or more (got {s.reps!r})")
        if s.id in seen:
            raise ValueError(f"the id {s.id} appears twice")
        seen.add(s.id)
    bound_name = "at_most" if search.better == "higher" else "at_least"
    for g in search.censored:
        if not isinstance(g.count, int) or isinstance(g.count, bool) or g.count < 1:
            raise ValueError(f"a censored group's count must be a whole number, 1 or more (got {g.count!r})")
        bound = g.at_most if search.better == "higher" else g.at_least
        if not _finite(bound):
            raise ValueError(f"a censored group needs a finite {bound_name} when {search.better} is better "
                             f"(got {bound!r})")
        if g.se is not None and not (_finite(g.se) and g.se > 0):
            raise ValueError(f"a censored group's se must be a number above 0 (got {g.se!r})")
        if not isinstance(g.reps, int) or isinstance(g.reps, bool) or g.reps < 1:
            raise ValueError(f"a censored group's reps must be a whole number, 1 or more (got {g.reps!r})")


def _check_spread(spread: Any) -> Any:
    if spread is None:
        return None
    if isinstance(spread, (list, tuple)):
        if len(spread) != 2 or not all(_finite(x) for x in spread) or not 0 < spread[0] < spread[1]:
            raise ValueError(f"a spread range must be two numbers, 0 < lo < hi (got {spread!r})")
        return (float(spread[0]), float(spread[1]))
    if not _finite(spread) or spread <= 0:
        raise ValueError(f"the spread must be a number above 0 (got {spread!r})")
    return float(spread)


def _num(cell: str, what: str, where: str, *, whole: bool = False, required: bool = True):
    cell = (cell or "").strip()
    if not cell:
        if required:
            raise ValueError(f"{where}: {what}: a number is required")
        return None
    try:
        x = int(cell) if whole else float(cell)
    except ValueError:
        raise ValueError(f"{where}: {what}: a {'whole number' if whole else 'number'} is required "
                         f"(got {cell!r})") from None
    if not whole and not math.isfinite(x):
        raise ValueError(f"{where}: {what}: a finite number is required (got {cell!r})")
    return x


def _from_csv(text: str, name: str) -> Search:
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
    idcol = cols.get("id") or cols.get("candidate")
    if idcol is None or "score" not in cols:
        raise ValueError(f"{name}: the header needs an id (or candidate) column and a score column")
    get = lambda row, k: (row.get(cols[k]) if k in cols else None) or ""
    parent, cands, cens = None, [], []
    for i, row in enumerate(reader, start=2):
        where = f"{name}:{i}"
        rid = (row.get(idcol) or "").strip()
        role = get(row, "role").strip().lower()
        se = _num(get(row, "se"), "se", where, required=False)
        reps = _num(get(row, "reps"), "reps", where, whole=True, required=False)
        reps = 1 if reps is None else reps  # an empty cell is 1; a 0 is refused by name below
        order = get(row, "order").strip() or None
        if not get(row, "score").strip() and (get(row, "at_most").strip() or get(row, "at_least").strip()):
            cens.append(Censored(count=_num(get(row, "count") or "1", "count", where, whole=True),
                                 at_most=_num(get(row, "at_most"), "at_most", where, required=False),
                                 at_least=_num(get(row, "at_least"), "at_least", where, required=False),
                                 se=se, reps=reps, order=order))
            continue
        if not rid:
            raise ValueError(f"{where}: id: a non-empty id is required")
        s = Score(rid, _num(get(row, "score"), "score", where), se=se, reps=reps, order=order)
        if role in PARENT_ROLES:
            if parent is not None:
                raise ValueError(f"{where}: a second parent row ({rid}); the first was {parent.id}")
            parent = s
        elif role and role != "candidate":
            raise ValueError(f"{where}: role must be parent or candidate, or empty (got {role!r})")
        else:
            cands.append(s)
    if parent is None:
        raise ValueError(f"{name}: no parent row: mark the score the loop started from with role parent")
    return Search(parent=parent, candidates=cands, censored=cens)


def _score_obj(d: Any, where: str) -> Score:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: an object with id and score is required")
    if not isinstance(d.get("id"), str) or not d["id"]:
        raise ValueError(f"{where}: id: a non-empty string is required")
    if not _finite(d.get("score")):
        raise ValueError(f"{where}: score: a finite number is required (got {d.get('score')!r})")
    return Score(d["id"], float(d["score"]), se=d.get("se"), reps=d.get("reps", 1), order=d.get("order"))


def _from_json(text: str, name: str) -> Search:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: not JSON: {exc}") from None
    if not isinstance(doc, dict) or doc.get("schema") != SCORES_SCHEMA:
        got = doc.get("schema") if isinstance(doc, dict) else None
        raise ValueError(f"{name}: schema must be {SCORES_SCHEMA} (got {got!r})")
    if "parent" not in doc:
        raise ValueError(f"{name}: no parent: give the score the loop started from as parent")
    parent = _score_obj(doc["parent"], f"{name}: parent")
    cands = [_score_obj(c, f"{name}: candidates[{i}]") for i, c in enumerate(doc.get("candidates") or [])]
    cens = []
    for i, g in enumerate(doc.get("censored") or []):
        if not isinstance(g, dict):
            raise ValueError(f"{name}: censored[{i}]: an object is required")
        cens.append(Censored(count=g.get("count", 1), at_most=g.get("at_most"), at_least=g.get("at_least"),
                             se=g.get("se"), reps=g.get("reps", 1), order=g.get("order")))
    spread = doc.get("spread")
    known = {"schema", "better", "scale", "spread", "parent", "candidates", "censored", "source"}
    return Search(parent=parent, candidates=cands, censored=cens, better=doc.get("better", "higher"),
                  scale=doc.get("scale", "linear"), spread=tuple(spread) if isinstance(spread, list) else spread,
                  spread_source=None if spread is None else "file", source=doc.get("source"),
                  extra={k: v for k, v in doc.items() if k not in known})


def read_scores(path: str | Path) -> Search:
    """A search from a CSV (a header row: id, score, and optional se, reps, order, role, at_most, at_least,
    count) or a `loopmath.gate-winner-scores/0.1` JSON file."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    search = _from_json(text, p.name) if text.lstrip().startswith("{") else _from_csv(text, p.name)
    _check_search(search)
    return search


def _arm_key(x: Any) -> tuple[str, str] | None:
    d = inputs.digest_of(x)
    if d:
        return ("digest", d)
    a = inputs.arm_of(x)
    if a:
        return ("arm", a)
    c = inputs.config_of(x)
    return ("config", c) if c else None


def _arm_name(key: tuple[str, str]) -> str:
    kind, value = key
    if kind == "digest":
        return value.split(":")[-1][:12]
    return value


def search_from_runs(rows: Sequence[Any], *, parent: str = "arm:base", rule: Any = "score:judge",
                     config: Any = None, spread: float | None = None) -> Search:
    """A search from harness-runs rows or run documents: each arm (harness digest, else arm label, else
    configuration) is a candidate; `parent` selects the parent arm, in `gate`'s selector forms.

    An arm's score is the mean of its task means over the task set every arm ran. Its standard error is
    sqrt(sum_t sigma_t^2 / r_t) / T, with sigma_t^2 the replicate variance on task t pooled over the arms with
    two or more runs there (a task with none takes the pool over all tasks), or `spread` squared when given
    (one run's SD on one task). One evaluation's SD is sqrt(sum_t sigma_t^2) / T."""
    from ..compare import observe, score_rule

    sr = score_rule(rule)
    if sr is None:
        raise ValueError(f"gate winner reads scores: the rule must be score:NAME (got {rule!r})")
    lower = sr[1]
    spread = _check_spread(spread)
    if isinstance(spread, tuple):
        raise ValueError("with runs, the spread is one number: one run's SD on one task")
    rows = list(rows)
    parent_key = inputs.resolve(rows, parent)
    parent_rows = inputs.select(rows, parent)
    if not parent_rows:
        raise ValueError(f"the parent {parent} matches no run; the runs have {inputs.present(rows)}")
    ids = {id(x) for x in parent_rows}
    groups: dict[tuple[str, str], list] = {parent_key: parent_rows}
    for x in rows:
        if id(x) in ids:
            continue
        key = _arm_key(x)
        if key is None:
            raise ValueError(f"run {inputs.name_of(x)} has no harness digest, arm or configuration to group by")
        groups.setdefault(key, []).append(x)
    scales: set[str] = set()
    cells: dict[tuple[str, str], dict[str, list[float]]] = {}
    costs: dict[str, float | None] = {}
    for key, xs in groups.items():
        by_task: dict[str, list[float]] = {}
        usd = []
        for x in xs:
            o = observe(x, rule, config=config).outcome
            if o is None:
                continue
            by_task.setdefault(str(inputs.task_of(x)), []).append(float(o))
            scales.add(inputs.scale_of(x, rule) or "linear")
            usd.append(inputs.usd_of(x))
        if not by_task:
            raise ValueError(f"arm {_arm_name(key)} has no run scored under {rule}")
        cells[key] = by_task
        costs[_arm_name(key)] = (sum(usd) / len(usd)) if usd and all(u is not None for u in usd) else None
    if len(scales) > 1:
        raise ValueError(f"the runs mix score scales ({', '.join(sorted(scales))}); give runs on one scale")
    tasks = sorted(set().union(*[set(c) for c in cells.values()]))
    missing = [(key, sorted(set(tasks) - set(c))) for key, c in cells.items() if set(tasks) - set(c)]
    if missing:
        shown = "; ".join(f"{_arm_name(k)} has no scored run on {', '.join(m[:6])}{' and more' if len(m) > 6 else ''}"
                          for k, m in missing[:6])
        raise ValueError(f"every arm must run the same tasks, or their scores differ by task difficulty: {shown}")
    T = len(tasks)
    if spread is not None:
        sigma2 = {t: spread * spread for t in tasks}
        source = "given"
    else:
        ss_all = df_all = 0.0
        sigma2 = {}
        for t in tasks:
            ss = df = 0.0
            for c in cells.values():
                xs = c[t]
                if len(xs) >= 2:
                    m = sum(xs) / len(xs)
                    ss += sum((v - m) ** 2 for v in xs)
                    df += len(xs) - 1
            if df:
                sigma2[t] = ss / df
            ss_all += ss
            df_all += df
        if not df_all:
            raise SpreadNeeded("no arm ran any task twice, so the replicate spread is unknown: give the spread "
                               "(one run's SD on one task)")
        for t in tasks:
            sigma2.setdefault(t, ss_all / df_all)
        source = "runs"
    sd_eval = math.sqrt(sum(sigma2.values())) / T
    scored = []
    for key, c in cells.items():
        y = sum(sum(c[t]) / len(c[t]) for t in tasks) / T
        se = math.sqrt(sum(sigma2[t] / len(c[t]) for t in tasks)) / T
        scored.append((key, Score(_arm_name(key), y, se=se, sd=sd_eval)))
    parent_score = scored[0][1]
    return Search(parent=parent_score, candidates=[s for _, s in scored[1:]],
                  better="lower" if lower else "higher", scale=scales.pop() if scales else "linear",
                  spread=sd_eval, spread_source=source, kind="runs", costs=costs)


# The posterior.


def _ses(search: Search, spread: float | None):
    """Each score's standard error and one evaluation's SD under this spread (None: their own se)."""

    def one(s):
        se = s.se if s.se is not None else spread / math.sqrt(s.reps)
        sd = s.sd if s.sd is not None else (s.se * math.sqrt(s.reps) if s.se is not None else spread)
        return se, sd

    return one


def _data(search: Search, spread: float | None, w: int) -> tuple[Data, float, float]:
    one = _ses(search, spread)
    sign = 1.0 if search.better == "higher" else -1.0
    y = np.array([sign * c.score for c in search.candidates])
    s = np.array([one(c)[0] for c in search.candidates])
    se_p, sd_p = one(search.parent)
    bound = np.array([sign * (g.at_most if search.better == "higher" else g.at_least) for g in search.censored])
    cs = np.array([g.se if g.se is not None else spread / math.sqrt(g.reps) for g in search.censored])
    count = np.array([g.count for g in search.censored], dtype=float)
    d = Data(y_p=sign * search.parent.score, s_p=se_p, y=y, s=s, w=w, bound=bound, count=count, s_c=cs)
    return d, sd_p, one(search.candidates[w])[1]


def _shape_reasons(d: Data) -> list[str]:
    """Regimes where the ranges ran narrow in the recovery sims, read from the scores alone."""
    out = []
    s_med = float(np.median(d.s))
    if d.s_p <= PRECISE_PARENT * s_med:
        out.append(f"the parent's se is {d.s_p / s_med:.2g} of a typical candidate's, so the range rests on the "
                   "model's shape for the candidates: in simulated searches with a parent this precise and 16 or "
                   "more candidates, 80% ranges covered 80% to 90% when the candidates' true scores were normal, "
                   "equal or rising along the search, but 62% to 84% when a few were better than the rest, most "
                   "were worse, or the scores had heavy tails")
    n = len(d.y)
    if n >= HEAVY_MIN_N and not len(d.count):
        t2 = max(float(np.var(d.y)) - float(np.mean(d.s ** 2)), 0.0)  # the candidates' spread, by moments
        z = (d.y - d.y.mean()) / np.sqrt(t2 + d.s ** 2)
        kurt = float(np.mean(z ** 4) / np.mean(z ** 2) ** 2 - 3.0)
        chance = 2.0 * math.sqrt(24.0 / n)  # about 2 SE of the excess kurtosis of n normal scores
        if kurt > chance:
            out.append(f"the candidates' scores have heavier tails than a normal spread (excess kurtosis "
                       f"{kurt:.2g}, against {chance:.2g} by chance): in simulated searches with t3 deviations "
                       "where this showed, 80% ranges covered 64% to 71% at 256 candidates and 76% to 79% at 64; "
                       "it also shows when a few candidates are far better than the rest, where they covered 73% "
                       "to 86%")
    return out


def _interval(m: Mixture) -> dict:
    q = m.quantiles([0.025, 0.1, 0.5, 0.9, 0.975])
    return {"mean": _r(m.mean), "median": _r(q[2]), "lo": _r(q[1]), "hi": _r(q[3]),
            "bands": {"80": [_r(q[1]), _r(q[3])], "95": [_r(q[0]), _r(q[4])]}}


def _confirmed_diffs(base: Sequence[Any], cand: Sequence[Any], rule: Any) -> list[float]:
    """The confirmatory pairs' differences, candidate minus base in the score's own direction, paired the way
    compare pairs them (pair key, else task in start order); a pair with an unscored run is left out."""
    from ..compare import observe, pair_up

    pairs, _, _ = pair_up([observe(x, rule) for x in base], [observe(x, rule) for x in cand])
    return [float(c.outcome) - float(b.outcome) for _, b, c in pairs
            if b.outcome is not None and c.outcome is not None]


def _update(m: Mixture, diffs: list[float], sd_pair: np.ndarray) -> Mixture:
    """D's posterior after the confirmatory pairs, each normal part updated in closed form by the pairs' mean,
    taken as normal with variance sd_pair^2 / n (sd_pair: one pair's difference SD under that part)."""
    n = len(diffs)
    dbar = float(np.mean(diffs))
    lv = sd_pair ** 2 / n
    tot = m.v + lv
    logw = np.log(m.w) - 0.5 * (np.log(tot) + (dbar - m.m) ** 2 / tot)
    w = np.exp(logw - logw.max())
    prec = 1.0 / m.v + 1.0 / lv
    return Mixture(w=w / w.sum(), m=(m.m / m.v + dbar / lv) / prec, v=1.0 / prec, log_evidence=m.log_evidence,
                   hyper=m.hyper)


# The projection: the real gate, from its actual state, on fresh evaluations drawn with D from the posterior.

ASSUMES = (
    "each projected pair is one fresh evaluation per arm, independent, with the spread as its SD",
    "the base arm's mean is held at the parent's posterior mean; its uncertainty and its dependence with the "
    "gain are left out",
    "held confirmatory pairs update the gain as a normal mean with the spread's variance",
    "a projected run costs its arm's mean over the held confirmatory runs (else over its search runs), when "
    "those are priced",
    "with a spread range, each projected evaluation's SD is the range's spreads averaged by their weights, not "
    "a spread drawn with the gain",
)


def _proj_row(arm: str, x: float, j: int, name: str, scale: str, better: str, usd: float | None,
              keyed: bool) -> dict:
    row = {"schema": inputs.ROWS_SCHEMA, "task_id": f"projected-{j}", "arm": arm,
           "outcome": {"name": name, "score": float(x), "judge": "projection", "scale": scale, "better": better}}
    if keyed:  # the held runs carry pair keys, and compare takes all or none
        row["pair"] = f"projected-{j}"
    if usd is not None:
        row["cost"] = {"usd": usd}
    return row


def _project(post: Mixture, search: Search, policy: GatePolicy, base: list, cand: list, held: int,
             sd_b: float, sd_c: float, draws: int, seed: int, costs: tuple[float | None, float | None]) -> dict:
    from ..compare import observe, score_rule

    name, lower = score_rule(policy.rule)
    better = "lower" if lower else "higher"
    usd_b, usd_c = costs
    held_rows = [*base, *cand]
    scale = (inputs.scale_of(held_rows[0], policy.rule) if held_rows else None) or search.scale
    keyed = any(observe(x, policy.rule).pair for x in held_rows)
    rng = np.random.default_rng(seed)
    gains = post.sample(rng, draws)
    sign = 1.0 if search.better == "higher" else -1.0
    centre = sign * post.hyper["theta_p"]  # the parent's posterior mean, in the scores' own direction
    floor = policy.floor or 0.0
    adopt, more, real, infeasible = [], [], [], 0
    for g in gains:
        mb, mc = centre, centre + sign * g
        if scale == "fraction":
            try:
                beta_params(mb, sd_b)
                beta_params(mc, sd_c)
            except ValueError:
                infeasible += 1
                continue
        rows_b, rows_c, j = list(base), list(cand), 0
        d = decide(rows_b, rows_c, policy)
        while d.action == "replicate" and d.n_more > 0 and j < policy.r_max:
            xb = draw_evaluations(mb, sd_b, scale, rng, d.n_more)
            xc = draw_evaluations(mc, sd_c, scale, rng, d.n_more)
            for xbi, xci in zip(xb, xc):
                rows_b.append(_proj_row("base", xbi, j, name, scale, better, usd_b, keyed))
                rows_c.append(_proj_row("candidate", xci, j, name, scale, better, usd_c, keyed))
                j += 1
            d = decide(rows_b, rows_c, policy)
        adopt.append(d.action == "adopt")
        more.append(j)
        real.append(g > floor)
    share = infeasible / draws
    if share > INFEASIBLE_MAX:
        return {"withheld": True, "draws": draws, "infeasible_share": _r(share, 4), "held_pairs": held,
                "reason": f"{share:.0%} of the drawn gains put a mean at or past the bound of the fraction scale, "
                          "or ask for a spread wider than a fraction allows there; the gate's action stands"}
    adopt_a, more_a, real_a = np.array(adopt), np.array(more), np.array(real)
    out = {
        "withheld": False,
        "draws": draws,
        "held_pairs": held,
        "p_adopt": _r(adopt_a.mean(), 4),
        "p_adopt_if_real": _r(adopt_a[real_a].mean(), 4) if real_a.any() else None,
        "p_adopt_if_not": _r(adopt_a[~real_a].mean(), 4) if (~real_a).any() else None,
        "mean_more": _r(more_a.mean(), 4),
        "p80_more": int(np.percentile(more_a, 80)),
        "infeasible_share": _r(share, 4),
        "conditioned_on": "feasible draws" if infeasible else None,
        "assumes": list(ASSUMES),
    }
    if usd_b is not None and usd_c is not None:
        out["usd_more"] = _r(more_a.mean() * (usd_b + usd_c), 4)
    return out


def _mean_usd(rows: Sequence[Any]) -> float | None:
    usd = [inputs.usd_of(x) for x in rows]
    return sum(usd) / len(usd) if usd and all(u is not None for u in usd) else None


def default_policy(**kw: Any) -> GatePolicy:
    """`GatePolicy(**kw)` with the value stop stated, not left to the gate's default, wherever the gate has the
    setting (a gate without it always stops on value): the action never rides on that default."""
    if any(f.name == "value_stop" for f in fields(GatePolicy)):
        kw.setdefault("value_stop", VALUE_STOP)
    return GatePolicy(**kw)


def assess(search: Search, *, spread: Any = None, winner: str | None = None, policy: GatePolicy | None = None,
           confirm: tuple[Sequence[Any], Sequence[Any]] | None = None, prior: str = "sparse", draws: int = 200,
           seed: int = 0, points: tuple[int, int] = DEFAULT_POINTS) -> WinnerVerdict:
    """The winner's selection-adjusted gain, the gate's action on the confirmatory runs, and the projection.

    `spread` is one evaluation's SD across reruns, or (lo, hi) to integrate over a range; it fills the scores
    with no se of their own. `confirm` is (base runs, candidate runs) for `gate.decide`: with scores input,
    each confirmatory run is one complete evaluation. `policy` defaults to `default_policy()`. `draws` is the
    projection's draws (0: none)."""
    from ..compare import score_rule

    policy = default_policy() if policy is None else policy

    _check_search(search)
    stated = spread is not None
    spread = _check_spread(spread if stated else search.spread)
    missing = [s.id for s in [search.parent, *search.candidates] if s.se is None]
    if spread is None and (missing or any(g.se is None for g in search.censored)):
        raise SpreadNeeded(f"{len(missing) or 'some'} scores have no standard error ({', '.join(missing[:4])}"
                           f"{' and more' if len(missing) > 4 else ''}): give the replicate spread, one "
                           "evaluation's SD across reruns, or a range of spreads")
    if prior not in ("auto", "ridge", "sparse"):
        raise ValueError(f"prior must be auto, ridge or sparse (got {prior!r})")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 0:
        raise ValueError(f"draws must be a whole number, 0 or more (got {draws!r})")
    sign = 1.0 if search.better == "higher" else -1.0
    ys = [sign * c.score for c in search.candidates]
    ids = [c.id for c in search.candidates]
    if winner is None:
        w = int(np.argmax(ys))
        bound_name = "at_most" if search.better == "higher" else "at_least"
        for g in search.censored:
            b = g.at_most if search.better == "higher" else g.at_least
            if sign * b > ys[w] + 1e-12:
                raise ValueError(f"a censored group's {bound_name} {b:g} is better than the best scored "
                                 f"candidate ({search.candidates[w].score:g}); the best must be scored")
    elif winner in ids:
        w = ids.index(winner)
    else:
        raise ValueError(f"winner {winner} is not a scored candidate; the candidates: {', '.join(ids[:8])}"
                         f"{' and more' if len(ids) > 8 else ''}")
    if search.better == "lower" and policy.rule == GatePolicy().rule:
        policy = replace(policy, rule=f"{policy.rule}:lower")  # the gate reads the scores the way the search does

    reasons: list[str] = []
    by_spread: list[dict] = []
    if isinstance(spread, tuple):
        lo, hi = spread
        parts, sds = [], []
        for x in np.geomspace(lo, hi, RANGE_POINTS):
            d, sd_p, sd_w = _data(search, float(x), w)
            parts.append(fit(d, scale=search.scale, view=prior, points=points))
            sds.append((sd_p, sd_w))
        post, a = combine(parts, np.zeros(len(parts)))
        sd_pair = np.concatenate([np.full(len(p.w), math.hypot(*sd)) for p, sd in zip(parts, sds)])
        sd_b = float(sum(ai * sd[0] for ai, sd in zip(a, sds)))
        sd_c = float(sum(ai * sd[1] for ai, sd in zip(a, sds)))
        for x in np.geomspace(lo, hi, SHOWN_POINTS):
            d, _, _ = _data(search, float(x), w)
            m = fit(d, scale=search.scale, view=prior, points=points)
            by_spread.append({"spread": _r(x), "gain": _interval(m), "p_better": _r(m.sf(0.0), 4)})
        spread_info = {"range": [lo, hi], "unit": "evaluation", "source": "range",
                       "weights": [_r(x, 4) for x in a]}
        reasons.append(f"the spread is an assumed range, {lo:g} to {hi:g}, not a measured one: the gain integrates "
                       "over it, weighing each spread by how well it fits the scores, and by_spread shows each")
    else:
        d, sd_b, sd_c = _data(search, spread, w)
        post = fit(d, scale=search.scale, view=prior, points=points)
        sd_pair = np.full(len(post.w), math.hypot(sd_b, sd_c))
        source = ("runs" if not stated and search.spread_source == "runs" else
                  "given" if spread is not None else "se")
        spread_info = {"value": spread, "unit": "evaluation", "source": source}

    lead = sign * (search.candidates[w].score - search.parent.score)
    n_scored, n_cens = len(search.candidates), int(sum(g.count for g in search.censored))
    total = n_scored + n_cens
    if winner is None:
        reasons.insert(0, f"the lead of {lead:+.4g} is the best of {total} candidates' scores; the model keeps "
                          f"{post.mean:+.4g} of it")
    else:
        reasons.insert(0, f"the named winner {search.candidates[w].id} leads by {lead:+.4g}; the model keeps "
                          f"{post.mean:+.4g} of it")
    if n_cens:
        reasons.append(f"{n_cens} candidates are known only as no better than a bound (censored)")
    if total < 4:
        reasons.append("fewer than 4 candidates: the shrinkage leans on the prior")
    mid = math.sqrt(spread[0] * spread[1]) if isinstance(spread, tuple) else spread
    reasons += _shape_reasons(_data(search, mid, w)[0])

    # The gate, on the confirmatory runs only: the search's own scores never adopt.
    base, cand = (list(confirm[0]), list(confirm[1])) if confirm else ([], [])
    gate = decide(base, cand, policy).to_json()
    held = int((gate.get("comparison") or {}).get("n_pairs") or 0)
    executed = max(len(base), len(cand))
    stopped = gate["action"] in ("adopt", "reject")
    scored_rule = score_rule(policy.rule) is not None
    runs_confirm = search.kind == "runs" and bool(base or cand)

    gain_confirmed, post_c = None, post
    if (base or cand) and scored_rule and not runs_confirm:
        diffs = [sign * x for x in _confirmed_diffs(base, cand, policy.rule)]
        if diffs:
            post_c = _update(post, diffs, sd_pair)
            gain_confirmed = {**_interval(post_c), "pairs": len(diffs), "p_better": _r(post_c.sf(0.0), 4)}

    projection = None
    if stopped:
        projection = {"withheld": False, "stopped": gate["action"], "draws": 0, "held_pairs": held,
                      "p_adopt": 1.0 if gate["action"] == "adopt" else 0.0, "p_adopt_if_real": None,
                      "p_adopt_if_not": None, "mean_more": 0, "p80_more": 0, "infeasible_share": 0.0}
    elif draws:
        why = None
        # each arm's future runs cost what its held runs did (else its search runs): the gate prices pairs and
        # stops on value from these, with or without a budget, so the projected gate must see them too
        costs: tuple[float | None, float | None] = (_mean_usd(base), _mean_usd(cand)) if base and cand else (
            search.costs.get(search.parent.id), search.costs.get(search.candidates[w].id))
        if not scored_rule:
            why = "the projection draws scores, and the rule is an acceptance rule"
        elif runs_confirm:
            why = ("the confirmatory runs are task runs and the projection draws whole evaluations; the gate's "
                   "action stands")
        elif (policy.budget_usd is not None or policy.max_cost_ratio is not None) and None in costs:
            why = "a budget or cost ratio needs each future run's cost, and an arm has no known cost"
        if why:
            projection = {"withheld": True, "draws": draws, "held_pairs": held, "reason": why}
        else:
            projection = _project(post_c, search, policy, base, cand, held, sd_b, sd_c, draws, seed, costs)
        if projection["withheld"]:
            reasons.append(f"projection withheld: {projection['reason']}")

    buy = {"now": 0 if stopped else int(gate["n_more"]),
           "at_least": 0 if stopped else max(policy.r_min - held, 0),
           "expected": projection["mean_more"] if projection and not projection["withheld"] else None,
           "at_most": 0 if stopped else max(policy.r_max - executed, 0)}
    one = _ses(search, spread[0] if isinstance(spread, tuple) else spread)
    cw = search.candidates[w]
    floor = policy.floor
    return WinnerVerdict(
        winner=cw.id, parent=search.parent.id, better=search.better, scale=search.scale,
        n_candidates={"scored": n_scored, "censored": n_cens, "total": total},
        observed_lead=float(lead), gain=_interval(post), p_better=_r(post.sf(0.0), 4),
        p_above_floor=_r(post.sf(floor), 4) if floor is not None else None, floor=floor,
        shrinkage=_r(1.0 - post.mean / lead, 4) if lead > 0 else None, gain_confirmed=gain_confirmed,
        spread=spread_info, by_spread=by_spread,
        prior={"view": prior, **{k: (_r(sign * v if k in ("mu", "theta_p") else v) if isinstance(v, float) else v)
                                 for k, v in post.hyper.items()}},  # mu and theta_p in the scores' direction
        gate=gate, projection=projection, buy=buy, reasons=reasons,
        inputs={"parent": {"id": search.parent.id, "score": search.parent.score, "se": _r(one(search.parent)[0], 9)},
                "winner": {"id": cw.id, "score": cw.score, "se": _r(one(cw)[0], 9)}},
        source=search.source,
    )
