"""Workflow search (spec 05 section 1a): every shape, width, round limit and per-piece setting, without
listing the configurations.

For a fixed shape case (workflow structure, widths, `K_max`) the belief's run row is a sum of per-piece
terms, and the expected run cost is a sum over pieces once the setting of the piece a repair loop's
gates judge is fixed (spec 04). The search uses that structure and is exact for the model:

1. Front. Per case, the exact Pareto front of (expected run cost, posterior mean u of the success or
   score predictor), by a dynamic program over pieces, conditioned on the judged piece's setting.
2. Thompson set. Per posterior draw, the exact optimum for the default pick (lowest cost per accepted
   result) and for each curve level, with a branch and bound over (case, draw).
3. The recommender rescores the front and the most frequent winners with `predict_many`, then
   `polish` predicts the pick's one-piece neighbours until the pick stops moving.

Copies. Two or more width-1 pieces of one role with the same inputs, none reachable from another, are
copies. With one setting on every copy they are the width form's workflow, so only the width form is
searched; other mixes are listed once, strongest first, and kept only when their copies differ (in
model with `copies="model"`, in any setting with `"setting"`).

The tables read the fit's rows the way `predict_many` does (`FitState._task_parts`, `_cached`,
`heads`). Each row is split into a case part and a setting part by calling the belief's own row
builders with the piece's setting varied: the terms that change are the setting part. The split keeps
the unseen variances additive by moving any node found in both parts into the setting part.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.special import expit, ndtr

from ..belief.design import Structure, _piece_reach, cost_rest, gate_rest, run_rest, structure
from ..belief.forest import canonical_model_id
from ..belief.state import MEAN_NODES, MIN_SCORE_SWITCH, N_DRAWS, PREDICT_SOURCE, _GW, _GX, transform
from ..types import AcceptanceRule, Configuration, Piece, Prediction, Setting, Task, Workflow
from ..workflows.ids import make_config
from ..workflows.shapes import (MAX_ROUNDS, MAX_WIDTH, MIDDLES, MIN_ROUNDS, MIN_WIDTH, ShapeParams, build_shape,
                               shape_name)
from .curve import LEVELS

METHOD = "exact_front+thompson"
COPY_RULES = ("model", "setting")
EFFORT_RANK = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}
MAX_MIXES = 50_000  # a copy group with more mixes than this is not searched (its case is listed in stats)
TIE = 1e-12  # u values within this of each other are equal (float sums in another order)
OBJECTIVES = ("default",) + tuple(f"p{lv}" for lv in LEVELS)


class NotExact(ValueError):
    """A case outside the exact method: two repair loops, loop gates judging different pieces, a row
    term that couples two pieces' settings, or a copy group with too many mixes."""


# ---------------------------------------------------------------- settings and copies

def setting_key(s: Setting) -> tuple[str, str, str]:
    """What the belief reads of a setting: harness, canonical model and effort."""
    return (s.harness or "unknown", canonical_model_id(s.model), (s.effort or "default").lower())


def copies_differ(settings: Sequence[Setting], combo: Sequence[int], rule: str) -> bool:
    if rule == "model":
        return len({setting_key(settings[i])[1] for i in combo}) > 1
    return len(set(combo)) > 1


def copy_order(settings: Sequence[Setting]) -> list[int]:
    """Setting indices strongest first: models in the settings' order, then effort from the highest."""
    first: dict[str, int] = {}
    for s in settings:
        first.setdefault(setting_key(s)[1], len(first))
    return sorted(range(len(settings)), key=lambda i: (first[setting_key(settings[i])[1]],
                                                      -EFFORT_RANK.get(setting_key(settings[i])[2], -1), i))


def mix_count(n_settings: int, k: int) -> int:
    return math.comb(n_settings + k - 1, k)


def differing_mix_count(settings: Sequence[Setting], k: int, rule: str) -> int:
    """How many mixes `group_combos` lists, counted without listing them: every multiset of k settings less the
    ones whose copies do not differ (one setting k times, or under "model" one model's settings only)."""
    if rule == "model":
        per_model = Counter(setting_key(s)[1] for s in settings)
        return mix_count(len(settings), k) - sum(mix_count(n, k) for n in per_model.values())
    return mix_count(len(settings), k) - len(settings)


def group_combos(settings: Sequence[Setting], k: int, rule: str) -> list[tuple[int, ...]]:
    """A copy group's options: each mix once, in copy order, whose copies differ (the rest is the width form)."""
    order = copy_order(settings)
    out = []
    for t in itertools.combinations_with_replacement(range(len(order)), k):
        combo = tuple(order[j] for j in t)
        if copies_differ(settings, combo, rule):
            out.append(combo)
    return out


def canonical_mix(settings: Sequence[Setting], chosen: Sequence[Setting], rule: str) -> tuple[int, ...] | None:
    """The copy-ordered mix of `chosen` as setting indices, or None when it is not in the space (a setting
    outside it, or copies that do not differ)."""
    index = {setting_key(s): i for i, s in enumerate(settings)}
    try:
        idx = [index[setting_key(s)] for s in chosen]
    except KeyError:
        return None
    rank = {i: r for r, i in enumerate(copy_order(settings))}
    combo = tuple(sorted(idx, key=rank.__getitem__))
    return combo if copies_differ(settings, combo, rule) else None


def copy_groups(wf: Workflow, pieces: Iterable[str]) -> list[tuple[str, ...]]:
    """Copies among `pieces`: two or more width-1 pieces of one role with the same inputs, none reachable
    from another."""
    ins: dict[str, set] = {}
    for a, b in wf.edges:
        ins.setdefault(b, set()).add(a)
    reach = _piece_reach(wf)
    info = {p.id: p for p in wf.pieces}
    by: dict[tuple, list[str]] = {}
    for p in pieces:
        if max(1, int(info[p].width or 1)) == 1:
            by.setdefault((info[p].role, frozenset(ins.get(p, ()))), []).append(p)
    return [tuple(g) for g in by.values()
            if len(g) > 1 and not any(q in reach.get(p, ()) for p in g for q in g if q != p)]


def width_form(wf: Workflow, group: Sequence[str]) -> Workflow:
    """The workflow with a copy group as one piece of width k: the id without its copy suffix, the copies'
    edges and artifacts merged (`diff-implement-1` becomes `diff-implement`)."""
    first = group[0]
    stem = first.rsplit("-", 1)[0] if "-" in first and first.rsplit("-", 1)[1].isdigit() else first
    ids = {p.id for p in wf.pieces}
    new = stem if stem == first or stem not in ids else first

    def rename(name: str) -> str:
        if name in group:
            return new
        for g in group:
            if g in name:
                return name.replace(g, new)
        return name

    pieces: list[Piece] = []
    for p in wf.pieces:
        if p.id not in group:
            pieces.append(p)
        elif p.id == first:
            pieces.append(dataclasses.replace(p, id=new, width=len(group)))
    gates = tuple(dataclasses.replace(g, after=rename(g.after), on_fail=rename(g.on_fail) if g.on_fail else g.on_fail)
                  for g in wf.control.gates)
    extra = dict(wf.extra)
    kinds = extra.get("artifact_kinds")
    if isinstance(kinds, dict):
        extra["artifact_kinds"] = {rename(a): k for a, k in kinds.items()}
    return dataclasses.replace(wf, pieces=tuple(pieces),
                               artifacts=tuple(dict.fromkeys(rename(a) for a in wf.artifacts)),
                               edges=tuple(dict.fromkeys((rename(a), rename(b)) for a, b in wf.edges)),
                               control=dataclasses.replace(wf.control, gates=gates), extra=extra)


def belief_key(wf: Workflow) -> tuple:
    """Everything the belief's rows read of a workflow: equal keys give equal rows for equal settings."""
    ids = {p.id for p in wf.pieces}
    made = {a: p for p, a in wf.edges if p in ids and a not in ids}
    flow = {(made[a], q) for a, q in wf.edges if a in made and q in ids} | {e for e in wf.edges if set(e) <= ids}
    pieces = tuple((p.id, p.role, max(1, int(p.width or 1))) for p in wf.pieces)
    gates = tuple((g.after, g.rule, g.on_fail) for g in wf.control.gates)
    return (wf.id, pieces, tuple(sorted(flow)), gates, max(1, int(wf.control.budget_rounds or 1)))


# ---------------------------------------------------------------- objective

@dataclass
class Objective:
    """How u (the linear predictor behind g, times `sign`) maps to g, and the rescue in dollars."""

    head: str  # "success" or "score:<name>"
    rescue: float
    sign: float = 1.0  # -1 for a lower-is-better score
    center: float = 0.0
    scale: float = 1.0
    sd: float = 1.0
    tt: float = 0.0

    def g(self, u: np.ndarray) -> np.ndarray:
        """g at u, per draw or at the posterior mean; increasing in u."""
        if self.head == "success":
            return expit(u)
        return ndtr((self.center + self.scale * self.sign * np.asarray(u) - self.tt) * self.sign / max(self.sd, 1e-12))


def objective_for(fs: Any, task: Task, rule: AcceptanceRule | None, rescue_usd: float | None) -> Objective:
    """The head g comes from, as `FitState._predict` switches it: a score head once it has
    `MIN_SCORE_SWITCH` runs in the task's type and the target transforms."""
    rescue = float(rescue_usd or 0.0)
    if rule is not None and rule.score is not None and f"score:{rule.score.name}" in fs.heads:
        name = rule.score.name
        info = fs.score_info(name)
        tt = transform(float(rule.score.target), info.get("scale") or "linear")
        if tt is not None and fs._score_support(name, task) >= MIN_SCORE_SWITCH:
            head = fs.heads[f"score:{name}"]
            better = rule.score.better or info.get("better") or "higher"
            return Objective(f"score:{name}", rescue, 1.0 if better == "higher" else -1.0, head.center, head.scale,
                             head.scale * head.sigma, tt)
    return Objective("success", rescue)


def searchable(belief: Any) -> bool:
    """Whether the belief exposes the rows the search reads (a fitted state; the test fakes do not)."""
    return all(hasattr(belief, a) for a in ("heads", "_task_parts", "_cached", "_score_support", "score_info"))


# ---------------------------------------------------------------- shape cases and the space

@dataclass
class Case:
    key: str
    workflow: Workflow
    origin: str  # "builder" | "recorded" | "width_form"
    st: Structure
    judged: str | None = None  # the piece every loop gate judges (None: no loop)
    loop: list[str] = field(default_factory=list)
    outside: list[str] = field(default_factory=list)  # pieces outside the loop, copy groups as one name
    groups: dict[str, tuple[str, ...]] = field(default_factory=dict)  # "copies:a+b" -> its pieces
    combos: dict[str, np.ndarray] = field(default_factory=dict)  # copy group -> (mixes, k) setting indices

    def n_configs(self, n_settings: int, mixes: dict[int, int]) -> int:
        grouped = sum(len(g) for g in self.groups.values())
        n = n_settings ** (len(self.st.pieces) - grouped)
        for g in self.groups.values():
            n *= mixes[len(g)]
        return n


def make_case(key: str, wf: Workflow, origin: str, any_setting: Setting, copies: bool = True) -> Case:
    """A case from a workflow. Raises NotExact for loop structures outside the method."""
    cfg = Configuration("cfg_case", wf, {p.id: any_setting for p in wf.pieces})
    st = structure(cfg)
    c = Case(key, wf, origin, st)
    if len(st.loops) > 1:
        raise NotExact("more than one repair loop")
    if st.loops:
        gidx, members = st.loops[0]
        judged = {st.gates[gi].judged for gi in gidx}
        if len(judged) != 1:
            raise NotExact("loop gates judge different pieces")
        c.judged = judged.pop()
        c.loop = list(members)
    c.outside = [p for p in st.pieces if p not in c.loop]
    if copies:
        for grp in copy_groups(wf, c.outside):
            name = "copies:" + "+".join(grp)
            c.groups[name] = grp
            c.outside = [p for p in c.outside if p not in grp] + [name]
    return c


def builder_workflows(widths: Sequence[int], rounds: Sequence[int]) -> list[tuple[str, Workflow]]:
    """The D33 shapes over the widths and round limits: front none or plan, middle implement, best_of_n or team,
    back none or review."""
    out = []
    for plan in (False, True):
        for middle in MIDDLES:
            for review in (False, True):
                for w in ((1,) if middle == "implement" else widths):
                    for k in (rounds if review else (1,)):
                        p = ShapeParams(plan=plan, middle=middle, width=w, review=review, budget_rounds=k)
                        out.append((f"{shape_name(p)} w{w} k{k}", build_shape(p)))
    return out


@dataclass
class Space:
    settings: list[Setting]
    cases: list[Case]
    copies: str = "model"
    widths: tuple[int, int] = (MIN_WIDTH, MAX_WIDTH)
    rounds: tuple[int, int] = (MIN_ROUNDS, MAX_ROUNDS)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (case key, why) not searched

    def mixes(self) -> dict[int, int]:
        ks = {len(g) for c in self.cases for g in c.groups.values()}
        return {k: differing_mix_count(self.settings, k, self.copies) for k in ks}

    def configurations(self) -> int:
        mixes = self.mixes()
        return int(sum(c.n_configs(len(self.settings), mixes) for c in self.cases))

    def summary(self) -> dict[str, Any]:
        return {"shapes": len({c.workflow.id for c in self.cases}), "cases": len(self.cases),
                "settings": len(self.settings), "configurations": self.configurations(),
                "widths": list(self.widths), "rounds": list(self.rounds),
                "recorded_shapes": sum(1 for c in self.cases if c.origin != "builder"), "copies": self.copies}


def space_settings(pairs: Sequence[tuple[Configuration, str]]) -> list[Setting]:
    """The allowed settings: every setting in the candidates whose model the catalog and edit candidates use
    (`models.allowed`, so `--models` limits the space), else every setting. Deduplicated by what the belief
    reads; the objects of the usual, user and recorded configurations win, so their ids match; ordered as the
    catalog lists them (models in `models.allowed` order)."""
    generated = [c for c, o in pairs if o in ("catalog", "edit")]
    models = {setting_key(s)[1] for c in generated for s in c.settings.values()}
    first = [c for c, o in pairs if o not in ("catalog", "edit")]
    objects: dict[tuple, Setting] = {}
    for cfg in first + generated:
        for s in cfg.settings.values():
            objects.setdefault(setting_key(s), s)
    order: dict[tuple, None] = {}
    for cfg in generated + first:
        for s in cfg.settings.values():
            k = setting_key(s)
            if not models or k[1] in models:
                order.setdefault(k, None)
    return [objects[k] for k in order]


def space_from(pairs: Sequence[tuple[Configuration, str]], usual: Configuration | None = None, *,
               widths: Sequence[int] = tuple(range(MIN_WIDTH, MAX_WIDTH + 1)),
               rounds: Sequence[int] = tuple(range(MIN_ROUNDS, MAX_ROUNDS + 1)), copies: str = "model",
               settings: Sequence[Setting] | None = None) -> Space:
    """The space the candidates span: the D33 builder cases over `widths` and `rounds`, every other structure
    among the candidates (and the width form of each copy group), and the allowed settings."""
    if copies not in COPY_RULES:
        raise ValueError(f"unknown search copies rule {copies!r}; expected one of {', '.join(COPY_RULES)}")
    sets = list(settings) if settings is not None else space_settings(pairs)
    sp = Space(sets, [], copies, (min(widths), max(widths)), (min(rounds), max(rounds)))
    if not sets:
        return sp
    s0 = sets[0]
    seen: set[tuple] = set()

    def add(key: str, wf: Workflow, origin: str) -> None:
        bk = belief_key(wf)
        if bk in seen:
            return
        seen.add(bk)
        try:
            case = make_case(key, wf, origin, s0)
        except NotExact as e:
            sp.skipped.append((key, str(e)))
            return
        sp.cases.append(case)
        for grp in case.groups.values():
            add(f"{key} width form", width_form(wf, grp), "width_form")

    for key, wf in builder_workflows(widths, rounds):
        add(key, wf, "builder")
    extra = ([usual] if usual is not None else []) + [c for c, _ in pairs]
    for cfg in extra:
        add(f"{cfg.workflow.id} [{cfg.id}]", cfg.workflow, "recorded")
    mixes = sp.mixes()
    too_many = {k for k, n in mixes.items() if n > MAX_MIXES}
    if too_many:
        kept = []
        for c in sp.cases:
            if any(len(g) in too_many for g in c.groups.values()):
                sp.skipped.append((c.key, f"a copy group with more than {MAX_MIXES:,} mixes"))
            else:
                kept.append(c)
        sp.cases = kept
    return sp


# ---------------------------------------------------------------- splitting rows

def _minus(row: Sequence[tuple], common: Counter) -> tuple:
    left = Counter(common)
    out = []
    for t in row:
        if left[t] > 0:
            left[t] -= 1
        else:
            out.append(t)
    return tuple(out)


def split_rows(rows: Sequence[Sequence[tuple]]) -> tuple[tuple, list[tuple]]:
    """One row per setting of a piece -> (case part, setting part per setting). The case part is what every
    row shares; a node found in both parts moves into every setting part, so the parts' unseen variances add."""
    first = rows[0]
    n = len(first)
    if all(len(r) == n for r in rows):
        vary = [i for i in range(n) if any(r[i] != first[i] for r in rows)]
        vs = set(vary)
        case = [first[i] for i in range(n) if i not in vs]
        parts = [tuple(r[i] for i in vary) for r in rows]
    else:
        common = Counter(first)
        for r in rows[1:]:
            common &= Counter(r)
        case = _multiset_take(first, common)
        parts = [_minus(r, common) for r in rows]
    snodes = {t[0] for p in parts for t in p}
    moved = tuple(t for t in case if t[0] in snodes)
    if moved:
        case = [t for t in case if t[0] not in snodes]
        parts = [p + moved for p in parts]
    return tuple(case), parts


def _multiset_take(row: Sequence[tuple], common: Counter) -> list:
    left = Counter(common)
    out = []
    for t in row:
        if left[t] > 0:
            left[t] -= 1
            out.append(t)
    return out


def cost_terms(st: Structure, piece: str, k: int, setting: Setting, effort: float = 1.0) -> tuple:
    """A piece's cost row without the task part, as `FitState._plan` builds it (with the psrc and fsrc nodes of
    the source a prediction is for, and the effort terms at `effort`, `FitState.effort_for(task)`)."""
    return cost_rest(st, piece, k, setting, source=PREDICT_SOURCE, effort=effort)


class _Varied:
    """`st` with one piece's setting replaced for the duration of a row build."""

    def __init__(self, st: Structure, piece: str, setting: Setting):
        self.st, self.piece, self.setting = st, piece, setting

    def __enter__(self) -> Structure:
        self.old = self.st.settings[self.piece]
        self.st.settings[self.piece] = self.setting
        return self.st

    def __exit__(self, *exc: Any) -> None:
        self.st.settings[self.piece] = self.old


# ---------------------------------------------------------------- tables

class Tables:
    """Per case: for every piece and setting, the posterior-mean cost and u (stage 1) and their per-draw values
    (stage 2), from the fit's own rows and heads. Setting parts are evaluated once per distinct term tuple and
    shared by every case; case parts once per case, piece and round."""

    def __init__(self, fs: Any, task: Task, obj: Objective, settings: Sequence[Setting], draw_idx: np.ndarray,
                 copies: str = "model"):
        self.fs, self.task, self.obj, self.settings, self.copies = fs, task, obj, list(settings), copies
        self.S = len(self.settings)
        self.tp = fs._task_parts(task)
        self.effort = fs.effort_for(task) if hasattr(fs, "effort_for") else 1.0  # spec 04 section 1, under a timebox
        self.half_u = 0.5 * fs.heads["cost"].sigma ** 2
        self.draw_idx = draw_idx
        tg = self.tp["gate"]
        self.nodes = MEAN_NODES if tg.s2 > 0 else 1  # the unseen task effect on the gates, by Gauss-Hermite
        n = len(tg.known)
        w = (_GW[:, None] * np.ones((1, n)) / n) if self.nodes > 1 else np.full((1, n), 1.0 / n)
        self.omega = w.ravel()  # mean-path weights over the tiled (node, draw) entries
        self.omega_fold = w.sum(axis=0)  # the same, summed over the nodes
        self._memo: dict = {}
        self._keys: dict = {}
        self._run_memo: dict = {}

    def _key(self, obj: Any) -> int:
        """A small id for a hashable value (term tuples are long)."""
        k = self._keys.get(obj)
        if k is None:
            k = self._keys[obj] = len(self._keys)
        return k

    def _stack(self, head: str, rows: Sequence[tuple]):
        key = ("stack", head, self._key(tuple(rows)))
        hit = self._memo.get(key)
        if hit is None:
            rs = self.fs._cached(self.fs.heads[head], list(rows))
            hit = self._memo[key] = (np.array([r.mu for r in rs]), np.stack([r.known for r in rs]),
                                     np.stack([r.draws for r in rs]), np.array([r.s2 for r in rs]))
        return hit

    # cost rows -----------------------------------------------------
    def _cost_split(self, st: Structure, piece: str, k: int):
        """(case part, setting parts) of a piece's cost row in round k. Rounds after the first reuse the first
        round's setting parts; the rows are checked on the last setting."""
        base = self._memo.get(("cost_split", id(st), piece, 1))
        if base is None:
            base = split_rows([cost_terms(st, piece, 1, s, self.effort) for s in self.settings])
            self._memo[("cost_split", id(st), piece, 1)] = base
        if k == 1:
            return base
        parts = base[1]
        case = _minus(cost_terms(st, piece, k, self.settings[0], self.effort), Counter(parts[0]))
        check = cost_terms(st, piece, k, self.settings[-1], self.effort)
        if Counter(case) + Counter(parts[-1]) != Counter(check):
            raise NotExact(f"the setting part of {piece}'s cost row changes with the round")
        if {t[0] for t in case} & {t[0] for p in parts for t in p}:
            raise NotExact(f"{piece}'s cost row shares a node between its case and setting parts")
        return case, parts

    def cost_arrays(self, st: Structure, piece: str, k: int):
        """Log cost per setting: mean path (S, N_DRAWS) with the unseen variances as s2 / 2, per draw (S, D),
        and the case part's draws (D,) (the round effect)."""
        case, parts = self._cost_split(st, piece, k)
        key = ("cost", self._key(case), self._key(tuple(parts)))
        hit = self._memo.get(key)
        if hit is None:
            _, sknown, sdraws, ss2 = self._stack("cost", parts)
            _, cknown, cdraws, cs2 = self._stack("cost", [case])
            tc = self.tp["cost"]
            mean_path = tc.known[None, :] + cknown + sknown + 0.5 * (tc.s2 + cs2[0] + ss2)[:, None] + self.half_u
            draws = (tc.draws[None, :] + cdraws + sdraws)[:, self.draw_idx] + self.half_u
            hit = self._memo[key] = (mean_path, draws, cdraws[0][self.draw_idx])
        return hit

    # gate rows ----------------------------------------------------
    def gate_arrays(self, st: Structure, gi: int, k: int):
        """Pass chance per setting of the judged piece: key, mean path (S, E) and per draw (S, D)."""
        g = st.gates[gi]
        rows = []
        for s in self.settings:
            with _Varied(st, g.judged, s) as v:
                rows.append(gate_rest(v, g, k))
        case, parts = split_rows(rows)
        key = ("gate", self._key(case), self._key(tuple(parts)))
        hit = self._memo.get(key)
        if hit is None:
            _, _, sdraws, _ = self._stack("gate", parts)
            _, _, cdraws, _ = self._stack("gate", [case])
            tg = self.tp["gate"]
            mean_path = tg.known[None, :] + cdraws + sdraws
            if self.nodes > 1:
                shift = math.sqrt(tg.s2) * _GX[:, None]
                mean_path = (mean_path[:, None, :] + shift[None, :, :]).reshape(self.S, -1)
            draws = (tg.draws[None, :] + cdraws + sdraws)[:, self.draw_idx]
            hit = self._memo[key] = (key, expit(mean_path), expit(draws))
        return hit

    # run rows -----------------------------------------------------
    def _run_split(self, case: Case):
        """The run row as a case part plus one setting part per piece: {piece: parts}, case terms.
        Cases with the same pieces and flow share the per-piece parts; a hit is checked on the last setting."""
        st = case.st
        r0 = run_rest(st)
        mkey = (st.workflow_id, tuple(st.pieces), tuple(st.roles[p] for p in st.pieces),
                tuple(sorted(case.workflow.edges)), len(r0))
        hit = self._run_memo.get(mkey)
        if hit is not None:
            vary, parts = hit
            for p in st.pieces:
                with _Varied(st, p, self.settings[-1]) as v:
                    row = run_rest(v)
                want = list(r0)
                for i, t in zip(vary[p], parts[p][-1]):
                    want[i] = t
                if tuple(want) != row:
                    hit = None
                    break
        if hit is None:
            vary, parts = {}, {}
            for p in st.pieces:
                rows = []
                for s in self.settings:
                    with _Varied(st, p, s) as v:
                        rows.append(run_rest(v))
                if any(len(r) != len(r0) for r in rows):
                    raise NotExact("run rows change length with a setting")
                vary[p] = [i for i in range(len(r0)) if any(r[i] != r0[i] for r in rows)]
                parts[p] = [tuple(r[i] for i in vary[p]) for r in rows]
            used = [i for p in st.pieces for i in vary[p]]
            if len(used) != len(set(used)):
                raise NotExact("a run row term couples two pieces' settings")
            self._run_memo[mkey] = (vary, parts)
        used = {i for p in st.pieces for i in vary[p]}
        return {p: parts[p] for p in st.pieces}, tuple(t for i, t in enumerate(r0) if i not in used)

    # a case -------------------------------------------------------
    def _reach_weights(self, st: Structure, members: list[str], after: dict, K: int):
        """Per loop member and round, the chance to reach it per setting of the judged piece: mean path (S, N)
        folded over the gates' nodes, and per draw (S, D)."""
        gates = {(gi, k): self.gate_arrays(st, gi, k)
                 for v in members for gi in after.get(v, ()) for k in range(1, K + 1)}
        sig = ("reach", K, tuple((i, tuple(gates[(gi, k)][0] for gi in after.get(v, ()) for k in range(1, K + 1)))
                                 for i, v in enumerate(members)))
        if sig in self._memo:
            wm, wd = self._memo[sig]
            return dict(zip(members, wm)), dict(zip(members, wd))
        E, D = len(self.omega), len(self.draw_idx)
        r_m, r_d = np.ones((self.S, E)), np.ones((self.S, D))
        ws_m: dict[str, dict[int, np.ndarray]] = {v: {} for v in members}
        ws_d: dict[str, dict[int, np.ndarray]] = {v: {} for v in members}
        for k in range(1, K + 1):
            m_m, m_d = r_m, r_d
            for v in members:
                ws_m[v][k], ws_d[v][k] = m_m, m_d
                for gi in after.get(v, ()):
                    _, pm, pd = gates[(gi, k)]
                    m_m, m_d = m_m * pm, m_d * pd
            r_m, r_d = r_m - m_m, r_d - m_d
        n = len(self.omega_fold)
        ws_m = {v: {k: (w * self.omega).reshape(self.S, self.nodes, n).sum(axis=1) for k, w in ws_m[v].items()}
                for v in members}
        self._memo[sig] = ([ws_m[v] for v in members], [ws_d[v] for v in members])
        return ws_m, ws_d

    def case_tables(self, case: Case) -> dict:
        """Everything the dynamic programs need for one case.

        u[p]: (mean (S,), draws (S, D)) of the piece's part of the run row, times the objective's sign; u0: the
        case part, (scalar, (D,)). cout[p]: mean cost (S,) and per-draw cost (S, D) of a piece outside the loop.
        With a loop and J the judged piece: cJ, J's expected loop cost per setting (mean (S,), per draw
        (S, D)); cloop[v], member v's mean cost given J's setting (S_J, S_v); base[v], v's round-1 cost per draw
        (S_v, D); Bv[v] (S_J, D), the factor v's round-1 cost is multiplied by per draw (the chance to reach
        each round times the round effect); B = Bv[J]."""
        st = case.st
        parts, u0_terms = self._run_split(case)
        head = self.obj.head
        sign = self.obj.sign
        out: dict[str, Any] = {"u": {}, "cout": {}, "cloop": {}, "base": {}}
        for p in st.pieces:
            mu, _, d, _ = self._stack(head, parts[p])
            out["u"][p] = (sign * mu, sign * d[:, self.draw_idx])
        mu0, _, d0, _ = self._stack(head, [u0_terms])
        t = self.tp[head]
        out["u0"] = (sign * (t.mu + mu0[0]), sign * (t.draws + d0[0])[self.draw_idx])
        singles = [q for q in case.outside if q not in case.groups] + [q for g in case.groups.values() for q in g]
        for p in singles:
            lm, ld, _ = self.cost_arrays(st, p, 1)
            w = st.widths[p]
            out["cout"][p] = (w * (np.exp(lm) @ self.omega_fold), w * np.exp(ld))
        for name, grp in case.groups.items():
            combos = np.array(group_combos(self.settings, len(grp), self.copies), dtype=int).reshape(-1, len(grp))
            case.combos[name] = combos
            for key in ("cout", "u"):
                out[key][name] = tuple(sum(out[key][p][a][combos[:, i]] for i, p in enumerate(grp)) for a in (0, 1))
        if case.judged is None:
            return out
        J, members = case.judged, case.loop
        after: dict[str, list[int]] = {}
        for gi in st.loops[0][0]:
            after.setdefault(st.gates[gi].after, []).append(gi)
        ws_m, ws_d = self._reach_weights(st, members, after, st.k_max)
        D = len(self.draw_idx)
        Bs = {}
        for v in members:
            w = st.widths[v]
            cm_tot = np.zeros((self.S, self.S))
            B = np.zeros((self.S, D))
            _, ld1, c1 = self.cost_arrays(st, v, 1)
            for k in range(1, st.k_max + 1):
                lm, _, ck = self.cost_arrays(st, v, k)
                cm_tot += ws_m[v][k] @ (w * np.exp(lm)).T
                B += ws_d[v][k] * np.exp(ck - c1)[None, :]
            Bs[v] = B
            out["base"][v] = w * np.exp(ld1)
            if v == J:
                out["cJ"] = (np.diag(cm_tot).copy(), B * out["base"][v])
            else:
                out["cloop"][v] = cm_tot
        out["B"] = Bs[J]
        out["Bv"] = Bs
        out["shared_B"] = all(np.allclose(Bs[v], Bs[J], rtol=1e-12, atol=0) for v in members)
        return out


# ---------------------------------------------------------------- Pareto fronts

def front_mask(cost: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Non-dominated points (lower cost, higher u), ties kept once; 1-D arrays."""
    order = np.lexsort((-u, cost))
    uo = u[order]
    best = np.maximum.accumulate(uo)
    keep = np.ones(len(uo), bool)
    keep[1:] = uo[1:] > best[:-1] + TIE
    mask = np.zeros(len(u), bool)
    mask[order[keep]] = True
    return mask


@dataclass
class Front:
    """Partial or full choices: cost, u and the setting (or mix) index of every piece chosen so far."""

    cost: np.ndarray
    u: np.ndarray
    idx: np.ndarray  # (n, pieces so far)

    def prune(self) -> "Front":
        m = front_mask(self.cost, self.u)
        return Front(self.cost[m], self.u[m], self.idx[m])


def merge(a: Front, cost_b: np.ndarray, u_b: np.ndarray) -> Front:
    """a plus one more piece with per-option (cost, u), pruned."""
    ib = np.arange(len(cost_b))
    m = front_mask(cost_b, u_b)
    ib, cost_b, u_b = ib[m], cost_b[m], u_b[m]
    c = (a.cost[:, None] + cost_b[None, :]).ravel()
    u = (a.u[:, None] + u_b[None, :]).ravel()
    ia = np.repeat(np.arange(len(a.cost)), len(ib))
    idx = np.concatenate([a.idx[ia], np.tile(ib, len(a.cost))[:, None]], axis=1)
    return Front(c, u, idx).prune()


def _empty() -> Front:
    return Front(np.zeros(1), np.zeros(1), np.zeros((1, 0), int))


def case_front(case: Case, tab: dict) -> tuple[Front, list[str]]:
    """Stage 1: the exact (E[C_run], u) front of a case, and the piece order of `idx`."""
    order = list(case.outside)
    f = _empty()
    for p in case.outside:
        f = merge(f, tab["cout"][p][0], tab["u"][p][0])
    u0 = tab["u0"][0]
    if case.judged is None:
        return Front(f.cost, f.u + u0, f.idx), order
    J = case.judged
    others = [v for v in case.loop if v != J]
    cJ, uJ = tab["cJ"][0], tab["u"][J][0]
    parts = []
    for j in range(len(cJ)):
        g = Front(f.cost + cJ[j], f.u + uJ[j], np.concatenate([f.idx, np.full((len(f.cost), 1), j)], axis=1))
        for v in others:
            g = merge(g, tab["cloop"][v][j], tab["u"][v][0])
        parts.append(g)
    allf = Front(np.concatenate([p.cost for p in parts]), np.concatenate([p.u for p in parts]),
                 np.concatenate([p.idx for p in parts])).prune()
    return Front(allf.cost, allf.u + u0, allf.idx), order + [J] + others


# ---------------------------------------------------------------- stage 2: per-draw optima

def _best_per_draw(cost: np.ndarray, u: np.ndarray, obj: Objective) -> dict:
    """cost, u: (D, N) options per draw -> per objective (value, argmin) per draw. `default` minimizes
    C + (1 - g) R; `pXX` minimizes C subject to g >= XX / 100 (inf when none reaches it)."""
    g = obj.g(u)
    ell = cost + (1.0 - g) * obj.rescue
    out = {"default": (ell.min(axis=1), ell.argmin(axis=1))}
    for lv in LEVELS:
        c = np.where(g >= lv / 100.0, cost, np.inf)
        out[f"p{lv}"] = (c.min(axis=1), c.argmin(axis=1))
    return out


def front_rows(cost: np.ndarray, u: np.ndarray):
    """Per-row fronts of (D, N) arrays, padded with inf: cost, u, index (D, M). A cost tie may keep a dominated
    point, which is harmless."""
    D, N = cost.shape
    if N == 1:
        return cost, u, np.zeros((D, 1), int)
    order = np.argsort(cost, axis=1)
    co = np.take_along_axis(cost, order, 1)
    uo = np.take_along_axis(u, order, 1)
    keep = np.empty((D, N), bool)
    keep[:, 0] = True
    keep[:, 1:] = uo[:, 1:] > np.maximum.accumulate(uo, axis=1)[:, :-1]
    rows, cols = np.nonzero(keep)
    pos = np.cumsum(keep, axis=1)[rows, cols] - 1
    M = int(pos.max()) + 1
    c = np.full((D, M), np.inf)
    uu = np.full((D, M), -np.inf)
    ix = np.zeros((D, M), int)
    c[rows, pos] = co[rows, cols]
    uu[rows, pos] = uo[rows, cols]
    ix[rows, pos] = order[rows, cols]
    return c, uu, ix


def _merge_rows(fc, fu, fidx, pc, pu, pix):
    """Per draw, the front of every pair (a point of the running front, a point of the piece's front)."""
    D = fc.shape[0]
    if fc.shape[1] == 1 and fidx.shape[2] == 0:
        return fc + pc, fu + pu, pix[:, :, None]
    c = (fc[:, :, None] + pc[:, None, :]).reshape(D, -1)
    u = (fu[:, :, None] + pu[:, None, :]).reshape(D, -1)
    a_ix = np.repeat(np.arange(fc.shape[1]), pc.shape[1])
    b_ix = np.tile(np.arange(pc.shape[1]), fc.shape[1])
    c2, u2, sel = front_rows(c, np.where(np.isfinite(c), u, -np.inf))
    prev = fidx[np.arange(D)[:, None], a_ix[sel]]
    new = np.take_along_axis(pix, b_ix[sel], 1)[:, :, None]
    return c2, u2, np.concatenate([prev, new], axis=2)


def _draw_subset(tab: dict, dsel: np.ndarray) -> dict:
    """The case tables on some draws only (every per-draw array has draws on its last axis)."""
    out = dict(tab)
    out["u"] = {p: (m, d[..., dsel]) for p, (m, d) in tab["u"].items()}
    out["u0"] = (tab["u0"][0], tab["u0"][1][dsel])
    out["cout"] = {p: (m, d[..., dsel]) for p, (m, d) in tab["cout"].items()}
    if "B" in tab:
        out["B"] = tab["B"][..., dsel]
        out["Bv"] = {v: d[..., dsel] for v, d in tab["Bv"].items()}
        out["base"] = {v: d[..., dsel] for v, d in tab["base"].items()}
    return out


def nondominated3(a: np.ndarray, b: np.ndarray, u: np.ndarray):
    """Per row of (D, S) arrays, the points not dominated on (a low, b low, u high); exact duplicates keep the
    lowest index. Returns kept indices (D, M) padded with 0 and a validity mask."""
    D, S = a.shape
    ge = (a[:, :, None] <= a[:, None, :]) & (b[:, :, None] <= b[:, None, :]) & (u[:, :, None] >= u[:, None, :])
    strict = (a[:, :, None] < a[:, None, :]) | (b[:, :, None] < b[:, None, :]) | (u[:, :, None] > u[:, None, :])
    idx = np.arange(S)
    dom = ge & (strict | (idx[:, None] < idx[None, :])[None])  # [d, j', j]: j' dominates j
    dom[:, idx, idx] = False
    keep = ~dom.any(axis=1)
    rows, cols = np.nonzero(keep)
    pos = np.cumsum(keep, axis=1)[rows, cols] - 1
    M = int(pos.max()) + 1
    sel = np.zeros((D, M), int)
    ok = np.zeros((D, M), bool)
    sel[rows, pos] = cols
    ok[rows, pos] = True
    return sel, ok


def case_bounds(case: Case, tab: dict):
    """Per draw, a lower bound on the case's run cost and an upper bound on its u (each piece at its own best):
    (C_lo (D,), U_hi (D,)). In the loop, given J's setting j, member v costs Bv[v][j] times its round-1 cost, so
    the bound is the smallest over j of J's loop cost plus each other member's own factor times its cheapest
    round-1 cost. Members' factors differ when a gate sits between them, so J's factor B then does not stand in
    for theirs."""
    c = sum(tab["cout"][p][1].min(axis=0) for p in case.outside) if case.outside else 0.0
    u = tab["u0"][1] + sum(tab["u"][p][1].max(axis=0) for p in case.outside + case.loop)
    if case.judged is not None:
        J, base = case.judged, tab["base"]
        rest = [v for v in case.loop if v != J]
        if tab["shared_B"]:
            loop = tab["B"] * (base[J] + sum(base[v].min(axis=0) for v in rest))
        else:
            loop = tab["Bv"][J] * base[J] + sum(tab["Bv"][v] * base[v].min(axis=0) for v in rest)
        c = c + loop.min(axis=0)
    return c + np.zeros_like(u), u


def bound_values(obj: Objective, c_lo: np.ndarray, u_hi: np.ndarray) -> dict:
    """Lower bounds of every objective per draw (g is increasing in u)."""
    g = obj.g(u_hi)
    out = {"default": c_lo + (1.0 - g) * obj.rescue}
    for lv in LEVELS:
        out[f"p{lv}"] = np.where(g >= lv / 100.0, c_lo, np.inf)
    return out


def _loop_per_j(case, tab, obj, fc, fu, fidx, u0, order, others):
    """The loop when members' round weights differ (a gate between them): per J setting j, every other member's
    per-draw cost is Bv[j] base_v, so the others' front is merged per j; exact, S_J times the work."""
    J = case.judged
    D = fc.shape[0]
    best = None
    for j in range(tab["B"].shape[0]):
        lc = (tab["Bv"][J][j] * tab["base"][J][j])[:, None]
        lu = tab["u"][J][1][j][:, None]
        lix = np.zeros((D, 1, 0), int)
        for v in others:
            pc, pu, pix = front_rows((tab["Bv"][v][j][None, :] * tab["base"][v]).T, tab["u"][v][1].T)
            lc, lu, lix = _merge_rows(lc, lu, lix, pc, pu, pix)
        M = fc.shape[1]
        C = (lc[:, :, None] + fc[:, None, :]).reshape(D, -1)
        U = (lu[:, :, None] + fu[:, None, :]).reshape(D, -1) + u0
        res = _best_per_draw(np.where(np.isfinite(C), C, np.inf), U, obj)
        if best is None:
            best = {k: (np.full(D, np.inf), np.zeros((D, fidx.shape[2] + 1 + len(others)), int)) for k in res}
        for key, (val, arg) in res.items():
            li, fi = np.divmod(arg, M)
            ch = np.concatenate([fidx[np.arange(D), fi], np.full((D, 1), j), lix[np.arange(D), li]], axis=1)
            better = val < best[key][0]
            best[key][0][better] = val[better]
            best[key][1][better] = ch[better]
    return best, order + [J] + others


def case_thompson(case: Case, tab: dict, obj: Objective, dsel: np.ndarray | None = None):
    """Per draw, the case's optimum for every objective: {objective: (value (D,), choices (D, pieces))} and the
    piece order of the choices. Exact: per draw, cost and eta are sums over pieces given J's setting, and
    dominated options of a piece (or of the merged pieces) never enter an optimum. `dsel`: only these draws."""
    if dsel is not None:
        tab = _draw_subset(tab, dsel)
    D = tab["u0"][1].shape[0]
    order = list(case.outside)
    fc, fu, fidx = np.zeros((D, 1)), np.zeros((D, 1)), np.zeros((D, 1, 0), int)
    for p in case.outside:
        fc, fu, fidx = _merge_rows(fc, fu, fidx, *front_rows(tab["cout"][p][1].T, tab["u"][p][1].T))
    u0 = tab["u0"][1][:, None]
    if case.judged is None:
        res = _best_per_draw(fc, fu + u0, obj)
        return {k: (v, fidx[np.arange(D), a]) for k, (v, a) in res.items()}, order
    J = case.judged
    others = [v for v in case.loop if v != J]
    if not tab["shared_B"]:
        return _loop_per_j(case, tab, obj, fc, fu, fidx, u0, order, others)
    B = tab["B"].T  # (D, S_J)
    cJ1 = tab["base"][J].T
    uJ = tab["u"][J][1].T
    # the other members' per-draw front on round-1 cost: B[j] > 0 scales all of them, so it does not depend on j
    lc, lu, lix = np.zeros((D, 1)), np.zeros((D, 1)), np.zeros((D, 1, 0), int)
    for v in others:
        lc, lu, lix = _merge_rows(lc, lu, lix, *front_rows(tab["base"][v].T, tab["u"][v][1].T))
    # J's setting j costs B[j] (cJ1[j] + L) with u uJ[j]: j' dominates j for every L >= 0 when it is no worse on
    # (B cJ1, B, -uJ)
    jsel, jok = nondominated3(B * cJ1, B, uJ)
    r = np.arange(D)[:, None]
    Bs, cs, us = B[r, jsel], cJ1[r, jsel], uJ[r, jsel]
    Ml = lc.shape[1]
    Lc = Bs[:, :, None] * (cs[:, :, None] + lc[:, None, :])
    Lc = np.where(jok[:, :, None] & np.isfinite(Lc), Lc, np.inf).reshape(D, -1)
    Lu = (us[:, :, None] + lu[:, None, :]).reshape(D, -1)
    Lu = np.where(np.isfinite(Lc), Lu, -np.inf)
    lac, lau, lsel = front_rows(Lc, Lu)
    M = fc.shape[1]
    C = (lac[:, :, None] + fc[:, None, :]).reshape(D, -1)
    U = (lau[:, :, None] + fu[:, None, :]).reshape(D, -1) + u0
    res = _best_per_draw(np.where(np.isfinite(C), C, np.inf), U, obj)
    out = {}
    for key, (val, arg) in res.items():
        a, fi = np.divmod(arg, M)
        jp, li = np.divmod(lsel[np.arange(D), a], Ml)
        j = jsel[np.arange(D), jp]
        out[key] = (val, np.concatenate([fidx[np.arange(D), fi], j[:, None], lix[np.arange(D), li]], axis=1))
    return out, order + [J] + others


def to_config(case: Case, piece_order: Sequence[str], choice: Sequence[int], settings: Sequence[Setting]
              ) -> Configuration:
    pick: dict[str, Setting] = {}
    for p, i in zip(piece_order, choice):
        if p in case.groups:
            pick.update({q: settings[int(j)] for q, j in zip(case.groups[p], case.combos[p][int(i)])})
        else:
            pick[p] = settings[int(i)]
    return make_config(case.workflow, pick)


# ---------------------------------------------------------------- the search

@dataclass
class Found:
    configs: list[tuple[Configuration, str]]  # (configuration, "front" | "thompson"), for the rescore
    wins: dict[str, dict[str, float]]  # every draw winner: objective -> share of draws it is that objective's best
    front: list[tuple[Configuration, float, float]]  # the global front: (configuration, run cost, g at the mean)
    stats: dict[str, Any]

    @property
    def on_front(self) -> set[str]:
        return {c.id for c, _, _ in self.front}


def draw_index(n_draws: int) -> np.ndarray:
    """The fit's draws stage 2 reads: evenly spaced draws of the first half and their antithetic partners in the
    second half (`HeadState.unit` is N_DRAWS / 2 normals and their negatives), so an even n is whole pairs."""
    n = max(1, min(int(n_draws), N_DRAWS))
    half = N_DRAWS // 2
    m = (n + 1) // 2
    first = np.arange(0, half, max(1, half // m))[:m]
    return np.sort(np.concatenate([first, first + half]))[:n]


def search(belief: Any, task: Task, rule: AcceptanceRule | None, rescue_usd: float | None, space: Space, *,
           draws: int = 200, per_objective: int = 100) -> Found:
    """Stages 1 and 2 over every case of `space`: the global front, and per objective the `per_objective`
    configurations that are best in the most draws."""
    t0 = time.perf_counter()
    obj = objective_for(belief, task, rule, rescue_usd)
    draw_idx = draw_index(draws)
    D = len(draw_idx)
    tabs = Tables(belief, task, obj, space.settings, draw_idx, space.copies)
    skipped = list(space.skipped)
    cases, case_tabs = [], []
    for c in space.cases:
        try:
            case_tabs.append(tabs.case_tables(c))
            cases.append(c)
        except NotExact as e:
            skipped.append((c.key, str(e)))
    t1 = time.perf_counter()
    fronts = [(c, *case_front(c, tab)) for c, tab in zip(cases, case_tabs)]
    t2 = time.perf_counter()
    # per draw, the best over all cases for every objective; cases in order of their mean bound, and a case runs
    # only on the draws where one of its bounds beats the incumbent
    bounds = [bound_values(obj, *case_bounds(c, tab)) for c, tab in zip(cases, case_tabs)]
    best_val = {k: np.full(D, np.inf) for k in OBJECTIVES}
    best_cfg: dict[str, list] = {k: [None] * D for k in OBJECTIVES}
    visits = 0
    for i in sorted(range(len(cases)), key=lambda i: (float(np.mean(bounds[i]["default"])), i)):
        c, tab = cases[i], case_tabs[i]
        active = np.zeros(D, bool)
        for k in OBJECTIVES:
            active |= bounds[i][k] < best_val[k]
        dsel = np.flatnonzero(active)
        if len(dsel) == 0:
            continue
        visits += len(dsel)
        res, order = case_thompson(c, tab, obj, dsel=dsel)
        for key, (val, choice) in res.items():
            better = val < best_val[key][dsel]
            for n in np.flatnonzero(better):
                best_cfg[key][dsel[n]] = (i, order, tuple(choice[n]))
            best_val[key][dsel[better]] = val[better]
    t3 = time.perf_counter()
    made: dict[tuple, Configuration] = {}

    def config_of(i: int, order: Sequence[str], choice: tuple) -> Configuration:
        k = (i, tuple(order), choice)
        if k not in made:
            made[k] = to_config(cases[i], order, choice, space.settings)
        return made[k]

    counts: dict[str, dict[str, int]] = {}
    by_id: dict[str, Configuration] = {}
    for key, lst in best_cfg.items():
        for item in lst:
            if item is None:
                continue
            cfg = config_of(*item)
            by_id[cfg.id] = cfg
            cnt = counts.setdefault(cfg.id, {})
            cnt[key] = cnt.get(key, 0) + 1
    # the global front across cases
    front: list[tuple[Configuration, float, float]] = []
    if fronts:
        allc = np.concatenate([f.cost for _, f, _ in fronts])
        allu = np.concatenate([f.u for _, f, _ in fronts])
        owner = np.concatenate([np.full(len(f.cost), i) for i, (_, f, _) in enumerate(fronts)])
        pos = np.concatenate([np.arange(len(f.cost)) for _, f, _ in fronts])
        m = front_mask(allc, allu)
        for i, j in zip(owner[m], pos[m]):
            _, f, order = fronts[i]
            cfg = config_of(int(i), order, tuple(int(x) for x in f.idx[j]))
            front.append((cfg, float(f.cost[j]), float(obj.g(f.u[j]))))
        front.sort(key=lambda x: (x[1], -x[2], x[0].id))
    out: dict[str, tuple[Configuration, str]] = {c.id: (c, "front") for c, _, _ in front}
    for key in OBJECTIVES:
        ranked = sorted(((cnt[key], cid) for cid, cnt in counts.items() if key in cnt), key=lambda x: (-x[0], x[1]))
        for _, cid in ranked[:per_objective]:
            out.setdefault(cid, (by_id[cid], "thompson"))
    wins = {cid: {k: round(v / D, 6) for k, v in cnt.items()} for cid, cnt in counts.items()}
    t4 = time.perf_counter()
    stats = {"method": METHOD, "exact": not skipped, "space": space.summary(), "draws": D,
             "front_points": len(front), "thompson_configs": len(counts), "found": len(out),
             "pruned_share": round(1.0 - visits / max(1, D * len(cases)), 6),
             "seconds": {"tables": t1 - t0, "front": t2 - t1, "thompson": t3 - t2, "assemble": t4 - t3,
                         "total": t4 - t0},
             "skipped": [{"case": k, "why": why} for k, why in skipped]}
    return Found(list(out.values()), wins, front, stats)


# ---------------------------------------------------------------- stage 3 polish

def neighbors(cfg: Configuration, space: Space) -> list[Configuration]:
    """Every configuration one piece's setting away from `cfg` in the space: a copy's new setting moves it to its
    place in copy order, and a mix whose copies no longer differ is left out (it is the width form's)."""
    try:
        case = make_case("polish", cfg.workflow, "polish", space.settings[0])
    except NotExact:
        return []
    out = []
    for p in case.st.pieces:
        for s in space.settings:
            cur = cfg.settings.get(p)
            if cur is not None and setting_key(cur) == setting_key(s):
                continue
            pick = {**cfg.settings, p: s}
            ok = True
            for grp in case.groups.values():
                if p not in grp:
                    continue
                combo = canonical_mix(space.settings, [pick[q] for q in grp], space.copies)
                if combo is None:
                    ok = False
                    break
                pick.update({q: space.settings[i] for q, i in zip(grp, combo)})
            if ok:
                out.append(make_config(cfg.workflow, pick))
    return out


def polish(space: Space, configs: dict[str, Configuration], preds: list[Prediction],
           predict: Callable[[list[Configuration]], list[Prediction]],
           choose: Callable[[list[Prediction]], str | None], rounds: int = 3) -> list[Prediction]:
    """Predict the chosen configuration's one-piece neighbours and choose again, until the choice stops moving
    (at most `rounds` times). This covers the part of the mean chance that depends on the posterior variance,
    which the front and the draws do not see. Adds to `configs` and returns the new predictions."""
    added: list[Prediction] = []
    for _ in range(rounds):
        cid = choose(preds + added)
        if cid is None or cid not in configs:
            break
        new = [c for c in neighbors(configs[cid], space) if c.id not in configs]
        if not new:
            break
        for c in new:
            configs[c.id] = c
        added += predict(new)
        if choose(preds + added) == cid:
            break
    return added
