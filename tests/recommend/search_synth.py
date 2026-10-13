"""A synthetic fitted store for the workflow search tests, and brute force over a search space.

Runs come from tests/belief/simdata.py's generator over the builder shapes (widths 2 and 3, round limits 1 and 3),
the recorded shapes with a tests gate, and a best of three written as three copies, with the six fictional settings
mixed per piece and a `perf` score; the fit uses no prior and a fixed clock, so its draws are fixed. No real data.
"""

from __future__ import annotations

import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "belief"))

import simdata as SD  # noqa: E402

from loopmath.belief.design import run_rest, structure  # noqa: E402
from loopmath.recommend import search as S  # noqa: E402
from loopmath.recommend.curve import LEVELS  # noqa: E402
from loopmath.types import AcceptanceRule, Control, Piece, ScoreTarget, Task, Workflow  # noqa: E402
from loopmath.workflows.ids import make_config  # noqa: E402

RULE = AcceptanceRule(name="tests", definition="tests pass")
SCORE = {"name": "perf", "scale": "linear", "better": "higher", "center": 2000.0, "spread": 300.0}
PERF_HIGH = AcceptanceRule(name="perf>=2100", definition="perf >= 2100", requires=(),
                           score=ScoreTarget(name="perf", target=2100.0, better="higher", scale="linear"))
PERF_LOW = AcceptanceRule(name="perf<=1900", definition="perf <= 1900", requires=(),
                          score=ScoreTarget(name="perf", target=1900.0, better="lower", scale="linear"))
COPIES = Workflow("plan_best_of_3_copies", 1, "Plan, three implementer copies, a referee",
                  (Piece("plan", "planner"), Piece("implement-1", "implementer"), Piece("implement-2", "implementer"),
                   Piece("implement-3", "implementer"), Piece("select", "referee")),
                  ("plan_doc", "diff-implement-1", "diff-implement-2", "diff-implement-3", "pick"),
                  (("plan", "plan_doc"), ("plan_doc", "implement-1"), ("plan_doc", "implement-2"),
                   ("plan_doc", "implement-3"), ("implement-1", "diff-implement-1"),
                   ("implement-2", "diff-implement-2"), ("implement-3", "diff-implement-3"),
                   ("diff-implement-1", "select"), ("diff-implement-2", "select"), ("diff-implement-3", "select"),
                   ("select", "pick")), Control())
TASK = Task(id="tsk_search_synth", type="feature", repo="acme/api", title="")
SETTINGS = list(SD.SETTINGS)
USUAL = make_config(SD.IR, {"implement": SETTINGS[0], "review": SETTINGS[1]})
NOW = datetime.fromisoformat("2026-09-24T12:00:00-07:00")

_FITS: dict[tuple, dict] = {}


def build(home: Path, n_runs: int = 1600, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    shapes = [wf for _, wf in S.builder_workflows((2, 3), (1, 3))] + [SD.SWEEP, SD.PIR, SD.BON, COPIES]
    configs = [make_config(wf, {p.id: SETTINGS[int(rng.integers(len(SETTINGS)))] for p in wf.pieces})
               for wf in shapes for _ in range(8)]
    docs, _ = SD.simulate(n_runs, seed=seed, configs=configs, score=SCORE)
    from loopmath.belief import fit as fitmod

    fitmod.fit(home, docs=docs, no_prior=True, now=NOW)


def synth_fit(tmp_path_factory, n_runs: int = 1600, seed: int = 11) -> dict:
    """The fitted store, built once per session: {"home", "state"}."""
    key = (n_runs, seed)
    if key not in _FITS:
        from loopmath.belief.state import load_latest

        home = tmp_path_factory.mktemp(f"search_synth_{seed}")
        build(home, n_runs, seed)
        _FITS[key] = {"home": home, "state": load_latest(home)}
    return _FITS[key]


def space(cases_of=None, *, widths=(3,), rounds=(3,), copies="model", settings=None, extra=()) -> S.Space:
    """A search space over the builder shapes (and `extra` recorded configurations), keeping the cases whose
    key `cases_of` accepts."""
    sp = S.space_from([(c, "recorded") for c in extra], USUAL, widths=widths, rounds=rounds, copies=copies,
                      settings=list(settings or SETTINGS))
    if cases_of is not None:
        sp.cases = [c for c in sp.cases if cases_of(c)]
    return sp


def enumerate_space(sp: S.Space):
    """Every configuration of the space, each copy mix once."""
    sets = sp.settings
    for c in sp.cases:
        ok = {len(g): set(S.group_combos(sets, len(g), sp.copies)) for g in c.groups.values()}
        for combo in itertools.product(range(len(sets)), repeat=len(c.st.pieces)):
            pick = dict(zip(c.st.pieces, combo))
            if all(tuple(pick[q] for q in g) in ok[len(g)] for g in c.groups.values()):
                yield make_config(c.workflow, {p: sets[i] for p, i in pick.items()})


def brute(fs, task: Task, rule, rescue_usd: float, sp: S.Space) -> dict:
    """Every configuration of the space predicted: ids, predictions, draws, mean cost and u."""
    cfgs = list(enumerate_space(sp))
    out = fs._predict_many(task, cfgs, rule, rescue_usd)
    obj = S.objective_for(fs, task, rule, rescue_usd)
    tp = fs._task_parts(task)[obj.head]
    rows = fs._cached(fs.heads[obj.head], [run_rest(structure(c)) for c in cfgs])
    return {"configs": cfgs, "ids": [c.id for c in cfgs], "preds": [p for p, _ in out], "draws": [d for _, d in out],
            "u": obj.sign * np.array([tp.mu + r.mu for r in rows]),
            "cost": np.array([p.cost.usd.mean for p, _ in out]), "obj": obj}


def curve_and_pick(preds, g: dict):
    """The curve's row per level (a config id or None) and the default pick, as the recommender chooses them."""
    rows = {}
    for lv in LEVELS:
        ok = [p for p in preds if p.p_success.mean >= lv / 100]
        rows[lv] = None if not ok else min(ok, key=lambda p: (p.cost.usd.mean, p.ell.usd.mean, p.config)).config
    return rows, min(preds, key=lambda p: (p.ell.usd.mean, p.cost.usd.mean, p.config)).config


def draw_winners(b: dict, idx: np.ndarray) -> dict[str, set[str]]:
    """Per objective, the configurations that are best in some draw of `idx`."""
    ell = np.stack([d["ell"] for d in b["draws"]])[:, idx]
    g = np.stack([d["g"] for d in b["draws"]])[:, idx]
    cost = np.stack([d["run"].exp_usd for d in b["draws"]])[:, idx]
    ids = b["ids"]
    out = {"default": {ids[i] for i in ell.argmin(axis=0)}}
    for lv in LEVELS:
        feas = np.where(g >= lv / 100, cost, np.inf)
        out[f"p{lv}"] = {ids[i] for i, ok in zip(feas.argmin(axis=0), np.isfinite(feas.min(axis=0))) if ok}
    return out
