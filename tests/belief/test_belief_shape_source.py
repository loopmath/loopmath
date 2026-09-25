"""Shapes, position x source and role groups (spec 04 sections 1 and 2; lane 2B, FINDINGS F5).

Synthetic bundles only: a graph run in one data source and solo in another, as the shipped bundle has them.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import design as D
from loopmath.belief import fit as F
from loopmath.belief import lookahead as L
from loopmath.belief import state as S
from loopmath.belief.design import copy_groups, rows_for_config, run_rest, setting_terms, shape_key, structure
from loopmath.belief.state import load_latest
from loopmath.types import AcceptanceRule, Control, Piece, ScoreTarget, Setting, Workflow

NOW = datetime.fromisoformat("2026-09-24T12:00:00-07:00")
Z80 = 1.2815515655446004
OPUS = Setting("claude-code", "claude-opus-9", "high")
FABLE = Setting("claude-code", "claude-fable-9", "medium")
PI = Workflow("plan_implement", 1, "Plan, implement",
              (Piece("plan", "planner"), Piece("implement", "implementer")), ("plan_doc", "patch"),
              (("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch")), Control())
COPIES = Workflow("best_of_n", 1, "Three copies, then a referee",
                  (Piece("implement-1", "implementer"), Piece("implement-2", "implementer"),
                   Piece("implement-3", "implementer"), Piece("select", "referee")),
                  ("patch-1", "patch-2", "patch-3", "pick"),
                  (("implement-1", "patch-1"), ("implement-2", "patch-2"), ("implement-3", "patch-3"),
                   ("patch-1", "select"), ("patch-2", "select"), ("patch-3", "select"), ("select", "pick")),
                  Control())
SCORE = {"name": "perf", "scale": "linear", "better": "higher", "center": 2000.0, "spread": 300.0}
GAP = -1.2  # source A's cost level against source B's


def _bundle(*, user_pi: int = 0, score: dict | None = None, seed: int = 3) -> list[tuple[str, dict]]:
    """Graph PI only in source A (`sweep`, cheaper by GAP), solo only in source B (`e0`), no other
    effect on cost, and `user_pi` runs of PI by the user. PI and solo pieces cost the same for everyone."""
    rng = np.random.default_rng(seed)
    truth = simdata.Truth(np.random.default_rng(seed + 1), scale_mult=0.0)
    truth.effects["cost"] = {"source:sweep": GAP}
    truth.effects["tokens"] = {"source:sweep": GAP}
    out = []
    for origin, wf, n, prefix in (("sweep", PI, 150, "run_a"), ("e0", simdata.SOLO, 150, "run_b"),
                                  ("user", PI, user_pi, "run_u")):
        for i in range(n):
            task = simdata.make_task(rng, i, n_tasks=40)
            doc = simdata.simulate_run(truth, task, simdata.config(wf, OPUS), f"{prefix}{i:04d}", rng,
                                       source=origin, score=score)
            out.append((origin, doc))
    return out


def _strip(monkeypatch, *prefixes):
    """The test hook: every cost row without the terms whose node starts with one of `prefixes`, keeping
    every other term. `"psrc:"` alone removes psrc alone; with `"fsrc:"` too, the rows are as before 0.2."""
    orig = D.cost_rest

    def cost_rest(*args, **kwargs):
        return tuple(t for t in orig(*args, **kwargs) if not t[0].startswith(prefixes))

    for module in (D, S, L):
        monkeypatch.setattr(module, "cost_rest", cost_rest)


BEFORE_02 = ("psrc:", "fsrc:")


def _fit(tmp_path_factory, docs):
    home = tmp_path_factory.mktemp("home")
    F.fit(home, docs=docs, no_prior=True, now=NOW)
    return load_latest(home)


def _contrast(state, a, b, piece="implement"):
    """The user's cost ratio of `piece` in configuration a against b: (mean, lo, hi) of its 80% interval,
    and the unseen nodes that do not cancel, in closed form (the task part cancels)."""
    h = state.heads["cost"]
    Xk, Xv, vnodes, vphi = h.matrices([state._plan(a)["cost"][(piece, 1)], state._plan(b)["cost"][(piece, 1)]])
    dk, dv = Xk[0] - Xk[1], (Xv[0] - Xv[1]).toarray().ravel()
    mu = float((dk @ h.mean).ravel()[0])
    sd = math.sqrt(float(np.sum(np.asarray(dk @ h.U) ** 2)) + float(np.sum(dv ** 2 * vphi ** 2)))
    return math.exp(mu), math.exp(mu - Z80 * sd), math.exp(mu + Z80 * sd), [v for v, d in zip(vnodes, dv) if d]


@pytest.fixture(scope="module")
def f5(tmp_path_factory):
    out = {"psrc": _fit(tmp_path_factory, _bundle()), "learned": _fit(tmp_path_factory, _bundle(user_pi=20))}
    for key, prefixes in (("no_psrc", ("psrc:",)), ("before", BEFORE_02)):
        with pytest.MonkeyPatch.context() as mp:
            _strip(mp, *prefixes)
            state = _fit(tmp_path_factory, _bundle())
            out[key] = _contrast(state, simdata.config(PI, OPUS), simdata.config(simdata.SOLO, OPUS))
    return out


# ---------------------------------------------------------------- 1, 2: aliasing and learning

def test_a_graph_from_one_source_does_not_carry_that_sources_cost_level(f5):
    """F5: with PI only in the cheap source and the rows as before 0.2, the gap between sources lands on
    PI's topology, so the user's PI implement looks cheaper than solo's. With psrc the 80% interval keeps 1.
    psrc alone (fsrc kept, which takes part of the gap itself) moves the mean toward 1 and widens the
    interval: x0.68 (0.19 to 2.39) against x0.58 (0.28 to 1.20) without psrc, x0.39 (0.23 to 0.65) before."""
    mean, lo, hi, unseen = _contrast(f5["psrc"], simdata.config(PI, OPUS), simdata.config(simdata.SOLO, OPUS))
    assert lo < 1 < hi
    assert set(unseen) == {"psrc:plan_implement#1|user", "psrc:solo#0|user"}
    _, lo_b, hi_b, unseen_b = f5["before"]
    assert hi_b < 1 and not unseen_b
    mean0, lo0, hi0, unseen0 = f5["no_psrc"]
    assert not unseen0
    assert mean > mean0 >= f5["before"][0]  # equal without fsrc
    assert math.log(hi / lo) > 1.5 * math.log(hi0 / lo0)


def test_the_users_own_runs_of_a_graph_set_its_position_x_source_node(f5):
    state = f5["learned"]
    assert "psrc:plan_implement#1|user" in state.heads["cost"].index
    assert "psrc:solo#0|user" not in state.heads["cost"].index
    mean, lo, hi, unseen = _contrast(state, simdata.config(PI, OPUS), simdata.config(simdata.SOLO, OPUS))
    assert unseen == ["psrc:solo#0|user"]
    _, lo_new, hi_new, _ = _contrast(f5["psrc"], simdata.config(PI, OPUS), simdata.config(simdata.SOLO, OPUS))
    assert math.log(hi / lo) < 0.8 * math.log(hi_new / lo_new)
    assert lo < 1 < hi and abs(math.log(mean)) < 0.3


# ---------------------------------------------------------------- 3: shape keying

def test_shape_keys_follow_the_role_set():
    assert structure(simdata.config(simdata.IR, OPUS)).shape == "implement_review"  # catalog shape
    assert structure(simdata.config(PI, OPUS)).shape == "plan_implement"
    worker = Workflow("plan_implement", 1, "Plan, then a worker",
                      (Piece("plan", "planner"), Piece("implement", "worker")), ("plan_doc", "patch"),
                      (("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "patch")), Control())
    st = structure(simdata.config(worker, OPUS))
    assert st.shape == "plan_implement~planner+worker"
    assert ("topology:plan_implement~planner+worker", None, 1.0) in run_rest(st)
    rows = rows_for_config(simdata.make_task(np.random.default_rng(0), 0), simdata.config(worker, OPUS), "sweep")
    nodes = {n for n, _, _ in rows["cost"][("implement", 1)]}
    assert {"topology:plan_implement~planner+worker", "position:plan_implement~planner+worker#1",
            "psrc:plan_implement~planner+worker#1|sweep"} <= nodes
    assert shape_key("my_flow", ["planner"]) == "my_flow"  # not in the catalog: the id


def test_width_three_and_three_copies_share_one_topology_node():
    wide = structure(simdata.config(simdata.BON, OPUS))
    copies = structure(simdata.config(COPIES, OPUS))
    assert wide.shape == copies.shape == "best_of_n"
    assert run_rest(wide)[0] == run_rest(copies)[0] == ("topology:best_of_n", None, 1.0)


def test_only_cost_and_tokens_rows_carry_psrc():
    rows = rows_for_config(simdata.make_task(np.random.default_rng(0), 0), simdata.config(simdata.IR, OPUS), "user")
    assert all(any(n.startswith("psrc:implement_review#") and n.endswith("|user") for n, _, _ in r)
               for r in rows["cost"].values())
    assert not any(n.startswith("psrc:") for r in [*rows["gate"].values(), rows["run"]] for n, _, _ in r)


# ---------------------------------------------------------------- 4: additivity (lane 2A's contract)

def _weights(terms) -> Counter:
    out: Counter = Counter()
    for node, _, value in terms:
        out[node] += value
    return out


def _diff(a, b) -> dict[str, float]:
    wa, wb = _weights(a), _weights(b)
    return {n: round(wa[n] - wb[n], 12) for n in set(wa) | set(wb) if not math.isclose(wa[n], wb[n], abs_tol=1e-12)}


def test_one_pieces_setting_changes_only_its_own_rows():
    before = simdata.config(simdata.PIR, OPUS)
    after = simdata.config(simdata.PIR, OPUS, FABLE, OPUS)  # implement at fable
    task = simdata.make_task(np.random.default_rng(0), 0)
    rb, ra = rows_for_config(task, before, "user"), rows_for_config(task, after, "user")
    for key in rb["cost"]:
        assert (rb["cost"][key] == ra["cost"][key]) == (key[0] != "implement")
    n = 3
    expected = _weights(setting_terms("claude-code", "claude-fable-9", "medium", 1 / n)
                        + (("role_family:implementer|fable", None, 1.0),))
    expected.subtract(_weights(setting_terms("claude-code", "claude-opus-9", "high", 1 / n)
                               + (("role_family:implementer|opus", None, 1.0),)))
    assert _diff(ra["run"], rb["run"]) == {k: round(v, 12) for k, v in expected.items() if abs(v) > 1e-12}


def test_run_cost_is_the_sum_over_pieces_of_width_times_the_row(sim_fit):
    """A gate-free shape's expected run cost is sum over pieces of width x exp(row + sigma^2 / 2), averaged
    over the known draws with the unseen effect integrated out."""
    s = sim_fit["state"]
    task = simdata.make_task(np.random.default_rng(5), 0)
    for wf in (simdata.BON, COPIES):
        cfg = simdata.config(wf, OPUS)
        plan, tparts = s._plan(cfg), s._task_parts(task)
        h = s.heads["cost"]
        total = np.zeros(S.N_DRAWS)
        for (piece, _k), rest in plan["cost"].items():
            known, s2 = s._known(h, tparts["cost"], rest)
            total += plan["st"].widths[piece] * np.exp(known + 0.5 * s2 + 0.5 * h.sigma ** 2)
        assert math.isclose(s.predict(task, cfg).cost.usd.mean, float(total.mean()), rel_tol=1e-9)


# ---------------------------------------------------------------- 6: a new user's recommendation

def test_a_new_users_pick_is_not_decided_by_the_source_gap(tmp_path_factory, monkeypatch):
    """A new user, a shipped-style bundle (PI only in the cheap source) and a score target. PI costs twice
    solo in truth, for the same score. PI stays a candidate, unflagged, and the pick is solo; without psrc
    the source gap makes PI the pick, with a narrower cost interval (7.5x from lo to hi, 12x with fsrc,
    against 4.3x here)."""
    from loopmath.recommend.engine import recommend
    from loopmath.recommend.engine import Settings

    # The pair is ranked with the search off; with it on (spec 05 section 1a), the search's finds replace the
    # catalog candidate and the pick is still solo.

    rule = AcceptanceRule("perf>=2000", "perf reaches 2000", requires=(), score=ScoreTarget("perf", 2000.0, "higher"))
    task = simdata.make_task(np.random.default_rng(9), 0, n_tasks=40)
    pi, solo = simdata.config(PI, OPUS), simdata.config(simdata.SOLO, OPUS)
    docs = _bundle(score=SCORE)

    def pick(state, search=False):
        rec = recommend(state, task, rule, usual=solo, usual_from="default", configs=[(pi, "catalog")],
                        settings=Settings(search=search))
        return rec, {c.config.id: c for c in rec.candidates}

    state = _fit(tmp_path_factory, docs)
    rec, cands = pick(state)
    assert pi.id in cands and cands[pi.id].origin == "catalog"
    assert not any("source" in note for note in rec.notes)
    assert rec.default.config.id == solo.id
    rec_s, _ = pick(state, search=True)
    assert rec_s.search is not None and rec_s.default.config.id == solo.id
    assert not any("source" in note for note in rec_s.notes)
    # psrc alone (fsrc kept) widens PI's interval (12x against 10.2x), but fsrc alone keeps the pick at
    # solo; "without psrc" above is the rows as before 0.2, psrc and fsrc stripped (4.3x, PI picked)
    wide = cands[pi.id].prediction.cost.usd
    with pytest.MonkeyPatch.context() as mp:
        _strip(mp, "psrc:")
        _, cands0 = pick(_fit(tmp_path_factory, docs))
    alone = cands0[pi.id].prediction.cost.usd
    assert wide.hi / wide.lo > alone.hi / alone.lo
    _strip(monkeypatch, *BEFORE_02)
    rec_b, cands_b = pick(_fit(tmp_path_factory, docs))
    assert rec_b.default.config.id == pi.id
    narrow = cands_b[pi.id].prediction.cost.usd
    assert wide.hi / wide.lo > 1.5 * narrow.hi / narrow.lo


# ---------------------------------------------------------------- 8: role groups

def test_copy_groups_are_pieces_of_one_role_with_no_path_between_them():
    st = structure(simdata.config(COPIES, OPUS))
    assert st.copies == {"implement-1": 3, "implement-2": 3, "implement-3": 3, "select": 1}
    assert structure(simdata.config(simdata.IR, OPUS)).copies == {"implement": 1, "review": 1}
    chain = Workflow("twice", 1, "Implement twice in a row",
                     (Piece("first", "implementer"), Piece("second", "implementer")), ("p1", "p2"),
                     (("first", "p1"), ("p1", "second"), ("second", "p2")), Control())
    assert structure(simdata.config(chain, OPUS)).copies == {"first": 1, "second": 1}
    assert copy_groups(["a", "b", "c"], {"a": "worker", "b": "worker", "c": "reviewer"},
                       {"a": {"c"}, "b": {"c"}}) == {"a": 2, "b": 2, "c": 1}


def test_uniform_copies_and_one_wide_piece_differ_only_in_the_width_control():
    wide, copies = run_rest(structure(simdata.config(simdata.BON, OPUS))), run_rest(structure(simdata.config(COPIES, OPUS)))
    assert _diff(wide, copies) == {"control:width": round(math.log2(3), 12)}
    w = _weights(copies)
    assert math.isclose(w["role:implementer"], 1.0) and math.isclose(w["role_family:implementer|opus"], 1.0)
    assert math.isclose(w["family:opus"], 1.0) and math.isclose(w["effort:high"], 1.0)


def test_one_copys_setting_changes_its_setting_terms_and_its_share_of_role_x_family():
    before = run_rest(structure(simdata.config(COPIES, OPUS)))
    after = run_rest(structure(simdata.config(COPIES, FABLE, OPUS, OPUS, OPUS)))  # implement-1 at fable
    expected = _weights(setting_terms("claude-code", "claude-fable-9", "medium", 1 / 4)
                        + (("role_family:implementer|fable", None, 1 / 3),))
    expected.subtract(_weights(setting_terms("claude-code", "claude-opus-9", "high", 1 / 4)
                               + (("role_family:implementer|opus", None, 1 / 3),)))
    assert _diff(after, before) == {k: round(v, 12) for k, v in expected.items() if abs(v) > 1e-12}
