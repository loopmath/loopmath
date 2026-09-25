"""The workflow search against brute force on a synthetic fit (spec 05 section 1a).

Every configuration of a small space is predicted one by one; the search must find the same (run cost, chance)
front, every configuration that is best in some draw, and, after the polish, the same curve rows and default pick.
"""

from __future__ import annotations

import numpy as np
import pytest

from loopmath.recommend import search as S
from loopmath.recommend.curve import LEVELS, curve, default_pick, rescue_cost
from loopmath.recommend.engine import polish, predict_with_medians
from loopmath.types import Setting
from loopmath.workflows.ids import make_config

import search_synth as H


@pytest.fixture(scope="module")
def fs(tmp_path_factory):
    return H.synth_fit(tmp_path_factory)["state"]


def sweep_and_copies():
    """A tests gate inside the repair loop (members' round weights differ), and three implementer copies."""
    return [make_config(wf, {p.id: H.SETTINGS[0] for p in wf.pieces}) for wf in (H.SD.SWEEP, H.COPIES)]


def small() -> S.Space:
    """The builder shapes without a plan in front of a best of n or a team, and SWEEP: about 1,000 configurations."""
    return H.space(lambda c: not c.key.startswith(("plan_best_of_n", "plan_team")), extra=sweep_and_copies()[:1])


def check(fs, rule, sp: S.Space) -> dict:
    """E1 to E3; returns what the caller may check further."""
    rescue = rescue_cost("redo_usual", fs.predict(H.TASK, H.USUAL, rule))
    found = S.search(fs, H.TASK, rule, rescue.usd, sp, draws=200, per_objective=40)
    assert found.stats["exact"] and found.stats["skipped"] == []
    b = H.brute(fs, H.TASK, rule, rescue.usd, sp)
    ids, idx = b["ids"], {cid: i for i, cid in enumerate(b["ids"])}
    assert len(ids) == len(idx) == sp.configurations()

    # E1: the exact (run cost, u) front, with its costs and chances
    brute_front = {ids[i] for i in np.flatnonzero(S.front_mask(b["cost"], b["u"]))}
    assert found.on_front == brute_front
    for cfg, cost, g in found.front:
        i = idx[cfg.id]
        assert cost == pytest.approx(b["cost"][i], rel=1e-9)
        assert g == pytest.approx(float(b["obj"].g(b["u"][i])), abs=1e-9)

    # E2: every draw's optimum for every objective is among the draw winners
    winners = H.draw_winners(b, S.draw_index(200))
    won: dict[str, set[str]] = {}
    for cid, w in found.wins.items():
        for k in w:
            won.setdefault(k, set()).add(cid)
    for key in S.OBJECTIVES:
        assert winners[key] <= won.get(key, set()), key
    shares = {k: sum(w.get(k, 0.0) for w in found.wins.values()) for k in S.OBJECTIVES}
    assert shares["default"] == pytest.approx(1.0, abs=1e-5)

    # E3: the rescored union and the polish give the brute-force curve rows and default pick
    union = {c.id: c for c, _ in found.configs}
    if H.USUAL.id in idx:  # the recommender always keeps the usual; its shape is in the space there
        union[H.USUAL.id] = H.USUAL
    got = fs._predict_many(H.TASK, list(union.values()), rule, rescue.usd)
    preds = [p for p, _ in got]
    g = {p.config: d["g"] for p, d in got}

    def predict(cfgs):
        out = fs._predict_many(H.TASK, cfgs, rule, rescue.usd)
        g.update({p.config: d["g"] for p, d in out})
        return [p for p, _ in out]

    n0 = len(union)
    preds += S.polish(sp, union, preds, predict, lambda ps: H.curve_and_pick(ps, g)[1])
    assert all(cid in idx for cid in list(union)[n0:])  # the polish stays in the space
    rows_s, pick_s = H.curve_and_pick(preds, g)
    rows_b, pick_b = H.curve_and_pick(b["preds"], g)
    assert pick_s == pick_b
    assert rows_s == rows_b
    return {"found": found, "brute": b, "levels": [lv for lv in LEVELS if rows_b[lv] is not None]}


@pytest.mark.parametrize("rule", [H.RULE, H.PERF_HIGH, H.PERF_LOW], ids=["success", "score_higher", "score_lower"])
def test_the_search_finds_the_brute_force_front_draw_optima_and_picks(fs, rule):
    sp = small()
    assert any(c.judged is not None and len(c.loop) > 1 for c in sp.cases)
    obj = S.objective_for(fs, H.TASK, rule, 1.0)
    assert obj.head == ("success" if rule.score is None else "score:perf")
    assert obj.sign == (-1.0 if rule is H.PERF_LOW else 1.0)
    out = check(fs, rule, sp)
    assert out["levels"], "the space reaches no curve level; the check would be empty"


def test_a_tests_gate_in_the_loop_is_searched_per_judged_setting(fs):
    """SWEEP's loop has a gate after implement and after review, so review is reached with a chance that
    depends on implement's setting: the per-J merge, exact against brute force."""
    sp = H.space(lambda c: c.origin == "recorded", extra=sweep_and_copies()[:1])
    (case,) = sp.cases
    assert case.judged == "implement" and set(case.loop) == {"implement", "review"}
    rescue = rescue_cost("redo_usual", fs.predict(H.TASK, H.USUAL, H.RULE))
    obj = S.objective_for(fs, H.TASK, H.RULE, rescue.usd)
    tab = S.Tables(fs, H.TASK, obj, sp.settings, S.draw_index(200)).case_tables(case)
    assert not tab["shared_B"]
    out = check(fs, H.RULE, sp)
    # the stage-2 cost bound is below the cheapest configuration's run cost in every draw, by brute force
    runs = np.stack([d["run"].exp_usd for d in out["brute"]["draws"]])[:, S.draw_index(200)]
    c_lo, _ = S.case_bounds(case, tab)
    assert np.all(c_lo <= runs.min(axis=0) + 1e-9)


@pytest.mark.parametrize("rule", [H.RULE, H.PERF_HIGH, H.PERF_LOW], ids=["success", "score_higher", "score_lower"])
def test_every_cases_bounds_are_below_its_optimum_in_every_draw(fs, rule):
    """The branch and bound skips a case on a draw when its bound does not beat the incumbent, so the bound must
    be at or below the case's own optimum for every objective and draw: with and without a gate inside the loop
    (members' round factors differ, so J's factor does not stand in for theirs)."""
    sp = small()
    rescue = rescue_cost("redo_usual", fs.predict(H.TASK, H.USUAL, rule))
    obj = S.objective_for(fs, H.TASK, rule, rescue.usd)
    tables = S.Tables(fs, H.TASK, obj, sp.settings, S.draw_index(200))
    shared = set()
    for case in sp.cases:
        tab = tables.case_tables(case)
        if case.judged is not None:
            shared.add(tab["shared_B"])
        bound = S.bound_values(obj, *S.case_bounds(case, tab))
        opt, _ = S.case_thompson(case, tab, obj)
        for key, (val, _) in opt.items():
            assert np.all(bound[key] <= val + 1e-9 * np.maximum(1.0, np.abs(val))), (case.key, key)
    assert shared == {True, False}


@pytest.mark.parametrize("rule", [H.RULE, H.PERF_HIGH], ids=["success", "score_higher"])
def test_the_branch_and_bound_keeps_every_draw_winner_when_a_gated_loop_competes(fs, rule, monkeypatch):
    """SWEEP (a gate inside its loop) against the builder shapes on two settings: a loop bound that took J's round
    factor for every member skipped SWEEP on draws it wins. The pruned search keeps exactly the winners of an
    unpruned one, and E1 to E3 hold against brute force."""
    sets = [H.SETTINGS[0], H.SETTINGS[2]]
    sweep = make_config(H.SD.SWEEP, {p.id: sets[0] for p in H.SD.SWEEP.pieces})
    sp = H.space(extra=[sweep], settings=sets)
    rescue = rescue_cost("redo_usual", fs.predict(H.TASK, H.USUAL, rule))
    pruned = S.search(fs, H.TASK, rule, rescue.usd, sp, draws=200, per_objective=100)
    with monkeypatch.context() as mp:
        mp.setattr(S, "case_bounds", lambda case, tab: (np.full(tab["u0"][1].shape, -np.inf),
                                                        np.full(tab["u0"][1].shape, np.inf)))
        unpruned = S.search(fs, H.TASK, rule, rescue.usd, sp, draws=200, per_objective=100)
    assert pruned.wins == unpruned.wins
    check(fs, rule, sp)


@pytest.mark.parametrize("copies", S.COPY_RULES)
def test_copies_are_searched_once_per_mix_under_both_rules(fs, copies):
    sp = H.space(lambda c: c.origin != "builder", extra=sweep_and_copies()[1:], copies=copies)
    assert [c.origin for c in sp.cases] == ["recorded", "width_form"]
    out = check(fs, H.RULE, sp)
    copies_found = 0
    for cfg, _ in out["found"].configs:
        if "implement-1" not in cfg.settings:  # the width form: one implement piece of width 3
            assert cfg.workflow.pieces[1].width == 3
            continue
        copies_found += 1
        chosen = [cfg.settings[p] for p in ("implement-1", "implement-2", "implement-3")]
        mix = S.canonical_mix(sp.settings, chosen, copies)
        assert mix is not None, "copies that do not differ belong to the width form"
        assert [sp.settings[i] for i in mix] == chosen, "copies are in copy order"
    assert copies_found


def curve_rows(preds) -> dict[int, str]:
    """The recommender's curve: reached level -> configuration id."""
    return {lv: r.config for r in curve(preds) if r.reached for lv in r.levels}


def test_every_reached_row_is_polished_not_only_the_default_pick(fs):
    """Spec 05 section 1a step 4, with the case built from brute force so it does not hang on the fit's posterior
    (without fsrc no small space here puts a row off the plug-in front). Start from every configuration but one
    row's, R: that row falls to a configuration one piece from R, and R is not one piece from the default pick, so
    the default pick's polish misses it. The recommender's polish (the default pick, then every reached row, no
    goal) finds R, and every row and the default pick match brute force."""
    unseen = [Setting("claude-code", "claude-haiku-9", "high"), Setting("codex", "gpt-9-nova", "high")]
    sp = H.space(settings=H.SETTINGS[4:] + unseen)
    tried = 0
    for rule in (H.PERF_LOW, H.RULE, H.PERF_HIGH):
        rescue = rescue_cost("redo_usual", fs.predict(H.TASK, H.USUAL, rule))
        b = H.brute(fs, H.TASK, rule, rescue.usd, sp)
        rows_b, pick_b = curve_rows(b["preds"]), default_pick(b["preds"]).config

        def predict(cfgs, rule=rule, rescue=rescue):
            return predict_with_medians(fs, H.TASK, cfgs, rule, rescue)[0]

        cfg = {c.id: c for c in b["configs"]}
        near_pick = {c.id for c in S.neighbors(cfg[pick_b], sp)}
        for lv, r in sorted(rows_b.items(), reverse=True):
            start = [p for p in b["preds"] if p.config != r]
            fell = curve_rows(start).get(lv)
            tried += 1
            if r == pick_b or r in near_pick or fell is None or r not in {c.id for c in S.neighbors(cfg[fell], sp)}:
                continue
            default_only = {cid: c for cid, c in cfg.items() if cid != r}
            S.polish(sp, default_only, start, predict, lambda ps: default_pick(ps).config)
            assert r not in default_only, "R is not one piece from the default pick, so its polish misses R"
            by_id = {cid: c for cid, c in cfg.items() if cid != r}
            n, polished = polish(fs, H.TASK, rule, rescue, sp, by_id, start, {}, None)
            assert r in by_id, f"the {lv}% row fell one piece from R and the rows' polish did not reach R"
            assert curve_rows(polished) == rows_b
            assert default_pick(polished).config == pick_b
            assert n == 1 and set(by_id) == set(cfg)  # R alone is new, and the polish stays in the space
            return
    pytest.fail(f"none of {tried} rows is one piece from where it falls and out of the default pick's reach; "
                "the fit moved, so try other settings")
