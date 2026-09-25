"""The workflow search's parts (spec 05 section 1a): row splits that add back to the fit's rows, fronts and the
dominance prune against quadratic checks, and the copy groups, mixes and width forms."""

from __future__ import annotations

import dataclasses
import time
from collections import Counter

import numpy as np
import pytest

from loopmath.belief.design import gate_rest, structure
from loopmath.recommend import search as S
from loopmath.types import Configuration, Control, Gate, Piece, Setting, Workflow
from loopmath.workflows.ids import make_config

import search_synth as H


@pytest.fixture(scope="module")
def fs(tmp_path_factory):
    return H.synth_fit(tmp_path_factory)["state"]


def every_case_space() -> S.Space:
    extra = [make_config(wf, {p.id: H.SETTINGS[0] for p in wf.pieces}) for wf in (H.SD.SWEEP, H.COPIES)]
    return H.space(widths=(2, 3), rounds=(1, 3), extra=extra)


def random_configs(sp: S.Space, rng: np.random.Generator, per_case: int = 3) -> list[tuple[S.Case, Configuration]]:
    out = []
    for c in sp.cases:
        for _ in range(per_case):
            out.append((c, make_config(c.workflow, {p: sp.settings[int(rng.integers(len(sp.settings)))]
                                                    for p in c.st.pieces})))
    return out


# ---------------------------------------------------------------- row splits

def t(node: str, v: float = 1.0) -> tuple:
    return (node, None, v)


def test_split_rows_keeps_shared_positions_in_the_case_part():
    case, parts = S.split_rows([(t("a"), t("s1"), t("b")), (t("a"), t("s2"), t("b"))])
    assert case == (t("a"), t("b"))
    assert parts == [(t("s1"),), (t("s2"),)]


def test_split_rows_falls_back_to_a_multiset_when_lengths_differ():
    case, parts = S.split_rows([(t("a"), t("s1"), t("b")), (t("a"), t("b"), t("s2"), t("s3"))])
    assert case == (t("a"), t("b"))
    assert parts == [(t("s1"),), (t("s2"), t("s3"))]


def test_a_node_in_both_parts_moves_into_every_setting_part():
    """Unseen variances add per node: a node split across the parts would lose its cross term."""
    case, parts = S.split_rows([(t("x", 0.5), t("x", 0.25), t("a")), (t("x", 0.5), t("x", 0.75), t("a"))])
    assert case == (t("a"),)
    assert parts == [(t("x", 0.25), t("x", 0.5)), (t("x", 0.75), t("x", 0.5))]


def test_search_rows_add_up_to_the_fits_rows_for_random_configurations(fs):
    """The cost, gate and run rows the search assembles from case and setting parts are the rows
    `FitState._plan` builds for the whole configuration: a piece's cost row reads only its own setting, a gate's
    only its judged piece's, and the run row's per-piece parts do not overlap."""
    sp = every_case_space()
    obj = S.objective_for(fs, H.TASK, H.RULE, 1.0)
    tabs = S.Tables(fs, H.TASK, obj, sp.settings, S.draw_index(200))
    index = {S.setting_key(s): i for i, s in enumerate(sp.settings)}
    checked = Counter()
    for case, cfg in random_configs(sp, np.random.default_rng(5)):
        plan = fs._plan(cfg)
        for (p, k), row in plan["cost"].items():
            cpart, parts = tabs._cost_split(case.st, p, k)
            assert Counter(cpart) + Counter(parts[index[S.setting_key(cfg.settings[p])]]) == Counter(row)
            checked["cost"] += 1
        for (gi, k), row in plan["gate"].items():
            g = case.st.gates[gi]
            rows = []
            for s in sp.settings:
                with S._Varied(case.st, g.judged, s) as v:
                    rows.append(gate_rest(v, g, k))
            gpart, parts = S.split_rows(rows)
            assert Counter(gpart) + Counter(parts[index[S.setting_key(cfg.settings[g.judged])]]) == Counter(row)
            checked["gate"] += 1
        parts, rpart = tabs._run_split(case)
        run = Counter(rpart)
        for p in case.st.pieces:
            run += Counter(parts[p][index[S.setting_key(cfg.settings[p])]])
        assert run == Counter(plan["run"])
        checked["run"] += 1
    assert min(checked.values()) > 0 and set(checked) == {"cost", "gate", "run"}


def test_tables_give_each_configurations_mean_cost_and_u(fs):
    """Outside a loop, a configuration's mean run cost and u are the sums of its pieces' table entries."""
    sp = every_case_space()
    obj = S.objective_for(fs, H.TASK, H.RULE, 1.0)
    tabs = S.Tables(fs, H.TASK, obj, sp.settings, S.draw_index(200))
    index = {S.setting_key(s): i for i, s in enumerate(sp.settings)}
    cases = [c for c in sp.cases if c.judged is None]
    assert any(c.groups for c in cases)
    pairs = random_configs(S.Space(sp.settings, cases), np.random.default_rng(9))
    tp = fs._task_parts(H.TASK)["success"]
    for case, cfg in pairs:
        tab = tabs.case_tables(case)
        pick = {p: index[S.setting_key(cfg.settings[p])] for p in case.st.pieces}
        cost = sum(tab["cout"][p][0][pick[p]] for p in case.st.pieces)
        u = tab["u0"][0] + sum(tab["u"][p][0][pick[p]] for p in case.st.pieces)
        pred = fs.predict(H.TASK, cfg, H.RULE)
        row = fs._cached(fs.heads["success"], [S.run_rest(structure(cfg))])[0]
        assert cost == pytest.approx(pred.cost.usd.mean, rel=1e-9)
        assert u == pytest.approx(tp.mu + row.mu, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------- fronts and the prune

def quadratic_front(cost: np.ndarray, u: np.ndarray) -> set[tuple[float, float]]:
    pts = set(zip(cost.tolist(), u.tolist()))
    return {(c, v) for c, v in pts if not any((c2 <= c and v2 >= v) and (c2, v2) != (c, v) for c2, v2 in pts)}


def test_front_mask_is_the_non_dominated_set_with_ties_kept_once():
    rng = np.random.default_rng(1)
    for _ in range(50):
        cost = rng.integers(0, 8, 40).astype(float)
        u = rng.integers(0, 8, 40).astype(float)
        m = S.front_mask(cost, u)
        kept = list(zip(cost[m].tolist(), u[m].tolist()))
        assert len(kept) == len(set(kept))
        assert set(kept) == quadratic_front(cost, u)


def test_u_values_within_tie_are_equal():
    m = S.front_mask(np.array([1.0, 1.0, 2.0]), np.array([0.5, 0.5 + S.TIE / 10, 0.5 + S.TIE / 10]))
    assert m.sum() == 1


def test_merge_is_the_front_of_every_sum():
    rng = np.random.default_rng(2)
    a = S.Front(rng.random(30), rng.random(30), np.arange(30)[:, None]).prune()
    cb, ub = rng.random(25), rng.random(25)
    got = S.merge(a, cb, ub)
    sums_c = (a.cost[:, None] + cb[None, :]).ravel()
    sums_u = (a.u[:, None] + ub[None, :]).ravel()
    assert set(zip(got.cost.tolist(), got.u.tolist())) == quadratic_front(sums_c, sums_u)
    for c, v, (i, j) in zip(got.cost, got.u, got.idx):
        assert c == pytest.approx(a.cost[list(a.idx[:, 0]).index(i)] + cb[j])


def test_nondominated3_keeps_exactly_the_undominated_points():
    rng = np.random.default_rng(3)
    a, b, u = (rng.integers(0, 5, (20, 12)).astype(float) for _ in range(3))
    sel, ok = S.nondominated3(a, b, u)
    for d in range(20):
        want = set()
        pts = [(a[d, j], b[d, j], u[d, j]) for j in range(12)]
        for j, p in enumerate(pts):
            dominated = any(q[0] <= p[0] and q[1] <= p[1] and q[2] >= p[2] and (q != p or i < j)
                            for i, q in enumerate(pts) if i != j)
            if not dominated:
                want.add(j)
        assert set(sel[d][ok[d]].tolist()) == want


def test_every_objectives_best_is_on_the_per_draw_front():
    """Stage 2 keeps only per-draw fronts: every objective (C + (1 - g) R, or C with g at a level) is lowest
    on the front, because it rises with C and falls with u."""
    rng = np.random.default_rng(4)
    cost, u = rng.random((6, 50)), rng.normal(size=(6, 50))
    obj = S.Objective("success", 2.0)
    fc, fu, fi = S.front_rows(cost, u)
    full = S._best_per_draw(cost, u, obj)
    front = S._best_per_draw(fc, fu, obj)
    for key in S.OBJECTIVES:
        assert np.array_equal(full[key][0], front[key][0]), key


@pytest.mark.parametrize("n", [200, 100, 150, 40, 400])
def test_draw_index_takes_whole_antithetic_pairs(fs, n):
    """Stage 2's draws are pairs of a normal and its negative (the fit's `unit`), so the per-draw means of the
    known nodes average to the posterior mean; 200 is every second draw, as before the pairs."""
    idx = S.draw_index(n)
    half = S.N_DRAWS // 2
    assert len(idx) == n == len(set(idx.tolist()))
    assert {i + half for i in idx if i < half} == {i for i in idx if i >= half}
    unit = fs.heads["cost"].unit[:, idx]
    assert np.allclose(unit.mean(axis=1), 0.0, atol=1e-12)
    if n == 200:
        assert np.array_equal(idx, np.arange(0, S.N_DRAWS, 2))


# ---------------------------------------------------------------- copies and the space

A, B, C, D = (Setting("claude-code", "claude-opus-9", "high"), Setting("claude-code", "claude-opus-9", "xhigh"),
              Setting("codex", "gpt-9-astra", "xhigh"), Setting("codex", "gpt-9-sol", "high"))


def test_copy_groups_are_same_role_same_inputs_and_not_chained():
    assert S.copy_groups(H.COPIES, [p.id for p in H.COPIES.pieces]) == [("implement-1", "implement-2", "implement-3")]
    chain = Workflow("chain", 1, "Two implementers in a row",
                     (Piece("implement-1", "implementer"), Piece("implement-2", "implementer")), ("a", "b"),
                     (("implement-1", "a"), ("a", "implement-2"), ("implement-2", "b")), Control())
    assert S.copy_groups(chain, ["implement-1", "implement-2"]) == []
    wide = dataclasses.replace(H.COPIES, pieces=tuple(dataclasses.replace(p, width=2) if p.id == "implement-3" else p
                                                      for p in H.COPIES.pieces))
    assert S.copy_groups(wide, [p.id for p in wide.pieces]) == [("implement-1", "implement-2")]


def test_the_width_form_merges_the_copies_into_one_piece():
    wf = S.width_form(H.COPIES, ("implement-1", "implement-2", "implement-3"))
    assert [(p.id, p.width) for p in wf.pieces] == [("plan", 1), ("implement", 3), ("select", 1)]
    assert wf.artifacts == ("plan_doc", "diff-implement", "pick")
    assert set(wf.edges) == {("plan", "plan_doc"), ("plan_doc", "implement"), ("implement", "diff-implement"),
                             ("diff-implement", "select"), ("select", "pick")}
    assert S.belief_key(wf) != S.belief_key(H.COPIES)


def test_copy_order_is_models_in_order_then_the_highest_effort():
    assert S.copy_order([A, B, C, D]) == [1, 0, 2, 3]
    assert S.copy_order([C, A, B]) == [0, 2, 1]


@pytest.mark.parametrize("rule, n", [("model", 20 - 4 - 2), ("setting", 20 - 4)])
def test_group_combos_are_each_mix_once_whose_copies_differ(rule, n):
    sets = [A, B, C, D]
    combos = S.group_combos(sets, 3, rule)
    assert len(combos) == n == len(set(combos))
    assert S.mix_count(4, 3) == 20
    rank = {i: r for r, i in enumerate(S.copy_order(sets))}
    for combo in combos:
        assert [rank[i] for i in combo] == sorted(rank[i] for i in combo)
        assert S.copies_differ(sets, combo, rule)
    assert ((0, 0, 1) in combos or (1, 0, 0) in combos) == (rule == "setting")


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("rule", S.COPY_RULES)
def test_mixes_are_counted_without_listing_them(rule, k):
    E = Setting("codex", "gpt-9-astra", "high")
    for sets in ([A, B, C, D], [A, B, C, D, E], [C, E, A], [A, B], [D]):
        assert S.differing_mix_count(sets, k, rule) == len(S.group_combos(sets, k, rule)), [s.model for s in sets]


@pytest.mark.parametrize("k, n", [(4, 60), (6, 30)])
def test_an_oversized_copy_group_is_skipped_before_its_mixes_are_listed(monkeypatch, k, n):
    """The cap is checked on the counted mixes, so a group over it costs nothing to skip: nothing lists its
    mixes (listing four copies over 60 settings took 2 s, six over 30 took 8 s)."""
    models = [f"claude-opus-9-{i}" for i in range(n // 10)] + [f"gpt-9-{i}" for i in range(n // 10)]
    sets = [Setting("codex" if m.startswith("gpt") else "claude-code", m, e)
            for m in models for e in ("low", "medium", "high", "xhigh", "max")]
    assert len({S.setting_key(s) for s in sets}) == n
    ids = [f"implement-{i + 1}" for i in range(k)]
    wf = Workflow(f"{k}_copies", 1, f"{k} copies", tuple(Piece(p, "implementer") for p in ids), ("issue", "diff"),
                  tuple([("issue", p) for p in ids] + [(p, "diff") for p in ids]))
    cfg = make_config(wf, {p: sets[0] for p in ids})

    def listed(*a, **kw):
        raise AssertionError("a copy group's mixes were listed while building the space")

    monkeypatch.setattr(S, "group_combos", listed)
    t0 = time.perf_counter()
    sp = S.space_from([(cfg, "recorded")], settings=sets)
    assert time.perf_counter() - t0 < 1.0  # about 0.01 s
    assert S.differing_mix_count(sets, k, "model") > S.MAX_MIXES
    assert (f"{wf.id} [{cfg.id}]", f"a copy group with more than {S.MAX_MIXES:,} mixes") in sp.skipped
    assert not any(c.groups for c in sp.cases) and sp.summary()["configurations"] > 0


def test_canonical_mix_orders_copies_and_leaves_out_what_is_not_in_the_space():
    sets = [A, B, C, D]
    assert S.canonical_mix(sets, [D, A, C], "model") == (0, 2, 3)
    assert S.canonical_mix(sets, [C, D, A], "model") == (0, 2, 3)
    assert S.canonical_mix(sets, [A, B, A], "model") is None  # one model: the width form's
    assert S.canonical_mix(sets, [A, B, A], "setting") == (1, 0, 0)
    assert S.canonical_mix(sets, [A, Setting("codex", "gpt-9-luna", "low")], "model") is None


def test_space_settings_follow_the_catalog_models_and_keep_the_usuals_objects():
    usual_a = Setting("claude-code", "claude-opus-9", "high", extra={"from": "usual"})
    usual = make_config(H.SD.IR, {"implement": usual_a, "review": Setting("codex", "gpt-9-luna", "low")})
    catalog = [make_config(H.SD.SOLO, {"implement": s}) for s in (C, A, D)]
    sets = S.space_settings([(usual, "usual")] + [(c, "catalog") for c in catalog])
    assert [S.setting_key(s) for s in sets] == [S.setting_key(s) for s in (C, A, D)]
    assert sets[1] is usual_a  # the usual's object, so its id matches
    no_catalog = S.space_settings([(usual, "usual")])
    assert [s.model for s in no_catalog] == ["claude-opus-9", "gpt-9-luna"]


def test_cases_outside_the_method_are_listed_not_searched(fs, monkeypatch):
    two_loops = Workflow("two_loops", 1, "Two repair loops",
                         (Piece("implement", "implementer"), Piece("review", "reviewer"), Piece("test", "tester")),
                         ("patch", "notes", "report"),
                         (("implement", "patch"), ("patch", "review"), ("review", "notes"), ("notes", "test"),
                          ("test", "report")),
                         Control(gates=(Gate("g_review", "review", "review_approve", on_fail="implement"),
                                        Gate("g_test", "test", "tests_pass", on_fail="test")), budget_rounds=2))
    cfg = make_config(two_loops, {p.id: H.SETTINGS[0] for p in two_loops.pieces})
    try:
        S.make_case("x", two_loops, "recorded", H.SETTINGS[0])
    except S.NotExact:
        pass
    else:
        pytest.skip("this structure is inside the method")
    sp = H.space(lambda c: c.origin != "builder", extra=[cfg])
    assert [k for k, _ in sp.skipped] == [f"two_loops [{cfg.id}]"]
    found = S.search(fs, H.TASK, H.RULE, 1.0, S.Space(sp.settings, H.space(widths=(2,), rounds=(1,)).cases,
                                                      skipped=sp.skipped))
    assert found.stats["exact"] is False and found.stats["skipped"][0]["case"] == f"two_loops [{cfg.id}]"
    copies = make_config(H.COPIES, {p.id: H.SETTINGS[0] for p in H.COPIES.pieces})
    monkeypatch.setattr(S, "MAX_MIXES", 10)
    sp = H.space(extra=[copies], widths=(2,), rounds=(1,))
    assert any(k == f"plan_best_of_3_copies [{copies.id}]" and "mixes" in why for k, why in sp.skipped)
    assert not any(c.groups for c in sp.cases)


def test_neighbors_change_one_piece_and_stay_in_the_space():
    sp = H.space(extra=[make_config(H.COPIES, {p.id: H.SETTINGS[0] for p in H.COPIES.pieces})], widths=(2,),
                 rounds=(1,))
    sets = sp.settings
    cfg = make_config(H.COPIES, {"plan": sets[2], "implement-1": sets[1], "implement-2": sets[4],
                                 "implement-3": sets[5], "select": sets[3]})
    around = S.neighbors(cfg, sp)
    assert len({c.id for c in around}) == len(around) and cfg.id not in {c.id for c in around}
    single = [p for p in cfg.settings if p not in ("implement-1", "implement-2", "implement-3")]
    for n in around:
        moved = [p for p in single if n.settings[p] != cfg.settings[p]]
        mix_now = Counter(S.setting_key(n.settings[q]) for q in ("implement-1", "implement-2", "implement-3"))
        mix_was = Counter(S.setting_key(cfg.settings[q]) for q in ("implement-1", "implement-2", "implement-3"))
        assert len(moved) + (mix_now != mix_was) == 1
        chosen = [n.settings[q] for q in ("implement-1", "implement-2", "implement-3")]
        assert S.canonical_mix(sets, chosen, sp.copies) is not None
    assert len([n for n in around if n.settings["plan"] != cfg.settings["plan"]]) == len(sets) - 1
    ir = make_config(H.SD.IR, {"implement": sets[0], "review": sets[1]})
    assert len(S.neighbors(ir, sp)) == 2 * (len(sets) - 1)
    assert all(sum(n.settings[p] != ir.settings[p] for p in ir.settings) == 1 for n in S.neighbors(ir, sp))
