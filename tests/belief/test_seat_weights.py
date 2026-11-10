"""Seat weights of the success and score rows (0.4.2, `design.seat_weights`): the writer at 1, the planner at 0.22
and the reviewers at 0.11 between them, in place of 1/n; `average` keeps 1/n, and a fit is read with the rule it
was made under. The three numbers are a placeholder from one experiment on 8 coding tasks."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest
import simdata

from loopmath.belief import design as D
from loopmath.belief import fit as F
from loopmath.belief.state import load
from loopmath.types import AcceptanceRule, Configuration, Control, Piece, Setting, Workflow

NOW = datetime.fromisoformat("2026-09-23T12:00:00-07:00")
STRONG, WEAK, THIRD, FOURTH = simdata.SETTINGS[0], simdata.SETTINGS[5], simdata.SETTINGS[3], simdata.SETTINGS[4]
RULE = AcceptanceRule(name="tests", definition="tests pass")
SETTING_LEVELS = ("provider:", "family:", "model:", "effort:", "family_effort:", "harness:")


def _workflow(wid: str, *pieces: tuple[str, str], width: dict[str, int] | None = None) -> Workflow:
    """A chain of pieces `(id, role)`, each handing one artifact to the next."""
    ids = [p for p, _ in pieces]
    edges = []
    for a, b in zip(ids, ids[1:]):
        edges += [(a, f"{a}_out"), (f"{a}_out", b)]
    edges.append((ids[-1], f"{ids[-1]}_out"))
    return Workflow(wid, 1, wid, tuple(Piece(p, r, width=(width or {}).get(p, 1)) for p, r in pieces),
                    tuple(f"{p}_out" for p in ids), tuple(edges), Control())


PIRR = _workflow("plan_implement_review2", ("plan", "planner"), ("implement", "implementer"),
                 ("review-1", "reviewer"), ("review-2", "reviewer"))
TWO_WRITERS = _workflow("implement_fix", ("implement", "implementer"), ("fix", "implementer"), ("review", "reviewer"))
NO_WRITER = _workflow("plan_review", ("plan", "planner"), ("review", "reviewer"))
TESTED = _workflow("implement_test", ("implement", "implementer"), ("test", "tester"))
SWARM = _workflow("swarm", ("plan", "planner"), ("work", "worker"), ("review", "reviewer"), width={"work": 3})


def _st(wf: Workflow, *settings: Setting) -> D.Structure:
    return D.structure(simdata.config(wf, *(settings or (STRONG,))))


def _weights(wf: Workflow, seats: str = D.SEAT_RULE) -> dict[str, float]:
    return D.seat_weights(_st(wf), seats)


def test_the_role_rule_is_writer_1_planner_022_reviewers_011():
    assert D.SEAT_RULE == "role-v1" and D.SEAT_RULES["role-v1"] == {"implementer": 1.0, "planner": 0.22,
                                                                      "reviewer": 0.11}
    assert _weights(simdata.SOLO) == {"implement": 1.0}
    assert _weights(simdata.IR) == {"implement": 1.0, "review": 0.11}
    assert _weights(simdata.PIR) == {"plan": 0.22, "implement": 1.0, "review": 0.11}
    # two reviewers: their mean counts 0.11, so each counts 0.055; the weights are not rescaled to sum to 1
    assert _weights(PIRR) == {"plan": 0.22, "implement": 1.0, "review-1": 0.055, "review-2": 0.055}
    assert sum(_weights(PIRR).values()) == pytest.approx(1.33)


def test_average_is_one_over_the_number_of_pieces():
    for wf in (simdata.SOLO, simdata.IR, simdata.PIR, PIRR, simdata.BON, SWARM):
        n = len(wf.pieces)
        assert _weights(wf, D.SEAT_AVERAGE) == {p.id: 1.0 / n for p in wf.pieces}
    assert _weights(simdata.PIR, "a rule this loopmath does not know") == _weights(simdata.PIR, D.SEAT_AVERAGE)


def test_pieces_of_one_role_share_its_weight():
    assert _weights(TWO_WRITERS) == {"implement": 0.5, "fix": 0.5, "review": 0.11}
    wide = _workflow("wide", ("implement", "implementer"), ("review", "reviewer"), width={"implement": 3})
    assert _weights(wide) == {"implement": 1.0, "review": 0.11}  # one wide piece is one writer


@pytest.mark.parametrize("wf", [simdata.BON, SWARM, TESTED, NO_WRITER], ids=lambda wf: wf.id)
def test_a_workflow_the_rule_does_not_cover_keeps_the_average(wf):
    """A piece of a role the rule does not name (`referee`, `worker`, `tester`), or no writer: 1/n, as in 0.4.1."""
    st = _st(wf, STRONG, WEAK)
    assert D.seat_weights(st) == D.seat_weights(st, D.SEAT_AVERAGE) == {p: 1.0 / len(st.pieces) for p in st.pieces}
    assert D.run_rest(st) == D.run_rest(st, D.SEAT_AVERAGE)


@pytest.mark.parametrize("role", ["implementer", "planner", "reviewer", "tester", "referee", "worker", "editor"])
def test_one_piece_is_at_1_whatever_its_role(role):
    st = _st(_workflow("one", ("only", role)))
    assert D.seat_weights(st) == D.seat_weights(st, D.SEAT_AVERAGE) == {"only": 1.0}
    assert D.run_rest(st) == D.run_rest(st, D.SEAT_AVERAGE)


def _by_node(terms) -> dict[str, float]:
    out: dict[str, float] = {}
    for node, _, value in terms:
        out[node] = out.get(node, 0.0) + value
    return out


def test_the_run_row_carries_each_setting_at_its_seat_weight_and_the_rest_unchanged():
    st = _st(PIRR, THIRD, STRONG, WEAK, FOURTH)  # planner, writer and two reviewers on four different models
    row, old = _by_node(D.run_rest(st)), _by_node(D.run_rest(st, D.SEAT_AVERAGE))
    want = {"plan": 0.22, "implement": 1.0, "review-1": 0.055, "review-2": 0.055}
    by_rule: dict[str, float] = {}
    by_average: dict[str, float] = {}
    for piece, w in want.items():  # pieces that share a provider or a harness add up on its node
        h, m, e = D._s(st.settings[piece])
        for node, _, _ in D.setting_terms(h, m, e):
            by_rule[node] = by_rule.get(node, 0.0) + w
            by_average[node] = by_average.get(node, 0.0) + 0.25
    models = [n for n in by_rule if n.startswith("model:")]
    assert sorted(by_rule[n] for n in models) == [0.055, 0.055, 0.22, 1.0]  # four models, one per piece
    for node in by_rule:
        assert row[node] == pytest.approx(by_rule[node]), node
        assert old[node] == pytest.approx(by_average[node]), node
    # topology, control, role, role x family and step terms do not depend on the seat rule
    rest = {n: v for n, v in row.items() if not n.startswith(SETTING_LEVELS)}
    assert rest == {n: v for n, v in old.items() if not n.startswith(SETTING_LEVELS)}
    assert (rest["role:planner"], rest["role:implementer"], rest["role:reviewer"]) == (1.0, 1.0, 2.0)  # 1 a piece
    assert set(row) == set(old)  # the same nodes, so a fit of either rule has a column for every term


def test_harness_factors_follow_the_seat_weight():
    cfg = simdata.config(simdata.PIR, STRONG)
    st = D.structure(cfg, ("skill:a",))
    row, old = _by_node(D.run_rest(st)), _by_node(D.run_rest(st, D.SEAT_AVERAGE))
    node = D.factor_node(STRONG.harness, "skill:a")
    assert row[node] == pytest.approx(1.33) and row[f"harness:{STRONG.harness}"] == pytest.approx(1.33)
    assert old[node] == pytest.approx(1.0) and old[f"harness:{STRONG.harness}"] == pytest.approx(1.0)


def test_a_strong_writer_with_a_weak_reviewer_reads_above_the_reverse():
    """With one model ahead of another and nothing else, the role rule ranks the workflow by its writer; the
    average cannot tell the two apart."""
    strong, weak = (D.model_path(D.canonical_model_id(s.model))[2] for s in (STRONG, WEAK))
    effect = {f"model:{strong}": 1.0, f"model:{weak}": -1.0}

    def eta(writer: Setting, reviewer: Setting, seats: str) -> float:
        st = D.structure(Configuration("c", simdata.IR, {"implement": writer, "review": reviewer}))
        return sum(effect.get(node, 0.0) * value for node, _, value in D.run_rest(st, seats))

    assert eta(STRONG, WEAK, D.SEAT_RULE) == pytest.approx(0.89)
    assert eta(WEAK, STRONG, D.SEAT_RULE) == pytest.approx(-0.89)
    assert eta(STRONG, WEAK, D.SEAT_AVERAGE) == eta(WEAK, STRONG, D.SEAT_AVERAGE) == 0.0


def test_a_solo_row_is_the_same_under_both_rules():
    task = simdata.make_task(np.random.default_rng(0), 0)
    for cfg in (c for c in simdata.all_configs() if c.workflow.id == "solo"):
        st = D.structure(cfg)
        assert D.run_rest(st, D.SEAT_RULE) == D.run_rest(st, D.SEAT_AVERAGE)
        assert D.run_row(task, "user", st) == D.run_row(task, "user", st, seats=D.SEAT_AVERAGE)
        new, old = (D.rows_for_config(task, cfg, seats=s) for s in (D.SEAT_RULE, D.SEAT_AVERAGE))
        assert (new["run"], new["cost"], new["gate"]) == (old["run"], old["cost"], old["gate"])


def test_cost_and_gate_rows_do_not_read_the_seat_rule():
    task = simdata.make_task(np.random.default_rng(0), 0)
    cfg = simdata.config(simdata.PIR, STRONG, WEAK)
    new, old = (D.rows_for_config(task, cfg, seats=s) for s in (D.SEAT_RULE, D.SEAT_AVERAGE))
    assert new["cost"] == old["cost"] and new["gate"] == old["gate"] and new["run"] != old["run"]


def test_the_config_key_chooses_the_rule_and_follows_the_level_when_unset():
    assert D.seat_rule({}) == D.seat_rule(None) == D.seat_rule({"belief": {"level": "store"}}) == "role-v1"
    assert D.seat_rule({"belief": {"seat_weights": "average"}}) == "average"
    assert D.seat_rule({"belief": {"level": "shared"}}) == "average"  # `shared` stays 0.3.5's fit
    assert D.seat_rule({"belief": {"level": "shared", "seat_weights": "role"}}) == "role-v1"
    assert D.seat_rule({"belief": {"seat_weights": "thirds"}}) == "role-v1"  # malformed: the default
    assert D.seat_rule({"belief": "store"}) == "role-v1"


def test_config_set_takes_the_two_values_and_refuses_another(tmp_path, capsys, monkeypatch):
    from loopmath.cli import main
    from loopmath.store.config import CHOICES, KEYS

    assert CHOICES["belief.seat_weights"] == ("role", "average") and "belief.seat_weights" in KEYS
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path))
    for value in ("average", "role"):
        assert main(["config", "set", "belief.seat_weights", value]) == 0
        assert f'seat_weights = "{value}"' in (tmp_path / "config.toml").read_text(encoding="utf-8")
    capsys.readouterr()
    assert main(["config", "set", "belief.seat_weights", "writer"]) != 0
    assert "belief.seat_weights" in capsys.readouterr().err
    assert 'seat_weights = "role"' in (tmp_path / "config.toml").read_text(encoding="utf-8")


# loopmath 0.4.1's fit of `_solo_store`: single-piece runs only, the default config, a two-result benchmark table
# and no shipped runs. Per head: its nodes, the sum of the means and the sum of the draws; then two predictions
# (configuration, chance of success with its range, mean dollars)
GOLDEN_041_SOLO = {"seed_key": "c727cdc8e41257799e305ca8b9f6efe360fd5b95f3631def577068d81cf1547c",
                   "cost": (76, -1.519508296030857, -607.803318412343),
                   "gate": (0, 0.0, 0.0),
                   "rate": (76, -16.303977866237737, -6521.591146495094),
                   "success": (78, 0.24027440889725707, 96.10976355890278),
                   "predict": (("cfg_5fdc3482e806", 0.14821505696245207, 0.02420550066596439, 0.3507603323522399,
                                1.1426372045766413),
                               ("cfg_bf5ad529f195", 0.37161914211348324, 0.07182913232843588, 0.7492448227625546,
                                0.03806089174170237))}


def _solo_store(tmp_path):
    table = tmp_path / "benchmarks.toml"
    table.write_text("\n".join([
        '[[benchmark]]\nid = "tb"\nmetric = "pass rate"\nkind = "success"\nreference = "claude-opus-9"\n'
        'reference_effort = "max"\nruns = 200\n',
        '[[result]]\nbenchmark = "tb"\nmodel = "claude-opus-9"\neffort = "max"\nvalue = 0.6\n',
        '[[result]]\nbenchmark = "tb"\nmodel = "gpt-9-astra"\neffort = "high"\nvalue = 0.7\n']), encoding="utf-8")
    solo = [c for c in simdata.all_configs() if c.workflow.id == "solo"]
    docs, _ = simdata.simulate(60, seed=11, source="live", n_tasks=20, configs=solo)
    return table, solo, docs


def _config(home, **keys):
    home.mkdir(parents=True, exist_ok=True)
    lines = ["[belief]"] + [f"{k} = {json.dumps(v)}" for k, v in keys.items()]
    (home / "config.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_store_of_solo_runs_fits_and_predicts_as_041(tmp_path):
    """Solo workflows are unchanged: on a store with single-piece runs only, the fit, its draws and its solo
    predictions are 0.4.1's to the digit, under the role rule (the default) and under `average` alike."""
    table, solo, docs = _solo_store(tmp_path)
    task = simdata.make_task(np.random.default_rng(0), 0, n_tasks=20)
    _config(tmp_path / "average", seat_weights="average")
    states = {}
    for name in ("default", "average"):
        path = F.fit(tmp_path / name, docs=docs, benchmarks=table, now=NOW, bundle_dir=tmp_path / "none")
        st = states[name] = load(path)
        assert st.meta["seed_key"] == GOLDEN_041_SOLO["seed_key"], name
        assert sorted(st.heads) == ["cost", "gate", "rate", "success"]
        for head in ("cost", "gate", "rate", "success"):
            nodes, mean, draws = GOLDEN_041_SOLO[head]
            h = st.heads[head]
            assert len(h.index) == nodes, (name, head)
            assert float(np.sum(h.mean)) == pytest.approx(mean, rel=1e-9, abs=1e-9), (name, head)
            assert float(np.sum(h.draws)) == pytest.approx(draws, rel=1e-9, abs=1e-9), (name, head)
        for cfg, want in zip((solo[0], solo[-1]), GOLDEN_041_SOLO["predict"]):
            p = st.predict(task, cfg)
            got = (cfg.id, p.p_success.mean, p.p_success.lo, p.p_success.hi, p.cost.usd.mean)
            assert got == pytest.approx(want, rel=1e-9), name
    assert states["default"].seat_rule == "role-v1" and states["average"].seat_rule == "average"
    for head in ("cost", "gate", "rate", "success"):  # the two rules give one fit here, bit for bit
        a, b = states["default"].heads[head], states["average"].heads[head]
        assert np.array_equal(a.mean, b.mean) and np.array_equal(a.draws, b.draws), head


def test_a_fit_records_its_rule_and_older_loopmath_refuses_a_role_fit_by_name(tmp_path, monkeypatch):
    docs, _ = simdata.simulate(60, seed=2, source="live", n_tasks=20)
    metas = {}
    for name, keys in (("default", {}), ("average", {"seat_weights": "average"}), ("shared", {"level": "shared"}),
                       ("shared_role", {"level": "shared", "seat_weights": "role"})):
        _config(tmp_path / name, **keys)
        path = F.fit(tmp_path / name, docs=docs, no_prior=True, now=NOW)
        metas[name] = json.loads((path / "meta.json").read_text())
        assert D.other_design(metas[name]) is None
    assert metas["default"]["model_options"]["belief.seat_weights"] == "role-v1"
    assert metas["default"]["design_version"] == D.OPT_IN_DESIGN_VERSION
    # `average` writes what 0.4.1 writes: no seat option, and under `shared` no options at all (version 4)
    assert "belief.seat_weights" not in metas["average"]["model_options"]
    assert metas["shared"]["design_version"] == D.DESIGN_VERSION and "model_options" not in metas["shared"]
    assert metas["shared_role"]["model_options"] == {"belief.seat_weights": "role-v1"}
    assert metas["shared_role"]["design_version"] == D.OPT_IN_DESIGN_VERSION
    # multi-piece runs: the rule moves the success head's estimates and no other head's (the seed key is over
    # every row, so the other heads' random draws are new, around the same means)
    new, old = load(tmp_path / "default" / "fits" / "latest"), load(tmp_path / "average" / "fits" / "latest")
    assert new.meta["seed_key"] != old.meta["seed_key"]
    assert not np.array_equal(new.heads["success"].mean, old.heads["success"].mean)
    for head in ("cost", "rate", "gate"):
        assert np.array_equal(new.heads[head].mean, old.heads[head].mean), head
    # a value this loopmath does not know, and a loopmath without the option (0.4.1), refuse the fit by name
    other = {**metas["default"], "model_options": {**metas["default"]["model_options"],
                                                   "belief.seat_weights": "role-v9"}}
    assert "`belief.seat_weights` = \"role-v9\"" in D.other_design(other)
    monkeypatch.delitem(D.MODEL_OPTIONS, "belief.seat_weights")
    why = D.other_design(metas["default"])
    assert "`belief.seat_weights`" in why and "run `loopmath fit` for a new fit" in why
    assert D.other_design(metas["average"]) is None


def test_a_fit_predicts_with_the_rule_it_was_made_under(tmp_path):
    """A fit from before 0.4.2 has no seat option: it keeps 1/n rows until the next fit, in `predict` and in
    `recommend`'s search tables alike."""
    from loopmath.recommend import search

    docs, _ = simdata.simulate(60, seed=2, source="live", n_tasks=20)
    _config(tmp_path / "average", seat_weights="average")
    new = load(F.fit(tmp_path / "default", docs=docs, no_prior=True, now=NOW))
    old = load(F.fit(tmp_path / "average", docs=docs, no_prior=True, now=NOW))
    assert (new.seat_rule, old.seat_rule) == ("role-v1", "average")
    assert D.fit_seat_rule({}) == D.fit_seat_rule({"model_options": None}) == "average"
    assert D.fit_seat_rule({"model_options": {"belief.seat_weights": "role-v9"}}) == "average"
    cfg = simdata.config(simdata.PIR, STRONG, WEAK)
    st = D.structure(cfg)
    assert new._plan(cfg)["run"] == D.run_rest(st, "role-v1") != D.run_rest(st, "average") == old._plan(cfg)["run"]
    task = simdata.make_task(np.random.default_rng(0), 0, n_tasks=20)
    for state in (new, old):
        obj = search.objective_for(state, task, RULE, 1.0)
        tables = search.Tables(state, task, obj, list(simdata.SETTINGS), search.draw_index(50))
        assert tables.seats == state.seat_rule
