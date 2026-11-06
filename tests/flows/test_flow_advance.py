"""`advance` on the example flow: verdicts, the screen readout, the subset check and the single checks. Runs come
from `next_runs` specs, with qualities and dollars from a known additive truth and `quality` and `fit` given."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import warnings
import zlib
from pathlib import Path

import numpy as np
import pytest

from loopmath.flows import check as check_mod
from loopmath.flows import judges
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows.folder import FLOW_FILE, load, save_plan
from loopmath.flows.records import Experiment
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.ocp.conformance import catalog_resolver

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
FIT = {"calibrated": False, "n_person": 0, "judges": [], "s_run": None}
FITTED = {"calibrated": False, "n_person": 0, "judges": [{"id": "j-claude", "kind": "model", "noise": 0.08}],
          "s_run": 0.1}
# Changes in quality and in dollars per run, by (edit kind, piece): the shipped truth's numbers. The example's
# factors are 1 extract on Sonnet, 2 drop the critic, 3 synthesize at medium, 4 scout width 2, 5 round budget 1.
SHIPPED = {("model", "extract"): (0.0, -0.45), ("drop", "critic"): (-0.2, -0.15),
           ("effort", "synthesize"): (-0.03, -0.2), ("width", "scout"): (0.01, 0.15), ("budget", None): (0.0, 0.0),
           ("merge", None): (-0.05, -0.3)}
EXTRACT, SYNTH, BOTH = "extract-claude-sonnet-5-5", "synthesize-effort-medium", "set-1-3"
NOISY = {"sd": 0.05, "cost_sd": 0.1}  # check runs: a little noise, as in the check tests
VERDICT_ONLY = {"name": "claims-check", "definition": "the claims check passes", "requires": ["claims-check"]}


def _saved(tmp_path: Path, change=None):
    target = tmp_path / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    if change is not None:
        doc = json.loads((target / FLOW_FILE).read_text(encoding="utf-8"))
        change(doc["flow"])
        c = doc["flow"]["configuration"]
        c["id"] = config_id(c["workflow"], c["settings"], resolve=catalog_resolver)
        doc["flow"]["digest"] = flow_digest(doc["flow"])
        (target / FLOW_FILE).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    save_plan(target, plan_mod.make_plan(load(target), quality={}, fit=FIT))
    return load(target)


def _index(run_id: str) -> int:
    return int(run_id.split(".")[-2])


def _exp(plan, exp_id):
    return next(e for e in plan.experiments if e.id == exp_id)


def _level(arm, truth) -> tuple[float, float]:
    """An arm's quality and dollars per run: the base's 0.6 and 1.0 plus each of its edits' changes."""
    q, usd = 0.6, 1.0
    for e in arm.edits:
        dq, du = truth.get((e.kind, e.piece), (0.0, 0.0))
        q, usd = q + dq, usd + du
    return q, usd


def _ran(plan, specs, truth, docs=(), quality=None, *, experiment="screen", k=None, sd=0.0, cost_sd=0.0, seed=0,
         loss=None, accept=None):
    """`docs` and `quality` plus the runs of `experiment` among `specs` (pairs 1 to k), from `truth` with seeded
    noise. `loss` maps an experiment id to a quality loss its candidate runs take on top, which the screen did not
    see; `accept(arm)` gives the claims check's verdict."""
    arms = {a.id: a for a in plan.arms}
    docs, quality = list(docs), dict(quality or {})
    for s in specs:
        if s.experiment != experiment or (k is not None and _index(s.run_id) > k):
            continue
        rng = np.random.default_rng([seed, zlib.crc32(s.run_id.encode())])
        q, usd = _level(arms[s.arm], truth)
        if s.run_id.endswith(".a"):
            q -= (loss or {}).get(experiment, 0.0)
        q += sd * rng.standard_normal()
        usd *= float(np.exp(cost_sd * rng.standard_normal()))
        run = {"id": s.run_id, "flow": s.flow.to_dict(), "task": {"id": s.input},
               "slate": {"id": s.slate, "members": list(s.extra["slate_members"])}}
        if accept is not None:
            verdict = "pass" if accept(arms[s.arm]) else "fail"
            run["signals"] = [{"id": f"sig_{s.run_id}_v", "kind": "verdict", "name": "claims-check", "value": verdict}]
        docs.append({"ocp": "0.4", "kind": "run", "run": run, "nodes": [], "attempts": [{"cost": {"usd": usd}}]})
        quality[s.run_id] = (float(q), None)
    return docs, quality


def _advance(folder, docs, quality, fit=FIT):
    plan = plan_mod.advance(folder, docs, quality=quality, fit=fit)
    return plan, next_mod.next_runs(folder, n=1000, seed=1, docs=docs, quality=quality, fit=fit)


def _screened(folder, truth=SHIPPED, fit=FIT, **world):
    """The plan and specs once every screen pair is run and scored."""
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=fit)
    docs, quality = _ran(folder.plan, first, truth, **world)
    return (docs, quality), *_advance(folder, docs, quality, fit)


def _effect(readout, kind, piece):
    return next(e for e in readout["effects"] if (e["edit"]["kind"], e["edit"].get("piece")) == (kind, piece))


def _run_ids(plan):
    return [run_id for e in plan.experiments for key, _, _ in plan_mod.pairs(plan, e)
            for run_id in plan_mod.run_ids(key)]


# The screen readout


def test_the_readout_recovers_the_truth_and_chooses_the_cheaper_set(tmp_path):
    folder = _saved(tmp_path)
    _, plan, specs = _screened(folder)
    readout = _exp(plan, "screen").extra["readout"]
    assert readout["status"] == "chosen" and readout["outcome"] == "quality"
    assert (readout["pairs"], readout["kept_pairs"], readout["rank"], readout["df"]) == (16, 16, 5, 11)
    for (kind, piece), (dq, du) in SHIPPED.items():
        if kind == "merge":
            continue
        e = _effect(readout, kind, piece)
        assert e["quality"]["mean"] == pytest.approx(dq, abs=1e-6) and e["cost"]["mean"] == pytest.approx(du, abs=1e-6)
        assert e["quality"]["lo"] == e["quality"]["hi"] == e["quality"]["mean"]  # no noise: the intervals collapse
    # Extract on Sonnet and synthesize at medium; budget 1 saves nothing, so a tie goes to the smaller set; drop
    # critic loses 0.2 and scout width 2 costs more.
    assert readout["chosen"] == [1, 3] and readout["ranking"] == [1, 3]
    assert readout["predicted"] == pytest.approx({"quality": -0.03, "cost": -0.65})
    assert readout["reasons"][0].startswith("screen readout: 16 kept pairs, 5 factors, rank 5")
    subset = _exp(plan, "subset")
    assert subset.arms == (BOTH,) and subset.state == "planned" and subset.verdict is None
    assert {s.experiment for s in specs} == {"merge-extract-synthesize", "subset"}


def test_with_the_truths_noise_drop_critic_stays_out(tmp_path):
    folder = _saved(tmp_path)
    _, plan, _ = _screened(folder, sd=0.12, cost_sd=0.25, seed=1)
    readout = _exp(plan, "screen").extra["readout"]
    # Round budget 1's estimate saves a tenth of a cent, so it joins; drop critic's whole interval is past -0.1.
    assert readout["status"] == "chosen" and readout["chosen"] == [1, 3, 5]
    critic = _effect(readout, "drop", "critic")["quality"]
    assert critic["lo"] < critic["mean"] < critic["hi"] < -0.1


def test_the_readout_waits_for_every_screen_score(tmp_path):
    folder = _saved(tmp_path)
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    late = next(s.run_id for s in first if _index(s.run_id) == 7)
    del quality[late]
    plan, specs = _advance(folder, docs, quality)
    assert _exp(plan, "screen").state == "done" and "readout" not in _exp(plan, "screen").extra
    assert _exp(plan, "subset").arms == () and _exp(plan, "subset").state == "planned"
    assert {s.experiment for s in specs} == {"merge-extract-synthesize"}  # next_runs stops at the subset
    readout = check_mod.screen_readout(folder, plan, _exp(plan, "screen"), docs, quality=quality, fit=FIT)
    assert readout["status"] == "wait" and readout["waiting"] == 1
    assert readout["blocking"] == [late] and readout["reasons"][0].endswith(f": {late}")
    # The wait names exactly the kept runs with no outcome, in pair order, run a before run b.
    pair = f"run_{plan.id}.screen"
    for gone in (f"{pair}.003.b", f"{pair}.012.a", f"{pair}.012.b"):
        del quality[gone]
    readout = check_mod.screen_readout(folder, plan, _exp(plan, "screen"), docs, quality=quality, fit=FIT)
    assert (readout["status"], readout["waiting"]) == ("wait", 3)
    assert readout["blocking"] == [f"{pair}.003.b", late, f"{pair}.012.a", f"{pair}.012.b"]


def test_the_subset_arm_is_a_plan_arm_sized_at_the_fitted_spreads(tmp_path):
    folder = _saved(tmp_path)
    saved = folder.plan
    _, plan, specs = _screened(folder, fit=FITTED)
    assert plan.id == saved.id and plan.arms[:len(saved.arms)] == saved.arms
    assert plan.experiments[1] == saved.experiments[1]
    assert [(e.id, e.arms, e.pair_cap, e.design) for e in plan.experiments[:1]] == \
        [(e.id, e.arms, e.pair_cap, e.design) for e in saved.experiments[:1]]
    arm = next(a for a in plan.arms if a.id == BOTH)
    c = arm.configuration
    assert c["id"] == config_id(c["workflow"], c["settings"], resolve=catalog_resolver)
    assert [e.kind for e in arm.edits] == ["model", "effort"]
    figure = plan_mod.margin_table(margins=(0.1,), s_run=0.1, s_judge=0.08, k=1)[0]["check_pairs"]
    subset = _exp(plan, "subset")
    assert subset.pair_cap == figure == subset.design["pairs"] != _exp(plan, "merge-extract-synthesize").pair_cap
    assert subset.design["spreads"]["label"] == "fitted" and subset.design["after"] == "screen"
    assert subset.split == "holdout" and subset.reference == "base"
    assert sorted({_index(s.run_id) for s in specs if s.experiment == "subset"}) == list(range(1, figure + 1))


@pytest.mark.parametrize("change, match", [
    ({"s_run": 50.0}, r"^margin 0\.1: needs 3\d{6} check pairs at sd_pair 70\.71, above the limit of 1000000$"),
    ({"s_run": 1e200}, r"^s_run 1e\+200: a spread's square is past what a float holds$"),
    ({"judges": [{"id": "j-claude", "kind": "model", "noise": 1e200}]},
     r"^judge noise \[1e\+200\]: the model judges' mean square noise is past what a float holds$"),
], ids=["past-the-limit", "s_run", "noise"])
def test_a_subset_sized_past_the_pair_limit_is_refused_by_name(tmp_path, monkeypatch, change, match):
    """Spreads fitted after the plan was made can put the new subset check past the pair limit, or past what a
    float holds: `advance` refuses by name, and no pair of an oversized check is ever listed."""
    from loopmath.screen import design as screen_design

    listed = plan_mod.pairs

    def pairs(plan, exp):
        if (exp.pair_cap or 0) > 10 ** 6:
            raise AssertionError(f"{exp.id}: {exp.pair_cap} pairs listed, past the pair limit")
        return listed(plan, exp)

    def fold_over(*args, **kwargs):
        raise AssertionError("advance builds no screen")

    folder = _saved(tmp_path)
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    monkeypatch.setattr(plan_mod, "pairs", pairs)
    monkeypatch.setattr(screen_design, "fold_over", fold_over)
    with pytest.raises(ValueError, match=match):
        plan_mod.advance(folder, docs, quality=quality, fit={**FITTED, **change})
    figure = plan_mod.margin_table(margins=(0.1,), s_run=0.1, s_judge=0.08, k=1)[0]["check_pairs"]
    assert _exp(plan_mod.advance(folder, docs, quality=quality, fit=FITTED), "subset").pair_cap == figure


# The subset check and the single checks


def test_a_subset_that_adopts_ends_the_chain(tmp_path):
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder)
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", **NOISY)
    plan, specs = _advance(folder, docs, quality)
    subset = _exp(plan, "subset")
    assert (subset.state, subset.verdict, subset.extra["stop"]) == ("done", "adopt", "decided")
    assert [e.kind for e in plan.experiments] == ["screen", "merge", "subset"]
    assert plan_mod.adopted(plan) == BOTH
    assert {s.experiment for s in specs} == {"merge-extract-synthesize"}


def test_a_subset_that_rejects_checks_its_factors_one_at_a_time(tmp_path):
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder)
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", k=8, loss={"subset": 0.3},
                         **NOISY)
    plan, specs = _advance(folder, docs, quality)
    assert (_exp(plan, "subset").state, _exp(plan, "subset").verdict) == ("done", "reject")
    single = plan.experiments[-1]
    assert (single.id, single.kind, single.arms, single.reference, single.split, single.state) == \
        (f"single-{EXTRACT}", "single", (EXTRACT,), "base", "holdout", "planned")
    assert {s.experiment for s in specs} == {"merge-extract-synthesize", single.id}
    rejected = (docs, quality, specs)

    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment=single.id, **NOISY)
    plan, specs = _advance(folder, docs, quality)
    assert _exp(plan, single.id).verdict == "adopt" and plan_mod.adopted(plan) == EXTRACT
    second = plan.experiments[-1]
    assert (second.id, second.arms, second.reference) == (f"single-{BOTH}", (BOTH,), EXTRACT)
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment=second.id, k=8, loss={second.id: 0.3},
                         **NOISY)
    plan, specs = _advance(folder, docs, quality)
    assert _exp(plan, second.id).verdict == "reject" and plan.experiments[-1].id == second.id
    assert plan_mod.adopted(plan) == EXTRACT and _exp(plan, "subset").extra["chain"] == [single.id, second.id]
    assert {s.experiment for s in specs} == {"merge-extract-synthesize"}

    # Single 1 rejects instead: single 2 is synthesize at medium alone, against the base.
    docs, quality, specs = rejected
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment=single.id, k=8, loss={single.id: 0.3},
                         **NOISY)
    plan, _ = _advance(folder, docs, quality)
    assert _exp(plan, single.id).verdict == "reject"
    other = plan.experiments[-1]
    assert (other.id, other.arms, other.reference) == (f"single-{SYNTH}", (SYNTH,), "base")
    assert plan_mod.adopted(plan) == "base" and _exp(plan, "subset").extra["chain"] == [single.id, other.id]


def test_one_advance_goes_from_the_screen_to_single_1(tmp_path):
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder)
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", k=8, loss={"subset": 0.3},
                         **NOISY)
    assert folder.plan.experiments[-1].arms == ()  # the saved plan has not seen the screen
    plan = plan_mod.advance(folder, docs, quality=quality, fit=FIT)
    assert [(e.id, e.state) for e in plan.experiments] == [
        ("screen", "done"), ("merge-extract-synthesize", "planned"), ("subset", "done"),
        (f"single-{EXTRACT}", "planned")]


def test_singles_follow_the_ranking_and_reuse_the_canonical_arm(tmp_path):
    # Synthesize at medium saves more than extract on Sonnet here, so it is checked first.
    truth = {**SHIPPED, ("model", "extract"): (0.0, -0.1), ("effort", "synthesize"): (-0.03, -0.4)}
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder, truth)
    readout = _exp(plan, "screen").extra["readout"]
    assert readout["chosen"] == [1, 3] and readout["ranking"] == [3, 1]
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment="subset", k=8, loss={"subset": 0.3}, **NOISY)
    plan, specs = _advance(folder, docs, quality)
    first = plan.experiments[-1]
    assert (first.arms, first.reference) == ((SYNTH,), "base")
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment=first.id, **NOISY)
    plan, _ = _advance(folder, docs, quality)
    second = plan.experiments[-1]
    assert (second.id, second.arms, second.reference) == (f"single-{BOTH}", (BOTH,), SYNTH)
    edits = {(e.kind, e.piece) for e in next(a for a in plan.arms if a.id == BOTH).edits}
    assert [a.id for a in plan.arms if {(e.kind, e.piece) for e in a.edits} == edits] == [BOTH]
    assert len({a.id for a in plan.arms}) == len(plan.arms)


@pytest.mark.parametrize("second_too", [False, True])
def test_a_single_off_the_ranked_chain_adopts_nothing(tmp_path, second_too):
    # Ranking [3, 1], and the plan already holds extract on Sonnet against synthesize at medium, adopted. Once
    # synthesize at medium adopts, the chain's next single is the set of both, so that old edge stays off it.
    truth = {**SHIPPED, ("model", "extract"): (0.0, -0.1), ("effort", "synthesize"): (-0.03, -0.4)}
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder, truth)
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment="subset", k=8, loss={"subset": 0.3}, **NOISY)
    plan, specs = _advance(folder, docs, quality)
    first = plan.experiments[-1]
    old = Experiment(id=f"single-{EXTRACT}", kind="single", arms=(EXTRACT,), reference=SYNTH, split="holdout",
                     pair_cap=first.pair_cap, design=first.design, state="done", verdict="adopt")
    save_plan(folder.dir, dataclasses.replace(plan, experiments=plan.experiments[:-1] + (old, first)))
    folder = load(folder.dir)
    assert plan_mod.adopted(folder.plan) == "base"
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment=first.id)
    plan, specs = _advance(folder, docs, quality)
    second = plan.experiments[-1]
    assert _exp(plan, first.id).verdict == "adopt"
    assert (second.arms, second.reference, second.state) == ((BOTH,), SYNTH, "planned")
    assert _exp(plan, "subset").extra["chain"] == [first.id, second.id] and plan_mod.adopted(plan) == SYNTH
    if second_too:
        save_plan(folder.dir, plan)
        folder = load(folder.dir)
        docs, quality = _ran(plan, specs, truth, docs, quality, experiment=second.id)
        plan, _ = _advance(folder, docs, quality)
        assert _exp(plan, second.id).verdict == "adopt" and plan_mod.adopted(plan) == BOTH


def test_a_setting_on_a_dropped_piece_is_skipped_in_the_chain(tmp_path):
    # Factors: 1 extract on Sonnet, 2 drop critic, 3 synthesize at medium, 4 critic effort high, 5 scout width 2,
    # 6 budget 1. A critic setting changes nothing once the critic is dropped, and the fold-over reads it at half
    # its effect: drop 0.35, critic effort 0.15, extract 0.1 saved.
    truth = {("model", "extract"): (0.0, -0.1), ("drop", "critic"): (0.0, -0.5), ("effort", "critic"): (0.0, -0.3),
             ("effort", "synthesize"): (-0.2, 0.1), ("width", "scout"): (0.0, 0.15)}
    folder = _saved(tmp_path, lambda flow: flow["allow"]["pieces"]["critic"].update(efforts=["high"]))
    (docs, quality), plan, specs = _screened(folder, truth)
    readout = _exp(plan, "screen").extra["readout"]
    assert readout["chosen"] == [1, 2, 4] and readout["ranking"] == [2, 4, 1]
    assert _effect(readout, "drop", "critic")["cost"]["mean"] == pytest.approx(-0.35, abs=1e-6)
    assert _exp(plan, "subset").arms == ("set-1-2",)  # the arm leaves out the setting on the dropped critic
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment="subset", k=8, loss={"subset": 0.3}, **NOISY)
    plan, specs = _advance(folder, docs, quality)
    drop = plan.experiments[-1]
    assert (drop.id, drop.arms, drop.reference) == ("single-drop-critic", ("drop-critic",), "base")
    assert "skipped" not in _exp(plan, "subset").extra
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment=drop.id, **NOISY)
    plan, specs = _advance(folder, docs, quality)
    third = plan.experiments[-1]
    assert (third.id, third.arms, third.reference) == ("single-set-1-2", ("set-1-2",), "drop-critic")
    skipped = _exp(plan, "subset").extra["skipped"]
    assert [s["factor"] for s in skipped] == [4] and "changes nothing" in skipped[0]["reason"]
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment=third.id, **NOISY)
    plan, _ = _advance(folder, docs, quality)
    assert _exp(plan, third.id).verdict == "adopt" and plan_mod.adopted(plan) == "set-1-2"
    assert [(e.id, e.reference) for e in plan.experiments if e.kind == "single"] == [
        ("single-drop-critic", "base"), ("single-set-1-2", "drop-critic")]
    assert len({e.id for e in plan.experiments}) == len(plan.experiments)
    assert len(set(_run_ids(plan))) == len(_run_ids(plan))
    save_plan(folder.dir, plan)
    loaded = load(folder.dir)
    assert loaded.plan == plan and plan_mod.advance(loaded, docs, quality=quality, fit=FIT) == plan


@pytest.mark.parametrize("verdict", ["reject", "adopt"])
def test_a_taken_single_id_gets_a_number(tmp_path, verdict):
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder)
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", k=8, loss={"subset": 0.3},
                         **NOISY)
    plan = plan_mod.advance(folder, docs, quality=quality, fit=FIT)
    # The same arm against another reference already holds the id single 1 would take.
    single = plan.experiments[-1]
    other = Experiment(id=single.id, kind="single", arms=(EXTRACT,), reference=SYNTH, split="holdout",
                       design=single.design, pair_cap=single.pair_cap, state="done", verdict=verdict)
    save_plan(folder.dir, dataclasses.replace(plan, experiments=plan.experiments[:-1] + (other,)))
    folder = load(folder.dir)
    plan, specs = _advance(folder, docs, quality)
    first = f"single-{EXTRACT}-2"
    assert [(e.id, e.reference, e.state) for e in plan.experiments[-2:]] == [
        (f"single-{EXTRACT}", SYNTH, "done"), (first, "base", "planned")]
    assert len(set(_run_ids(plan))) == len(_run_ids(plan))
    # The single against synthesize at medium is not on the chain, so its verdict adopts nothing.
    assert plan_mod.adopted(plan) == "base" and _exp(plan, "subset").extra["chain"] == [first]
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment=first, **NOISY)
    plan, _ = _advance(folder, docs, quality)
    assert _exp(plan, first).verdict == "adopt" and plan_mod.adopted(plan) == EXTRACT
    assert (plan.experiments[-1].arms, plan.experiments[-1].reference) == ((BOTH,), EXTRACT)


def test_adopted_reads_the_subsets_chain_and_refuses_a_broken_one(tmp_path):
    plan = _saved(tmp_path).plan
    one = Experiment(id="single-a", kind="single", arms=(EXTRACT,), reference="base", split="holdout",
                     state="done", verdict="adopt")
    two = dataclasses.replace(one, id="single-b", arms=(BOTH,), reference=EXTRACT)
    off = dataclasses.replace(one, id="single-c", arms=(SYNTH,))

    def chained(*chain):
        subset = dataclasses.replace(_exp(plan, "subset"), state="done", verdict="reject", extra={"chain": list(chain)})
        exps = tuple(subset if e.kind == "subset" else e for e in plan.experiments) + (one, two, off)
        return plan_mod.adopted(dataclasses.replace(plan, experiments=exps))

    assert chained() == "base" and chained("single-a") == EXTRACT and chained("single-a", "single-b") == BOTH
    with pytest.raises(ValueError, match="'single-c' in the subset's chain is not a single check against 'extract"):
        chained("single-a", "single-c")
    for wrong in ("single-b", "merge-extract-synthesize", "single-x"):
        with pytest.raises(ValueError, match=f"'{wrong}' in the subset's chain is not a single check against 'base'"):
            chained(wrong)


def test_two_factors_check_both_then_each_in_factor_order(tmp_path):
    def two(flow):
        flow["allow"] = {"pieces": {"extract": {"models": [{"id": "claude-sonnet-5-5"}]},
                                    "synthesize": {"efforts": ["medium"]}}}
        flow["ext"].pop("dev.loopmath.hypothesis_edits")
    folder = _saved(tmp_path, two)
    saved = folder.plan
    assert [(e.id, e.arms) for e in saved.experiments] == [("subset", ("set-1-2",))]
    specs = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(saved, specs, SHIPPED, experiment="subset", k=8, loss={"subset": 0.3}, **NOISY)
    plan, specs = _advance(folder, docs, quality)
    single = plan.experiments[-1]
    assert (single.id, single.arms, single.reference) == (f"single-{EXTRACT}", (EXTRACT,), "base")
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment=single.id, **NOISY)
    plan, _ = _advance(folder, docs, quality)
    assert [(e.id, e.arms, e.reference) for e in plan.experiments[1:]] == [
        (single.id, (EXTRACT,), "base"), ("single-set-1-2", ("set-1-2",), EXTRACT)]


def test_one_factor_is_its_own_chain(tmp_path):
    def one(flow):
        flow["allow"] = {"pieces": {"extract": {"models": [{"id": "claude-sonnet-5-5"}]}}}
        flow["ext"].pop("dev.loopmath.hypothesis_edits")
    folder = _saved(tmp_path, one)
    specs = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    assert [(e.id, e.arms) for e in folder.plan.experiments] == [(f"single-{EXTRACT}", (EXTRACT,))]
    docs, quality = _ran(folder.plan, specs, SHIPPED, experiment=f"single-{EXTRACT}", **NOISY)
    plan, specs = _advance(folder, docs, quality)
    assert [(e.state, e.verdict) for e in plan.experiments] == [("done", "adopt")]
    assert plan_mod.adopted(plan) == EXTRACT and specs == []


# Verdicts


def test_a_done_verdict_holds_when_more_documents_arrive(tmp_path):
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder)
    adopt = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", **NOISY)
    plan, _ = _advance(folder, *adopt)
    assert _exp(plan, "subset").verdict == "adopt"
    save_plan(folder.dir, plan)
    folder = load(folder.dir)
    lose = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", loss={"subset": 0.4}, **NOISY)
    assert check_mod.check_experiment(folder, plan, _exp(plan, "subset"), lose[0], quality=lose[1],
                                      fit=FIT).verdict == "reject"
    again, _ = _advance(folder, *lose)
    assert again == plan and plan_mod.adopted(again) == BOTH


def test_running_checks_say_more_and_merges_stand_apart(tmp_path):
    # The shipped merge loses 0.05, half the margin, where the check figure promises no power; here it loses
    # nothing.
    truth = {**SHIPPED, ("merge", None): (0.0, -0.3)}
    folder = _saved(tmp_path)
    merge = "merge-extract-synthesize"
    specs = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, specs, truth, experiment=merge, k=4, **NOISY)
    plan, _ = _advance(folder, docs, quality)
    assert (_exp(plan, merge).state, _exp(plan, merge).verdict) == ("running", "more")
    docs, quality = _ran(folder.plan, specs, truth, experiment=merge, **NOISY)
    plan, _ = _advance(folder, docs, quality)
    assert (_exp(plan, merge).state, _exp(plan, merge).verdict, _exp(plan, merge).extra["stop"]) == \
        ("done", "adopt", "decided")
    assert plan_mod.adopted(plan) == "base"
    docs, quality = _ran(folder.plan, specs, truth, docs, quality)  # the screen
    plan, specs = _advance(folder, docs, quality)
    docs, quality = _ran(plan, specs, truth, docs, quality, experiment="subset", **NOISY)
    plan, _ = _advance(folder, docs, quality)
    assert plan_mod.adopted(plan) == BOTH and _exp(plan, merge).verdict == "adopt"


# Readouts that choose nothing


def test_no_subset_when_every_factor_costs_more(tmp_path):
    truth = {key: (dq, abs(du) + 0.1) for key, (dq, du) in SHIPPED.items()}
    folder = _saved(tmp_path)
    _, plan, specs = _screened(folder, truth)
    subset = _exp(plan, "subset")
    assert (subset.state, subset.verdict, subset.arms, subset.extra["stop"]) == ("done", "reject", (), "no subset")
    readout = _exp(plan, "screen").extra["readout"]
    assert readout["status"] == "stop" and readout["chosen"] == [] and len(readout["effects"]) == 5
    assert "no set of factors" in readout["reasons"][-1]
    assert [e.id for e in plan.experiments] == ["screen", "merge-extract-synthesize", "subset"]
    assert {s.experiment for s in specs} == {"merge-extract-synthesize"}


def _unplanned(docs, keep):
    """The screen's documents with run a of each pair not in `keep` (pair numbers) on another arm."""
    out = json.loads(json.dumps(docs))
    for doc in out:
        if doc["run"]["id"].endswith(".a") and _index(doc["run"]["id"]) not in keep:
            doc["run"]["flow"]["arm"] = "base"
    return out


def test_too_few_kept_pairs_or_an_unpriced_run_stops_the_readout(tmp_path):
    folder = _saved(tmp_path)
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    few = _unplanned(docs, range(1, 6))
    plan, _ = _advance(folder, few, quality)
    readout = _exp(plan, "screen").extra["readout"]
    assert (readout["kept_pairs"], len(readout["excluded"])) == (5, 11)
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) == ("reject", "too few pairs")
    unpriced = json.loads(json.dumps(docs))
    unpriced[3]["attempts"] = [{"cost": {}}]
    plan, _ = _advance(folder, unpriced, quality)
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) == ("reject", "cost unknown")
    assert "effects" not in _exp(plan, "screen").extra["readout"]
    assert [e.id for e in plan.experiments] == ["screen", "merge-extract-synthesize", "subset"]


def test_kept_pairs_whose_columns_are_not_identified_stop_the_readout(tmp_path):
    folder = _saved(tmp_path)
    rows = _exp(folder.plan, "screen").design["rows"]
    same = [i for i, row in enumerate(rows, start=1) if row[0] == row[1]]
    assert len(same) >= 6
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    plan, _ = _advance(folder, _unplanned(docs, same), quality)
    readout = _exp(plan, "screen").extra["readout"]
    assert readout["status"] == "stop" and readout["stop"] == "not identified"
    assert (readout["kept_pairs"], readout["rank"], readout["unidentified"]) == (len(same), 4, [1, 2])
    assert "effects" not in readout
    assert "factors 1, 2" in readout["reasons"][0] and "in the span of the other columns" in readout["reasons"][0]
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) == ("reject", "not identified")


def test_a_full_rank_kept_set_still_reads_the_truth(tmp_path):
    folder = _saved(tmp_path)
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    plan, _ = _advance(folder, _unplanned(docs, range(3, 17)), quality)
    readout = _exp(plan, "screen").extra["readout"]
    assert (readout["kept_pairs"], readout["rank"], readout["df"]) == (14, 5, 9)
    assert _effect(readout, "model", "extract")["cost"]["mean"] == pytest.approx(-0.45, abs=1e-6)
    assert _effect(readout, "drop", "critic")["quality"]["mean"] == pytest.approx(-0.2, abs=1e-6)
    assert readout["chosen"] == [1, 3] and _exp(plan, "subset").arms == (BOTH,)
    # An excluded pair is not in the fit, so a run of it with no outcome does not hold the choice back.
    unscored = {run_id: q for run_id, q in quality.items() if run_id != f"run_{plan.id}.screen.001.a"}
    plan, _ = _advance(folder, _unplanned(docs, range(3, 17)), unscored)
    assert _exp(plan, "screen").extra["readout"] == readout and _exp(plan, "subset").arms == (BOTH,)


def test_an_excluded_runs_outcome_changes_no_readout(tmp_path):
    # An excluded pair is in no fit and no stop, so the readout neither waits for its runs nor reads them: an
    # outcome that arrives later leaves every stop, and the choice, as they were.
    folder, costly = _saved(tmp_path / "shipped"), _saved(tmp_path / "costly")
    screen = _exp(folder.plan, "screen")
    same = [i for i, row in enumerate(screen.design["rows"], start=1) if row[0] == row[1]]
    docs, quality = _ran(folder.plan, next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT), SHIPPED)
    unpriced = _unplanned(docs, range(1, 6))
    next(d for d in unpriced if d["run"]["id"].endswith(".001.b"))["attempts"] = [{"cost": {}}]
    truth = {key: (dq, abs(du) + 0.1) for key, (dq, du) in SHIPPED.items()}
    c_docs, c_quality = _ran(costly.plan, next_mod.next_runs(costly, n=1000, seed=1, quality={}, fit=FIT), truth)
    wide = dataclasses.replace(screen, design={**screen.design, "factors": screen.design["factors"] * 4})
    worlds = {"too few pairs": (folder, screen, _unplanned(docs, range(1, 6)), quality, range(1, 6)),
              "not identified": (folder, screen, _unplanned(docs, same), quality, same),
              "cost unknown": (folder, screen, unpriced, quality, range(1, 6)),
              "too many factors": (folder, wide, _unplanned(docs, range(1, 6)), quality, range(1, 6)),
              "no subset": (costly, _exp(costly.plan, "screen"), _unplanned(c_docs, range(3, 17)), c_quality,
                            range(3, 17)),
              None: (folder, screen, _unplanned(docs, range(3, 17)), quality, range(3, 17))}
    for stop, (f, exp, world, scored, keep) in worlds.items():
        unscored = {run_id: q for run_id, q in scored.items() if _index(run_id) in keep}
        assert len(unscored) == 2 * len(keep) < len(scored)
        readouts = [check_mod.screen_readout(f, f.plan, exp, world, quality=q, fit=FIT) for q in (unscored, scored)]
        assert readouts[0] == readouts[1] and readouts[0]["status"] != "wait" and readouts[0].get("stop") == stop
        first, second = (_advance(f, world, q)[0] for q in (unscored, scored))
        if stop is None:  # the subset's sizing record counts every scored pair of the flow, kept or not
            spreads = [_exp(p, "subset").design["spreads"] for p in (first, second)]
            assert spreads[0].pop("scored_pairs") == 14 and spreads[1].pop("scored_pairs") == 16
        assert first == second


def test_more_factors_than_the_readout_enumerates_stop_it(tmp_path):
    folder = _saved(tmp_path)
    screen = _exp(folder.plan, "screen")
    wide = dataclasses.replace(screen, design={**screen.design, "factors": screen.design["factors"] * 4})
    readout = check_mod.screen_readout(folder, folder.plan, wide, [], quality={}, fit=FIT)
    assert (readout["status"], readout["stop"], readout["chosen"]) == ("stop", "too many factors", [])
    assert "at most 16" in readout["reasons"][0]


# Numbers no float holds


def _strict(value):
    """`value` through standard JSON both ways: no NaN or infinity is written or read."""
    def refuse(constant):
        raise ValueError(f"{constant} is not standard JSON")
    return json.loads(json.dumps(value, allow_nan=False), parse_constant=refuse)


def _quiet(tmp_path, folder, docs, quality):
    """advance and experiment_states with warnings as errors, the states, the readout and the plan (saved in a copy
    of the folder) each read back as standard JSON; the plan and the screen's readout."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = plan_mod.advance(folder, docs, quality=quality, fit=FIT)
        states = check_mod.experiment_states(folder, docs, quality=quality, fit=FIT)
    _strict(states)
    where = tmp_path / f"saved-{len(list(tmp_path.glob('saved-*')))}"
    shutil.copytree(folder.dir, where)
    _strict(json.loads(save_plan(where, plan).read_text(encoding="utf-8")))
    readout = _exp(plan, "screen").extra["readout"]
    _strict(readout)
    return plan, readout


def _priced(docs, end, attempts):
    """A copy of `docs` with the attempts of the run whose id ends `end` replaced, and that run's id."""
    out = json.loads(json.dumps(docs))
    doc = next(d for d in out if d["run"]["id"].endswith(end))
    doc["attempts"] = attempts
    return out, doc["run"]["id"]


def _shipped_screen(folder):
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    return _ran(folder.plan, first, SHIPPED)


def test_a_run_cost_no_float_holds_stops_the_readout_as_cost_unknown(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _shipped_screen(folder)
    huge, b4 = _priced(docs, ".004.b", [{"cost": {"usd": 1e308}}, {"cost": {"usd": 1e308}}])
    plan, readout = _quiet(tmp_path, folder, huge, quality)
    line = (f"1 screen pair has a run whose dollar cost is not finite (NaN, or past what a float holds) ({b4}), "
            "so no saving can be estimated: no subset check")
    assert (readout["status"], readout["stop"], readout["reasons"]) == ("stop", "cost unknown", [line])
    assert readout["nonfinite"] == [f"cost:{b4}"] and "effects" not in readout and readout["chosen"] == []
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) == ("reject", "cost unknown")
    native, _ = _priced(docs, ".004.b", [{"cost": {"usd": 10 ** 400}}])  # an integer no float holds
    plan, readout = _quiet(tmp_path, folder, native, quality)
    assert (readout["status"], readout["stop"], readout["reasons"]) == ("stop", "cost unknown", [line])
    assert readout["nonfinite"] == [f"cost:{b4}"] and "effects" not in readout and readout["chosen"] == []
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) == ("reject", "cost unknown")
    both, _ = _priced(huge, ".002.a", [{"cost": {}}])
    plan, readout = _quiet(tmp_path, folder, both, quality)
    assert readout["reasons"] == ["1 screen pair have a run with no dollar cost, so no saving can be estimated: no "
                                  "subset check", line]
    assert readout["nonfinite"] == [f"cost:{b4}"]
    more, a4 = _priced(huge, ".004.a", [{"cost": {"usd": float("inf")}}])
    more, b2 = _priced(more, ".002.b", [{"cost": {"usd": 1e308}}, {"cost": {"usd": 1e308}}])
    plan, readout = _quiet(tmp_path, folder, more, quality)
    assert readout["nonfinite"] == [f"cost:{b2}", f"cost:{a4}", f"cost:{b4}"]  # pair order, run a before run b
    assert readout["reasons"] == [f"2 screen pairs have a run whose dollar cost is not finite (NaN, or past what a "
                                  f"float holds) ({b2}, {a4}, {b4}), so no saving can be estimated: no subset check"]
    assert _exp(plan, "subset").extra["stop"] == "cost unknown"


def test_a_run_cost_of_nan_stops_the_readout_as_cost_unknown(tmp_path):
    """NaN dollars, which standard JSON cannot hold, so set on the document in code: the readout stops as it does
    for a missing cost (`cost unknown`, the subset rejected), naming the run in `nonfinite` as `cost:<run id>` and
    in its reason as not finite."""
    folder = _saved(tmp_path)
    docs, quality = _shipped_screen(folder)
    nan, b4 = _priced(docs, ".004.b", [])
    next(d for d in nan if d["run"]["id"] == b4)["attempts"] = [{"cost": {"usd": float("nan")}}]
    plan, readout = _quiet(tmp_path, folder, nan, quality)
    line = (f"1 screen pair has a run whose dollar cost is not finite (NaN, or past what a float holds) ({b4}), "
            "so no saving can be estimated: no subset check")
    assert (readout["status"], readout["stop"], readout["reasons"]) == ("stop", "cost unknown", [line])
    assert readout["nonfinite"] == [f"cost:{b4}"] and "effects" not in readout and readout["chosen"] == []
    missing, _ = _priced(docs, ".004.b", [{"cost": {}}])
    plan_m, readout_m = _quiet(tmp_path, folder, missing, quality)
    assert (readout_m["status"], readout_m["stop"], "nonfinite" in readout_m) == ("stop", "cost unknown", False)
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) \
        == (_exp(plan_m, "subset").verdict, _exp(plan_m, "subset").extra["stop"]) == ("reject", "cost unknown")


@pytest.mark.parametrize("usd", [1e308, 1e300, 1e200])
def test_fitted_numbers_no_float_holds_stop_the_readout_by_name(tmp_path, usd):
    folder = _saved(tmp_path)
    docs, quality = _shipped_screen(folder)
    for doc in docs:
        if doc["run"]["id"].endswith(".b"):
            doc["attempts"] = [{"cost": {"usd": usd}}]
    plan, readout = _quiet(tmp_path, folder, docs, quality)
    # each residual square overflows; at 1e308 the mean dollars, and so every set's dollar level, do too
    names = [f"cost.{what}:{j}" for j in range(1, 6) for what in ("sd", "lo", "hi")]
    names += ["cost.mean", "level.cost"] if usd == 1e308 else []
    assert (readout["status"], readout["stop"], readout["nonfinite"]) == ("stop", "not finite", names)
    assert readout["reasons"] == [f"the readout's numbers {', '.join('nonfinite:' + n for n in names)} are past what "
                                  "a float holds, so no effect can be read: no subset check"]
    assert (readout["rank"], readout["df"], readout["chosen"]) == (5, 11, [])
    assert not {"effects", "ranking", "predicted"} & set(readout)
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) == ("reject", "not finite")


def test_a_mean_cost_no_float_holds_stops_the_readout_by_name(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _shipped_screen(folder)
    for doc in docs:
        doc["attempts"] = [{"cost": {"usd": 1.5e308}}]  # every difference 0: the effects hold, the mean overflows
    plan, readout = _quiet(tmp_path, folder, docs, quality)
    assert (readout["stop"], readout["nonfinite"]) == ("not finite", ["cost.mean", "level.cost"])
    assert readout["reasons"] == ["the readout's numbers nonfinite:cost.mean, nonfinite:level.cost are past what a "
                                  "float holds, so no effect can be read: no subset check"]
    assert not {"effects", "ranking", "predicted"} & set(readout)
    assert _exp(plan, "subset").extra["stop"] == "not finite"


def test_a_run_cost_no_float_holds_leaves_a_new_check_unpriced_with_a_reason(tmp_path, monkeypatch):
    """A recorded base run outside the screen whose dollars are an integer no float holds: the readout chooses as
    before, and the new subset check is sized with no price, the mean named, in standard JSON. `mean_run_cost`
    raises there, as the store's reader does; any other error from the reader raises through the sizing."""
    folder = _saved(tmp_path)
    merge = "merge-extract-synthesize"
    specs = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, specs, SHIPPED, experiment=merge, k=1)
    docs, quality = _ran(folder.plan, specs, SHIPPED, docs, quality)  # the screen
    base = [d["run"]["id"] for d in docs if d["run"]["flow"]["arm"] == "base"]
    reference = next(s.run_id for s in specs if s.experiment == merge and s.arm == "base" and _index(s.run_id) == 1)
    native, _ = _priced(docs, reference, [{"cost": {"usd": 10 ** 400}}])
    with pytest.raises(OverflowError):
        plan_mod.mean_run_cost(folder, native)
    plan, readout = _quiet(tmp_path, folder, native, quality)
    assert (readout["status"], readout["chosen"]) == ("chosen", [1, 3]) and "nonfinite" not in readout
    subset = _exp(plan, "subset")
    assert subset.arms == (BOTH,) and subset.state == "planned"
    assert subset.design["price"] == {"usd": None, "per_run": None, "from_runs": len(base),
                                      "reason": f"the mean dollars of the {len(base)} recorded runs is past what a "
                                                "float holds"}
    assert _exp(plan, merge).verdict == "more"

    def broken(doc):
        raise TypeError("a shape the reader refuses")

    monkeypatch.setattr(plan_mod, "run_cost", broken)  # the check and the readout read check's own
    with pytest.raises(TypeError, match="a shape the reader refuses") as raised:
        plan_mod.advance(folder, docs, quality=quality, fit=FIT)
    assert "size" in {entry.name for entry in raised.traceback}


@pytest.mark.parametrize("which", ["cost", "quality"])
def test_a_prediction_no_float_holds_stops_the_readout_by_name(tmp_path, monkeypatch, which):
    # Injected at the fit: effects past half the float range on factors 1 and 2, each finite, so only a set with
    # both sums past what a float holds. No finite world found reaches this boundary on its own.
    real, seen = check_mod._regress, []

    def regress(rows_x, inverse, d, df):
        effect, se = real(rows_x, inverse, d, df)
        seen.append(None)
        if (len(seen) % 2 == 0) == (which == "cost"):  # quality's fit, then cost's, for each readout
            effect = effect.copy()
            effect[:2] = -1e308 if which == "cost" else 1e308
        assert np.all(np.isfinite(effect)) and np.all(np.isfinite(se))
        return effect, se

    monkeypatch.setattr(check_mod, "_regress", regress)
    folder = _saved(tmp_path)
    docs, quality = _shipped_screen(folder)
    plan, readout = _quiet(tmp_path, folder, docs, quality)
    assert seen and (readout["status"], readout["stop"]) == ("stop", "not finite")
    assert readout["nonfinite"] == [f"predicted.{which}"]
    assert readout["reasons"] == [f"the readout's numbers nonfinite:predicted.{which} are past what a float holds, "
                                  "so no effect can be read: no subset check"]
    assert readout["chosen"] == [] and not {"effects", "ranking", "predicted"} & set(readout)
    assert (readout["rank"], readout["df"]) == (5, 11)
    assert (_exp(plan, "subset").verdict, _exp(plan, "subset").extra["stop"]) == ("reject", "not finite")


# The readouts of finite numbers, as before any number was checked: the canonical JSON's sha256, recorded once.
FINITE_SHAS = {"shipped": "d63dbd98ec86094ae1c3c73c0d7d89121d22bfc8b10ecb6c63d10863ab1b439a",
               "noise": "ace0914c34add43c81ab122d80a1dc1af907ee2be8d14e0cacd2c1ef002ac6b6",
               "cheapest": "1bf5212a9b998c9f721fe8e915d98dc98b688b52ff2d9f606120f6251a8a96d4",
               "best": "d6e2cffc5853aaa0858ff275981c2b9a5d4beaae0224e5dfc9d3b8fdf8585131",
               "cpar": "0e930a04ce1e9be252130eabc59457fcb0ece120920b1639e405ee2a6a4928ed",
               "unpriced": "c2946391b723bdc8786d9647eaa6f0a860eb9f3ecac5d24600d252aee470628b"}


def test_readouts_of_finite_numbers_keep_their_bytes(tmp_path):
    truth = {**SHIPPED, ("effort", "synthesize"): (-0.15, -0.2), ("width", "scout"): (0.15, 0.15)}
    readouts = {}
    _, plan, _ = _screened(_saved(tmp_path / "shipped"))
    readouts["shipped"] = _exp(plan, "screen").extra["readout"]
    _, plan, _ = _screened(_saved(tmp_path / "noise"), sd=0.12, cost_sd=0.25, seed=1)
    readouts["noise"] = _exp(plan, "screen").extra["readout"]
    best = _saved(tmp_path / "best", lambda flow: flow.update(
        objective={"kind": "best_under_cost", "score": "quality", "margin": 0.1, "cost_cap_usd": 5.0}))
    cpar = _saved(tmp_path / "cpar", lambda flow: flow.update(objective={"kind": "cost_per_accepted", "margin": 0.1},
                                                              acceptance_rule=VERDICT_ONLY))
    for name, folder, world in (("cheapest", _saved(tmp_path / "cheapest"), {}), ("best", best, {}),
                                ("cpar", cpar, {"accept": lambda arm: all(e.kind != "drop" for e in arm.edits)})):
        _, plan, _ = _screened(folder, truth, **world)
        readouts[name] = _exp(plan, "screen").extra["readout"]
    folder = _saved(tmp_path / "unpriced")
    docs, quality = _shipped_screen(folder)
    unpriced, _ = _priced(docs, docs[3]["run"]["id"], [{"cost": {}}])
    plan, _ = _advance(folder, unpriced, quality)
    readouts["unpriced"] = _exp(plan, "screen").extra["readout"]
    assert {name: hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
            for name, r in readouts.items()} == FINITE_SHAS


# Other objectives


def test_each_objective_chooses_by_its_own_aim(tmp_path):
    # Synthesize at medium loses 0.15, past the margin; scout width 2 gains 0.15 at 0.15 dollars; dropping the
    # critic fails the claims check.
    truth = {**SHIPPED, ("effort", "synthesize"): (-0.15, -0.2), ("width", "scout"): (0.15, 0.15)}
    cheapest = _saved(tmp_path / "cheapest")
    best = _saved(tmp_path / "best", lambda flow: flow.update(
        objective={"kind": "best_under_cost", "score": "quality", "margin": 0.1, "cost_cap_usd": 5.0}))
    cpar = _saved(tmp_path / "cpar", lambda flow: flow.update(objective={"kind": "cost_per_accepted", "margin": 0.1},
                                                              acceptance_rule=VERDICT_ONLY))
    readouts = {}
    for name, folder, world in (("cheapest", cheapest, {}), ("best", best, {}),
                                ("cpar", cpar, {"accept": lambda arm: all(e.kind != "drop" for e in arm.edits)})):
        _, plan, _ = _screened(folder, truth, **world)
        readouts[name] = _exp(plan, "screen").extra["readout"]
    assert readouts["cheapest"]["chosen"] == [1, 3, 4]  # 0.5 saved, quality change 0
    assert readouts["best"]["chosen"] == [4]  # the only gain
    assert readouts["cpar"]["chosen"] == [1, 3] and readouts["cpar"]["outcome"] == "acceptance"
    critic = _effect(readouts["cpar"], "drop", "critic")["quality"]
    assert critic["mean"] == pytest.approx(-1.0, abs=1e-6)  # every run without the critic fails the check


# Determinism, records and judges


def test_advance_is_deterministic_and_a_saved_plan_is_a_fixed_point(tmp_path):
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder, sd=0.12, cost_sd=0.25, seed=1)
    docs, quality = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", k=8, loss={"subset": 0.3},
                         **NOISY)
    plan = plan_mod.advance(folder, docs, quality=quality, fit=FIT)
    assert plan_mod.advance(folder, docs, quality=quality, fit=FIT) == plan
    assert plan.experiments[-1].kind == "single" and "readout" in _exp(plan, "screen").extra
    save_plan(folder.dir, plan)
    loaded = load(folder.dir)
    assert loaded.plan == plan and loaded.plan.to_dict() == plan.to_dict()
    assert plan_mod.advance(loaded, docs, quality=quality, fit=FIT) == plan


def test_advance_never_calls_the_judges_without_a_score(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("judges called")
    monkeypatch.setattr(judges, "fit", refuse)
    monkeypatch.setattr(judges, "combined_quality", refuse)
    monkeypatch.setattr(judges, "unpooled_quality", refuse)
    folder = _saved(tmp_path)
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    unscored = plan_mod.advance(folder, docs)
    assert _exp(unscored, "screen").state == "done" and _exp(unscored, "subset").arms == ()
    plan = plan_mod.advance(folder, docs, quality=quality, fit=FIT)
    assert _exp(plan, "subset").arms == (BOTH,)
