"""`experiment_states` on the example flow: every key at each stage of one loop, the pairs a check keeps, the
screen's effects, the next pair's value, and nothing saved. The worlds are `test_flow_advance.py`'s."""

from __future__ import annotations

import copy
import json

from loopmath import compare
from loopmath.flows import check as check_mod
from loopmath.flows import judges
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows.folder import PLAN_FILE, load, save_plan

from test_flow_advance import (BOTH, EXTRACT, FIT, NOISY, SHIPPED, VERDICT_ONLY,  # noqa: E402  (same folder)
                               _advance, _exp, _ran, _saved, _screened, _unplanned)

MERGE = "merge-extract-synthesize"
FLAT = {**SHIPPED, ("merge", None): (0.0, -0.3)}  # the merge loses nothing, so its check runs on to the cap


def _states(folder, docs, quality, fit=FIT):
    states = check_mod.experiment_states(folder, docs, quality=quality, fit=fit)
    for s in states:
        assert tuple(s) == check_mod.STATE_KEYS and isinstance(s["arms"], list)
    return {s["id"]: s for s in states}


def _row(s):
    return s["state"], s["verdict"], s["pairs"], s["pairs_left"]


def _cpar(tmp_path):
    return _saved(tmp_path / "cpar", lambda flow: flow.update(objective={"kind": "cost_per_accepted", "margin": 0.1},
                                                              acceptance_rule=VERDICT_ONLY))


def _no_drop(arm):
    return all(e.kind != "drop" for e in arm.edits)


def _observations(docs, exp_id):
    """(reference, candidate) observations of `exp_id`'s pairs as compare reads them: the claims check's verdict
    and the run's dollars, run a the candidate."""
    ref, cand = [], []
    for doc in sorted(docs, key=lambda d: d["run"]["id"]):
        run = doc["run"]
        if f".{exp_id}." not in run["id"]:
            continue
        a = run["id"].endswith(".a")
        (cand if a else ref).append(compare.Observation(
            task=run["task"]["id"], arm="cand" if a else "base", outcome=run["signals"][-1]["value"] == "pass",
            usd=doc["attempts"][0]["cost"]["usd"], run=run["id"], pair=run["id"][len("run_"):-2],
            experiment=exp_id))
    return ref, cand


def test_every_experiment_has_every_key_at_each_stage(tmp_path):
    folder = _saved(tmp_path)
    fresh = _states(folder, [], {})
    assert [(s["kind"], s["arms"] == [], *_row(s), s["effects"], s["next_pair_value"]) for s in fresh.values()] == \
        [("screen", False, "planned", None, 0, 16, [], None), ("merge", False, "planned", None, 0, 24, [], None),
         ("subset", True, "planned", None, 0, 24, [], None)]

    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    s = _states(folder, *_ran(folder.plan, first, FLAT, k=5))["screen"]
    assert (*_row(s), s["effects"]) == ("running", None, 5, 11, [])

    docs, quality = _ran(folder.plan, first, FLAT)
    plan, specs = _advance(folder, docs, quality)
    states = _states(folder, docs, quality)
    assert _row(states["screen"]) == ("done", None, 16, 0)
    assert states["subset"]["arms"] == [BOTH] and _row(states["subset"]) == ("planned", None, 0, 24)

    s = _states(folder, *_ran(plan, specs, FLAT, docs, quality, experiment="subset", k=4, **NOISY))["subset"]
    assert (*_row(s), s["next_pair_value"]) == ("running", "more", 4, 20, None)

    docs, quality = _ran(plan, specs, FLAT, docs, quality, experiment="subset", k=8, loss={"subset": 0.3}, **NOISY)
    plan, specs = _advance(folder, docs, quality)
    single = plan.experiments[-1]
    states = _states(folder, docs, quality)
    assert _row(states["subset"]) == ("done", "reject", 8, 0)
    assert (states[single.id]["kind"], states[single.id]["arms"], *_row(states[single.id])) == \
        ("single", [EXTRACT], "planned", None, 0, single.pair_cap)

    s = _states(folder, *_ran(plan, specs, FLAT, docs, quality, experiment=single.id, k=3, **NOISY))[single.id]
    assert _row(s) == ("running", "more", 3, single.pair_cap - 3)
    docs, quality = _ran(plan, specs, FLAT, docs, quality, experiment=single.id, **NOISY)
    docs, quality = _ran(plan, specs, FLAT, docs, quality, experiment=MERGE, **NOISY)
    states = _states(folder, docs, quality)
    assert _row(states[single.id]) == ("done", "adopt", single.pair_cap, 0)
    assert _row(states[MERGE]) == ("done", "adopt", 24, 0)
    assert [s["next_pair_value"] for s in states.values()] == [None] * len(states)


def test_a_fresh_flow_calls_no_judges(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("judges called")
    monkeypatch.setattr(judges, "fit", refuse)
    monkeypatch.setattr(judges, "combined_quality", refuse)
    monkeypatch.setattr(judges, "unpooled_quality", refuse)
    folder = _saved(tmp_path)
    states = check_mod.experiment_states(folder, [])
    assert [(s["state"], s["pairs"], s["pairs_left"], s["effects"]) for s in states] == \
        [("planned", 0, 16, []), ("planned", 0, 24, []), ("planned", 0, 24, [])]
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, _ = _ran(folder.plan, first, SHIPPED)
    screen, merge, subset = check_mod.experiment_states(folder, docs)
    assert (screen["state"], screen["pairs"], screen["pairs_left"], screen["effects"]) == ("done", 16, 0, [])
    assert (subset["arms"], subset["state"]) == ([], "planned")


def test_nothing_is_saved_and_the_states_are_advances(tmp_path):
    folder = _saved(tmp_path)
    before = (folder.dir / PLAN_FILE).read_bytes()
    saved = copy.deepcopy(folder.plan)
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    states = check_mod.experiment_states(folder, docs, quality=quality, fit=FIT)
    advanced = plan_mod.advance(folder, docs, quality=quality, fit=FIT)
    assert advanced != saved  # the readout and the subset's arm are new
    assert [(s["id"], s["kind"], s["arms"], s["state"], s["verdict"]) for s in states] == \
        [(e.id, e.kind, list(e.arms), e.state or "planned", e.verdict) for e in advanced.experiments]
    assert (folder.dir / PLAN_FILE).read_bytes() == before and folder.plan == saved
    assert load(folder.dir).plan == saved


def test_the_screens_effects_are_its_readouts(tmp_path):
    folder = _saved(tmp_path)
    (docs, quality), plan, _ = _screened(folder)
    effects = _states(folder, docs, quality)["screen"]["effects"]
    assert effects == _exp(plan, "screen").extra["readout"]["effects"] and len(effects) == 5
    assert set(effects[0]) == {"edit", "quality", "cost"} and set(effects[0]["quality"]) == {"mean", "lo", "hi"}
    effects[0]["quality"]["mean"] = 9.0  # a copy: the plan's readout is not touched
    assert _states(folder, docs, quality)["screen"]["effects"][0]["quality"]["mean"] != 9.0

    # A readout that stops with effects (no subset) shows them; one that stops before them (not identified) does not.
    truth = {key: (dq, abs(du) + 0.1) for key, (dq, du) in SHIPPED.items()}
    costly = _saved(tmp_path / "costly")
    (docs, quality), plan, _ = _screened(costly, truth)
    assert _exp(plan, "subset").extra["stop"] == "no subset"
    assert _states(costly, docs, quality)["screen"]["effects"] == _exp(plan, "screen").extra["readout"]["effects"]
    rows = _exp(folder.plan, "screen").design["rows"]
    same = [i for i, row in enumerate(rows, start=1) if row[0] == row[1]]
    first = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, first, SHIPPED)
    docs = _unplanned(docs, same)
    plan, _ = _advance(folder, docs, quality)
    assert _exp(plan, "subset").extra["stop"] == "not identified"
    assert _states(folder, docs, quality)["screen"]["effects"] == []


def test_a_checks_pairs_are_the_pairs_check_keeps(tmp_path):
    folder = _saved(tmp_path)
    arm = _exp(folder.plan, MERGE).arms[0]
    specs = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(folder.plan, specs, FLAT, experiment=MERGE, k=5, **NOISY)
    r = check_mod.check(folder, arm, docs, quality=quality, fit=FIT)
    s = _states(folder, docs, quality)[MERGE]
    assert (s["pairs"], s["pairs_left"]) == (r.pairs, r.more_pairs) == (5, 19)

    # One run of a pair is no pair.
    half = [d for d in docs if d["run"]["id"].endswith(".001.a")]
    assert len(half) == 1 and _row(_states(folder, half, quality)[MERGE]) == ("running", "more", 0, 24)

    # A pair with a run that is not the one its pair planned is not counted, as `check` does not count it.
    wrong = json.loads(json.dumps(docs))
    next(d for d in wrong if d["run"]["id"].endswith(".002.a"))["run"]["flow"]["arm"] = "base"
    r = check_mod.check(folder, arm, wrong, quality=quality, fit=FIT)
    s = _states(folder, wrong, quality)[MERGE]
    assert (s["pairs"], s["pairs_left"]) == (r.pairs, r.more_pairs) == (4, 20) and len(r.extra["excluded"]) == 1

    # Every pair recorded, one score still missing: more, with no pair left to run.
    docs, quality = _ran(folder.plan, specs, FLAT, experiment=MERGE, **NOISY)
    late = {run_id: q for run_id, q in quality.items() if not run_id.endswith(".024.b")}
    r = check_mod.check(folder, arm, docs, quality=late, fit=FIT)
    assert (r.verdict, r.pairs, r.more_pairs) == ("more", 24, 0)
    assert _row(_states(folder, docs, late)[MERGE]) == ("running", "more", 24, 0)
    assert _row(_states(folder, docs, quality)[MERGE]) == ("done", "adopt", 24, 0)


def test_the_next_pairs_value_is_compares_for_a_cost_per_accepted_check(tmp_path):
    cpar = _cpar(tmp_path)
    plan = cpar.plan
    exp = _exp(plan, MERGE)
    specs = next_mod.next_runs(cpar, n=1000, seed=1, quality={}, fit=FIT)
    docs, quality = _ran(plan, specs, SHIPPED, experiment=MERGE, k=8, accept=_no_drop, **NOISY)
    s = _states(cpar, docs, quality)[MERGE]
    c = compare.paired(*_observations(docs, MERGE), rule=compare.ACCEPT, objective="cpar", max_pairs=exp.pair_cap,
                       lookahead=True, seed=plan.seed)
    assert (s["state"], s["verdict"], c.verdict, c.n_pairs) == ("running", "more", "more", 8)
    assert s["next_pair_value"] == c.next_pair_value_usd and s["next_pair_value"] > 0

    # Pricing the next pair changes nothing else in the check.
    on = check_mod.check_experiment(cpar, plan, exp, docs, quality=quality, fit=FIT, value=True)
    off = check_mod.check_experiment(cpar, plan, exp, docs, quality=quality, fit=FIT)
    assert on.extra["next_pair_value"] == s["next_pair_value"] and "next_pair_value" not in off.extra
    assert [(r.verdict, r.extra["stop"], r.p_quality, r.p_cost, r.threshold, r.pairs, r.more_pairs, r.extra["waiting"])
            for r in (on, off)] == [(off.verdict, None, off.p_quality, off.p_cost, off.threshold, 8, 16, 0)] * 2

    # A done check, a screen and a planned check have no value.
    docs, quality = _ran(plan, specs, SHIPPED, experiment=MERGE, accept=_no_drop, **NOISY)
    states = _states(cpar, docs, quality)
    assert states[MERGE]["state"] == "done"
    assert [s["next_pair_value"] for s in states.values()] == [None] * len(states)


def test_rule_d_and_best_under_cost_checks_have_no_value(tmp_path):
    best = _saved(tmp_path / "best", lambda flow: flow.update(
        objective={"kind": "best_under_cost", "score": "quality", "margin": 0.1, "cost_cap_usd": 5.0}))
    cheapest = _saved(tmp_path / "cheapest")
    for folder in (best, cheapest):
        specs = next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT)
        docs, quality = _ran(folder.plan, specs, FLAT, experiment=MERGE, k=8, **NOISY)
        s = _states(folder, docs, quality)[MERGE]
        assert (s["state"], s["verdict"], s["pairs"], s["next_pair_value"]) == ("running", "more", 8, None)
        r = check_mod.check_experiment(folder, folder.plan, _exp(folder.plan, MERGE), docs, quality=quality, fit=FIT,
                                       value=True)
        assert r.extra["next_pair_value"] is None


def test_a_saved_done_verdict_holds_with_no_value(tmp_path):
    folder = _saved(tmp_path)
    (docs, quality), plan, specs = _screened(folder)
    plan, _ = _advance(folder, *_ran(plan, specs, SHIPPED, docs, quality, experiment="subset", **NOISY))
    assert _exp(plan, "subset").verdict == "adopt"
    save_plan(folder.dir, plan)
    folder = load(folder.dir)
    lose = _ran(plan, specs, SHIPPED, docs, quality, experiment="subset", loss={"subset": 0.4}, **NOISY)
    r = check_mod.check_experiment(folder, plan, _exp(plan, "subset"), lose[0], quality=lose[1], fit=FIT)
    assert r.verdict == "reject"  # what the new documents alone would say
    s = _states(folder, *lose)["subset"]
    assert (*_row(s), s["next_pair_value"]) == ("done", "adopt", r.pairs, 0, None)
