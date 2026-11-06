"""The check of one change, on a saved plan of the example flow: the merge check against the base, runs built
from `next_runs` specs, with given qualities and dollars."""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from loopmath import compare
from loopmath.compare import model as compare_model
from loopmath.flows import check as check_mod
from loopmath.flows import judges
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows import simulate
from loopmath.flows.folder import FLOW_FILE, load, save_plan
from loopmath.flows.records import CheckResult
from loopmath.ocp.canonical import config_id, flow_digest
from loopmath.ocp.conformance import catalog_resolver, validate_doc
from loopmath.store.runs import run_cost

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "flows" / "lit-brief"
TRUTH = json.loads((EXAMPLE / "truth.json").read_text(encoding="utf-8"))
FIT = {"calibrated": False, "n_person": 0, "judges": [], "s_run": None}
ARM = "merge-extract-synthesize"
CAP = 24


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


def _specs(folder, experiment=ARM):
    return [s for s in next_mod.next_runs(folder, n=1000, seed=1, quality={}, fit=FIT) if s.experiment == experiment]


def _index(run_id: str) -> int:
    return int(run_id.split(".")[-2])


def _doc(spec, usd=1.0, signals=None) -> dict:
    run = {"id": spec.run_id, "flow": spec.flow.to_dict(), "task": {"id": spec.input},
           "slate": {"id": spec.slate, "members": list(spec.extra["slate_members"])}}
    if signals:
        run["signals"] = signals
    doc = {"ocp": "0.4", "kind": "run", "run": run, "nodes": []}
    if usd is not None:
        doc["attempts"] = [{"cost": {"usd": usd}}]
    return doc


def _world(folder, k, *, loss=0.0, cost=0.7, sd=0.05, seed=0):
    """Run documents and qualities for the first k pairs of the merge check."""
    rng = np.random.default_rng(seed)
    docs, quality = [], {}
    for s in sorted(_specs(folder), key=lambda s: s.run_id):
        cand = s.run_id.endswith(".a")
        usd = round(float((cost if cand else 1.0) * np.exp(0.1 * rng.standard_normal())), 6)
        q = 0.6 - (loss if cand else 0.0) + sd * rng.standard_normal()
        if _index(s.run_id) <= k:
            docs.append(_doc(s, usd))
            quality[s.run_id] = (float(q), None)
    return docs, quality


def _check(folder, docs, quality, arm=ARM):
    return check_mod.check(folder, arm, docs, quality=quality, fit=FIT)


# The verdicts


def test_a_clear_win_adopts_at_the_cap_and_not_before(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, CAP)
    r = _check(folder, docs, quality)
    assert (r.arm, r.reference, r.pairs, r.pair_cap, r.margin, r.threshold) == (ARM, "base", CAP, CAP, 0.1, 0.95)
    assert r.verdict == "adopt" and r.more_pairs is None and r.extra["stop"] == "decided"
    assert r.p_quality >= 0.95 and r.p_cost >= 0.95
    assert r.extra["rule"] == "one_look_early_reject" and r.extra["reasons"][0].startswith("check rule")
    before = _check(folder, [d for d in docs if _index(d["run"]["id"]) < CAP], quality)
    assert before.p_quality >= 0.95 and before.p_cost >= 0.95
    assert before.verdict == "more" and before.more_pairs == 1 and before.extra["stop"] is None


def test_a_clear_loss_rejects_before_the_cap(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, 6, loss=0.3)
    r = _check(folder, docs, quality)
    assert r.verdict == "reject" and r.extra["stop"] == "decided" and r.pairs == 6 and r.p_quality <= 0.05
    assert r.more_pairs is None


def test_undecided_before_the_cap_asks_for_the_rest(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, 4)
    r = _check(folder, docs, quality)
    assert r.verdict == "more" and r.pairs == 4 and r.more_pairs == CAP - 4 and r.extra["stop"] is None


def test_a_candidate_that_is_not_cheaper_is_rejected_at_the_cap(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, CAP, cost=1.0)
    r = _check(folder, docs, quality)
    assert 0.05 < r.p_cost < 0.95 and r.p_quality >= 0.95
    assert r.verdict == "reject" and r.extra["stop"] == "cap"


def test_a_dearer_candidate_is_rejected_early_on_cost(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, 8, cost=1.6)
    r = _check(folder, docs, quality)
    assert r.p_cost <= 0.05 and r.verdict == "reject" and r.extra["stop"] == "decided"


def test_an_unknown_cost_never_adopts(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, CAP)
    docs[3]["attempts"] = [{"cost": {}}]
    early = _check(folder, [d for d in docs if _index(d["run"]["id"]) <= 10], quality)
    assert early.p_cost is None and early.p_quality is not None
    assert early.verdict == "more" and early.more_pairs == CAP - 10
    r = _check(folder, docs, quality)
    assert r.p_cost is None and r.verdict == "reject" and r.extra["stop"] == "cost unknown"
    assert any("no dollar cost" in line for line in r.extra["reasons"])


# The final look


def test_at_the_cap_the_final_look_waits_for_every_score(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, CAP)
    some = {k: v for k, v in quality.items() if _index(k) <= 3}
    r = _check(folder, docs, some)
    assert r.pairs == CAP and r.extra["waiting"] == CAP - 3 and r.extra["scored_pairs"] == 3
    assert r.verdict == "more" and r.more_pairs == 0 and r.extra["stop"] is None
    assert any("final look waits" in line and "no pair past the cap" in line for line in r.extra["reasons"])
    nulls = {**quality, **{k: (None, None) for k in quality if _index(k) == 7}}
    assert _check(folder, docs, nulls).verdict == "more"
    assert _check(folder, docs, quality).verdict == "adopt"  # the scores arrive: one final look


def test_an_early_reject_still_fires_while_scores_wait(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, CAP, loss=0.3)
    r = _check(folder, docs, {k: v for k, v in quality.items() if _index(k) <= 6})
    assert r.verdict == "reject" and r.extra["stop"] == "decided" and r.extra["waiting"] == CAP - 6


# Which runs count


def test_lone_runs_and_runs_of_another_plan_revision_or_experiment_are_not_counted(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, 10)
    want = _check(folder, docs, quality)
    specs = {s.run_id: s for s in _specs(folder)}
    lone = specs[f"run_{folder.plan.id}.{ARM}.011.a"]
    other_plan = [_doc(specs[f"run_{folder.plan.id}.{ARM}.012.{x}"]) for x in "ab"]
    for d in other_plan:
        d["run"]["flow"]["plan"] = "pln_000000000000"
    other_revision = [_doc(specs[f"run_{folder.plan.id}.{ARM}.013.{x}"]) for x in "ab"]
    for d in other_revision:
        d["run"]["flow"]["version"] = 2
    screen = [_doc(s) for s in _specs(folder, "screen")[:6]]
    extra = [_doc(lone)] + other_plan + other_revision + screen
    more_quality = {**quality, **{d["run"]["id"]: (0.1, None) for d in extra}}
    got = _check(folder, docs + extra, more_quality)
    assert got == want and got.pairs == 10 and got.extra["excluded"] == []


@pytest.mark.parametrize("change, words", [
    (lambda run, folder: run["flow"].update(arm="base"), "run.flow.arm"),
    (lambda run, folder: run["flow"].update(split="train"), "run.flow.split"),
    (lambda run, folder: run.update(configuration={"id": folder.plan.arms[0].configuration["id"]}),
     "run.configuration.id"),
    (lambda run, folder: run["task"].update(id="q01"), "task.id"),
    (lambda run, folder: run["slate"].update(id="slt_other"), "run.slate"),
    (lambda run, folder: run["slate"].update(members=[run["id"], "run_other"]), "run.slate"),
    (lambda run, folder: run["flow"].pop("arm"), "run.flow.arm is missing"),
    (lambda run, folder: run["flow"].pop("split"), "run.flow.split is missing"),
], ids=["arm", "split", "configuration", "train-input", "slate-id", "slate-members", "no-arm", "no-split"])
def test_a_run_that_is_not_the_planned_one_leaves_its_pair_out(tmp_path, change, words):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, 10)
    key = f"{folder.plan.id}.{ARM}.005"
    target = next(d for d in docs if d["run"]["id"] == f"run_{key}.a")
    change(target["run"], folder)
    r = _check(folder, docs, quality)
    assert r.pairs == 9 and r.more_pairs == CAP - 9
    (gone,) = r.extra["excluded"]
    assert gone["pair"] == key and gone["run"] == f"run_{key}.a" and words in gone["reason"]
    assert any("not the one its pair planned" in line for line in r.extra["reasons"])


def test_both_runs_on_a_train_input_leave_the_pair_out(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, 10)
    for d in docs:
        if _index(d["run"]["id"]) == 2:
            d["run"]["task"]["id"] = "q01"
    r = _check(folder, docs, quality)
    assert r.pairs == 9 and r.extra["excluded"][0]["pair"].endswith(".002")


# Refusals and reading


def test_refusals(tmp_path):
    plain = tmp_path / "plain"
    shutil.copytree(EXAMPLE, plain)
    with pytest.raises(ValueError, match="no plan.json"):
        check_mod.check(load(plain), ARM, [])
    folder = _saved(tmp_path / "stale")
    doc = json.loads(folder.path.read_text(encoding="utf-8"))
    doc["flow"]["objective"]["margin"] = 0.05
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    folder.path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        check_mod.check(load(folder.dir), ARM, [])
    folder = _saved(tmp_path / "ok")
    with pytest.raises(ValueError, match="not in plan"):
        check_mod.check(folder, "no-such-arm", [])
    with pytest.raises(ValueError, match="reference"):
        check_mod.check(folder, "base", [])
    screen_arm = next(a for a in folder.plan.experiments[0].arms if a != "base")
    with pytest.raises(ValueError, match="only in the screen"):
        check_mod.check(folder, screen_arm, [])


def test_an_unknown_objective_kind_is_refused(tmp_path):
    folder = _saved(tmp_path, lambda flow: flow["objective"].update(kind="fastest"))
    with pytest.raises(ValueError, match="objective kind 'fastest'"):
        check_mod.check(folder, ARM, [])


def test_no_judge_is_called_without_a_measured_score(tmp_path, monkeypatch):
    folder = _saved(tmp_path)
    docs, _ = _world(folder, 10)

    def refuse(*a, **k):
        raise AssertionError("judges called")

    monkeypatch.setattr(judges, "fit", refuse)
    monkeypatch.setattr(judges, "combined_quality", refuse)
    monkeypatch.setattr(judges, "unpooled_quality", refuse)
    r = check_mod.check(folder, ARM, docs)
    assert r.p_quality is None and r.p_cost is not None and r.verdict == "more" and r.extra["waiting"] == 10


def test_the_check_is_deterministic_and_round_trips(tmp_path):
    folder = _saved(tmp_path)
    docs, quality = _world(folder, 12)
    one = _check(folder, docs, quality)
    assert one == _check(folder, list(reversed(docs)), quality)
    assert CheckResult.from_dict(json.loads(json.dumps(one.to_dict()))) == one


def test_no_pair_yet_asks_for_the_cap(tmp_path):
    folder = _saved(tmp_path)
    r = _check(folder, [], {})
    assert (r.verdict, r.pairs, r.more_pairs, r.p_quality, r.p_cost) == ("more", 0, CAP, None, None)


# The rule switch


def test_the_rule_is_one_switch(tmp_path, monkeypatch):
    folder = _saved(tmp_path)
    loss_docs, loss_q = _world(folder, 6, loss=0.3)
    win_docs, win_q = _world(folder, 12)
    monkeypatch.setattr(check_mod, "RULE", "one_look")
    r = _check(folder, loss_docs, loss_q)
    assert r.verdict == "more" and r.extra["rule"] == "one_look" and "reject at any look" not in r.extra["reasons"][0]
    monkeypatch.setattr(check_mod, "RULE", "every_look")
    r = _check(folder, win_docs, win_q)
    assert r.threshold == pytest.approx(0.995032, abs=1e-6)
    assert r.verdict == ("adopt" if min(r.p_quality, r.p_cost) >= r.threshold else "more")
    assert _check(folder, loss_docs, loss_q).verdict == "reject"
    monkeypatch.setattr(check_mod, "RULE", "one_look_early_reject")
    assert _check(folder, win_docs, win_q).verdict == "more"
    with pytest.raises(ValueError, match="rule 'sometimes'"):
        check_mod._decide(0.5, 0.5, "open", 0.95, rule="sometimes")


@pytest.mark.parametrize("p_quality, p_cost, look, rule, want", [
    (0.99, 0.99, "open", "one_look_early_reject", ("more", None)),
    (0.99, 0.99, "final", "one_look_early_reject", ("adopt", "decided")),
    (0.99, 0.90, "final", "one_look_early_reject", ("reject", "cap")),
    (0.04, 0.99, "open", "one_look_early_reject", ("reject", "decided")),
    (0.99, 0.05, "open", "one_look_early_reject", ("reject", "decided")),
    (0.04, 0.99, "open", "one_look", ("more", None)),
    (0.99, 0.99, "waiting", "one_look_early_reject", ("more", None)),
    (0.99, None, "cost unknown", "one_look_early_reject", ("reject", "cost unknown")),
    (0.999, 0.999, "open", "every_look", ("adopt", "decided")),
    (None, 0.99, "open", "one_look_early_reject", ("more", None)),
])
def test_decide(p_quality, p_cost, look, rule, want):
    assert check_mod._decide(p_quality, p_cost, look, 0.995032 if rule == "every_look" else 0.95, rule=rule) == want


# The other objective kinds


ACCEPT_RULE = {"name": "quality>=0.7+claims-check", "definition": "quality at least 0.7 and the claims check passes",
               "requires": ["claims-check"], "score": {"name": "quality", "target": 0.7, "better": "higher",
                                                        "scale": "fraction"}}


def _verdict(run_id, value):
    return [{"id": f"sig_{run_id}_v", "kind": "verdict", "name": "claims-check", "value": value}]


CPAR = {"kind": "cost_per_accepted", "margin": 0.1}
VERDICT_ONLY = {"name": "claims-check", "definition": "the claims check passes", "requires": ["claims-check"]}


def _cpar(tmp_path, rule=None):
    def change(flow):
        flow["objective"] = dict(CPAR)
        if rule is not None:
            flow["acceptance_rule"] = rule
    return _saved(tmp_path, change)


def test_cost_per_accepted_reads_each_run_under_the_flows_rule(tmp_path):
    folder = _cpar(tmp_path)
    assert folder.flow.acceptance_rule == ACCEPT_RULE
    accepted = check_mod._acceptance(folder.flow)
    spec = _specs(folder)[0]
    q = {spec.run_id: (0.8, None)}
    passing = _verdict(spec.run_id, "pass")
    assert accepted(_doc(spec, signals=passing), q) is True
    assert accepted(_doc(spec, signals=_verdict(spec.run_id, "fail")), q) is False
    assert accepted(_doc(spec, signals=passing), {spec.run_id: (0.6, None)}) is False  # under the target
    assert accepted(_doc(spec), q) is None  # no verdict yet
    assert accepted(_doc(spec, signals=passing), {}) is None  # no quality yet
    newer = [{**passing[0], "observed_at": "2026-11-03T10:00:00-08:00"},
             {**passing[0], "id": "sig_old", "value": "fail", "observed_at": "2026-11-02T10:00:00-08:00"}]
    assert accepted(_doc(spec, signals=newer), q) is True  # the newest verdict of a name counts
    revert = passing + [{"id": "sig_rv", "kind": "event", "name": "revert", "observed_at": "2026-11-01T00:00:00Z"}]
    assert accepted(_doc(spec, signals=revert), q) is False
    # A failed run is not accepted when nothing else decides it; complete passing evidence stands, as in compare.
    failed = _doc(spec)
    failed["attempts"] = [{"status": "failed", "cost": {"usd": 1.0}}]
    assert accepted(failed, q) is False
    failed["run"]["signals"] = passing
    assert accepted(failed, q) is True
    doc = _doc(spec, signals=passing)
    doc["attempts"] = [{"status": "failed", "cost": {"usd": 1.0}}]
    assert compare.observe(doc, "accept").failed and accepted(doc, q) is True


def _with_rule(folder, **change):
    """The folder with its loaded flow's acceptance rule changed, as a document written before the rule's
    members were bounded reads."""
    rule = {**folder.flow.acceptance_rule, **change}
    return dataclasses.replace(folder, flow=dataclasses.replace(folder.flow, acceptance_rule=rule))


@pytest.mark.parametrize("days", [10 ** 9, 10 ** 400, -10 ** 400, float("inf"), float("nan"), "x"],
                         ids=["1e9", "1e400", "-1e400", "inf", "nan", "text"])
def test_an_acceptance_window_no_timedelta_takes_is_refused_by_name(tmp_path, days):
    folder = _cpar(tmp_path)
    bad = _with_rule(folder, window_days=days)
    for call in (lambda: check_mod._acceptance(bad.flow), lambda: _check(bad, [], {})):
        with pytest.raises(ValueError, match="^acceptance_rule.window_days "):
            call()
    spec = _specs(folder)[0]
    revert = {"id": "sig_rv", "kind": "event", "name": "revert", "observed_at": "2026-10-31T00:00:00Z"}
    doc = _doc(spec, signals=_verdict(spec.run_id, "pass") + [revert])
    doc["run"]["ended_at"] = "2026-10-01T00:00:00Z"
    q = {spec.run_id: (0.8, None)}
    assert check_mod._acceptance(_with_rule(folder, window_days=999999999).flow)(doc, q) is False
    assert check_mod._acceptance(folder.flow)(doc, q) is True  # 14 days: the revert came after them


@pytest.mark.parametrize("target", [10 ** 400, float("inf"), float("nan"), True, "0.7"],
                         ids=["1e400", "inf", "nan", "bool", "text"])
def test_an_acceptance_target_no_float_holds_is_refused_by_name(tmp_path, target):
    folder = _cpar(tmp_path)
    bad = _with_rule(folder, score={**ACCEPT_RULE["score"], "target": target})
    for call in (lambda: check_mod._acceptance(bad.flow), lambda: _check(bad, [], {})):
        with pytest.raises(ValueError, match="^acceptance_rule.score.target "):
            call()
    spec = _specs(folder)[0]
    doc = _doc(spec, signals=_verdict(spec.run_id, "pass"))
    for good, quality, want in ((0.7, (0.8, None), True), (10 ** 300, (0.8, None), False), (None, None, True)):
        accepted = check_mod._acceptance(_with_rule(folder, score={**ACCEPT_RULE["score"], "target": good}).flow)
        assert accepted(doc, {} if quality is None else {spec.run_id: quality}) is want


def _cpar_world(folder, k, *, quality_on=True, missing=(), rates=(0.85, 0.6), seed=4):
    """Docs, injected quality and the Observations compare would get, for the first k pairs."""
    rng = np.random.default_rng(seed)
    docs, quality, ref, cand = [], {}, [], []
    target = (folder.flow.acceptance_rule.get("score") or {}).get("target")
    for s in sorted(_specs(folder), key=lambda s: s.run_id):
        i, is_cand = _index(s.run_id), s.run_id.endswith(".a")
        verdict = "pass" if rng.random() < rates[0 if is_cand else 1] else "fail"
        q = 0.8 if i % 5 else 0.65  # every fifth pair's runs fall under a 0.7 target
        usd = round(float((0.7 if is_cand else 1.0) * np.exp(0.2 * rng.standard_normal())), 6)
        if i > k:
            continue
        silent = s.run_id in missing
        docs.append(_doc(s, usd, signals=None if silent else _verdict(s.run_id, verdict)))
        if quality_on:
            quality[s.run_id] = (q, None)
        outcome = None if silent else (verdict == "pass" and (target is None or q >= target))
        (cand if is_cand else ref).append(compare.Observation(
            task=s.input, arm="cand" if is_cand else "base", outcome=outcome, usd=usd, run=s.run_id,
            pair=s.run_id[len("run_"):-2], experiment=ARM))
    return docs, quality, ref, cand


def _same_as_compare(r, ref, cand, plan):
    c = compare.paired(ref, cand, rule="accept", objective="cpar", max_pairs=CAP, lookahead=False, seed=plan.seed)
    assert (r.p_quality, r.p_cost) == (c.p_better, c.p_cheaper)
    assert r.threshold == round(c.adopt_threshold, 6) and r.extra["rule"] == "compare"
    assert r.extra["reasons"][-len(c.reasons):] == c.reasons
    return c


def test_cost_per_accepted_takes_compares_verdict_on_those_readings(tmp_path):
    folder = _cpar(tmp_path)
    one = f"run_{folder.plan.id}.{ARM}.009.a"
    docs, quality, ref, cand = _cpar_world(folder, 12, missing={one})
    r = _check(folder, docs, quality)
    c = _same_as_compare(r, ref, cand, folder.plan)
    assert (r.verdict, r.extra["stop"]) == (c.verdict, c.stop)
    assert r.extra["waiting"] == 1 and r.extra["scored_pairs"] == 11 and r.pairs == 12
    docs[0]["attempts"] = [{"cost": {}}]
    unknown = _check(folder, docs, quality)
    assert (unknown.verdict, unknown.p_quality, unknown.p_cost) == ("more", None, None)


def test_a_verdict_only_rule_needs_no_quality(tmp_path, monkeypatch):
    folder = _cpar(tmp_path, VERDICT_ONLY)
    docs, quality, ref, cand = _cpar_world(folder, CAP, quality_on=False)
    assert quality == {}

    def refuse(*a, **k):
        raise AssertionError("judges called")

    monkeypatch.setattr(judges, "fit", refuse)
    monkeypatch.setattr(judges, "combined_quality", refuse)
    monkeypatch.setattr(judges, "unpooled_quality", refuse)
    r = _check(folder, docs, {})  # the quality and fit are given (empty), so no judge is called
    c = _same_as_compare(r, ref, cand, folder.plan)
    assert r.extra["waiting"] == 0 and r.pairs == CAP and (r.verdict, r.extra["stop"]) == (c.verdict, c.stop)
    assert r.verdict != "more"  # the final look happens without a single quality score


def test_a_missing_required_verdict_waits_at_the_cap(tmp_path, monkeypatch):
    folder = _cpar(tmp_path)
    one = f"run_{folder.plan.id}.{ARM}.019.b"
    docs, quality, ref, cand = _cpar_world(folder, CAP, missing={one}, rates=(0.6, 0.6), seed=11)

    def refuse(*a, **k):
        raise AssertionError("judges called")

    monkeypatch.setattr(judges, "fit", refuse)
    monkeypatch.setattr(judges, "combined_quality", refuse)
    monkeypatch.setattr(judges, "unpooled_quality", refuse)
    r = _check(folder, docs, quality)  # every run has a number quality; one still lacks its verdict
    c = _same_as_compare(r, ref, cand, folder.plan)
    assert c.verdict == "reject" and c.stop == "cap"  # compare alone would close the check
    assert (r.verdict, r.more_pairs, r.extra["stop"], r.extra["waiting"]) == ("more", 0, None, 1)
    assert any("an acceptance reading" in line for line in r.extra["reasons"])


def test_best_under_cost_refuses_an_adopt_over_the_cost_cap(tmp_path):
    objective = {"kind": "best_under_cost", "score": "quality", "margin": 0.1}  # E215: a cost cap above 0
    roomy = _saved(tmp_path / "roomy", lambda flow: flow.update(objective={**objective, "cost_cap_usd": 5.0}))
    docs, quality = _world(roomy, 12, loss=-0.2)
    r = _check(roomy, docs, quality)
    assert r.verdict == "adopt" and r.extra["rule"] == "compare" and r.p_quality >= r.threshold
    tight = _saved(tmp_path / "tight", lambda flow: flow.update(objective={**objective, "cost_cap_usd": 0.5}))
    docs, quality = _world(tight, 12, loss=-0.2)
    r = _check(tight, docs, quality)
    assert r.verdict == "reject" and r.extra["stop"] == "cost cap"


def test_best_under_cost_never_adopts_with_a_run_unpriced(tmp_path):
    objective = {"kind": "best_under_cost", "score": "quality", "margin": 0.1, "cost_cap_usd": 1.0}
    folder = _saved(tmp_path, lambda flow: flow.update(objective=objective))
    docs, quality = _world(folder, CAP, loss=-0.2)
    early = [d for d in docs if _index(d["run"]["id"]) <= 12]
    assert _check(folder, early, quality).verdict == "adopt"  # every run priced: compare's adopt, under the cap
    target = next(d for d in early if d["run"]["id"].endswith(".003.a"))
    target["attempts"] = [{"cost": {}}]
    r = _check(folder, early, quality)
    assert (r.verdict, r.more_pairs, r.extra["stop"], r.p_cost) == ("more", CAP - 12, None, None)
    assert any("no dollar cost yet" in line for line in r.extra["reasons"])
    r = _check(folder, docs, quality)
    assert (r.verdict, r.extra["stop"]) == ("reject", "cost unknown")


# Dollars no float holds, and dollars the cost model states no range for


BUC = {"kind": "best_under_cost", "score": "quality", "margin": 0.1, "cost_cap_usd": 1.0}
KINDS = ("cheapest", "cpar", "buc")
SPREAD = ("the log costs spread far outside the cost model's range (a log-cost sd of 0.05 to 3), so no cost range "
          "is stated")
RANGED = {"1e300": ("a", 1e300), "1e308": ("a", 1e308), "zero": ("a", 0.0), "zero-base": ("b", 0.0)}


def _kind_world(tmp_path, kind, k):
    """A saved folder whose objective is `kind`, and its world's docs and quality for the first k pairs."""
    if kind == "cpar":
        folder = _cpar(tmp_path)
        docs, quality, _, _ = _cpar_world(folder, k)
    elif kind == "buc":
        folder = _saved(tmp_path, lambda flow: flow.update(objective=dict(BUC)))
        docs, quality = _world(folder, k, loss=-0.2)
    else:
        folder = _saved(tmp_path)
        docs, quality = _world(folder, k)
    return folder, docs, quality


def _costed(docs, run_id, attempts):
    """A copy of `docs` with the run's attempts replaced."""
    out = json.loads(json.dumps(docs))
    next(d for d in out if d["run"]["id"] == run_id)["attempts"] = attempts
    return out


def _strict(value):
    """`value` through standard JSON both ways: no NaN or infinity is written or read."""
    def refuse(constant):
        raise ValueError(f"{constant} is not standard JSON")
    return json.loads(json.dumps(value, allow_nan=False), parse_constant=refuse)


def _loop(tmp_path, folder, docs, quality, *, quiet=True):
    """check, experiment_states and advance on `docs`, with warnings as errors when `quiet`; the result, the states
    and the advanced plan, saved in a copy of the folder, each read back as standard JSON."""
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("error")
        r = _check(folder, docs, quality)
        states = check_mod.experiment_states(folder, docs, quality=quality, fit=FIT)
        plan = plan_mod.advance(folder, docs, quality=quality, fit=FIT)
    _strict(r.to_dict())
    _strict(states)
    where = tmp_path / f"saved-{len(list(tmp_path.glob('saved-*')))}"
    shutil.copytree(folder.dir, where)
    _strict(json.loads(save_plan(where, plan).read_text(encoding="utf-8")))
    return r, states, plan


def _read(r):
    return (r.verdict, r.extra["stop"], r.p_quality, r.p_cost, r.threshold, r.pairs, r.more_pairs,
            r.extra["waiting"], r.extra["scored_pairs"])


def _check_exp(plan):
    return next(e for e in plan.experiments if e.id == ARM)


def _without(plan, key):
    """`plan` with `key` left out of the check's extra."""
    return dataclasses.replace(plan, experiments=tuple(
        dataclasses.replace(e, extra={x: v for x, v in e.extra.items() if x != key}) if e.id == ARM else e
        for e in plan.experiments))


HUGE = {"sum": [{"cost": {"usd": 1e308}}, {"cost": {"usd": 1e308}}],  # each a float; their sum is not
        "integer": [{"cost": {"usd": 10 ** 400}}]}  # an integer no float holds, which the store's reader cannot convert


@pytest.mark.parametrize("k", [12, CAP])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dollars", list(HUGE))
def test_a_run_cost_no_float_holds_reads_as_a_missing_cost(tmp_path, kind, k, dollars):
    folder, docs, quality = _kind_world(tmp_path, kind, k)
    run_id = f"run_{folder.plan.id}.{ARM}.003.a"
    huge = _costed(docs, run_id, HUGE[dollars])
    doc = next(d for d in huge if d["run"]["id"] == run_id)
    if dollars == "sum":
        assert run_cost(doc)["usd"] == math.inf
    else:
        with pytest.raises(OverflowError):
            run_cost(doc)
    a, states_a, plan_a = _loop(tmp_path, folder, huge, quality)
    m, states_m, plan_m = _loop(tmp_path, folder, _costed(docs, run_id, [{"cost": {}}]), quality)
    assert _read(a) == _read(m)
    assert _read(a)[:2] == (("more", None) if k < CAP else ("reject", "cost unknown"))
    line = (f"1 run of the check has dollars that are not finite (NaN, or past what a float holds) ({run_id}): "
            "read as no dollar cost")
    assert line in a.extra["reasons"] and [x for x in a.extra["reasons"] if x != line] == m.extra["reasons"]
    assert a.extra["nonfinite"] == [f"cost:{run_id}"] and "nonfinite" not in m.extra
    assert a.extra.get("compare_reasons") == m.extra.get("compare_reasons")
    assert states_a == states_m
    assert _check_exp(plan_a).extra.get("nonfinite") == ([f"cost:{run_id}"] if k == CAP else None)
    assert _without(plan_a, "nonfinite") == plan_m


@pytest.mark.parametrize("k", [12, CAP])
@pytest.mark.parametrize("kind", KINDS)
def test_a_run_cost_of_nan_reads_as_a_missing_cost(tmp_path, kind, k):
    """NaN dollars, which standard JSON cannot hold, so set on the document in code: read as a missing cost, with
    the same verdict, states and plan, named in `nonfinite` as `cost:<run id>` and in the reasons as not finite."""
    folder, docs, quality = _kind_world(tmp_path, kind, k)
    run_id = f"run_{folder.plan.id}.{ARM}.003.a"
    nan = _costed(docs, run_id, [])
    doc = next(d for d in nan if d["run"]["id"] == run_id)
    doc["attempts"] = [{"cost": {"usd": float("nan")}}]
    assert math.isnan(run_cost(doc)["usd"])
    a, states_a, plan_a = _loop(tmp_path, folder, nan, quality)
    m, states_m, plan_m = _loop(tmp_path, folder, _costed(docs, run_id, [{"cost": {}}]), quality)
    assert _read(a) == _read(m)
    assert _read(a)[:2] == (("more", None) if k < CAP else ("reject", "cost unknown"))
    line = (f"1 run of the check has dollars that are not finite (NaN, or past what a float holds) ({run_id}): "
            "read as no dollar cost")
    assert line in a.extra["reasons"] and [x for x in a.extra["reasons"] if x != line] == m.extra["reasons"]
    assert a.extra["nonfinite"] == [f"cost:{run_id}"] and "nonfinite" not in m.extra
    assert a.extra.get("compare_reasons") == m.extra.get("compare_reasons")
    assert states_a == states_m
    assert _check_exp(plan_a).extra.get("nonfinite") == ([f"cost:{run_id}"] if k == CAP else None)
    assert _without(plan_a, "nonfinite") == plan_m


def test_a_valid_integer_cost_no_float_holds_reads_as_a_missing_cost(tmp_path):
    """A simulated run whose first attempt costs 10**400 dollars, an integer: the run validates and stays standard
    JSON, and check, experiment_states and advance read it as a sum no float holds, a missing cost named in
    `nonfinite`, where the store's reader would raise."""
    folder = _saved(tmp_path)
    _, quality = _world(folder, 12)
    specs = [s for s in _specs(folder) if _index(s.run_id) <= 12]
    docs = simulate.simulate(folder, arm="base", n=len(specs), seed=5, truth=TRUTH, specs=specs)
    target = next(d for d in docs if d["run"]["id"].endswith(".003.a"))
    target["attempts"][0]["cost"]["usd"] = 10 ** 400
    assert not [f for f in validate_doc(target) if f.level == "error"]
    assert _strict(target)["attempts"][0]["cost"]["usd"] == 10 ** 400
    with pytest.raises(OverflowError):
        run_cost(target)
    r, states, plan = _loop(tmp_path, folder, docs, quality)
    assert _read(r)[:2] == ("more", None)
    assert r.extra["nonfinite"] == [f"cost:{target['run']['id']}"]


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_only_an_overflow_in_the_cost_reader_is_read_as_a_missing_cost(tmp_path, monkeypatch, error):
    """The dollar guard catches OverflowError from the store's reader and nothing else: any other error from it
    raises through check, experiment_states and advance."""
    folder, docs, quality = _kind_world(tmp_path, "cheapest", 12)

    def broken(doc):
        raise error("a shape the reader refuses")

    monkeypatch.setattr(check_mod, "run_cost", broken)
    monkeypatch.setattr(plan_mod, "run_cost", broken)
    with pytest.raises(error, match="a shape the reader refuses"):
        _check(folder, docs, quality)
    with pytest.raises(error, match="a shape the reader refuses"):
        check_mod.experiment_states(folder, docs, quality=quality, fit=FIT)
    with pytest.raises(error, match="a shape the reader refuses"):
        plan_mod.advance(folder, docs, quality=quality, fit=FIT)


def _ranged_why(case, k):
    if case in ("zero", "zero-base"):
        who = "candidate" if case == "zero" else "base"
        return (f"the {who}'s recorded costs mix $0 and priced runs (1 of {k} at $0); a $0 run among priced ones "
                "spreads the log costs far outside what the cost model reads, so no cost range is stated")
    return SPREAD


def test_the_cost_models_range_reasons_are_the_ones_the_check_reads():
    assert compare_model.COST_RANGE_WHY == SPREAD and SPREAD.endswith(check_mod.NO_RANGE)
    assert _ranged_why("zero", 12).endswith(check_mod.NO_RANGE)


@pytest.mark.parametrize("k", [12, CAP])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("case", list(RANGED))
def test_dollars_the_cost_model_cannot_range_read_as_a_missing_cost(tmp_path, case, kind, k):
    folder, docs, quality = _kind_world(tmp_path, kind, k)
    side, usd = RANGED[case]
    run_id = f"run_{folder.plan.id}.{ARM}.003.{side}"
    r, states_r, plan_r = _loop(tmp_path, folder, _costed(docs, run_id, [{"cost": {"usd": usd}}]), quality)
    m, states_m, plan_m = _loop(tmp_path, folder, _costed(docs, run_id, [{"cost": {}}]), quality)
    why = _ranged_why(case, k)
    assert r.extra["cost_range"] == why and "cost_range" not in m.extra and "nonfinite" not in r.extra
    assert r.extra["reasons"].count(f"cost-range:{why}") == 1
    assert _read(r) == _read(m)
    assert _read(r)[:2] == (("more", None) if k < CAP else ("reject", "cost unknown"))
    shown = ("the cost model states no cost range for the check's dollars, so the candidate cannot be shown cheaper: "
             "keep the base" if k == CAP else "the cost model states no cost range for the check's dollars, so the "
             "check cannot adopt; at the cap that is a reject")
    assert shown in r.extra["reasons"] and not any("no dollar cost" in line for line in r.extra["reasons"])
    if kind == "cpar":  # compare is not asked with a missing cost, so its answer on the outcomes is not read
        assert r.extra["reasons"][-2:] == [f"cost-range:{why}", shown]
    assert [(s["id"], s["verdict"], s["next_pair_value"]) for s in states_r] == \
        [(s["id"], s["verdict"], s["next_pair_value"]) for s in states_m]
    assert _check_exp(plan_r).extra.get("cost_range") == (why if k == CAP else None)
    assert _without(plan_r, "cost_range") == plan_m


@pytest.mark.parametrize("k", [10, 12, CAP])
def test_compares_default_objective_is_cpar_until_it_states_no_cost_range(tmp_path, k):
    folder = _cpar(tmp_path)
    docs, quality, ref, cand = _cpar_world(folder, k)

    def blob(c):
        return json.dumps(dataclasses.asdict(c), sort_keys=True, default=str)

    for lookahead in (False, True):
        named = compare.paired(ref, cand, rule=compare.ACCEPT, objective="cpar", max_pairs=CAP, lookahead=lookahead,
                               seed=folder.plan.seed)
        default = compare.paired(ref, cand, rule=compare.ACCEPT, max_pairs=CAP, lookahead=lookahead,
                                 seed=folder.plan.seed)
        assert blob(default) == blob(named) and default.objective == "cpar" and not default.outcomes_only
    for case, (side, usd) in RANGED.items():
        run_id = f"run_{folder.plan.id}.{ARM}.003.{side}"
        obs = [dataclasses.replace(o, usd=usd) if o.run == run_id else o for o in ref + cand]
        base, priced = obs[:len(ref)], obs[len(ref):]
        c = compare.paired(base, priced, rule=compare.ACCEPT, max_pairs=CAP, lookahead=True, seed=folder.plan.seed)
        assert c.outcomes_only and c.cost_base is None and c.cost_cand is None
        assert check_mod._cost_range(c) == _ranged_why(case, k)
        with pytest.raises(ValueError, match="objective cpar, the cost per accepted result, needs every run's cost"):
            compare.paired(base, priced, rule=compare.ACCEPT, objective="cpar", max_pairs=CAP, lookahead=True,
                           seed=folder.plan.seed)


@pytest.mark.parametrize("usd", [None, 1e300])
@pytest.mark.parametrize("kind", ["cheapest", "cpar"])
def test_a_compare_shape_error_still_raises_through_the_check(tmp_path, monkeypatch, kind, usd):
    folder, docs, quality = _kind_world(tmp_path, kind, 12)
    if usd is not None:
        docs = _costed(docs, f"run_{folder.plan.id}.{ARM}.003.a", [{"cost": {"usd": usd}}])
    real = check_mod._observations

    def repeated(*args, **kwargs):
        ref, cand = real(*args, **kwargs)
        return [*ref, ref[0]], cand

    monkeypatch.setattr(check_mod, "_observations", repeated)
    for call in (lambda: _check(folder, docs, quality),
                 lambda: check_mod.experiment_states(folder, docs, quality=quality, fit=FIT),
                 lambda: plan_mod.advance(folder, docs, quality=quality, fit=FIT)):
        with pytest.raises(ValueError, match="each run is evidence once") as raised:
            call()
        assert any(entry.name == "pair_up" for entry in raised.traceback)


@pytest.mark.parametrize("k", [12, CAP])
@pytest.mark.parametrize("usd", [1e308, 1e300])
def test_every_cost_per_accepted_run_at_a_huge_cost_stays_standard_json(tmp_path, usd, k):
    # Compare's own arithmetic on these dollars may warn; the check, the states and the plan stay finite.
    folder = _cpar(tmp_path)
    docs, quality, _, _ = _cpar_world(folder, k)
    for doc in docs:
        doc["attempts"] = [{"cost": {"usd": usd}}]
    r, states, plan = _loop(tmp_path, folder, docs, quality, quiet=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        valued = check_mod.check_experiment(folder, folder.plan, _check_exp(folder.plan), docs, quality=quality,
                                            fit=FIT, value=True)
    _strict(valued.to_dict())
    assert "nonfinite" not in r.extra and "cost_range" not in r.extra and _read(valued) == _read(r)
    assert _read(r)[:2] == (("more", None) if k < CAP else ("reject", "cap"))
    value = valued.extra["next_pair_value"]
    assert value is None or math.isfinite(value)
    if usd == 1e300 and k < CAP:
        assert value is not None  # the value path is read, and its number is finite
    assert next(s for s in states if s["id"] == ARM)["next_pair_value"] == (value if k < CAP else None)
    done = _check_exp(plan)
    assert (done.state, done.verdict) == (("running", "more") if k < CAP else ("done", "reject"))


# Signals that are not a list

ABSENT = object()  # the run's `run.signals` left out
UNREADABLE = {"5": 5, "true": True, "text": "text", "object": {"k": 1}, "null": None, "false": False, "zero": 0,
              "empty-text": "", "empty-object": {}}  # each B's SIGNALS_NOT_LIST
READABLE = {"absent": ABSENT, "empty-list": []}
NO_SIGNALS = "read with no signals: run.signals is not a list (extra.unreadable)"
# Changes in quality and in dollars per run, by (edit kind, piece), for the screen runs below.
SHIPPED = {("model", "extract"): (0.0, -0.45), ("drop", "critic"): (-0.2, -0.15),
           ("effort", "synthesize"): (-0.03, -0.2), ("width", "scout"): (0.01, 0.15), ("budget", None): (0.0, 0.0)}


def _signals(docs, shape, *run_ids):
    """A copy of `docs` with each of the runs' `run.signals` set to `shape`, or left out for ABSENT."""
    out = json.loads(json.dumps(docs))
    for doc in out:
        if doc["run"]["id"] in run_ids:
            doc["run"].pop("signals", None)
            if shape is not ABSENT:
                doc["run"]["signals"] = json.loads(json.dumps(shape))
    return out


def _unnamed(r):
    """The check's result without `extra["unreadable"]` and its reason line."""
    extra = {x: v for x, v in r.extra.items() if x != "unreadable"}
    return dataclasses.replace(r, extra={**extra, "reasons": [x for x in r.extra["reasons"]
                                                              if not x.endswith(NO_SIGNALS)]})


@pytest.mark.parametrize("k", [12, CAP])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", list(UNREADABLE) + list(READABLE))
def test_a_run_whose_signals_are_not_a_list_reads_as_one_with_none_and_is_named(tmp_path, shape, kind, k):
    """The run is read as with its signals left out, in its pair as any run, by the rule that reads the pair then,
    and named in `extra["unreadable"]` and one reason line; a done check keeps the name in the plan. Absent
    signals and an empty list are read and named by nothing."""
    folder, docs, quality = _kind_world(tmp_path, kind, k)
    run_id = f"run_{folder.plan.id}.{ARM}.003.a"
    a, states_a, plan_a = _loop(tmp_path, folder, _signals(docs, {**UNREADABLE, **READABLE}[shape], run_id), quality)
    m, states_m, plan_m = _loop(tmp_path, folder, _signals(docs, ABSENT, run_id), quality)
    assert _read(a) == _read(m) and _unnamed(a) == _unnamed(m) and states_a == states_m
    done = _check_exp(plan_a).state == "done"
    assert done == ((kind, k) != ("cheapest", 12))
    if shape in READABLE:
        assert a.to_dict() == m.to_dict() and plan_a == plan_m
        return
    assert a.extra["unreadable"] == [f"signals:{run_id}"] and "unreadable" not in m.extra
    assert [x for x in a.extra["reasons"] if x.endswith(NO_SIGNALS)] == [f"1 run {NO_SIGNALS}"]
    assert not any(x.endswith(NO_SIGNALS) for x in m.extra["reasons"])
    assert _check_exp(plan_a).extra.get("unreadable") == ([f"signals:{run_id}"] if done else None)
    assert _without(plan_a, "unreadable") == plan_m


def test_a_done_checks_name_is_in_the_saved_plan(tmp_path):
    folder, docs, quality = _kind_world(tmp_path, "cheapest", CAP)
    run_id = f"run_{folder.plan.id}.{ARM}.003.a"
    plan = plan_mod.advance(folder, _signals(docs, 5, run_id), quality=quality, fit=FIT)
    assert (_check_exp(plan).state, _check_exp(plan).verdict) == ("done", "adopt")
    saved = _strict(json.loads(save_plan(folder.dir, plan).read_text(encoding="utf-8")))
    exp = next(e for e in saved["experiments"] if e["id"] == ARM)
    assert exp["unreadable"] == [f"signals:{run_id}"]  # an experiment's extra is written beside its fields
    assert load(folder.dir).plan == plan


@pytest.mark.parametrize("shape", list(UNREADABLE))
def test_with_no_score_anywhere_every_pair_waits_and_each_run_is_named_in_pair_order(tmp_path, shape):
    """No run holds a score, so the judges are not fitted and every pair of a score kind waits, as with the signals
    left out; the check names each kept run, the candidate before the reference, pair by pair."""
    folder, docs, _ = _kind_world(tmp_path, "cheapest", 12)
    ids = sorted(d["run"]["id"] for d in docs)
    every = _signals(docs, UNREADABLE[shape], *ids)
    a, m = check_mod.check(folder, ARM, every), check_mod.check(folder, ARM, docs)
    assert _unnamed(a) == _unnamed(m) and _read(a)[:2] == ("more", None) and a.extra["waiting"] == 12
    assert a.extra["unreadable"] == [f"signals:run_{folder.plan.id}.{ARM}.{i:03d}.{s}" for i in range(1, 13)
                                     for s in "ab"]
    assert f"24 runs {NO_SIGNALS}" in a.extra["reasons"]
    assert check_mod.experiment_states(folder, every) == check_mod.experiment_states(folder, docs)
    assert plan_mod.advance(folder, every) == plan_mod.advance(folder, docs)


@pytest.mark.parametrize("shape", list(UNREADABLE))
def test_a_failed_run_whose_signals_are_not_a_list_is_not_accepted(tmp_path, shape):
    """cost_per_accepted: a failed run with no readable signals reads False, as with its signals left out, so its
    pair does not wait."""
    folder = _cpar(tmp_path)
    docs, quality, _, _ = _cpar_world(folder, 12)
    run_id = f"run_{folder.plan.id}.{ARM}.003.a"
    failed = _costed(docs, run_id, [{"status": "failed", "cost": {"usd": 0.7}}])
    accepted = check_mod._acceptance(folder.flow)
    results = []
    for value in (UNREADABLE[shape], ABSENT):
        changed = _signals(failed, value, run_id)
        assert accepted(next(d for d in changed if d["run"]["id"] == run_id), quality) is False
        results.append(_check(folder, changed, quality))
    a, m = results
    assert _unnamed(a) == _unnamed(m) and a.extra["waiting"] == 0
    assert a.extra["unreadable"] == [f"signals:{run_id}"]


def test_members_that_are_not_objects_are_skipped_by_the_acceptance_reader(tmp_path):
    """Members that are not objects before a run's own signals: the acceptance reader reads the run's own
    verdicts, and the check names nothing."""
    folder = _cpar(tmp_path)
    docs, quality, _, _ = _cpar_world(folder, 12)
    mixed = json.loads(json.dumps(docs))
    for doc in mixed:
        doc["run"]["signals"] = ["text", 5] + doc["run"]["signals"]
    accepted = check_mod._acceptance(folder.flow)
    assert [accepted(d, quality) for d in mixed] == [accepted(d, quality) for d in docs]
    a, m = _check(folder, mixed, quality), _check(folder, docs, quality)
    assert a.to_dict() == m.to_dict() and "unreadable" not in a.extra


def test_an_error_in_the_shape_test_is_not_caught(tmp_path, monkeypatch):
    folder, docs, quality = _kind_world(tmp_path, "cheapest", 12)
    fault = TypeError("an error in the shape test")

    def broken(doc):
        raise fault

    monkeypatch.setattr(check_mod, "unreadable_signals", broken)
    for call in (lambda: _check(folder, docs, quality),
                 lambda: check_mod.experiment_states(folder, docs, quality=quality, fit=FIT),
                 lambda: plan_mod.advance(folder, docs, quality=quality, fit=FIT)):
        with pytest.raises(TypeError) as info:
            call()
        assert info.value is fault


def _screen_world(folder, *, accept=None):
    """Docs and quality for every screen pair, from SHIPPED with no noise; `accept(arm)` gives the claims check's
    verdict on each run's signals."""
    arms = {a.id: a for a in folder.plan.arms}
    docs, quality = [], {}
    for s in _specs(folder, "screen"):
        q, usd = 0.6, 1.0
        for e in arms[s.arm].edits:
            dq, du = SHIPPED.get((e.kind, e.piece), (0.0, 0.0))
            q, usd = q + dq, usd + du
        signals = None if accept is None else _verdict(s.run_id, "pass" if accept(arms[s.arm]) else "fail")
        docs.append(_doc(s, round(usd, 6), signals=signals))
        quality[s.run_id] = (q, None)
    return docs, quality


def _screen(plan):
    return next(e for e in plan.experiments if e.kind == "screen")


def _readouts(tmp_path, folder, docs, quality, run_id, shape):
    """The screen readout with the run's signals in `shape` and with them left out, and the advanced plans, each
    saved and read back as standard JSON."""
    out = []
    for value in (shape, ABSENT):
        changed = _signals(docs, value, run_id)
        readout = check_mod.screen_readout(folder, folder.plan, _screen(folder.plan), changed, quality=quality,
                                           fit=FIT)
        plan = plan_mod.advance(folder, changed, quality=quality, fit=FIT)
        where = tmp_path / f"saved-{len(list(tmp_path.glob('saved-*')))}"
        shutil.copytree(folder.dir, where)
        _strict(json.loads(save_plan(where, plan).read_text(encoding="utf-8")))
        out.append((_strict(readout), plan))
    return out


def _without_readout_name(plan):
    def strip(e):
        readout = {x: v for x, v in e.extra["readout"].items() if x != "unreadable"}
        return dataclasses.replace(e, extra={**e.extra, "readout": readout})
    return dataclasses.replace(plan, experiments=tuple(strip(e) if e.kind == "screen" else e
                                                       for e in plan.experiments))


@pytest.mark.parametrize("shape", list(UNREADABLE))
@pytest.mark.parametrize("status", ["chosen", "stop"])
def test_a_screen_readout_names_a_run_whose_signals_are_not_a_list(tmp_path, status, shape):
    """A done screen read on the given quality: the readout is the one with the run's signals left out plus
    `unreadable`, chosen or stopped (a run with no dollars), and the saved readout keeps the name."""
    folder = _saved(tmp_path)
    docs, quality = _screen_world(folder)
    run_id = f"run_{folder.plan.id}.screen.003.a"
    if status == "stop":
        docs = _costed(docs, run_id, [{"cost": {}}])
    (a, plan_a), (m, plan_m) = _readouts(tmp_path, folder, docs, quality, run_id, UNREADABLE[shape])
    assert a == {**m, "unreadable": [f"signals:{run_id}"]} and a["status"] == status
    assert _screen(plan_a).extra["readout"] == a and _without_readout_name(plan_a) == plan_m


@pytest.mark.parametrize("shape", list(UNREADABLE))
@pytest.mark.parametrize("kind", ["cheapest", "cpar"])
def test_a_waiting_screen_names_its_run_whose_signals_are_not_a_list(tmp_path, kind, shape):
    """The screen waits for a run with no reading: no quality for it, or, for cost_per_accepted, no verdict in
    its signals. The wait names it, as with its signals left out plus `unreadable`; a wait is never saved."""
    if kind == "cpar":
        folder = _cpar(tmp_path, VERDICT_ONLY)
        docs, quality = _screen_world(folder, accept=lambda arm: all(e.kind != "drop" for e in arm.edits))
    else:
        folder = _saved(tmp_path)
        docs, quality = _screen_world(folder)
    run_id = f"run_{folder.plan.id}.screen.003.a"
    if kind == "cheapest":
        del quality[run_id]
    (a, plan_a), (m, plan_m) = _readouts(tmp_path, folder, docs, quality, run_id, UNREADABLE[shape])
    assert a["status"] == "wait" and a["blocking"] == [run_id]
    assert a == {**m, "unreadable": [f"signals:{run_id}"]}
    assert "readout" not in _screen(plan_a).extra and plan_a == plan_m
