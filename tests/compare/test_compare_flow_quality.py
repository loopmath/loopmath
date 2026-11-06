"""Flow runs in compare: the score a flow run's own rule names is the judges' unpooled estimate.

A flow run (one with `run.flow`) that a judge scored under its own rule's score name reads that score from
`flows.judges.current_quality`, fitted once over the runs of both arms at each call site, and never reads a
judge's raw score. A run the judges give no unpooled estimate is not scored, and its judge warnings go on
`Observation.notes` and into the comparison's reasons. A flow run no judge scored under that name, any other
score name, and every coding run read as before. The runs are B's calibrated fixture: both arms of the example
flow on five inputs twice, two model judges on every run and a person on eight.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from loopmath import compare
from loopmath.belief import rules as belief_rules
from loopmath.belief.design import Unusable
from loopmath.compare import units
from loopmath.flows import judges
from loopmath.flows import next as next_mod
from loopmath.flows import plan as plan_mod
from loopmath.flows import simulate as sim
from loopmath.flows.folder import load, save_plan
from tests.flows.test_flow_judges import CALIBRATED_PROBE, EXAMPLE, TRUTH, calibrated_runs
from tests.flows.test_flow_plan import FAR, far_run_corpus

RULE = "score:quality"
NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone(timedelta(hours=-8)))
JUDGES = ("j-astra", "j-claude", "person")
POOLED = judges.combined_quality  # the pooled reader, kept as an oracle; `no_pooled_reading` refuses it in the tests


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flow-quality") / "lit-brief"
    docs = calibrated_runs(directory)
    return docs, load(directory).flow


@pytest.fixture(scope="module")
def far(tmp_path_factory):
    directory = tmp_path_factory.mktemp("flow-quality-far") / "lit-brief"
    docs, level = far_run_corpus(directory)
    return docs, load(directory).flow, level


@pytest.fixture(autouse=True)
def no_pooled_reading(monkeypatch):
    """Compare never reads the pooled posterior: `combined_quality` raises in every test here."""
    def refuse(*args, **kwargs):
        raise AssertionError("compare read judges.combined_quality")

    monkeypatch.setattr(judges, "combined_quality", refuse)


def _id(doc: dict) -> str:
    return doc["run"]["id"]


def _by_arm(docs):
    """The flow's two arms, base and drop-critic, as compare's base and candidate."""
    return ([d for d in docs if d["run"]["flow"]["arm"] == "base"],
            [d for d in docs if d["run"]["flow"]["arm"] != "base"])


def _by_repeat(docs):
    """The first repeat of each arm and input (all eight person-scored runs) against the second (none)."""
    return [d for d in docs if _id(d).endswith("-0")], [d for d in docs if _id(d).endswith("-1")]


def _changed(docs, run_id, change):
    """A copy of `docs` with `change` applied to a copy of run `run_id`."""
    out = copy.deepcopy(docs)
    change(next(d for d in out if _id(d) == run_id))
    return out


def _linear(doc, only=None, *, ranged):
    """Each judge's quality score of `doc` (only those of `only`, when given) on a 1 to 5 linear scale, with its
    range on the signal when `ranged`."""
    for s in doc["run"]["signals"]:
        ref = (s.get("source") or {}).get("ref")
        if s.get("kind") == "score" and s.get("name") == "quality" and (only is None or ref in only):
            s["value"], s["scale"] = 1 + 4 * s["value"], "linear"
            if ranged:
                s["ext"] = {judges.JUDGE_EXT: {"range": [1, 5]}}


def _paired(monkeypatch, base, cand, rule=RULE, **kw):
    """`compare.paired`, and the (quality map, observations) of each `_observations` call it makes."""
    seen = []
    real = compare._observations

    def spy(runs, rule_, now, config, **k):
        out = real(runs, rule_, now, config, **k)
        seen.append((k.get("quality"), out))
        return out

    monkeypatch.setattr(compare, "_observations", spy)
    try:
        return compare.paired(base, cand, rule=rule, now=NOW, **kw), seen
    finally:
        monkeypatch.setattr(compare, "_observations", real)


def _outcomes(seen) -> dict:
    """Run id to outcome, from the first two `_observations` calls (the base arm and the candidate arm)."""
    return {o.run: o.outcome for _, obs in seen[:2] for o in obs}


def _notes(seen) -> dict:
    return {o.run: o.notes for _, obs in seen[:2] for o in obs}


def _accepted(doc, quality):
    ev = belief_rules.evidence(doc, belief_rules.resolve_rule(doc), now=NOW,
                               quality=quality.get(_id(doc), (None, None)))
    return None if ev.z is None else bool(ev.z >= 0.5)


def _warnings(c) -> list[str]:
    return [r for r in c.reasons if "no score from the judges" in r or "read without" in r
            or "name no judge" in r]


def _reasons(doc, quality, readable=None) -> tuple[str, ...]:
    """The outcome reader's judge reasons for the run's score, in its order, given the run's entry in `quality`.

    A run whose stored rule the resolver refuses has no rule to be read under: its reasons are those of
    `readable`, the same run under a rule the resolver takes, read with no entry and less the reason for no
    value, then `rule:<the resolver's reason>` for its named refusal (`Unusable`), or `rule:unreadable` for a
    TypeError or ValueError it gives no reason for. Any other error of the resolver's call propagates."""
    try:
        rule = belief_rules.resolve_rule(doc)
    except Unusable as exc:
        refused = f"rule:{exc}"
    except (TypeError, ValueError):
        refused = "rule:unreadable"
    else:
        ev = belief_rules.evidence(doc, rule, now=NOW, quality=quality.get(_id(doc), (None, None)))
        return tuple(r for r in ev.reasons
                     if r.startswith(("unranged:", "unattributed:", "outside-range:", "nonfinite:", "nonnumeric:",
                                      "unfitted:")))
    assert readable is not None, "a run whose rule is refused is read through its readable twin"
    return (*[r for r in _reasons(readable, {}) if not r.startswith("unfitted:")], refused)


def _unattributed(doc, refs=None, ref="missing"):
    """Each judge's and person's quality score of `doc` (of the judges in `refs`, when given) with `ref` as its
    `source.ref`, or with none when it is "missing": a score that names no judge."""
    for s in doc["run"]["signals"]:
        source = s.get("source") or {}
        if s.get("kind") == "score" and source.get("kind") in ("judge", "user") \
                and (refs is None or source.get("ref") in refs):
            if ref == "missing":
                source.pop("ref", None)
            else:
                source["ref"] = ref


class Counter:
    """Counts calls of the real `current_quality` and `quality_map`."""

    def __init__(self, monkeypatch):
        self.current = self.maps = 0
        real_current, real_map = judges.current_quality, judges.quality_map

        def current(*a, **k):
            self.current += 1
            return real_current(*a, **k)

        def qmap(*a, **k):
            self.maps += 1
            return real_map(*a, **k)

        monkeypatch.setattr(judges, "current_quality", current)
        monkeypatch.setattr(judges, "quality_map", qmap)

    def reset(self):
        self.current = self.maps = 0

    def calls(self) -> int:
        """Fits made: `current_quality` calls, each with its one `quality_map`, and no one-run fit beside them."""
        assert self.maps == self.current, (self.current, self.maps)
        return self.current


# The joint fit


def test_the_score_rule_reads_the_joint_fit(cal, monkeypatch):
    docs, _ = cal
    base, cand = _by_arm(docs)
    joint = judges.current_quality(docs)
    probe = next(d for d in docs if _id(d) == CALIBRATED_PROBE)
    assert abs(joint[CALIBRATED_PROBE][0] - judges.current_quality([probe])[CALIBRATED_PROBE][0]) > 0.01
    c, seen = _paired(monkeypatch, base, cand)
    assert _outcomes(seen) == pytest.approx({_id(d): joint[_id(d)][0] for d in docs}, abs=1e-12)
    for _, obs in seen[:2]:
        assert all((o.scale, o.better, o.notes) == ("fraction", "higher", ()) for o in obs)
    assert c.n_pairs == 10 and _warnings(c) == []


def test_both_arms_read_one_fit_of_the_judges(cal, monkeypatch):
    """The person-scored runs sit in one arm only. Each judge has one bias in the compare, the same for both
    arms: every run reads the fit over both arms, which a fit of each arm alone does not give."""
    docs, flow = cal
    base, cand = _by_repeat(docs)
    assert all(any((s.get("source") or {}).get("kind") == "user" for s in d["run"]["signals"]) for d in base[:4])
    assert not any((s.get("source") or {}).get("kind") == "user" for d in cand for s in d["run"]["signals"])
    joint, alone = judges.current_quality(base + cand), judges.current_quality(cand)
    bias = {name: {j["id"]: j["bias"] for j in judges.fit(flow, side)["judges"]}
            for name, side in (("base", base), ("cand", cand))}
    assert abs(bias["base"]["j-claude"] - bias["cand"]["j-claude"]) > 0.01
    assert max(abs(joint[_id(d)][0] - alone[_id(d)][0]) for d in cand) > 0.01
    c, seen = _paired(monkeypatch, base, cand)
    assert seen[0][0] is seen[1][0]  # one map for both arms
    assert _outcomes(seen) == pytest.approx({_id(d): joint[_id(d)][0] for d in base + cand}, abs=1e-12)
    assert c.n_pairs == 10


def test_the_acceptance_rule_reads_the_joint_fit(cal, monkeypatch):
    docs, _ = cal
    base, cand = _by_repeat(docs)
    joint = judges.current_quality(docs)
    _, seen = _paired(monkeypatch, base, cand, rule="accept")
    assert _outcomes(seen) == {_id(d): _accepted(d, joint) for d in docs}
    assert any(v is not None for v in _outcomes(seen).values())


def test_one_run_is_fitted_alone(cal):
    docs, _ = cal
    probe = next(d for d in docs if _id(d) == CALIBRATED_PROBE)
    alone = judges.current_quality([probe])
    o = compare.observe(probe, RULE)
    assert (o.outcome, o.scale, o.better, o.notes) == (alone[CALIBRATED_PROBE][0], "fraction", "higher", ())
    assert compare.observe(probe, "accept", now=NOW).outcome == _accepted(probe, alone)


def test_a_run_far_from_its_arm_mean_reads_its_own_value(cal, monkeypatch):
    """Compare pools nothing itself: a run's value is its entry in `current_quality`, however far it sits from
    the rest of its arm."""
    docs, _ = cal
    base, cand = _by_arm(docs)
    real = judges.current_quality
    far = _id(cand[0])

    def current(*a, **k):
        out = dict(real(*a, **k))
        out[far] = (0.02, 0.01)
        return out

    monkeypatch.setattr(judges, "current_quality", current)
    _, seen = _paired(monkeypatch, base, cand)
    got = _outcomes(seen)
    assert got[far] == 0.02
    rest = [got[_id(d)] for d in cand[1:]]
    assert min(rest) - 0.02 > 0.2  # far from its arm, and read as given


def test_a_run_far_from_its_cell_reads_its_unpooled_estimate(far, monkeypatch):
    """With the real reader, on 5 runs a cell: `FAR`, 0.35 below its cell, reads its unpooled estimate, the
    `current_quality` entry for the same runs, as every run does. The pooled combined quality, pulled toward the
    cell, is more than 0.01 above it; it is read only as an oracle, before compare runs."""
    docs, flow, level = far
    pooled, _ = POOLED(flow, next(d for d in docs if _id(d) == FAR), judges.fit(flow, docs))
    every = judges.current_quality(docs)
    _, seen = _paired(monkeypatch, *_by_arm(docs))
    got = _outcomes(seen)
    assert got == {run_id: q for run_id, (q, _) in every.items()}
    assert abs(got[FAR] - (level - 0.35)) < 0.05 and got[FAR] + 0.01 < pooled


# Readers: what counts as a judge's score


def test_ranged_linear_scores_read_as_fractions(cal, monkeypatch):
    docs, _ = cal
    linear = copy.deepcopy(docs)
    for d in linear:
        _linear(d, ranged=True)
    for rule in (RULE, "accept"):
        _, frac = _paired(monkeypatch, *_by_arm(docs), rule=rule)
        _, lin = _paired(monkeypatch, *_by_arm(linear), rule=rule)
        assert _outcomes(lin) == pytest.approx(_outcomes(frac), abs=1e-9)
        assert all(n == () for n in _notes(lin).values())


def test_unranged_scores_alone_give_no_score(cal, monkeypatch):
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: _linear(d, ranged=False))
    raw = {s["value"] for d in changed if _id(d) == CALIBRATED_PROBE for s in d["run"]["signals"]
           if s.get("kind") == "score"}
    assert CALIBRATED_PROBE not in judges.current_quality(changed)
    for rule in (RULE, "accept"):
        c, seen = _paired(monkeypatch, *_by_arm(changed), rule=rule)
        o = next(o for _, obs in seen[:2] for o in obs if o.run == CALIBRATED_PROBE)
        assert o.notes == ("unranged:j-astra", "unranged:j-claude", "unfitted:quality"), o
        assert o.outcome in ((None,) if rule == RULE else (None, False)) and o.scale is None
        assert not raw & {o.outcome}
        assert _warnings(c) == [
            "1 candidate run gets no score from the judges: unfitted:quality (no unpooled estimate from the "
            "judges fit)",
            "1 candidate run read without judge j-astra, whose linear score has no range: unranged:j-astra",
            "1 candidate run read without judge j-claude, whose linear score has no range: unranged:j-claude"]


def test_an_unranged_score_beside_a_usable_one_reads_the_usable_one(cal, monkeypatch):
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: _linear(d, {"j-claude"}, ranged=False))
    joint = judges.current_quality(changed)
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    assert _outcomes(seen)[CALIBRATED_PROBE] == pytest.approx(joint[CALIBRATED_PROBE][0], abs=1e-12)
    assert _notes(seen)[CALIBRATED_PROBE] == ("unranged:j-claude",)
    assert _warnings(c) == ["1 candidate run read without judge j-claude, whose linear score has no range: "
                            "unranged:j-claude"]
    assert not any("unscored" in r for r in c.reasons)


@pytest.mark.parametrize("ref", ["missing", None, 7])
def test_unattributed_scores_alone_give_no_score(cal, monkeypatch, ref):
    """Scores that name no judge are never read raw: a run with only such scores is not scored, its notes are
    the outcome reader's reasons, and one line names the scores and points to `loopmath flow score`."""
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: _unattributed(d, ref=ref))
    probe = next(d for d in changed if _id(d) == CALIBRATED_PROBE)
    ids = [s["id"] for s in probe["run"]["signals"] if s.get("kind") == "score"]
    raw = {s["value"] for s in probe["run"]["signals"] if s.get("kind") == "score"}
    got = judges.judged(probe, "quality")
    assert len(ids) == 2 and got.usable == [] and got.unattributed == ids
    assert got.reasons == [f"unattributed:{i}" for i in ids]
    joint = judges.current_quality(changed)
    assert CALIBRATED_PROBE not in joint
    for rule in (RULE, "accept"):
        c, seen = _paired(monkeypatch, *_by_arm(changed), rule=rule)
        o = next(o for _, obs in seen[:2] for o in obs if o.run == CALIBRATED_PROBE)
        assert o.notes == tuple(f"unattributed:{i}" for i in ids) + ("unfitted:quality",) == _reasons(probe, joint)
        assert o.outcome == (None if rule == RULE else _accepted(probe, joint)) and o.scale is None
        assert not raw & {o.outcome}
        assert _warnings(c) == [
            f"judge scores that name no judge are not read (2 in candidate runs): unattributed:{ids[0]}, "
            f"unattributed:{ids[1]}; record a judge's score with loopmath flow score",
            "1 candidate run gets no score from the judges: unfitted:quality (no unpooled estimate from the "
            "judges fit)"]


def test_an_unattributed_score_beside_usable_ones_is_left_out(cal, monkeypatch):
    """With j-astra's score naming no judge, the run reads the joint fit's value from its other judges, and the
    score is named in its notes and the reasons."""
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: _unattributed(d, refs=("j-astra",)))
    probe = next(d for d in changed if _id(d) == CALIBRATED_PROBE)
    (sid,) = judges.judged(probe, "quality").unattributed
    joint = judges.current_quality(changed)
    assert joint[CALIBRATED_PROBE][0] is not None
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    assert _outcomes(seen)[CALIBRATED_PROBE] == pytest.approx(joint[CALIBRATED_PROBE][0], abs=1e-12)
    assert _notes(seen)[CALIBRATED_PROBE] == (f"unattributed:{sid}",) == _reasons(probe, joint)
    assert _warnings(c) == [f"judge scores that name no judge are not read (1 in candidate runs): "
                            f"unattributed:{sid}; record a judge's score with loopmath flow score"]
    assert not any("unscored" in r for r in c.reasons)
    _, seen = _paired(monkeypatch, *_by_arm(changed), rule="accept")
    assert _outcomes(seen)[CALIBRATED_PROBE] == _accepted(probe, joint)


def test_unattributed_entries_keep_their_multiplicity(cal, monkeypatch):
    """Two scores with no id that name no judge are two entries, in the notes and the count, as in the outcome
    reader's reasons."""
    docs, _ = cal

    def anonymous(d):
        _unattributed(d)
        for s in d["run"]["signals"]:
            if s.get("kind") == "score":
                del s["id"]

    changed = _changed(docs, CALIBRATED_PROBE, anonymous)
    probe = next(d for d in changed if _id(d) == CALIBRATED_PROBE)
    got = judges.judged(probe, "quality")
    assert got.usable == [] and got.unattributed == ["?", "?"]
    assert got.reasons == ["unattributed:?", "unattributed:?"]
    joint = judges.current_quality(changed)
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    assert _notes(seen)[CALIBRATED_PROBE] == ("unattributed:?", "unattributed:?", "unfitted:quality") \
        == _reasons(probe, joint)
    assert ("judge scores that name no judge are not read (2 in candidate runs): unattributed:?; record a judge's "
            "score with loopmath flow score") in c.reasons


def test_unattributed_entries_carry_through_cases_and_past_five_names(cal, monkeypatch):
    """Every score that names no judge stays a note and a count, run by run and case by case; the line names
    five and says how many more there are."""
    docs, _ = cal
    chosen = ("run_cal-drop-critic-q04-0", "run_cal-drop-critic-q04-1", "run_cal-drop-critic-q05-0",
              CALIBRATED_PROBE)
    changed = copy.deepcopy(docs)
    for d in changed:
        if _id(d) in chosen:
            _unattributed(d)
    ids = sorted(f"unattributed:{s['id']}" for d in changed if _id(d) in chosen for s in d["run"]["signals"]
                 if s.get("kind") == "score")
    assert len(ids) > 5
    line = (f"judge scores that name no judge are not read ({len(ids)} in candidate runs): {', '.join(ids[:5])} "
            f"and {len(ids) - 5} more; record a judge's score with loopmath flow score")
    joint = judges.current_quality(changed)
    base, cand = _by_arm(changed)
    c, seen = _paired(monkeypatch, base, cand)
    assert line in c.reasons
    assert sorted(w for o in seen[1][1] for w in o.notes if w.startswith("unattributed:")) == ids
    case_b, case_c = units.by_case(base, cand, rule=RULE)
    assert sorted(w for o in case_c for w in o.notes if w.startswith("unattributed:")) == ids
    q05 = next(o for o in case_c if o.task == "q05")
    assert sorted(q05.notes) == sorted(w for d in changed if _id(d) in chosen[2:] for w in _reasons(d, joint))
    assert line in compare.paired(case_b, case_c, rule=RULE).reasons


def test_a_flow_run_no_judge_scored_reads_its_score_as_before(cal, monkeypatch):
    docs, _ = cal

    def unjudged(d):
        d["run"]["signals"] = [s for s in d["run"]["signals"] if s.get("kind") != "score"] + [
            {"id": "sig_own", "kind": "score", "name": "quality", "value": 0.42, "scale": "fraction",
             "observed_at": "2026-01-05T10:00:00+00:00", "source": {"kind": "orchestrator", "ref": "synthesize"}}]

    changed = _changed(docs, CALIBRATED_PROBE, unjudged)
    probe = next(d for d in changed if _id(d) == CALIBRATED_PROBE)
    _, seen = _paired(monkeypatch, *_by_arm(changed))
    o = next(o for _, obs in seen[:2] for o in obs if o.run == CALIBRATED_PROBE)
    assert (o.outcome, o.scale, o.notes) == (0.42, "fraction", ())
    assert o.outcome == compare.last_score_signal(probe, "quality")["value"]


def test_another_score_name_reads_as_before(cal, monkeypatch):
    docs, _ = cal
    changed = copy.deepcopy(docs)
    for k, d in enumerate(changed):
        d["run"]["signals"].append({"id": f"sig_speed_{k}", "kind": "score", "name": "speed", "value": 0.1 * (k % 7),
                                    "observed_at": "2026-01-05T10:00:00+00:00",
                                    "source": {"kind": "judge", "ref": "j-claude"}})
    _, seen = _paired(monkeypatch, *_by_arm(changed), rule="score:speed")
    assert _outcomes(seen) == {_id(d): compare.last_score_signal(d, "speed")["value"] for d in changed}
    assert all(n == () for n in _notes(seen).values())


def test_an_unpooled_value_of_zero_is_measured(cal, monkeypatch):
    docs, _ = cal
    base, cand = _by_arm(docs)
    real = judges.current_quality

    def current(*a, **k):
        return {**real(*a, **k), CALIBRATED_PROBE: (0.0, 0.05)}

    monkeypatch.setattr(judges, "current_quality", current)
    _, seen = _paired(monkeypatch, base, cand)
    assert _outcomes(seen)[CALIBRATED_PROBE] == 0.0 and _notes(seen)[CALIBRATED_PROBE] == ()
    _, seen = _paired(monkeypatch, base, cand, rule="accept")
    assert _outcomes(seen)[CALIBRATED_PROBE] is False


def test_a_refused_fit_keeps_its_reason(cal, monkeypatch):
    """More judges with readings than a fit takes: every run of the group gets the refused fit's entry, whose
    reason reaches the notes, the reasons and the acceptance reader unchanged."""
    docs, _ = cal
    n = judges.MAX_SCORED_JUDGES + 1 - len(JUDGES)  # one judge past the limit, with the three there

    def crowded(d):
        d["run"]["signals"] += [{"id": f"sig_x{k}", "kind": "score", "name": "quality", "value": 0.5,
                                 "scale": "fraction", "observed_at": "2026-01-05T10:00:00+00:00",
                                 "source": {"kind": "judge", "ref": f"j-x{k:02d}"}} for k in range(n)]

    changed = _changed(docs, CALIBRATED_PROBE, crowded)
    reason = f"too-many-judges:{judges.MAX_SCORED_JUDGES + 1}"
    entries = judges.current_quality(changed)
    assert {getattr(v, "reason", None) for v in entries.values()} == {reason} and len(entries) == len(docs)
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    assert set(_outcomes(seen).values()) == {None}
    assert set(_notes(seen).values()) == {(reason,)}
    assert _warnings(c) == [f"10 base runs and 10 candidate runs get no score from the judges: {reason} (more "
                            "judges with readings than the judges fit takes)"]
    given = []
    real = belief_rules.evidence

    def evidence(doc, rule, **k):
        given.append(k.get("quality"))
        return real(doc, rule, **k)

    monkeypatch.setattr(belief_rules, "evidence", evidence)  # around compare's own read: the map is given
    obs = compare._observations(changed, "accept", NOW, None, quality=entries)
    assert len(given) == len(docs) and all(q is entries[_id(d)] for q, d in zip(given, changed))
    assert {o.notes for o in obs} == {(reason,)}


def test_warnings_count_source_runs_through_cases(cal, monkeypatch):
    """Several warnings on several runs: each is named once, with the runs per arm that carry it, whether the
    arms are compared run by run or case by case."""
    docs, _ = cal
    changed = copy.deepcopy(docs)
    by_id = {_id(d): d for d in changed}
    _linear(by_id["run_cal-base-q01-0"], {"j-claude"}, ranged=False)
    _linear(by_id["run_cal-base-q01-1"], {"j-claude", "j-astra"}, ranged=False)
    _linear(by_id["run_cal-drop-critic-q02-0"], ranged=False)
    expected = [
        "1 base run and 1 candidate run get no score from the judges: unfitted:quality (no unpooled estimate "
        "from the judges fit)",
        "1 candidate run read without judge person, whose linear score has no range: unranged:person",
        "1 base run and 1 candidate run read without judge j-astra, whose linear score has no range: "
        "unranged:j-astra",
        "2 base runs and 1 candidate run read without judge j-claude, whose linear score has no range: "
        "unranged:j-claude"]
    base, cand = _by_arm(changed)
    c, _ = _paired(monkeypatch, base, cand)
    assert sorted(_warnings(c)) == sorted(expected)
    case_b, case_c = units.by_case(base, cand, rule=RULE)
    q01 = next(o for o in case_b if o.task == "q01")
    # the first repeat keeps a score from j-astra and the person; the second, with no person, has none
    assert sorted(q01.notes) == ["unfitted:quality", "unranged:j-astra", "unranged:j-claude", "unranged:j-claude"]
    assert q01.outcome is not None
    assert sorted(_warnings(compare.paired(case_b, case_c, rule=RULE))) == sorted(expected)


# Scores the reader leaves out: outside their range, or no float holds them


def _valued(doc, values):
    """The quality scores of `doc` from each judge or person in `values` set to its `(scale, value)`: a linear
    scale with the range [0, 10] on the signal, any other scale as written (None: no scale)."""
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s.get("name") == "quality" and (s.get("source") or {}).get("ref") in values:
            scale, s["value"] = values[s["source"]["ref"]]
            s.pop("scale", None)
            s.pop("ext", None)
            if scale is not None:
                s["scale"] = scale
            if scale == "linear":
                s["ext"] = {judges.JUDGE_EXT: {"range": [0, 10]}}


def _newer(doc, ref, value, **changes):
    """A newer quality score of `value` from `ref` beside its own, with `changes` to the copied signal."""
    old = next(s for s in doc["run"]["signals"] if s.get("kind") == "score" and s.get("name") == "quality"
               and (s.get("source") or {}).get("ref") == ref)
    new = {**copy.deepcopy(old), "id": f"sig_newer_{ref}", "value": value, "observed_at": "2026-02-01T10:00:00+00:00"}
    new.update(changes)
    doc["run"]["signals"].append(new)


UNREAD = {  # the outcome reader's own cases (tests/belief/test_belief_flow_outcome.py): values, its reasons
    "outside": ({"j-claude": ("linear", 1e308), "j-astra": ("fraction", 1.5), "person": ("log", -3.0)},
                ["outside-range:j-astra", "outside-range:j-claude", "outside-range:person"]),
    "nonfinite": ({"j-claude": ("linear", 2 ** 1024), "j-astra": ("fraction", float("inf")),
                   "person": ("fraction", float("nan"))},
                  ["nonfinite:j-astra", "nonfinite:j-claude", "nonfinite:person"]),
    "mixed": ({"j-claude": ("linear", 2 ** 1024), "j-astra": ("linear", 11), "person": ("fraction", float("-inf"))},
              ["outside-range:j-astra", "nonfinite:j-claude", "nonfinite:person"]),
    "nonnumeric": ({"j-claude": ("fraction", "high"), "j-astra": ("fraction", None), "person": ("fraction", True)},
                   ["nonnumeric:j-astra", "nonnumeric:j-claude", "nonnumeric:person"]),
    "mixed-nonnumeric": ({"j-claude": ("linear", 2 ** 1024), "j-astra": ("linear", 11), "person": ("fraction", "high")},
                         ["outside-range:j-astra", "nonfinite:j-claude", "nonnumeric:person"]),
}


def _by_id(docs) -> dict:
    return {_id(d): d for d in docs}


def _three_judged(docs) -> str:
    """The first run scored by both model judges and the person."""
    return next(_id(d) for d in docs if {j.id for j in judges.signal_judges([d], "quality")} == set(JUDGES))


@pytest.mark.parametrize("scale, value", [("fraction", 1.5), ("linear", 10.5), ("linear", 1e308)])
def test_a_score_outside_its_range_beside_a_usable_one_is_named(cal, monkeypatch, scale, value):
    """j-claude's score outside its range: the run reads the joint fit's value from j-astra, and the judge is
    named in its notes, as the outcome reader names it, and in the reasons."""
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: _valued(d, {"j-claude": (scale, value)}))
    probe = next(d for d in changed if _id(d) == CALIBRATED_PROBE)
    got = judges.judged(probe, "quality")
    assert got.outside_range == ["j-claude"] and got.reasons == ["outside-range:j-claude"]
    joint = judges.current_quality(changed)
    assert joint[CALIBRATED_PROBE][0] is not None
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    assert _outcomes(seen)[CALIBRATED_PROBE] == pytest.approx(joint[CALIBRATED_PROBE][0], abs=1e-12)
    assert _notes(seen)[CALIBRATED_PROBE] == ("outside-range:j-claude",) == _reasons(probe, joint)
    assert _warnings(c) == ["1 candidate run read without judge j-claude, whose newest score lies outside its "
                            "range: outside-range:j-claude"]
    assert not any("unscored" in r for r in c.reasons)
    _, seen = _paired(monkeypatch, *_by_arm(changed), rule="accept")
    assert _outcomes(seen)[CALIBRATED_PROBE] == _accepted(probe, joint)
    assert _notes(seen)[CALIBRATED_PROBE] == ("outside-range:j-claude",)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 2 ** 1024], ids=["inf", "nan", "2**1024"])
def test_a_score_no_float_holds_leaves_its_judge_read_from_its_other_score(cal, monkeypatch, value):
    """A newer j-claude score that no float holds is no reading: every run reads as if it were not there, with
    j-claude read from its finite score, and the judge is named in the run's notes and in the reasons."""
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: _newer(d, "j-claude", value))
    probe = next(d for d in changed if _id(d) == CALIBRATED_PROBE)
    got = judges.judged(probe, "quality")
    assert got.nonfinite == ["j-claude"] and got.reasons == ["nonfinite:j-claude"]
    assert got.usable == judges.judged(_by_id(docs)[CALIBRATED_PROBE], "quality").usable
    joint = judges.current_quality(changed)
    for rule in (RULE, "accept"):
        c, seen = _paired(monkeypatch, *_by_arm(changed), rule=rule)
        _, before = _paired(monkeypatch, *_by_arm(docs), rule=rule)
        if rule == RULE:
            assert _outcomes(seen) == pytest.approx(_outcomes(before), abs=1e-12)
        else:
            assert _outcomes(seen) == _outcomes(before)
            assert _outcomes(seen)[CALIBRATED_PROBE] == _accepted(probe, joint)
        assert _notes(seen) == {**_notes(before), CALIBRATED_PROBE: ("nonfinite:j-claude",)}
        assert _notes(seen)[CALIBRATED_PROBE] == _reasons(probe, joint)
        assert _warnings(c) == ["1 candidate run read without the scores of judge j-claude that are not finite as a "
                                "float: nonfinite:j-claude"]


@pytest.mark.parametrize("value", [None, True, "high", [0.5], {"v": 0.5}],
                         ids=["null", "bool", "text", "list", "object"])
def test_a_score_that_is_not_a_number_leaves_its_judge_read_from_its_other_score(cal, monkeypatch, value):
    """A newer j-claude score that is not a number is no reading: every run reads as if it were not there, with
    j-claude read from its own number, and the judge is named in the run's notes and in the reasons with its own
    line, never the line for a run with no score."""
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: _newer(d, "j-claude", value))
    probe = next(d for d in changed if _id(d) == CALIBRATED_PROBE)
    got = judges.judged(probe, "quality")
    assert got.nonnumeric == ("j-claude",) and got.reasons == ["nonnumeric:j-claude"]
    assert got.usable == judges.judged(_by_id(docs)[CALIBRATED_PROBE], "quality").usable
    joint = judges.current_quality(changed)
    for rule in (RULE, "accept"):
        c, seen = _paired(monkeypatch, *_by_arm(changed), rule=rule)
        _, before = _paired(monkeypatch, *_by_arm(docs), rule=rule)
        if rule == RULE:
            assert _outcomes(seen) == pytest.approx(_outcomes(before), abs=1e-12)
        else:
            assert _outcomes(seen) == _outcomes(before)
            assert _outcomes(seen)[CALIBRATED_PROBE] == _accepted(probe, joint)
        assert _notes(seen) == {**_notes(before), CALIBRATED_PROBE: ("nonnumeric:j-claude",)}
        assert _notes(seen)[CALIBRATED_PROBE] == _reasons(probe, joint)
        assert _warnings(c) == ["1 candidate run read without the scores of judge j-claude that are not numbers: "
                                "nonnumeric:j-claude"]


def test_a_warning_kind_with_no_line_of_its_own_keeps_the_default_line(cal, monkeypatch):
    """A warning kind compare has no line for reads as a run with no score from the judges, with nothing more,
    and does not raise."""
    docs, _ = cal
    _, seen = _paired(monkeypatch, *_by_arm(docs))
    o = dataclasses.replace(seen[1][1][0], notes=("later-kind:j-claude",))
    assert compare._judge_warnings([], [o]) == ["1 candidate run gets no score from the judges: later-kind:j-claude"]
    assert compare._judge_warnings([o, o], []) == ["2 base runs get no score from the judges: later-kind:j-claude"]


@pytest.mark.parametrize("case", sorted(UNREAD))
def test_judge_scores_none_of_which_reads_give_no_score_and_name_each_judge(cal, monkeypatch, case):
    """Every judge's and the person's scores outside their range, not finite as a float or not numbers, beside a
    newer harness score under the name: the run is judged, so the harness score is never read raw. It has no
    score, its notes are the outcome reader's reasons in its order, and the reasons name each judge once and the
    missing value."""
    docs, _ = cal
    values, reasons = UNREAD[case]
    target = _three_judged(docs)

    def unread(d):
        _valued(d, values)
        d["run"]["signals"].append({"id": "sig_harness", "kind": "score", "name": "quality", "value": 0.9,
                                    "scale": "fraction", "better": "higher", "tier": "reported",
                                    "observed_at": "2026-11-01T12:00:00-07:00", "source": {"kind": "harness"}})

    changed = _changed(docs, target, unread)
    probe = _by_id(changed)[target]
    got = judges.judged(probe, "quality")
    assert not got.usable and got.reasons == reasons
    joint = judges.current_quality(changed)
    assert target not in joint
    runs = f"1 {'base' if probe['run']['flow']['arm'] == 'base' else 'candidate'} run"
    lines = {"outside-range": "{runs} read without judge {j}, whose newest score lies outside its range: {w}",
             "nonfinite": "{runs} read without the scores of judge {j} that are not finite as a float: {w}",
             "nonnumeric": "{runs} read without the scores of judge {j} that are not numbers: {w}"}
    expected = [lines[w.partition(":")[0]].format(runs=runs, j=w.partition(":")[2], w=w) for w in reasons] + [
        f"{runs} gets no score from the judges: unfitted:quality (no unpooled estimate from the judges fit)"]
    for rule in (RULE, "accept"):
        c, seen = _paired(monkeypatch, *_by_arm(changed), rule=rule)
        o = next(o for _, obs in seen[:2] for o in obs if o.run == target)
        assert o.notes == tuple(reasons) + ("unfitted:quality",) == _reasons(probe, joint)
        assert o.outcome == (None if rule == RULE else _accepted(probe, joint)) and o.scale is None
        assert o.outcome != 0.9
        assert sorted(_warnings(c)) == sorted(expected)


@pytest.mark.parametrize("read", [True, False])
def test_every_kind_of_left_out_score_is_named_in_the_readers_order(cal, monkeypatch, read):
    """One run with j-astra's score unranged, two scores with no id that name no judge, the person's score
    outside its range with a newer one that is not a number, and a newer j-claude score no float holds: the notes
    are the outcome reader's reasons, in its order and multiplicity, whether or not j-claude's finite score still
    reads."""
    docs, _ = cal
    target = _three_judged(docs)

    def every_kind(d):
        _linear(d, {"j-astra"}, ranged=False)
        _valued(d, {"person": ("fraction", 1.5)} if read else {"person": ("fraction", 1.5), "j-claude": ("log", 2.0)})
        _newer(d, "j-claude", float("inf"))
        _newer(d, "person", "high")
        for k in range(2):
            _newer(d, "j-claude", 0.5, source={"kind": "judge"})
            del d["run"]["signals"][-1]["id"]

    changed = _changed(docs, target, every_kind)
    probe = _by_id(changed)[target]
    got = judges.judged(probe, "quality")
    outside = ["person"] if read else ["j-claude", "person"]
    assert got.unranged == ["j-astra"] and got.unattributed == ["?", "?"]
    assert got.outside_range == outside and got.nonfinite == ["j-claude"] and got.nonnumeric == ("person",)
    assert bool(got.usable) is read
    reasons = (["unranged:j-astra", "unattributed:?", "unattributed:?"] + [f"outside-range:{j}" for j in outside]
               + ["nonfinite:j-claude", "nonnumeric:person"])
    assert got.reasons == reasons
    want = reasons + ([] if read else ["unfitted:quality"])
    joint = judges.current_quality(changed)
    assert (target in joint) is read
    arm = "base" if probe["run"]["flow"]["arm"] == "base" else "candidate"
    expected = [f"1 {arm} run read without judge j-astra, whose linear score has no range: unranged:j-astra",
                f"judge scores that name no judge are not read (2 in {arm} runs): unattributed:?; record a judge's "
                "score with loopmath flow score"]
    expected += [f"1 {arm} run read without judge {j}, whose newest score lies outside its range: outside-range:{j}"
                 for j in outside]
    expected.append(f"1 {arm} run read without the scores of judge j-claude that are not finite as a float: "
                    "nonfinite:j-claude")
    expected.append(f"1 {arm} run read without the scores of judge person that are not numbers: nonnumeric:person")
    if not read:
        expected.append(f"1 {arm} run gets no score from the judges: unfitted:quality (no unpooled estimate from the "
                        "judges fit)")
    for rule in (RULE, "accept"):
        c, seen = _paired(monkeypatch, *_by_arm(changed), rule=rule)
        o = next(o for _, obs in seen[:2] for o in obs if o.run == target)
        assert list(o.notes) == want and o.notes == _reasons(probe, joint)
        if rule == RULE:
            assert o.outcome == (pytest.approx(joint[target][0], abs=1e-12) if read else None)
        else:
            assert o.outcome == _accepted(probe, joint)
        assert sorted(_warnings(c)) == sorted(expected)


def test_left_out_scores_count_source_runs_through_cases(cal, monkeypatch):
    """j-claude's score outside its range with a newer one that is not a number, and a newer j-astra score no float
    holds, on two base runs of one case and one candidate run: each is named once with the runs per arm that carry
    it, whether the arms are compared run by run or case by case, and each run still reads a score."""
    docs, _ = cal
    chosen = ("run_cal-base-q01-0", "run_cal-base-q01-1", "run_cal-drop-critic-q02-0")
    changed = copy.deepcopy(docs)
    for d in changed:
        if _id(d) in chosen:
            _valued(d, {"j-claude": ("fraction", 1.5)})
            _newer(d, "j-astra", float("inf"))
            _newer(d, "j-claude", "high")
    expected = [
        "2 base runs and 1 candidate run read without the scores of judge j-astra that are not finite as a float: "
        "nonfinite:j-astra",
        "2 base runs and 1 candidate run read without the scores of judge j-claude that are not numbers: "
        "nonnumeric:j-claude",
        "2 base runs and 1 candidate run read without judge j-claude, whose newest score lies outside its range: "
        "outside-range:j-claude"]
    base, cand = _by_arm(changed)
    c, seen = _paired(monkeypatch, base, cand)
    assert _warnings(c) == expected
    assert all(_outcomes(seen)[r] is not None for r in chosen)
    assert all(_notes(seen)[r] == ("outside-range:j-claude", "nonfinite:j-astra", "nonnumeric:j-claude")
               for r in chosen)
    case_b, case_c = units.by_case(base, cand, rule=RULE)
    q01 = next(o for o in case_b if o.task == "q01")
    assert sorted(q01.notes) == ["nonfinite:j-astra", "nonfinite:j-astra", "nonnumeric:j-claude", "nonnumeric:j-claude",
                                 "outside-range:j-claude", "outside-range:j-claude"]
    assert _warnings(compare.paired(case_b, case_c, rule=RULE)) == expected


def test_left_out_scores_read_the_given_map_and_fit_nothing_again(cal, monkeypatch):
    """With the map given, a run whose scores the reader partly leaves out reads its entry, and a run missing
    from the map has no score, with the reader's reasons before `unfitted`: neither is fitted again."""
    docs, _ = cal

    def partly(d):
        _valued(d, {"j-claude": ("fraction", 1.5)})
        _newer(d, "j-astra", float("nan"))

    changed = _changed(docs, CALIBRATED_PROBE, partly)
    entries = judges.current_quality(changed)
    without = {k: v for k, v in entries.items() if k != CALIBRATED_PROBE}
    n = Counter(monkeypatch)
    for rule in (RULE, "accept"):
        for quality, notes in ((entries, ("outside-range:j-claude", "nonfinite:j-astra")),
                               (without, ("outside-range:j-claude", "nonfinite:j-astra", "unfitted:quality"))):
            obs = compare._observations(changed, rule, NOW, None, quality=quality)
            o = next(o for o in obs if o.run == CALIBRATED_PROBE)
            assert o.notes == notes
            if rule == RULE:
                assert o.outcome == (None if quality is without else pytest.approx(entries[CALIBRATED_PROBE][0],
                                                                                   abs=1e-12))
    assert n.calls() == 0


# One fit per call site


def test_one_fit_per_call_site(cal, monkeypatch):
    docs, _ = cal
    base, cand = _by_arm(docs)
    n = Counter(monkeypatch)
    for rule in (RULE, "accept"):
        n.reset()
        compare.paired(base, cand, rule=rule, now=NOW)
        assert n.calls() == 1, rule
    n.reset()
    c = compare.paired(base, cand, rule="review", now=NOW)  # nothing scored: the rules that would score it
    assert c.n_pairs == 0 and "the rules that score them" in " ".join(c.reasons)
    assert n.calls() == 1
    for call in (lambda: units.by_case(base, cand, rule=RULE),
                 lambda: compare._rules_that_score(base + cand),
                 lambda: compare._scoring_rules(base, cand, RULE, NOW, None),
                 lambda: compare.check_rule(RULE, base, cand)):
        n.reset()
        call()
        assert n.calls() == 1
    missing = _changed(docs, CALIBRATED_PROBE, lambda d: _linear(d, ranged=False))
    assert CALIBRATED_PROBE not in judges.current_quality(missing)
    n.reset()
    compare.paired(*_by_arm(missing), rule=RULE, now=NOW)
    assert n.calls() == 1  # a run missing from the map is not fitted again


def test_runs_with_no_flow_run_fit_nothing(cal, monkeypatch):
    from tests.compare.test_compare_paired import _doc
    from tests.compare.test_from_runs_unit import score_rows

    docs, _ = cal
    base, cand = _by_arm(docs)
    observed = ([compare.observe(d, RULE) for d in base], [compare.observe(d, RULE) for d in cand])
    coding = ([_doc(f"run_b{i}", f"t{i}", "pass" if i % 2 else "fail", 1.0) for i in range(6)],
              [_doc(f"run_c{i}", f"t{i}", "pass", 1.2) for i in range(6)])
    n = Counter(monkeypatch)
    compare.paired(*coding, rule="tests", now=NOW)
    compare.paired(*coding, rule="accept", now=NOW)
    compare.paired(*observed, rule=RULE)
    compare.paired(*score_rows(3, 2), rule="score:q")
    units.by_case(*score_rows(3, 2), rule="score:q")
    compare._rules_that_score(coding[0] + coding[1])
    assert (n.current, n.maps) == (0, 0)


def test_coding_runs_do_not_call_the_judges(monkeypatch):
    """Call isolation: with `current_quality` raising, coding runs compare as they did."""
    from tests.compare.test_compare_paired import _doc

    base = [_doc(f"run_b{i}", f"t{i}", "pass" if i % 3 else "fail", 1.0) for i in range(8)]
    cand = [_doc(f"run_c{i}", f"t{i}", "pass" if i % 4 else "fail", 0.9) for i in range(8)]
    before = {rule: compare.paired(base, cand, rule=rule, now=NOW).to_dict() for rule in ("tests", "accept")}

    def refuse(*a, **k):
        raise AssertionError("current_quality called on coding runs")

    monkeypatch.setattr(judges, "current_quality", refuse)
    assert {rule: compare.paired(base, cand, rule=rule, now=NOW).to_dict() for rule in before} == before


def test_one_pass_iterables(cal):
    docs, _ = cal
    base, cand = _by_arm(docs)
    for rule in (RULE, "accept"):
        lists = compare.paired(base, cand, rule=rule, now=NOW).to_dict()
        assert compare.paired(iter(base), iter(cand), rule=rule, now=NOW).to_dict() == lists
    lists = units.by_case(base, cand, rule=RULE)
    once = units.by_case(iter(base), iter(cand), rule=RULE)
    assert once == lists and [[o.notes for o in side] for side in once] == [[o.notes for o in side] for side in lists]


def test_coding_and_flow_runs_in_one_compare(cal, monkeypatch):
    """Coding runs scored on the fraction scale beside flow runs: the coding runs read as they do alone, and
    the flow runs read the fit over the flow runs of both arms."""
    from tests.compare.test_compare_paired import _doc

    docs, _ = cal
    flow_b, flow_c = _by_arm(docs)

    def scored(doc, value):
        doc["run"]["signals"] = [{"id": f"sig_{_id(doc)}", "kind": "score", "name": "quality", "value": value,
                                  "scale": "fraction", "better": "higher",
                                  "observed_at": "2026-01-15T11:00:00-08:00", "tier": "verified"}]
        return doc

    coding_b = [scored(_doc(f"run_b{i}", f"t{i}", None, 1.0), 0.4 + 0.05 * i) for i in range(6)]
    coding_c = [scored(_doc(f"run_c{i}", f"t{i}", None, 1.1), 0.45 + 0.05 * i) for i in range(6)]
    alone = {o.run: o for o in compare._observations(coding_b + coding_c, RULE, NOW, None)}
    joint = judges.current_quality(docs)
    c, seen = _paired(monkeypatch, coding_b + flow_b, coding_c + flow_c)
    got = {o.run: o for _, obs in seen[:2] for o in obs}
    for run_id, o in alone.items():
        assert got[run_id] == o and got[run_id].notes == ()
        assert json.dumps(o.__dict__, sort_keys=True, default=str) == json.dumps(got[run_id].__dict__,
                                                                                  sort_keys=True, default=str)
    assert {d: got[d].outcome for d in joint} == pytest.approx({d: v[0] for d, v in joint.items()}, abs=1e-12)
    assert c.n_pairs == 16 and c.verdict in ("adopt", "reject", "more")


# The compare command


def test_the_command_reads_one_fit_per_call_site(cal, monkeypatch, tmp_path, capsys):
    """`check-change from-runs` on a store holding the flow's runs: one fit for its rule check, one for its own
    read of both arms, and one in `paired` when it pairs runs (none when it pairs cases, which carry their
    runs' readings). The warnings reach its reasons."""
    from tests.compare.test_check_change_cli import home as _home  # noqa: F401
    from tests.compare.test_check_change_cli import lm

    docs, _ = cal
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "home").mkdir()
    changed = _changed(docs, "run_cal-drop-critic-q02-0", lambda d: _linear(d, ranged=False))
    src = tmp_path / "ocp"
    src.mkdir()
    for d in changed:
        (src / f"{_id(d)}.ocp.json").write_text(json.dumps(d), encoding="utf-8")
    code, _, err = lm(capsys, "run", "import", str(src), "--finish", "--no-fit", "--allow-simulated", "--json")
    assert code == 0, err
    arms = ("--base-label", "arm=base", "--cand-label", "arm=drop-critic", "--rule", RULE, "--json")
    n = Counter(monkeypatch)
    for unit, fits in (("case", 2), ("run", 3)):
        n.reset()
        code, out, err = lm(capsys, "check-change", "from-runs", *arms, "--unit", unit)
        assert code == 0, err
        assert out["unit"]["unit"] == unit and n.calls() == fits, (unit, n.current)
        assert "1 candidate run gets no score from the judges: unfitted:quality (no unpooled estimate from the " \
               "judges fit)" in out["comparison"]["reasons"]


# Admission comes before the flow runs are taken


def _maps(monkeypatch, call) -> list:
    """The quality maps `call` passes to `_observations`, in compare and in compare.units."""
    seen = []
    real = compare._observations

    def spy(runs, rule_, now, config, **k):
        seen.append(k.get("quality"))
        return real(runs, rule_, now, config, **k)

    monkeypatch.setattr(compare, "_observations", spy)
    monkeypatch.setattr(units, "_observations", spy)
    try:
        call()
    finally:
        monkeypatch.setattr(compare, "_observations", real)
        monkeypatch.setattr(units, "_observations", real)
    return seen


def _coding_score(run_id, task, value):
    from tests.compare.test_compare_paired import _doc

    doc = _doc(run_id, task, None, 1.0)
    doc["run"]["signals"] = [{"id": f"sig_{run_id}", "kind": "score", "name": "quality", "value": value,
                              "scale": "fraction", "better": "higher", "observed_at": "2026-01-15T11:00:00-08:00",
                              "tier": "verified"}]
    return doc


def test_the_map_is_current_quality_over_every_run_document(cal, monkeypatch):
    """Coding documents are admitted with the flow runs: a coding run given first with a flow run's id keeps
    that flow copy out of the fit, as `current_quality` on the same documents does, through cases and the rule
    diagnostics as well as a plain compare."""
    docs, _ = cal
    flow_b, flow_c = _by_arm(docs)
    unique = ([_coding_score(f"run_b{i}", f"t{i}", 0.5) for i in range(3)] + flow_b,
              [_coding_score(f"run_c{i}", f"t{i}", 0.6) for i in range(3)] + flow_c)
    twin = "run_cal-base-q01-0"
    repeated = ([_coding_score(twin, "t9", 0.5)] + flow_b, flow_c)
    for base, cand in (unique, repeated):
        expected = judges.current_quality(base + cand)
        calls = [lambda: compare.paired(base, cand, rule=RULE, now=NOW)] if base is unique[0] else []
        calls += [lambda: units.by_case(base, cand, rule=RULE),
                  lambda: compare._rules_that_score(base + cand),
                  lambda: compare._scoring_rules(base, cand, "accept", NOW, None)]
        for call in calls:
            maps = _maps(monkeypatch, call)
            assert maps and all(m == expected for m in maps)
    assert twin not in judges.current_quality(repeated[0] + repeated[1])
    case_b, _ = units.by_case(*repeated, rule=RULE)
    q01 = next(o for o in case_b if o.task == "q01")
    assert q01.notes == ("unfitted:quality",)
    assert q01.outcome == pytest.approx(judges.current_quality(repeated[0] + repeated[1])["run_cal-base-q01-1"][0])


def test_a_named_rule_on_another_score_name_reads_as_before(cal, monkeypatch):
    """The shared map is for the runs' own score name. A named acceptance rule on another score name reads each
    flow run as `evidence` does with no map (its own reading of that name), with no judge warnings."""
    docs, _ = cal
    changed = copy.deepcopy(docs)
    for k, d in enumerate(changed):
        for j in ("j-claude", "j-astra"):
            d["run"]["signals"].append({"id": f"sig_speed_{k}_{j}", "kind": "score", "name": "speed",
                                        "value": 0.1 + 0.08 * (k % 9), "scale": "fraction",
                                        "observed_at": "2026-01-05T10:00:00+00:00",
                                        "source": {"kind": "judge", "ref": j}})
    speed = belief_rules.stored_rule({"run": {"acceptance_rule": {
        "name": "speed>=0.4", "definition": "speed at least 0.4", "requires": [],
        "score": {"name": "speed", "target": 0.4, "better": "higher", "scale": "fraction"}}}})
    _, seen = _paired(monkeypatch, *_by_arm(changed), rule=speed)
    expected = {}
    for d in changed:
        ev = belief_rules.evidence(d, speed, now=NOW)
        expected[_id(d)] = None if ev.z is None else bool(ev.z >= 0.5)
    assert _outcomes(seen) == expected and len(set(expected.values()) - {None}) == 2
    assert all(n == () for n in _notes(seen).values())


# A stored rule the resolver refuses: a judged run reads no score, never a judge's raw value

REFUSED = {  # a change to the stored rule, and the reason the run reads under it
    "window_days 10**10": (lambda r: r.update(window_days=10 ** 10), "rule:window_days past 999999999"),
    "window_days 1e10": (lambda r: r.update(window_days=1e10), "rule:window_days past 999999999"),
    "target 10**400": (lambda r: r["score"].update(target=10 ** 400), "rule:score target not a finite number"),
    "target inf": (lambda r: r["score"].update(target=float("inf")), "rule:score target not a finite number"),
    "target high": (lambda r: r["score"].update(target="high"), "rule:unreadable"),
    "requires 5": (lambda r: r.update(requires=5), "rule:unreadable"),
}


@pytest.fixture(scope="module")
def lit_base():
    """Four base runs of the example flow, simulated at seed 5. The first one's newest quality score is the
    person's, 0.5243, and its own rule reads quality."""
    return sim.simulate(load(EXAMPLE), arm="base", n=4, seed=5, truth=TRUTH)


def test_the_control_run_reads_the_map_not_the_raw_score(lit_base):
    docs = copy.deepcopy(lit_base)
    probe = docs[0]
    assert compare.last_score_signal(probe, "quality")["value"] == 0.5243
    joint = judges.current_quality(docs)
    o = next(o for o in compare._observations(docs, RULE, NOW, None) if o.run == _id(probe))
    assert o.outcome == pytest.approx(joint[_id(probe)][0], abs=1e-12) and o.outcome != 0.5243
    assert o.notes == ()


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_a_judged_run_whose_rule_is_refused_reads_no_score(lit_base, case):
    """The resolver refuses the run's stored rule, so no rule names its score: alone or beside the other runs it
    has no score under `score:quality`, with the refusal named, and never reads the person's raw 0.5243."""
    change, reason = REFUSED[case]
    docs = copy.deepcopy(lit_base)
    probe = docs[0]
    twin = copy.deepcopy(probe)
    change(probe["run"]["acceptance_rule"])
    with pytest.raises((TypeError, ValueError)):
        belief_rules.resolve_rule(probe)
    got = judges.judged(probe, "quality")
    assert len(got.usable) == 3 and got.reasons == []
    assert compare.last_score_signal(probe, "quality")["value"] == 0.5243
    joint = judges.current_quality(docs)
    assert _id(probe) not in joint
    alone = compare.observe(probe, RULE, now=NOW)
    beside = next(o for o in compare._observations(docs, RULE, NOW, None) if o.run == _id(probe))
    for o in (alone, beside):
        assert (o.outcome, o.scale, o.better, o.notes) == (None, None, None, (reason,))
        assert o.notes == _reasons(probe, joint, readable=twin)


def test_a_refused_rule_is_named_in_the_comparison(cal, monkeypatch):
    """One candidate run with its rule refused has no score, and the comparison names the refusal in one line;
    every other run reads the joint fit's value."""
    docs, _ = cal
    changed = _changed(docs, CALIBRATED_PROBE, lambda d: REFUSED["window_days 10**10"][0](d["run"]["acceptance_rule"]))
    joint = judges.current_quality(changed)
    assert CALIBRATED_PROBE not in joint
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    outcomes = _outcomes(seen)
    assert outcomes.pop(CALIBRATED_PROBE) is None
    assert outcomes == pytest.approx({r: v[0] for r, v in joint.items()}, abs=1e-12)
    assert _notes(seen)[CALIBRATED_PROBE] == ("rule:window_days past 999999999",)
    assert _notes(seen)[CALIBRATED_PROBE] == _reasons(_by_id(changed)[CALIBRATED_PROBE], joint,
                                                      readable=_by_id(docs)[CALIBRATED_PROBE])
    assert _warnings(c) == ["1 candidate run gets no score from the judges: rule:window_days past 999999999 (the "
                            "run's stored rule is refused, so no score is read from its judges)"]


def test_a_refused_rule_keeps_the_readers_reasons_before_it(cal, monkeypatch):
    """A refused rule on a run whose judge scores already carry a reason: the notes are the reader's reasons in
    its order, then the refusal, and each has its line."""
    docs, _ = cal

    def both(d):
        _newer(d, "j-claude", float("inf"))
        REFUSED["window_days 10**10"][0](d["run"]["acceptance_rule"])

    changed = _changed(docs, CALIBRATED_PROBE, both)
    probe = _by_id(changed)[CALIBRATED_PROBE]
    got = judges.judged(probe, "quality")
    assert got.usable and got.reasons == ["nonfinite:j-claude"]
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    assert _outcomes(seen)[CALIBRATED_PROBE] is None
    assert _notes(seen)[CALIBRATED_PROBE] == ("nonfinite:j-claude", "rule:window_days past 999999999")
    twin = _by_id(_changed(docs, CALIBRATED_PROBE, lambda d: _newer(d, "j-claude", float("inf"))))[CALIBRATED_PROBE]
    assert _notes(seen)[CALIBRATED_PROBE] == _reasons(probe, {}, readable=twin)
    assert _warnings(c) == [
        "1 candidate run read without the scores of judge j-claude that are not finite as a float: nonfinite:j-claude",
        "1 candidate run gets no score from the judges: rule:window_days past 999999999 (the run's stored rule is "
        "refused, so no score is read from its judges)"]


def test_only_the_rule_call_is_guarded(cal, monkeypatch):
    """A shape error outside the resolver's call still raises, whether the run's rule resolves or is refused, and
    under `accept` a refused rule still raises the resolver's refusal, as before."""
    docs, _ = cal
    probe = next(d for d in docs if _id(d) == CALIBRATED_PROBE)
    refused = copy.deepcopy(probe)
    REFUSED["window_days 10**10"][0](refused["run"]["acceptance_rule"])
    with pytest.raises(Unusable, match="window_days past 999999999"):
        compare.observe(refused, compare.ACCEPT, now=NOW)

    def broken(doc, name):
        raise TypeError("a shape error in the judges' reader")

    monkeypatch.setattr(judges, "judged", broken)
    for doc in (probe, refused):
        with pytest.raises(TypeError, match="shape error in the judges"):
            compare.observe(doc, RULE, now=NOW)


def test_a_refused_rule_on_a_run_no_judge_scored_reads_as_before(lit_base):
    """A flow run no judge scored under the name has no judge value to fall back on: with its rule refused it
    reads its newest score under the name, as before."""
    doc = copy.deepcopy(lit_base[0])
    doc["run"]["signals"] = [s for s in doc["run"]["signals"]
                             if not (s.get("kind") == "score" and s.get("name") == "quality")]
    doc["run"]["signals"].append({"id": "sig_harness", "kind": "score", "name": "quality", "value": 0.9,
                                  "scale": "fraction", "better": "higher", "tier": "reported",
                                  "observed_at": "2026-11-01T12:00:00-07:00", "source": {"kind": "harness"}})
    REFUSED["window_days 10**10"][0](doc["run"]["acceptance_rule"])
    assert not any(judges.judged(doc, "quality"))
    o = compare.observe(doc, RULE, now=NOW)
    assert (o.outcome, o.scale, o.notes) == (0.9, "fraction", ())


def _refused_candidates(docs) -> tuple[list[dict], list[str]]:
    """A copy of `docs` with every candidate run's stored rule refused (`window_days` 10**10), and their ids."""
    changed = copy.deepcopy(docs)
    ids = []
    for d in changed:
        if d["run"]["flow"]["arm"] != "base":
            REFUSED["window_days 10**10"][0](d["run"]["acceptance_rule"])
            ids.append(_id(d))
    return changed, ids


def test_a_comparison_whose_candidate_rules_are_all_refused_names_them(cal, monkeypatch):
    """Every candidate run's stored rule refused: no pair is scored, and the comparison says so, naming the
    refusal among the rules that score the runs and in the judge warnings, rather than raising."""
    docs, _ = cal
    changed, ids = _refused_candidates(docs)
    assert len(ids) == 10
    c, seen = _paired(monkeypatch, *_by_arm(changed))
    assert c.n_pairs == 0
    assert all(_outcomes(seen)[r] is None and _notes(seen)[r] == ("rule:window_days past 999999999",) for r in ids)
    assert all(_notes(seen)[r] == _reasons(_by_id(changed)[r], {}, readable=_by_id(docs)[r]) for r in ids)
    assert c.reasons[0] == ("no pair is scored under the rule score:quality (10 pairs, none with both runs scored); "
                            "the rules that score them: claims-check (10 of 10 pairs), none for 10 runs whose stored "
                            "rule is refused: rule:window_days past 999999999")
    assert _warnings(c) == ["10 candidate runs get no score from the judges: rule:window_days past 999999999 (the "
                            "run's stored rule is refused, so no score is read from its judges)"]


def test_the_command_still_refuses_a_refused_stored_rule_by_name(cal, monkeypatch, tmp_path, capsys):
    """The rule check stays strict: `check-change from-runs` on a store holding one run whose stored rule the
    resolver refuses (a run stored before validation refused that rule) exits on the resolver's named refusal,
    with no traceback, byte for byte as before; under --json the error is the one object."""
    from loopmath import cli

    docs, _ = cal
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "home").mkdir()
    src = tmp_path / "ocp"
    src.mkdir()
    for d in docs:
        (src / f"{_id(d)}.ocp.json").write_text(json.dumps(d), encoding="utf-8")
    assert cli.main(["run", "import", str(src), "--finish", "--no-fit", "--allow-simulated", "--json"]) == 0
    stored = tmp_path / "home" / "runs" / f"{CALIBRATED_PROBE}.ocp.json"
    doc = json.loads(stored.read_text(encoding="utf-8"))
    REFUSED["window_days 10**10"][0](doc["run"]["acceptance_rule"])
    stored.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(Unusable, match="window_days past 999999999"):
        compare.check_rule(RULE, *_by_arm([doc if _id(d) == CALIBRATED_PROBE else d for d in docs]))
    arms = ["check-change", "from-runs", "--base-label", "arm=base", "--cand-label", "arm=drop-critic", "--rule", RULE]
    seen = []
    for extra in ([], ["--json"]):
        capsys.readouterr()
        code = cli.main(arms + extra)
        seen.append((code, *capsys.readouterr()))
    assert seen == [(1, "", "error: window_days past 999999999\n"),
                    (1, '{\n "schema": "loopmath.check-change/1",\n "ok": false,\n "error": "window_days past '
                        '999999999",\n "exit": 1\n}\n', "error: window_days past 999999999\n")]


def test_the_all_refused_comparison_guards_only_the_rule_call(cal, monkeypatch):
    """Past the resolver's call nothing is caught: a shape error in naming the other rules still raises, and
    under `accept` the refused rules still raise the resolver's refusal, as before."""
    docs, _ = cal
    changed, _ = _refused_candidates(docs)
    with pytest.raises(Unusable, match="window_days past 999999999"):
        compare.paired(*_by_arm(changed), rule=compare.ACCEPT, now=NOW)

    def broken(*args, **kwargs):
        raise TypeError("a shape error in the rule choices")

    monkeypatch.setattr(compare, "rule_choices", broken)
    with pytest.raises(TypeError, match="shape error in the rule choices"):
        compare.paired(*_by_arm(changed), rule=RULE, now=NOW)


class _ResolverFault(RuntimeError):
    """An error of the resolver's call that is neither its named refusal nor a TypeError or ValueError."""


def test_a_resolver_error_that_is_no_refusal_is_not_caught(cal, monkeypatch):
    """Only the resolver's named refusal reads as a refused rule: any other error from its call propagates, the
    very instance raised, from `observe` and from `paired`."""
    docs, _ = cal
    real = belief_rules.resolve_rule
    fault = _ResolverFault("the resolver failed")

    def faulty(doc, *args, **kwargs):
        # only compare's guarded call fails: the judges' fit admits the run through the real resolver
        if sys._getframe(1).f_code.co_name == "_run_rule" and _id(doc) == CALIBRATED_PROBE:
            raise fault
        return real(doc, *args, **kwargs)

    monkeypatch.setattr(belief_rules, "resolve_rule", faulty)
    probe = next(d for d in docs if _id(d) == CALIBRATED_PROBE)
    for read in (lambda: compare.observe(probe, RULE, now=NOW),
                 lambda: compare.paired(*_by_arm(docs), rule=RULE, now=NOW)):
        with pytest.raises(_ResolverFault) as info:
            read()
        assert info.value is fault


def test_the_reasons_helper_reads_the_resolver_by_its_own_call(cal, monkeypatch):
    """`_reasons` on real refused documents gives the refusal after its twin's reasons, named or unreadable; an
    error of the resolver's call that is no refusal propagates from it, the very instance raised."""
    docs, _ = cal
    probe = _by_id(docs)[CALIBRATED_PROBE]
    for case in ("window_days 10**10", "target 10**400", "target high", "requires 5"):
        refused = copy.deepcopy(probe)
        REFUSED[case][0](refused["run"]["acceptance_rule"])
        assert _reasons(refused, {}, readable=probe) == (REFUSED[case][1],)
    fault = _ResolverFault("the resolver failed")

    def faulty(doc, *args, **kwargs):
        raise fault

    monkeypatch.setattr(belief_rules, "resolve_rule", faulty)
    with pytest.raises(_ResolverFault) as info:
        _reasons(probe, {}, readable=probe)
    assert info.value is fault


def test_one_refused_run_is_named_in_the_singular(cal, monkeypatch):
    """One readable base run and one candidate run whose stored rule is refused: no pair is scored, and the
    comparison names the one run in the singular."""
    docs, _ = cal
    base = [copy.deepcopy(d) for d in docs if _id(d) == "run_cal-base-q05-1"]
    cand = [copy.deepcopy(d) for d in docs if _id(d) == CALIBRATED_PROBE]
    REFUSED["window_days 10**10"][0](cand[0]["run"]["acceptance_rule"])
    c, seen = _paired(monkeypatch, base, cand)
    assert c.n_pairs == 0 and _notes(seen)[CALIBRATED_PROBE] == ("rule:window_days past 999999999",)
    assert _outcomes(seen)["run_cal-base-q05-1"] is not None
    assert _notes(seen)[CALIBRATED_PROBE] == _reasons(cand[0], {}, readable=_by_id(docs)[CALIBRATED_PROBE])
    assert c.reasons[0] == ("no pair is scored under the rule score:quality (1 pair, none with both runs scored); "
                            "the rules that score them: claims-check (1 of 1 pairs), none for 1 run whose stored rule "
                            "is refused: rule:window_days past 999999999")
    assert _warnings(c) == ["1 candidate run gets no score from the judges: rule:window_days past 999999999 (the "
                            "run's stored rule is refused, so no score is read from its judges)"]


# Signals that are not a list


ABSENT = object()  # the run's `run.signals` left out
UNREADABLE = {"5": 5, "true": True, "text": "text", "object": {"k": 1}, "null": None, "false": False, "zero": 0,
              "empty-text": "", "empty-object": {}}  # each B's SIGNALS_NOT_LIST
READABLE = {"absent": ABSENT, "empty-list": []}
NO_SIGNALS = "read with no signals: run.signals is not a list: unreadable:signals"
BASE_PROBE = "run_cal-base-q05-1"


def _shaped(docs, shape, *run_ids):
    """A copy of `docs` with each of the runs' `run.signals` set to `shape`, or left out for ABSENT."""
    out = copy.deepcopy(docs)
    for doc in out:
        if _id(doc) in run_ids:
            doc["run"].pop("signals", None)
            if shape is not ABSENT:
                doc["run"]["signals"] = copy.deepcopy(shape)
    return out


def _unnamed(c) -> dict:
    """The comparison as a dict, without the line naming runs read with no signals."""
    return {**dataclasses.asdict(c), "reasons": [r for r in c.reasons if not r.endswith(NO_SIGNALS)]}


WHERE = {"base": ([BASE_PROBE], f"1 base run {NO_SIGNALS}"),
         "cand": ([CALIBRATED_PROBE], f"1 candidate run {NO_SIGNALS}"),
         "every": (None, f"10 base runs and 10 candidate runs {NO_SIGNALS}")}


@pytest.mark.parametrize("shape", list(UNREADABLE) + list(READABLE))
@pytest.mark.parametrize("rule", [RULE, compare.ACCEPT])
@pytest.mark.parametrize("where", list(WHERE))
def test_a_run_whose_signals_are_not_a_list_reads_as_one_with_none_and_is_named(cal, monkeypatch, where, rule,
                                                                                shape):
    """One run in the base arm, one in the candidate arm, or every run: each reads as with its signals left out,
    with `unreadable:signals` after the reader's own notes and one line in the reasons. Absent signals and an
    empty list are named by nothing."""
    docs, _ = cal
    ids, line = WHERE[where]
    ids = ids or [_id(d) for d in docs]
    named = shape in UNREADABLE
    a, seen_a = _paired(monkeypatch, *_by_arm(_shaped(docs, {**UNREADABLE, **READABLE}[shape], *ids)), rule=rule)
    m, seen_m = _paired(monkeypatch, *_by_arm(_shaped(docs, ABSENT, *ids)), rule=rule)
    assert _outcomes(seen_a) == _outcomes(seen_m)
    token = ("unreadable:signals",) if named else ()
    assert _notes(seen_a) == {r: n + (token if r in ids else ()) for r, n in _notes(seen_m).items()}
    assert _unnamed(a) == _unnamed(m) and not any(r.endswith(NO_SIGNALS) for r in m.reasons)
    assert [r for r in a.reasons if r.endswith(NO_SIGNALS)] == ([line] if named else [])


@pytest.mark.parametrize("shape", list(UNREADABLE))
def test_one_run_read_alone_carries_the_note_after_the_readers_own(cal, shape):
    """`observe` on one run: under each rule the outcome is the one with the signals left out, and the notes are
    that reading's notes then `unreadable:signals`."""
    docs, _ = cal
    for rule in (RULE, compare.ACCEPT):
        a = compare.observe(_by_id(_shaped(docs, UNREADABLE[shape], CALIBRATED_PROBE))[CALIBRATED_PROBE], rule, now=NOW)
        m = compare.observe(_by_id(_shaped(docs, ABSENT, CALIBRATED_PROBE))[CALIBRATED_PROBE], rule, now=NOW)
        assert (a.outcome, a.scale, a.better) == (m.outcome, m.scale, m.better)
        assert a.notes == m.notes + ("unreadable:signals",)


@pytest.mark.parametrize("shape", list(UNREADABLE) + list(READABLE))
def test_rule_choices_read_a_run_whose_signals_are_not_a_list_as_one_with_none(cal, shape):
    docs, _ = cal
    value = {**UNREADABLE, **READABLE}[shape]
    for ids in ([CALIBRATED_PROBE], [_id(d) for d in docs]):
        assert compare.rule_choices(*_by_arm(_shaped(docs, value, *ids))) == \
            compare.rule_choices(*_by_arm(_shaped(docs, ABSENT, *ids)))


def test_members_that_are_not_objects_are_skipped_by_the_score_and_name_readers(cal):
    """Members that are not objects before each run's own signals: the score reader and the rule choices' name
    reader read the runs' own signals, and no run is named."""
    docs, _ = cal
    mixed = copy.deepcopy(docs)
    for doc in mixed:
        doc["run"]["signals"] = ["text", 5] + doc["run"]["signals"]
    for doc, other in zip(docs, mixed):
        assert compare.last_score_signal(other, "quality") == compare.last_score_signal(doc, "quality")
        for rule in (RULE, compare.ACCEPT):
            assert "unreadable:signals" not in compare.observe(other, rule, now=NOW).notes
    assert compare._names(mixed) == compare._names(docs)


# Members that are not objects, read once as the plan reads them


@pytest.fixture(scope="module")
def lit_eight(tmp_path_factory):
    """The first eight runs the example flow's plan asks for, simulated on the base arm at seed 1."""
    target = tmp_path_factory.mktemp("flow-quality-eight") / "lit-brief"
    shutil.copytree(EXAMPLE, target)
    truth = json.loads((target / "truth.json").read_text(encoding="utf-8"))
    save_plan(target, plan_mod.make_plan(load(target)))
    folder = load(target)
    specs = next_mod.next_runs(folder, n=8, seed=1)
    return sim.simulate(folder, arm="base", n=len(specs), seed=1, truth=truth, specs=specs), folder.flow


def _plan_map(flow, docs) -> dict:
    """The plan's own reading of `docs` (`plan.quality_and_fit`, neither given) as plain tuples, less the runs it
    gives no estimate, which `current_quality` leaves out."""
    quality, _ = plan_mod.quality_and_fit(flow, {_id(d): d for d in docs}, None, None)
    return {r: tuple(q) for r, q in quality.items() if q[0] is not None}


def _members(doc):
    """Members that are not objects, text and a number, before the run's own signals."""
    doc["run"]["signals"] = ["text", 5] + doc["run"]["signals"]


def _handed(monkeypatch) -> list:
    """The list of documents each `judges.current_quality` call is handed, in the list returned."""
    handed, real = [], judges.current_quality

    def spy(docs, *args, **kwargs):
        handed.append(list(docs))
        return real(docs, *args, **kwargs)

    monkeypatch.setattr(judges, "current_quality", spy)
    return handed


def _corpora(cal, lit_eight) -> list:
    """(flow, docs, the two sides, the probe run) for the calibrated runs, its candidate probe, and for the
    eight simulated base runs, four against four, the first run."""
    (docs, flow), (eight, flow8) = cal, lit_eight
    return [(flow, docs, _by_arm, CALIBRATED_PROBE), (flow8, eight, lambda ds: (ds[:4], ds[4:]), _id(eight[0]))]


def test_members_that_are_not_objects_are_read_as_the_plan_reads_them(cal, lit_eight, monkeypatch):
    """Members that are not objects before one flow run's own signals: compare's map is the plan's own reading
    of the same documents, for that run and every other, the comparison is the one without those members, in one
    fit, and the run read alone reads as it does without them. No run gets no score from the judges for them."""
    for flow, docs, sides, probe in _corpora(cal, lit_eight):
        bad = _changed(docs, probe, _members)
        quality = compare._flow_quality(*sides(bad))
        assert {r: tuple(q) for r, q in quality.items()} == _plan_map(flow, bad) == _plan_map(flow, docs)
        assert probe in quality
        counter = Counter(monkeypatch)
        a, seen_a = _paired(monkeypatch, *sides(bad))
        assert counter.calls() == 1
        m, seen_m = _paired(monkeypatch, *sides(docs))
        assert _outcomes(seen_a) == _outcomes(seen_m) and _notes(seen_a) == _notes(seen_m)
        assert _outcomes(seen_a)[probe] == quality[probe][0] and not any(
            n.startswith("unfitted:") for n in _notes(seen_a)[probe])
        assert dataclasses.asdict(a) == dataclasses.asdict(m)
        alone, clean = compare.observe(_by_id(bad)[probe], RULE, now=NOW), compare.observe(_by_id(docs)[probe], RULE,
                                                                                          now=NOW)
        assert alone == clean and alone.outcome is not None


def test_a_flow_run_with_members_that_are_not_objects_goes_in_as_a_copy(cal, lit_eight, monkeypatch):
    """The flow run reaches `current_quality` as a copy whose signals are its object members, in order, and
    every other run as the very document given; the documents given are unchanged."""
    for _, docs, sides, probe in _corpora(cal, lit_eight):
        bad = _changed(docs, probe, _members)
        before = json.dumps(bad, sort_keys=True)
        handed = _handed(monkeypatch)
        compare.paired(*sides(bad), rule=RULE, now=NOW)
        given = _by_id(bad)
        assert len(handed) == 1 and sorted(_id(d) for d in handed[0]) == sorted(given)
        for doc in handed[0]:
            if _id(doc) == probe:
                assert doc is not given[probe] and doc == _by_id(docs)[probe]
            else:
                assert doc is given[_id(doc)]
        assert json.dumps(bad, sort_keys=True) == before


def test_a_run_that_is_not_a_flow_run_goes_in_as_it_is(cal, monkeypatch):
    """A run with no `run.flow` whose signals hold members that are not objects, beside the flow runs: it reaches
    `current_quality` as the very document given, gets no entry, and the flow runs read as the plan reads them."""
    docs, flow = cal
    other = copy.deepcopy(_by_id(docs)[BASE_PROBE])
    other["run"].pop("flow")
    other["run"]["id"] = "run_not-a-flow-run"
    _members(other)
    before = json.dumps(other, sort_keys=True)
    handed = _handed(monkeypatch)
    base, cand = _by_arm(docs)
    quality = compare._flow_quality(base + [other], cand)
    assert len(handed) == 1 and len(handed[0]) == len(docs) + 1
    assert all(doc is given for doc, given in zip(handed[0], base + [other] + cand)) and _id(other) not in quality
    assert {r: tuple(q) for r, q in quality.items()} == _plan_map(flow, docs)
    assert json.dumps(other, sort_keys=True) == before


@pytest.mark.parametrize("shape", list(UNREADABLE) + list(READABLE))
def test_signals_that_are_not_a_list_go_in_as_they_are(cal, monkeypatch, shape):
    """A flow run whose `run.signals` is not a list, left out or an empty list reaches `current_quality` as the
    very document given, and the comparison keeps its `unreadable:signals` note and line."""
    docs, _ = cal
    shaped = _shaped(docs, {**UNREADABLE, **READABLE}[shape], CALIBRATED_PROBE)
    handed = _handed(monkeypatch)
    c, seen = _paired(monkeypatch, *_by_arm(shaped))
    given = _by_id(shaped)
    assert len(handed) == 1 and all(doc is given[_id(doc)] for doc in handed[0])
    named = shape in UNREADABLE
    assert ("unreadable:signals" in _notes(seen)[CALIBRATED_PROBE]) == named
    assert [r for r in c.reasons if r.endswith(NO_SIGNALS)] == ([f"1 candidate run {NO_SIGNALS}"] if named else [])


@pytest.mark.parametrize("rule", [RULE, compare.ACCEPT])
def test_attempts_that_are_text_still_raise_where_they_raise(cal, rule):
    """A top-level `attempts` that is text is a shape error of its own, which B's reader raises as it reads the
    run's signals. With the signals not a list, as with them left out, no signal is read, so neither is it."""
    docs, _ = cal
    stored = copy.deepcopy(_by_id(docs)[CALIBRATED_PROBE])
    stored["attempts"] = "text"
    with pytest.raises(AttributeError, match="'str' object has no attribute 'get'") as info:
        compare.observe(stored, rule, now=NOW)
    assert Path(str(info.traceback[-1].path)).parts[-2:] == ("belief", "design.py")
    read = {}
    for shape in ("5", "null", "absent"):
        o = compare.observe(_shaped([stored], {**UNREADABLE, **READABLE}[shape], CALIBRATED_PROBE)[0], rule, now=NOW)
        read[shape] = (o.outcome, o.usd, o.notes)
    unread = (None, None, ("unreadable:signals",))
    assert read == {"5": unread, "null": unread, "absent": (None, None, ())}


def test_an_error_in_the_shape_test_is_not_caught(cal, monkeypatch):
    from loopmath.belief import outcome

    docs, _ = cal
    real = outcome.unreadable_signals
    fault = TypeError("an error in the shape test")

    def broken(doc, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_observe_ocp":
            raise fault
        return real(doc, *args, **kwargs)

    monkeypatch.setattr(outcome, "unreadable_signals", broken)
    probe = _by_id(docs)[CALIBRATED_PROBE]
    for read in (lambda: compare.observe(probe, RULE, now=NOW), lambda: compare.observe(probe, compare.ACCEPT, now=NOW),
                 lambda: compare.paired(*_by_arm(docs), rule=RULE, now=NOW)):
        with pytest.raises(TypeError) as info:
            read()
        assert info.value is fault


def test_the_command_reads_a_stored_run_whose_signals_are_not_a_list(cal, monkeypatch, tmp_path, capsys):
    """`check-change from-runs` on a store holding one run whose `run.signals` is not a list: under each rule it
    exits 0 with the comparison it gives with the signals left out, plus the one line."""
    from tests.compare.test_check_change_cli import lm

    docs, _ = cal
    monkeypatch.setenv("LOOPMATH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOOPMATH_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "home").mkdir()
    src = tmp_path / "ocp"
    src.mkdir()
    for d in docs:
        (src / f"{_id(d)}.ocp.json").write_text(json.dumps(d), encoding="utf-8")
    code, _, err = lm(capsys, "run", "import", str(src), "--finish", "--no-fit", "--allow-simulated", "--json")
    assert code == 0, err
    stored = tmp_path / "home" / "runs" / f"{CALIBRATED_PROBE}.ocp.json"
    doc = json.loads(stored.read_text(encoding="utf-8"))
    arms = ("check-change", "from-runs", "--base-label", "arm=base", "--cand-label", "arm=drop-critic", "--json")
    seen = {}
    for shape in ["absent"] + list(UNREADABLE):
        stored.write_text(json.dumps(_shaped([doc], {**UNREADABLE, **READABLE}[shape], CALIBRATED_PROBE)[0]),
                          encoding="utf-8")
        for rule in (RULE, "accept"):
            code, out, err = lm(capsys, *arms, "--rule", rule)
            assert code == 0, (shape, rule, err)
            seen[shape, rule] = out
    for (shape, rule), out in seen.items():
        reasons = out["comparison"]["reasons"]
        assert [r for r in reasons if r.endswith(NO_SIGNALS)] == ([] if shape == "absent" else
                                                                   [f"1 candidate run {NO_SIGNALS}"])
        unnamed = {**out, "comparison": {**out["comparison"],
                                         "reasons": [r for r in reasons if not r.endswith(NO_SIGNALS)]}}
        assert unnamed == seen["absent", rule], (shape, rule)
