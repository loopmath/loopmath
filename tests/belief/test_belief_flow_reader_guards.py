"""Reader guards on the belief readers. A stored rule's score target no finite float holds, or a `window_days` no
timedelta holds, is refused by name (`Unusable`, a ValueError), so the fit drops the run by name and receipts and
views read the default rule. A rule built in code with such a window reads no outcome, with a reason, and a score
value no float holds is no measured value, with a reason. These reads leak no arithmetic error, and coding runs
beside such a run keep their rows."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
from pathlib import Path

import pytest

import simdata
from loopmath.belief import fit as belief_fit
from loopmath.belief.design import Unusable
from loopmath.belief.outcome import parse_ts
from loopmath.belief.rules import evidence, resolve_rule, resolve_rule_or_default
from loopmath.flows import judges
from loopmath.types import DEFAULT_RULE, AcceptanceRule, ScoreTarget

NOW = parse_ts("2026-11-03T12:00:00-08:00")
TARGET = "score target not a finite number"
WINDOW = "window_days past 999999999"
_spec = importlib.util.spec_from_file_location(
    "flow_judges_helpers", Path(__file__).resolve().parents[1] / "flows" / "test_flow_judges.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
QUALITY = AcceptanceRule(name="quality>=0.5", definition="quality at least 0.5", requires=(),
                         score=ScoreTarget(name="quality", target=0.5, better="higher", scale="fraction"))


def _coding(n=40, seed=37):
    docs, _ = simdata.simulate(n, seed=seed, source="live", prefix="run_g")
    return docs


def _run(doc, run_id, rule=None, signals=()):
    """A copy of `doc` as `run_id`, its tests passing, with `rule` stored and `signals` added."""
    doc = copy.deepcopy(doc)
    doc["run"]["id"] = run_id
    for s in doc["run"]["signals"]:
        if s.get("kind") == "verdict" and s.get("name") == "tests":
            s["value"] = "pass"
    if rule is not None:
        doc["run"]["acceptance_rule"] = {"name": "tests", "definition": "the task's tests pass", "requires": ["tests"],
                                         **rule}
    doc["run"]["signals"].extend(copy.deepcopy(list(signals)))
    return doc


def _beside(coding, extra):
    """`collect_rows` with `extra` among the coding runs: the coding runs' rows equal theirs alone on every head.
    Returns the mixed result."""
    alone = belief_fit.collect_rows(iter(coding), now=NOW)
    mixed = belief_fit.collect_rows(iter(coding[:20] + extra + coding[20:]), now=NOW)
    names = {d["run"]["id"] for d in coding}
    assert alone[0].keys() <= mixed[0].keys()
    for head, a in alone[0].items():
        m = mixed[0][head]
        keep = [i for i, r in enumerate(m.runs) if r in names]
        for part in ("runs", "rows", "y", "w", "sources", "types"):
            assert [getattr(m, part)[i] for i in keep] == getattr(a, part), (head, part)
    return mixed


def _admitted(mixed, run_id):
    return any(run_id in h.runs for h in mixed[0].values())


def _event(name, at):
    return {"id": f"sig_{name}", "kind": "event", "name": name, "observed_at": at, "tier": "reported",
            "source": {"kind": "user"}}


def _score(k, value, at, scale="fraction"):
    return {"id": f"sig_v{k}", "kind": "score", "name": "quality", "value": value, "better": "higher",
            "scale": scale, "observed_at": at, "tier": "reported", "source": {"kind": "harness"}}


@pytest.mark.parametrize("target", [10 ** 400, -(10 ** 400), float("inf"), float("nan")])
def test_a_stored_target_no_finite_float_holds_is_refused_by_name(target):
    coding = _coding()
    score = {"name": "quality", "better": "higher", "scale": "linear"}
    doc = _run(coding[0], "run_g_target", {"name": "quality-rule", "requires": [], "score": {**score, "target": target}})
    with pytest.raises(Unusable, match=TARGET):
        resolve_rule(doc)
    assert resolve_rule_or_default(doc) == DEFAULT_RULE  # receipts and views: the default for a malformed rule
    mixed = _beside(coding, [doc])
    assert mixed[2][TARGET] == 1 and not _admitted(mixed, "run_g_target")
    near = _run(coding[0], "run_g_near", {"name": "quality-rule", "requires": [], "score": {**score, "target": 2 ** 53}})
    assert resolve_rule(near).score.target == 2.0 ** 53


@pytest.mark.parametrize("days", [10 ** 9, -(10 ** 9), 10 ** 400, float("inf")])
def test_a_stored_window_no_timedelta_holds_is_refused_by_name(days):
    coding = _coding()
    doc = _run(coding[0], "run_g_window", {"window_days": days})
    with pytest.raises(Unusable, match=WINDOW):
        resolve_rule(doc)
    assert resolve_rule_or_default(doc) == DEFAULT_RULE
    mixed = _beside(coding, [doc])
    assert mixed[2][WINDOW] == 1 and not _admitted(mixed, "run_g_window")


def test_window_days_0_and_999999999_stay_usable():
    """0 reads as 14 days, so a revert 30 days after the run's end is not late; 999999999 days reads, so it is."""
    coding = _coding()
    late = [_event("revert", "2026-10-20T11:00:00-07:00")]
    zero = _run(coding[0], "run_g_w0", {"window_days": 0}, late)
    wide = _run(coding[0], "run_g_wmax", {"window_days": 999999999}, late)
    assert resolve_rule(zero).window_days == 14 and resolve_rule(wide).window_days == 999999999
    ev0, evw = (evidence(d, resolve_rule(d), now=NOW) for d in (zero, wide))
    assert ev0.z == 1.0 and "late:revert" not in ev0.reasons
    assert evw.z == 0.0 and "late:revert" in evw.reasons
    mixed = _beside(coding, [zero, wide])
    assert WINDOW not in mixed[2] and _admitted(mixed, "run_g_w0") and _admitted(mixed, "run_g_wmax")


@pytest.mark.parametrize("days", [10 ** 9, 10 ** 400, float("inf"), float("nan")])
def test_a_window_built_in_code_no_timedelta_holds_reads_no_outcome(days):
    """The outcome read itself (`outcome_evidence`'s window): a rule built in code, or from a stored record's
    dict, reads no outcome with the reason `unusable:window_days`, and raises nothing."""
    doc = _run(_coding(3)[0], "run_g_code")
    for rule in (dataclasses.replace(DEFAULT_RULE, window_days=days),
                 AcceptanceRule.from_dict({**DEFAULT_RULE.to_dict(), "window_days": days})):
        ev = evidence(doc, rule, now=NOW)
        assert ev.z is None and "unusable:window_days" in ev.reasons
    assert evidence(doc, dataclasses.replace(DEFAULT_RULE, window_days=999999999), now=NOW).z == 1.0


@pytest.mark.parametrize("scale", ["linear", "fraction"])
def test_a_score_value_no_float_holds_is_no_measured_value(scale):
    """A score of 2**1024 is skipped as no value, with the reason `overflow:<name>`: the rule reads the last value
    a float holds, or `missing:` with none, the raw scores (the final loop, under any rule) leave it out, and the
    run is read beside coding runs without raising. Only an int can be past the float range."""
    coding = _coding()
    older = _score(0, 0.7, "2026-09-20T11:01:00-07:00", scale)
    huge = _score(1, 2 ** 1024, "2026-09-20T11:05:00-07:00", scale)
    some = _run(coding[0], "run_g_some", signals=[older, huge])
    ev = evidence(some, QUALITY, now=NOW)
    assert ev.scores["quality"] == 0.7 and ev.z == 1.0 and "overflow:quality" in ev.reasons
    only = _run(coding[0], "run_g_only", signals=[huge])
    ev = evidence(only, QUALITY, now=NOW)
    assert ev.z is None and {"missing:quality", "overflow:quality"} <= set(ev.reasons) and "quality" not in ev.scores
    raw = evidence(only, resolve_rule(only), now=NOW)  # the stored tests rule: only the raw loop reads the score
    assert raw.z == 1.0 and "quality" not in raw.scores and "overflow:quality" in raw.reasons
    plain = _run(coding[0], "run_g_plain", signals=[older])
    assert not [r for r in evidence(plain, QUALITY, now=NOW).reasons if r.startswith("overflow:")]
    mixed = _beside(coding, [some, only])
    assert _admitted(mixed, "run_g_some") and _admitted(mixed, "run_g_only")


def test_a_flow_judge_score_no_float_holds_reaches_no_reader(tmp_path):
    """A flow run whose judges' linear scores are all 2**1024 and the newest scores: the judges skip them as no
    reading, and the final raw loop, which the judges' filter does not guard, skips them too, with
    `overflow:quality`, so `evidence` and `collect_rows` raise nothing."""
    cal = H.calibrated_runs(tmp_path / "lit-brief")
    doc = copy.deepcopy(cal[0])
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"]["kind"] == "judge":
            s["scale"], s["value"], s["ext"] = "linear", 2 ** 1024, {judges.JUDGE_EXT: {"range": [0, 10]}}
            s["observed_at"] = "2026-11-01T12:00:00-07:00"
    ev = evidence(doc, resolve_rule(doc), now=NOW)
    assert "overflow:quality" in ev.reasons
    belief_fit.collect_rows(iter(cal[1:] + [doc]), now=NOW)
