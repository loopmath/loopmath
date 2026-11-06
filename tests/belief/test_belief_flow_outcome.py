"""The outcome reader and the belief fit on flow runs: a flow run's rule score is its unpooled estimate
under the current judges fit (`judges.current_quality`), not the pooled combined quality, a raw judge score is
never read, runs without `run.flow` are read as before,
calibration sees only the runs the fit admits, a judge score with no judge named, outside its range, not finite
as a float or not a number is no reading and is never read raw, the belief fit counts each skipped judge score by
its reason, a range no float reads makes no reader raise, and a run no reader can walk (a document or `run` that
is not an object, signals that are not a list of objects) is unreadable, counted once by name, and never stops
another run's finish or the report, and the review readers pass over a signal that is not an object."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import simdata
from loopmath.belief import fit as belief_fit
from loopmath.belief.outcome import _last, parse_ts, signal_list, unreadable_signals
from loopmath.belief.rules import REVIEW_RULE, evidence, resolve_rule, review_only
from loopmath.belief.state import transform
from loopmath.flows import judges
from loopmath.ocp.conformance import validate_doc
from loopmath.ocp.emit import validate_strict
from loopmath.store import Store
from loopmath.store import finish as store_finish
from loopmath.store.report import report_payload
from loopmath.store.runs import run_cost
from loopmath.types import AcceptanceRule, ScoreTarget
from tests.store.store_helpers import fake_settle

NOW = parse_ts("2026-11-03T12:00:00-08:00")
_spec = importlib.util.spec_from_file_location(
    "flow_judges_helpers", Path(__file__).resolve().parents[1] / "flows" / "test_flow_judges.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)
QUALITY = AcceptanceRule(name="quality>=0.5", definition="quality at least 0.5", requires=(),
                         score=ScoreTarget(name="quality", target=0.5, better="higher", scale="fraction"))


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    return H.calibrated_runs(tmp_path_factory.mktemp("cal") / "lit-brief")


def _coding(n=60, seed=31):
    """Coding runs (no `run.flow`) with three `quality` scores from judges at three times; the newest is 0.4."""
    docs, _ = simdata.simulate(n, seed=seed, source="live", prefix="run_c")
    for i, d in enumerate(docs):
        sig = d["run"].setdefault("signals", [])
        for k, at in enumerate(("2026-09-20T11:00:00-07:00", "2026-09-20T11:05:00-07:00", "2026-09-20T11:02:00-07:00")):
            sig.append({"id": f"sig_q{k}", "kind": "score", "name": "quality", "value": 0.3 + 0.1 * k + 0.001 * i,
                        "better": "higher", "scale": "fraction", "observed_at": at, "tier": "reported",
                        "source": {"kind": "judge", "ref": f"j{k}"}})
    return docs


def _judge_values(doc):
    return [s["value"] for s in doc["run"]["signals"] if s.get("kind") == "score"
            and (s.get("source") or {}).get("kind") in ("judge", "user")]


def _score_rows(out):
    hr = out[0].get("score:quality")
    return {} if hr is None else dict(zip(hr.runs, hr.y))


# The reader


def test_run_without_flow_ignores_quality():
    doc = _coding(3)[0]
    plain = evidence(doc, QUALITY, now=NOW)
    assert plain.scores["quality"] == 0.4  # the newest of the three
    for quality in ((0.1, 0.01), (None, None)):
        assert evidence(doc, QUALITY, now=NOW, quality=quality) == plain


def test_flow_run_reads_its_combined_quality(cal):
    doc = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    rule = resolve_rule(doc)
    assert rule.score.name == "quality" and rule.requires == ("claims-check",)
    claims = _last([s for s in doc["run"]["signals"] if s.get("kind") == "verdict"
                    and s.get("name") == "claims-check"])["value"]
    high = evidence(doc, rule, now=NOW, quality=(0.9, 0.01))
    assert high.scores["quality"] == 0.9 and high.tier == "reported"
    assert high.z == (1.0 if claims == "pass" else 0.0)
    low = evidence(doc, rule, now=NOW, quality=(0.5, 0.01))
    assert low.z == 0.0 and "below_target:quality" in low.reasons
    # no keyword: the current fit of the documents the caller has, here the run alone
    own = judges.current_quality([doc])[H.CALIBRATED_PROBE]
    alone = evidence(doc, rule, now=NOW)
    assert alone == evidence(doc, rule, now=NOW, quality=own) and alone.scores["quality"] == own[0]
    assert not set(_judge_values(doc)) & set(alone.scores.values())


def test_no_combined_value_reads_no_score(cal):
    """A judged flow run with no value from the judges fit has no score and a named reason; its raw judge scores are never
    read, whether it was given (None, None) or has no map entry."""
    doc = cal[0]
    rule = resolve_rule(doc)
    ev = evidence(doc, rule, now=NOW, quality=(None, None))
    assert "quality" not in ev.scores and "unfitted:quality" in ev.reasons
    assert ev.z in (None, 0.0) and not set(_judge_values(doc)) & set(ev.scores.values())
    if ev.z is None:
        assert "below_target:quality" not in ev.reasons


def test_flow_run_without_judge_scores_reads_as_before(cal):
    doc = copy.deepcopy(cal[0])
    doc["run"]["signals"] = [s for s in doc["run"]["signals"] if s.get("kind") != "score"]
    doc["run"]["signals"].append({"id": "sig_h", "kind": "score", "name": "quality", "value": 0.83,
                                  "better": "higher", "scale": "fraction", "tier": "verified",
                                  "observed_at": "2026-01-05T10:00:00+00:00", "source": {"kind": "harness"}})
    bare = copy.deepcopy(doc)
    del bare["run"]["flow"]
    rule = resolve_rule(doc)
    assert evidence(doc, rule, now=NOW) == evidence(bare, rule, now=NOW)
    assert evidence(doc, rule, now=NOW).scores["quality"] == 0.83


# The belief fit


def test_collect_rows_reads_the_corpus_fit(cal):
    """Calibrated integration: each flow run's score row is its unpooled estimate under the corpus fit, which
    for the probe run differs from its own value by more than 0.01; the score head describes a fraction."""
    out = belief_fit.collect_rows(iter(cal), now=NOW)
    rows, every = _score_rows(out), judges.current_quality(cal)
    assert set(rows) == set(every) == {d["run"]["id"] for d in cal}
    assert all(rows[k] == transform(every[k][0], "fraction") for k in rows)
    assert out[1]["quality"] == {"scale": "fraction", "better": "higher", "unit": None}
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    assert abs(every[H.CALIBRATED_PROBE][0] - judges.current_quality([probe])[H.CALIBRATED_PROBE][0]) > 0.01
    success = out[0]["success"]
    for run_id, z in zip(success.runs, success.y):
        doc = next(d for d in cal if d["run"]["id"] == run_id)
        assert z == evidence(doc, resolve_rule(doc), now=NOW, quality=every[run_id]).z


def test_coding_rows_unchanged_beside_flow_runs(cal):
    coding = _coding()
    alone = belief_fit.collect_rows(iter(coding), now=NOW)
    mixed = belief_fit.collect_rows(iter(coding[:30] + cal + coding[30:]), now=NOW)
    names = {d["run"]["id"] for d in coding}
    for head in ("success", "score:quality"):
        a, m = alone[0][head], mixed[0][head]
        keep = [i for i, r in enumerate(m.runs) if r in names]
        assert [m.runs[i] for i in keep] == a.runs
        assert [m.y[i] for i in keep] == a.y and [m.w[i] for i in keep] == a.w
    flow = {k: v for k, v in _score_rows(mixed).items() if k not in names}
    every = judges.current_quality(coding + cal)
    assert flow == {k: transform(v[0], "fraction") for k, v in every.items()}


def test_configured_q_seed_weights_cover_flow_runs(cal):
    out = belief_fit.collect_rows(iter(cal), now=NOW, q_by_tier={"reported": 0.9})
    success = out[0]["success"]
    assert len(success.seed_w) == len(success.runs) > 0 and set(success.seed_w) == {0.95}
    assert set(success.w) == {0.9}


def _wild(doc, run_id=None, person=True):
    """A copy with every judge score at 0.01 (and a person score added), under `run_id` if given."""
    doc = copy.deepcopy(doc)
    if run_id is not None:
        doc["run"]["id"] = run_id
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score":
            s["value"] = 0.01
    if person and not any((s.get("source") or {}).get("kind") == "user" for s in doc["run"]["signals"]):
        doc["run"]["signals"].append({"id": "sig_wild_person", "kind": "score", "name": "quality", "value": 0.01,
                                      "better": "higher", "scale": "fraction", "tier": "reported",
                                      "observed_at": "2026-01-05T10:00:00+00:00",
                                      "source": {"kind": "user", "ref": "person"}})
    return doc


def test_admission_decides_what_calibrates(cal):
    """Runs the belief fit leaves out cannot move the retained flow runs' values or the person count: a shipped
    copy of a stored run, an origin in `without`, a source in `without`, an unusable document and a repeated run
    id, each with very different judge values. A one-pass iterator works."""
    stored = [("user", d) for d in cal]
    base = judges.current_quality(stored)
    flow = H.load(H.EXAMPLE).flow

    def n_person(pairs, without=()):
        return judges.fit(flow, [d for _, d, _ in belief_fit.admitted(pairs, without=without)])["n_person"]

    swept = _wild(cal[1], "run_swept")
    swept["run"]["task"]["source"] = {"kind": "sweep"}
    unusable = _wild(cal[2], "run_unusable")
    del unusable["run"]["configuration"]
    cases = {
        "shipped copy": (stored + [("shipped-x", _wild(cal[0]))], ()),
        "origin in without": (stored + [("sweep", _wild(cal[3], "run_other"))], ("sweep",)),
        "source in without": (stored + [(None, swept)], ("sweep",)),
        "unusable": (stored + [(None, unusable)], ()),
        "repeated id": (stored + [(None, _wild(cal[4]))], ()),
    }
    for name, (pairs, without) in cases.items():
        assert judges.current_quality(iter(pairs), without=without) == base, name
        assert n_person(pairs, without) == 8, name
        out = belief_fit.collect_rows(iter(pairs), now=NOW, without=without)
        assert _score_rows(out) == {k: transform(v[0], "fraction") for k, v in base.items()}, name
    # the same extra runs, admitted, do move the values
    assert judges.current_quality(stored + [(None, _wild(cal[3], "run_other"))]) != base


def test_unranged_linear_reading_has_no_row(cal):
    doc = copy.deepcopy(cal[0])
    doc["run"]["id"] = "run_unranged"
    doc["run"]["signals"] = [s for s in doc["run"]["signals"] if not (s.get("kind") == "score"
                                                                      and s["source"]["ref"] != "j-astra")]
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score":
            s["value"], s["scale"] = 3.5, "linear"
    out = belief_fit.collect_rows(iter(cal + [doc]), now=NOW)
    assert "run_unranged" not in _score_rows(out) and out[2][judges.UNRANGED] == 1
    assert "run_unranged" not in judges.current_quality(cal + [doc])
    ev = evidence(doc, resolve_rule(doc), now=NOW)
    assert "quality" not in ev.scores and "unranged:j-astra" in ev.reasons
    if ev.z is None:
        assert "run_unranged" not in out[0]["success"].runs


def _crowded(docs, extra):
    """Copies of `docs` with `extra` more model judges, j03 and up, scoring every run beside the corpus's three."""
    out = copy.deepcopy(docs)
    for doc in out:
        sig = next(s for s in doc["run"]["signals"] if s.get("kind") == "score" and s["source"]["ref"] == "j-claude")
        doc["run"]["signals"] += [dict(sig, id=f"{sig['id']}-j{k:02d}", value=min(1.0, sig["value"] + 0.01 * k),
                                       source={**sig["source"], "ref": f"j{k:02d}"}) for k in range(3, 3 + extra)]
    return out


def test_too_many_judges_read_no_score(cal):
    """Above `judges.MAX_SCORED_JUDGES` (13) judges with readings the fit is refused: its runs read no score, with
    the reason `too-many-judges:<n>`, and never a raw judge value. 13 fit. The counts are fixed, not taken from
    the limit."""
    docs, reason = _crowded(cal, 11), "too-many-judges:14"  # with j-claude, j-astra and the person
    quality = judges.current_quality(docs)
    assert set(quality) == {d["run"]["id"] for d in docs}
    assert all(isinstance(v, judges.Unfitted) and v == (None, None) and v.reason == reason
               for v in quality.values())
    out = belief_fit.collect_rows(iter(docs), now=NOW)
    assert _score_rows(out) == {} and out[2][judges.REFUSED] == len(docs)
    person = [d for d in docs if any(s.get("kind") == "score" and s["source"]["kind"] == "user"
                                     for s in d["run"]["signals"])]
    assert len(person) == 8  # each alone is scored by 14 judges too
    for doc in person:
        rule = resolve_rule(doc)
        for ev in (evidence(doc, rule, now=NOW), evidence(doc, rule, now=NOW, quality=quality[doc["run"]["id"]])):
            assert "quality" not in ev.scores and reason in ev.reasons and "unfitted:quality" not in ev.reasons
            assert ev.z in (None, 0.0) and not set(_judge_values(doc)) & set(ev.scores.values())
    fewer = _crowded(cal, 10)
    quality = judges.current_quality(fewer)
    assert set(quality) == {d["run"]["id"] for d in fewer}
    assert all(type(v) is tuple and v[0] is not None for v in quality.values())
    out = belief_fit.collect_rows(iter(fewer), now=NOW)
    assert set(_score_rows(out)) == set(quality) and judges.REFUSED not in out[2]


# Readings no reader may take raw or fail on


def _anonymous(doc, refs=None, ref="missing"):
    """A copy whose judge and person scores (of the judges in `refs`, or all) have `ref` as `source.ref`, or no
    `ref` at all when it is "missing"."""
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"].get("kind") in ("judge", "user") \
                and (refs is None or s["source"].get("ref") in refs):
            if ref == "missing":
                del s["source"]["ref"]
            else:
                s["source"]["ref"] = ref
    return doc


@pytest.mark.parametrize("ref", ["missing", None, 7])
def test_unattributed_scores_read_no_score(cal, ref):
    """A judge's or person's score with no string `source.ref` counts as judged and is never read raw: a run with
    only such scores has no score and the reason `unattributed:<signal id>`, in `evidence`, `current_quality`
    and the belief fit, and the other runs' values are those of the fit without the run."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    anon = _anonymous(probe, ref=ref)
    ids = [s["id"] for s in anon["run"]["signals"] if s.get("kind") == "score"]
    assert ids and judges.judged(anon, "quality") == ([], [], ids, [], [], ())
    assert judges.current_quality(others + [anon]) == judges.current_quality(others)
    rule = resolve_rule(anon)
    for quality in (None, (None, None)):
        ev = evidence(anon, rule, now=NOW, quality=quality)
        assert "quality" not in ev.scores and "unfitted:quality" in ev.reasons
        assert [r for r in ev.reasons if r.startswith("unattributed:")] == [f"unattributed:{i}" for i in ids]
        assert ev.z in (None, 0.0) and not set(_judge_values(anon)) & set(ev.scores.values())
    out = belief_fit.collect_rows(iter(others + [anon]), now=NOW)
    assert _score_rows(out) == _score_rows(belief_fit.collect_rows(iter(others), now=NOW))
    assert H.CALIBRATED_PROBE not in _score_rows(out) and out[1]["quality"]["scale"] == "fraction"
    success = dict(zip(out[0]["success"].runs, out[0]["success"].y))
    assert success.get(H.CALIBRATED_PROBE) == evidence(anon, rule, now=NOW, quality=(None, None)).z


def test_an_unattributed_score_beside_readings_is_left_out(cal):
    """With j-astra's score unattributed, the run reads its other judges' value, as if that score were not there,
    and names the signal."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    part = _anonymous(probe, refs=("j-astra",))
    ids = [s["id"] for s in probe["run"]["signals"] if s.get("kind") == "score" and s["source"]["ref"] == "j-astra"]
    tiers, unranged, unattributed, outside, nonfinite, nonnumeric = judges.judged(part, "quality")
    assert ids and tiers and not unranged and unattributed == ids and not outside and not nonfinite and not nonnumeric
    want = judges.current_quality(others + [H._strip(probe, ("j-claude", "person"))])
    assert judges.current_quality(others + [part]) == want
    ev = evidence(part, resolve_rule(part), now=NOW, quality=want[H.CALIBRATED_PROBE])
    assert ev.scores["quality"] == want[H.CALIBRATED_PROBE][0] and "unfitted:quality" not in ev.reasons
    assert [r for r in ev.reasons if r.startswith("unattributed:")] == [f"unattributed:{i}" for i in ids]
    rows = _score_rows(belief_fit.collect_rows(iter(others + [part]), now=NOW))
    assert rows[H.CALIBRATED_PROBE] == transform(want[H.CALIBRATED_PROBE][0], "fraction")


def _linear(doc, span):
    """A copy whose model judges' scores are linear, with `span` as the range on each signal."""
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"]["kind"] == "judge":
            s["scale"], s["ext"] = "linear", {judges.JUDGE_EXT: {"range": span}}
    return doc


@pytest.mark.parametrize("span", [[0, 2 ** 1024], [2 ** 60, 2 ** 60 + 1], [0, 2 ** 53 + 1]])
def test_unusable_ranges_beside_coding_runs(cal, span, tmp_path):
    """Flow runs whose model judges' linear scores carry a range no float reads, beside coding runs: no reader
    raises, each such reading is skipped as unranged, the coding runs' rows are unchanged, and the whole belief
    fit runs."""
    coding, bad = _coding(), [_linear(d, span) for d in cal]
    for doc in bad:
        assert judges.judged(doc, "quality")[1] == ["j-astra", "j-claude"]
        assert {"unranged:j-astra", "unranged:j-claude"} <= set(evidence(doc, resolve_rule(doc), now=NOW).reasons)
    alone = belief_fit.collect_rows(iter(coding), now=NOW)
    mixed = belief_fit.collect_rows(iter(coding[:30] + bad + coding[30:]), now=NOW)
    assert mixed[2][judges.UNRANGED] == 2 * len(bad)
    names = {d["run"]["id"] for d in coding}
    for head, a in alone[0].items():
        m = mixed[0][head]
        keep = [i for i, r in enumerate(m.runs) if r in names]
        for part in ("runs", "rows", "y", "w", "sources", "types"):
            assert [getattr(m, part)[i] for i in keep] == getattr(a, part), (head, part)
    flow = {k: v for k, v in _score_rows(mixed).items() if k not in names}
    assert flow == {k: transform(v[0], "fraction") for k, v in judges.current_quality(coding + bad).items()}
    assert set(flow) == {d["run"]["id"] for d in bad if any(s.get("kind") == "score" and s["source"]["kind"] == "user"
                                                             for s in d["run"]["signals"])}
    path = belief_fit.fit(tmp_path / "home", docs=coding[:30] + bad + coding[30:], no_prior=True, now=NOW,
                          bundle_dir=tmp_path / "no-bundle")
    assert (path / "meta.json").is_file()


# The unpooled estimate, every statistical reader's value


def _valued(doc, values, harness=None):
    """A copy whose scores from each judge in `values` (ref to `(scale, value)`) hold that value on that scale: a
    linear one with the range [0, 10] on its signal, None for no scale. With `harness`, a newer harness score under
    the name."""
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"].get("ref") in values:
            scale, s["value"] = values[s["source"]["ref"]]
            s.pop("scale", None)
            s.pop("ext", None)
            if scale is not None:
                s["scale"] = scale
            if scale == "linear":
                s["ext"] = {judges.JUDGE_EXT: {"range": [0, 10]}}
    if harness is not None:
        doc["run"]["signals"].append({"id": "sig_harness", "kind": "score", "name": "quality", "value": harness,
                                      "scale": "fraction", "better": "higher", "tier": "reported",
                                      "observed_at": "2026-11-01T12:00:00-07:00", "source": {"kind": "harness"}})
    return doc


OUTSIDE = [("linear", 10.5), ("linear", -1), ("linear", 1e308), ("fraction", 1.5), ("fraction", -1e308),
           ("log", 7.0), (None, 2.0), ("fraction", -0.5)]


@pytest.mark.parametrize("scale, value", OUTSIDE)
def test_a_reading_outside_its_range_is_left_out(cal, scale, value):
    """j-claude's scores outside their range (linear: outside [0, 10] on the signal; any other: outside [0, 1]):
    the run reads its other judges' value, as if j-claude's scores were not there, names `outside-range:j-claude`,
    and no reader raises."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    doc = _valued(probe, {"j-claude": (scale, value)})
    got = judges.judged(doc, "quality")
    assert got.outside_range == ["j-claude"] and got.reasons == ["outside-range:j-claude"]
    assert len(got.usable) == len(judges.judged(probe, "quality").usable) - 1
    want = judges.current_quality(others + [H._strip(probe, ("j-astra", "person"))])
    assert judges.current_quality(others + [doc]) == want
    ev = evidence(doc, resolve_rule(doc), now=NOW, quality=want[H.CALIBRATED_PROBE])
    assert ev.scores["quality"] == want[H.CALIBRATED_PROBE][0] and "outside-range:j-claude" in ev.reasons
    rows = _score_rows(belief_fit.collect_rows(iter(others + [doc]), now=NOW))
    assert rows[H.CALIBRATED_PROBE] == transform(want[H.CALIBRATED_PROBE][0], "fraction")


def test_a_bound_is_in_range(cal):
    """A score on a bound of its range is a reading: 0 and 10 on [0, 10], 0 and 1 on any other scale."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    n = len(judges.judged(probe, "quality").usable)
    for scale, value, want in (("linear", 0, 0.0), ("linear", 10, 1.0), ("fraction", 0, 0.0), ("fraction", 1, 1.0),
                               ("log", 1, 1.0)):
        doc = _valued(probe, {"j-claude": (scale, value)})
        got = judges.judged(doc, "quality")
        assert not got.outside_range and len(got.usable) == n
        assert judges._readings(doc, judges.signal_judges([doc], "quality"))["j-claude"] == want


UNREAD = {
    "outside": ({"j-claude": ("linear", 1e308), "j-astra": ("fraction", 1.5), "person": ("log", -3.0)},
                ["outside-range:j-astra", "outside-range:j-claude", "outside-range:person"]),
    "nonfinite": ({"j-claude": ("linear", 2 ** 1024), "j-astra": ("fraction", float("inf")),
                   "person": ("fraction", float("nan"))},
                  ["nonfinite:j-astra", "nonfinite:j-claude", "nonfinite:person"]),
    "mixed": ({"j-claude": ("linear", 2 ** 1024), "j-astra": ("linear", 11), "person": ("fraction", float("-inf"))},
              ["outside-range:j-astra", "nonfinite:j-claude", "nonfinite:person"]),
    "nonnumeric": ({"j-claude": ("fraction", None), "j-astra": (None, True), "person": ("fraction", "0.8")},
                   ["nonnumeric:j-astra", "nonnumeric:j-claude", "nonnumeric:person"]),
    "mixed2": ({"j-claude": (None, [0.5]), "j-astra": ("linear", 2 ** 1024), "person": ("linear", 11)},
               ["outside-range:person", "nonfinite:j-astra", "nonnumeric:j-claude"]),
}


@pytest.mark.parametrize("case", sorted(UNREAD))
def test_judge_scores_none_of_which_reads_give_no_score(cal, case):
    """Every judge's and the person's scores outside their range, not finite as a float or not a number, beside a
    newer harness score under the name: the run counts as judged, so the harness score is never read raw. It has
    no score, names each judge, and is left out of `current_quality` and the score rows; no reader raises."""
    values, reasons = UNREAD[case]
    probe = next(d for d in cal if {j.id for j in judges.signal_judges([d], "quality")} == set(values))
    others = [d for d in cal if d is not probe]
    doc = _valued(probe, values, harness=0.9)
    got = judges.judged(doc, "quality")
    assert not got.usable and got.reasons == reasons
    assert judges.current_quality(others + [doc]) == judges.current_quality(others)
    rule = resolve_rule(doc)
    for quality in (None, (None, None)):
        ev = evidence(doc, rule, now=NOW, quality=quality)
        assert "quality" not in ev.scores and "unfitted:quality" in ev.reasons and ev.z in (None, 0.0)
        assert [r for r in ev.reasons if r.startswith(("outside-range:", "nonfinite:", "nonnumeric:"))] == reasons
    out = belief_fit.collect_rows(iter(others + [doc]), now=NOW)
    assert _score_rows(out) == _score_rows(belief_fit.collect_rows(iter(others), now=NOW))
    success = dict(zip(out[0]["success"].runs, out[0]["success"].y))
    assert success.get(probe["run"]["id"]) == evidence(doc, rule, now=NOW, quality=(None, None)).z


def test_a_flow_run_without_judge_scores_reads_a_raw_value_no_float_holds_as_none(cal):
    """A flow run with no judge score under the name reads its rule score raw, as a run without a flow does: a
    harness score of 2**1024 is no value (`missing:quality`, `overflow:quality`), and no reader raises."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    doc = _valued(H._strip(probe, ()), {}, harness=2 ** 1024)
    assert not any(judges.judged(doc, "quality"))
    ev = evidence(doc, resolve_rule(doc), now=NOW)
    assert "quality" not in ev.scores and {"missing:quality", "overflow:quality"} <= set(ev.reasons)
    belief_fit.collect_rows(iter(others + [doc]), now=NOW)


THREE = {"j-claude", "j-astra", "person"}  # the judges of the calibrated runs a person scored
NONNUMBERS = {"null": None, "true": True, "false": False, "text": "0.8", "list": [0.5], "object": {"value": 0.5}}
JUDGE_DROPS = (*judges.DROP_COUNTS.values(), judges.REFUSED)
OLDER = "2026-01-01T00:00:00+00:00"  # before every calibrated run's scores


def _three(cal, k=0):
    """The `k`th calibrated run scored by j-claude, j-astra and the person."""
    return [d for d in cal if {j.id for j in judges.signal_judges([d], "quality")} == THREE][k]


def _no_number(doc, shape, refs=None):
    """A copy whose judge and person scores (of the judges in `refs`, or all) hold `NONNUMBERS[shape]`, or no
    value at all when `shape` is "missing"."""
    doc = copy.deepcopy(doc)
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"].get("kind") in ("judge", "user") \
                and (refs is None or s["source"].get("ref") in refs):
            if shape == "missing":
                del s["value"]
            else:
                s["value"] = copy.deepcopy(NONNUMBERS[shape])
    return doc


def _again(doc, ref, suffix, **changes):
    """A copy with one more score from `ref` under the name: its first, with `changes` and the id suffixed."""
    doc = copy.deepcopy(doc)
    first = next(s for s in doc["run"]["signals"] if s.get("kind") == "score" and s["source"].get("ref") == ref)
    doc["run"]["signals"].append(dict(copy.deepcopy(first), id=first["id"] + suffix, **changes))
    return doc


def _unranged(doc, refs):
    """A copy whose scores from the judges in `refs` are linear 5 with no range on the signal."""
    doc = _valued(doc, {r: ("linear", 5) for r in refs})
    for s in doc["run"]["signals"]:
        if s.get("kind") == "score" and s["source"].get("ref") in refs:
            del s["ext"]
    return doc


def _judge_drops(out):
    """The belief fit's judge drop counts (`collect_rows`'s third value), each one written."""
    return {k: n for k, n in out[2].items() if k in JUDGE_DROPS}


@pytest.mark.parametrize("named", [True, False], ids=["named", "no-ref"])
@pytest.mark.parametrize("shape", sorted(NONNUMBERS) + ["missing"])
def test_a_judge_score_that_is_no_number_reads_no_score(cal, shape, named):
    """Every judge's and the person's score under the name holds no number (`NONNUMBERS`, or no value), j-claude's
    twice, beside a newer harness score of 0.9: the run counts as judged, so 0.9 is never read raw. With a string
    `source.ref` each judge is named once, as `nonnumeric:` only (a bool is not `nonfinite:`); with none each
    signal is `unattributed:`. The run has no score and is left out of `current_quality` and the score rows, and
    the belief fit counts it under that one name."""
    probe = _three(cal)
    others = [d for d in cal if d is not probe]
    doc = _no_number(_again(probe, "j-claude", "-older", observed_at=OLDER), shape)
    doc = _valued(doc if named else _anonymous(doc), {}, harness=0.9)
    ids = [s["id"] for s in doc["run"]["signals"] if s.get("kind") == "score" and s["source"]["kind"] != "harness"]
    assert len(ids) == 4
    if named:
        want = judges.Judged([], [], [], [], [], ("j-astra", "j-claude", "person"))
        reasons, drops = ["nonnumeric:j-astra", "nonnumeric:j-claude", "nonnumeric:person"], {judges.NONNUMERIC: 3}
    else:
        want = judges.Judged([], [], ids, [], [])
        reasons, drops = [f"unattributed:{i}" for i in ids], {judges.UNATTRIBUTED: 4}
    got = judges.judged(doc, "quality")
    assert got == want and got.reasons == reasons
    assert judges.current_quality(others + [doc]) == judges.current_quality(others)
    rule = resolve_rule(doc)
    for quality in (None, (None, None)):
        ev = evidence(doc, rule, now=NOW, quality=quality)
        assert "quality" not in ev.scores and "unfitted:quality" in ev.reasons and ev.z in (None, 0.0)
        assert [r for r in ev.reasons if r.startswith(("unattributed:", "nonfinite:", "nonnumeric:"))] == reasons
        assert 0.9 not in ev.scores.values()
    out = belief_fit.collect_rows(iter(others + [doc]), now=NOW)
    assert _score_rows(out) == _score_rows(belief_fit.collect_rows(iter(others), now=NOW))
    assert _judge_drops(out) == drops
    success = dict(zip(out[0]["success"].runs, out[0]["success"].y))
    assert success.get(probe["run"]["id"]) == evidence(doc, rule, now=NOW, quality=(None, None)).z


def test_null_at_the_declared_unmeasured_shape_is_judged(cal):
    """Every judge's and the person's score is null in the shape of a declared score not yet measured (no scale,
    tier asserted), beside a newer harness score of 0.9: the document conforms, strictly too, and the run counts
    as judged, names each judge as `nonnumeric:` and has no score, so 0.9 is never read."""
    probe = _three(cal)
    others = [d for d in cal if d is not probe]
    doc = _valued(probe, {}, harness=0.9)
    doc["run"]["signals"] = [{"id": s["id"], "kind": "score", "name": "quality", "value": None,
                              "observed_at": s["observed_at"], "tier": "asserted", "source": s["source"]}
                             if s.get("kind") == "score" and s["source"]["kind"] in ("judge", "user") else s
                             for s in doc["run"]["signals"]]
    assert validate_doc(doc) == [] and [f for f in validate_strict(doc) if f["level"] == "error"] == []
    reasons = ["nonnumeric:j-astra", "nonnumeric:j-claude", "nonnumeric:person"]
    got = judges.judged(doc, "quality")
    assert got == judges.Judged([], [], [], [], [], ("j-astra", "j-claude", "person")) and got.reasons == reasons
    assert judges.current_quality(others + [doc]) == judges.current_quality(others)
    ev = evidence(doc, resolve_rule(doc), now=NOW)
    assert "quality" not in ev.scores and "unfitted:quality" in ev.reasons and ev.z in (None, 0.0)
    assert [r for r in ev.reasons if r.startswith("nonnumeric:")] == reasons
    out = belief_fit.collect_rows(iter(others + [doc]), now=NOW)
    assert probe["run"]["id"] not in _score_rows(out) and _judge_drops(out) == {judges.NONNUMERIC: 3}


@pytest.mark.parametrize("kind", ["harness", "orchestrator"])
@pytest.mark.parametrize("value", [0.7, 2 ** 1024, None, True, "0.8"], ids=["0.7", "2**1024", "null", "true", "text"])
def test_a_harness_score_with_a_ref_is_never_a_judge_score(cal, kind, value):
    """A flow run whose only score under the name comes from a harness or an orchestrator with a string
    `source.ref` is not judged, whatever its value: no `unattributed:`, `nonfinite:` or `nonnumeric:` reason and
    no judge drop count. It reads the score raw, as a run without judge scores does: 0.7 reads 0.7, and no number
    reads `missing:quality` (2**1024 `overflow:quality` too)."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    doc = H._strip(probe, ())
    doc["run"]["signals"].append({"id": "sig_harness", "kind": "score", "name": "quality", "value": value,
                                  "scale": "fraction", "better": "higher", "tier": "reported",
                                  "observed_at": "2026-11-01T12:00:00-07:00", "source": {"kind": kind, "ref": "h-run"}})
    got = judges.judged(doc, "quality")
    assert got == judges.Judged([], [], [], [], []) and not any(got)
    ev = evidence(doc, resolve_rule(doc), now=NOW)
    assert not [r for r in ev.reasons if r.startswith(("unattributed:", "nonfinite:", "nonnumeric:"))]
    if value == 0.7:
        assert ev.scores["quality"] == 0.7
    else:
        assert "quality" not in ev.scores and "missing:quality" in ev.reasons
        assert ("overflow:quality" in ev.reasons) == (value == 2 ** 1024)
    assert _judge_drops(belief_fit.collect_rows(iter(others + [doc]), now=NOW)) == {}


@pytest.mark.parametrize("shape", ["null", "true", "text"])
def test_a_non_number_beside_readings_is_left_out(cal, shape):
    """j-claude's score holds no number, beside j-astra's reading and a newer harness score of 0.9: the run reads
    its value from j-astra alone, as if j-claude's score were not there, never 0.9, and names
    `nonnumeric:j-claude`."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    doc = _valued(_no_number(probe, shape, refs=("j-claude",)), {}, harness=0.9)
    got = judges.judged(doc, "quality")
    assert got.nonnumeric == ("j-claude",) and got.reasons == ["nonnumeric:j-claude"] and len(got.usable) == 1
    want = judges.current_quality(others + [H._strip(probe, ("j-astra",))])
    assert judges.current_quality(others + [doc]) == want
    ev = evidence(doc, resolve_rule(doc), now=NOW, quality=want[H.CALIBRATED_PROBE])
    assert ev.scores["quality"] == want[H.CALIBRATED_PROBE][0] != 0.9 and "nonnumeric:j-claude" in ev.reasons
    out = belief_fit.collect_rows(iter(others + [doc]), now=NOW)
    assert _score_rows(out)[H.CALIBRATED_PROBE] == transform(want[H.CALIBRATED_PROBE][0], "fraction")
    assert _judge_drops(out) == {judges.NONNUMERIC: 1}


def test_a_non_number_does_not_erase_a_judge_reading(cal):
    """j-claude gains a newer null score and an older text one beside its number: its reading is still that
    number, so the run reads the unchanged run's value, and it names `nonnumeric:j-claude` once."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    doc = _again(_again(probe, "j-claude", "-newer", value=None, observed_at="2026-10-01T00:00:00+00:00"),
                 "j-claude", "-older", value="0.8", observed_at=OLDER)
    got = judges.judged(doc, "quality")
    assert got.nonnumeric == ("j-claude",) and got.reasons == ["nonnumeric:j-claude"]
    assert got.usable == judges.judged(probe, "quality").usable == ["reported", "reported"]
    js = judges.signal_judges([probe], "quality")
    assert judges._readings(doc, js) == judges._readings(probe, js)
    want = judges.current_quality(others + [probe])
    assert judges.current_quality(others + [doc]) == want
    ev = evidence(doc, resolve_rule(doc), now=NOW, quality=want[H.CALIBRATED_PROBE])
    assert ev.scores["quality"] == want[H.CALIBRATED_PROBE][0] and "nonnumeric:j-claude" in ev.reasons
    out = belief_fit.collect_rows(iter(others + [doc]), now=NOW)
    assert _score_rows(out) == _score_rows(belief_fit.collect_rows(iter(others + [probe]), now=NOW))
    assert _judge_drops(out) == {judges.NONNUMERIC: 1}


def _drop_runs(cal, kind):
    """The flow runs of the drop count item `kind`, and the judge drop counts the belief fit gives for them,
    written out: one per judge, or per signal when unattributed."""
    a, b = _three(cal, 0), _three(cal, 1)
    return {
        "unranged": ([_unranged(a, ("j-claude", "j-astra"))], {judges.UNRANGED: 2}),
        "unattributed": ([_anonymous(a, refs=("j-claude", "j-astra"))], {judges.UNATTRIBUTED: 2}),
        "outside-range": ([_valued(a, {"j-claude": ("linear", 50), "j-astra": ("fraction", 1.5)})],
                          {judges.OUTSIDE_RANGE: 2}),
        "nonfinite": ([_valued(a, {"j-claude": ("linear", 2 ** 1024), "j-astra": ("fraction", float("inf"))})],
                      {judges.NONFINITE: 2}),
        "nonnumeric": ([_no_number(_no_number(a, "null", refs=("j-claude",)), "text", refs=("j-astra",))],
                       {judges.NONNUMERIC: 2}),
        # run a: j-claude's two scores a list, j-astra 2**1024, the person linear 11; run b: j-claude unranged,
        # j-astra's two scores and the person's with no ref
        "mixed": ([_valued(_no_number(_again(a, "j-claude", "-older", observed_at=OLDER), "list", refs=("j-claude",)),
                           {"j-astra": ("linear", 2 ** 1024), "person": ("linear", 11)}),
                   _anonymous(_again(_unranged(b, ("j-claude",)), "j-astra", "-older", observed_at=OLDER),
                              refs=("j-astra", "person"))],
                  {judges.UNRANGED: 1, judges.UNATTRIBUTED: 3, judges.OUTSIDE_RANGE: 1, judges.NONFINITE: 1,
                   judges.NONNUMERIC: 1}),
    }[kind]


@pytest.mark.parametrize("kind", ["unranged", "unattributed", "outside-range", "nonfinite", "nonnumeric", "mixed",
                                  "refused"])
def test_the_belief_fit_counts_each_judge_drop_by_name(cal, kind):
    """Beside coding runs and the other calibrated runs, flow runs whose judge scores the reader skips: the belief
    fit counts each under its reason's count (`judges.DROP_COUNTS`), one per judge, or per signal when
    unattributed, and the runs of a fit refused for too many judges under `judges.REFUSED` beside them. No other
    judge count is written, not even a zero."""
    if kind == "refused":
        flow = _crowded(cal, 11)
        flow[0] = _no_number(flow[0], "null", refs=("j05",))
        want = {judges.REFUSED: 20, judges.NONNUMERIC: 1}
    else:
        runs, want = _drop_runs(cal, kind)
        ids = {d["run"]["id"] for d in runs}
        flow = [d for d in cal if d["run"]["id"] not in ids] + runs
    assert _judge_drops(belief_fit.collect_rows(iter(_coding(20) + flow), now=NOW)) == want


def test_the_whole_belief_fit_runs_beside_unread_judge_scores(cal, tmp_path):
    """Beside coding runs, a flow run with one linear judge score of 1e308 on [0, 10], and one whose judges' scores
    are all 2**1024 with no person score: the whole belief fit runs."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    others = [d for d in cal if d is not probe]
    huge = _valued(probe, {"j-claude": ("linear", 1e308)})
    gone = H._strip(_valued(probe, {"j-claude": ("linear", 2 ** 1024), "j-astra": ("linear", 2 ** 1024)}),
                    ("j-claude", "j-astra"))
    for k, doc in enumerate((huge, gone)):
        path = belief_fit.fit(tmp_path / f"home{k}", docs=_coding(20) + others + [doc], no_prior=True, now=NOW,
                              bundle_dir=tmp_path / "no-bundle")
        assert (path / "meta.json").is_file()


def test_unpooled_quality_equals_current_quality(cal):
    """With the fit the caller holds (`judges.fit(flow, docs)`), `unpooled_quality` gives `current_quality`'s value
    for every run, so a caller never fits twice. Off the fit it is the plug-in formula, near a refit's value; no
    reading and a refused fit read nothing."""
    flow = H.load(H.EXAMPLE).flow
    fit, every = judges.fit(flow, cal), judges.current_quality(cal)
    assert all(judges.unpooled_quality(flow, d, fit) == every[d["run"]["id"]] for d in cal)
    new = copy.deepcopy(cal[3])
    new["run"]["id"] = "run_new"
    for s in new["run"]["signals"]:
        if s.get("kind") == "score":
            s["value"] = round(min(1.0, s["value"] + 0.1), 6)
    values = {s["source"]["ref"]: s["value"] for s in new["run"]["signals"] if s.get("kind") == "score"}
    rows = {r["id"]: r for r in fit["judges"]}
    d = {j: rows[j]["noise"] ** -2 for j in values}
    total = sum(d.values())
    mean = sum(d[j] * (values[j] - rows[j]["bias"]) for j in values) / total
    sd = (1.0 / total + sum((d[j] / total) ** 2 * rows[j]["bias_sd"] ** 2 for j in values)) ** 0.5
    got = judges.unpooled_quality(flow, new, fit)
    assert abs(got[0] - mean) < 1e-12 and abs(got[1] - sd) < 1e-12
    assert abs(got[0] - judges.current_quality(cal + [new])["run_new"][0]) < 0.01
    assert judges.unpooled_quality(flow, H._strip(new, ()), fit) == (None, None)
    refused = judges.unpooled_quality(flow, new, {**fit, "refused": "too-many-judges:14"})
    assert isinstance(refused, judges.Unfitted) and refused.reason == "too-many-judges:14"


# A run no reader can walk: a document or `run` that is not an object, signals that are not a list of objects


MEMBERS = {"text": ["x"], "number": [5], "null": [None], "list": [[]], "true": [True], "two": ["x", 7]}
CONTAINERS = {"object": {"a": 1}, "empty-object": {}, "text": "xy", "empty-text": "", "number": 5, "zero": 0,
              "null": None, "true": True, "false": False}
SIGNAL_SHAPES = [f"member-{k}" for k in MEMBERS] + [f"container-{k}" for k in CONTAINERS]
NOT_OBJECT = {"text": "x", "number": 5, "zero": 0, "true": True, "list": ["x"], "empty-list": [], "null": None,
              "false": False, "empty-text": ""}
RUN_SHAPES = ["document-list", "document-text"] + [f"run-{k}" for k in NOT_OBJECT]
BEFORE = {"config": "cfg_aaaaaaaaaaaa", "p_success": {"mean": 0.7, "lo": 0.6, "hi": 0.8, "level": 0.8},
          "cost": {"usd": {"mean": 1.0, "lo": 0.5, "hi": 1.5, "level": 0.8},
                   "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "ell": {"usd": {"mean": 1.2, "lo": 0.6, "hi": 1.8, "level": 0.8},
                  "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "rounds": {"mean": 1, "lo": 1, "hi": 1, "level": 0.8}, "per_piece": {}, "support": 3}


def _victim(cal):
    """The calibrated run a test breaks: the first one that is not the probe."""
    return next(d for d in cal if d["run"]["id"] != H.CALIBRATED_PROBE)


def _bad(doc, shape):
    """A copy of `doc` in `shape` (`SIGNAL_SHAPES`, `RUN_SHAPES`), and the reason the belief fit counts it under:
    bad members appended after the good signals, the signal list or `run` replaced, or the document a list or
    text."""
    kind, name = shape.split("-", 1)
    doc = copy.deepcopy(doc)
    if kind == "member":
        doc["run"]["signals"].extend(copy.deepcopy(MEMBERS[name]))
        return doc, "unreadable: signal not an object"
    if kind == "container":
        doc["run"]["signals"] = copy.deepcopy(CONTAINERS[name])
        return doc, "unreadable: signals not a list"
    if kind == "run":
        doc["run"] = copy.deepcopy(NOT_OBJECT[name])
        return doc, "unreadable: run not an object"
    return ([doc] if name == "list" else "x"), "unreadable: run not an object"


def _plain(x):
    """`x` as plain values to compare: dataclasses and objects as dicts, arrays as lists, floats by repr."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.floating, float)):
        return repr(float(x))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if hasattr(x, "__dict__"):
        return {"__type": type(x).__name__, **{k: _plain(v) for k, v in sorted(vars(x).items())}}
    return repr(x)


def _rows(out):
    """Everything `collect_rows` gives but its dropped counts, as plain values: each head's design rows and terms,
    targets, weights, runs, sources and types, the score metadata, the runs by source, the support, the checks,
    the overlap, the labels and the feature tally."""
    return _plain([v for i, v in enumerate(out) if i != 2])


def _plus(dropped, reason):
    """`collect_rows`' dropped counts with one more under `reason`."""
    want = Counter(dropped)
    want[reason] += 1
    return want


@pytest.mark.parametrize("shape", SIGNAL_SHAPES)
def test_a_misshapen_signal_list_is_unreadable_once_by_name(cal, shape):
    """A flow run whose `run.signals` is not a list (null, false, 0, empty text and an empty object too) or holds
    signals that are not objects is unreadable before it is parsed, beside coding runs and the other calibrated
    runs. The belief fit counts it once under its reason's name, which `fit` words as unreadable, and every head,
    row and other count, and `current_quality`, are those with the run removed."""
    victim = _victim(cal)
    bad, reason = _bad(victim, shape)
    kept = [d for d in cal if d is not victim]
    coding = _coding(20)
    out = belief_fit.collect_rows(iter(coding + kept[:3] + [bad] + kept[3:]), now=NOW)
    gone = belief_fit.collect_rows(iter(coding + kept), now=NOW)
    assert _rows(out) == _rows(gone)
    assert Counter(out[2]) == _plus(gone[2], reason)
    assert belief_fit.reason_note(reason) == "not a readable OCP v0.3 run"
    assert judges.current_quality(kept[:3] + [bad] + kept[3:]) == judges.current_quality(kept)


@pytest.mark.parametrize("shape", ["absent", "empty"])
def test_absent_signals_and_an_empty_list_are_read(cal, shape):
    """No `run.signals`, or an empty list, is a run with no signals: admitted, with nothing dropped."""
    doc = copy.deepcopy(_victim(cal))
    if shape == "absent":
        del doc["run"]["signals"]
    else:
        doc["run"]["signals"] = []
    dropped = Counter()
    assert [pr.run_id for _, _, pr in belief_fit.admitted([doc], dropped=dropped)] == [doc["run"]["id"]]
    assert not dropped


def test_a_misshapen_signal_list_keeps_the_admission_order(cal):
    """The signals check takes a parse failure's place. An origin in `without` and a shipped copy of a stored run
    count as before; a misshapen stored copy still stands for its run, so its shipped copy is not read; and a
    misshapen copy never takes its run id, so a well-formed copy before or after it is admitted."""
    good = _victim(cal)
    bad, reason = _bad(good, "member-text")
    rid = good["run"]["id"]

    def admit(pairs, without=()):
        dropped, overlap = Counter(), Counter()
        ids = [pr.run_id for _, _, pr in belief_fit.admitted(pairs, without=without, dropped=dropped,
                                                               overlap=overlap)]
        return ids, dict(dropped), dict(overlap)

    assert admit([("sweep", bad)], ("sweep",)) == ([], {"without sweep": 1}, {})
    assert admit([("user", good), ("shipped-x", bad)]) == ([rid], {belief_fit.SHIPPED_COPY: 1}, {"shipped-x": 1})
    assert admit([("user", bad), ("shipped-x", good)]) == ([], {reason: 1, belief_fit.SHIPPED_COPY: 1},
                                                           {"shipped-x": 1})
    assert admit([(None, bad), (None, good)]) == ([rid], {reason: 1}, {})
    assert admit([(None, good), (None, bad)]) == ([rid], {reason: 1}, {})


@pytest.mark.parametrize("shape", RUN_SHAPES)
def test_a_document_or_run_that_is_no_object_is_unreadable_once_by_name(cal, shape):
    """A document that is not an object, or whose `run` is not one (null, false, empty text and an empty list
    too), has no run id to read. It is unreadable before any admission rule, counted once by name whatever its
    origin, even one in `without`, and every head, row and other count, and `current_quality`, are those without
    it."""
    victim = _victim(cal)
    bad, reason = _bad(victim, shape)
    kept = [d for d in cal if d is not victim]
    coding = _coding(20)
    gone = belief_fit.collect_rows(iter(coding + kept), now=NOW)
    for origin, without in ((None, ()), ("user", ()), ("shipped-x", ()), ("sweep", ("sweep",))):
        out = belief_fit.collect_rows(iter(coding + kept[:3] + [(origin, bad)] + kept[3:]), now=NOW,
                                      without=without)
        assert _rows(out) == _rows(gone), origin
        assert Counter(out[2]) == _plus(gone[2], reason), origin
    assert judges.current_quality(kept[:3] + [bad] + kept[3:]) == judges.current_quality(kept)


def test_a_user_copy_with_no_run_object_shadows_no_shipped_copy(cal):
    """A stored copy whose `run` is not an object has no run id, so it stands for no run: a shipped copy beside it
    is admitted, with no overlap counted."""
    good = _victim(cal)
    dropped, overlap = Counter(), Counter()
    pairs = [("user", _bad(good, "run-text")[0]), ("shipped-x", good)]
    ids = [pr.run_id for _, _, pr in belief_fit.admitted(pairs, dropped=dropped, overlap=overlap)]
    assert ids == [good["run"]["id"]]
    assert (dict(dropped), dict(overlap)) == ({"unreadable: run not an object": 1}, {})


def test_the_signals_shape_test_is_shared_by_name(cal):
    """`outcome.unreadable_signals` is the shape test the belief fit counts by, public for every reader: the reason
    for each misshapen signal list, None for absent signals, an empty list, a calibrated run and a `run` that is not
    an object (`admitted` names that one itself). `outcome.signal_list` gives a list as it is, members and all, and
    no signals for any other value."""
    doc = _victim(cal)
    for shape in SIGNAL_SHAPES:
        bad, reason = _bad(doc, shape)
        assert unreadable_signals(bad) == reason, shape
        assert signal_list(bad["run"]) == (bad["run"]["signals"] if shape.startswith("member") else []), shape
    absent = copy.deepcopy(doc)
    del absent["run"]["signals"]
    empty = {**doc, "run": {**doc["run"], "signals": []}}
    assert [unreadable_signals(d) for d in (doc, absent, empty, _bad(doc, "run-text")[0])] == [None] * 4
    assert signal_list(absent["run"]) == [] and signal_list(doc["run"]) is doc["run"]["signals"]


def test_a_document_with_no_run_still_reads_no_configuration(cal):
    """A document with no `run` is not misshapen: it is parsed, and reads as a run with no configuration."""
    doc = copy.deepcopy(_victim(cal))
    del doc["run"]
    dropped = Counter()
    assert list(belief_fit.admitted([doc], dropped=dropped)) == []
    assert dropped == Counter({"no configuration": 1})


def test_a_parser_attribute_error_is_never_swallowed(cal, monkeypatch):
    """The checks are no catch-all: a well-shaped run whose parser raises AttributeError (a bug) still raises,
    from `admitted` and from the belief fit."""
    def broken(*args, **kwargs):
        raise AttributeError("a parser bug")

    monkeypatch.setattr(belief_fit, "parse_run", broken)
    with pytest.raises(AttributeError, match="a parser bug"):
        list(belief_fit.admitted([_victim(cal)]))
    with pytest.raises(AttributeError, match="a parser bug"):
        belief_fit.collect_rows(iter(cal), now=NOW)


def _store_reads(root, cal, victim, shape):
    """The calibrated runs imported into a store at `root`, the probe open and the rest finished, then the
    victim's file written by hand: in `shape` (`_bad`), with no `run.signals` ("absent"), or the victim never
    imported ("gone"). The probe is finished over its receipt. Gives its receipt's `after`, and
    `current_quality`, `collect_rows` over the store's finished runs, and the report."""
    probe = H.CALIBRATED_PROBE
    store = Store(root)
    for doc in cal:
        if not (shape == "gone" and doc is victim):
            store.import_run(copy.deepcopy(doc), finished=doc["run"]["id"] != probe)
    if shape != "gone":
        path = store.run_path(victim["run"]["id"])
        doc = json.loads(path.read_text())
        if shape == "absent":
            del doc["run"]["signals"]
        else:
            doc = _bad(doc, shape)[0]
        path.write_text(json.dumps(doc))
    store.write_receipt({"id": f"rct_{probe[4:]}", "rec": None, "run": probe, "fit": "fit_1", "before": BEFORE})
    store_finish.finish_run(store, probe, settle=fake_settle(), no_fit=True)
    docs = list(store.finished_docs())
    return {"after": next(r for r in store.receipts_for(probe) if r.get("after"))["after"],
            "quality": judges.current_quality(docs), "rows": belief_fit.collect_rows(iter(docs), now=NOW),
            "report": report_payload(store)}


def _changed(a, b, path=""):
    """The dotted paths at which two payloads differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            yield from _changed(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
    elif a != b:
        yield path


@pytest.mark.parametrize("shape", ["member-text", "container-object", "container-number"])
def test_one_misshapen_stored_run_leaves_finish_and_the_report(cal, shape, tmp_path):
    """One other finished flow run's file written by hand after import, with a signal that is not an object, or a
    signal list that is no list (an object, or a number, which B's readers read as no signals). The probe still
    finishes and the report still reads. The probe's receipt, `current_quality` and the belief fit's heads and
    rows are those of the store without that run, the fit counting it once by name. The report is that of the same
    run with no signals, which differs from the store without it only by the run's own row: one more finished
    run, read as unknown, with its known dollars."""
    victim = _victim(cal)
    reason = _bad(victim, shape)[1]
    got = {k: _store_reads(tmp_path / k, cal, victim, s) for k, s in (("bad", shape), ("absent", "absent"),
                                                                       ("gone", "gone"))}
    bad, gone = got["bad"], got["gone"]
    assert bad["after"] == gone["after"]
    assert bad["quality"] == gone["quality"]
    assert _rows(bad["rows"]) == _rows(gone["rows"])
    assert Counter(bad["rows"][2]) == _plus(gone["rows"][2], reason)
    assert bad["report"] == got["absent"]["report"]
    r, g = bad["report"], gone["report"]
    c, k = r["cost_per_accepted"], g["cost_per_accepted"]
    src = next(s for s in c["by_source"] if c["by_source"][s] != k["by_source"].get(s))
    assert (r["runs"]["total"], r["runs"]["finished"], c["unknown"], c["accepted"]) == (
        g["runs"]["total"] + 1, g["runs"]["finished"] + 1, k["unknown"] + 1, k["accepted"])
    assert (c["by_source"][src]["runs"], c["by_source"][src]["unknown"]) == (
        k["by_source"][src]["runs"] + 1, k["by_source"][src]["unknown"] + 1)
    usd, n, a = run_cost(victim)["usd_known"], c["accepted"], c["by_source"][src]["accepted"]
    assert usd > 0 and n > 0 and a > 0
    assert c["total_usd"] == pytest.approx(k["total_usd"] + usd, abs=1e-6)
    assert c["by_source"][src]["usd"] == pytest.approx(k["by_source"][src]["usd"] + usd, abs=1e-6)
    assert c["usd"] == pytest.approx((k["total_usd"] + usd) / n, abs=1e-6)
    assert c["by_source"][src]["usd_per_accepted"] == pytest.approx((k["by_source"][src]["usd"] + usd) / a, abs=1e-6)
    follow = {"runs.total", "runs.finished", "cost_per_accepted.unknown", "cost_per_accepted.total_usd",
              "cost_per_accepted.usd", *(f"cost_per_accepted.by_source.{src}.{f}"
                                         for f in ("runs", "unknown", "usd", "usd_per_accepted"))}
    assert set(_changed(r, g)) == follow


@pytest.mark.parametrize("value", [{"a": 1}, "xy", 5, True, False, 0, None],
                         ids=["object", "text", "number", "true", "false", "zero", "null"])
def test_b_readers_take_a_signal_list_of_another_type_as_none(cal, value):
    """B's readers read a `run.signals` that is no list as no signals, never walking it. On a flow run and on a
    coding run, the outcome under a score rule and under the review rule, the judges' reading and the review check
    are those of the run with no `run.signals`."""
    probe = next(d for d in cal if d["run"]["id"] == H.CALIBRATED_PROBE)
    for doc in (probe, _coding(2)[0]):
        odd, none = copy.deepcopy(doc), copy.deepcopy(doc)
        odd["run"]["signals"] = value
        del none["run"]["signals"]
        for rule in (QUALITY, REVIEW_RULE):
            assert evidence(odd, rule, now=NOW) == evidence(none, rule, now=NOW), rule.name
        assert judges.judged(odd, "quality") == judges.judged(none, "quality")
        assert review_only(odd) == review_only(none)


def _reviewed(doc, value, tests=False):
    """A copy of a coding run whose verdict is one `review` at `value`, beside its `tests` verdict when `tests`."""
    out = copy.deepcopy(doc)
    signals = out["run"]["signals"]
    verdict = next(s for s in signals if s.get("kind") == "verdict")
    review = {**verdict, "id": "sig_review", "name": "review", "value": value,
              "observed_at": "2026-09-20T11:10:00-07:00"}
    out["run"]["signals"] = [s for s in signals if tests or s is not verdict] + [review]
    return out


def _members(doc, shape):
    """`doc` with the signals of `MEMBERS[shape]`, which are not objects, before its own and again after them."""
    bad = MEMBERS[shape]
    return {**doc, "run": {**doc["run"], "signals": [*bad, *doc["run"]["signals"], *bad]}}


def _only(doc, shape):
    """`doc` whose signals are only those of `MEMBERS[shape]`, or none when `shape` is None."""
    return {**doc, "run": {**doc["run"], "signals": list(MEMBERS[shape]) if shape else []}}


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_review_only_skips_a_signal_that_is_not_an_object(shape):
    """`review_only` reads a run's verdicts past its signals that are not objects (`rules._verdicts` keeps only
    objects): a review verdict alone is review-only and one beside a `tests` verdict is not, with them before and
    after the verdicts or without them; signals that are all such, or none, hold no review verdict."""
    doc = _coding(2)[0]
    alone, both = _reviewed(doc, "pass"), _reviewed(doc, "pass", tests=True)
    assert (review_only(alone), review_only(both)) == (True, False)
    assert (review_only(_members(alone, shape)), review_only(_members(both, shape))) == (True, False)
    assert (review_only(_only(doc, shape)), review_only(_only(doc, None))) == (False, False)


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_the_review_rule_skips_a_signal_that_is_not_an_object(shape):
    """The review rule's reading passes over signals that are not objects (`rules.evidence` keeps only objects): a
    run's evidence with them before and after its signals is its evidence without them, for an accepted, a rejected
    and an unknown review; signals that are all such read as none, with no review."""
    doc = _coding(2)[0]
    for value, z, reason in (("pass", 1.0, None), ("reject", 0.0, None), ("maybe", None, "unknown:review=maybe")):
        clean = _reviewed(doc, value)
        ev = evidence(clean, REVIEW_RULE, now=NOW)
        assert ev.z == z and (reason is None or reason in ev.reasons), (value, ev)
        assert evidence(_members(clean, shape), REVIEW_RULE, now=NOW) == ev, value
    none = evidence(_only(doc, None), REVIEW_RULE, now=NOW)
    assert none.z is None and "missing:review" in none.reasons
    assert evidence(_only(doc, shape), REVIEW_RULE, now=NOW) == none
