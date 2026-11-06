"""The store reads a flow run (`run.flow`) through the current judges fit: `run finish`, a late signal's re-score,
the flagged pass, the 0.2.7 settlement and `report` give it its value in one fit of the store's finished flow runs,
its own copy first, so a stored copy never wins. A run that fit leaves out reads no score and no raw judge value,
with the reason passed on. Runs without `run.flow` read as before: no fit, and no other run file read for them."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime

import pytest

from loopmath.belief import rules
from loopmath.belief.rules import resolve_rule
from loopmath.flows import judges
from loopmath.store import Store
from loopmath.store import finish as F
from loopmath.store import report as report_module
from loopmath.store import runs as R
from loopmath.store.ids import parse_ts
from loopmath.store.report import report_payload
from loopmath.types import Signal
from tests.flows.test_flow_judges import CALIBRATED_PROBE, calibrated_runs
from tests.store.store_helpers import fake_settle, finished_doc

P = CALIBRATED_PROBE
Q = "run_cal-drop-critic-q04-1"  # a second drop-critic run, with no person score
TARGET = 0.52  # each rule's target: between the probe's value in the store's fit and in a fit of it alone
BEFORE = {"config": "cfg_aaaaaaaaaaaa", "p_success": {"mean": 0.7, "lo": 0.6, "hi": 0.8, "level": 0.8},
          "cost": {"usd": {"mean": 1.0, "lo": 0.5, "hi": 1.5, "level": 0.8},
                   "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "ell": {"usd": {"mean": 1.2, "lo": 0.6, "hi": 1.8, "level": 0.8},
                  "tokens": {"mean": 1, "lo": 0, "hi": 2, "level": 0.8}},
          "rounds": {"mean": 1, "lo": 1, "hi": 1, "level": 0.8}, "per_piece": {}, "support": 3}
CODING = "run_coding1"
NOW = "2026-11-04T12:00:00Z"
# The receipt `run finish` wrote for CODING (`_coding_store`, the clock at NOW) before flow runs were read through
# the store's fit, 1272 bytes.
CODING_RECEIPT_SHA256 = "39c9ced683aae9214664b85d28a08f15e18cdcdc34531a87297881ef65277c0f"


@pytest.fixture(scope="module")
def cal(tmp_path_factory):
    """The 20 calibrated runs, each rule's target at TARGET."""
    docs = calibrated_runs(tmp_path_factory.mktemp("cal") / "lit-brief")
    for doc in docs:
        doc["run"]["acceptance_rule"]["score"]["target"] = TARGET
    return docs


@pytest.fixture
def fits(monkeypatch):
    """Each `judges.current_quality` call, as the number of documents it is given, and how many runs were fitted
    alone (a `quality_map` call outside one)."""
    seen = {"current": [], "alone": 0}
    real_current, real_map = judges.current_quality, judges.quality_map
    inside: list[bool] = []

    def current(docs, **kw):
        docs = list(docs)
        seen["current"].append(len(docs))
        inside.append(True)
        try:
            return real_current(docs, **kw)
        finally:
            inside.pop()

    def quality_map(items):
        if not inside:
            seen["alone"] += 1
        return real_map(items)

    monkeypatch.setattr(judges, "current_quality", current)
    monkeypatch.setattr(judges, "quality_map", quality_map)
    return seen


@pytest.fixture
def readings(monkeypatch):
    """Run id to the `quality` values the store's sites read a run with (`evidence_for`), in order."""
    seen: dict[str, list] = {}
    real = F.evidence_for

    def evidence_for(doc, rule, **kw):
        seen.setdefault(doc["run"]["id"], []).append(kw.get("quality"))
        return real(doc, rule, **kw)

    monkeypatch.setattr(F, "evidence_for", evidence_for)
    monkeypatch.setattr(report_module, "evidence_for", evidence_for)
    return seen


@pytest.fixture
def scans(monkeypatch):
    """How many times the store's finished run files are read (`Store.finished_docs`)."""
    count = [0]
    real = Store.finished_docs

    def finished_docs(self):
        count[0] += 1
        return real(self)

    monkeypatch.setattr(Store, "finished_docs", finished_docs)
    return count


def _store(tmp_path, docs, open_=()):
    store = Store(tmp_path / "home")
    for doc in docs:
        store.import_run(copy.deepcopy(doc), finished=doc["run"]["id"] not in open_)
    return store


def _opened(store, run):
    """The receipt `run start --rec` writes: its id and before part, no after."""
    store.write_receipt({"id": f"rct_{run[4:]}", "rec": None, "run": run, "fit": "fit_1", "before": BEFORE})
    return f"rct_{run[4:]}"


def _finished(store, *runs):
    for run in runs:
        _opened(store, run)
        F.finish_run(store, run, settle=fake_settle(), no_fit=True)


def _receipt(store, run):
    return next(r for r in store.receipts_for(run) if r.get("after"))


def _person(run, value, at="2026-01-08T00:00:00+00:00"):
    return Signal(id=f"sig_{run[4:]}_late", run=run, kind="score", name="quality", value=value, better="higher",
                  scale="fraction", observed_at=at, source={"kind": "user", "ref": "person"})


def _evidence(doc, quality):
    return rules.evidence(doc, resolve_rule(doc), now=datetime.now().astimezone(), quality=quality)


def _others(cal, *runs):
    return [d for d in cal if d["run"]["id"] not in (P, *runs)]


def _before_027(doc):
    """`doc` with its rule read otherwise before 0.2.7 (`excludes_events: []`), so a settlement re-scores it."""
    doc = copy.deepcopy(doc)
    doc["run"]["acceptance_rule"]["excludes_events"] = []
    assert F.read_otherwise_before_027(doc)
    return doc


def _scored(store, run, after):
    """A scored receipt of `run`, named in its run file as loopmath names the receipts it writes."""
    rid = f"rct_{run[4:]}"
    store.write_receipt({"id": rid, "rec": None, "run": run, "fit": "fit_1", "before": BEFORE, "after": after})
    store._update(run, lambda d: R.add_event(d, "receipt_written", detail=rid))
    return rid


# The value


def test_the_probe_reads_otherwise_in_the_stores_fit_than_alone(cal):
    probe = next(d for d in cal if d["run"]["id"] == P)
    every, alone = judges.current_quality(cal)[P], judges.current_quality([probe])[P]
    assert abs(every[0] - alone[0]) > 0.01 and every[0] < TARGET <= alone[0]
    assert _evidence(probe, every).z == 0.0 and _evidence(probe, alone).z == 1.0


def test_finish_reads_a_flow_run_through_the_stores_fit(cal, tmp_path, fits):
    store = _store(tmp_path, cal, open_=(P,))
    _opened(store, P)
    out = F.finish_run(store, P, settle=fake_settle(), no_fit=True)
    assert fits == {"current": [20], "alone": 0}  # the run first, then the 19 finished
    done = store.run_doc(P)
    want = judges.current_quality(list(store.finished_docs()))[P]
    after = _receipt(store, P)["after"]
    assert after["scores"]["quality"] == want[0]
    assert after["z"] == out["evidence"]["z"] == _evidence(done, want).z == 0.0
    assert _evidence(done, judges.current_quality([done])[P]).z == 1.0  # alone, it would be accepted


def test_a_stale_stored_copy_never_wins(cal, tmp_path):
    """A second file holding the probe with other judge scores, named to be read first: finish and a late
    signal's re-score read the run's own copy."""
    store = _store(tmp_path, cal, open_=(P,))
    stale = copy.deepcopy(next(d for d in cal if d["run"]["id"] == P))
    for s in stale["run"]["signals"]:
        if s.get("kind") == "score":
            s["value"] = 0.95
    R.set_store_ext(stale, state=R.FINISHED)
    (store.home / "runs" / "a-stale.ocp.json").write_text(json.dumps(stale), encoding="utf-8")
    assert next(store.finished_docs())["run"]["signals"] == stale["run"]["signals"]
    others = _others(cal)
    _finished(store, P)
    want = judges.current_quality([store.run_doc(P), *others])[P]
    assert abs(want[0] - judges.current_quality([stale, *others])[P][0]) > 0.01
    assert _receipt(store, P)["after"]["scores"]["quality"] == want[0]
    assert store.add_signal(P, _person(P, 0.3)) == "late"
    assert F.rescore_run(store, P)["rescored"] == [f"rct_{P[4:]}"]
    want = judges.current_quality([store.run_doc(P), *others])[P]
    assert _receipt(store, P)["after"]["scores"]["quality"] == want[0]


def test_the_flagged_pass_rescores_each_flow_run_through_the_stores_fit(cal, tmp_path, fits):
    store = _store(tmp_path, cal, open_=(P, Q))
    _finished(store, P, Q)
    before = judges.current_quality(list(store.finished_docs()))
    for run in (P, Q):
        assert store.add_signal(run, _person(run, 0.2)) == "late"
    fits["current"].clear()
    got = F.rescore_flagged(store)
    assert sorted(got["rescored"]) == sorted(f"rct_{r[4:]}" for r in (P, Q)) and got["failed"] == []
    assert fits == {"current": [21, 21], "alone": 0}  # one fit for each flagged run, under its lock
    want = judges.current_quality(list(store.finished_docs()))
    for run in (P, Q):
        rc = _receipt(store, run)
        assert want[run][0] != before[run][0] and rc["after"]["scores"]["quality"] == want[run][0]
        assert rc["after"]["z"] == _evidence(store.run_doc(run), want[run]).z and not rc["rescore"]
    assert F.rescore_flagged(store)["rescored"] == []  # a second pass changes nothing


def test_settlement_reads_flow_runs_through_one_fit_of_the_whole_store(cal, tmp_path, fits, readings):
    """Only two flow runs and a coding run are read otherwise before 0.2.7: one fit, of all 20 flow runs, reads
    both. The coding run is read as before, with no quality."""
    docs = [_before_027(d) if d["run"]["id"] in (P, Q) else d for d in cal]
    coding = finished_doc(CODING)
    coding["run"]["acceptance_rule"] = {"name": "tests", "definition": "tests", "requires": ["tests"],
                                        "excludes_events": []}
    coding["run"]["signals"] = [{"id": "sig_tests", "kind": "verdict", "name": "tests", "value": "pass",
                                 "observed_at": "2026-09-20T10:05:00-07:00", "source": {"kind": "harness"},
                                 "tier": "verified"}]
    store = _store(tmp_path, [*docs, coding])
    alone = judges.current_quality([store.run_doc(P)])[P]
    _scored(store, P, {"cost": {"usd": 1.0, "tokens": 1}, "z": 1.0, "q": 0.95, "scores": {"quality": alone[0]}})
    _scored(store, Q, {"cost": {"usd": 1.0, "tokens": 1}, "z": None, "q": 0.95, "scores": {}, "rescore": True})
    _scored(store, CODING, {"cost": {"usd": 1.0, "tokens": 1}, "z": 0.0, "q": 0.95, "scores": {}})
    fits["current"].clear()
    readings.clear()
    got = F.settle_rule_receipts(store)
    assert got["checked"] == sorted([P, Q, CODING]) and got["failed"] == []
    assert sorted(got["rescored"]) == sorted(f"rct_{r[4:]}" for r in (P, Q, CODING))
    assert fits == {"current": [20], "alone": 0}
    want = judges.current_quality(list(store.finished_docs()))
    for run in (P, Q):
        after = _receipt(store, run)["after"]
        assert after["scores"]["quality"] == want[run][0]
        assert after["z"] == _evidence(store.run_doc(run), want[run]).z
    assert _receipt(store, P)["after"]["z"] == 0.0
    assert readings == {P: [want[P]], Q: [want[Q]], CODING: [None]}
    assert _receipt(store, CODING)["after"]["z"] == 1.0


def test_report_reads_flow_runs_through_one_fit_of_the_whole_store(cal, tmp_path, fits, readings):
    """A window holding only the probe of the flow runs: its z is read in the fit of all 20. A coding run in the
    window is read with no quality, and a written receipt's z is kept."""
    store = _store(tmp_path, [*cal, finished_doc(CODING)])
    since = parse_ts(next(d for d in cal if d["run"]["id"] == P)["run"]["started_at"])
    fits["current"].clear()
    readings.clear()
    out = report_payload(store, since=since)
    assert out["runs"]["finished"] == 2 and fits == {"current": [20], "alone": 0}
    want = judges.current_quality(list(store.finished_docs()))[P]
    probe = next(e for e in out["exploration"]["runs"] if e["run"] == P)
    assert probe["z"] == _evidence(store.run_doc(P), want).z == 0.0
    assert readings == {P: [want], CODING: [None]}
    fits["current"].clear()
    report_payload(store)
    assert fits["current"] == [20]  # one fit for the 20 flow rows
    rid = _scored(store, P, {"cost": {"usd": 1.0, "tokens": 1}, "z": 1.0, "q": 0.95, "scores": {}})
    held = store.receipt_path(rid).read_bytes()
    probe = next(e for e in report_payload(store, since=since)["exploration"]["runs"] if e["run"] == P)
    assert probe["z"] == 1.0 and store.receipt_path(rid).read_bytes() == held


# A run the fit leaves out


@pytest.mark.parametrize("case", ["missing", "refused"])
def test_a_flow_run_the_fit_leaves_out_reads_no_score_at_every_site(case, cal, tmp_path, monkeypatch, readings):
    """The fit gives the probe no value (`missing`) or a refused fit's value with its reason (`refused`): finish,
    a late signal's re-score, the settlement and report read no score, pass the value on as it is, and never fit
    the run alone."""
    refused = judges.Unfitted("too-many-judges:9")

    def current(docs, **kw):
        return {} if case == "missing" else {d["run"]["id"]: refused for d in docs}

    monkeypatch.setattr(judges, "current_quality", current)
    monkeypatch.setattr(judges, "quality_map", lambda items: pytest.fail("a run was fitted alone"))
    expected = (None, None) if case == "missing" else refused
    reason = "unfitted:quality" if case == "missing" else "too-many-judges:9"
    store = _store(tmp_path, [_before_027(d) if d["run"]["id"] == P else d for d in cal], open_=(P,))

    def unscored():
        assert readings[P][-1] == expected and (case == "missing" or readings[P][-1] is refused)
        after = _receipt(store, P)["after"]
        assert "quality" not in after["scores"] and reason in after["reasons"]
        assert after["z"] == _evidence(store.run_doc(P), (None, None)).z

    _finished(store, P)
    unscored()
    store.add_signal(P, _person(P, 0.9))
    F.rescore_run(store, P)
    unscored()
    store.flag_rescore(P)
    assert F.settle_rule_receipts(store)["rescored"] == [f"rct_{P[4:]}"]
    unscored()
    out = report_payload(store)
    assert readings[Q] == [expected] and (case == "missing" or readings[Q][0] is refused)
    assert next(e for e in out["exploration"]["runs"] if e["run"] == Q)["z"] == _evidence(store.run_doc(Q),
                                                                                          (None, None)).z


SCORED_BEFORE = {**BEFORE, "scores": {"quality": {"name": "quality", "unit": None, "better": "higher",
                                                  "value": {"mean": 0.5, "lo": 0.0, "hi": 1.0, "level": 0.8},
                                                  "p_reach": 0.5, "support": 3}}}
MOVED = {"p_success": {"from": 0.7, "to": 0.6}, "cost_usd": {"from": 1.0, "to": 1.1}}
EARLIER = {"at": "2026-01-02T00:00:00+00:00", "before": {"z": None, "q": 0.95, "tier": "reported"}, "reasons": []}


def _numeric(store, run):
    """`run` finished with its value in the store's fit, a prediction of that score before it, and what a refit
    and an earlier re-score add: `fit_after`, `moved`, a history entry, a score name the run's signals do not
    hold and unknown members. The receipt as written."""
    rid = f"rct_{run[4:]}"
    store.write_receipt({"id": rid, "rec": None, "run": run, "fit": "fit_1", "before": SCORED_BEFORE})
    F.finish_run(store, run, settle=fake_settle(), no_fit=True)
    value = judges.current_quality(list(store.finished_docs()))[run][0]
    rc = _receipt(store, run)
    assert value is not None and rc["after"]["scores"]["quality"] == value
    assert rc["scored"]["scores"]["quality"]["value"] == value
    rc["after"].update(fit_after="fit_2", moved=MOVED, rescored=[EARLIER], held={"kept": True})
    rc["after"]["scores"]["legacy"] = 3.0
    rc["ext"] = {"kept": True}
    store.write_receipt(rc)
    return rc


@pytest.mark.parametrize("site", ["flagged", "settlement"])
@pytest.mark.parametrize("case", ["missing", "refused"])
def test_a_value_the_fit_no_longer_gives_leaves_the_rescored_receipt(case, site, cal, tmp_path, monkeypatch):
    """The probe's receipt holds its value in the store's fit; then the fit leaves it out (`missing`) or is refused
    (`refused`), and a person's late score flags it. The flagged pass, or the settlement, re-scores it: the value
    is gone from the scores and the scored part, no raw judge value takes its place, z and the reason are the
    unfitted reading's, and the cost, history, refit members, other score names and unknown members are kept."""
    docs = [_before_027(d) if site == "settlement" and d["run"]["id"] == P else d for d in cal]
    store = _store(tmp_path, docs, open_=(P,))
    old = _numeric(store, P)
    refused = judges.Unfitted("too-many-judges:9")
    monkeypatch.setattr(judges, "current_quality",
                        lambda docs, **kw: {} if case == "missing" else {d["run"]["id"]: refused for d in docs})
    monkeypatch.setattr(judges, "quality_map", lambda items: pytest.fail("a run was fitted alone"))
    assert store.add_signal(P, _person(P, 0.9)) == "late"
    got = F.rescore_flagged(store) if site == "flagged" else F.settle_rule_receipts(store)
    assert got["rescored"] == [old["id"]] and got["failed"] == []
    rc = _receipt(store, P)
    after = rc["after"]
    assert after["scores"] == {"legacy": 3.0}  # no value, raw or read before, for the rule's score
    assert "quality" not in rc["scored"]["scores"]
    assert (after["reasons"].count("unfitted:quality" if case == "missing" else "too-many-judges:9") == 1
            and after["z"] == _evidence(store.run_doc(P), (None, None)).z)
    assert after["cost"] == old["after"]["cost"] and after["fit_after"] == "fit_2" and after["moved"] == MOVED
    assert after["held"] == {"kept": True} and rc["ext"] == {"kept": True} and rc["rescore"] is False
    assert after["rescored"][0] == EARLIER and after["rescored"][1]["before"]["z"] == old["after"]["z"]
    assert len(after["rescored"]) == 2


def test_a_flow_run_no_judge_scored_keeps_its_scores_when_rescored(cal, tmp_path, monkeypatch):
    """A flow run with no judge score under the rule's score name is read as before: a re-score keeps the score
    names its evidence does not hold, that one too."""
    docs = copy.deepcopy(cal)
    probe = next(d for d in docs if d["run"]["id"] == P)
    probe["run"]["signals"] = [s for s in probe["run"]["signals"] if s.get("kind") != "score"]
    store = _store(tmp_path, docs)
    monkeypatch.setattr(judges, "current_quality", lambda docs, **kw: {})
    rid = _scored(store, P, {"cost": {"usd": 1.0, "tokens": 1}, "z": None, "q": 0.95, "scores": {"quality": 0.5}})
    store.flag_rescore(P)
    assert F.rescore_flagged(store)["rescored"] == [rid]
    assert _receipt(store, P)["after"]["scores"] == {"quality": 0.5}


def _judge_scores(doc):
    return [s for s in doc["run"]["signals"]
            if s.get("kind") == "score" and (s.get("source") or {}).get("kind") in ("judge", "user")]


def _unattributed(doc):
    for s in _judge_scores(doc):
        s["source"].pop("ref")


def _outside(doc):
    for s in _judge_scores(doc):
        s.update(scale="linear", value=1e308, ext={judges.JUDGE_EXT: {"range": [0, 10]}})


def _unranged(doc):
    for s in _judge_scores(doc):
        s["scale"] = "linear"


@pytest.mark.parametrize("change", [_unattributed, _outside, _unranged], ids=lambda f: f.__name__[1:])
def test_judge_scores_that_give_no_value_reach_the_receipt_with_their_reasons(change, cal, tmp_path, fits):
    """The probe's judge scores name no judge, lie outside their range, or are linear with no range: the store's
    fit leaves the run out, and its receipt reads no score, with the reader's reasons."""
    doc = copy.deepcopy(next(d for d in cal if d["run"]["id"] == P))
    change(doc)
    store = _store(tmp_path, [doc if d["run"]["id"] == P else d for d in cal], open_=(P,))
    _finished(store, P)
    done = store.run_doc(P)
    got = judges.judged(done, "quality")
    assert got.reasons and not got.usable and P not in judges.current_quality(list(store.finished_docs()))
    after = _receipt(store, P)["after"]
    assert "quality" not in after["scores"] and {"unfitted:quality", *got.reasons} <= set(after["reasons"])
    assert fits["alone"] == 0


def test_late_judge_scores_no_float_holds_are_named_on_the_receipt(cal, tmp_path, fits):
    """Late scores of 2**1024 from both model judges are no reading: the run keeps its value in the store's fit,
    and the re-scored receipt names each judge."""
    store = _store(tmp_path, cal, open_=(P,))
    _finished(store, P)
    held = _receipt(store, P)["after"]["scores"]["quality"]
    for judge in ("j-claude", "j-astra"):
        store.add_signal(P, Signal(id=f"sig_late_{judge}", run=P, kind="score", name="quality", value=2 ** 1024,
                                   better="higher", scale="fraction", observed_at="2026-01-08T00:00:00+00:00",
                                   source={"kind": "judge", "ref": judge}))
    assert F.rescore_run(store, P)["rescored"] == [f"rct_{P[4:]}"]
    want = judges.current_quality(list(store.finished_docs()))[P]
    after = _receipt(store, P)["after"]
    assert after["scores"]["quality"] == want[0] == held and fits["alone"] == 0
    assert {"nonfinite:j-astra", "nonfinite:j-claude"} <= set(after["reasons"])


# Runs without a flow


def _coding_store(tmp_path, cal):
    """The calibrated flow runs, finished, and CODING open with a passing tests verdict and a judge's score."""
    store = _store(tmp_path, cal)
    coding = finished_doc(CODING)
    coding["run"]["signals"] = [
        {"id": "sig_tests", "kind": "verdict", "name": "tests", "value": "pass",
         "observed_at": "2026-09-20T10:05:00-07:00", "source": {"kind": "harness"}, "tier": "verified"},
        {"id": "sig_quality", "kind": "score", "name": "quality", "value": 0.61, "scale": "fraction",
         "better": "higher", "observed_at": "2026-09-20T10:06:00-07:00", "source": {"kind": "judge", "ref": "j-claude"},
         "tier": "reported"}]
    store.import_run(coding, finished=False)
    _opened(store, CODING)
    return store


def test_a_coding_run_in_a_store_with_flow_runs_finishes_and_rescores_as_before(cal, tmp_path, monkeypatch, fits,
                                                                                scans):
    store = _coding_store(tmp_path, cal)
    monkeypatch.setattr(F, "now_iso", lambda: NOW)
    scans[0] = 0
    F.finish_run(store, CODING, settle=fake_settle(), no_fit=True)
    data = store.receipt_path(f"rct_{CODING[4:]}").read_bytes()
    assert (len(data), hashlib.sha256(data).hexdigest()) == (1272, CODING_RECEIPT_SHA256)
    assert fits["current"] == [] and scans[0] == 0
    store.add_signal(CODING, Signal(id="sig_quality2", run=CODING, kind="score", name="quality", value=0.4,
                                    better="higher", scale="fraction", observed_at="2026-09-20T10:07:00-07:00",
                                    source={"kind": "judge", "ref": "j-claude"}))
    F.rescore_run(store, CODING)
    assert _receipt(store, CODING)["after"]["scores"]["quality"] == 0.4  # the newest score, read raw as before
    assert fits == {"current": [], "alone": 0} and scans[0] == 0


@pytest.mark.parametrize("value", [2 ** 1024, "high"], ids=["nonfinite", "nonnumeric"])
def test_a_coding_run_whose_judge_score_gives_no_value_keeps_its_scores_when_rescored(value, cal, tmp_path, fits):
    """A coding run is read raw, as before, though a judge scored it under the rule's score name: when that score
    gives no value (not finite, or no number), a re-score keeps the value read before, as it keeps any score name
    the evidence does not hold. Only a flow run's value is dropped (`_unscored`)."""
    store = _store(tmp_path, cal)
    coding = finished_doc(CODING)
    coding["run"]["acceptance_rule"] = {"name": "quality>=0.5", "definition": "quality at least 0.5",
                                        "requires": [], "score": {"name": "quality", "target": 0.5,
                                                                  "better": "higher", "scale": "fraction"}}
    store.import_run(coding)
    store.add_signal(CODING, Signal(id="sig_quality", run=CODING, kind="score", name="quality", value=value,
                                    better="higher", scale="fraction", observed_at="2026-09-20T10:06:00-07:00",
                                    source={"kind": "judge", "ref": "j-claude"}))
    assert any(judges.judged(store.run_doc(CODING), "quality"))
    rid = _scored(store, CODING,
                  {"cost": {"usd": 1.0, "tokens": 1}, "z": None, "q": 0.95, "scores": {"quality": 0.5}})
    store.flag_rescore(CODING)
    assert F.rescore_flagged(store)["rescored"] == [rid]
    assert _receipt(store, CODING)["after"]["scores"] == {"quality": 0.5}
    assert fits == {"current": [], "alone": 0}


def test_loops_over_coding_runs_only_fit_nothing(tmp_path, fits, scans, readings):
    coding = finished_doc(CODING)
    coding["run"]["acceptance_rule"] = {"name": "tests", "definition": "tests", "requires": ["tests"],
                                        "excludes_events": []}
    store = _store(tmp_path, [coding, finished_doc("run_coding2")])
    _scored(store, CODING, {"cost": {"usd": 1.0, "tokens": 1}, "z": 1.0, "q": 0.95, "scores": {}})
    scans[0] = 0
    assert F.settle_rule_receipts(store)["checked"] == [CODING]
    assert scans[0] == 1  # the settlement's own read of the finished runs
    scans[0] = 0
    report_payload(store)
    assert scans[0] == 0 and fits == {"current": [], "alone": 0}
    assert set(readings) == {CODING, "run_coding2"} and all(q is None for v in readings.values() for q in v)


def test_the_fit_of_the_flow_runs_equals_the_fit_of_every_run(cal, tmp_path):
    """The store gives the judges fit only its flow runs; the fit admits no other, so the value is the same."""
    store = _coding_store(tmp_path, [d for d in cal if d["run"]["id"] != P])
    F.finish_run(store, CODING, settle=fake_settle(), no_fit=True)  # a finished coding run with a judge's score
    store.import_run(finished_doc("run_coding2"))
    probe = next(d for d in cal if d["run"]["id"] == P)
    every = list(store.finished_docs())
    assert len(every) == 21 and len(F._flow_docs(store)) == 19
    assert F._flow_quality(store, probe) == judges.current_quality([probe, *every])[P]
    assert judges.current_quality(F._flow_docs(store)) == judges.current_quality(every)
