"""Receipts: before from a stored recommendation, after at finish, re-score on a late event (spec 05 section 6)."""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from loopmath.recommend import receipts as R
from loopmath.recommend.engine import Settings, recommend
from loopmath.types import DEFAULT_RULE, Evidence, Receipt

from recommend_fakes import IR, PIR, TASK, FakeBelief, Num, cfg, solo

USUAL = cfg(IR, implement="opus", review="astra")
GOAL = cfg(PIR, plan="opusx", implement="opus", review="astra")
CHEAP = solo("luna")
EDITED = solo("fable")


def stored_rec() -> tuple[dict, FakeBelief]:
    b = FakeBelief({USUAL.id: Num(0.80, 2.00), GOAL.id: Num(0.90, 3.00), CHEAP.id: Num(0.55, 0.40),
                    EDITED.id: Num(0.60, 0.70)},
                   {CHEAP.id: {"usd": 0.3, "success_pp": 0.0, "cost_pct": -30.0, "score": None}}, {CHEAP.id: 0.3})
    rec = recommend(b, TASK, DEFAULT_RULE, usual=USUAL, usual_from="history",
                    configs=[(GOAL, "catalog"), (CHEAP, "catalog")], settings=Settings(goal="p90"))
    payload = rec.payload()
    payload["rec"] = "rec_TEST"
    payload["fit"] = {"id": b.fit_id}
    payload["candidates"] = [c.to_dict() for c in rec.candidates]
    # what the store reads back is JSON
    return json.loads(json.dumps(payload)), b


def test_before_receipt_uses_the_stored_prediction():
    rec, b = stored_rec()
    r = R.before_receipt(rec, run="run_1", config=GOAL)
    assert r.rec == "rec_TEST" and r.run == "run_1" and r.fit == b.fit_id and r.id.startswith("rct_")
    assert r.before.config == GOAL.id and r.before.cost.usd.mean == 3.0
    assert r.after is None and r.scored is None
    assert Receipt.from_dict(json.loads(json.dumps(r.to_dict()))) == r


def test_an_edited_configuration_is_predicted_again_with_the_rec_rescue():
    rec, b = stored_rec()
    with pytest.raises(ValueError):
        R.before_receipt(rec, run="run_2", config=EDITED)
    r = R.before_receipt(rec, run="run_2", config=EDITED, task=TASK, belief=b, rule=DEFAULT_RULE)
    assert r.before.config == EDITED.id and r.before.cost.usd.mean == 0.70
    rescue = rec["rescue"]["usd"]
    assert r.before.ell.usd.mean == pytest.approx(0.70 + 0.40 * rescue)


def test_after_receipt_scores_cost_z_and_surprise():
    rec, _ = stored_rec()
    r = R.before_receipt(rec, run="run_1", config=GOAL)
    done = R.finish_receipt(r, cost_usd=2.5, tokens=450_000, evidence=Evidence(1.0, 0.98, "verified"), rounds=1.0)
    s = done.scored
    assert done.after["cost"] == {"usd": 2.5, "tokens": 450_000} and done.after["z"] == 1.0
    assert s["cost_in_interval"] is True  # 2.10 to 3.90
    assert s["tokens_in_interval"] is True
    p1 = 0.98 * 0.90 + 0.02 * 0.10
    assert s["log_score_z"] == pytest.approx(math.log(p1), abs=1e-6)
    assert s["surprise"] < 0  # below the median on log cost
    far = R.finish_receipt(r, cost_usd=9.0, tokens=None, evidence=Evidence(None, 0.95, "reported"))
    assert far.scored["cost_in_interval"] is False and far.scored["surprise"] > 1.28
    assert far.scored["log_score_z"] is None and far.scored["tokens_in_interval"] is None


def test_a_late_revert_rescores_the_receipt():
    rec, _ = stored_rec()
    r = R.before_receipt(rec, run="run_1", config=GOAL)
    done = R.finish_receipt(r, cost_usd=2.5, tokens=450_000, evidence=Evidence(1.0, 0.98, "verified"))
    late = R.rescore(done, None, Evidence(0.0, 0.98, "verified", reasons=("late:revert",)),
                     observed_at="2026-09-25T10:00:00-07:00")
    assert late.after["z"] == 0.0 and late.after["reasons"] == ["late:revert"]
    assert late.after["cost"] == done.after["cost"]
    assert late.scored["log_score_z"] < done.scored["log_score_z"]
    assert late.scored["cost_in_interval"] == done.scored["cost_in_interval"]
    assert late.after["rescored"][0]["before"]["z"] == 1.0
    with pytest.raises(ValueError):
        R.rescore(r, None, Evidence(0.0, 0.98, "verified"))


def test_score_rule_receipt_scores_each_score():
    from recommend_fakes import score_rule
    rule = score_rule("runtime_s", "<=", 200.0)
    b = FakeBelief({USUAL.id: Num(0.8, 2.0, score=190.0, p_reach=0.6)}, score_name="runtime_s", score_unit="s")
    r = R.before_receipt(None, run="run_3", config=USUAL, task=TASK, belief=b, rule=rule)
    done = R.finish_receipt(r, cost_usd=2.0, tokens=None, evidence=Evidence(1.0, 0.95, "reported",
                                                                             scores={"runtime_s": 180.0}))
    sc = done.scored["scores"]["runtime_s"]
    assert sc == {"value": 180.0, "predicted": 190.0, "in_interval": True, "p_reach": 0.6}


def test_ocp_receipt_has_the_spec_01_shape_and_exploration_terms():
    rec, _ = stored_rec()
    r = R.before_receipt(rec, run="run_x", config=CHEAP)
    o = R.ocp_receipt(r, rec)
    assert set(o) == {"before"} and o["before"]["predicted"]["p_success"]["level"] == 0.8
    assert o["before"]["rec"] == "rec_TEST" and o["before"]["predicted"]["cost_usd"]["mean"] == 0.40
    assert o["before"]["gain_per_run"]["usd"] == 0.3 and o["before"]["price_usd"] == 0.40
    assert o["before"]["payback_runs"] == pytest.approx(0.4 / 0.3, rel=1e-5)
    done = R.finish_receipt(r, cost_usd=0.5, tokens=90_000, evidence=Evidence(0.0, 0.95, "reported"))
    o = R.ocp_receipt(done, rec)
    assert o["after"] == {"cost_usd": 0.5, "tokens": 90_000, "signals": []}


def run_doc(config: dict, *, rec_id: str | None = "rec_TEST", signals=("sig_tests",)) -> dict:
    """An OCP v0.3 run document with two attempts (the shape of lane 1's full-fields example)."""
    cfg = {**config, "source": "exploration", **({"rec": rec_id} if rec_id else {})}
    cost1 = {"input_tokens": 8000, "cached_input_tokens": 300000, "output_tokens": 12000, "usd": 1.10}
    cost2 = {"input_tokens": 2000, "cached_input_tokens": 100000, "cache_creation_tokens": 20000,
             "output_tokens": 8000, "usd": 1.40}
    return {"ocp": "0.3",
            "run": {"id": "run_doc1", "configuration": cfg, "ended_at": "2026-09-23T18:00:00-07:00",
                    "signals": [{"id": s, "kind": "verdict", "name": "tests", "value": "pass"} for s in signals]},
            "attempts": [{"id": "implement.a1", "round": 1, "cost": cost1},
                         {"id": "review.a1", "round": 2, "cost": cost2}]}


def test_an_unpriced_attempt_leaves_the_run_dollars_unknown():
    # D62, D67: $1.10 priced plus an attempt without dollars is not a $1.10 run; tokens stay the full sum
    rec, _ = stored_rec()
    doc = run_doc(GOAL.to_dict())
    del doc["attempts"][1]["cost"]["usd"]
    got = R.realized_from_doc(doc)
    assert (got["cost_usd"], got["cost_usd_known"], got["tokens"]) == (None, 1.10, 450_000)
    r = R.after_receipt(doc, rec, Evidence(1.0, 0.98, "verified"), "fit_now")
    assert r.after["cost"] == {"usd": None, "tokens": 450_000}
    assert r.scored["cost_in_interval"] is None and r.scored["surprise"] is None
    assert r.scored["tokens_in_interval"] is not None
    assert R.ocp_receipt(r, rec)["after"]["cost_usd"] is None
    doc["attempts"][1]["cost"] = None  # no cost object at all: the same
    assert R.realized_from_doc(doc)["cost_usd"] is None
    assert R.realized_from_doc(run_doc(GOAL.to_dict()))["cost_usd"] == pytest.approx(2.5)
    assert R.realized_from_doc({"run": {"id": "run_x"}, "attempts": []})["cost_usd"] is None


def test_after_receipt_reads_the_run_document_d16():
    rec, _ = stored_rec()
    doc = run_doc(GOAL.to_dict())
    r = R.after_receipt(doc, rec, Evidence(1.0, 0.98, "verified"), "fit_now")
    assert r.rec == "rec_TEST" and r.run == "run_doc1" and r.fit == "fit_20260923170000"
    assert r.after["cost"] == {"usd": 2.5, "tokens": 450_000.0} and r.after["rounds"] == 2.0
    assert r.after["signals"] == ["sig_tests"] and r.after["finished_at"] == "2026-09-23T18:00:00-07:00"
    assert r.scored["cost_in_interval"] is True
    # the receipt stored at run start keeps its id
    start = R.before_receipt(rec, run="run_doc1", config=GOAL, receipt_id="rct_START")
    r2 = R.after_receipt(doc, rec, Evidence(1.0, 0.98, "verified"), None, receipt=start.to_dict())
    assert r2.id == "rct_START" and r2.scored == r.scored
    # a bare `run` object works too
    bare = {**doc["run"], "attempts": doc["attempts"]}
    assert R.after_receipt(bare, rec, Evidence(1.0, 0.98, "verified")).after["cost"]["usd"] == 2.5


def test_after_receipt_is_none_without_a_prediction_and_uses_the_belief_for_an_edit():
    rec, b = stored_rec()
    ev = Evidence(1.0, 0.95, "reported")
    assert R.after_receipt(run_doc(EDITED.to_dict()), rec, ev, "fit_now") is None
    assert R.after_receipt(run_doc(GOAL.to_dict(), rec_id=None), None, ev) is None
    assert R.after_receipt({"run": {"id": "run_x"}}, rec, ev) is None
    r = R.after_receipt(run_doc(EDITED.to_dict()), rec, ev, "fit_now", belief=b, task=TASK, rule=DEFAULT_RULE)
    assert r.before.config == EDITED.id and r.fit == b.fit_id and r.after["cost"]["usd"] == 2.5


def test_rescore_with_the_run_document_picks_up_the_late_signal():
    rec, _ = stored_rec()
    r = R.after_receipt(run_doc(GOAL.to_dict()), rec, Evidence(1.0, 0.98, "verified"))
    late_doc = run_doc(GOAL.to_dict(), signals=("sig_tests", "sig_revert"))
    late = R.rescore(r.to_dict(), late_doc, Evidence(0.0, 0.98, "verified", reasons=("late:revert",)))
    assert late.id == r.id and late.after["signals"] == ["sig_tests", "sig_revert"]
    assert late.after["z"] == 0.0 and late.scored["log_score_z"] < r.scored["log_score_z"]
    assert R.ocp_receipt(late, rec)["after"]["signals"] == ["sig_tests", "sig_revert"]


def test_with_fit_after_records_how_the_prediction_moved():
    rec, b = stored_rec()
    r = R.after_receipt(run_doc(GOAL.to_dict()), rec, Evidence(1.0, 0.98, "verified"))
    b.nums[GOAL.id] = Num(0.92, 2.80)
    moved = R.with_fit_after(r, "fit_after1", b.predict(TASK, GOAL))
    assert moved.after["fit_after"] == "fit_after1" and moved.scored == r.scored
    assert moved.after["moved"]["p_success"] == {"from": 0.90, "to": 0.92}
    o = R.ocp_receipt(moved, rec)["after"]
    assert o["fit_after"] == "fit_after1" and o["moved"]["p_success"]["to"] == 0.92


def check_ocp_receipt(o: dict) -> None:
    """The constraints of `$defs/receipt` in the OCP v0.3 schema (lane 1) that loopmath's output can break."""
    assert set(o) <= {"before", "after", "ext"}
    b = o["before"]
    for key in ("rec", "fit"):
        assert key not in b or (isinstance(b[key], str) and len(b[key]) <= 200), key
    for name, iv in b["predicted"].items():
        assert isinstance(iv["mean"], (int, float)) and all(isinstance(iv[k], (int, float)) for k in ("lo", "hi"))
        assert 0 <= iv.get("level", 0.8) <= 1
    for key in ("price_usd", "payback_runs"):
        assert b.get(key) is None or isinstance(b[key], (int, float)), key
    if "after" in o:
        a = o["after"]
        assert a["cost_usd"] is None or isinstance(a["cost_usd"], (int, float))
        assert a["tokens"] is None or (isinstance(a["tokens"], int) and not isinstance(a["tokens"], bool)
                                       and a["tokens"] >= 0)
        assert isinstance(a["signals"], list)
        assert a.get("fit_after") is None or (isinstance(a["fit_after"], str) and len(a["fit_after"]) <= 200)


def test_ocp_receipts_fit_the_schema_with_and_without_a_recommendation():
    rec, b = stored_rec()
    ev = Evidence(1.0, 0.95, "reported")
    linked = R.before_receipt(rec, run="run_doc1", config=GOAL)
    standalone = R.before_receipt(None, run="run_doc1", config=GOAL, task=TASK, belief=b, rule=DEFAULT_RULE)
    assert standalone.rec is None
    for r, source in ((linked, rec), (standalone, None)):
        o = R.ocp_receipt(r, source)
        check_ocp_receipt(o)
        done = R.after_receipt(run_doc(GOAL.to_dict(), rec_id=r.rec), source, ev, receipt=r)
        o = R.ocp_receipt(R.with_fit_after(done, "fit_after1", b.predict(TASK, GOAL)), source)
        check_ocp_receipt(o)
        assert o["after"]["tokens"] == 450_000 and isinstance(o["after"]["tokens"], int)
    assert "rec" not in R.ocp_receipt(standalone)["before"]
    assert R.ocp_receipt(linked, rec)["before"]["rec"] == "rec_TEST"


EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "spec" / "examples" / "v0.3" / "full-fields.ocp.json"


@pytest.mark.parametrize("linked", [True, False])
def test_receipt_passes_strict_ocp_validation(linked):
    """Lane 1's strict validator on its full-fields example with our receipt."""
    from loopmath.ocp.emit import validate_strict
    doc = json.loads(EXAMPLE.read_text())
    rec, b = stored_rec()
    r = (R.before_receipt(rec, run=doc["run"]["id"], config=GOAL) if linked else
         R.before_receipt(None, run=doc["run"]["id"], config=GOAL, task=TASK, belief=b, rule=DEFAULT_RULE))
    for receipt in (r, R.after_receipt(doc, rec if linked else None, Evidence(1.0, 0.95, "reported"), receipt=r)):
        doc["run"]["receipt"] = R.ocp_receipt(receipt, rec if linked else None)
        findings = validate_strict(doc)
        assert [f for f in findings if f["level"] == "error"] == []
