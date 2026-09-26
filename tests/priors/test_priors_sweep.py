"""The sweep converter (lane 11): synthetic contract v3 runs, no real sweep data."""

from __future__ import annotations

import copy
import json

import pytest

from loopmath.belief.design import parse_run, structure
from loopmath.logmatch.tariff import price_table, table_id
from loopmath.priors import ocpdoc
from loopmath.priors.show import run_summary
from loopmath.priors.build import BundleError, run_sweep
from loopmath.priors.sweep import batch_of, convert_sweep_run, iter_sweep
from loopmath.priors.validate import validate_bundle_doc

SEP = "·"


def _att(tid, n, state, start, end, *, receipt=None, reason=None, tokens=None, cost=None, cause=None, model=None,
         actor="developer", evidence="reported", effort="high"):
    a = {"id": f"{tid}{SEP}a{n}", "n": n, "state": state, "actor": actor, "started_at": start, "ended_at": end,
         "outcome": {"result": state, "evidence": evidence}}
    if model:
        a["model"] = model
    if receipt:
        a["outcome"]["receipt"] = receipt
    if reason:
        a["outcome"]["reason"] = reason
    if cause:
        a["cause"] = cause
    if tokens is not None:
        a["ext"] = {"tokens": tokens, "reasoning_effort": effort}
        if cost is not None:
            a["ext"]["cost_usd"] = cost
    return a


def _tok(inp, out, cr, cw):
    return {"input": inp, "output": out, "cache_read": cr, "cache_write": cw, "total": inp + out + cr + cw}


def accepted_run():
    """Default planner (opus5, xhigh), fable high developer, sol reviewer; round 1 fails the gate, round 2 passes."""
    return {
        "dagr": 3,
        "run": {"id": "run-sweep0830-t3-bugfix--claude-fable-5--high@rev-gpt-5.6-sol-xhigh",
                "title": "t3-bugfix under fable", "started_at": "2026-08-30T18:00:00Z"},
        "generated_at": "2026-08-30T19:00:00Z",
        "ext": {"experiment": {"sweep": "sweep0830", "task": "t3-bugfix",
                               "arm": {"id": "claude-fable-5--high", "family": "claude", "model": "claude-fable-5",
                                       "effort": "high"},
                               "reviewer": {"model": "gpt-5.6-sol", "effort": "xhigh", "family": "codex"},
                               "accepted": True, "dev_attempts": 2, "planner_shared": True}},
        "tasks": [
            {"id": "PLAN", "kind": "plan", "state": "done", "attempts": [
                _att("PLAN", 1, "done", "2026-08-30T18:00:00Z", "2026-08-30T18:01:00Z", actor="planner",
                     model=f"opus5{SEP}xhigh", effort="xhigh", receipt="plan.md: 10 lines \u2014 shared",
                     tokens=_tok(2, 1000, 10000, 5000), cost=0.1)]},
            {"id": "DEV", "kind": "impl", "state": "done", "attempts": [
                _att("DEV", 1, "rejected", "2026-08-30T18:10:00Z", "2026-08-30T18:12:00Z",
                     reason="gate failed (exit 1)", tokens=_tok(10, 2000, 100000, 20000), cost=0.5),
                _att("DEV", 2, "done", "2026-08-30T18:12:01Z", "2026-08-30T18:14:00Z",
                     receipt="developer CLI ok; gate pass; reviewer approved",
                     cause={"type": "gate_failed", "ref": f"GATE{SEP}a1"}, tokens=_tok(5, 1000, 50000, 1000), cost=0.2)]},
            {"id": "REV", "kind": "review", "state": "done", "attempts": [
                _att("REV", 1, "done", "2026-08-30T18:14:01Z", "2026-08-30T18:15:00Z", actor="reviewer",
                     receipt="verdict: approve=True, 0 blocking", tokens=_tok(1000, 300, 2000, 0))]},
            {"id": "GATE", "kind": "gate", "state": "done", "attempts": [
                _att("GATE", 1, "failed", "2026-08-30T18:12:00Z", "2026-08-30T18:12:01Z", actor="harness",
                     receipt="gate/run_gate.sh workdir: exit 1 \u2014 GATE FAIL: 9 of 36 checks failed",
                     evidence="verified"),
                _att("GATE", 2, "done", "2026-08-30T18:14:00Z", "2026-08-30T18:14:01Z", actor="harness",
                     receipt="gate/run_gate.sh workdir: exit 0 (pass)", evidence="verified")]},
        ],
        "events": [],
    }


def rows_for(doc, *, cli_ok=True, timed_out=False, earlier=0):
    rows = []
    for i in range(earlier):  # a superseded execution before the recorded one
        rows.append({"record": "attempt", "attempt": 1, "started_at": f"2026-08-30T10:0{i}:00Z",
                     "dev_cli_ok": False, "dev_timed_out": False, "max_attempts": 3})
    dev = [t for t in doc["tasks"] if t["id"] == "DEV"][0]["attempts"]
    for a in dev:
        rows.append({"record": "attempt", "attempt": a["n"], "started_at": a["started_at"], "dev_cli_ok": cli_ok,
                     "dev_timed_out": timed_out, "max_attempts": 3, "reviewer_verdict_parse_ok": True,
                     "tokens": {"developer": {"thinking": 123}, "planner": {"thinking": 45}, "reviewer": None}})
    return rows


def crashed_run():
    """Every developer round crashed at once: an infrastructure error, not a model failure."""
    doc = accepted_run()
    doc["run"]["id"] = "run-sweep0830-t7-sql--claude-sonnet-5--max@plan-gpt-5.6-luna-low@rev-claude-opus-5-xhigh"
    exp = doc["ext"]["experiment"]
    exp.update({"task": "t7-sql", "accepted": False, "dev_attempts": 3,
                "arm": {"id": "claude-sonnet-5--max", "family": "claude", "model": "claude-sonnet-5", "effort": "max"},
                "planner": {"model": "gpt-5.6-luna", "effort": "low", "family": "codex"},
                "reviewer": {"model": "claude-opus-5", "effort": "xhigh", "family": "claude"}})
    tasks = {t["id"]: t for t in doc["tasks"]}
    tasks["DEV"]["state"] = "failed"
    tasks["DEV"]["attempts"] = [
        _att("DEV", n, "failed", f"2026-08-30T18:1{n}:00Z", f"2026-08-30T18:1{n}:02Z",
             reason="developer CLI failed: rc=1 timed_out=False is_error=True | 0.0k out tokens",
             tokens=_tok(0, 0, 0, 0), cost=0.0) for n in (1, 2, 3)]
    tasks["GATE"]["attempts"] = [
        _att("GATE", n, "failed", f"2026-08-30T18:1{n}:02Z", f"2026-08-30T18:1{n}:03Z", actor="harness",
             receipt="gate/run_gate.sh workdir: exit 2 \u2014 GATE FAIL: /Users/someone/secret/path",
             evidence="verified") for n in (1, 2, 3)]
    tasks["REV"]["state"] = "canceled"
    tasks["REV"]["attempts"] = []
    return doc


def test_accepted_run_structure():
    src = accepted_run()
    doc, warnings = convert_sweep_run(src, rows_for(src), stem="t3-bugfix--claude-fable-5--high@rev-gpt-5.6-sol-xhigh")
    assert warnings == []
    assert doc["ocp"] == "0.3"
    run = doc["run"]
    assert run["task"]["id"] == "sweep0830/t3-bugfix"
    assert run["task"]["type"] == "bug_fix"
    assert run["task"]["source"] == {"kind": "sweep", "ref": "sweep0830"}
    assert run["task"]["features"]["size"] == "s"
    cfg = run["configuration"]
    assert cfg["source"] == "designed"
    assert [p["id"] for p in cfg["workflow"]["pieces"]] == ["plan", "implement", "review"]
    assert cfg["workflow"]["control"]["budget"] == 3
    assert cfg["workflow"]["control"]["gates"] == ["implement", "review"]
    assert cfg["settings"]["plan"]["model"]["id"] == "opus-5"  # default planner read from the label
    assert cfg["settings"]["plan"]["effort"] == "xhigh"
    assert cfg["settings"]["implement"]["harness"] == "claude-code"
    assert cfg["settings"]["review"]["harness"] == "codex"
    assert cfg["settings"]["review"]["model"]["family"] == "sol"
    verdicts = {s["name"]: s["value"] for s in run["signals"] if s["kind"] == "verdict"}
    assert verdicts == {"tests": "pass", "referee": "accept"}
    score = [s for s in run["signals"] if s["kind"] == "score"]
    assert score and score[0]["value"] == 1.0 and score[0]["scale"] == "fraction"
    assert run["acceptance_rule"]["requires"] == ["tests", "referee"]
    assert validate_bundle_doc(doc)[0] == []


def test_tokens_and_dollars_follow_the_price_table():
    src = accepted_run()
    doc, _ = convert_sweep_run(src, rows_for(src), stem="x")
    dev1 = next(a for a in doc["attempts"] if a["id"] == "implement.a1")
    assert dev1["cost"]["input_tokens"] == 10
    assert dev1["cost"]["cached_input_tokens"] == 100000
    assert dev1["cost"]["cache_creation_tokens"] == 20000
    assert dev1["cost"]["output_tokens"] == 2000
    assert dev1["cost"]["reasoning_tokens"] == 123
    r = price_table().rate("claude-fable-5")
    expected = (10 * r["input"] + 100000 * r["cache_read"] + 20000 * r["cache_write"] + 2000 * r["output"]) / 1e6
    assert dev1["cost"]["usd"] == pytest.approx(expected)
    assert dev1["cost"]["tariff"]["id"] == ocpdoc.tariff()["id"] == table_id()  # lane 2's tariff id
    plan = next(a for a in doc["attempts"] if a["node"] == "plan")
    assert plan["ext"]["dev.loopmath.shared_across_runs"] is True


def test_repair_round_and_cause_link_the_gate():
    src = accepted_run()
    doc, _ = convert_sweep_run(src, rows_for(src), stem="x")
    dev2 = next(a for a in doc["attempts"] if a["id"] == "implement.a2")
    assert dev2["round"] == 2
    assert dev2["cause"] == {"type": "gate_failed", "ref": "tests.a1"}
    gate1 = next(a for a in doc["attempts"] if a["id"] == "tests.a1")
    assert gate1["harness"] == "command" and "cost" not in gate1
    review = next(a for a in doc["attempts"] if a["node"] == "review")
    assert review["n"] == 1 and review["round"] == 2  # the round of the developer attempt it followed


def delayed_review_run():
    """The first review comes after developer round 3: the gate failed in rounds 1 and 2."""
    doc = accepted_run()
    doc["ext"]["experiment"]["dev_attempts"] = 3
    tasks = {t["id"]: t for t in doc["tasks"]}
    tasks["DEV"]["attempts"] = [
        _att("DEV", 1, "rejected", "2026-08-30T18:10:00Z", "2026-08-30T18:12:00Z", reason="gate failed (exit 1)",
             tokens=_tok(10, 2000, 100000, 20000)),
        _att("DEV", 2, "rejected", "2026-08-30T18:12:01Z", "2026-08-30T18:14:00Z", reason="gate failed (exit 1)",
             cause={"type": "gate_failed", "ref": f"GATE{SEP}a1"}, tokens=_tok(5, 1000, 50000, 1000)),
        _att("DEV", 3, "done", "2026-08-30T18:14:01Z", "2026-08-30T18:16:00Z",
             receipt="developer CLI ok; gate pass; reviewer approved",
             cause={"type": "gate_failed", "ref": f"GATE{SEP}a2"}, tokens=_tok(5, 1000, 50000, 1000))]
    tasks["GATE"]["attempts"] = [
        _att("GATE", n, state, start, end, actor="harness", evidence="verified",
             receipt=("gate/run_gate.sh workdir: exit 0 (pass)" if state == "done"
                      else "gate/run_gate.sh workdir: exit 1, GATE FAIL: 9 of 36 checks failed"))
        for n, state, start, end in ((1, "failed", "2026-08-30T18:12:00Z", "2026-08-30T18:12:01Z"),
                                     (2, "failed", "2026-08-30T18:14:00Z", "2026-08-30T18:14:01Z"),
                                     (3, "done", "2026-08-30T18:16:00Z", "2026-08-30T18:16:01Z"))]
    tasks["REV"]["attempts"] = [
        _att("REV", 1, "done", "2026-08-30T18:16:01Z", "2026-08-30T18:17:00Z", actor="reviewer",
             receipt="verdict: approve=True, 0 blocking", tokens=_tok(1000, 300, 2000, 0))]
    return doc


def _delayed_rows(src):
    rows = rows_for(src)
    for r in rows:  # the reviewer only ran in round 3; its row alone has reviewer tokens and a parsed verdict
        ran = r["attempt"] == 3
        r["tokens"] = {**r["tokens"], "reviewer": {"thinking": 77} if ran else None}
        r["reviewer_verdict_parse_ok"] = True if ran else None
    return rows


def test_delayed_first_review_is_in_the_round_it_followed():
    src = delayed_review_run()
    doc, warnings = convert_sweep_run(src, _delayed_rows(src), stem="x")
    assert warnings == [] and validate_bundle_doc(doc)[0] == []
    review = next(a for a in doc["attempts"] if a["node"] == "review")
    assert (review["n"], review["round"]) == (1, 3)
    assert review["cost"]["reasoning_tokens"] == 77  # from the round 3 row, not the round 1 row
    rounds = {a["id"]: a.get("round") for a in doc["attempts"]}
    assert [rounds[f"implement.a{k}"] for k in (1, 2, 3)] == [1, 2, 3]
    # The gate passed only after round 3, and the review ran in that round and no other.
    gates = [a for a in doc["attempts"] if a["node"] == "tests"]
    passed_after = [k for k, g in enumerate(gates, start=1) if g["outcome"]["result"] == "done"]
    assert passed_after == [3] == [a["round"] for a in doc["attempts"] if a["node"] == "review"]
    verdicts = {s["name"]: s["value"] for s in doc["run"]["signals"] if s["kind"] == "verdict"}
    assert verdicts == {"tests": "pass", "referee": "accept"}


def test_delayed_first_review_gives_lane5_the_right_gates():
    src = delayed_review_run()
    doc, _ = convert_sweep_run(src, _delayed_rows(src), stem="x")
    parsed = parse_run(doc)
    st = structure(parsed.config)
    gates = [(st.gates[g].after, k, v) for g, k, v in parsed.gates]
    assert gates == [("implement", 1, False), ("implement", 2, False), ("implement", 3, True), ("review", 3, True)]
    assert [a.round for a in parsed.attempts if a.piece == "review"] == [3]


def crashed_review_run():
    """Round 1 passes the gate but the reviewer CLI crashes (the harness writes a rejection); round 2 is approved."""
    doc = accepted_run()
    tasks = {t["id"]: t for t in doc["tasks"]}
    tasks["DEV"]["attempts"][0].update({"state": "rejected", "outcome": {
        "result": "rejected", "evidence": "reported", "reason": "reviewer verdict unparseable (treated as reject)"}})
    tasks["GATE"]["attempts"][0].update({"state": "done", "outcome": {
        "result": "done", "evidence": "verified", "receipt": "gate/run_gate.sh workdir: exit 0 (pass)"}})
    crashed = _att("REV", 1, "failed", "2026-08-30T18:12:01Z", "2026-08-30T18:12:02Z", actor="reviewer",
                   reason="reviewer CLI failed: rc=1 timed_out=False is_error=True", tokens=_tok(0, 0, 0, 0))
    approved = _att("REV", 2, "done", "2026-08-30T18:14:01Z", "2026-08-30T18:15:00Z", actor="reviewer",
                    receipt="verdict: approve=True, 0 blocking", tokens=_tok(1000, 300, 2000, 0))
    tasks["REV"]["attempts"] = [crashed, approved]
    tasks["DEV"]["attempts"][1]["started_at"] = "2026-08-30T18:12:03Z"  # the repair starts after the crash
    return doc


def test_crashed_review_judges_nothing():
    src = crashed_review_run()
    rows = rows_for(src)
    rows[0].update({"review_ran": True, "review_rc": 1, "review_timed_out": False})
    rows[1].update({"review_ran": True, "review_rc": 0, "review_timed_out": False})
    for given in (rows, []):  # the attempts.jsonl flags, else the run file's reason text
        doc, _ = convert_sweep_run(src, given, stem="x")
        reviews = [a for a in doc["attempts"] if a["node"] == "review"]
        assert [(a["round"], (a.get("ext") or {}).get("dev.loopmath.infra_error")) for a in reviews] == [
            (1, True), (2, None)]
        parsed = parse_run(doc)
        st = structure(parsed.config)
        gates = [(st.gates[g].after, k, v) for g, k, v in parsed.gates]
        assert ("review", 1, False) not in gates and ("review", 2, True) in gates  # no rejection from a crash
        assert validate_bundle_doc(doc)[0] == []


def test_crashed_rounds_are_errors_not_failures():
    src = crashed_run()
    doc, warnings = convert_sweep_run(src, rows_for(src, cli_ok=False, earlier=2), stem="t7-sql--x")
    assert warnings == []
    run = doc["run"]
    verdicts = {s["name"]: s["value"] for s in run["signals"] if s["kind"] == "verdict"}
    assert verdicts == {"tests": "error"}  # the review never ran
    info = run["ext"]["dev.loopmath.prior"]
    assert info["infra_error"] is True and info["infra_rounds"] == 3
    assert info["superseded_rows"] == 2 and info["later_rows"] == 0
    assert all(a["ext"]["dev.loopmath.infra_error"] for a in doc["attempts"] if a["node"] == "implement")
    assert not [s for s in run["signals"] if s["kind"] == "score"]
    assert run["configuration"]["settings"]["plan"]["model"]["id"] == "gpt-5.6-luna"
    assert validate_bundle_doc(doc)[0] == []


def test_timed_out_round_has_unknown_cost_not_zero():
    src = crashed_run()
    rows = rows_for(src, cli_ok=False, timed_out=True)
    doc, _ = convert_sweep_run(src, rows, stem="x")
    devs = [a for a in doc["attempts"] if a["node"] == "implement"]
    assert all("cost" not in a for a in devs)
    assert all("dev.loopmath.tokens_unknown" in a["ext"] for a in devs)
    assert all("dev.loopmath.infra_error" not in a["ext"] for a in devs)  # a timeout is the model's
    assert doc["run"]["ext"]["dev.loopmath.prior"]["cost_complete"] is False
    assert {s["name"]: s["value"] for s in doc["run"]["signals"]}["tests"] == "fail"
    summary = run_summary([doc])
    assert summary["cost_unknown_attempts"] == 3 and summary["unpriced_attempts"] == 0  # gates are not counted


def test_config_id_is_stable_across_key_order():
    src = accepted_run()
    doc, _ = convert_sweep_run(src, rows_for(src), stem="x")
    cfg = doc["run"]["configuration"]
    shuffled_settings = {k: dict(reversed(list(v.items()))) for k, v in reversed(list(cfg["settings"].items()))}
    shuffled_wf = dict(reversed(list(cfg["workflow"].items())))
    assert ocpdoc.config_id(shuffled_wf, shuffled_settings) == cfg["id"]
    other = copy.deepcopy(cfg["settings"])
    other["implement"]["effort"] = "max"
    assert ocpdoc.config_id(cfg["workflow"], other) != cfg["id"]
    renamed = {**cfg["workflow"], "id": "something_else", "title": "t"}
    assert ocpdoc.config_id(renamed, cfg["settings"]) == cfg["id"]


def test_mismatch_between_flag_and_verdicts_is_warned():
    src = accepted_run()
    src["ext"]["experiment"]["accepted"] = False
    _, warnings = convert_sweep_run(src, rows_for(src), stem="x")
    assert warnings and "accepted=False" in warnings[0]


def test_unlabelled_task_is_refused():
    src = accepted_run()
    src["ext"]["experiment"]["task"] = "t9-unknown"
    with pytest.raises(ValueError, match="no label"):
        convert_sweep_run(src, [], stem="x")


def test_iter_sweep_reads_a_results_folder(tmp_path):
    (tmp_path / "dagr").mkdir()
    runs = [("t3-bugfix--claude-fable-5--high@rev-gpt-5.6-sol-xhigh", accepted_run()),
            ("t7-sql--claude-sonnet-5--max@plan-gpt-5.6-luna-low@rev-claude-opus-5-xhigh", crashed_run())]
    lines = []
    for stem, doc in runs:
        (tmp_path / "dagr" / f"{stem}.run.json").write_text(json.dumps(doc))
        for row in rows_for(doc, cli_ok=stem.startswith("t3")):
            lines.append(json.dumps({**row, "run_id": stem}))
    (tmp_path / "attempts.jsonl").write_text("\n".join(lines) + "\n")
    out = list(iter_sweep(tmp_path))
    assert [p.name.split("--")[0] for p, _, _ in out] == ["t3-bugfix", "t7-sql"]
    assert all(w == [] for _, _, w in out)
    assert out[1][1]["run"]["ext"]["dev.loopmath.prior"]["infra_error"] is True


# ---------------------------------------------------------------- batches (23B)
STEM = "t3-bugfix--claude-fable-5--high@rev-gpt-5.6-sol-xhigh"


def second_batch_run():
    """accepted_run as a later batch (sweep0925) writes it: its own run id prefix, the same task."""
    src = accepted_run()
    src["run"]["id"] = src["run"]["id"].replace("run-sweep0830-", "run-sweep0925-")
    src["ext"]["experiment"]["sweep"] = "sweep0925"
    return src


def results_folder(tmp_path, name, runs):
    """A results folder: `dagr/<stem>.run.json` and attempts.jsonl rows that name each run by its stem."""
    d = tmp_path / name
    (d / "dagr").mkdir(parents=True)
    lines = []
    for stem, src in runs:
        (d / "dagr" / f"{stem}.run.json").write_text(json.dumps(src))
        lines += [json.dumps({**r, "run_id": stem}) for r in rows_for(src)]
    (d / "attempts.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_a_second_batch_has_its_own_run_id_and_the_same_task():
    first, _ = convert_sweep_run(accepted_run(), rows_for(accepted_run()), stem=STEM)
    src = second_batch_run()
    doc, warnings = convert_sweep_run(src, rows_for(src), stem=STEM)
    run = doc["run"]
    assert warnings == []
    assert (first["run"]["id"], run["id"]) == (f"sweep0830/{STEM}", f"sweep0925/{STEM}")
    assert run["task"]["id"] == first["run"]["task"]["id"] == "sweep0830/t3-bugfix"  # one task node, both batches
    assert run["task"]["source"] == {"kind": "sweep", "ref": "sweep0925"}
    assert {s["source"]["ref"] for s in run["signals"]} == {"sweep0925 harness"}
    assert json.loads(json.dumps(doc).replace("sweep0925", "sweep0830")) == first  # nothing else depends on it


def test_the_batch_comes_from_the_run_id_when_the_experiment_does_not_name_it():
    src = second_batch_run()
    del src["ext"]["experiment"]["sweep"]
    assert batch_of(src) == "sweep0925"
    src["run"]["id"] = "t3-bugfix"
    with pytest.raises(ValueError, match="no sweep batch"):
        convert_sweep_run(src, [], stem="x")


def test_iter_sweep_matches_a_second_batchs_rows(tmp_path):
    """The rows name a run by its stem; the batch's own `run-<batch>-` prefix is stripped to find them.

    Only the rows say these rounds crashed and how much they thought, so both prove the match."""
    stem = "t7-sql--claude-sonnet-5--max@plan-gpt-5.6-luna-low@rev-claude-opus-5-xhigh"
    src = crashed_run()
    src["run"]["id"] = f"run-sweep0925-{stem}"
    src["ext"]["experiment"]["sweep"] = "sweep0925"
    for att in [t for t in src["tasks"] if t["id"] == "DEV"][0]["attempts"]:
        att["outcome"]["reason"] = "rc=1"
    d = tmp_path / "results"
    (d / "dagr").mkdir(parents=True)
    (d / "dagr" / f"{stem}.run.json").write_text(json.dumps(src))
    (d / "attempts.jsonl").write_text("\n".join(json.dumps({**r, "run_id": stem}) for r in rows_for(src, cli_ok=False)))
    [(_, doc, warnings)] = list(iter_sweep(d))
    assert warnings == [] and doc["run"]["id"] == f"sweep0925/{stem}"
    assert doc["run"]["ext"]["dev.loopmath.prior"]["infra_error"] is True
    impl = [a for a in doc["attempts"] if a["node"] == "implement"]
    assert len(impl) == 3 and all(a["ext"]["dev.loopmath.infra_error"] for a in impl)
    assert {a["cost"]["reasoning_tokens"] for a in impl} == {123}


def test_run_sweep_reads_one_folder_per_batch(tmp_path):
    a = results_folder(tmp_path, "a", [(STEM, accepted_run())])
    b = results_folder(tmp_path, "b", [(STEM, second_batch_run())])
    res = run_sweep([a, b])
    assert sorted(d["run"]["id"] for d in res.docs) == [f"sweep0830/{STEM}", f"sweep0925/{STEM}"]
    one = {"fable-5": {"runs": 1, "accepted": 1, "not_accepted": 0, "no_verdict": 0, "infra_error_runs": 0,
                       "cost_incomplete_runs": 0}}
    assert res.counts["batches"] == {"sweep0830": one, "sweep0925": one}
    assert sorted(res.inputs["batches"]) == ["sweep0830", "sweep0925"] and res.inputs["files"] == 4
    assert any(n.startswith("sweep0925:") for n in res.notes)
    assert run_sweep(a).counts["batches"] == {"sweep0830": one}  # one folder, as before
    with pytest.raises(BundleError, match="one batch"):
        run_sweep([a, results_folder(tmp_path, "c", [(STEM, accepted_run())])])  # sweep0830 twice
