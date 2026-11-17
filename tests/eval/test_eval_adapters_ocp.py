"""`adapters/ocp.py`: readings and pairs to OCP signals and preferences and back, and harness-runs rows in.

The round trips keep every reading, repeat and correction (acceptance check 2). All data is synthetic.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import unquote

import pytest

from loopmath.eval import contracts as C
from loopmath.eval.adapters import ocp as A
from loopmath.ocp.conformance import SIMULATE_KEY, validate_doc

STUDY = Path(__file__).resolve().parent / "data" / "study-small"
NS, KEPT = A.NS, A.KEPT
AT = "2026-02-03T10:00:00-08:00"


def document(run: dict, **more) -> dict:
    """A small OCP 0.4 run document for one row of `runs.jsonl`."""
    settings = {s["step"]: {"harness": "cli", "effort": s.get("effort"),
                            "model": {"raw": s["model"]["id"], "id": s["model"]["id"], "family": s["model"]["family"]}}
                for s in run["steps"]}
    workflow = {"pieces": [{"id": s["step"], "role": "dev"} for s in run["steps"]], "edges": []}
    body = {"id": run["run"], "task": {"id": run["input"]}, "labels": {"arm": run["arm"]},
            "configuration": {"workflow": workflow, "settings": settings}, "signals": [], **more}
    return {"ocp": "0.4", "producer": {"name": "example", "version": "1"}, "privacy": {"profile": "metadata_only"},
            "run": body, "nodes": [], "edges": [], "attempts": [], "artifacts": []}


def run_row(name: str, input_id: str = "case-01", arm: str = "baseline", family: str = "beta") -> dict:
    return {"run": name, "input": input_id, "arm": arm, "simulated": False,
            "steps": [{"step": "final", "model": {"id": f"model-{family}", "family": family}, "effort": "low"}]}


def errors(doc: dict) -> list[str]:
    return [f"{f.code} {f.path} {f.message}" for f in validate_doc(doc) if f.level == "error"]


def scores_of(batch: A.Batch) -> list[dict]:
    return [C.validate_score({**row, "schema": C.SCHEMA_SCORES}, "scores.jsonl", "") for row in batch.scores]


def pairs_of(batch: A.Batch) -> list[dict]:
    return [C.validate_pair({**row, "schema": C.SCHEMA_PAIRS}, "pairs.jsonl", "") for row in batch.pairs]


def by_id(rows: list[dict]) -> dict:
    return {row["id"]: row for row in rows}


def through_ocp(runs: list[dict], scores: list[dict], pairs: list[dict], persons=()) -> tuple[A.Batch, list[dict]]:
    """Export the rows, merge each entry into its run's document, check the documents and read them back."""
    entries, not_used = A.export(runs, scores, pairs, persons=persons)
    assert not_used == []
    rows = {row["run"]: row for row in runs}
    docs = [A.merge(document(rows[entry["run"]]), entry) for entry in entries]
    for doc in docs:
        assert errors(doc) == []
    docs = json.loads(json.dumps(docs))  # as written to files and read again
    return A.read_documents([(f"{doc['run']['id']}.ocp.json", doc) for doc in docs]), docs


# ---------------------------------------------------------------------------------------------------
# Acceptance check 2: round trips

def test_study_to_ocp_and_back_keeps_every_reading_repeat_and_correction():
    study = C.read_study(STUDY)
    assert (len(study.runs), len(study.scores), len(study.pairs)) == (16, 305, 18)
    batch, docs = through_ocp(study.runs, study.scores, study.pairs, persons=["reviewer"])
    assert batch.not_used == []
    assert sum(len(doc["run"]["signals"]) for doc in docs) == 305
    assert sum(len(doc["run"].get("preferences", [])) for doc in docs) == 36  # each pair sits in both runs' files
    back, pairs_back = scores_of(batch), pairs_of(batch)
    assert by_id(back) == by_id(study.scores) and len(back) == 305
    assert by_id(pairs_back) == by_id(study.pairs) and len(pairs_back) == 18
    # the correction and the repeats are still there, and still read the same way
    assert by_id(back)["s0305"]["replaces"] == "s0021"
    live, dropped = C.live(back, "scores.jsonl")
    assert len(live) == 304 and [entry["id"] for entry in dropped] == ["s0021"]
    def repeats(rows):
        calls = {}
        for row in rows:
            calls.setdefault((row["run"], row["step"], row["component"], row["judge"]), set()).add(row["call"])
        return sum(1 for ids in calls.values() if len(ids) > 1)
    assert repeats(back) == repeats(study.scores) > 0
    # the runs come back with their inputs, arms and step settings
    study_runs = {row["run"]: row for row in study.runs}
    for row in batch.runs:
        row = C.validate_run({**row, "schema": C.SCHEMA_RUNS}, "runs.jsonl", "")
        want = study_runs[row["run"]]
        assert (row["input"], row["arm"], row["steps"]) == (want["input"], want["arm"], want["steps"])
        assert "ext" not in row


def test_a_person_is_written_as_a_user_and_a_model_as_a_judge():
    study = C.read_study(STUDY)
    entries, _ = A.export(study.runs, study.scores, study.pairs, persons=["reviewer"])
    kinds = {(s["source"]["ref"], s["source"]["kind"]) for entry in entries for s in entry["signals"]}
    assert kinds == {("reviewer", "user"), ("judge-strong", "judge"), ("judge-cheap", "judge")}
    judges = {(p["judge"]["ext"][NS]["id"], p["judge"]["kind"]) for entry in entries for p in entry["preferences"]}
    assert judges == {("reviewer", "user"), ("judge-strong", "model")}


def test_a_signal_and_a_preference_hold_the_members_of_the_mapping():
    reading = C.validate_score({"schema": C.SCHEMA_SCORES, "id": "s2", "run": "r1", "step": "draft",
                                "component": "depth", "judge": "judge-a", "call": "c9", "value": 6.5,
                                "rubric": "brief@1", "observed_at": AT, "replaces": "s1", "assignment": "a7",
                                "cost_usd": 0.004}, "scores.jsonl", "")
    assert A.signal(reading) == {
        "id": "s2", "kind": "score", "name": "depth", "value": 6.5, "observed_at": AT,
        "source": {"kind": "judge", "ref": "judge-a"}, "tier": "reported",
        "ext": {NS: {"call": "c9", "step": "draft", "rubric": "brief@1", "replaces": "s1", "assignment": "a7",
                     "cost_usd": 0.004}}}
    pair = C.validate_pair({"schema": C.SCHEMA_PAIRS, "id": "p1", "a": "r1", "b": "r2", "step": "final",
                            "component": "overall", "judge": "judge-a", "call": "pc1", "shown_first": "b",
                            "verdict": "b", "rubric": "brief@1", "observed_at": AT}, "pairs.jsonl", "")
    assert A.preference(pair, "eval.case-01") == {
        "id": "p1", "slate": "eval.case-01", "winner": "r2", "members": ["r1", "r2"],
        "judge": {"kind": "model", "ext": {NS: {"id": "judge-a"}}}, "observed_at": AT, "tier": "reported",
        "ext": {NS: {"component": "overall", "step": "final", "call": "pc1", "shown_first": "b",
                     "rubric": "brief@1"}}}
    tie = A.preference({**pair, "verdict": "tie"}, "eval.case-01")
    assert tie["winner"] == "tie"


def test_levels_items_counts_and_a_time_ocp_does_not_write_come_back_unchanged():
    runs = [run_row("r1"), run_row("r2", arm="strong", family="alpha")]
    raw = [
        {"id": "s1", "run": "r1", "component": "clarity", "judge": "judge-a", "call": "k1", "value": "good"},
        {"id": "s2", "run": "r1", "component": "tone", "judge": "judge-a", "call": "k1", "value": 3},
        {"id": "s3", "run": "r1", "component": "facts", "judge": "judge-a", "call": "k2", "value": 1, "item": "i1"},
        {"id": "s4", "run": "r1", "component": "facts", "judge": "judge-a", "call": "k2", "value": 0, "item": "i2"},
        {"id": "s5", "run": "r2", "component": "facts", "judge": "judge-a", "call": "k3", "value": 7, "of": 9},
        {"id": "s6", "run": "r2", "component": "clarity", "judge": "judge-a", "call": "k4", "value": "ok",
         "observed_at": "2026-02-03T18:00:00Z", "ext": {"org.example.notes": {"batch": 4}}},
        {"id": "s7", "run": "r2", "component": "depth", "judge": "judge-a", "call": "k5", "value": 2.5,
         "observed_at": "2026-02-03 10:00:00+00:00"},
    ]
    scores = [C.validate_score({"schema": C.SCHEMA_SCORES, "step": "final", "rubric": "brief@1", "observed_at": AT,
                                **row}, "scores.jsonl", "") for row in raw]
    signals = by_id([A.signal(row) for row in scores])
    assert (signals["s1"]["value"], signals["s1"]["ext"][NS]["level"]) == (None, "good")
    assert signals["s2"]["value"] == 3 and "level" not in signals["s2"]["ext"][NS]
    assert signals["s3"]["ext"][NS]["item"] == "i1" and signals["s5"]["ext"][NS]["of"] == 9
    assert signals["s7"]["observed_at"] == "2026-02-03T10:00:00+00:00"
    assert signals["s7"]["ext"][NS]["observed_at"] == "2026-02-03 10:00:00+00:00"
    assert signals["s6"]["ext"][NS]["ext"] == {"org.example.notes": {"batch": 4}}
    batch, _ = through_ocp(runs, scores, [])
    assert by_id(scores_of(batch)) == by_id(scores)


def test_ocp_to_study_and_back_gives_the_same_signals_and_preferences():
    """A document made elsewhere, with members the tables have no column for, is written back as it came."""
    a, b = run_row("r1"), run_row("r2", arm="strong", family="alpha")
    slate = {"id": "slate-7", "members": ["r1", "r2"], "blinded": True}
    signals = [
        {"id": "g1", "kind": "score", "name": "depth", "value": 4, "observed_at": AT, "tier": "verified",
         "unit": "points", "at_attempt": "att-1", "source": {"kind": "judge", "ref": "judge-a", "model": "model-x"},
         "ext": {NS: {"call": "k1", "step": "final", "rubric": "brief@1", "later": True},
                 "org.example.grader": {"queue": "night"}}},
        {"id": "g2", "kind": "score", "name": "clarity", "value": None, "observed_at": AT, "tier": "reported",
         "source": {"kind": "user", "ref": "reviewer"},
         "ext": {NS: {"call": "k2", "step": "final", "rubric": "brief@1", "level": "ok", "replaces": "g0"}}},
        {"id": "g0", "kind": "score", "name": "clarity", "value": None, "observed_at": AT, "tier": "reported",
         "source": {"kind": "user", "ref": "reviewer"},
         "ext": {NS: {"call": "k0", "step": "final", "rubric": "brief@1", "level": "poor"}}},
    ]
    preference = {"id": "f1", "slate": "slate-7", "winner": "r2", "members": ["r1", "r2"], "observed_at": AT,
                  "tier": "verified", "judge": {"kind": "referee", "model": "model-x", "blinded": True,
                                                "ext": {NS: {"id": "judge-a"}}},
                  "ext": {NS: {"component": "overall", "step": "final", "call": "pk1", "shown_first": "a",
                               "rubric": "brief@1"}, "org.example.grader": {"queue": "night"}}}
    doc_a = document(a, signals=signals, preferences=[preference], slate=slate)
    doc_b = document(b, preferences=[preference], slate=slate)
    assert errors(doc_a) == [] and errors(doc_b) == []
    batch = A.read_documents([("a.json", doc_a), ("b.json", doc_b)])
    assert batch.not_used == [] and len(batch.pairs) == 1  # the copy in the second file is read once
    runs = [C.validate_run({**row, "schema": C.SCHEMA_RUNS}, "runs.jsonl", "") for row in batch.runs]
    scores, pairs = scores_of(batch), pairs_of(batch)
    assert by_id(scores)["g1"]["ext"][KEPT] == {
        "tier": "verified", "unit": "points", "at_attempt": "att-1", "source": {"kind": "judge", "model": "model-x"},
        "ext": {"org.example.grader": {"queue": "night"}}, "eval": {"later": True}}
    assert by_id(scores)["g2"]["value"] == "ok" and by_id(scores)["g2"]["replaces"] == "g0"
    assert pairs[0]["verdict"] == "b" and pairs[0]["judge"] == "judge-a"
    entries, not_used = A.export(runs, scores, pairs, persons=["reviewer"])
    assert not_used == []
    out = {entry["run"]: entry for entry in entries}
    assert by_id(out["r1"]["signals"]) == by_id(signals)
    assert out["r1"]["preferences"] == [preference] == out["r2"]["preferences"]
    assert out["r1"]["slate"] == slate == out["r2"]["slate"]
    again = A.read_documents([("a.json", A.merge(document(a), out["r1"])), ("b.json", A.merge(document(b), out["r2"]))])
    assert by_id(scores_of(again)) == by_id(scores) and pairs_of(again) == pairs


# ---------------------------------------------------------------------------------------------------
# Records made by other tools

def test_signals_with_no_eval_member_the_newest_is_the_reading_and_earlier_ones_are_corrections():
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)

    def score(ident, name, value, minute, judge="judge-a", kind="judge"):
        return {"id": ident, "kind": "score", "name": name, "value": value, "tier": "reported",
                "observed_at": f"2026-02-03T10:{minute:02d}:00-08:00", "source": {"kind": kind, "ref": judge}}

    signals = [score("n3", "depth", 7, 30), score("n1", "depth", 5, 10), score("n2", "depth", 6, 20),
               score("m1", "depth", 4, 5, judge="judge-b"), score("x1", "speed", 3, 5),
               {"id": "t1", "kind": "score", "name": "depth", "value": 9, "tier": "verified", "observed_at": AT,
                "source": {"kind": "test"}},
               {"id": "v1", "kind": "verdict", "name": "tests", "value": "pass", "tier": "verified",
                "observed_at": AT}]
    doc = document(run_row("r1"), signals=signals)
    assert errors(doc) == []
    with_rubric = A.read_documents([("r1.json", doc)], rubric=rubric)
    rows = by_id(scores_of(with_rubric))
    # a signal id of another tool is unique in its document only: the reading is named within its run
    assert set(rows) == {"r1.n1", "r1.n2", "r1.n3", "r1.m1"}
    assert (rows["r1.n2"]["replaces"], rows["r1.n3"]["replaces"]) == ("r1.n1", "r1.n2")
    assert "replaces" not in rows["r1.n1"]
    assert [row["id"] for row in C.live(list(rows.values()), "scores.jsonl")[0]] == ["r1.n3", "r1.m1"]
    assert rows["r1.n3"] == {"schema": C.SCHEMA_SCORES, "id": "r1.n3", "run": "r1", "step": "final",
                             "component": "depth", "judge": "judge-a", "call": "r1.n3", "value": 7,
                             "rubric": "brief@1", "observed_at": "2026-02-03T10:30:00-08:00", "replaces": "r1.n2",
                             "ext": {KEPT: {"id": "n3"}}}
    assert with_rubric.not_used == [
        {"source": "r1.json", "kind": "signal", "id": "x1", "reason": '"speed" is not a component of the rubric'},
        {"source": "r1.json", "kind": "signal", "id": "t1",
         "reason": "a score with no judge: source.kind is not judge or user, or source.ref is missing"}]
    bare = A.read_documents([("r1.json", doc)])
    assert bare.scores == [] and {entry["reason"] for entry in bare.not_used if entry["id"] != "t1"} == {
        "the signal names no rubric; give --rubric"}


def test_two_documents_of_another_tool_may_use_the_same_signal_ids():
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)

    def score(ident, value, minute):
        return {"id": ident, "kind": "score", "name": "depth", "value": value, "tier": "reported",
                "observed_at": f"2026-02-03T10:{minute:02d}:00-08:00", "source": {"kind": "judge", "ref": "judge-a"}}

    # each document numbers its signals from s1; in both, s2 is a later score of the same judge and component
    runs = [run_row("r1"), run_row("r2", arm="strong"), run_row("team/r 3", input_id="case-02")]
    docs = [document(runs[0], signals=[score("s1", 5, 10), score("s2", 6, 20)]),
            document(runs[1], signals=[score("s1", 8, 10), score("s2", 9, 20)]),
            document(runs[2], signals=[score("s1", 2, 10)])]
    assert [errors(doc) for doc in docs] == [[], [], []]
    named = [(f"d{n}.json", doc) for n, doc in enumerate(docs)]
    batch = A.read_documents(named, rubric=rubric)
    rows = scores_of(batch)
    assert batch.not_used == [] and len(rows) == 5 and len({row["id"] for row in rows}) == 5
    # by hand: run, signal id and value of every reading, the call, and what each later score corrects. The
    # "/" and the space of "team/r 3" are written %2F and %20 (0.4.4 wrote "_", so a run "team_r_3" gave the
    # same id)
    assert [(row["run"], row["id"], row["call"], row["value"], row.get("replaces")) for row in rows] == [
        ("r1", "r1.s1", "r1.s1", 5, None), ("r1", "r1.s2", "r1.s2", 6, "r1.s1"),
        ("r2", "r2.s1", "r2.s1", 8, None), ("r2", "r2.s2", "r2.s2", 9, "r2.s1"),
        ("team/r 3", "team%2Fr%203.s1", "team%2Fr%203.s1", 2, None)]
    assert [(row["id"], row["value"]) for row in C.live(rows, "scores.jsonl")[0]] == [
        ("r1.s2", 6), ("r2.s2", 9), ("team%2Fr%203.s1", 2)]
    # the same documents again give the same rows: the ids do not depend on what was read before
    assert scores_of(A.read_documents(list(reversed(named)), rubric=rubric)) == [
        *rows[4:], *rows[2:4], *rows[:2]]
    # on the way out each signal has the id its document gave it, and the reading's id beside it
    entries, not_used = A.export(runs, rows, [])
    assert not_used == [] and [[s["id"] for s in entry["signals"]] for entry in entries] == [
        ["s1", "s2"], ["s1", "s2"], ["s1"]]
    assert entries[1]["signals"][1]["ext"][NS] == {"id": "r2.s2", "call": "r2.s2", "step": "final",
                                                    "rubric": "brief@1", "replaces": "r2.s1"}
    for entry, doc in zip(entries, docs):
        for out, given in zip(entry["signals"], doc["run"]["signals"]):
            assert {k: v for k, v in out.items() if k != "ext"} == given
    # and back in, with the eval member now on every signal: the same five readings, member for member
    merged = [A.merge(document(run), entry) for run, entry in zip(runs, entries)]
    assert [errors(doc) for doc in merged] == [[], [], []]
    again = A.read_documents([(f"m{n}.json", doc) for n, doc in enumerate(json.loads(json.dumps(merged)))])
    assert scores_of(again) == rows


def test_a_preference_with_no_eval_member_is_read_on_the_named_component_or_listed_as_not_used():
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)
    runs = [run_row("r1"), run_row("r2", arm="strong"), run_row("r3", input_id="case-02")]
    prefs = [
        {"id": "f1", "slate": "sl-1", "winner": "r1", "observed_at": AT,
         "judge": {"kind": "model", "model": "model x"}},
        {"id": "f2", "slate": "sl-1", "winner": "tie", "members": ["r1", "r2"]},
        {"id": "f3", "slate": "sl-1", "winner": "r1", "members": ["r1", "r2", "r9"]},
        {"id": "f4", "slate": "sl-1", "winner": "r1", "members": ["r1", "r3"]},
        {"id": "f5", "slate": "sl-1", "winner": "r1", "members": ["r1", "r8"]},
    ]
    slate = {"id": "sl-1", "members": ["r1", "r2"]}
    docs = [("r1.json", document(runs[0], preferences=prefs, slate=slate, started_at="2026-02-03T09:00:00-08:00")),
            ("r2.json", document(runs[1], preferences=prefs[:2], slate=slate)), ("r3.json", document(runs[2]))]
    batch = A.read_documents(docs, rubric=rubric, pair_component="depth")
    rows = by_id(pairs_of(batch))
    # a preference id of another tool is unique in its slate: the pair is named within the slate, and the
    # copies of f1 and f2 in r2.json are the same two pairs
    assert set(rows) == {"sl-1.f1", "sl-1.f2"} and len(batch.pairs) == 2
    assert rows["sl-1.f1"] == {"schema": C.SCHEMA_PAIRS, "id": "sl-1.f1", "a": "r1", "b": "r2", "step": "final",
                               "component": "depth", "judge": "model_x", "call": "sl-1.f1", "shown_first": None,
                               "verdict": "a", "rubric": "brief@1", "observed_at": AT,
                               "ext": {KEPT: {"judge": {"kind": "model", "model": "model x"}, "slate": "sl-1",
                                              "id": "f1"}}}
    assert rows["sl-1.f2"]["verdict"] == "tie" and rows["sl-1.f2"]["observed_at"] == "2026-02-03T09:00:00-08:00"
    assert rows["sl-1.f2"]["judge"] == "referee"
    assert {entry["id"]: entry["reason"] for entry in batch.not_used} == {
        "f3": "the preference does not name exactly two runs",
        "f4": "the two runs have different inputs",
        "f5": "the preference does not resolve to two runs of the study"}
    overall = A.read_documents(docs, rubric=rubric)
    assert pairs_of(overall)[0]["component"] is None  # read on the overall component
    assert all(entry["reason"] == "the preference names no rubric; give --rubric"
               for entry in A.read_documents(docs).not_used if entry["id"] in ("f1", "f2"))
    known = A.read_documents(docs[:1], rubric=rubric, inputs={"r2": "case-01"})  # r2 is already in the study
    assert set(by_id(pairs_of(known))) == {"sl-1.f1", "sl-1.f2"}


def test_scoped_ids_are_one_to_one_and_an_independent_decoder_reads_both_ids_back():
    # by hand: the collision of review 2 item 10, and the characters that are written as %XX
    assert (A.scoped_id("a.b", "c"), A.scoped_id("a", "b.c")) == ("a%2Eb.c", "a.b.c")
    assert A.scoped_id("r1", "s1") == "r1.s1" and A.scoped_id("run-7_x", "sig.3") == "run-7_x.sig.3"
    assert A.scoped_id("team/r 3", "s1") == "team%2Fr%203.s1" and A.scoped_id("team_r_3", "s1") == "team_r_3.s1"
    assert A.scoped_id("50%", "a@b") == "50%25.a%40b" and A.scoped_id("x", "%2E") == "x.%252E"
    assert A.scoped_id("x　y", "é\tz") == "x%E3%80%80y.é%09z"  # per UTF-8 byte; letters stay
    # every pair over a small alphabet of the troublesome characters: the scope ends at the first "." and
    # urllib's percent decoding gives both ids back, so no two pairs share an id
    chars = ["a", ".", "%", "2", "E", " ", "/", "@", "　"]
    texts = chars + [x + y for x in chars for y in chars] + [
        x + y + z for x in chars[:5] for y in chars[:5] for z in chars[:5]]
    seen = {}
    for scope in texts:
        for record in texts:
            ident = A.scoped_id(scope, record)
            head, dot, tail = ident.partition(".")
            assert dot and (unquote(head), unquote(tail)) == (scope, record), (scope, record, ident)
            assert not any(ch.isspace() or ch in "/@" for ch in ident)
            assert seen.setdefault(ident, (scope, record)) == (scope, record)
    assert len(seen) == len(texts) ** 2 == 46225


def test_runs_a_dot_b_and_a_with_dotted_signal_ids_give_four_readings_and_their_corrections():
    """Review 2 item 10: run "a.b" with signals c, d and run "a" with signals b.c, b.d gave the ids a.b.c and
    a.b.d twice in 0.4.4."""
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)

    def score(ident, value, minute):
        return {"id": ident, "kind": "score", "name": "depth", "value": value, "tier": "reported",
                "observed_at": f"2026-02-03T10:{minute:02d}:00-08:00", "source": {"kind": "judge", "ref": "judge-a"}}

    runs = [run_row("a.b"), run_row("a", arm="strong")]
    docs = [document(runs[0], signals=[score("d", 6, 20), score("c", 5, 10)]),
            document(runs[1], signals=[score("b.c", 8, 10), score("b.d", 9, 20)])]
    assert [errors(doc) for doc in docs] == [[], []]
    rows = scores_of(A.read_documents([("ab.json", docs[0]), ("a.json", docs[1])], rubric=rubric))
    # by hand: the later score of each run corrects the earlier one of the same run, by its new id
    assert [(row["run"], row["id"], row["call"], row["value"], row.get("replaces"), row["ext"][KEPT]["id"])
            for row in rows] == [
        ("a.b", "a%2Eb.c", "a%2Eb.c", 5, None, "c"), ("a.b", "a%2Eb.d", "a%2Eb.d", 6, "a%2Eb.c", "d"),
        ("a", "a.b.c", "a.b.c", 8, None, "b.c"), ("a", "a.b.d", "a.b.d", 9, "a.b.c", "b.d")]
    assert [row["id"] for row in C.live(rows, "scores.jsonl")[0]] == ["a%2Eb.d", "a.b.d"]
    # out with the ids each document gave, the reading ids and corrections beside them, and back in unchanged
    entries, not_used = A.export(runs, rows, [])
    assert not_used == [] and [[(s["id"], s["ext"][NS]["id"], s["ext"][NS].get("replaces")) for s in entry["signals"]]
                               for entry in entries] == [
        [("c", "a%2Eb.c", None), ("d", "a%2Eb.d", "a%2Eb.c")], [("b.c", "a.b.c", None), ("b.d", "a.b.d", "a.b.c")]]
    merged = json.loads(json.dumps([A.merge(document(run), entry) for run, entry in zip(runs, entries)]))
    assert [errors(doc) for doc in merged] == [[], []]
    assert scores_of(A.read_documents([("m0.json", merged[0]), ("m1.json", merged[1])])) == rows


def test_preferences_of_another_tool_are_named_within_their_slate():
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)
    runs = [run_row("r1"), run_row("r2", arm="strong"), run_row("r3", input_id="case-02"),
            run_row("r4", input_id="case-02", arm="strong")]

    def pref(winner, minute):
        return {"id": "f1", "slate": "sl-1" if winner in ("r1", "r2") else "sl.2", "winner": winner,
                "observed_at": f"2026-02-03T10:{minute:02d}:00-08:00", "judge": {"kind": "model", "model": "m"}}

    one, two = pref("r2", 1), pref("r3", 2)  # both tools number their slate's preferences from f1
    slates = [{"id": "sl-1", "members": ["r1", "r2"]}, {"id": "sl.2", "members": ["r3", "r4"]}]
    docs = [(f"{run['run']}.json", document(run, preferences=[p], slate=s))
            for run, p, s in zip(runs, (one, one, two, two), (slates[0], slates[0], slates[1], slates[1]))]
    batch = A.read_documents(docs, rubric=rubric)
    rows = pairs_of(batch)
    # by hand: one pair per slate, read once from the two files that hold it; "." in a slate is %2E
    assert batch.not_used == [] and [(row["id"], row["call"], row["a"], row["b"], row["verdict"],
                                      row["ext"][KEPT]["id"]) for row in rows] == [
        ("sl-1.f1", "sl-1.f1", "r1", "r2", "b", "f1"), ("sl%2E2.f1", "sl%2E2.f1", "r3", "r4", "a", "f1")]
    entries, not_used = A.export([C.validate_run({**row, "schema": C.SCHEMA_RUNS}, "runs.jsonl", "")
                                  for row in batch.runs], [], rows)
    out = {entry["run"]: entry for entry in entries}
    assert not_used == [] and out["r1"]["preferences"] == out["r2"]["preferences"]
    assert [(p["id"], p["slate"], p["winner"], p["ext"][NS]["id"]) for p in out["r1"]["preferences"] + out["r3"][
        "preferences"]] == [("f1", "sl-1", "r2", "sl-1.f1"), ("f1", "sl.2", "r3", "sl%2E2.f1")]
    for entry, (_, doc) in zip(entries, docs):  # what the other tool wrote comes back as it was
        given, back = doc["run"]["preferences"][0], entry["preferences"][0]
        assert {k: back[k] for k in given if k != "judge"} == {k: v for k, v in given.items() if k != "judge"}
        assert {k: v for k, v in back["judge"].items() if k != "ext"} == given["judge"]
    merged =json.loads(json.dumps([A.merge(document(run), out[run["run"]]) for run in runs]))
    assert pairs_of(A.read_documents([(f"m{n}.json", doc) for n, doc in enumerate(merged)])) == rows
    # one id in one slate with two contents: both come through, and the import refuses the second by id
    other = dict(one, winner="r1")
    clash = A.read_documents([("r1.json", document(runs[0], preferences=[one], slate=slates[0])),
                              ("r2.json", document(runs[1], preferences=[other], slate=slates[0]))], rubric=rubric)
    assert [(row["id"], row["verdict"]) for row in pairs_of(clash)] == [("sl-1.f1", "b"), ("sl-1.f1", "a")]


# ---------------------------------------------------------------------------------------------------
# Run rows from documents

def test_a_run_row_reads_the_task_the_arm_the_settings_the_cost_and_the_simulated_mark():
    doc = document(run_row("r1"))
    doc["run"]["flow"] = {"id": "f", "version": 1, "digest": "sha256:" + "0" * 64, "arm": "strong", "split": "explore"}
    doc["run"]["ext"] = {SIMULATE_KEY: {"seed": 1}}
    doc["run"]["configuration"]["settings"]["final"] = {"harness": "cli", "model": {"raw": "Vendor/Model-Large@2"},
                                                        "ext": {NS: {"variant": "terse"}}}
    doc["attempts"] = [{"id": "a1", "cost": {"usd": 0.25}}, {"id": "a2", "cost": {"usd": 0.5}}, {"id": "a3"}]
    huge = dict(doc, attempts=[{"id": "a1", "cost": {"usd": 10 ** 400}}])
    assert "cost_usd" not in A.run_row(huge, "r1.json")  # a cost no number can hold is left out, not raised
    assert A.run_row(doc, "r1.json") == {
        "run": "r1", "input": "case-01", "arm": "strong", "split": "explore", "cost_usd": 0.75, "simulated": True,
        "steps": [{"step": "final", "model": {"id": "Vendor/Model-Large@2", "family": "vendor"}, "variant": "terse"}]}
    C.validate_run({**A.run_row(doc, "r1.json"), "schema": C.SCHEMA_RUNS}, "runs.jsonl", "")
    del doc["run"]["flow"], doc["run"]["labels"]
    doc["run"]["configuration"]["id"] = "cfg-1"
    assert A.run_row(doc, "r1.json")["arm"] == "cfg-1"


def test_steps_come_from_the_attempts_when_the_run_has_no_configuration():
    doc = document(run_row("r1"))
    del doc["run"]["configuration"]
    doc["attempts"] = [{"id": "a1", "node": "draft", "model": {"id": "model-small", "family": "beta"}},
                       {"id": "a2", "node": "final", "model": "model-large", "effort": "high"},
                       {"id": "a3", "node": "draft", "model": {"id": "model-mid", "family": "beta"}}]
    assert A.run_row(doc, "r1.json")["steps"] == [
        {"step": "final", "model": {"id": "model-large", "family": "model"}, "effort": "high"},
        {"step": "draft", "model": {"id": "model-mid", "family": "beta"}}]


@pytest.mark.parametrize("edit, message", [
    (lambda doc: doc["run"].pop("labels"),
     'r1.json: run: run "r1" names no arm; state run.flow.arm, run.labels.arm or a configuration id'),
    (lambda doc: doc["run"].pop("configuration"),
     'r1.json: run: run "r1" has no step settings; give run.configuration.settings, or attempts with their models'),
    (lambda doc: doc["run"].pop("task"),
     "r1.json: run: the document has no run.id or no run.task.id; give an OCP run document at version 0.3 or "
     "later, with its task"),
], ids=["no-arm", "no-steps", "no-task"])
def test_a_run_document_that_cannot_give_a_run_row_is_refused(edit, message):
    doc = document(run_row("r1"))
    edit(doc)
    with pytest.raises(C.ContractError) as caught:
        A.run_row(doc, "r1.json")
    assert str(caught.value) == message


def test_a_document_that_is_not_a_run_or_breaks_ocp_is_refused_by_file_and_place():
    with pytest.raises(C.ContractError) as caught:
        A.read_documents([("flow.json", {"ocp": "0.4", "kind": "flow", "flow": {}})])
    assert str(caught.value) == "flow.json: not an OCP run document; give run documents, one per file"
    doc = document(run_row("r1"), signals=[{"id": "g1", "kind": "score", "name": "depth", "value": "high",
                                             "observed_at": AT, "tier": "reported"}])
    with pytest.raises(C.ContractError) as caught:
        A.read_documents([("r1.json", doc)])
    assert caught.value.code == "bad_ocp" and caught.value.file == "r1.json"
    assert "E195" in str(caught.value) and "loopmath ocp validate" in str(caught.value)


# ---------------------------------------------------------------------------------------------------
# harness-runs rows

def harness_row(**more) -> dict:
    return {"schema": "loopmath.harness-runs/0.1", "task_id": "case-01", "arm": "baseline", "attempt": 1,
            "harness": "cli", "model": "model-small", "effort": "low", "started_at": "2026-02-03T09:00:00-08:00",
            "ended_at": "2026-02-03T09:05:00-08:00", "cost": {"usd": 0.2},
            "outcome": {"accepted": True}, **more}


def test_harness_rows_give_runs_and_readings(tmp_path):
    rubric = C.read_document(STUDY / "rubric.json", C.SCHEMA_RUBRIC)
    rows = [
        harness_row(outcome={"accepted": True, "judges": [
            {"id": "judge-a", "score": 6, "observed_at": AT},
            {"id": "judge-a", "score": 1, "observed_at": AT,
             "ext": {NS: {"component": "numbers_right", "call": "k1", "id": "h1", "rubric": "brief@1"}}},
            {"id": "judge-b", "score": None, "observed_at": AT,
             "ext": {NS: {"component": "clarity", "level": "good", "call": "k2", "rubric": "brief@1"}}},
            {"id": "judge-b", "score": None}]}),
        harness_row(arm="strong", model="model-large", ext={SIMULATE_KEY: {"seed": 2}}),
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    batch = A.read_rows(path, rubric=rubric)
    runs = [C.validate_run({**row, "schema": C.SCHEMA_RUNS}, "runs.jsonl", "") for row in batch.runs]
    assert [(row["input"], row["arm"], row["simulated"], row["cost_usd"]) for row in runs] == [
        ("case-01", "baseline", False, 0.2), ("case-01", "strong", True, 0.2)]
    assert runs[0]["steps"] == [{"step": "final", "model": {"id": "model-small", "family": "model"}, "effort": "low"}]
    name = runs[0]["run"]
    scores = by_id(scores_of(batch))
    assert set(scores) == {f"{name}:judge:0", "h1", f"{name}:judge:2"}
    assert scores[f"{name}:judge:0"]["component"] == "overall" and scores[f"{name}:judge:0"]["value"] == 6
    assert (scores["h1"]["component"], scores["h1"]["call"], scores["h1"]["value"]) == ("numbers_right", "k1", 1)
    assert scores[f"{name}:judge:2"]["value"] == "good"
    assert batch.not_used == [{"source": "rows.jsonl line 1", "kind": "signal", "id": f"{name}:judge:3",
                               "reason": "a declared score with no value yet"}]
    assert batch.source[("runs", name)] == "rows.jsonl line 1"
    assert A.read_rows(path).scores and all(  # without a rubric only the readings that name theirs are read
        row["rubric"] == "brief@1" for row in A.read_rows(path).scores)


def test_a_harness_row_that_breaks_its_format_or_names_no_arm_is_refused(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"schema": "loopmath.harness-runs/0.1", "arm": "baseline"}) + "\n", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        A.read_rows(path)
    assert caught.value.code == "bad_rows" and str(caught.value).startswith("rows.jsonl: ")
    row = harness_row()
    del row["arm"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(C.ContractError) as caught:
        A.read_rows(path)
    assert caught.value.code in ("no_arm", "bad_rows")


# ---------------------------------------------------------------------------------------------------
# Export entries

def test_export_names_what_it_cannot_place_and_keeps_to_the_runs_asked_for():
    study = C.read_study(STUDY)
    entries, not_used = A.export(study.runs, study.scores, study.pairs, only=["strong-01", "nobody", "strong-01"])
    assert [entry["run"] for entry in entries] == ["strong-01"]
    assert not_used == [{"kind": "run", "id": "nobody", "reason": "no such run in the study"}]
    assert entries[0]["slate"] == {"id": "eval.case-01", "members": ["baseline-01", "strong-01"]}
    assert {p["id"] for p in entries[0]["preferences"]} == {
        row["id"] for row in study.pairs if "strong-01" in (row["a"], row["b"])}
    stray = dict(study.scores[0], id="s9999", run="gone")
    _, not_used = A.export(study.runs, [stray], [dict(study.pairs[0], b="gone")])
    assert not_used == [{"kind": "reading", "id": "s9999", "reason": 'no run "gone" in the study'},
                        {"kind": "pair", "id": "p001", "reason": 'no run "gone" in the study'}]


def test_merge_adds_only_what_the_document_does_not_hold():
    study = C.read_study(STUDY)
    entries, _ = A.export(study.runs, study.scores, study.pairs, only=["baseline-01"])
    doc = document(study.runs[0])
    once = A.merge(doc, entries[0])
    assert doc["run"]["signals"] == [] and "preferences" not in doc["run"]  # the input is not changed
    assert A.merge(once, entries[0]) == once
    assert len(once["run"]["signals"]) == len(entries[0]["signals"]) and once["run"]["slate"] == entries[0]["slate"]
    with pytest.raises(ValueError, match="the entry is for run"):
        A.merge(document(study.runs[1]), entries[0])


def test_merge_keeps_one_preference_id_apart_on_two_slates_and_refuses_one_off_the_runs_slate():
    run = run_row("r1")

    def entry(on: str, other: str, **more) -> dict:
        record = {"id": "f1", "slate": on, "winner": "r1", "members": ["r1", other], "observed_at": AT,
                  "judge": {"kind": "model"}}
        return {"run": "r1", "signals": [], "preferences": [record], **more}

    one = entry("slate-1", "r2", slate={"id": "slate-1", "members": ["r1", "r2"]})
    two = entry("slate-2", "r3", slates=[{"id": "slate-2", "members": ["r1", "r3"]}])
    # no slate in the document: both preferences are kept, and the run gets no slate, as neither covers both
    doc = A.merge(A.merge(document(run), two), one)
    assert [(p["slate"], p["id"]) for p in doc["run"]["preferences"]] == [("slate-2", "f1"), ("slate-1", "f1")]
    assert "slate" not in doc["run"] and errors(doc) == []
    assert A.merge(doc, one) == doc and A.merge(doc, two) == doc  # by slate and id: nothing twice
    # a document on slate-1 takes slate-1's preference and refuses slate-2's by name; OCP finds E194 in it
    on_one = A.merge(document(run), one)
    assert on_one["run"]["slate"] == one["slate"] and errors(on_one) == []
    with pytest.raises(ValueError, match=r"^the preference 'f1' is on the slate 'slate-2', and the document gives the "
                                         r"run 'r1' the slate 'slate-1' \(OCP E194\)"):
        A.merge(on_one, two)
    forced = copy.deepcopy(on_one)
    forced["run"]["preferences"] += two["preferences"]
    assert [line.split(" ")[0] for line in errors(forced)] == ["E194"]


def test_the_adapter_is_not_a_command_and_hands_over_to_scores(capsys):
    import argparse

    code = A.run(argparse.Namespace(eval_command="scores export", study=str(STUDY), run=["baseline-01"], json=False))
    out = capsys.readouterr().out
    assert code == 0 and json.loads(out)["run"] == "baseline-01"
