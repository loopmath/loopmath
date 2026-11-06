"""The prior bundle reads a run's signals as the belief model does: the share reduction, the run summary and the
bundle's own checks skip a signal that is not an object and read signals that are not a list as none, so none of
them raises. The OCP checker names such a run (E010): a strict build refuses the bundle on it, a lenient build
lists it, and the reduction leaves the member where it was."""

from __future__ import annotations

import copy
import gzip
import json
import re

import pytest

from loopmath.priors.build import BundleError, SourceResult, build_bundle
from loopmath.priors.reduce import reduce_for_bundle
from loopmath.priors.show import run_summary
from loopmath.priors.validate import local_checks, validate_bundle_doc

from test_priors_bundle import _sweep_docs  # noqa: E402  (same folder)

MEMBERS = {"text": ["text"], "number": [5], "null": [None], "list": [[]], "true": [True], "two": ["x", 7]}
CONTAINERS = {"object": {"a": 1}, "empty-object": {}, "text": "xy", "empty-text": "", "number": 5, "zero": 0,
              "null": None, "true": True, "false": False}


def _members(doc: dict, shape: str) -> dict:
    """`doc` with the signals of `MEMBERS[shape]`, which are not objects, before its own and again after them."""
    out = copy.deepcopy(doc)
    bad = MEMBERS[shape]
    out["run"]["signals"] = [*copy.deepcopy(bad), *out["run"]["signals"], *copy.deepcopy(bad)]
    return out


def _signals(doc: dict, signals: object) -> dict:
    """`doc` with a copy of `signals` as its `run.signals`."""
    out = copy.deepcopy(doc)
    out["run"]["signals"] = copy.deepcopy(signals)
    return out


def _review_run(value: str) -> dict:
    """The shape of the run that found the raise: one run whose rule requires a `review` verdict."""
    review = {"id": "s1", "kind": "verdict", "name": "review", "value": value, "tier": "reported",
              "observed_at": "2026-11-04T00:00:00-08:00"}
    return {"run": {"id": "r1", "task": {"id": "t1", "type": "coding", "source": {"kind": "sweep"}},
                    "configuration": {"workflow": {"id": "w1"}}, "acceptance_rule": {"requires": ["review"]},
                    "signals": [review]}, "attempts": []}


def _score_run(values: tuple) -> dict:
    """A run whose rule is a score: its last value against the target decides."""
    scores = [{"id": f"s{i}", "kind": "score", "name": "tests_pass_fraction", "value": v, "tier": "reported",
               "observed_at": f"2026-11-04T00:0{i}:00-08:00"} for i, v in enumerate(values)]
    doc = _signals(_review_run("pass"), scores)
    doc["run"]["acceptance_rule"] = {"score": {"name": "tests_pass_fraction", "target": 0.5, "better": "higher"}}
    return doc


def _lines(folder) -> dict:
    with gzip.open(folder / "sweep.jsonl.gz", "rb") as fh:
        return {json.loads(line)["run"]["id"]: line for line in fh.read().splitlines()}


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_the_run_summary_reads_a_verdict_past_a_signal_that_is_not_an_object(shape):
    for value, outcome in (("pass", "accepted"), ("reject", "not accepted")):
        clean = _review_run(value)
        assert run_summary([clean])["outcomes"] == {outcome: 1}
        assert run_summary([_members(clean, shape)]) == run_summary([clean])
    assert run_summary([_signals(_review_run("pass"), MEMBERS[shape])])["outcomes"] == {"unknown": 1}
    assert run_summary([_signals(_review_run("pass"), [])])["outcomes"] == {"unknown": 1}


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_the_run_summary_reads_a_score_past_a_signal_that_is_not_an_object(shape):
    for values, outcome in (((0.2, 0.9), "accepted"), ((0.9, 0.2), "not accepted")):
        clean = _score_run(values)
        assert run_summary([clean])["outcomes"] == {outcome: 1}
        assert run_summary([_members(clean, shape)]) == run_summary([clean])
    assert run_summary([_signals(_score_run((0.9,)), MEMBERS[shape])])["outcomes"] == {"unknown": 1}


@pytest.mark.parametrize("shape", sorted(CONTAINERS))
def test_the_run_summary_reads_signals_that_are_not_a_list_as_none(shape):
    """Both walks: the verdicts a rule requires, and the scores of a rule with a score and no requires."""
    runs = [(_review_run("pass"), "accepted"), (_review_run("reject"), "not accepted"),
            (_score_run((0.2, 0.9)), "accepted"), (_score_run((0.9, 0.2)), "not accepted")]
    for doc, outcome in runs:
        assert run_summary([doc])["outcomes"] == {outcome: 1}
        assert run_summary([_signals(doc, CONTAINERS[shape])])["outcomes"] == {"unknown": 1}


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_the_reduction_keeps_a_signal_that_is_not_an_object_where_it_was(shape):
    clean = _sweep_docs()[0]
    clean["run"]["signals"][-1]["source"] = {"kind": "orchestrator", "ref": "~/notes/referee.txt"}
    want = reduce_for_bundle(clean, salt="s")
    assert want["run"]["signals"][-1]["source"] == {"kind": "orchestrator"}
    assert reduce_for_bundle(_members(clean, shape), salt="s") == _members(want, shape)


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_the_checks_name_each_signal_that_is_not_an_object(shape):
    clean = reduce_for_bundle(_sweep_docs()[0], salt="s")
    assert validate_bundle_doc(clean)[0] == []
    bad = _members(clean, shape)
    n, total = len(MEMBERS[shape]), len(bad["run"]["signals"])
    assert local_checks(bad) == []
    found = sorted((f["level"], f["code"], f["path"]) for f in validate_bundle_doc(bad)[0])
    at = [*range(n), *range(total - n, total)]  # the members at both ends, one finding each
    assert found == sorted(("error", "E010", f"$['run']['signals'][{i}]") for i in at)


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_a_local_finding_keeps_its_index_past_a_skipped_signal(shape):
    clean = reduce_for_bundle(_sweep_docs()[0], salt="s")
    good = clean["run"]["signals"]
    good[-1]["id"] = good[0]["id"]  # a duplicate id: a local check (L006), not the OCP checker's
    n = len(MEMBERS[shape])
    at = n + len(good) - 1
    assert [(f["code"], f["path"]) for f in local_checks(clean)] == [("L006", f"$.run.signals[{len(good) - 1}]")]
    doc = _signals(clean, [*MEMBERS[shape], *good])
    assert local_checks(doc) == [{"level": "error", "code": "L006", "path": f"$.run.signals[{at}]",
                                  "message": "duplicate signal id"}]
    found = sorted((f["code"], f["path"]) for f in validate_bundle_doc(doc)[0])
    assert found == sorted([*(("E010", f"$['run']['signals'][{i}]") for i in range(n)),
                            ("L006", f"$.run.signals[{at}]")])


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_a_strict_build_refuses_the_bundle_and_names_the_run(tmp_path, shape):
    docs = _sweep_docs()
    docs[0] = _members(docs[0], shape)
    named = f"sweep: {docs[0]['run']['id']}: E010 schema: {MEMBERS[shape][0]!r} is not of type 'object'"
    with pytest.raises(BundleError, match=re.escape(named)):
        build_bundle(tmp_path, [SourceResult("sweep", docs, {}, "sweep/1")], salt="x")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("shape", sorted(MEMBERS))
def test_a_lenient_build_lists_the_run_and_keeps_the_other_bytes(tmp_path, shape):
    clean = build_bundle(tmp_path / "clean", [SourceResult("sweep", _sweep_docs(), {}, "sweep/1")], salt="x")
    docs = _sweep_docs()
    named = docs[0]["run"]["id"]
    docs[0] = _members(docs[0], shape)
    got = build_bundle(tmp_path / "bad", [SourceResult("sweep", docs, {}, "sweep/1")], salt="x", strict=False)
    assert got["problems"] == [f"sweep: {named}: E010 schema: {MEMBERS[shape][0]!r} is not of type 'object'"]
    assert got["sources"]["sweep"]["validation"]["errors"] == 2 * len(MEMBERS[shape])
    assert got["sources"]["sweep"]["summary"] == clean["sources"]["sweep"]["summary"]
    lines, clean_lines = _lines(tmp_path / "bad"), _lines(tmp_path / "clean")
    assert sorted(lines) == sorted(clean_lines) and named in lines and len(lines) == 2
    for run_id, line in clean_lines.items():
        if run_id == named:
            assert json.loads(lines[run_id]) == _members(json.loads(line), shape)
        else:
            assert lines[run_id] == line


@pytest.mark.parametrize("shape", sorted(CONTAINERS))
def test_signals_that_are_not_a_list_are_kept_and_name_the_run(tmp_path, shape):
    docs = _sweep_docs()
    docs[0]["run"]["signals"] = copy.deepcopy(CONTAINERS[shape])
    reduced = reduce_for_bundle(docs[0], salt="x")
    assert reduced["run"]["signals"] == CONTAINERS[shape]
    assert local_checks(reduced) == []
    assert run_summary([reduced])["outcomes"] == {"unknown": 1}
    named = f"sweep: {docs[0]['run']['id']}: E010 schema: {CONTAINERS[shape]!r} is not of type 'array'"
    with pytest.raises(BundleError, match=re.escape(named)):
        build_bundle(tmp_path, [SourceResult("sweep", docs, {}, "sweep/1")], salt="x")
    assert not list(tmp_path.iterdir())
