"""OCP 0.4 `run.flow`: E218 (a run of a flow carries its configuration, the flow's digest and one of its inputs),
W219 (judge and user scores name judges of the flow) and W220 (judge and user scores lie in the linear range they
are read by); and the `simulated` cost basis (W200, E171)."""

from __future__ import annotations

import json

import pytest

from loopmath.cli import main as cli_main
from loopmath.ocp import conformance
from loopmath.ocp.canonical import flow_digest

from ._common import EXAMPLES, codes, load

V04 = EXAMPLES / "v0.4"
GOLDEN = EXAMPLES / "golden" / "v0.4"
FLOW_FILE = V04 / "lit-brief.flow.ocp.json"
RUN_FILES = [V04 / "lit-brief.run-1.ocp.json", V04 / "lit-brief.run-2.ocp.json"]
DIGEST = "flw_774ab7a14c50"
SIMULATE = "dev.loopmath.simulate"


def _run() -> dict:
    return load(RUN_FILES[0])


def _found(run, *flows) -> list[tuple[str, str]]:
    """The run's findings, checked with the given flow documents (the example flow when none is given)."""
    docs = [*(flows or (load(FLOW_FILE),)), run]
    return [(f.code, f.path) for f in conformance.validate_many(docs)[-1]]


def _flow_with(change) -> dict:
    """The example flow with a change, its declared digest made to hold again."""
    doc = load(FLOW_FILE)
    change(doc["flow"])
    doc["flow"]["digest"] = flow_digest(doc["flow"])
    return doc


# E218, one document ------------------------------------------------------------------------------

def test_run_flow_without_the_configuration_is_e218():
    run = _run()
    del run["run"]["configuration"]
    alone = [(f.level, f.code, f.path) for f in conformance.validate_doc(run) if f.code == "E218"]
    assert alone == [("error", "E218", "$['run']['flow']")]
    assert [c for c in _found(run) if c[0] == "E218"] == [("E218", "$['run']['flow']")]


def test_run_flow_is_unknown_below_0_4():
    run = _run()
    del run["run"]["configuration"]
    del run["producer"]["capabilities"]["flow"]
    run["ocp"] = "0.3"
    assert "E218" not in codes(conformance.validate_doc(run))
    assert "E218" not in codes(conformance.validate_many([load(FLOW_FILE), run])[1])


# E218 and W219, across documents ----------------------------------------------------------------

def test_the_example_runs_validate_with_their_flow():
    assert conformance.validate_files([FLOW_FILE, *RUN_FILES]) == [[], [], []]
    assert [conformance.validate_file(p) for p in RUN_FILES] == [[], []]


def test_a_digest_that_is_not_the_flows_is_e218():
    run = _run()
    run["run"]["flow"]["digest"] = "flw_000000000000"
    [finding] = conformance.validate_many([load(FLOW_FILE), run])[1]
    assert (finding.level, finding.code, finding.path) == ("error", "E218", "$['run']['flow']['digest']")
    assert finding.message == (f"run.flow.digest 'flw_000000000000' is not the digest of flow 'lit-brief' version 1 "
                               f"('{DIGEST}')")
    del run["run"]["flow"]["digest"]  # a run that gives no digest is not compared
    assert _found(run) == []


def test_a_task_the_flow_does_not_list_is_e218():
    run = _run()
    run["run"]["task"]["id"] = "q99"
    [finding] = conformance.validate_many([load(FLOW_FILE), run])[1]
    assert (finding.code, finding.path) == ("E218", "$['run']['task']['id']")
    assert finding.message == "task.id 'q99' is not among the inputs.ids of flow 'lit-brief' version 1 (20 ids)"


def test_a_flow_without_input_ids_takes_any_task():
    def no_ids(flow):
        del flow["inputs"]["ids"], flow["inputs"]["holdout"]

    flow = _flow_with(no_ids)
    assert conformance.validate_doc(flow) == []
    run = _run()
    run["run"]["task"]["id"] = "q99"
    run["run"]["flow"]["digest"] = flow["flow"]["digest"]
    assert _found(run, flow) == []


def test_a_run_of_a_flow_not_in_the_set_is_not_checked():
    run = _run()
    run["run"]["flow"]["version"] = 2
    run["run"]["flow"]["digest"] = "flw_000000000000"
    run["run"]["task"]["id"] = "q99"
    run["run"]["signals"][0]["source"]["ref"] = "j-gemini"
    assert _found(run) == []


def test_with_two_flows_of_one_id_and_version_each_check_passes_on_either():
    other = _flow_with(lambda flow: flow["inputs"].update(ids=["q99"], holdout=[]))
    assert other["flow"]["digest"] != DIGEST
    run = _run()
    assert _found(run, other, load(FLOW_FILE)) == []
    run["run"]["task"]["id"] = "q99"
    run["run"]["flow"]["digest"] = other["flow"]["digest"]
    assert _found(run, load(FLOW_FILE), other) == []
    run["run"]["flow"]["digest"] = "flw_000000000000"
    assert _found(run, load(FLOW_FILE), other) == [("E218", "$['run']['flow']['digest']")]


def test_a_judge_or_user_score_naming_no_judge_of_the_flow_is_w219():
    run = _run()
    assert [(s["source"]["kind"], s["source"]["ref"]) for s in run["run"]["signals"]][2] == ("user", "person")
    run["run"]["signals"][0]["source"]["ref"] = "j-gemini"
    del run["run"]["signals"][2]["source"]["ref"]
    found = conformance.validate_many([load(FLOW_FILE), run])[1]
    assert [(f.level, f.code, f.path) for f in found] == [
        ("warning", "W219", "$['run']['signals'][0]['source']"),
        ("warning", "W219", "$['run']['signals'][2]['source']")]
    assert "source.ref 'j-gemini'" in found[0].message and "judges: j-astra, j-claude, person" in found[0].message
    assert "no source.ref" in found[1].message
    assert conformance.validate_doc(run) == []  # without the flow in the set nothing is compared


def test_w219_reads_only_judge_and_user_scores():
    run = _run()
    signal = run["run"]["signals"][0]
    verdict = {"id": "sig_verdict", "kind": "verdict", "name": "accepted", "value": "accept",
               "observed_at": signal["observed_at"], "tier": "reported", "at_attempt": signal["at_attempt"],
               "source": {"kind": "judge", "ref": "j-gemini"}}
    run["run"]["signals"].append(verdict)
    run["run"]["signals"][1]["source"] = {"kind": "metric", "ref": "rouge"}
    assert _found(run) == []


def test_the_flow_goldens_need_the_flow_and_main_reports_them(capsys):
    flow = str(FLOW_FILE)
    assert conformance.main([flow, *map(str, RUN_FILES)]) == 0
    capsys.readouterr()
    assert conformance.main([flow, str(GOLDEN / "E218-fail.ocp.json")]) == 1
    assert "error E218 $['run']['flow']['digest']" in capsys.readouterr().out
    assert conformance.main([flow, str(GOLDEN / "W219-fail.ocp.json")]) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "warning W219 $['run']['signals'][1]['source']" in out
    assert conformance.main([str(GOLDEN / "E218-fail.ocp.json"), str(GOLDEN / "W219-fail.ocp.json")]) == 0


# W220, alone ------------------------------------------------------------------------------------

JUDGE_NS = "dev.loopmath.judge"
VALUE = "$['run']['signals'][0]['value']"
HUGE = 10**400  # an integer JSON can write and no float can hold


def _ranged(value, span=(0, 10), **signal) -> dict:
    """The example run with j-claude's score linear at `value` and `signal`'s members, its signal giving the range
    `span` (no range member when None)."""
    run = _run()
    first = run["run"]["signals"][0]
    first.update({"scale": "linear", "value": value, **signal})
    if span is not None:
        first["ext"] = {JUDGE_NS: {"range": list(span)}}
    return run


def _w220(findings) -> list[tuple[str, str, str]]:
    return [(f.level, f.path, f.message) for f in findings if f.code == "W220"]


def _own(value, lo, hi) -> str:
    return f"score 'quality' has value {value!r}, outside the range [{lo!r}, {hi!r}] its signal gives"


@pytest.mark.parametrize("value, span", [(-1, (0, 10)), (10.5, (0, 10)), (-11, (-10, -1)), (0, (-10, -1)),
                                         (2**53 + 2, (0, 2**53)), (2**53 + 1, (0, 2**53)),
                                         (-(2**53) - 1, (-(2**53), 0)), (2**53 + 1, (0, 2.0**53))],
                         ids=["below", "above", "below a negative range", "above a negative range", "past 2**53",
                              "one past 2**53, its float on the bound", "one below -(2**53), its float on the bound",
                              "one past a float bound"])
def test_a_judge_score_outside_its_signals_range_warns(value, span):
    findings = conformance.validate_doc(_ranged(value, span))
    assert [(f.level, f.code, f.path, f.message) for f in findings] == [("warning", "W220", VALUE, _own(value, *span))]


@pytest.mark.parametrize("value, span", [(0, (0, 10)), (10, (0, 10)), (10.0, (0, 10)), (5, (0, 10)),
                                         (-10, (-10, -1)), (-1, (-10, -1)), (2**53, (0, 2**53)),
                                         (-(2**53), (-(2**53), 0))],
                         ids=["low bound", "high bound", "high bound as a float", "inside", "negative low bound",
                              "negative high bound", "bound 2**53", "bound -(2**53)"])
def test_a_judge_score_in_its_signals_range_or_on_a_bound_does_not_warn(value, span):
    assert conformance.validate_doc(_ranged(value, span)) == []


def test_a_person_score_and_a_judge_score_without_a_ref_warn_alone():
    for source in ({"kind": "user", "ref": "person"}, {"kind": "judge"}, {"kind": "judge", "ref": 3}):
        run = _ranged(12, source=source)
        assert _w220(conformance.validate_doc(run)) == [("warning", VALUE, _own(12, 0, 10))], source


def _member(run, member):
    run["run"]["signals"][0]["ext"] = {JUDGE_NS: member}
    return run


NOT_WARNED = {
    "no range member": lambda: _ranged(12, None),
    "log scale": lambda: _ranged(12, scale="log"),
    "no scale": lambda: (lambda run: (run["run"]["signals"][0].pop("scale"), run)[1])(_ranged(12)),
    "falling range": lambda: _ranged(12, (10, 0)),
    "equal bounds": lambda: _ranged(12, (5, 5)),
    "range past 2**53": lambda: _ranged(12, (0, 2**60)),
    "infinite bound": lambda: _ranged(12, (0, float("inf"))),
    "range of booleans": lambda: _ranged(12, (False, True)),
    "range of one number": lambda: _ranged(12, (1,)),
    "range an object": lambda: _member(_ranged(12, None), {"range": {"lo": 0, "hi": 10}}),
    "member not an object": lambda: _member(_ranged(12, None), [0, 10]),
    "a command score": lambda: _ranged(12, source={"kind": "command", "ref": "claims-check"}),
    "no source": lambda: (lambda run: (run["run"]["signals"][0].pop("source"), run)[1])(_ranged(12)),
    "a declared null score": lambda: _ranged(None),
}


@pytest.mark.parametrize("case", NOT_WARNED, ids=NOT_WARNED)
def test_a_score_without_a_usable_range_or_value_does_not_warn(case):
    run = NOT_WARNED[case]()
    assert "W220" not in codes(conformance.validate_doc(run))
    assert "W220" not in codes(conformance.validate_many([load(FLOW_FILE), run])[1])


@pytest.mark.parametrize("value", [HUGE, -HUGE, float("nan"), float("inf"), True, "12"],
                         ids=["huge", "huge negative", "NaN", "infinite", "boolean", "string"])
def test_a_value_no_float_holds_gets_its_value_finding_and_no_warning(value):
    found = codes(conformance.validate_doc(_ranged(value)))
    assert "W220" not in found and "E195" in found
    if value in (HUGE, -HUGE):
        assert found == ["E195"]


def test_w220_is_a_0_4_rule():
    run = _ranged(12)
    run["ocp"] = "0.3"
    del run["producer"]["capabilities"]["flow"]
    assert "W220" not in codes(conformance.validate_doc(run))
    assert "W220" not in codes(conformance.validate_many([_linear_flow(), run])[1])


# W220, across documents -------------------------------------------------------------------------

def _linear_flow(span=(0, 10), judge=0) -> dict:
    """The example flow with one judge (j-claude by default) linear over `span`."""
    return _flow_with(lambda flow: flow["judges"][judge].update(scale="linear", range=list(span)))


def _run_of(flow, value, **signal) -> dict:
    """The example run of `flow` (its digest), j-claude's score linear at `value` with no range of its own."""
    run = _ranged(value, None, **signal)
    run["run"]["flow"]["digest"] = flow["flow"]["digest"]
    return run


def _judged(value, lo, hi, judge="j-claude") -> str:
    return (f"score 'quality' has value {value!r}, outside the range [{lo!r}, {hi!r}] of judge {judge!r} in flow "
            "'lit-brief' version 1")


@pytest.mark.parametrize("value, span", [(-1, (0, 10)), (12, (0, 10)), (-11, (-10, -1)), (0, (-10, -1)),
                                         (2**53 + 1, (0, 2**53)), (-(2**53) - 1, (-(2**53), 0))],
                         ids=["below", "above", "below a negative range", "above a negative range",
                              "one past 2**53, its float on the bound", "one below -(2**53), its float on the bound"])
def test_a_score_outside_its_flow_judges_range_warns(value, span):
    flow = _linear_flow(span)
    run = _run_of(flow, value)
    assert conformance.validate_doc(run) == []  # alone the run gives no range
    found = conformance.validate_many([flow, run])
    assert found[0] == [] and _w220(found[1]) == [("warning", VALUE, _judged(value, *span))]
    assert [f.code for f in found[1]] == ["W220"]


@pytest.mark.parametrize("value, span", [(0, (0, 10)), (10, (0, 10)), (-10, (-10, -1)), (-1, (-10, -1)),
                                         (2**53, (0, 2**53)), (-(2**53), (-(2**53), 0))],
                         ids=["low bound", "high bound", "negative low bound", "negative high bound", "bound 2**53",
                              "bound -(2**53)"])
def test_a_score_on_its_flow_judges_bound_does_not_warn(value, span):
    flow = _linear_flow(span)
    assert conformance.validate_many([flow, _run_of(flow, value)]) == [[], []]


def test_a_person_score_is_held_to_the_person_judges_range():
    flow = _linear_flow((1, 5), judge=2)
    run = _run_of(flow, 3)
    run["run"]["signals"][2].update(value=6, scale="linear")
    assert _w220(conformance.validate_many([flow, run])[1]) == [
        ("warning", "$['run']['signals'][2]['value']", _judged(6, 1, 5, "person"))]


def test_the_flow_judges_range_is_read_whatever_the_signals_scale():
    flow = _linear_flow()
    run = _run_of(flow, 12, scale="log")
    assert _w220(conformance.validate_many([flow, run])[1]) == [("warning", VALUE, _judged(12, 0, 10))]


def test_a_run_version_written_as_a_float_still_names_the_flow():
    flow = _linear_flow()
    run = _run_of(flow, 12)
    run["run"]["flow"]["version"] = 1.0
    assert _w220(conformance.validate_many([flow, run])[1]) == [("warning", VALUE, _judged(12, 0, 10))]


def test_no_matching_flow_or_no_usable_judge_range_gives_no_warning():
    flow = _linear_flow()
    run = _run_of(flow, 12)
    run["run"]["flow"]["version"] = 2
    assert "W220" not in codes(conformance.validate_many([flow, run])[1])  # no flow of that version
    for span in ((10, 0), (0, 2**1024), (0, 1e308)):  # E214 refuses each, in the flow, before any float conversion
        bad = _linear_flow(span)
        found = conformance.validate_many([bad, _run_of(bad, 12)])
        assert codes(found[0]) == ["E214"] and "W220" not in codes(found[1]), span
    fraction = load(FLOW_FILE)  # j-claude is a fraction judge: no range to hold the score to
    run = _ranged(12, None, scale="log")
    assert "W220" not in codes(conformance.validate_many([fraction, run])[1])


def test_a_score_from_no_judge_or_no_ref_is_left_to_w219():
    flow = _linear_flow()
    for source in ({"kind": "command", "ref": "j-claude"}, {"kind": "judge"}, {"kind": "judge", "ref": "j-gemini"}):
        run = _run_of(flow, 12, source=source)
        assert "W220" not in codes(conformance.validate_many([flow, run])[1]), source


@pytest.mark.parametrize("order", ["accepting first", "accepting last"])
def test_with_two_flows_of_one_id_and_version_the_score_warns_only_outside_both(order):
    narrow, wide = _linear_flow((0, 10)), _linear_flow((0, 20))
    flows = [wide, narrow] if order == "accepting first" else [narrow, wide]
    run = _run_of(narrow, 12)
    assert "W220" not in codes(conformance.validate_many([*flows, run])[-1])
    run = _run_of(narrow, 25)
    assert _w220(conformance.validate_many([*flows, run])[-1]) == [
        ("warning", VALUE, f"score 'quality' has value 25, outside the range "
                           f"{'[0, 20] or [0, 10]' if order == 'accepting first' else '[0, 10] or [0, 20]'} of judge "
                           "'j-claude' in flow 'lit-brief' version 1")]


def test_a_flow_without_the_judge_range_does_not_count():
    ranged, plain = _linear_flow((0, 10)), load(FLOW_FILE)
    run = _run_of(ranged, 12)
    assert _w220(conformance.validate_many([plain, ranged, run])[-1]) == [("warning", VALUE, _judged(12, 0, 10))]


def test_outside_both_its_own_and_the_judges_range_gives_two_warnings_in_order():
    flow = _linear_flow((0, 10))
    run = _run_of(flow, 12)
    run["run"]["signals"][0]["ext"] = {JUDGE_NS: {"range": [0, 5]}}
    found = conformance.validate_many([flow, run])[1]
    assert [(f.level, f.code, f.path, f.message) for f in found] == [
        ("warning", "W220", VALUE, _own(12, 0, 5)), ("warning", "W220", VALUE, _judged(12, 0, 10))]


def test_a_value_no_float_holds_gets_no_cross_document_warning():
    flow = _linear_flow()
    found = conformance.validate_many([flow, _run_of(flow, HUGE)])[1]
    assert codes(found) == ["E195"]


def test_w220_files_pass_with_their_warning_printed(capsys, tmp_path):
    fail, passed = GOLDEN / "W220-fail.ocp.json", GOLDEN / "W220-pass.ocp.json"
    assert conformance.main([str(passed), str(fail)]) == 0
    out = capsys.readouterr().out
    assert f"PASS  {fail}  (0 errors, 1 warnings)" in out
    assert "warning W220 $['run']['signals'][1]['value']: score 'quality' has value 12, outside the range [0, 10] " \
           "its signal gives" in out
    assert cli_main(["ocp", "validate", str(fail)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PASS") and "warning W220 $['run']['signals'][1]['value']" in out
    assert cli_main(["ocp", "validate", "--json", str(fail)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["files"][0]["warnings"] == 1
    assert [(f["level"], f["code"]) for f in payload["files"][0]["findings"]] == [("warning", "W220")]
    flow = _linear_flow()
    run = tmp_path / "run.ocp.json"
    run.write_text(json.dumps(_run_of(flow, 12)), encoding="utf-8")
    together = tmp_path / "flow.ocp.json"
    together.write_text(json.dumps(flow), encoding="utf-8")
    assert cli_main(["ocp", "validate", str(together), str(run)]) == 0
    assert "warning W220 $['run']['signals'][0]['value']" in capsys.readouterr().out


# The simulated cost basis -----------------------------------------------------------------------

def _simulated() -> dict:
    return load(GOLDEN / "E171-pass.ocp.json")


def test_a_simulated_run_with_the_mark_passes():
    doc = _simulated()
    assert {a["cost"]["basis"] for a in doc["attempts"]} == {"simulated"}
    assert isinstance(doc["run"]["ext"][SIMULATE], dict)
    assert doc["producer"]["name"] == "loopmath/harness-runs"
    assert conformance.validate_doc(doc) == []


@pytest.mark.parametrize("mark", [None, True, "seed 1103", [1103]])
def test_simulated_costs_without_the_mark_are_e171(mark):
    doc = _simulated()
    if mark is None:
        del doc["run"]["ext"][SIMULATE]
    else:
        doc["run"]["ext"][SIMULATE] = mark
    found = conformance.validate_doc(doc)
    assert [(f.code, f.path) for f in found] == [
        ("E171", f"$['attempts'][{i}]['cost']['basis']") for i in range(len(doc["attempts"]))]


def test_a_dagr_producer_gets_no_simulated_exception():
    doc = _simulated()
    doc["producer"]["name"] = "dagr"
    assert set(codes(conformance.validate_doc(doc))) == {"E171"}


def test_other_producers_may_say_simulated_from_0_4():
    doc = _simulated()
    doc["producer"]["name"] = "acme-sim"
    del doc["run"]["ext"][SIMULATE]
    assert conformance.validate_doc(doc) == []


def test_simulated_is_not_recommended_below_0_4():
    assert "simulated" not in conformance.RECOMMENDED_VOCABULARY["cost.basis"]
    assert conformance.RECOMMENDED_VOCABULARY_V04["cost.basis"] == {"measured", "allocated", "simulated"}
    doc = _simulated()
    doc["ocp"] = "0.3"
    del doc["run"]["flow"], doc["producer"]["capabilities"]["flow"]
    found = codes(conformance.validate_doc(doc))
    assert found == ["E171", "W200"]
    doc["producer"]["name"] = "acme-sim"
    assert codes(conformance.validate_doc(doc)) == ["W200"]
