"""The plugin eval reader: synthetic `claude plugin eval` results (tests/plugin_eval/pefixture.py, written from
Anthropic's docs with a URL per field) become `loopmath.harness-runs/0.1` rows. Claude Code is never run."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pefixture as F  # noqa: E402

from loopmath import harness_runs as HR  # noqa: E402
from loopmath.adapters import plugin_eval as PE  # noqa: E402
from loopmath.harness_runs.ocp import retained_text  # noqa: E402

OURS = "dev.loopmath.adapter.plugin-eval"
VENDOR = "com.anthropic.claude-code.plugin-eval"
MANIFEST = {"name": "lint-helper", "version": "0.4.0"}

# accepted at the default threshold 1.0 (a perfect run), per case and arm in file order; None is unscored
ACCEPTED = {
    ("fix-null-check", "with"): [True, False, True], ("fix-null-check", "without"): [False, False, True],
    ("rename-config-key", "with"): [True, True, False], ("rename-config-key", "without"): [False, False, False],
    ("add-skill-call", "with"): [True, False, None], ("add-skill-call", "without"): [False, False, False],
    ("write-docs-page", "with"): [True, True, False], ("write-docs-page", "without"): [True, True, False],
}
GRADES = {
    ("fix-null-check", "with"): [1.0, 0.75, 1.0], ("fix-null-check", "without"): [0.75, 0.5, 1.0],
    ("rename-config-key", "with"): [1.0, 1.0, 0.0], ("rename-config-key", "without"): [0.5, 0.5, 0.25],
    ("add-skill-call", "with"): [1.0, 0.75, None], ("add-skill-call", "without"): [0.75, 0.0, 0.75],
    ("write-docs-page", "with"): [1.0, 1.0, 0.75], ("write-docs-page", "without"): [1.0, 1.0, 0.5],
}


@pytest.fixture
def suite(tmp_path) -> F.Built:
    return F.write(tmp_path / "lint-helper", F.docs_suite(), manifest=MANIFEST)


def _key(row: dict) -> tuple[str, str, int]:
    return row["task_id"], row["arm"], int(row["pair"].rsplit("#", 1)[1])


def _by_key(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    out = {}
    for row in rows:
        assert _key(row) not in out, f"{_key(row)} read twice"
        out[_key(row)] = row
    return out


def _grade(row: dict):
    return row["outcome"]["score"] if "score" in (row["outcome"] or {}) else None


def test_every_run_is_one_checked_row(suite):
    got = PE.read(suite.path)
    assert got.format == "claude-plugin-eval/1"
    assert len(got.rows) == 24
    for row in got.rows:
        assert HR.check_row(row) == [], row
        assert row["schema"] == HR.SCHEMA and row["agent"] == "claude-code"
        assert row["effort"] is None and row["started_at"] is None and row["ended_at"] is None
    assert set(_by_key(got.rows)) == {(c, a, j) for (c, a), v in ACCEPTED.items() for j in range(1, len(v) + 1)}


def test_arms_pairs_ids_and_the_experiment_are_stable(suite, tmp_path):
    got = PE.read(suite.path)
    exp = got.rows[0]["experiment"]
    assert re.fullmatch(r"plugin-eval-[0-9a-f]{12}", exp)
    for row in got.rows:
        case, arm, j = _key(row)
        assert row["experiment"] == exp and row["pair"] == f"{case}#{j}" and arm in ("with", "without")
        assert row["id"] == f"plugin-eval:{exp}:{case}:{arm}:{j}"
        assert row["tool"] == {"name": "claude-plugin-eval", "version": F.CLAUDE_VERSION,
                               "run_id": f"{case}/{arm}/{j}", "task_id": case}
    assert PE.read(suite.path).rows == got.rows   # the same file reads the same
    moved = F.write(tmp_path / "elsewhere", suite.doc, manifest=MANIFEST, stamp="2026-10-21T09-00-00-000Z")
    assert PE.read(moved.path).rows[0]["experiment"] == exp   # the experiment is the content, never the path
    other = copy.deepcopy(suite.doc)
    other["costUsd"] = 2.5
    assert PE.read(F.write(tmp_path / "second", other).path).rows[0]["experiment"] != exp
    assert {r["experiment"] for r in PE.read(suite.path, experiment="my-suite").rows} == {"my-suite"}


def test_the_harness_record_names_the_plugin_from_its_manifest_or_the_flag(suite, tmp_path):
    rows = PE.read(suite.path).rows
    by_arm = {r["arm"]: r["harness"] for r in rows}
    assert by_arm["with"]["factors"] == ["plugin:lint-helper"] and by_arm["without"]["factors"] == []
    for rec in by_arm.values():
        assert rec["schema"] == "loopmath.harness/0.1" and rec["source"] == "plugin-eval"
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", rec["digest"]) and rec["parent"] is None
    assert by_arm["with"]["digest"] != by_arm["without"]["digest"]
    assert all(r["harness"] == by_arm[r["arm"]] for r in rows)   # one record per arm
    bare = F.write(tmp_path / "no-manifest", suite.doc)
    got = PE.read(bare.path)
    assert all(r["harness"] is None for r in got.rows)
    assert any("--plugin" in n for n in got.notes), got.notes
    flagged = PE.read(bare.path, plugin="lint-helper").rows
    assert {r["arm"]: r["harness"]["factors"] for r in flagged} == {"with": ["plugin:lint-helper"], "without": []}


def test_a_run_is_accepted_when_its_grade_reaches_the_threshold_one_by_default(suite):
    rows = _by_key(PE.read(suite.path).rows)
    for (case, arm), want in ACCEPTED.items():
        for j, acc in enumerate(want, 1):
            out = rows[(case, arm, j)]["outcome"]
            if acc is None:
                assert out is None
                continue
            assert out["accepted"] is acc and out["by"] == "claude-plugin-eval", (case, arm, j)
            assert out["grade"] == pytest.approx(GRADES[(case, arm)][j - 1]) and out["threshold"] == 1.0


def test_accept_at_sets_the_threshold(suite):
    rows = _by_key(PE.read(suite.path, accept_at=0.75).rows)
    assert rows[("fix-null-check", "with", 2)]["outcome"]["accepted"] is True
    assert rows[("fix-null-check", "without", 2)]["outcome"]["accepted"] is False
    assert rows[("fix-null-check", "without", 2)]["outcome"]["threshold"] == 0.75


def test_outcome_score_gives_the_grade_as_a_fraction(suite):
    rows = _by_key(PE.read(suite.path, outcome="score").rows)
    for (case, arm), want in GRADES.items():
        for j, g in enumerate(want, 1):
            out = rows[(case, arm, j)]["outcome"]
            if g is None:
                assert out is None
                continue
            assert out["name"] == "plugin-eval" and out["scale"] == "fraction" and out["judge"] == "claude-plugin-eval"
            assert out["score"] == pytest.approx(g)


def test_with_no_run_score_the_grade_comes_from_the_grader_list(tmp_path):
    built = F.write(tmp_path / "p", F.docs_suite(run_score=False), manifest=MANIFEST)
    rows = _by_key(PE.read(built.path, outcome="score").rows)
    assert rows[("fix-null-check", "with", 2)]["outcome"]["score"] == pytest.approx(0.75)
    assert rows[("add-skill-call", "with", 1)]["outcome"]["score"] == pytest.approx(1.0)   # scored: false is out
    assert rows[("add-skill-call", "without", 2)]["outcome"]["score"] == 0.0               # aborted scores 0


def test_a_run_whose_paid_graders_were_skipped_is_unscored_under_either_case_mean(tmp_path):
    for skipped_in_mean in (True, False):
        built = F.write(tmp_path / f"p{skipped_in_mean}", F.docs_suite(skipped_in_mean=skipped_in_mean))
        got = PE.read(built.path)
        row = _by_key(got.rows)[("add-skill-call", "with", 3)]
        assert row["outcome"] is None and row["status"] == "done"
        assert row["ext"][OURS]["run"]["skipped_paid_graders"] is True
        assert got.summary["unscored"] == 1


def test_an_errored_run_keeps_the_grade_the_file_gives_it_and_is_counted(suite):
    got = PE.read(suite.path)
    row = _by_key(got.rows)[("rename-config-key", "with", 3)]
    assert row["status"] == "done" and row["outcome"]["accepted"] is False and row["outcome"]["grade"] == 0.0
    assert row["ext"][OURS]["run"]["errored"] is True and "error" not in row["ext"][OURS]["run"]
    assert got.summary["errors"] == {"with": 1, "without": 0}
    assert any("2.1.299" in n for n in got.notes), got.notes


def test_an_errors_text_is_free_text_dropped_unless_asked_and_then_makes_the_document_full(suite):
    got = PE.read(suite.path)
    row = _by_key(got.rows)[("rename-config-key", "with", 3)]
    assert "error" not in row["ext"][VENDOR]["run"] and "run.error" in got.summary["dropped_text"]
    assert retained_text(row) == []   # metadata only, with no error text anywhere in the row
    assert "timed out" not in json.dumps(row)
    kept = _by_key(PE.read(suite.path, keep_text=True).rows)[("rename-config-key", "with", 3)]
    assert kept["ext"][VENDOR]["run"]["error"] == "timed out after 300s" and kept["ext"][OURS]["run"]["errored"]
    assert f"ext.{VENDOR}.run.error" in retained_text(kept)   # so the document's privacy is full


def test_an_aborted_run_scores_zero_and_keeps_where_it_stopped(suite):
    row = _by_key(PE.read(suite.path).rows)[("add-skill-call", "without", 2)]
    assert row["outcome"]["grade"] == 0.0 and row["outcome"]["accepted"] is False
    assert row["ext"][OURS]["run"]["aborted"] == {"server": "github", "tool": "create_issue"}


def test_cost_is_unknown_without_a_cost_file_and_the_suite_cost_is_never_split(suite):
    got = PE.read(suite.path)
    assert all(r.get("cost") is None for r in got.rows)
    assert got.summary["suite_cost_usd"] == 1.84
    for row in got.rows:
        assert row["ext"][OURS]["suite"]["cost_usd"] == 1.84


def test_unknown_fields_are_kept_and_free_text_is_dropped_unless_asked(suite):
    got = PE.read(suite.path)
    rows = _by_key(got.rows)
    kept = rows[("write-docs-page", "with", 1)]["ext"][VENDOR]
    assert kept["run"]["turns"] == 7
    assert kept["run"]["graders"][0] == {"weight": 2.0, "passed": True, "scored": True}
    assert "prompt" not in rows[("fix-null-check", "with", 1)]["ext"][VENDOR]["case"]
    assert {"case.prompt", "case.tags[]", "run.graders[].explanation", "run.graders[].name", "run.graders[].type",
            "suite.suite.name"} <= set(got.summary["dropped_text"])
    assert all(retained_text(r) == [] for r in got.rows)   # metadata only by default
    full = _by_key(PE.read(suite.path, keep_text=True).rows)
    assert full[("fix-null-check", "with", 1)]["ext"][VENDOR]["case"]["prompt"].startswith("loopmath fixture")
    assert retained_text(full[("fix-null-check", "with", 1)])   # so the document's privacy is full


@pytest.mark.parametrize("version", [2, "1", None])
def test_an_unknown_or_missing_schema_version_is_refused_by_name(tmp_path, version):
    doc = F.docs_suite()
    if version is None:
        del doc["schemaVersion"]
    else:
        doc["schemaVersion"] = version
    with pytest.raises(PE.FormatError, match="schemaVersion"):
        PE.read(F.write(tmp_path / "p", doc).path)


@pytest.mark.parametrize("raw, why", [("[1, 2]", "an object"), ("{\"schemaVersion\": 1}", "cases"),
                                      ("not json", "JSON")])
def test_a_file_that_is_not_a_result_is_refused_with_what_is_missing(tmp_path, raw, why):
    with pytest.raises(PE.FormatError, match=why):
        PE.read(F.write(tmp_path / "p", {}, raw=raw).path)


def test_two_cases_with_one_name_are_refused(tmp_path):
    doc = F.docs_suite()
    doc["cases"][1]["name"] = doc["cases"][0]["name"]
    with pytest.raises(PE.FormatError, match="fix-null-check"):
        PE.read(F.write(tmp_path / "p", doc).path)


@pytest.mark.parametrize("field", ["score", "delta"])
def test_run_grades_that_do_not_reproduce_the_case_scores_are_refused(tmp_path, field):
    doc = F.docs_suite()
    doc["cases"][0]["aggregates"][field] += 0.25
    with pytest.raises(PE.FormatError, match=r"fix-null-check.*" + F.CLAUDE_VERSION):
        PE.read(F.write(tmp_path / "p", doc).path)


def test_a_threshold_other_than_one_is_noticed_from_cases_passed(tmp_path):
    doc = F.document(F.docs_suite()["cases"], threshold=0.9)
    got = PE.read(F.write(tmp_path / "p", doc).path)
    assert any("--threshold" in n and "--accept-at" in n for n in got.notes), got.notes
    quiet = PE.read(F.write(tmp_path / "q", doc).path, accept_at=0.9)
    assert not any("--threshold" in n for n in quiet.notes), quiet.notes


def test_a_partial_suite_imports_what_ran_and_names_why(tmp_path):
    doc = F.document(F.docs_suite()["cases"][:2], partial="cost_ceiling")
    got = PE.read(F.write(tmp_path / "p", doc).path)
    assert len(got.rows) == 12 and got.summary["partial"] == "cost_ceiling"
    assert any("cost_ceiling" in n for n in got.notes)
    assert {r["ext"][OURS]["suite"]["partial_reason"] for r in got.rows} == {"cost_ceiling"}


def test_a_partial_reason_the_docs_do_not_list_is_free_text(tmp_path):
    doc = F.document(F.docs_suite()["cases"][:2], partial="stopped by the operator at 14:02")
    path = F.write(tmp_path / "p", doc).path
    got = PE.read(path)
    assert got.summary["partial"] == "stopped by the operator at 14:02"   # printed, not written
    assert any("stopped by the operator" in n for n in got.notes)
    assert "suite.partialReason" in got.summary["dropped_text"]
    for r in got.rows:
        assert r["ext"][OURS]["suite"]["partial_reason"] is None and r["ext"][OURS]["suite"]["partial"] is True
        assert "operator" not in json.dumps(r) and retained_text(r) == []
    kept = PE.read(path, keep_text=True).rows[0]
    assert kept["ext"][VENDOR]["suite"]["partialReason"] == "stopped by the operator at 14:02"
    assert f"ext.{VENDOR}.suite.partialReason" in retained_text(kept)


def test_replicate_runs_get_the_these_cases_note_and_one_run_cases_do_not(tmp_path):
    path = F.write(tmp_path / "r", F.docs_suite()).path
    for outcome, says in (("accept", "acceptance rows are compared run by run: the answer is about these cases"),
                          ("score", "check-change from-runs compares the cases, one pair each")):
        got = PE.read(path, outcome=outcome)
        note = [n for n in got.notes if "add cases rather than runs" in n]
        assert len(note) == 1 and note[0].startswith("12 pairs from 4 compared cases:"), got.notes
        assert says in note[0] and "docs/plugin-eval.md" in note[0], note
    one = PE.read(F.write(tmp_path / "o", F.twin_suite()).path, outcome="score")
    assert one.summary["pairs"] == 3 and not any("add cases" in n for n in one.notes), one.notes


def test_one_arm_only_imports_and_says_there_is_nothing_to_compare(tmp_path):
    cases = [F.case(c["name"], c["arms"]["with"], None) for c in F.docs_suite()["cases"]]
    got = PE.read(F.write(tmp_path / "p", F.document(cases)).path)
    assert {r["arm"] for r in got.rows} == {"with"}
    assert any("one arm" in n for n in got.notes), got.notes


def test_the_model_flag_fills_the_model_the_file_does_not_record(suite):
    assert {r["model"] for r in PE.read(suite.path).rows} == {None}
    assert {r["model"] for r in PE.read(suite.path, model="claude-sonnet-5").rows} == {"claude-sonnet-5"}


def test_the_summary_counts_cases_runs_and_pairs(suite):
    s = PE.read(suite.path).summary
    assert s["plugin"] == {"name": "lint-helper", "version": "0.4.0", "from": ".claude-plugin/plugin.json"}
    assert s["cases"] == 4 and s["runs"] == {"with": 12, "without": 12} and s["pairs"] == 12
    assert s["threshold"] == {"value": 1.0, "from": "default"}
    assert s["claude_version"] == F.CLAUDE_VERSION and s["partial"] is None
    json.dumps(s)   # plain JSON


# Comparable cases. The docs omit `cases[].aggregates.delta` "when the case ran one arm or the arms aren't
# comparable" (https://code.claude.com/docs/en/plugin-evals#json-result) and do not say when arms are not
# comparable. So only a case with both arms and a delta reaches the rows; any other case in a two-arm suite is
# withheld, its run grades kept in the summary. A missing delta is never read as 0 or as comparable.

def _no_delta(doc: dict, *names: str) -> dict:
    for c in doc["cases"]:
        if c["name"] in names:
            del c["aggregates"]["delta"]
    return doc


def test_a_two_arm_case_with_no_delta_is_withheld_and_its_run_grades_kept(suite, tmp_path):
    full = _by_key(PE.read(suite.path).rows)
    got = PE.read(F.write(tmp_path / "p", _no_delta(F.docs_suite(), "rename-config-key"), manifest=MANIFEST).path)
    assert "rename-config-key" not in {r["task_id"] for r in got.rows}
    assert len(got.rows) == 18 and got.summary["pairs"] == 9
    for key, row in _by_key(got.rows).items():   # the comparable cases read as before, except the experiment
        assert row["outcome"] == full[key]["outcome"] and row["pair"] == full[key]["pair"], key
    (held,) = got.summary["withheld"]
    assert held["case"] == "rename-config-key" and "no delta" in held["why"] and "not comparable" in held["why"]
    assert held["runs"] == {"with": [1.0, 1.0, 0.0], "without": [0.5, 0.5, 0.25]}
    assert any("rename-config-key" in n and "withheld" in n for n in got.notes), got.notes


def test_a_one_arm_case_in_a_two_arm_suite_is_withheld(tmp_path):
    doc = F.docs_suite()
    last = doc["cases"][-1]
    doc["cases"][-1] = F.case(last["name"], last["arms"]["with"], None)
    got = PE.read(F.write(tmp_path / "p", F.document(doc["cases"])).path)
    assert "write-docs-page" not in {r["task_id"] for r in got.rows}
    (held,) = got.summary["withheld"]
    assert held["case"] == "write-docs-page" and "one arm" in held["why"]
    assert held["runs"] == {"with": [1.0, 1.0, 0.75], "without": []}


def test_a_two_arm_suite_with_no_comparable_case_is_refused(tmp_path):
    doc = _no_delta(F.docs_suite(), *(c["name"] for c in F.docs_suite()["cases"]))
    with pytest.raises(PE.FormatError, match=r"no case is comparable.*fix-null-check.*not comparable"):
        PE.read(F.write(tmp_path / "p", doc).path)


def test_a_comparable_suite_withholds_nothing(suite):
    assert PE.read(suite.path).summary["withheld"] == []


# The input layout is provisional (the run's `score` and the grader list's key are assumed names). A run's shape
# that admits two readings is refused, and so is a run whose `score` and grader list disagree. Equal case means
# do not prove the per-run grades: [0, 1] and [0.5, 0.5] both average 0.5.

def _run_of(doc: dict, case: str, arm: str, j: int) -> dict:
    return next(c for c in doc["cases"] if c["name"] == case)["arms"][arm][j - 1]


def test_run_scores_that_disagree_with_their_graders_are_refused_though_the_case_mean_holds(tmp_path):
    doc = F.docs_suite()
    one, two = _run_of(doc, "fix-null-check", "with", 1), _run_of(doc, "fix-null-check", "with", 2)
    one["score"], two["score"] = two["score"], one["score"]   # 1.0 and 0.75 swapped: the same case mean and delta
    with pytest.raises(PE.FormatError, match=r"fix-null-check.*with.*run 1.*0\.75.*1(\.0)?\b"):
        PE.read(F.write(tmp_path / "p", doc).path)


@pytest.mark.parametrize("score", [True, False])
def test_a_run_with_two_grader_lists_is_refused_as_ambiguous(tmp_path, score):
    doc = F.docs_suite(run_score=score)
    r = _run_of(doc, "fix-null-check", "with", 1)
    r["checks"] = [dict(g) for g in r["graders"]]
    with pytest.raises(PE.FormatError, match=r"fix-null-check.*with.*run 1.*(two|2) grader lists.*checks.*graders"):
        PE.read(F.write(tmp_path / "p", doc).path)


@pytest.mark.parametrize(("member", "value"), [("passed", "yes"), ("passed", 1), ("passed", 0), ("scored", 1),
                                              ("weight", -1.0), ("weight", "2"), ("weight", True)])
def test_a_grader_item_out_of_shape_is_refused(tmp_path, member, value):
    doc = F.docs_suite(run_score=False)
    _run_of(doc, "fix-null-check", "with", 1)["graders"][0][member] = value
    with pytest.raises(PE.FormatError, match=rf"fix-null-check.*with.*run 1.*{member}"):
        PE.read(F.write(tmp_path / "p", doc).path)


def test_a_run_with_no_score_and_no_grader_list_is_refused(tmp_path):
    doc = F.docs_suite(run_score=False)
    del _run_of(doc, "fix-null-check", "with", 1)["graders"]
    with pytest.raises(PE.FormatError, match=r"fix-null-check.*with.*run 1.*no score and no grader list"):
        PE.read(F.write(tmp_path / "p", doc).path)


@pytest.mark.parametrize("score", [1.5, -0.1, "1", None])
def test_a_run_score_outside_zero_to_one_is_refused(tmp_path, score):
    doc = F.docs_suite()
    _run_of(doc, "fix-null-check", "with", 1)["score"] = score
    with pytest.raises(PE.FormatError, match=r"fix-null-check.*with.*run 1.*score"):
        PE.read(F.write(tmp_path / "p", doc).path)


def test_the_valid_fixtures_hold_the_declared_types():
    """The fixture's own guard: after a JSON round trip, every valid document holds the types the reader requires
    (booleans, not 1 or 0), so a strict reader is tested on valid input."""
    for doc in (F.docs_suite(), F.docs_suite(run_score=False), F.docs_suite(skipped_in_mean=False), F.twin_suite()):
        doc = json.loads(json.dumps(doc))
        for c in doc["cases"]:
            for arm in c["arms"].values():
                for r in arm:
                    assert r["error"] is None or isinstance(r["error"], str)
                    assert r["aborted"] is None or isinstance(r["aborted"], dict)
                    assert isinstance(r["skippedPaidGraders"], bool)
                    if "score" in r:
                        assert isinstance(r["score"], float) and 0.0 <= r["score"] <= 1.0
                    for g in r["graders"]:
                        assert isinstance(g["passed"], bool) and isinstance(g["scored"], bool), g
                        assert isinstance(g["weight"], float) and g["weight"] >= 0, g
    for bad in ({"passed": 1}, {"passed": "yes"}, {"passed": True, "scored": 0}, {"passed": True, "weight": True}):
        with pytest.raises(TypeError):
            F.grader("g", **bad)


# An ungraded run is not a documented zero. The docs say a run that ended on an API error "scores 0 with no
# grader results" (https://code.claude.com/docs/en/plugin-evals#runs-ended-on-an-api-error), and that other
# abnormal runs, such as a timeout, may still be graded (#json-result); the file has no member that tells the two
# apart, and the error is free text. So a run with neither a score nor a grader list is refused unless it is
# aborted (documented: score 0), whatever its error says and whatever the Claude Code version. An errored run
# with a score keeps it. In a case the file gives no delta (withheld; the docs print its delta as a dash when a
# run ended on an API error), an ungraded run is listed with no grade, never 0.

@pytest.mark.parametrize("error", ["timed out after 300s", "API Error: 429 rate_limit_error"])
def test_an_ungraded_errored_run_is_refused_whatever_the_error_or_version(tmp_path, error):
    for version in ("2.1.299", F.CLAUDE_VERSION):
        doc = F.docs_suite(run_score=False)
        doc["claudeVersion"] = version
        r = _run_of(doc, "rename-config-key", "with", 3)
        r["error"] = error
        del r["graders"]
        with pytest.raises(PE.FormatError, match=r"rename-config-key.*with.*run 3.*no score and no grader list"):
            PE.read(F.write(tmp_path / f"p{version}{len(error)}", doc).path)


def test_an_errored_run_with_a_score_and_no_grader_list_keeps_its_score(tmp_path):
    doc = F.docs_suite()
    del _run_of(doc, "rename-config-key", "with", 3)["graders"]   # error "timed out after 300s", score 0.0
    rows = _by_key(PE.read(F.write(tmp_path / "p", doc).path, outcome="score").rows)
    row = rows[("rename-config-key", "with", 3)]
    assert row["outcome"]["score"] == 0.0 and row["ext"][OURS]["run"]["errored"] is True


def test_an_ungraded_run_in_a_withheld_case_is_listed_with_no_grade(tmp_path):
    doc = F.docs_suite(run_score=False)
    del next(c for c in doc["cases"] if c["name"] == "rename-config-key")["aggregates"]["delta"]
    r = _run_of(doc, "rename-config-key", "with", 3)
    r["error"] = "API Error: 429 rate_limit_error"
    del r["graders"]
    got = PE.read(F.write(tmp_path / "p", doc).path)
    (held,) = got.summary["withheld"]
    assert held["case"] == "rename-config-key" and held["runs"]["with"] == [1.0, 1.0, None]
    assert len(got.rows) == 18
